"""Minimal OpenAI-compatible server for exllamav3 with MoE CPU offload.

Loopback only: TLS and bearer-token auth are Caddy's job, matching the other backends on
this host. Endpoints: /v1/models, /v1/chat/completions, /v1/completions (streaming and not),
/health.

Single in-flight generation by design -- this model holds ~95 GB of VRAM and its decode is
host-DRAM-bound, so concurrency buys aggregate throughput at the cost of every individual
request. A lock serialises rather than interleaves.
"""
import sys, os, re, json, time, uuid, asyncio, argparse, threading
sys.path.insert(0, "/opt/exllamav3")
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Any
import uvicorn

from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler import (
    GreedySampler, CustomSampler, SS_Temperature, SS_TopK, SS_TopP, SS_Sample,
)

app = FastAPI(title="exllamav3-openai")
STATE: dict[str, Any] = {}
GEN_LOCK = threading.Lock()


class ChatReq(BaseModel):
    model: str | None = None
    messages: list[dict]
    max_tokens: int | None = 1024
    temperature: float | None = 0.7
    top_p: float | None = 0.9
    top_k: int | None = 50
    stream: bool | None = False
    stop: Any = None
    reasoning_effort: str | None = None      # low | high | anything else -> Max
    tools: list | None = None
    tool_choice: Any = None


class CompReq(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int | None = 1024
    temperature: float | None = 0.7
    top_p: float | None = 0.9
    top_k: int | None = 50
    stream: bool | None = False
    stop: Any = None


def _sampler(temp, top_p, top_k):
    if not temp or temp <= 0:
        return GreedySampler()
    ss = [SS_Temperature(temp)]
    if top_k: ss.append(SS_TopK(int(top_k)))
    if top_p and top_p < 1.0: ss.append(SS_TopP(float(top_p)))
    ss.append(SS_Sample())
    return CustomSampler(ss)


# The generator stops on EOS by default; these are the template's turn markers, added so a
# stray continuation cannot run to max_tokens.
DEFAULT_STOPS = ["<|user|>", "<|observation|>", "<|endoftext|>"]


def _stops(stop):
    extra = [] if stop is None else ([stop] if isinstance(stop, str) else list(stop))
    return DEFAULT_STOPS + extra


def normalize_messages(messages):
    """Convert OpenAI-shaped history into what the chat template expects.

    The template renders an assistant tool call as

        '<tool_call>' + tc.name  ... for k, v in tc.arguments.items()

    i.e. a FLAT name and a dict of arguments. OpenAI sends the nested
    {"function": {"name": ..., "arguments": "<json string>"}} shape, so rendering a returned
    tool call raised

        jinja2.exceptions.UndefinedError: 'str object' has no attribute 'items'

    which surfaced as a 502 on the turn after a successful tool call -- the crash is in
    rendering the history, not in the call itself.
    """
    out = []
    for m in messages or []:
        m = dict(m)
        tcs = m.get("tool_calls")
        if tcs:
            flat = []
            for tc in tcs:
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                args = fn.get("arguments", tc.get("arguments") if isinstance(tc, dict) else {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except (ValueError, TypeError):
                        args = {"input": args}      # unparseable: keep it, do not crash
                if not isinstance(args, dict):
                    args = {"input": args}
                e = {"name": fn.get("name") or tc.get("name"), "arguments": args}
                if tc.get("id"):
                    e["id"] = tc["id"]
                    e["tool_call_id"] = tc["id"]
                flat.append(e)
            m["tool_calls"] = flat
        # tool results: the template calls visible_text() on content, so keep it a string
        if m.get("role") == "tool":
            c = m.get("content")
            if c is None:
                m["content"] = ""
            elif not isinstance(c, (str, list, dict)):
                m["content"] = str(c)
        out.append(m)
    return out


def _prompt_from_messages(messages, reasoning_effort=None):
    """Render via the model's own chat template.

    exl3's Tokenizer exposes hf_render_chat_template, not HF's apply_chat_template -- calling
    the latter silently fell through to a plain "role: content" transcript, and the model then
    kept writing both sides of the conversation because nothing in that text tells it where to
    stop.

    GLM-5.3's template reads reasoning_effort and emits "<|system|>Reasoning Effort: X". Its
    own rule is low/high honoured, anything else -> Max, so "max" gives maximum reasoning.
    """
    tok = STATE["tokenizer"]
    messages = normalize_messages(messages)
    eff = reasoning_effort or STATE.get("reasoning_effort") or "max"
    try:
        return tok.hf_render_chat_template(messages, add_generation_prompt=True,
                                           reasoning_effort=eff)
    except TypeError:
        # template does not take the kwarg
        return tok.hf_render_chat_template(messages, add_generation_prompt=True)
    except Exception as e:
        # A template error must not become a 502 with no body: report it as a 400 naming the
        # offending message, which is debuggable from the client side.
        raise HTTPException(status_code=400,
                            detail=f"chat template failed to render: {type(e).__name__}: {e}")


def _run(prompt_text, max_tokens, sampler, stops):
    """Blocking generation. Held under GEN_LOCK by the caller."""
    gen, tok = STATE["gen"], STATE["tokenizer"]
    ids = tok.encode(prompt_text)
    job = Job(input_ids=ids, max_new_tokens=int(max_tokens or 1024),
              stop_conditions=stops, sampler=sampler)
    gen.enqueue(job)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("text"):
                yield r["text"], None
            if r.get("eos"):
                yield "", {"prompt_tokens": int(r.get("prompt_tokens", 0)),
                           "completion_tokens": int(r.get("new_tokens", 0))}


# The chat template opens the assistant turn with "<think>", so generation STARTS inside the
# reasoning block and the model emits "...reasoning...</think>...answer...". Clients expect
# those separated (reasoning_content vs content), the way vLLM's reasoning parser does it;
# left inline, the reasoning renders as part of the reply.
THINK_CLOSE = "</think>"


class ThinkSplitter:
    """Streaming splitter: routes text to reasoning until </think>, then to content.

    The close tag can arrive split across chunks, so a tail of len(tag)-1 characters is held
    back until it is known not to be a prefix of the tag.
    """

    def __init__(self):
        self.in_think = True
        self.buf = ""

    def feed(self, text):
        """-> (reasoning_delta, content_delta)"""
        if not self.in_think:
            return "", text
        self.buf += text
        i = self.buf.find(THINK_CLOSE)
        if i >= 0:
            reasoning, content = self.buf[:i], self.buf[i + len(THINK_CLOSE):]
            self.buf, self.in_think = "", False
            return reasoning, content
        # hold back anything that could still become the close tag
        keep = 0
        for n in range(min(len(THINK_CLOSE) - 1, len(self.buf)), 0, -1):
            if THINK_CLOSE.startswith(self.buf[-n:]):
                keep = n
                break
        out, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        return out, ""

    def flush(self):
        """Whatever is left when generation ends without a close tag."""
        rest, self.buf = self.buf, ""
        return (rest, "") if self.in_think else ("", rest)


def split_think(text):
    """Non-streaming: -> (reasoning, content)."""
    i = text.find(THINK_CLOSE)
    if i < 0:
        return "", text
    return text[:i].strip(), text[i + len(THINK_CLOSE):].lstrip()


# ---------------------------------------------------------------------------------------
# Tool calling.
#
# The chat template renders tool CALLS and tool RESPONSES but has no `tools` variable, so it
# cannot declare what is available -- schemas have to be injected into the system turn. The
# model's native call format is the registered special tokens
#
#     <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
#
# not JSON. Without schemas it invents both a format and the function names (observed:
# <|toolCall|>{"name":"getWeatherByCity",...}, for a tool actually called get_weather).
# ---------------------------------------------------------------------------------------

TOOLCALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def _tool_spec(tools):
    lines = ["# Tools", "",
             "You have access to the following functions. Call one when it helps answer the "
             "user; otherwise answer directly.", ""]
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else t
        lines.append("## " + str(fn.get("name")))
        if fn.get("description"):
            lines.append(fn["description"])
        lines.append("Parameters (JSON Schema):")
        lines.append(json.dumps(fn.get("parameters", {}), ensure_ascii=False))
        lines.append("")
    lines += ["To call a function, emit exactly this and nothing else:", "",
              "<tool_call>FUNCTION_NAME"
              "<arg_key>ARGUMENT_NAME</arg_key><arg_value>VALUE</arg_value>"
              "</tool_call>", "",
              "Use one <arg_key>/<arg_value> pair per argument. Use the exact function and "
              "argument names given above."]
    return "\n".join(lines)


def _with_tools(messages, tools, tool_choice):
    """Prepend the tool spec to the system turn (creating one if absent)."""
    if not tools:
        return messages
    spec = _tool_spec(tools)
    if isinstance(tool_choice, dict):
        want = (tool_choice.get("function") or {}).get("name")
        if want:
            spec += f"\n\nYou must call the function `{want}` for this turn."
    elif tool_choice == "required":
        spec += "\n\nYou must call one of the functions above for this turn."
    elif tool_choice == "none":
        return messages
    out = list(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            out[i] = dict(m, content=spec + "\n\n" + str(m.get("content", "")))
            return out
    return [{"role": "system", "content": spec}] + out


def parse_tool_calls(text):
    """-> (clean_text, [openai tool_call, ...])"""
    calls = []
    for block in TOOLCALL_RE.findall(text or ""):
        args = {}
        for k, v in ARG_RE.findall(block):
            v = v.strip()
            try:
                args[k.strip()] = json.loads(v)      # numbers, bools, objects
            except (ValueError, TypeError):
                args[k.strip()] = v                   # plain string
        name = ARG_RE.split(block)[0]
        name = name.split("<arg_key>")[0].strip().strip("\n")
        if not name:
            continue
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name,
                                   "arguments": json.dumps(args, ensure_ascii=False)}})
    clean = TOOLCALL_RE.sub("", text or "").strip()
    return clean, calls


TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"


class ToolCallStreamer:
    """Splits a content stream into plain text and completed tool calls.

    Text before a call streams through unchanged. Once <tool_call> is seen the block is
    buffered until </tool_call> and then emitted whole -- a call is only useful complete, and
    the arguments arrive interleaved with their keys, so there is nothing coherent to emit
    incrementally. Both tags can straddle chunk boundaries, so a tail that could still become
    an opening tag is held back the same way the reasoning splitter does it.
    """

    def __init__(self):
        self.buf = ""
        self.in_call = False
        self.n = 0

    def feed(self, text):
        """-> (text_delta, [tool_call, ...])"""
        self.buf += text
        out_text, calls = "", []
        while True:
            if not self.in_call:
                i = self.buf.find(TOOL_OPEN)
                if i < 0:
                    keep = 0
                    for k in range(min(len(TOOL_OPEN) - 1, len(self.buf)), 0, -1):
                        if TOOL_OPEN.startswith(self.buf[-k:]):
                            keep = k
                            break
                    out_text += self.buf[:len(self.buf) - keep]
                    self.buf = self.buf[len(self.buf) - keep:]
                    return out_text, calls
                out_text += self.buf[:i]
                self.buf = self.buf[i:]
                self.in_call = True
            j = self.buf.find(TOOL_CLOSE)
            if j < 0:
                return out_text, calls
            block = self.buf[:j + len(TOOL_CLOSE)]
            self.buf = self.buf[j + len(TOOL_CLOSE):]
            self.in_call = False
            _, parsed = parse_tool_calls(block)
            for c in parsed:
                c["index"] = self.n
                self.n += 1
                calls.append(c)

    def flush(self):
        rest, self.buf = self.buf, ""
        if self.in_call:
            # generation stopped mid-call; salvage what parses rather than leaking tags
            _, parsed = parse_tool_calls(rest + TOOL_CLOSE)
            for c in parsed:
                c["index"] = self.n
                self.n += 1
            self.in_call = False
            return "", parsed
        return rest, []


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def _stream_response(prompt_text, req, kind):
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = STATE["model_name"]

    def gen_iter():
        with GEN_LOCK:
            if kind == "chat":
                yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"role": "assistant"},
                                         "finish_reason": None}]})
            sp = ThinkSplitter()
            tc = ToolCallStreamer()
            saw_tool = {"any": False}

            def emit_calls(calls):
                for c in calls:
                    saw_tool["any"] = True
                    yield _sse({"id": cid, "object": "chat.completion.chunk",
                                "created": created, "model": model,
                                "choices": [{"index": 0, "delta": {"tool_calls": [c]},
                                             "finish_reason": None}]})

            def emit(reasoning, content):
                if kind != "chat":
                    # /v1/completions has no reasoning channel; keep the raw stream intact
                    if reasoning or content:
                        yield _sse({"id": cid, "object": "text_completion",
                                    "created": created, "model": model,
                                    "choices": [{"index": 0, "text": reasoning + content,
                                                 "finish_reason": None}]})
                    return
                if reasoning:
                    yield _sse({"id": cid, "object": "chat.completion.chunk",
                                "created": created, "model": model,
                                "choices": [{"index": 0,
                                             "delta": {"reasoning_content": reasoning},
                                             "finish_reason": None}]})
                if content:
                    txt, calls = tc.feed(content)
                    if txt:
                        yield _sse({"id": cid, "object": "chat.completion.chunk",
                                    "created": created, "model": model,
                                    "choices": [{"index": 0, "delta": {"content": txt},
                                                 "finish_reason": None}]})
                    yield from emit_calls(calls)

            for text, usage in _run(prompt_text, req.max_tokens,
                                    _sampler(req.temperature, req.top_p, req.top_k),
                                    _stops(req.stop)):
                if text:
                    r, c = sp.feed(text)
                    yield from emit(r, c)
            r, c = sp.flush()
            yield from emit(r, c)
            if kind == "chat":
                txt, calls = tc.flush()
                if txt:
                    yield _sse({"id": cid, "object": "chat.completion.chunk",
                                "created": created, "model": model,
                                "choices": [{"index": 0, "delta": {"content": txt},
                                             "finish_reason": None}]})
                yield from emit_calls(calls)
            fin = {"id": cid, "created": created, "model": model,
                   "choices": [{"index": 0,
                                "finish_reason": "tool_calls" if saw_tool["any"] else "stop"}]}
            if kind == "chat":
                fin["object"] = "chat.completion.chunk"; fin["choices"][0]["delta"] = {}
            else:
                fin["object"] = "text_completion"; fin["choices"][0]["text"] = ""
            yield _sse(fin)
            yield "data: [DONE]\n\n"

    return StreamingResponse(gen_iter(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE.get("model_name"),
            "profile": STATE.get("profile"), "mcs": STATE.get("mcs"),
            "reasoning_effort": STATE.get("reasoning_effort"),
            "max_context": STATE.get("max_ctx")}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": STATE["model_name"], "object": "model",
                                        "created": STATE["started"], "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    msgs = _with_tools(req.messages, req.tools, req.tool_choice)
    text = _prompt_from_messages(msgs, req.reasoning_effort)
    if req.stream:
        return _stream_response(text, req, "chat")
    out, usage = [], {}
    with GEN_LOCK:
        for t, u in _run(text, req.max_tokens, _sampler(req.temperature, req.top_p, req.top_k),
                         _stops(req.stop)):
            out.append(t)
            if u: usage = u
    reasoning, content = split_think("".join(out))
    content, tool_calls = parse_tool_calls(content)
    msg = {"role": "assistant", "content": content or None}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion",
        "created": int(time.time()), "model": STATE["model_name"],
        "choices": [{"index": 0, "message": msg,
                     "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {**usage, "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)}})


@app.post("/v1/completions")
def completions(req: CompReq):
    if req.stream:
        return _stream_response(req.prompt, req, "text")
    out, usage = [], {}
    with GEN_LOCK:
        for t, u in _run(req.prompt, req.max_tokens,
                         _sampler(req.temperature, req.top_p, req.top_k), _stops(req.stop)):
            out.append(t)
            if u: usage = u
    return JSONResponse({
        "id": f"cmpl-{uuid.uuid4().hex[:24]}", "object": "text_completion",
        "created": int(time.time()), "model": STATE["model_name"],
        "choices": [{"index": 0, "text": "".join(out), "finish_reason": "stop"}],
        "usage": {**usage, "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(p, cache=True, default_cache_size=270336,
                        default_autosplit_max_batch_size=1)
    p.add_argument("-host", default="127.0.0.1")
    p.add_argument("-port", type=int, default=8095)
    p.add_argument("-served-name", default="glm-5.3-flash")
    p.add_argument("-reasoning-effort", default="max",
                   help="default per-request reasoning effort: low | high | max (default)")
    args = p.parse_args()

    print(" -- loading model, this takes several minutes at 165 GB", flush=True)
    model, config, cache, tokenizer, *_ = model_init.init(args, max_chunk_size=4096)
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer, max_chunk_size=4096)
    STATE.update(model=model, cache=cache, tokenizer=tokenizer, gen=gen,
                 model_name=args.served_name, started=int(time.time()),
                 profile=getattr(args, "moe_cpu_profile", None),
                 mcs=getattr(args, "moe_cpu_split", None),
                 reasoning_effort=args.reasoning_effort,
                 max_ctx=args.cache_size)
    print(f" -- ready on {args.host}:{args.port} as '{args.served_name}'", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
