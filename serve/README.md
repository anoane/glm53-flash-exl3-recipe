# Serving GLM-5.3-Flash over an OpenAI-compatible API

`serve_openai.py` is a small OpenAI-compatible server for exllamav3 with MoE CPU offload. It
exists because the profile flags (`-mcp`, `-mcs`, …) are passed straight through
`model_init.add_args`, so any exl3 CLI flag works, and because reasoning and tool calls need
model-specific handling that a generic server does not know about.

Endpoints: `/v1/chat/completions`, `/v1/completions` (both streaming and not), `/v1/models`,
`/health`.

## Running

```bash
MODEL_DIR=/models/GLM-5.3-Flash-exl3-4.05bpw ./serve.sh
```

Or as a service:

```bash
install -m644 glm53-sec.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now glm53-sec
```

TLS and auth are the reverse proxy's job — see `Caddyfile.example`. The backend binds
`127.0.0.1` and has no auth of its own.

Loading 165 GB takes about five minutes, during which the proxy returns **502**. That is a
load window, not a crash; `/health` answering is the ready signal, and `journalctl -u
glm53-sec` shows progress.

## The configuration, and why

```
-cs 1048576          1M context. KV here is 17,600 B/token (11 KV layers x 1600 B), so 1M
                     is 17.2 GiB and residency has to drop to 132 to pay for it.
-mcs 156             132 of 288 experts resident. 136 resident OOMs at 1M inside the KDA
                     prefill scratch; 132 is the measured fit.
-mcp code_injection  security code review. 79.8% held-out capture at R=132. secure_coding,
     -mcpm static    the intuitive choice, measures 62.6% -- see ../moe_profiles/README.md.
-reasoning-effort    default per-request reasoning level.
```

Expect ~33 tok/s at long context. 1M costs about 17% against the 270k configuration, because
residency falls from 160 to 132.

Single in-flight generation by design: decode is host-DRAM-bound, so concurrency trades every
individual request's latency for aggregate throughput. A lock serialises rather than
interleaves.

## Reasoning

The chat template opens the assistant turn with `<think>`, so generation starts inside the
reasoning block and emits `…reasoning…</think>…answer…`. The server splits that:
`reasoning_content` and `content` are separate fields, and separate deltas when streaming.
Left inline, the reasoning renders as part of the reply.

Effort is per request: `"reasoning_effort": "low" | "high" | "max"`. The template honours
`low` and `high` and maps anything else to Max, so **thinking cannot be switched off** on this
model — `off` is not available, only quieter.

## Tool calling

Pass `tools` and `tool_choice` exactly as OpenAI defines them. Three things are handled that a
generic server gets wrong here:

**Schemas must be injected.** The chat template renders tool *calls* and *responses* but has
no `tools` variable, so there is no path for definitions to reach the model. Without them it
invents both the format and the names — observed: `<|toolCall|>{"name":"getWeatherByCity"}`
for a function actually called `get_weather`, with an argument that was never in the schema.
The server writes the schemas into the system turn.

**The call format is special tokens, not JSON.**

    <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value></tool_call>

These are parsed into OpenAI `tool_calls`, with values JSON-decoded so numbers and booleans
arrive typed. Streaming emits `delta.tool_calls`; a call is buffered until `</tool_call>` and
emitted whole, because keys and values interleave and there is no coherent partial JSON to
send. Both tags may straddle chunk boundaries, so a tail that could still become a tag is
held back.

**History must be normalised on the way back in.** The template wants a flat `tc.name` and
`tc.arguments` as a dict; OpenAI sends nested `function.name` with `arguments` as a JSON
string. Rendering the OpenAI shape directly raises

    jinja2.exceptions.UndefinedError: 'str object' has no attribute 'items'

which appears as a 502 on the turn *after* a successful tool call — the crash is in rendering
the history, not in the call. The server converts the shape before rendering, and any template
failure is returned as a 400 with the exception text rather than dropping the connection.

## Client configuration

For an OpenAI-style client, the model is a normal provider with reasoning support:

```json
{
  "baseUrl": "https://sec.example.com:8443/v1",
  "api": "openai-completions",
  "models": [{
    "id": "glm-5.3-flash-sec",
    "reasoning": true,
    "contextWindow": 1048576,
    "maxTokens": 65536,
    "thinkingLevelMap": {
      "off": "low", "minimal": "low", "low": "low",
      "medium": "high", "high": "high", "xhigh": "max", "max": "max"
    },
    "compat": {
      "supportsReasoningEffort": true,
      "supportsUsageInStreaming": false,
      "maxTokensField": "max_tokens"
    }
  }]
}
```

`thinkingLevelMap` collapses seven levels onto three because the template honours only `low`
and `high`. `supportsUsageInStreaming` is false: usage is returned on the non-streaming path
only.
