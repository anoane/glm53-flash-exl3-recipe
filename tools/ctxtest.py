"""Two tests, both through the Generator on real held-out text.

  fit   : does -mcs 144 survive a REAL 262144 prefill, and what does it buy?
  discr : does cold rate rise with context LENGTH, or does it just track WHICH text is
          being read? Fixes the final window and varies only the padding before it.

Measurement hygiene applied throughout (each of these has burned us):
  - real wikitext, never synthetic filler (synthetic inverts the sign of a profile A/B)
  - warmup ABOVE 2048 tokens so the regime 0->1 DSA JIT + graph capture is paid first
  - first generation after load is discarded
  - Generator, never perf.py (get_test_state(clear=True) zeroes 34/45 recurrent layers)
  - cpu-assign/row reported beside every tok/s; tok/s alone cannot tell placement from warmup
  - real prefills only, never a fabricated past_len
"""
import sys, os, time, argparse, json
sys.path.insert(0, "/opt/exllamav3")
import torch
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler import GreedySampler

def corpus_tokens(tokenizer, path):
    txt = open(path, encoding="utf8", errors="replace").read()
    ids = tokenizer.encode(txt).reshape(-1)
    print(f" -- corpus {os.path.basename(path)}: {len(txt):,} bytes -> {ids.shape[0]:,} tokens "
          f"({len(txt)/max(ids.shape[0],1):.2f} B/tok)")
    return ids

def measure(gen, ids, ngen, label):
    job = Job(input_ids=ids.reshape(1, -1), max_new_tokens=ngen,
              stop_conditions=[], sampler=GreedySampler())
    t0 = time.time(); gen.enqueue(job); res = None; ttft = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if ttft is None and r.get("text"): ttft = time.time() - t0
            if r.get("eos"): res = r
    pt, nt = res["prompt_tokens"], res["new_tokens"]
    tp, tg = res["time_prefill"], res["time_generate"]
    cached = res.get("cached_tokens", 0)
    out = {"label": label, "prompt": int(pt), "cached": int(cached), "gen": int(nt),
           "prefill_s": round(tp, 2), "prefill_tps": round(pt / tp, 1) if tp else 0,
           "ttft_s": round(ttft or 0, 2),
           "decode_tps": round(nt / tg, 3) if tg else 0,
           "ms_per_tok": round(1000 * tg / nt, 2) if nt else 0}
    print("RESULT " + json.dumps(out)); sys.stdout.flush()
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(p, cache=True, default_cache_size=270336,
                        default_autosplit_max_batch_size=1)
    p.add_argument("-mode", required=True, choices=["fit", "discr"])
    p.add_argument("-corpus", default="/w/wikicache/wikitext-2-raw/wiki.test.raw")
    p.add_argument("-gen2", type=int, default=128)
    p.add_argument("-chunk", type=int, default=4096)
    args = p.parse_args()

    model, config, cache, tokenizer, *_ = model_init.init(args, max_chunk_size=args.chunk)
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer, max_chunk_size=args.chunk)
    toks = corpus_tokens(tokenizer, args.corpus)
    N = toks.shape[0]

    # Warm above index_topk=2048 so the regime-1 Triton JIT + graph capture is not billed
    # to a measured point, then discard the whole first generation.
    measure(gen, toks[:4096], 32, "WARMUP_DISCARD")

    if args.mode == "fit":
        # Does -mcs 144 survive a real 262144 prefill, and what is decode worth?
        for ctx in (32768, 262144):
            if N < ctx:
                print(f" !! corpus has {N:,} tokens, need {ctx:,} -- SKIPPING (never pad by "
                      f"repeating: a duplicated tail depresses cold rate)"); continue
            measure(gen, toks[:ctx], args.gen2, f"fit|ctx{ctx}|rep1")
        # drift control: identical context again, fully prefix-cached
        if N >= 262144:
            measure(gen, toks[:262144], args.gen2, "fit|ctx262144|rep2")

    else:
        # Length-vs-position discriminator. The FINAL 4096 tokens are byte-identical in every
        # arm, so decode starts from the same immediate context; only the amount of preceding
        # text changes. If cold rate tracks total length, it rises across arms. If it tracks
        # which text is being read, it stays flat.
        W = 4096
        tail = toks[N - W:]                      # fixed window, same in all three arms
        for pad in (0, 65536, 192512):
            if N < pad + W:
                print(f" !! need {pad+W:,} tokens, have {N:,} -- SKIPPING"); continue
            ids = torch.cat([toks[:pad], tail]) if pad else tail
            measure(gen, ids, args.gen2, f"discr|pad{pad}|ctx{ids.shape[0]}")
    print("CTXTEST_DONE")
