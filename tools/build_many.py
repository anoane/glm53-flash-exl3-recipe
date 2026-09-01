"""Build several MoE placement profiles from ONE model load.

Loading GLM-5.3-Flash at 4.05bpw with offload takes ~5 minutes; building N profiles
separately pays that N times. The routing hooks are per-module and cheap to re-arm, so the
model is loaded once and each corpus is profiled in turn.

Window size is per-corpus on purpose. -plen should sit near the context the profile will
serve, but a corpus only yields total_tokens/plen independent windows, and the held-out
split needs at least 4 to mean anything. Small corpora therefore get smaller windows rather
than too few of them.
"""
import sys, os, json, time
sys.path.insert(0, "/opt/exllamav3")
sys.path.insert(0, "/w")
import numpy as np, torch
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler import GreedySampler
import argparse
import moe_profile_build as B
import moe_profile_pack as P

JOBS = [
    # name        corpus                       nprompts  plen    gen
    ("security",  "/w/corpora/security.txt",         12, 65536, 192),
    ("systems",   "/w/corpora/systems.txt",          12, 65536, 192),
    ("swe",       "/w/corpora/swe.txt",              12, 65536, 192),
    ("embedded",  "/w/corpora/embedded.txt",         12, 65536, 192),
    ("cryptography", "/w/corpora/cryptography.txt",            8, 32768, 192),
    ("agentic",   "/w/corpora/agentic.json",         20,  8192, 160),
]

if __name__ == "__main__":
    p = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(p, cache=True, default_cache_size=98304,
                        default_autosplit_max_batch_size=1)
    p.add_argument("-outdir", required=True)
    p.add_argument("-resident", type=int, default=160)
    args = p.parse_args()

    for v in ("EXL3_MOE_PROFILE", "EXL3_MOE_CPU_SPLIT_STATS"):
        os.environ.pop(v, None)

    model, config, cache, tokenizer, *_ = model_init.init(args, max_chunk_size=4096)
    mods = B.find_moe(model)
    L, E = len(mods), mods[0].num_experts
    idx_of = {m.key: i for i, m in enumerate(mods)}
    print(f" -- {L} MoE layers x {E} experts", flush=True)

    gen = Generator(model=model, cache=cache, tokenizer=tokenizer, max_chunk_size=4096)
    os.makedirs(args.outdir, exist_ok=True)

    for name, corpus, nprompts, plen, ngen in JOBS:
        t0 = time.time()
        try:
            texts, label = B.load_corpus(corpus)
            windows = B.make_windows(texts, tokenizer, nprompts, plen)
            P_ = len(windows)
            dec = np.zeros((P_, L, E), dtype=np.int64)
            pre = np.zeros((P_, L, E), dtype=np.int64)
            state = {"pi": 0}

            originals = []
            for m in mods:
                orig = m.routing_fn
                originals.append((m, orig))
                def make(mod, fn):
                    def wrapper(bsz, cfg, z, params):
                        sel, w = fn(bsz, cfg, z, params)
                        try:
                            e = sel.detach().reshape(-1).to(torch.int64).cpu().numpy()
                            rows = sel.shape[0] if sel.dim() > 1 else 1
                            bank = dec if rows == 1 else pre
                            np.add.at(bank[state["pi"], idx_of[mod.key]], e, 1)
                        except Exception:
                            pass
                        return sel, w
                    return wrapper
                m.routing_fn = make(m, orig)

            print(f"\n=== {name}: {P_} windows x {plen} tok + {ngen} decode ===", flush=True)
            for pi, ids in enumerate(windows):
                state["pi"] = pi
                gen.enqueue(Job(input_ids=ids, max_new_tokens=ngen,
                                stop_conditions=[], sampler=GreedySampler()))
                while gen.num_remaining_jobs():
                    gen.iterate()
                if (pi + 1) % 4 == 0 or pi == P_ - 1:
                    print(f"    window {pi+1}/{P_}  ({time.time()-t0:.0f}s)", flush=True)

            for m, orig in originals:
                m.routing_fn = orig

            npz = os.path.join(args.outdir, name + ".npz")
            np.savez_compressed(npz, counts_decode=dec, counts_prefill=pre)
            from exllamav3.model.moe_profile import model_fingerprint
            fp = model_fingerprint(config, E)
            rep = B.capture_report(dec.astype(np.float64), args.resident, 0.34)
            meta = {"corpus": label, "layers": L, "experts": E, "prompts": P_,
                    "fingerprint": fp, "prompt_tokens": plen, "gen_tokens": ngen,
                    "layer_keys": [m.key for m in mods],
                    "decode_hits": int(dec.sum()), "prefill_hits": int(pre.sum())}
            if rep:
                h = rep["head"]
                meta["capture"] = {"resident": args.resident, "bank": "decode",
                                   "held_out": round(h["held_out"], 4),
                                   "in_sample": round(h["in_sample"], 4),
                                   "oracle": round(h["oracle"], 4),
                                   "uniform": round(h["uniform"], 4),
                                   "n_fit": rep["n_fit"], "n_test": rep["n_test"]}
                print(f"    capture @R={args.resident}: held-out {100*h['held_out']:.1f}% "
                      f"(uniform {100*h['uniform']:.1f}%, oracle {100*h['oracle']:.1f}%)", flush=True)
            json.dump(meta, open(os.path.join(args.outdir, name + ".meta.json"), "w"), indent=2)
            info = P.pack(npz, os.path.join(args.outdir, name + ".exl3moe"))
            os.remove(npz)
            print(f"    wrote {name}.exl3moe  {info['bytes']:,} bytes  ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"    !! {name} FAILED: {type(e).__name__}: {e}", flush=True)
    print("\nBUILD_MANY_DONE", flush=True)
