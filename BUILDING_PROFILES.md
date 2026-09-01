# Building expert-placement profiles

A profile is a per-expert selection census measured offline. It is valid for exactly one
`(model, checkpoint)` pair and is only worth its load time if it matches the workload on two
axes — **domain** and **context length**. Both were measured, and getting either wrong throws
away the entire benefit:

* a prose census applied to code is worth **1.01x** — nothing
* a code census on code is worth **1.49x**
* a census fitted on 1k windows loses most of its value by 256k, and this is a length effect
  rather than a content artifact: holding the trailing 4096 tokens byte-identical and varying
  only the preceding context still moved cold rate 20.8% -> 28.6%

## 1. Assemble a corpus

Anything the tokenizer can read: a `.txt`/`.utf8` file, or a `.json` list of strings.

    python tools/make_code_corpus.py /path/to/src -o code_corpus.txt

`make_code_corpus.py` walks a source tree deterministically (sorted, skipping `.git`,
`build`, `node_modules`, …) so the same tree yields byte-identical output and the profile is
reproducible.

For anything else, concatenate. The profiles shipped here were grouped from 25 calibration
categories into coherent domains, because routing is shared across related material and
because a corpus only yields `total_tokens / plen` independent windows:

```python
GROUPS = {
 "security":  ["app_cracking","code_injection","ctf_pwn","exploit_dev","memcorruption",
               "red_teaming","web_hacking","rev_eng","fuzzing"],
 "systems":   ["c_standards","c_commented","cpp_commented","os_internals","firmware",
               "mcu_sbc","hardware"],
 "embedded":  ["robotics","rf_sdr","mcu_sbc","firmware"],
 "swe":       ["swe_arch","swe_arch_text","dev_tooling","secure_coding","lang_breadth",
               "rust_cpp_go"],
 "cryptography": ["crypto","secure_coding"],
}
for name, members in GROUPS.items():
    open(f"{name}.txt", "w").write("".join(
        f"# ==== {m} ====\n" + open(f"calib_corpus/{m}.utf8").read() + "\n" for m in members))
```

The `agentic` profile is different in kind and is the one most likely to match production
traffic: 111 real task prompts across 17 families, used as a JSON list of strings.

```python
import json
d = json.load(open("p2_campaign_prompts.json"))
json.dump([x["prompt"] for x in d], open("agentic.json", "w"))
```

## 2. Size the windows

    total_tokens / plen  =  how many independent windows you get

`-plen` should sit near the context you intend to serve, but the held-out split needs at
least **4** windows to mean anything, and the builder refuses to report capture below that
rather than quietly printing an in-sample number. Small corpora therefore take smaller
windows rather than too few of them:

| corpus | size | `-nprompts` | `-plen` | why |
|---|---:|---:|---:|---|
| security | 9.4 MB | 12 | 65536 | ample; long-context profile |
| systems | 5.4 MB | 12 | 65536 | ample |
| swe | 4.8 MB | 12 | 65536 | ample |
| embedded | 3.3 MB | 12 | 65536 | just fits (786k tokens needed) |
| cryptography | 1.3 MB | 8 | 32768 | 64k would give only 5 windows |
| agentic | 0.8 MB | 20 | 8192 | short prompts; a short-context profile |

`cryptography` and `agentic` are **short-context profiles** and will underperform at 256k,
the same way a 1k-window census does. That is a property of the corpus, not a defect.

## 3. Build

One corpus:

```bash
python tools/moe_profile_build.py -m /models/GLM-5.3-Flash-exl3-4.05bpw \
    -cs 98304 -mcs 144 \
    -corpus security.txt -o security.exl3moe \
    -nprompts 12 -plen 65536 -gen 192 -resident 160
```

Several corpora — use `tools/build_many.py`, which loads the model **once**. Loading this
model with offload takes ~5 minutes, so six separate runs waste ~25 minutes on nothing:

```bash
python tools/build_many.py -m /models/GLM-5.3-Flash-exl3-4.05bpw \
    -cs 98304 -mcs 144 -outdir ./profiles -resident 160
```

Edit the `JOBS` table at the top to change the corpus list and per-corpus window sizes.

Notes on the flags:

* `-mcs 144` — the builder needs the model loaded, and this model does not fit resident.
  Placement does not affect routing, so any working `-mcs` gives the same census.
* `-gen 192` — decode tokens per window. Decode is the allocation signal, since that is the
  regime offload runs in. Prefer more windows over longer generations: diversity is what the
  held-out split measures.
* `-resident N` — the residency the capture report is scored at. Set it to what you will
  actually serve (`288 - mcs`), or the number will describe a configuration you never run.
* Building takes ~90 s per 64k window. Six profiles at the sizes above is ~85 minutes.

## 4. Read the capture report

```
       residency   held-out   uniform   oracle   in-sample
     160/288  55.6%      74.0%     55.6%    86.3%       82.4%
```

| column | meaning |
|---|---|
| **held-out** | capture on windows the ranking never saw. **This is the number.** |
| uniform | what random placement scores (`R/E`). At this level the profile is worthless. |
| oracle | fit on the held-out set itself: the ceiling for *any* static profile. |
| in-sample | scored on the fit set. Always flattering; ignore it. |

The tool warns when held-out sits within 3 points of uniform, and when in-sample exceeds
held-out by more than 25 points. Both mean the ranking has not generalized.

Sanity-check against throughput with `step_ms ≈ 10.8 + 0.77 × cold_percent`: a profile that
moves held-out capture from 55.6% (random) to 74% takes cold rate from 44.4% to 26%, i.e.
44.9 ms → 30.8 ms per token.

## 5. Audit later

The `.exl3moe` carries the per-prompt census in its own byte range, so a profile can be
re-scored at any time without a model:

```bash
python tools/moe_profile_build.py -score security.exl3moe -resident 160
```

Use this when the serving residency changes — a profile scored at `R=160` says nothing about
how it behaves at `R=108`.

## 6. Verify against the workload

The capture number is measured on the corpus, not on your traffic. Confirm live:

```bash
EXL3_MOE_HANDOFF_PROF=1 python tools/ctxtest.py -m /models/GLM \
    -cs 270336 -mcs 128 -mcp security -mcpm static -mode fit -corpus your_text.txt
```

Read `cpu-assign/row` from the handoff profiler: divided by `topk` (8) that is the live cold
rate. If it does not roughly match `1 - held_out`, the profile does not match the traffic.
