# GLM-5.3-Flash exl3 4.05bpw on one 96 GB GPU

Serving recipe for `turboderp/GLM-5.3-Flash-exl3` at 4.05bpw with MoE CPU offload, plus the
precomputed expert-placement profiles and the tooling to rebuild them.

The model is ~165 GB of weights against 96 GB of VRAM, so most experts live in system RAM and
the CPU computes them. Which experts stay resident decides almost everything about decode
speed: the step is host-DRAM-bound on cold expert weights at every context length.

## Documentation

| | |
|---|---|
| **[`moe_profiles/README.md`](moe_profiles/README.md)** | **The profile catalogue.** All 27 profiles: what corpus each was built from, its held-out capture and headroom, grouped by use (general / systems / security / engineering). Includes the broad-vs-narrow evidence and why `memcorruption` is not shipped. **Start here to pick a profile.** |
| **[`LOADING_PROFILES.md`](LOADING_PROFILES.md)** | **How to load them.** The four ways to name a profile — by name, by absolute path, from an external directory (`-mcpd` / `$EXL3_MOE_PROFILE_DIR`, incl. read-only container mounts), and several at once with weights. Also: when merging helps and when it reproduces the broad-group failure, `static` vs `seed`, checkpoint identity and `-mcpq`, and how to confirm a profile is actually working from `cpu-assign/row`. |
| **[`serve/README.md`](serve/README.md)** | **Serving it.** An OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, streaming) with reasoning split into `reasoning_content` and full tool-call support, plus a systemd unit and a reverse-proxy example for TLS and bearer auth. Documents the three model-specific things a generic server gets wrong here: tool schemas have to be injected because the chat template has no `tools` variable, calls come back as special tokens rather than JSON, and OpenAI-shaped history must be normalised before rendering or the template raises. |
| **[`BUILDING_PROFILES.md`](BUILDING_PROFILES.md)** | **How to build your own.** Assembling a corpus, sizing windows against corpus length (a corpus yields `total_tokens / plen` independent windows and the held-out split needs at least 4), building one or many profiles from a single model load, reading the capture report against `uniform` and `oracle`, and re-auditing a shipped profile later with `-score`. |

Everything below is the recipe itself: measurements, flags, and the settings to run.

## Measured

exllamav3 dev `0531096`. GLM-5.3-Flash-exl3 4.05bpw: 45 layers, 42 MoE, 288 experts, topk 8.
RTX PRO 6000 Blackwell 96 GB + Ryzen 9 9950X3D / 128 GB DDR5 (dual channel, ~57.5 GB/s
measured). Driven through the Generator on held-out text, `-cs 270336`, warm, first
generation discarded.

    BEFORE  -mcs 180                                stock: dynamic placement, no profile
    AFTER   -mcs 128 -mcp <profile> -mcpm static

| workload | context | before | after | |
|---|---:|---:|---:|---:|
| prose | 32,768  | 17.01 tok/s | **32.91** | 1.93x |
| prose | 262,144 | 16.82 / 16.95 | **40.24 / 39.71** | **2.37x** |
| code  | 32,768  | 16.82 | **32.78** | 1.95x |
| code  | 262,144 | 17.11 / 17.42 | **33.66 / 34.01** | 1.96x |

Prefill improves too: 573 -> 771 tok/s (prose 32k), 696 -> 889 (prose 256k), since resident
experts cut CPU work during prefill as well.

Two contributions multiply, and only one needs the patch:

| lever | needs code | contribution |
|---|---|---:|
| `-mcs 180` -> `-mcs 128` (residency) | no, a flag | ~1.35x |
| profile matched to domain **and** context | yes | ~1.64x |

## The governing model

Decode time is a straight line in cold-expert rate, fitted over 14 points and cross-validated
on 16 more (R^2 = 0.9994):

    step_ms  ~=  10.8  +  0.77 x cold_percent          cold_percent = (cpu-assign/row)/8 x 100

The slope is exactly `42 layers x 8 topk x 12.62 MB / 100 / 53 GB/s` -- the fit recovers the
measured AVX-512 CPU kernel bandwidth on its own. Price any change with this before running
it. Read `cold_percent` live with `EXL3_MOE_HANDOFF_PROF=1`.

Where the 256k step goes, four instruments agreeing:

| | ms/token |
|---|---:|
| exposed CPU-expert stall | 38.0 |
| GPU kernel work (all of it) | 14.5 |
| — of which attention / DSA / KV | 0.95 |

Attention grows **+0.137 ms across a 64x context increase**. It is not the bottleneck; cold
rate is the only quantity in the system that moves with context.

## Quick start

```bash
./download-model.sh                    # checkpoint
./build_exl3.sh                        # image, with the profile patch applied
cp moe_profiles/* /models/GLM-5.3-Flash-exl3-4.05bpw/moe_profiles/
./run_glm.sh
```

## CLI flags

### Offload

| flag | meaning |
|---|---|
| `-mcs N` / `--moe_cpu_split N` | run the tail **N** experts of every MoE layer on the CPU, overlapping with that layer's own GPU expert compute. `288 - N` stay resident. |
| `-mcl N` / `--moe_cpu_offload N` | whole-layer offload. Serialises; `-mcs` is better on this model. |
| `-mct N` / `--moe_cpu_threads N` | worker threads (default `cpu_count/2`). The kernel saturates at 4 of 24 cores; 24 threads **regresses** to 40.3 GB/s from oversubscription. Leave it alone. |
| `-cs N` / `--cache_size N` | KV cache in tokens. Buys residency: 1 expert/layer == 30,000 tokens of context. |

### Placement

| flag | meaning |
|---|---|
| `-mcp` / `--moe_cpu_profile` | comma-separated profiles, optional `:weight` each, e.g. `code_long:3,wiki_long:1` |
| `-mcpm` / `--moe_cpu_profile_mode` | `static` freezes the order (use this); `seed` starts hot and keeps adapting |
| `-mcpd` / `--moe_cpu_profile_dir` | extra search path |
| `-mcpq` / `--moe_cpu_profile_any_quant` | allow a checkpoint-fingerprint mismatch, with a warning |

Profiles resolve from `<model_dir>/moe_profiles`, `$EXL3_MOE_PROFILE_DIR`, or a path.

### Environment

| var | meaning |
|---|---|
| `EXL3_MOE_HANDOFF_PROF=1` | per-job `gap \| spin \| compute` and `cpu-assign/row` — the cold-rate discriminator |
| `EXL3_SPLIT_PROF=1` | `issue` / `wait` brackets: `wait` is the exposed GPU stall on CPU experts |
| `EXL3_MOE_CPU_SWAP=0` | disable dynamic swapping (implied by `-mcpm static`) |
| `EXL3_MOE_CPU_SWAP_MAX` / `_INTERVAL` | sweep budget (64) and period (128 steps) for `seed` mode |

## Profiles

27 profiles ship in `moe_profiles/`, one per register. Pick the one that matches your
traffic — domain mismatch costs the entire benefit (a prose census on code is worth 1.01x).

| if you serve | use |
|---|---|
| an agent / tool-calling workload | **`agentic`** — 111 real task prompts, 77% headroom |
| general chat, prose | `wiki_long` |
| generic source code | `code_long` |
| kernel, drivers, systems C | `os_internals`, `c_commented`, `c_standards` |
| exploitation, RE, CTF | `code_injection`, `rev_eng`, `ctf_pwn`, `exploit_dev` |
| firmware, MCU, RF, robotics | `firmware`, `mcu_sbc`, `rf_sdr`, `robotics` |
| architecture and design work | `swe_arch_text`, `swe_arch` |

Full catalogue — corpus contents, capture numbers, and which register each covers:
**[`moe_profiles/README.md`](moe_profiles/README.md)**.

Loading them from a path, an external directory, or several with weights:
**[`LOADING_PROFILES.md`](LOADING_PROFILES.md)**. Building your own:
**[`BUILDING_PROFILES.md`](BUILDING_PROFILES.md)**.

**Group by register, not by topic.** Six broad profiles were built by merging related
categories; every one scored below its own members. `security` merged nine categories and
landed at 14% of available headroom — barely above random — while eight of its nine parts
score 55-80% individually. `systems` merged seven and kept 60%, because they share one
language family. Two dissimilar categories did worse than seven similar ones.

## Recommended settings

```bash
# prose / general, up to 256k
-cs 270336 -mcs 128 -mcp wiki_long  -mcpm static      # 39.8 tok/s @256k

# coding
-cs 270336 -mcs 128 -mcp code_long  -mcpm static      # 34.0 tok/s @256k

# 1M context (costs ~17% throughput; 132 resident to pay for 17.2 GiB of KV)
-cs 1048576 -mcs 156 -mcp wiki_long -mcpm static      # 33.0 tok/s
```

`-mcs 128` (160 resident) is the ceiling at `-cs 270336`: **164 resident fails to load.**
Usable VRAM is 95.01 GiB, not the 97,887 MiB `nvidia-smi` reports, and allocation is flat
across a run — exl3 preallocates, so prefill never spikes above the load-time figure.

## What does not work

Each of these was measured, not assumed:

* **MTP / speculative decoding.** 0.63x at short context, 1.08x at 256k. A verify round
  multiplies cold-expert bytes; acceptance (0.462 at 32k) does not pay for it.
* **Quantized KV (`-cq`).** Hard-blocked above 2048 tokens on this model: the DSA sparse
  path asserts `qc is None` because its gather kernels read fp16 rows. fp16 KV only.
* **Non-uniform per-layer residency.** Greedy-optimal allocation at constant VRAM moves cold
  mass 28.822% -> 28.678%. 1.005x. Not worth the patch.
* **FP8 for non-expert projections.** exl3 already stores them at 4.05bpw, *half* the size of
  FP8; switching would make them bigger.
* **A faster CPU expert kernel.** It runs at 52 GB/s, 90% of this box's DDR5 read ceiling,
  and saturates at 4 of 24 cores. There is <8% available.
* **Streaming expert weights to VRAM instead of computing on CPU.** Same DDR5 read, plus a
  PCIe hop: ~1.0x. The win is not moving bytes faster, it is not moving them at all.

## Measurement notes

* `eval/perf.py` is **not valid** for placement or long-context claims: `measure_generate`
  builds state via `cache.get_test_state(clear=True)`, so 34 of 45 recurrent layers decode
  from zero state and its long-context routing is really short-context routing. Use
  `tools/ctxtest.py`, which drives the Generator.
* Report `cpu-assign/row` beside every tok/s. Throughput alone cannot separate a placement
  effect from a warm-up artifact.
* Warm up **above 2048 tokens**: crossing `index_topk` JITs and captures a new graph slot in
  every MLA module mid-generation, +5.45 ms/token over a 127-token measurement.
* Discard the first generation after load, and the first timed point after a fresh full
  prefill (a reproducible +4.4 to +5.4 ms).
* Use in-distribution text whenever a profile is in the loop. Synthetic filler inverts the
  sign of the result.
* Single replicates carry 8-13% noise here. Anything under ~15% needs repeats.

## Contents

    README.md                                  this file: measurements, flags, settings
    moe_profiles/README.md                     profile catalogue -- pick one here
    LOADING_PROFILES.md                        loading: paths, external dirs, weighted merges
    BUILDING_PROFILES.md                       building: corpora, windows, scoring, auditing

    0001-moe-expert-placement-profiles.patch   the exl3 patch (+107/-10 vs dev@0531096)
    Dockerfile.exl3, build_exl3.sh             image
    download-model.sh, run_glm.sh              checkpoint and serving
    moe_profiles/*.exl3moe                     precomputed profiles
    serve/serve_openai.py                      OpenAI-compatible server (reasoning + tools)
    serve/serve.sh, glm53-sec.service          launcher and systemd unit
    serve/Caddyfile.example                    TLS + bearer token in front of it
    tools/ctxtest.py                           the harness the numbers were taken with
    tools/moe_profile_build.py                 build a profile
    tools/moe_profile_pack.py                  census -> packed profile
    tools/make_code_corpus.py                  code corpus from a source tree

## Engine patch

`0001-moe-expert-placement-profiles.patch` applies to exllamav3 dev `0531096` (+107/-10) and
is inert unless `--moe_cpu_profile` is passed.

The same change, applied and with the tooling and 40 tests in place, is a branch on the
engine fork:

* branch — https://github.com/anoane/exllamav3-anemone/tree/moe-expert-profiles
* commit — https://github.com/anoane/exllamav3-anemone/commit/1149e2f0c12ca5906c0e68b4f8b87b30b8cb041d

The patch here is generated from that branch, so the two cannot drift:

```bash
git diff dev...moe-expert-profiles -- \
    exllamav3/model_init.py exllamav3/modules/block_sparse_mlp_cpu.py \
    > 0001-moe-expert-placement-profiles.patch
```
