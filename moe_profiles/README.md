# Expert-placement profiles for GLM-5.3-Flash-exl3 4.05bpw

27 profiles. Each is a measured census of which experts the router actually selects, so
placement starts on the hot set instead of converging there over thousands of decode steps.

Valid for exactly one `(model, checkpoint)` pair: `turboderp/GLM-5.3-Flash-exl3` at 4.05bpw.
Model identity is fatal on mismatch even with the override; checkpoint identity is fatal
unless `--moe_cpu_profile_any_quant`.

## Reading the numbers

`held-out` is capture on windows the ranking never saw — the fraction of routed experts that
land in VRAM. `uniform` (55.6% at 160 resident of 288) is what **random placement** scores.
`oracle` is fitting on the held-out set itself: the ceiling for any static profile.
**headroom** = `(held-out − uniform) / (oracle − uniform)`: how much of the achievable signal
the profile actually captures. A profile near 0% headroom is worthless however good its
raw capture looks.

All scored at `-resident 160` (i.e. `-mcs 128`).

## General purpose

| profile | held-out | headroom | what it is |
|---|---:|---:|---|
| `wiki_long` | 74.0% | 66% | wikitext-2, prose. Default for general chat. |
| `code_long` | 70.1% | — | 17.4 MB Python/C++/CUDA from a source tree. Generic code. |
| **`agentic`** | 77.3% | **77%** | **111 real task prompts, 17 families** (MISRA/AUTOSAR, networking stacks, distributed consensus, kernel drivers, Rust unsafe, DB storage engines, devops). Closest to production agent traffic — start here if you serve an agent. |

## Systems and low-level

| profile | held-out | headroom | what it is |
|---|---:|---:|---|
| `os_internals` | 86.9% | 78% | kernel internals, schedulers, memory management, syscalls |
| `c_commented` | 75.0% | 70% | heavily commented C, idiomatic style |
| `c_standards` | 75.6% | 62% | C standards text, MISRA/AUTOSAR-style rule material |
| `cpp_commented` | 76.6% | 64% | commented modern C++ |
| `rust_cpp_go` | 76.6% | 67% | Rust / C++ / Go systems code side by side |
| `firmware` | 78.6% | 65% | firmware, bootloaders, board bring-up |
| `mcu_sbc` | 76.3% | 64% | microcontroller and single-board-computer code |
| `hardware` | 74.7% | 61% | datasheets, register maps, peripheral programming |

## Security

| profile | held-out | headroom | what it is |
|---|---:|---:|---|
| `code_injection` | 85.8% | 80% | injection classes across web and native |
| `rev_eng` | 84.0% | 80% | reverse engineering, disassembly, binary analysis |
| `ctf_pwn` | 83.8% | 80% | CTF binary exploitation writeups and harnesses |
| `exploit_dev` | 80.2% | 79% | exploit development, ROP, mitigations |
| `red_teaming` | 78.7% | 71% | red-team tradecraft, tooling, operations |
| `app_cracking` | 77.3% | 67% | application cracking, licensing, anti-tamper |
| `fuzzing` | 75.2% | 60% | fuzzers, harness construction, coverage |
| `web_hacking` | 73.5% | 55% | web attack surface. **Marginal** — mixes JS, PHP, HTTP traces and prose. |

## Engineering and applied

| profile | held-out | headroom | what it is |
|---|---:|---:|---|
| `swe_arch_text` | 82.8% | 80% | software-architecture prose, design discussion |
| `swe_arch` | 80.5% | 78% | architecture in code: patterns, module structure |
| `rf_sdr` | 82.9% | 79% | RF and software-defined radio |
| `robotics` | 78.0% | 73% | robotics control, kinematics, ROS-style code |
| `cryptography` | 77.8% | 69% | ciphers, protocols, constant-time implementation |
| `dev_tooling` | 75.5% | 59% | build systems, CI, developer tooling |
| `secure_coding` | 71.2% | 50% | defensive coding practice. **Marginal.** |
| `lang_breadth` | 71.8% | 49% | many languages at once. **Marginal** by design. |

## Not shipped

`memcorruption` scored **9% headroom** (58.8% held-out against 55.6% for random) and is
excluded. It bundles heap/stack/UAF writeups, exploit code and analysis prose under one
label — three registers in one corpus, which is the same failure as the broad groups below.
A category *name* does not guarantee register homogeneity, and the held-out scorer is what
catches it.

## Why per-category and not per-topic

Six broad profiles were built first by grouping related categories, then every member was
built separately. Every group scored **below its own members**, ordered by how mixed its
registers were:

| broad group | its headroom | members individually |
|---|---:|---|
| `security` (9 categories: web + asm + prose + harnesses) | **14%** | 55–80% |
| `crypto` (2 categories) | 23% | `cryptography` alone **69%** |
| `swe` (6 categories, mixed languages + prose) | 39% | 49–78% |
| `embedded` (4 categories) | 48% | 63–79% |
| `systems` (7 categories, all C/C++/low-level) | 60% | 61–78% |

`security` merged nine categories and landed barely above random, yet eight of its nine parts
score 55–80% alone. `systems` merged seven and lost little, because they share one language
family and register.

**The rule is register homogeneity, not topic or category count.** Two dissimilar categories
(`crypto`) did worse than seven similar ones (`systems`). Build one profile per register, and
check it against `uniform` before shipping it.

## Loading

See the repository README for full CLI documentation. Briefly:

```bash
# by name, resolved under <model_dir>/moe_profiles
-mcp agentic -mcpm static -mcs 128

# by absolute path, no installation needed
-mcp /srv/profiles/os_internals.exl3moe -mcpm static -mcs 128

# from an external directory
-mcpd /srv/profiles -mcp exploit_dev -mcpm static -mcs 128

# several, weighted (counts are renormalized per layer, so corpus size does not decide it)
-mcp agentic:3,os_internals:2,c_commented:1 -mcpm static -mcs 128
```

A merge is worth it when your traffic genuinely spans registers. It is **not** a substitute
for the right single profile: merging is what the broad groups above did, and they lost most
of their signal. If one profile matches your workload, use it alone.
