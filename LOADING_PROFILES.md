# Loading expert-placement profiles

A profile is loaded with `--moe_cpu_profile` (`-mcp`). It is inert unless that flag is
passed: without it, placement falls back to upstream's runtime discovery.

## The four ways to name one

### 1. By name, installed next to the model

```bash
cp moe_profiles/agentic.* /models/GLM-5.3-Flash-exl3-4.05bpw/moe_profiles/
... -mcp agentic -mcpm static -mcs 128
```

Names resolve under `<model_dir>/moe_profiles`, then `$EXL3_MOE_PROFILE_DIR`, then
`~/.cache/exllamav3/moe_profiles`. Extension is inferred, preferring the packed format:
`.exl3moe` → `.safetensors` → `.npz` → `.json`.

### 2. By path, from anywhere

```bash
... -mcp /srv/profiles/os_internals.exl3moe -mcpm static -mcs 128
... -mcp ./experiments/run7/candidate.exl3moe -mcpm static
```

Anything containing a path separator is used verbatim. Nothing needs installing, which makes
this the convenient form for A/B-ing candidates.

### 3. From an external directory

```bash
export EXL3_MOE_PROFILE_DIR=/srv/shared/moe_profiles
... -mcp exploit_dev -mcpm static -mcs 128

# or per-invocation, without touching the environment
... -mcpd /srv/shared/moe_profiles -mcp exploit_dev -mcpm static
```

`-mcpd` prepends a directory to the search path. Useful when profiles are shared across
several model copies, or mounted read-only into a container:

```bash
docker run --rm --gpus all \
  -v /models/GLM:/model:ro \
  -v /srv/shared/moe_profiles:/profiles:ro \
  exllamav3 python3 -u serve.py -m /model \
    -mcpd /profiles -mcp agentic -mcpm static -mcs 128
```

### 4. Several at once, with weights

```bash
-mcp agentic:3,os_internals:2,c_commented:1 -mcpm static -mcs 128
```

Comma-separated, each with an optional `:weight` (default 1.0). Weights are relative, so
`3,2,1` and `6,4,2` are identical.

**Counts are normalized per layer before weighting.** A 9 MB corpus and a 0.8 MB one
contribute according to their weights, not their size — otherwise the largest corpus would
silently dominate every merge.

Mixing formats is fine: `-mcp base.exl3moe:2,/tmp/probe.npz:1`.

## When to merge, and when not to

Merging is right when your traffic genuinely spans registers — say an agent that both writes
kernel code and discusses architecture:

```bash
-mcp agentic:3,os_internals:1,swe_arch_text:1 -mcpm static
```

Merging is **wrong** as a substitute for choosing the right profile. Six broad profiles were
built by merging related categories, and every one scored below its own members:

| merged group | headroom | members individually |
|---|---:|---|
| `security` (9 categories) | **14%** | 55–80% |
| `swe` (6 categories) | 39% | 49–78% |
| `systems` (7 categories) | 60% | 61–78% |

A merge across dissimilar registers averages rankings that disagree, and the result can land
near random even when every input is strong. If one profile matches your workload, use it
alone. Verify any merge the same way you would a single profile — measure it.

Note the merge also costs the fast load path: a single packed source returns its stored
ranking directly, while a merge must renormalize per layer and re-sort. That is ~3 ms, so it
matters for correctness reasoning, not for speed.

## Modes

```bash
-mcpm static     # freeze the profile order. Use this.
-mcpm seed       # start from the profile, then keep adapting at runtime
```

`static` is what all the measured numbers use. `seed` is new capability — upstream refuses a
profile unless swapping is off — but adaptation measured no better than static here
(24.3% vs 24.9% cold over 8 turns at fixed context, inside noise), and swapping mid-stream
perturbs generation: perplexity 3.4924 under `seed` against 3.4904 under `static`.

## Checkpoint identity

A profile records the checkpoint it was measured on. On mismatch the loader refuses:

```
-mcpq / --moe_cpu_profile_any_quant     proceed anyway, with a warning
```

Model identity (architecture, layers, experts, `moe_intermediate_size`, `hidden_size`) is
**fatal even with the override** — a profile from a different architecture is meaningless.
Checkpoint identity (`checkpoint_sha`, `quant_method`, `bits`, `head_bits`, `codebook`) is
what `-mcpq` waives, for the case where you have requantized at the same geometry and want to
reuse a profile rather than rebuild it. Expect some degradation; verify it.

`checkpoint_sha` hashes safetensors **headers** only — 0.03 s and 19.7 MB of reads on a
165 GB checkpoint, not a full content hash.

## Verifying a profile is actually working

```bash
EXL3_MOE_HANDOFF_PROF=1 ... -mcp agentic -mcpm static -mcs 128
```

The worker prints `cpu-assign/row`. Divided by `topk` (8) that is the live cold rate, and it
should roughly match `1 − held_out` from the profile's capture report. If it does not, the
profile does not match your traffic — which is the failure that matters, and the only one
throughput alone cannot distinguish from a warm-up artifact.

Expected at `-mcs 128` with a well-matched profile: `cpu-assign/row` around 2.0–2.5 of 8.
Without a profile: around 4.5.
