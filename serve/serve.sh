#!/usr/bin/env bash
# GLM-5.3-Flash exl3 4.05bpw, OpenAI API on 127.0.0.1:8095.
# Caddy fronts it at sec.example.com:8443 with TLS + bearer token.
#
#   -cs 1048576  1M context. KV is 17,600 B/token here (11 KV layers x 1600 B), so 1M is
#                17.2 GiB and residency must drop to 132 to pay for it.
#   -mcs 156     132 of 288 experts resident. 136 resident OOMs at 1M in the KDA prefill
#                scratch; 132 is the measured fit.
#   -mcp code_injection   security code review: vulnerability patterns in source. Held-out
#                capture 79.8% at R=132 (cold 20.2%), the best security-relevant profile at
#                this residency. secure_coding, the intuitive pick, measures 62.6%.
set -euo pipefail
M=${MODEL_DIR:-/models/GLM-5.3-Flash-exl3-4.05bpw}
SERVE=${SERVE_DIR:-$(cd "$(dirname "$0")" && pwd)}
IMG=exllamav3-dev:sm120-profile

sync; echo 3 > /proc/sys/vm/drop_caches
exec docker run --rm --name glm53-sec \
  --gpus all --ipc=host --shm-size 32g --ulimit memlock=-1 \
  --network host \
  -v $M:/model:ro -v $SERVE:/w \
  $IMG \
  python3 -u /w/serve_openai.py -m /model \
    -cs 1048576 -mcs 156 \
    -mcp code_injection -mcpm static \
    -reasoning-effort max \
    -served-name glm-5.3-flash-sec \
    -host 127.0.0.1 -port 8095
