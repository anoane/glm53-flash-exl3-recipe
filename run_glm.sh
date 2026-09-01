#!/usr/bin/env bash
set -euxo pipefail
MODEL=${MODEL_DIR:-/models/GLM-5.3-Flash-exl3-4.05bpw}
SCRIPT="${SCRIPT:-/opt/exllamav3/eval/perf.py}"
CS="${CS:-1048576}"
MCT="${MCT:-16}"
OFFLOAD="${OFFLOAD:--mcl 24}"     # e.g. "-mcl 24" or "-mcs 164"
CQARG=""
[ -n "${CQ:-}" ] && CQARG="-cq ${CQ}"
ARGS="${ARGS:--max_length 16384}"
docker run --rm --name glm-exl3 \
  --gpus all --ipc=host --shm-size 32g --ulimit memlock=-1 \
  -v "$MODEL":/model:ro \
  -e EXL3_MOE_CPU_THREADS="$MCT" \
  exllamav3-dev:sm120 \
  python3 "$SCRIPT" -m /model -cs "$CS" $CQARG $OFFLOAD -mct "$MCT" $ARGS
echo GLM_RUN_DONE
