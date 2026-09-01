#!/usr/bin/env bash
set -euxo pipefail
REPO=turboderp/GLM-5.3-Flash-exl3
REV=4.05bpw
DEST=${MODEL_DIR:-/models/GLM-5.3-Flash-exl3-4.05bpw}
mkdir -p "$DEST"
hf download "$REPO" --revision "$REV" --local-dir "$DEST" --max-workers 8
echo GLM_DL_DONE
