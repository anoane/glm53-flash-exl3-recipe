#!/usr/bin/env bash
set -euxo pipefail
cd ${BUILD_DIR:-/opt/exl3build}
if [ ! -d exllamav3/.git ]; then
  git clone --branch dev https://github.com/turboderp-org/exllamav3.git exllamav3
fi
cd exllamav3 && git fetch origin dev && git reset --hard origin/dev
echo "exllamav3 dev HEAD: $(git rev-parse --short HEAD) $(git log -1 --format=%ci)"
cd ${BUILD_DIR:-/opt/exl3build}
docker build -f Dockerfile.exl3 -t exllamav3-dev:sm120 .
echo EXL3_BUILD_DONE
