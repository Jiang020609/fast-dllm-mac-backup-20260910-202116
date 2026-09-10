#!/bin/zsh
set -euo pipefail

cd /Users/z33/Fast-dLLM

/usr/bin/env \
  MTL_CAPTURE_ENABLED=1 \
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  /Users/z33/fastdllm-mps/bin/python \
  profile_one_mps_gemm.py \
    --label qkvo \
    --m 32 \
    --k 4096 \
    --n 4096 \
    --warmups 20 \
    --bench-iters 100 \
    --capture-repeats 5 \
    --open-xcode
