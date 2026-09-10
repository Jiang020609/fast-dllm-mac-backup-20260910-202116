#!/usr/bin/env bash
set -uo pipefail

RUN_ID="${1:?Missing RUN_ID}"

ROOT="$HOME/Fast-dLLM/v1/llada"
MODEL="$HOME/models/LLaDA-8B-Instruct"
ACCELERATE="$HOME/fastdllm-mps/bin/accelerate"

OUTROOT="$HOME/Fast-dLLM/results/cache_refresh_smoke_$RUN_ID"
LOGROOT="$HOME/Fast-dLLM/logs/cache_refresh_smoke_$RUN_ID"

mkdir -p "$OUTROOT" "$LOGROOT"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export PYTORCH_ENABLE_MPS_FALLBACK=0
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="$HOME/hf-cache"
export HF_HUB_CACHE="$HOME/hf-cache/hub"
export HF_DATASETS_CACHE="$HOME/hf-cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODEL_ARGS="model_path=$MODEL,device=mps,is_check_greedy=False,gen_length=256,steps=256,block_length=32,show_speed=True,use_cache=True,dual_cache=True,threshold=0.75"

FAILURES=0

for INTERVAL in 1 2; do
    export LLADA_CACHE_REFRESH_INTERVAL="$INTERVAL"

    OUTDIR="$OUTROOT/interval_$INTERVAL"
    LOG="$LOGROOT/interval_$INTERVAL.log"

    mkdir -p "$OUTDIR"

    echo
    echo "=================================================="
    echo "Cache refresh interval=$INTERVAL"
    echo "=================================================="

    "$ACCELERATE" launch \
      --num_processes 1 \
      --num_machines 1 \
      --mixed_precision no \
      --dynamo_backend no \
      "$ROOT/eval_llada_rotated_k_v2b_cache_refresh.py" \
        --model llada_dist \
        --model_args "$MODEL_ARGS" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size 1 \
        --limit 1 \
        --confirm_run_unsafe_code \
        --output_path "$OUTDIR/eval" \
        --log_samples \
      2>&1 | tee "$LOG"

    STATUS=${PIPESTATUS[0]}

    if [ "$STATUS" -ne 0 ]; then
        FAILURES=$((FAILURES + 1))
    fi
done

echo
echo "Failures=$FAILURES"
echo "OUTROOT=$OUTROOT"

exit "$FAILURES"
