#!/usr/bin/env bash
set -uo pipefail

RUN_ID="${1:?Missing RUN_ID}"
LIMIT="${2:-40}"

ROOT="$HOME/Fast-dLLM/v1/llada"
MODEL="$HOME/models/LLaDA-8B-Instruct"
ACCELERATE="$HOME/fastdllm-mps/bin/accelerate"

OUTROOT="$HOME/Fast-dLLM/results/gen512_v2_i2_gate_$RUN_ID"
LOGROOT="$HOME/Fast-dLLM/logs/gen512_v2_i2_gate_$RUN_ID"

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

MODEL_ARGS="model_path=$MODEL,device=mps,is_check_greedy=False,gen_length=512,steps=512,block_length=32,show_speed=True,use_cache=True,dual_cache=True,threshold=0.75"

FAILURES=0

run_one() {
    local MODE="$1"
    local TARGET="$2"
    local INTERVAL="$3"

    local OUTDIR="$OUTROOT/$MODE"
    local LOG="$LOGROOT/$MODE.log"

    mkdir -p "$OUTDIR"

    export LLADA_CACHE_REFRESH_INTERVAL="$INTERVAL"

    echo
    echo "============================================================"
    echo "mode=$MODE"
    echo "interval=$INTERVAL"
    echo "limit=$LIMIT"
    echo "target=$TARGET"
    echo "started=$(date)"
    echo "============================================================"

    "$ACCELERATE" launch \
      --num_processes 1 \
      --num_machines 1 \
      --mixed_precision no \
      --dynamo_backend no \
      "$TARGET" \
        --model llada_dist \
        --model_args "$MODEL_ARGS" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size 1 \
        --limit "$LIMIT" \
        --confirm_run_unsafe_code \
        --output_path "$OUTDIR/eval" \
        --log_samples \
      2>&1 | tee "$LOG"

    local STATUS=${PIPESTATUS[0]}

    if [[ "$STATUS" -eq 0 ]]; then
        echo "mode=$MODE completed"
    else
        echo "mode=$MODE failed status=$STATUS"
        FAILURES=$((FAILURES + 1))
    fi
}

run_one \
  "v2" \
  "$ROOT/eval_llada_rotated_k_v2_reference.py" \
  "1"

run_one \
  "i2" \
  "$ROOT/eval_llada_rotated_k_v2b_cache_refresh.py" \
  "2"

echo
echo "Failures=$FAILURES"
echo "OUTROOT=$OUTROOT"
echo "LOGROOT=$LOGROOT"

exit "$FAILURES"
