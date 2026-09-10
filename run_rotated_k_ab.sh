#!/usr/bin/env bash
set -uo pipefail

RUN_ID="${1:?Missing RUN_ID}"
REPEATS="${2:-5}"

ROOT="$HOME/Fast-dLLM/v1/llada"
MODEL="$HOME/models/LLaDA-8B-Instruct"
ACCELERATE="$HOME/fastdllm-mps/bin/accelerate"
PYTHON="$HOME/fastdllm-mps/bin/python"

OUTROOT="$HOME/Fast-dLLM/results/rotated_k_ab_$RUN_ID"
LOGROOT="$HOME/Fast-dLLM/logs/rotated_k_ab_$RUN_ID"

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

echo "=================================================="
echo "Rotated-K Cache A/B"
echo "RUN_ID=$RUN_ID"
echo "REPEATS=$REPEATS"
echo "Started=$(date)"
echo "=================================================="

for REPEAT in $(seq 1 "$REPEATS"); do
    for MODE in baseline rotated_k; do
        if [ "$MODE" = "baseline" ]; then
            TARGET="$ROOT/eval_llada.py"
        else
            TARGET="$ROOT/eval_llada_rotated_k_wrapper.py"
        fi

        OUTDIR="$OUTROOT/${MODE}_r${REPEAT}"
        LOG="$LOGROOT/${MODE}_r${REPEAT}.log"

        mkdir -p "$OUTDIR"

        echo
        echo "=================================================="
        echo "Mode=$MODE Repeat=$REPEAT"
        echo "Target=$TARGET"
        echo "Started=$(date)"
        echo "=================================================="

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
            --limit 1 \
            --confirm_run_unsafe_code \
            --output_path "$OUTDIR/eval" \
            --log_samples \
          2>&1 | tee "$LOG"

        STATUS=${PIPESTATUS[0]}

        if [ "$STATUS" -eq 0 ]; then
            echo "$MODE repeat=$REPEAT completed"
        else
            echo "$MODE repeat=$REPEAT failed status=$STATUS"
            FAILURES=$((FAILURES + 1))
        fi
    done
done

echo
echo "=================================================="
echo "Generating summary"
echo "=================================================="

"$PYTHON" \
  "$HOME/Fast-dLLM/summarize_rotated_k_ab.py" \
  "$LOGROOT" \
  "$OUTROOT"

SUMMARY_STATUS=$?

if [ "$SUMMARY_STATUS" -ne 0 ]; then
    FAILURES=$((FAILURES + 1))
fi

echo
echo "=================================================="
echo "Rotated-K A/B completed"
echo "Failures=$FAILURES"
echo "Finished=$(date)"
echo "OUTROOT=$OUTROOT"
echo "=================================================="

exit "$FAILURES"
