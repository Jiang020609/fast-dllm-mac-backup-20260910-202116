#!/bin/bash
set -uo pipefail

RUN_ID="${1:?Missing RUN_ID}"
START="${2:?Missing START}"
COUNT="${3:?Missing COUNT}"

ROOT="$HOME/Fast-dLLM/v1/llada"
MODEL="$HOME/models/LLaDA-8B-Instruct"
ACCELERATE="$HOME/fastdllm-mps/bin/accelerate"

END=$((START + COUNT))
OUTROOT="$HOME/Fast-dLLM/results/gsm8k_range_${START}_${END}_$RUN_ID"
LOGROOT="$HOME/Fast-dLLM/logs/gsm8k_range_${START}_${END}_$RUN_ID"

THRESHOLDS=(0.75 0.80 0.85 0.90)

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

SAMPLES_JSON="$(
"$HOME/fastdllm-mps/bin/python" - "$START" "$COUNT" <<'PY'
import json
import sys

start = int(sys.argv[1])
count = int(sys.argv[2])

print(json.dumps(
    {"gsm8k": list(range(start, start + count))},
    separators=(",", ":"),
))
PY
)"

echo "=================================================="
echo "GSM8K non-overlapping range benchmark"
echo "RUN_ID=$RUN_ID"
echo "Samples=$START..$((END - 1))"
echo "Count=$COUNT"
echo "Thresholds=${THRESHOLDS[*]}"
echo "Started=$(date)"
echo "=================================================="

FAILURES=0

for THRESHOLD in "${THRESHOLDS[@]}"; do
  TAG="${THRESHOLD/./p}"
  OUTDIR="$OUTROOT/threshold_$TAG"
  LOG="$LOGROOT/threshold_$TAG.log"

  mkdir -p "$OUTDIR"

  MODEL_ARGS="model_path=$MODEL,device=mps,is_check_greedy=False,gen_length=256,steps=256,block_length=32,show_speed=True,use_cache=True,dual_cache=True,threshold=$THRESHOLD"

  echo
  echo "=================================================="
  echo "Starting threshold=$THRESHOLD"
  echo "Samples=$START..$((END - 1))"
  echo "Started=$(date)"
  echo "=================================================="

  "$ACCELERATE" launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    "$ROOT/eval_llada.py" \
      --model llada_dist \
      --model_args "$MODEL_ARGS" \
      --tasks gsm8k \
      --num_fewshot 5 \
      --batch_size 1 \
      --samples "$SAMPLES_JSON" \
      --confirm_run_unsafe_code \
      --output_path "$OUTDIR/eval" \
      --log_samples \
    2>&1 | tee "$LOG"

  STATUS=${PIPESTATUS[0]}

  if [ "$STATUS" -eq 0 ]; then
    echo "threshold=$THRESHOLD completed at $(date)"
  else
    echo "threshold=$THRESHOLD failed status=$STATUS at $(date)"
    FAILURES=$((FAILURES + 1))
  fi
done

echo
echo "=================================================="
echo "Range benchmark completed"
echo "Samples=$START..$((END - 1))"
echo "Failures=$FAILURES"
echo "OUTROOT=$OUTROOT"
echo "Finished=$(date)"
echo "=================================================="

exit "$FAILURES"
