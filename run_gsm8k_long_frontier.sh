#!/bin/bash
set -uo pipefail

RUN_ID="${1:?Missing RUN_ID}"
LIMIT="${2:-150}"

ROOT="$HOME/Fast-dLLM/v1/llada"
MODEL="$HOME/models/LLaDA-8B-Instruct"
ACCELERATE="$HOME/fastdllm-mps/bin/accelerate"

OUTROOT="$HOME/Fast-dLLM/results/gsm8k_long_frontier_$RUN_ID"
LOGROOT="$HOME/Fast-dLLM/logs/gsm8k_long_frontier_$RUN_ID"
MONITOR_LOG="$LOGROOT/system_monitor.log"

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

monitor_system() {
    while true; do
        {
            echo
            echo "=================================================="
            echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
            echo "=================================================="

            echo "--- thermal ---"
            pmset -g therm 2>&1 || true

            echo "--- memory pressure ---"
            memory_pressure -Q 2>&1 || true

            echo "--- disk ---"
            df -h "$HOME" | tail -n 1

            echo "--- python processes ---"
            ps -axo pid,etime,%cpu,%mem,rss,command \
              | grep -E 'eval_llada|accelerate' \
              | grep -v grep \
              || true
        } >> "$MONITOR_LOG"

        sleep 60
    done
}

monitor_system &
MONITOR_PID=$!

cleanup() {
    kill "$MONITOR_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

FAILURES=0

echo "=================================================="
echo "GSM8K long threshold frontier"
echo "RUN_ID=$RUN_ID"
echo "LIMIT=$LIMIT"
echo "THRESHOLDS=${THRESHOLDS[*]}"
echo "Started=$(date)"
echo "=================================================="

for THRESHOLD in "${THRESHOLDS[@]}"; do
    TAG="${THRESHOLD/./p}"
    OUTDIR="$OUTROOT/threshold_$TAG"
    LOG="$LOGROOT/threshold_$TAG.log"

    mkdir -p "$OUTDIR"

    MODEL_ARGS="model_path=$MODEL,device=mps,is_check_greedy=False,gen_length=256,steps=256,block_length=32,show_speed=True,use_cache=True,dual_cache=True,threshold=$THRESHOLD"

    echo
    echo "=================================================="
    echo "Starting threshold=$THRESHOLD"
    echo "Samples=$LIMIT"
    echo "Started=$(date)"
    echo "Log=$LOG"
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
        --limit "$LIMIT" \
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

"$HOME/fastdllm-mps/bin/python" - \
  "$LOGROOT" "$OUTROOT" "$LIMIT" <<'PY'
import csv
import re
import sys
from pathlib import Path

logroot = Path(sys.argv[1])
outroot = Path(sys.argv[2])
limit = int(sys.argv[3])


def last_float(pattern: str, text: str):
    values = re.findall(pattern, text)
    return float(values[-1]) if values else None


def last_int(pattern: str, text: str):
    values = re.findall(pattern, text)
    return int(values[-1]) if values else None


rows = []

for threshold in ("0.75", "0.80", "0.85", "0.90"):
    tag = threshold.replace(".", "p")
    path = logroot / f"threshold_{tag}.log"

    if not path.exists():
        continue

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    latency = last_float(
        r"Total time taken:\s*([0-9.]+)\s*seconds",
        text,
    )

    tokens_per_second = last_float(
        r"Tokens per second:\s*([0-9.]+)",
        text,
    )

    total_tokens = last_int(
        r"Total number of tokens generated:\s*(\d+)",
        text,
    )

    total_nfe = last_int(
        r"Total NFE is\s*(\d+)",
        text,
    )

    flexible_em = last_float(
        r"\|gsm8k\|[^\n]*flexible-extract"
        r"[^\n]*\|\s*([0-9.]+)\|",
        text,
    )

    strict_em = last_float(
        r"\|[^\n]*strict-match"
        r"[^\n]*\|\s*([0-9.]+)\|",
        text,
    )

    rows.append({
        "threshold": threshold,
        "samples": limit,
        "total_latency_s": latency,
        "latency_per_sample_s": (
            latency / limit
            if latency is not None
            else None
        ),
        "tokens_per_second": tokens_per_second,
        "total_tokens": total_tokens,
        "total_nfe": total_nfe,
        "nfe_per_sample": (
            total_nfe / limit
            if total_nfe is not None
            else None
        ),
        "flexible_exact_match": flexible_em,
        "strict_exact_match": strict_em,
    })

summary = outroot / "gsm8k_long_frontier_summary.csv"

with summary.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)

header = [
    "Threshold",
    "Samples",
    "Total(h)",
    "Sec/sample",
    "NFE/sample",
    "Tokens/s",
    "Flex EM",
    "Strict EM",
]

formatted = []

for row in rows:
    formatted.append([
        row["threshold"],
        str(row["samples"]),
        (
            f"{row['total_latency_s'] / 3600:.3f}"
            if row["total_latency_s"] is not None
            else "-"
        ),
        (
            f"{row['latency_per_sample_s']:.3f}"
            if row["latency_per_sample_s"] is not None
            else "-"
        ),
        (
            f"{row['nfe_per_sample']:.2f}"
            if row["nfe_per_sample"] is not None
            else "-"
        ),
        (
            f"{row['tokens_per_second']:.3f}"
            if row["tokens_per_second"] is not None
            else "-"
        ),
        (
            f"{row['flexible_exact_match']:.4f}"
            if row["flexible_exact_match"] is not None
            else "-"
        ),
        (
            f"{row['strict_exact_match']:.4f}"
            if row["strict_exact_match"] is not None
            else "-"
        ),
    ])

widths = [
    max(
        len(header[index]),
        *(len(row[index]) for row in formatted),
    )
    for index in range(len(header))
]


def print_row(row):
    print("  ".join(
        value.rjust(widths[index])
        for index, value in enumerate(row)
    ))


print()
print("=" * 100)
print("GSM8K long threshold frontier")
print("=" * 100)

print_row(header)
print_row(["-" * width for width in widths])

for row in formatted:
    print_row(row)

print()
print("Summary:", summary)
PY

echo
echo "=================================================="
echo "Long benchmark completed"
echo "Failures: $FAILURES"
echo "Finished=$(date)"
echo "OUTROOT=$OUTROOT"
echo "=================================================="

exit "$FAILURES"
