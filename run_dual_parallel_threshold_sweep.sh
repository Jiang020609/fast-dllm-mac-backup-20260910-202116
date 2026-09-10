#!/bin/bash
set -uo pipefail

ROOT="$HOME/Fast-dLLM/v1/llada"
PY="$HOME/fastdllm-mps/bin/python"
ACCELERATE="$HOME/fastdllm-mps/bin/accelerate"
MODEL="$HOME/models/LLaDA-8B-Instruct"

RUN_ID="${1:?Missing RUN_ID}"
OUTROOT="$HOME/Fast-dLLM/results/dual_parallel_threshold_$RUN_ID"
LOGROOT="$HOME/Fast-dLLM/logs/dual_parallel_threshold_$RUN_ID"

mkdir -p "$OUTROOT" "$LOGROOT"
cd "$ROOT" || exit 1

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

THRESHOLDS=(0.80 0.85 0.90 0.95)
FAILURES=0

echo "============================================================"
echo "Fast-dLLM DualCache + Parallel threshold sweep"
echo "Run ID: $RUN_ID"
echo "Started: $(date)"
echo "Thresholds: ${THRESHOLDS[*]}"
echo "============================================================"

for THRESHOLD in "${THRESHOLDS[@]}"
do
    SAFE_THRESHOLD="${THRESHOLD/./p}"
    OUTDIR="$OUTROOT/threshold_$SAFE_THRESHOLD"
    CONFIG_LOG="$LOGROOT/threshold_$SAFE_THRESHOLD.log"

    mkdir -p "$OUTDIR"

    MODEL_ARGS="model_path=$MODEL,device=mps,is_check_greedy=False,gen_length=256,steps=256,block_length=32,show_speed=True,use_cache=True,dual_cache=True,threshold=$THRESHOLD"

    echo
    echo "============================================================"
    echo "Starting threshold=$THRESHOLD"
    echo "Time: $(date)"
    echo "Output: $OUTDIR"
    echo "============================================================"

    set +e

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
        --limit 1 \
        --confirm_run_unsafe_code \
        --output_path "$OUTDIR" \
        --log_samples \
      2>&1 | tee "$CONFIG_LOG"

    STATUS=${PIPESTATUS[0]}
    set -e

    if [ "$STATUS" -eq 0 ]; then
        echo "threshold=$THRESHOLD completed at $(date)"
    else
        FAILURES=$((FAILURES + 1))
        echo "threshold=$THRESHOLD failed, status=$STATUS"
    fi
done

echo
echo "============================================================"
echo "Building summary"
echo "============================================================"

"$PY" - "$OUTROOT" "$LOGROOT" <<'PY'
import csv
import glob
import json
import re
import sys
from pathlib import Path


outroot = Path(sys.argv[1])
logroot = Path(sys.argv[2])

thresholds = ["0.80", "0.85", "0.90", "0.95"]
rows = []


def last_float(pattern: str, text: str):
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def last_int(pattern: str, text: str):
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else None


def read_metrics(directory: Path):
    candidates = sorted(
        glob.glob(
            str(directory / "**" / "*.json"),
            recursive=True,
        )
    )

    for filename in reversed(candidates):
        try:
            with open(filename, encoding="utf-8") as file:
                data = json.load(file)
        except Exception:
            continue

        result = data.get("results", {}).get("gsm8k")
        if not isinstance(result, dict):
            continue

        flexible = None
        strict = None

        for key, value in result.items():
            lowered = key.lower()

            if "exact_match" not in lowered:
                continue

            if "stderr" in lowered:
                continue

            if "flexible" in lowered:
                flexible = value

            if "strict" in lowered:
                strict = value

        return flexible, strict, filename

    return None, None, None


for threshold in thresholds:
    safe = threshold.replace(".", "p")
    log_path = logroot / f"threshold_{safe}.log"
    result_dir = outroot / f"threshold_{safe}"

    text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.exists()
        else ""
    )

    total_time = last_float(
        r"Total time taken:\s*([0-9.]+)\s*seconds",
        text,
    )

    tokens_per_second = last_float(
        r"Tokens per second:\s*([0-9.]+)",
        text,
    )

    generated_tokens = last_int(
        r"Total number of tokens generated:\s*(\d+)",
        text,
    )

    nfe = last_int(
        r"Total NFE is\s*(\d+)",
        text,
    )

    flexible, strict, result_file = read_metrics(result_dir)

    rows.append(
        {
            "threshold": threshold,
            "latency_seconds": total_time,
            "generated_tokens": generated_tokens,
            "tokens_per_second": tokens_per_second,
            "nfe": nfe,
            "nfe_reduction_vs_256": (
                1 - nfe / 256
                if nfe is not None
                else None
            ),
            "flexible_exact_match": flexible,
            "strict_exact_match": strict,
            "result_file": result_file,
            "log_file": str(log_path),
        }
    )


summary_path = outroot / "dual_parallel_threshold_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)


header = [
    "Threshold",
    "Latency(s)",
    "NFE",
    "NFE↓",
    "Tokens/s",
    "Flex EM",
    "Strict EM",
]

formatted = []

for row in rows:
    formatted.append(
        [
            row["threshold"],
            (
                f"{row['latency_seconds']:.3f}"
                if row["latency_seconds"] is not None
                else "FAILED"
            ),
            (
                str(row["nfe"])
                if row["nfe"] is not None
                else "-"
            ),
            (
                f"{row['nfe_reduction_vs_256']:.1%}"
                if row["nfe_reduction_vs_256"] is not None
                else "-"
            ),
            (
                f"{row['tokens_per_second']:.3f}"
                if row["tokens_per_second"] is not None
                else "-"
            ),
            (
                str(row["flexible_exact_match"])
                if row["flexible_exact_match"] is not None
                else "-"
            ),
            (
                str(row["strict_exact_match"])
                if row["strict_exact_match"] is not None
                else "-"
            ),
        ]
    )


widths = [
    max(
        len(header[index]),
        *(len(row[index]) for row in formatted),
    )
    for index in range(len(header))
]


def print_row(row):
    print(
        "  ".join(
            value.rjust(widths[index])
            for index, value in enumerate(row)
        )
    )


print()
print("=" * 80)
print("DualCache + Parallel threshold sweep summary")
print("=" * 80)

print_row(header)
print_row(["-" * width for width in widths])

for row in formatted:
    print_row(row)

print()
print("Summary CSV:", summary_path)
PY

echo
echo "============================================================"
echo "DualCache + Parallel sweep completed"
echo "Failures: $FAILURES"
echo "Results: $OUTROOT"
echo "Logs: $LOGROOT"
echo "Finished: $(date)"
echo "============================================================"

exit "$FAILURES"
