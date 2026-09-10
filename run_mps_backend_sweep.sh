#!/bin/bash
set -euo pipefail

ROOT="$HOME/Fast-dLLM/v1/llada"
ACCELERATE="$HOME/fastdllm-mps/bin/accelerate"
MODEL="$HOME/models/LLaDA-8B-Instruct"

RUN_ID="${1:?Missing RUN_ID}"
OUTROOT="$HOME/Fast-dLLM/results/mps_backend_$RUN_ID"
LOGROOT="$HOME/Fast-dLLM/logs/mps_backend_$RUN_ID"

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

MODEL_ARGS="model_path=$MODEL,device=mps,is_check_greedy=False,gen_length=256,steps=256,block_length=32,show_speed=True,use_cache=True,dual_cache=True,threshold=0.80"

run_one () {
    local NAME="$1"
    local PREFER_METAL="$2"
    local FAST_MATH="$3"
    local REPEAT="$4"

    unset PYTORCH_MPS_PREFER_METAL
    unset PYTORCH_MPS_FAST_MATH

    if [ "$PREFER_METAL" = "1" ]; then
        export PYTORCH_MPS_PREFER_METAL=1
    fi

    if [ "$FAST_MATH" = "1" ]; then
        export PYTORCH_MPS_FAST_MATH=1
    fi

    local OUTDIR="$OUTROOT/${NAME}_r${REPEAT}"
    local LOG="$LOGROOT/${NAME}_r${REPEAT}.log"

    mkdir -p "$OUTDIR"

    echo
    echo "=================================================="
    echo "Mode=$NAME Repeat=$REPEAT"
    echo "PREFER_METAL=${PREFER_METAL}"
    echo "FAST_MATH=${FAST_MATH}"
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
        --limit 1 \
        --confirm_run_unsafe_code \
        --output_path "$OUTDIR" \
        --log_samples \
      2>&1 | tee "$LOG"
}

# 每种配置重复两次，降低偶然波动。
for REPEAT in 1 2
do
    run_one default       0 0 "$REPEAT"
    run_one prefer_metal  1 0 "$REPEAT"
    run_one fast_math     0 1 "$REPEAT"
    run_one both          1 1 "$REPEAT"
done

"$HOME/fastdllm-mps/bin/python" - "$LOGROOT" "$OUTROOT" <<'PY'
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

logroot = Path(sys.argv[1])
outroot = Path(sys.argv[2])

groups = defaultdict(list)

for path in sorted(logroot.glob("*.log")):
    match = re.match(
        r"(default|prefer_metal|fast_math|both)_r(\d+)\.log",
        path.name,
    )
    if not match:
        continue

    mode = match.group(1)
    repeat = int(match.group(2))
    text = path.read_text(encoding="utf-8", errors="replace")

    def last_float(pattern):
        values = re.findall(pattern, text)
        return float(values[-1]) if values else None

    def last_int(pattern):
        values = re.findall(pattern, text)
        return int(values[-1]) if values else None

    flex = last_float(
        r"\|gsm8k\|[^\n]*flexible-extract[^\n]*\|\s*([0-9.]+)\|"
    )
    strict = last_float(
        r"\|[^\n]*strict-match[^\n]*\|\s*([0-9.]+)\|"
    )

    groups[mode].append({
        "repeat": repeat,
        "latency": last_float(
            r"Total time taken:\s*([0-9.]+)\s*seconds"
        ),
        "tokens_per_second": last_float(
            r"Tokens per second:\s*([0-9.]+)"
        ),
        "nfe": last_int(r"Total NFE is\s*(\d+)"),
        "flex": flex,
        "strict": strict,
    })

rows = []

for mode in ["default", "prefer_metal", "fast_math", "both"]:
    values = groups[mode]

    latencies = [
        item["latency"] for item in values
        if item["latency"] is not None
    ]
    throughputs = [
        item["tokens_per_second"] for item in values
        if item["tokens_per_second"] is not None
    ]

    rows.append({
        "mode": mode,
        "runs": len(values),
        "latency_mean_s": statistics.mean(latencies),
        "latency_min_s": min(latencies),
        "tokens_per_second_mean": statistics.mean(throughputs),
        "nfe": values[-1]["nfe"],
        "flexible_em_min": min(item["flex"] for item in values),
        "strict_em_min": min(item["strict"] for item in values),
    })

baseline = next(
    row["latency_mean_s"]
    for row in rows
    if row["mode"] == "default"
)

for row in rows:
    row["speedup_vs_default"] = (
        baseline / row["latency_mean_s"]
    )

summary = outroot / "mps_backend_summary.csv"

with summary.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

header = [
    "Mode",
    "Mean(s)",
    "Min(s)",
    "Speedup",
    "Tokens/s",
    "NFE",
    "Flex",
    "Strict",
]

formatted = []

for row in rows:
    formatted.append([
        row["mode"],
        f"{row['latency_mean_s']:.3f}",
        f"{row['latency_min_s']:.3f}",
        f"{row['speedup_vs_default']:.3f}x",
        f"{row['tokens_per_second_mean']:.3f}",
        str(row["nfe"]),
        str(row["flexible_em_min"]),
        str(row["strict_em_min"]),
    ])

widths = [
    max(len(header[i]), *(len(row[i]) for row in formatted))
    for i in range(len(header))
]

def show(row):
    print("  ".join(
        value.rjust(widths[i])
        for i, value in enumerate(row)
    ))

print()
print("=" * 88)
print("MPS backend sweep")
print("=" * 88)
show(header)
show(["-" * width for width in widths])

for row in formatted:
    show(row)

print()
print("Summary:", summary)
PY
