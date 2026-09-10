#!/bin/bash
set -euo pipefail

ROOT="$HOME/Fast-dLLM/v1/llada"
PY="$HOME/fastdllm-mps/bin/python"
ACCELERATE="$HOME/fastdllm-mps/bin/accelerate"
MODEL="$HOME/models/LLaDA-8B-Instruct"

RUN_ID="${1:?Missing RUN_ID}"
BASE_PROFILE_DIR="${2:?Missing BASE_PROFILE_DIR}"

OUTDIR="$HOME/Fast-dLLM/results/gsm8k_dual_profile_$RUN_ID"

mkdir -p "$OUTDIR"
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

COMMON_ARGS="model_path=$MODEL,device=mps,is_check_greedy=False,gen_length=256,steps=256,block_length=32,show_speed=True,use_cache=True,dual_cache=True"

echo "===== DualCache GSM8K 5-shot profile ====="
echo "Started: $(date)"
echo "Output: $OUTDIR"

PROFILE_JSONL="$OUTDIR/dual_profile.jsonl" \
PROFILE_TAG="gsm8k_5shot_dual_cache" \
"$ACCELERATE" launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  "$ROOT/eval_llada_dual_profile.py" \
    --model llada_dist \
    --model_args "$COMMON_ARGS" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 1 \
    --confirm_run_unsafe_code \
    --output_path "$OUTDIR/dual_eval" \
    --log_samples

echo
echo "===== No-Cache / Prefix / DualCache comparison ====="

"$PY" - "$BASE_PROFILE_DIR" "$OUTDIR" <<'PY'
import json
import sys
from pathlib import Path

base_dir = Path(sys.argv[1])
dual_dir = Path(sys.argv[2])


def read_last(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not records:
        raise RuntimeError(f"No records in {path}")

    return records[-1]


nocache = read_last(
    base_dir / "nocache_profile.jsonl"
)

prefix = read_last(
    base_dir / "prefix_profile.jsonl"
)

dual = read_last(
    dual_dir / "dual_profile.jsonl"
)


def forward_seconds(record: dict) -> float:
    phases = record["phase_seconds"]

    return (
        phases.get("full_forward", 0.0)
        + phases.get("initial_full_forward", 0.0)
        + phases.get("cached_forward", 0.0)
        + phases.get("dual_cached_forward", 0.0)
    )


def transfer_seconds(record: dict) -> float:
    phases = record["phase_seconds"]

    return (
        phases.get("transfer_update", 0.0)
        + phases.get("initial_transfer", 0.0)
        + phases.get("iterative_transfer", 0.0)
    )


header = [
    "Mode",
    "Wall(s)",
    "NFE",
    "Forward(s)",
    "Transfer(s)",
    "Merge(s)",
    "Forward %",
    "MPS GiB",
]

rows = []

for name, record in [
    ("No-cache", nocache),
    ("Prefix cache", prefix),
    ("DualCache", dual),
]:
    wall = float(record["wall_seconds"])
    forward = forward_seconds(record)

    rows.append(
        [
            name,
            f"{wall:.3f}",
            str(record["nfe"]),
            f"{forward:.3f}",
            f"{transfer_seconds(record):.3f}",
            f"{record['phase_seconds'].get('block_merge', 0.0):.3f}",
            f"{forward / wall * 100:.2f}%",
            f"{record['mps_driver_allocated_gib']:.3f}",
        ]
    )

widths = [
    max(
        len(header[index]),
        *(len(row[index]) for row in rows),
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
print("=" * 100)
print("Matched GSM8K 5-shot cache profile")
print("=" * 100)

print_row(header)
print_row(["-" * width for width in widths])

for row in rows:
    print_row(row)

print()
print(
    "Prefix speedup over No-cache: "
    f"{nocache['wall_seconds'] / prefix['wall_seconds']:.3f}x"
)

print(
    "DualCache speedup over No-cache: "
    f"{nocache['wall_seconds'] / dual['wall_seconds']:.3f}x"
)

print(
    "DualCache speedup over Prefix: "
    f"{prefix['wall_seconds'] / dual['wall_seconds']:.3f}x"
)

print(
    "Forward reduction No-cache -> DualCache: "
    f"{(1 - forward_seconds(dual) / forward_seconds(nocache)) * 100:.2f}%"
)

print(
    "Forward reduction Prefix -> DualCache: "
    f"{(1 - forward_seconds(dual) / forward_seconds(prefix)) * 100:.2f}%"
)

print()
print("DualCache forward calls:")

print(
    "  initial full calls =",
    dual["phase_calls"].get(
        "initial_full_forward",
        0,
    ),
)

print(
    "  initial full mean  =",
    f"{dual['phase_mean_ms'].get('initial_full_forward', 0.0):.3f} ms",
)

print(
    "  cached calls       =",
    dual["phase_calls"].get(
        "dual_cached_forward",
        0,
    ),
)

print(
    "  cached mean        =",
    f"{dual['phase_mean_ms'].get('dual_cached_forward', 0.0):.3f} ms",
)

print()
print("DualCache per-block profile:")

print(
    "Block  PrefixTok  QueryTok  "
    "FullForward(s)  CachedCalls  "
    "CachedForward(s)  CachedMean(ms)  Merge(s)"
)

for block in dual["blocks"]:
    cached_calls = block["dual_cached_forward_calls"]
    cached_seconds = block["dual_cached_forward_seconds"]

    cached_mean_ms = (
        cached_seconds / cached_calls * 1000
        if cached_calls
        else 0.0
    )

    print(
        f"{block['block']:>5}  "
        f"{block['prefix_tokens']:>9}  "
        f"{block['current_block_tokens']:>8}  "
        f"{block['initial_full_forward_seconds']:>14.3f}  "
        f"{cached_calls:>11}  "
        f"{cached_seconds:>16.3f}  "
        f"{cached_mean_ms:>14.3f}  "
        f"{block['block_merge_seconds']:>8.3f}"
    )

print()
print("Profile files:")
print(base_dir / "nocache_profile.jsonl")
print(base_dir / "prefix_profile.jsonl")
print(dual_dir / "dual_profile.jsonl")
PY

echo
echo "===== DualCache profiling completed ====="
echo "Finished: $(date)"
echo "Results: $OUTDIR"
