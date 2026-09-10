#!/bin/bash
set -euo pipefail

ROOT="$HOME/Fast-dLLM/v1/llada"
PY="$HOME/fastdllm-mps/bin/python"
ACCELERATE="$HOME/fastdllm-mps/bin/accelerate"
MODEL="$HOME/models/LLaDA-8B-Instruct"

RUN_ID="${1:?Missing RUN_ID}"
OUTDIR="$HOME/Fast-dLLM/results/gsm8k_profile_$RUN_ID"

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

COMMON_ARGS="model_path=$MODEL,device=mps,is_check_greedy=False,gen_length=256,steps=256,block_length=32,show_speed=True"

echo "===== No-Cache matched profile ====="

PROFILE_JSONL="$OUTDIR/nocache_profile.jsonl" \
PROFILE_TAG="gsm8k_5shot_nocache" \
"$ACCELERATE" launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  "$ROOT/eval_llada_profile.py" \
    --model llada_dist \
    --model_args "$COMMON_ARGS" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 1 \
    --confirm_run_unsafe_code \
    --output_path "$OUTDIR/nocache_eval" \
    --log_samples

echo
echo "===== Prefix-Cache matched profile ====="

PROFILE_JSONL="$OUTDIR/prefix_profile.jsonl" \
PROFILE_TAG="gsm8k_5shot_prefix" \
"$ACCELERATE" launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  "$ROOT/eval_llada_profile.py" \
    --model llada_dist \
    --model_args "$COMMON_ARGS,use_cache=True" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 1 \
    --confirm_run_unsafe_code \
    --output_path "$OUTDIR/prefix_eval" \
    --log_samples

echo
echo "===== Printing profile comparison ====="

"$PY" - "$OUTDIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def load_one(name):
    path = root / name
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if not rows:
        raise RuntimeError(f"No records in {path}")
    return rows[-1]

no = load_one("nocache_profile.jsonl")
cache = load_one("prefix_profile.jsonl")

def forward_total(record):
    phases = record["phase_seconds"]
    return (
        phases.get("full_forward", 0.0)
        + phases.get("initial_full_forward", 0.0)
        + phases.get("cached_forward", 0.0)
    )

print()
print("=" * 92)
print("Matched GSM8K 5-shot phase profile")
print("=" * 92)

header = [
    "Mode",
    "Wall(s)",
    "NFE",
    "Forward(s)",
    "Transfer(s)",
    "KV crop(s)",
    "Forward %",
    "MPS GiB",
]

rows = []

for name, record in [
    ("No-cache", no),
    ("Prefix cache", cache),
]:
    phases = record["phase_seconds"]
    forward = forward_total(record)
    wall = record["wall_seconds"]

    rows.append([
        name,
        f"{wall:.3f}",
        str(record["nfe"]),
        f"{forward:.3f}",
        f"{phases.get('transfer_update', 0.0):.3f}",
        f"{phases.get('kv_crop', 0.0):.3f}",
        f"{forward / wall * 100:.2f}%",
        f"{record['mps_driver_allocated_gib']:.3f}",
    ])

widths = [
    max(len(header[i]), *(len(row[i]) for row in rows))
    for i in range(len(header))
]

def show(row):
    print("  ".join(
        value.rjust(widths[index])
        for index, value in enumerate(row)
    ))

show(header)
show(["-" * width for width in widths])
for row in rows:
    show(row)

print()
print(
    "Profile wall-time speedup: "
    f"{no['wall_seconds'] / cache['wall_seconds']:.3f}x"
)
print(
    "Total forward-time reduction: "
    f"{(1 - forward_total(cache) / forward_total(no)) * 100:.2f}%"
)

print()
print("No-cache forward:")
print(
    f"  calls={no['phase_calls'].get('full_forward', 0)}, "
    f"mean="
    f"{no['phase_mean_ms'].get('full_forward', 0.0):.3f} ms"
)

print()
print("Prefix-cache forward:")
print(
    f"  initial full calls="
    f"{cache['phase_calls'].get('initial_full_forward', 0)}, "
    f"mean="
    f"{cache['phase_mean_ms'].get('initial_full_forward', 0.0):.3f} ms"
)
print(
    f"  cached calls="
    f"{cache['phase_calls'].get('cached_forward', 0)}, "
    f"mean="
    f"{cache['phase_mean_ms'].get('cached_forward', 0.0):.3f} ms"
)

print()
print("Prefix-cache per-block profile:")
print(
    "Block  PrefixTok  CachedQueryTok  "
    "FullForward(s)  CachedCalls  CachedForward(s)  CachedMean(ms)"
)

for block in cache["blocks"]:
    calls = block["cached_forward_calls"]
    cached_seconds = block["cached_forward_seconds"]
    cached_mean = cached_seconds / calls * 1000 if calls else 0.0

    print(
        f"{block['block']:>5}  "
        f"{block['prefix_tokens']:>9}  "
        f"{block['cached_query_tokens']:>14}  "
        f"{block['initial_full_forward_seconds']:>14.3f}  "
        f"{calls:>11}  "
        f"{cached_seconds:>16.3f}  "
        f"{cached_mean:>14.3f}"
    )

print()
print("Raw profiles:")
print(root / "nocache_profile.jsonl")
print(root / "prefix_profile.jsonl")
PY

echo
echo "===== Matched profiling completed ====="
echo "Results: $OUTDIR"
