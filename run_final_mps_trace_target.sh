#!/bin/zsh
set -euo pipefail

ROOT="$HOME/Fast-dLLM/v1/llada"

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

exec "$HOME/fastdllm-mps/bin/python" \
  "$ROOT/eval_llada_mps_trace_wrapper.py" \
    --model llada_dist \
    --model_args \
"model_path=$HOME/models/LLaDA-8B-Instruct,device=mps,is_check_greedy=False,gen_length=256,steps=256,block_length=32,show_speed=True,use_cache=True,dual_cache=True,threshold=0.80" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 1 \
    --confirm_run_unsafe_code \
    --output_path \
"$HOME/Fast-dLLM/results/final_mps_trace_eval" \
    --log_samples
