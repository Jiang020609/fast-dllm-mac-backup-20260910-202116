import time

import torch
from transformers import AutoTokenizer

from generate import generate_with_prefix_cache
from model.modeling_llada import LLaDAModelLM


MODEL_PATH = "/Users/z33/models/LLaDA-8B-Instruct"
DEVICE = torch.device("mps")
DTYPE = torch.bfloat16


def sync() -> None:
    torch.mps.synchronize()


assert torch.backends.mps.is_available()

print("Loading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("Loading cache-enabled model...", flush=True)

start = time.perf_counter()

model = LLaDAModelLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
).eval().to(DEVICE)

sync()

print("Model device:", next(model.parameters()).device, flush=True)
print("Model dtype:", next(model.parameters()).dtype, flush=True)
print("Load time:", round(time.perf_counter() - start, 3), "s", flush=True)

question = (
    "A bookstore sold 27 books on Monday and twice as many on Tuesday. "
    "It sold 19 books on Wednesday. How many books did it sell in total?"
)

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": question}],
    add_generation_prompt=True,
    tokenize=False,
)

prompt = tokenizer(
    text,
    add_special_tokens=False,
    return_tensors="pt",
)["input_ids"].to(DEVICE)

print("Prompt tokens:", prompt.shape[1], flush=True)
print("Starting Prefix KV Cache generation...", flush=True)

sync()
start = time.perf_counter()

with torch.inference_mode():
    output, nfe = generate_with_prefix_cache(
        model,
        prompt,
        steps=64,
        gen_length=64,
        block_length=32,
        temperature=0.0,
        remasking="low_confidence",
        threshold=None,
        factor=None,
    )

sync()
latency = time.perf_counter() - start

answer = tokenizer.batch_decode(
    output[:, prompt.shape[1]:],
    skip_special_tokens=True,
)[0]

print("\n===== Result =====", flush=True)
print(answer, flush=True)
print("\nLatency:", round(latency, 3), "s", flush=True)
print("NFE:", nfe, flush=True)
print("Effective slots/s:", round(64 / latency, 3), flush=True)
print(
    "MPS driver memory:",
    round(torch.mps.driver_allocated_memory() / 2**30, 3),
    "GiB",
    flush=True,
)
print("Prefix cache smoke test passed", flush=True)
