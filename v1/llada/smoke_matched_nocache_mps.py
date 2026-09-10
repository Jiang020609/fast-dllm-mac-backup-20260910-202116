import time

import torch
from transformers import AutoTokenizer

from generate import generate
from model.modeling_llada import LLaDAModelLM


MODEL_PATH = "/Users/z33/models/LLaDA-8B-Instruct"
DEVICE = torch.device("mps")
DTYPE = torch.bfloat16

assert torch.backends.mps.is_available()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = LLaDAModelLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
).eval().to(DEVICE)

torch.mps.synchronize()

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

torch.mps.synchronize()
start = time.perf_counter()

with torch.inference_mode():
    output, nfe = generate(
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

torch.mps.synchronize()
latency = time.perf_counter() - start

answer = tokenizer.batch_decode(
    output[:, prompt.shape[1]:],
    skip_special_tokens=True,
)[0]

print("===== Matched No-Cache Smoke =====")
print("Device:", next(model.parameters()).device)
print("Dtype:", next(model.parameters()).dtype)
print("NFE:", nfe)
print("Latency:", round(latency, 3), "s")
print("Effective slots/s:", round(64 / latency, 3))
print(
    "MPS driver memory:",
    round(torch.mps.driver_allocated_memory() / 2**30, 3),
    "GiB",
)
print("Answer:", answer)
print("Matched no-cache smoke passed")
