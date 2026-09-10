from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from transformers import AutoConfig

# 只应用 Rotated-K v2b patch，不启动 lm-eval。
import rotated_k_v2b_runtime_patch  # noqa: F401

from model.modeling_llada import LLaDAModelLM
from rotated_k_position_context import position_forward_scope


DEVICE = torch.device("mps")
DTYPE = torch.bfloat16


def sync() -> None:
    torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--query-len", type=int, default=32)
    parser.add_argument("--past-len", type=int, default=1331)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    torch.set_grad_enabled(False)

    model_path = (
        Path.home()
        / "models"
        / "LLaDA-8B-Instruct"
    )

    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    if not hasattr(config, "train_max_sequence_length"):
        config.train_max_sequence_length = int(
            config.max_sequence_length
        )

    print("Loading model...", flush=True)

    model = LLaDAModelLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=DTYPE,
        local_files_only=True,
    ).to(DEVICE)

    model.eval()
    sync()

    batch_size = 1
    query_len = int(args.query_len)
    past_len = int(args.past_len)

    n_layers = int(config.n_layers)
    n_heads = int(config.n_heads)
    d_model = int(config.d_model)

    kv_heads = int(
        getattr(config, "n_kv_heads", None)
        or n_heads
    )
    head_dim = d_model // n_heads

    mask_token_id = int(
        getattr(config, "mask_token_id", None)
        or 126336
    )

    input_ids = torch.full(
        (batch_size, query_len),
        mask_token_id,
        dtype=torch.long,
        device=DEVICE,
    )

    cache_shape = (
        batch_size,
        kv_heads,
        past_len,
        head_dim,
    )

    past_key_values = []

    for _ in range(n_layers):
        key = torch.zeros(
            cache_shape,
            dtype=DTYPE,
            device=DEVICE,
        )
        value = torch.zeros_like(key)
        past_key_values.append((key, value))

    sync()

    block_start = past_len - query_len
    block_end = past_len

    if block_start < 0:
        raise ValueError("query_len cannot exceed past_len")

    def forward():
        nonlocal past_key_values

        with position_forward_scope(
            block_start=block_start,
            block_end=block_end,
        ):
            output = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        if output.past_key_values is not None:
            past_key_values = list(
                output.past_key_values
            )

        return output

    # 首次 graph 编译和缓存建立不进入 profile window。
    for _ in range(args.warmups):
        output = forward()
        del output

    sync()

    print(
        "SHAPE_INSPECTION blocks=0..4",
        flush=True,
    )

    blocks = model.model.transformer.blocks

    if len(blocks) < 5:
        raise RuntimeError(
            "Expected at least 5 blocks, got {}".format(
                len(blocks)
            )
        )

    handles = []

    def make_linear_hook(module_name):
        def hook(module, inputs, output):
            x = inputs[0]
            y = output

            if not isinstance(x, torch.Tensor):
                return

            if isinstance(y, (tuple, list)):
                y = y[0]

            # nn.Linear 保存的 weight 形状为 [N, K]，
            # 实际计算是 [M, K] @ [K, N]。
            m = x.numel() // x.shape[-1]
            k = int(x.shape[-1])
            n = int(module.weight.shape[0])

            print(
                "LINEAR {:<48} "
                "input={} weight={} output={} "
                "GEMM=[{},{}]x[{},{}]->[{},{}]".format(
                    module_name,
                    tuple(x.shape),
                    tuple(module.weight.shape),
                    tuple(y.shape),
                    m,
                    k,
                    k,
                    n,
                    m,
                    n,
                ),
                flush=True,
            )

        return hook

    # 只观察 block 0～4，与刚才的 gputrace 范围一致。
    for block_idx in range(5):
        block = blocks[block_idx]

        for sub_name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear):
                full_name = "block{}.{}".format(
                    block_idx,
                    sub_name,
                )

                handles.append(
                    module.register_forward_hook(
                        make_linear_hook(full_name)
                    )
                )

    start_time = time.perf_counter()

    try:
        # 同时记录可能存在的 mm/matmul/bmm/attention 算子形状。
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            output = forward()
            del output
            sync()
    finally:
        for handle in handles:
            handle.remove()

    elapsed = time.perf_counter() - start_time

    print(
        "\n===== ATEN MATRIX OPS =====",
        flush=True,
    )

    wanted_exact = {
        "aten::linear",
        "aten::mm",
        "aten::matmul",
        "aten::bmm",
        "aten::addmm",
    }

    for event in prof.key_averages(
        group_by_input_shape=True
    ):
        name = event.key

        if (
            name in wanted_exact
            or "scaled_dot_product_attention" in name
        ):
            print(
                "{} count={} input_shapes={}".format(
                    name,
                    event.count,
                    getattr(event, "input_shapes", None),
                ),
                flush=True,
            )

    print(
        "\nSHAPE_INSPECTION_END total_s={:.6f}".format(
            elapsed
        ),
        flush=True,
    )

    allocated_gib = (
        torch.mps.current_allocated_memory()
        / (1024 ** 3)
    )

    print(
        "MPS allocated GiB: {:.3f}".format(
            allocated_gib
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
