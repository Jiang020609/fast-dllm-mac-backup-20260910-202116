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
        "PROFILE_WINDOW_BEGIN iterations={}".format(
            args.iterations
        ),
        flush=True,
    )

    tb_dir = Path("./tb_logs/mps_cached_natural")
    tb_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = tensorboard_trace_handler(
        str(tb_dir)
    )

    def trace_handler(prof):
        print(
            "\nPYTORCH_OPERATOR_PROFILE\n",
            flush=True,
        )

        print(
            prof.key_averages(
                group_by_input_shape=True,
            ).table(
                sort_by="self_cpu_time_total",
                row_limit=30,
            ),
            flush=True,
        )

        tb_writer(prof)

    torch.mps.profiler.start(
        mode="interval,event",
        wait_until_completed=False,
    )

    start = time.perf_counter()

    try:
        with profile(
            activities=[
                ProfilerActivity.CPU,
            ],
            schedule=schedule(
                skip_first=5,
                wait=0,
                warmup=2,
                active=5,
                repeat=1,
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
        ) as prof:
            # 不在每次 forward 后同步，保留自然提交流水线。
            for _ in range(args.iterations):
                with record_function(
                    "decode_iteration"
                ):
                    with record_function(
                        "model_forward"
                    ):
                        output = forward()

                    del output

                prof.step()

        sync()
        elapsed = time.perf_counter() - start
    finally:
        torch.mps.profiler.stop()

    mean_ms = (
        elapsed * 1000.0 / args.iterations
    )

    print(
        "PROFILE_WINDOW_END total_s={:.6f} mean_ms={:.3f}".format(
            elapsed,
            mean_ms,
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
