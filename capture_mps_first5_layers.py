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
        "LAYER_CAPTURE_ARMED blocks=0..4",
        flush=True,
    )

    if not torch.mps.profiler.is_metal_capture_enabled():
        raise RuntimeError(
            "Metal capture is disabled. "
            "Run with MTL_CAPTURE_ENABLED=1."
        )

    blocks = model.model.transformer.blocks

    if len(blocks) < 5:
        raise RuntimeError(
            "Expected at least 5 blocks, got {}".format(
                len(blocks)
            )
        )

    capture_stem = "mps_cached_first5_layers_{}".format(
        time.strftime("%Y%m%d_%H%M%S")
    )

    capture_dir = Path("./traces")
    capture_dir.mkdir(parents=True, exist_ok=True)

    capture_cm = None
    capture_started = False
    capture_finished = False

    def begin_capture(module, inputs):
        nonlocal capture_cm
        nonlocal capture_started

        if capture_started:
            return

        # 排空 embedding、mask 等前置工作，
        # 让 trace 从 block 0 边界开始。
        sync()

        capture_cm = torch.mps.profiler.metal_capture(
            capture_stem
        )
        capture_cm.__enter__()
        capture_started = True

        print(
            "CAPTURE_LAYER_BEGIN block=0",
            flush=True,
        )

    def end_capture(module, inputs, output):
        nonlocal capture_finished

        if not capture_started or capture_finished:
            return

        assert capture_cm is not None
        capture_cm.__exit__(None, None, None)
        capture_finished = True

        print(
            "CAPTURE_LAYER_END block=4",
            flush=True,
        )

    start_hook = blocks[0].register_forward_pre_hook(
        begin_capture
    )
    end_hook = blocks[4].register_forward_hook(
        end_capture
    )

    start_time = time.perf_counter()

    try:
        # 只执行一个完整 forward；
        # capture 会在 block 4 后停止，
        # 后续 block 5..31 不进入 gputrace。
        output = forward()
        del output
        sync()
    finally:
        start_hook.remove()
        end_hook.remove()

        if (
            capture_started
            and not capture_finished
            and capture_cm is not None
        ):
            capture_cm.__exit__(None, None, None)

    elapsed = time.perf_counter() - start_time

    if not capture_started:
        raise RuntimeError("Capture never started")

    if not capture_finished:
        raise RuntimeError("Capture did not finish at block 4")

    matches = sorted(
        Path(".").glob(
            "*-{}.gputrace".format(capture_stem)
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected one gputrace, found: {}".format(
                matches
            )
        )

    capture_path = capture_dir / matches[0].name
    matches[0].replace(capture_path)

    print(
        "GPU_TRACE_SAVED {}".format(capture_path),
        flush=True,
    )

    print(
        "FULL_FORWARD_END total_s={:.6f}".format(
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
