from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import rotated_k_v2b_runtime_patch  # noqa: F401

from model.modeling_llada import LLaDAModelLM
from rotated_k_position_context import position_forward_scope


DEVICE = torch.device("mps")
DTYPE = torch.bfloat16


def sync() -> None:
    torch.mps.synchronize()


def gib(value: int) -> float:
    return value / (1024 ** 3)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query-len",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--past-lens",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 1331, 2048],
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
    )

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

    if not hasattr(
        config,
        "train_max_sequence_length",
    ):
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

    query_len = int(args.query_len)

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
        (1, query_len),
        mask_token_id,
        dtype=torch.long,
        device=DEVICE,
    )

    print()
    print(
        "past_len,query_len,mean_ms,"
        "allocated_gib,driver_allocated_gib"
    )

    with torch.inference_mode():
        for past_len in args.past_lens:
            if past_len < query_len:
                print(
                    f"SKIP past_len={past_len}: "
                    "smaller than query_len"
                )
                continue

            past_key_values = None
            output = None

            gc.collect()
            torch.mps.empty_cache()
            sync()

            cache_shape = (
                1,
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

                past_key_values.append(
                    (key, value)
                )

            sync()

            block_start = past_len - query_len
            block_end = past_len

            def forward():
                nonlocal past_key_values

                with position_forward_scope(
                    block_start=block_start,
                    block_end=block_end,
                ):
                    result = model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

                if result.past_key_values is not None:
                    past_key_values = list(
                        result.past_key_values
                    )

                return result

            try:
                for _ in range(args.warmups):
                    output = forward()
                    del output

                sync()

                start = time.perf_counter()

                for _ in range(args.iterations):
                    output = forward()
                    del output

                sync()

                elapsed_s = (
                    time.perf_counter() - start
                )

                mean_ms = (
                    elapsed_s
                    * 1000.0
                    / args.iterations
                )

                allocated = (
                    torch.mps.current_allocated_memory()
                )

                if hasattr(
                    torch.mps,
                    "driver_allocated_memory",
                ):
                    driver_allocated = (
                        torch.mps.driver_allocated_memory()
                    )
                else:
                    driver_allocated = 0

                print(
                    f"{past_len},"
                    f"{query_len},"
                    f"{mean_ms:.3f},"
                    f"{gib(allocated):.3f},"
                    f"{gib(driver_allocated):.3f}",
                    flush=True,
                )

            except RuntimeError as exc:
                print(
                    f"FAILED past_len={past_len}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                break

            finally:
                del past_key_values
                gc.collect()
                torch.mps.empty_cache()
                sync()


if __name__ == "__main__":
    main()
