from __future__ import annotations

import atexit
import statistics
import time

import torch

from generate import (
    get_num_transfer_tokens,
    get_transfer_index,
    get_transfer_index_dynamic,
)
from rotated_k_position_context import position_forward_scope


ALL_RESULTS: list[dict[str, float | int]] = []


def timed_model_call(callback):
    # 将前面的异步工作排空，避免算入本次 forward。
    torch.mps.synchronize()
    start = time.perf_counter()

    output = callback()

    # 确保本次 forward 真正执行完成。
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    return output, elapsed


@atexit.register
def print_aggregate_profile() -> None:
    if not ALL_RESULTS:
        return

    def mean(key: str) -> float:
        return statistics.mean(
            float(result[key])
            for result in ALL_RESULTS
        )

    total = mean("total_s")
    full = mean("full_s")
    cached = mean("cached_s")
    other = mean("other_s")

    print()
    print("=" * 80)
    print("COARSE_PROFILE_AGGREGATE")
    print(f"samples={len(ALL_RESULTS)}")
    print(f"mean_total_s={total:.6f}")
    print(f"mean_full_s={full:.6f}")
    print(f"mean_cached_s={cached:.6f}")
    print(f"mean_other_s={other:.6f}")
    print(f"full_pct={100.0 * full / total:.2f}")
    print(f"cached_pct={100.0 * cached / total:.2f}")
    print(f"other_pct={100.0 * other / total:.2f}")
    print(
        "mean_full_call_ms="
        f"{1000.0 * full / mean('full_calls'):.3f}"
    )
    print(
        "mean_cached_call_ms="
        f"{1000.0 * cached / mean('cached_calls'):.3f}"
    )
    print("=" * 80)


@torch.no_grad()
def generate_with_dual_cache_v2b_profiled(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    factor=None,
):
    batch_size = prompt.shape[0]
    prompt_length = int(prompt.shape[1])

    if gen_length % block_length != 0:
        raise ValueError(
            "gen_length must be divisible by block_length"
        )

    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        raise ValueError(
            "steps must be divisible by num_blocks"
        )

    steps_per_block = steps // num_blocks

    x = torch.full(
        (batch_size, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_length] = prompt

    nfe = 0
    full_time = 0.0
    cached_time = 0.0
    full_calls = 0
    cached_calls = 0

    torch.mps.synchronize()
    generation_start = time.perf_counter()

    for block_index in range(num_blocks):
        block_start = (
            prompt_length
            + block_index * block_length
        )
        block_end = block_start + block_length

        block_mask_index = (
            x[:, block_start:block_end] == mask_id
        )

        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index,
            steps_per_block,
        )

        def run_full_forward():
            with position_forward_scope():
                return model(
                    x,
                    use_cache=True,
                )

        out_full, elapsed = timed_model_call(
            run_full_forward
        )

        full_time += elapsed
        full_calls += 1
        nfe += 1

        past_key_values = out_full.past_key_values

        global_mask_index = x == mask_id
        global_mask_index[:, block_end:] = False

        if factor is None:
            quota0 = (
                None
                if threshold is not None
                else num_transfer_tokens[:, 0]
            )

            x0, transfer_index = get_transfer_index(
                out_full.logits,
                temperature,
                remasking,
                global_mask_index,
                x,
                quota0,
                threshold,
            )
        else:
            x0, transfer_index = (
                get_transfer_index_dynamic(
                    out_full.logits,
                    temperature,
                    remasking,
                    global_mask_index,
                    x,
                    None,
                    factor,
                )
            )

        x = torch.where(
            transfer_index,
            x0,
            x,
        )

        for local_step in range(
            1,
            steps_per_block,
        ):
            if (
                x[:, block_start:block_end]
                .eq(mask_id)
                .sum()
                == 0
            ):
                break

            def run_cached_forward():
                with position_forward_scope(
                    block_start=block_start,
                    block_end=block_end,
                ):
                    return model(
                        x[:, block_start:block_end],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

            cached_output, elapsed = timed_model_call(
                run_cached_forward
            )

            cached_time += elapsed
            cached_calls += 1

            logits_block = cached_output.logits

            # 显式接回返回的 cache，避免依赖隐式 aliasing。
            if cached_output.past_key_values is not None:
                past_key_values = (
                    cached_output.past_key_values
                )

            mask_block = (
                x[:, block_start:block_end] == mask_id
            )

            if factor is None:
                quota = (
                    None
                    if threshold is not None
                    else num_transfer_tokens[
                        :,
                        local_step,
                    ]
                )

                (
                    x0_block,
                    transfer_index_block,
                ) = get_transfer_index(
                    logits_block,
                    temperature,
                    remasking,
                    mask_block,
                    x[:, block_start:block_end],
                    quota,
                    threshold,
                )
            else:
                (
                    x0_block,
                    transfer_index_block,
                ) = get_transfer_index_dynamic(
                    logits_block,
                    temperature,
                    remasking,
                    mask_block,
                    x[:, block_start:block_end],
                    None,
                    factor,
                )

            old_block = x[
                :,
                block_start:block_end,
            ]

            new_block = torch.where(
                transfer_index_block,
                x0_block,
                old_block,
            )

            x[:, block_start:block_end] = new_block

            nfe += 1

    torch.mps.synchronize()
    total_time = (
        time.perf_counter()
        - generation_start
    )

    other_time = max(
        0.0,
        total_time
        - full_time
        - cached_time,
    )

    result = {
        "total_s": total_time,
        "full_s": full_time,
        "cached_s": cached_time,
        "other_s": other_time,
        "full_calls": full_calls,
        "cached_calls": cached_calls,
        "nfe": nfe,
    }

    ALL_RESULTS.append(result)

    sample_number = len(ALL_RESULTS)

    print(
        "COARSE_PROFILE "
        f"sample={sample_number} "
        f"total_s={total_time:.6f} "
        f"full_s={full_time:.6f} "
        f"cached_s={cached_time:.6f} "
        f"other_s={other_time:.6f} "
        f"full_calls={full_calls} "
        f"cached_calls={cached_calls} "
        f"nfe={nfe} "
        f"full_pct={100.0 * full_time / total_time:.2f} "
        f"cached_pct={100.0 * cached_time / total_time:.2f} "
        f"other_pct={100.0 * other_time / total_time:.2f}"
    )

    return x, nfe
