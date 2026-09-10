from __future__ import annotations

import os

import torch

from generate import (
    get_num_transfer_tokens,
    get_transfer_index,
    get_transfer_index_dynamic,
)
from rotated_k_position_context import position_forward_scope


def get_refresh_interval() -> int:
    interval = int(
        os.environ.get(
            "LLADA_CACHE_REFRESH_INTERVAL",
            "1",
        )
    )

    if interval < 1:
        raise ValueError(
            "LLADA_CACHE_REFRESH_INTERVAL must be >= 1"
        )

    return interval


@torch.no_grad()
def generate_with_dual_cache_v2b_refresh(
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
    refresh_interval = get_refresh_interval()

    x = torch.full(
        (batch_size, prompt_length + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_length] = prompt

    nfe = 0
    full_refreshes = 0
    reused_block_starts = 0
    past_key_values = None

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

        need_full_refresh = (
            past_key_values is None
            or block_index % refresh_interval == 0
        )

        if need_full_refresh:
            # 与原 v2b 相同：对整个序列重建 cache。
            with position_forward_scope():
                step0_output = model(
                    x,
                    use_cache=True,
                )

            past_key_values = (
                step0_output.past_key_values
            )
            nfe += 1
            full_refreshes += 1

            # 保持 interval=1 与原 v2b 的 step-0 语义一致。
            global_mask_index = x == mask_id
            global_mask_index[:, block_end:] = False

            if factor is None:
                quota0 = (
                    None
                    if threshold is not None
                    else num_transfer_tokens[:, 0]
                )

                x0, transfer_index = get_transfer_index(
                    step0_output.logits,
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
                        step0_output.logits,
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

        else:
            # 近似路径：
            # 不重建整个序列，只计算新 block，
            # 并在旧 rotated-K cache 中替换当前区间。
            with position_forward_scope(
                block_start=block_start,
                block_end=block_end,
            ):
                step0_output = model(
                    x[:, block_start:block_end],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            if step0_output.past_key_values is not None:
                past_key_values = (
                    step0_output.past_key_values
                )

            nfe += 1
            reused_block_starts += 1

            mask0 = (
                x[:, block_start:block_end] == mask_id
            )

            if factor is None:
                quota0 = (
                    None
                    if threshold is not None
                    else num_transfer_tokens[:, 0]
                )

                x0_block, transfer0_block = (
                    get_transfer_index(
                        step0_output.logits,
                        temperature,
                        remasking,
                        mask0,
                        x[:, block_start:block_end],
                        quota0,
                        threshold,
                    )
                )
            else:
                x0_block, transfer0_block = (
                    get_transfer_index_dynamic(
                        step0_output.logits,
                        temperature,
                        remasking,
                        mask0,
                        x[:, block_start:block_end],
                        None,
                        factor,
                    )
                )

            old_block = x[
                :,
                block_start:block_end,
            ]

            new_block = torch.where(
                transfer0_block,
                x0_block,
                old_block,
            )

            x = torch.cat(
                (
                    x[:, :block_start],
                    new_block,
                    x[:, block_end:],
                ),
                dim=1,
            )

        # 后续 refinement 与 v2b 保持一致。
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

            with position_forward_scope(
                block_start=block_start,
                block_end=block_end,
            ):
                cached_output = model(
                    x[:, block_start:block_end],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            if cached_output.past_key_values is not None:
                past_key_values = (
                    cached_output.past_key_values
                )

            logits_block = cached_output.logits

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

            x = torch.cat(
                (
                    x[:, :block_start],
                    new_block,
                    x[:, block_end:],
                ),
                dim=1,
            )

            nfe += 1

    print(
        "CACHE_REFRESH "
        f"interval={refresh_interval} "
        f"full_refreshes={full_refreshes} "
        f"reused_block_starts={reused_block_starts} "
        f"nfe={nfe}"
    )

    return x, nfe
