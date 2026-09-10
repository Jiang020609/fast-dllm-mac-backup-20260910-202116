from __future__ import annotations

import torch

from generate import (
    get_num_transfer_tokens,
    get_transfer_index,
    get_transfer_index_dynamic,
)
from rotated_k_position_context import position_forward_scope


@torch.no_grad()
def generate_with_dual_cache_v2b(
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
            "v2b currently requires gen_length % block_length == 0"
        )

    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        raise ValueError(
            "v2b currently requires steps % num_blocks == 0"
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

        # Full forward：没有 replacement bounds。
        with position_forward_scope():
            out_full = model(
                x,
                use_cache=True,
            )

        past_key_values = out_full.past_key_values
        nfe += 1

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
            # 保留原实现的 early-exit 语义。
            if (
                x[:, block_start:block_end]
                .eq(mask_id)
                .sum()
                == 0
            ):
                break

            # Cached forward：直接传 Python block bounds，
            # 不创建 replace_position bool tensor。
            with position_forward_scope(
                block_start=block_start,
                block_end=block_end,
            ):
                logits_block = model(
                    x[:, block_start:block_end],
                    past_key_values=past_key_values,
                    use_cache=True,
                ).logits

            mask_block = (
                x[:, block_start:block_end] == mask_id
            )

            if factor is None:
                quota = (
                    None
                    if threshold is not None
                    else num_transfer_tokens[:, local_step]
                )

                x0_block, transfer_index_block = (
                    get_transfer_index(
                        logits_block,
                        temperature,
                        remasking,
                        mask_block,
                        x[:, block_start:block_end],
                        quota,
                        threshold,
                    )
                )
            else:
                x0_block, transfer_index_block = (
                    get_transfer_index_dynamic(
                        logits_block,
                        temperature,
                        remasking,
                        mask_block,
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

    return x, nfe
