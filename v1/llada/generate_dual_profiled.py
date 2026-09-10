import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

from generate import (
    get_num_transfer_tokens,
    get_transfer_index,
    get_transfer_index_dynamic,
)


def sync() -> None:
    torch.mps.synchronize()


def timed(function):
    sync()
    start = time.perf_counter()
    result = function()
    sync()
    return result, time.perf_counter() - start


def write_profile(record: dict) -> None:
    output_path = os.environ.get("PROFILE_JSONL")

    if not output_path:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.no_grad()
def generate_with_dual_cache(
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
    tag = os.environ.get("PROFILE_TAG", "dual_cache")

    batch_size = prompt.shape[0]
    prompt_tokens = int(prompt.shape[1])

    assert gen_length % block_length == 0

    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0

    steps_per_block = steps // num_blocks

    phase_seconds = defaultdict(float)
    phase_calls = defaultdict(int)
    block_profiles = []

    def record_phase(name: str, seconds: float) -> None:
        phase_seconds[name] += seconds
        phase_calls[name] += 1

    sync()
    wall_start = time.perf_counter()

    def setup_sequence():
        sequence = torch.full(
            (batch_size, prompt_tokens + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        sequence[:, :prompt_tokens] = prompt
        return sequence

    x, elapsed = timed(setup_sequence)
    record_phase("setup", elapsed)

    full_sequence_tokens = int(x.shape[1])
    nfe = 0

    for block_index in range(num_blocks):
        block_start = prompt_tokens + block_index * block_length
        block_end = block_start + block_length

        def prepare_transfer_schedule():
            block_mask = x[:, block_start:block_end] == mask_id
            return get_num_transfer_tokens(
                block_mask,
                steps_per_block,
            )

        num_transfer_tokens, elapsed = timed(
            prepare_transfer_schedule
        )
        record_phase("transfer_schedule", elapsed)

        block_profile = {
            "block": block_index,
            "prefix_tokens": block_start,
            "full_sequence_tokens": full_sequence_tokens,
            "current_block_tokens": block_length,
            "initial_full_forward_calls": 0,
            "initial_full_forward_seconds": 0.0,
            "dual_cached_forward_calls": 0,
            "dual_cached_forward_seconds": 0.0,
            "initial_transfer_seconds": 0.0,
            "iterative_transfer_seconds": 0.0,
            "block_merge_seconds": 0.0,
            "remaining_check_seconds": 0.0,
            "replace_position_seconds": 0.0,
        }

        # 每个 block 先运行一次完整 forward，建立全序列 KV。
        output, elapsed = timed(
            lambda: model(
                x,
                use_cache=True,
            )
        )

        record_phase("initial_full_forward", elapsed)

        block_profile["initial_full_forward_calls"] += 1
        block_profile["initial_full_forward_seconds"] += elapsed

        past_key_values = output.past_key_values
        nfe += 1

        def make_replace_position():
            replace_position = torch.zeros_like(
                x,
                dtype=torch.bool,
            )
            replace_position[:, block_start:block_end] = True
            return replace_position

        replace_position, elapsed = timed(
            make_replace_position
        )

        record_phase("replace_position_setup", elapsed)
        block_profile["replace_position_seconds"] += elapsed

        global_mask_index = x == mask_id
        global_mask_index[:, block_end:] = False

        def initial_transfer():
            if factor is None:
                quota = (
                    None
                    if threshold is not None
                    else num_transfer_tokens[:, 0]
                )

                x0, transfer_index = get_transfer_index(
                    output.logits,
                    temperature,
                    remasking,
                    global_mask_index,
                    x,
                    quota,
                    threshold,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    output.logits,
                    temperature,
                    remasking,
                    global_mask_index,
                    x,
                    None,
                    factor,
                )

            return torch.where(
                transfer_index,
                x0,
                x,
            )

        x, elapsed = timed(initial_transfer)

        record_phase("initial_transfer", elapsed)
        block_profile["initial_transfer_seconds"] += elapsed

        del output

        # 剩余 steps 只对当前 block 做 forward。
        for local_step in range(1, steps_per_block):
            remaining, elapsed = timed(
                lambda: (
                    x[:, block_start:block_end] == mask_id
                ).sum()
            )

            record_phase("remaining_check", elapsed)
            block_profile["remaining_check_seconds"] += elapsed

            if int(remaining.item()) == 0:
                break

            logits_block, elapsed = timed(
                lambda: model(
                    x[:, block_start:block_end],
                    past_key_values=past_key_values,
                    use_cache=True,
                    replace_position=replace_position,
                ).logits
            )

            record_phase("dual_cached_forward", elapsed)

            block_profile["dual_cached_forward_calls"] += 1
            block_profile["dual_cached_forward_seconds"] += elapsed

            nfe += 1

            block_mask = (
                x[:, block_start:block_end] == mask_id
            )

            def iterative_transfer():
                current_block = x[:, block_start:block_end]

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
                            block_mask,
                            current_block,
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
                            block_mask,
                            current_block,
                            None,
                            factor,
                        )
                    )

                return torch.where(
                    transfer_index_block,
                    x0_block,
                    current_block,
                )

            new_block, elapsed = timed(iterative_transfer)

            record_phase("iterative_transfer", elapsed)
            block_profile["iterative_transfer_seconds"] += elapsed

            def merge_block():
                return torch.cat(
                    [
                        x[:, :block_start],
                        new_block,
                        x[:, block_end:],
                    ],
                    dim=1,
                )

            x, elapsed = timed(merge_block)

            record_phase("block_merge", elapsed)
            block_profile["block_merge_seconds"] += elapsed

            del logits_block
            del new_block

        block_profiles.append(block_profile)

    sync()
    wall_seconds = time.perf_counter() - wall_start

    attributed_seconds = sum(phase_seconds.values())

    record = {
        "mode": "dual_cache",
        "tag": tag,
        "prompt_tokens": prompt_tokens,
        "gen_length": gen_length,
        "block_length": block_length,
        "steps": steps,
        "num_blocks": num_blocks,
        "steps_per_block": steps_per_block,
        "nfe": nfe,
        "wall_seconds": wall_seconds,
        "phase_seconds": dict(phase_seconds),
        "phase_calls": dict(phase_calls),
        "phase_mean_ms": {
            name: (
                phase_seconds[name] / phase_calls[name] * 1000
                if phase_calls[name] > 0
                else 0.0
            )
            for name in phase_seconds
        },
        "unattributed_seconds": (
            wall_seconds - attributed_seconds
        ),
        "mps_current_allocated_gib": (
            torch.mps.current_allocated_memory() / 2**30
        ),
        "mps_driver_allocated_gib": (
            torch.mps.driver_allocated_memory() / 2**30
        ),
        "blocks": block_profiles,
    }

    write_profile(record)

    return x, nfe
