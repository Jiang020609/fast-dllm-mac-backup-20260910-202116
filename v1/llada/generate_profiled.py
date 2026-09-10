import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch

from generate import (
    generate_with_dual_cache,
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
    output = os.environ.get("PROFILE_JSONL")

    if not output:
        return

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def finalize_profile(
    *,
    mode: str,
    tag: str,
    prompt_tokens: int,
    gen_length: int,
    block_length: int,
    original_steps: int,
    nfe: int,
    wall_seconds: float,
    phase_seconds: dict,
    phase_calls: dict,
    blocks: list,
) -> None:
    attributed = sum(phase_seconds.values())
    unattributed = wall_seconds - attributed

    record = {
        "mode": mode,
        "tag": tag,
        "prompt_tokens": prompt_tokens,
        "gen_length": gen_length,
        "block_length": block_length,
        "steps": original_steps,
        "num_blocks": gen_length // block_length,
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
        "unattributed_seconds": unattributed,
        "mps_current_allocated_gib": (
            torch.mps.current_allocated_memory() / 2**30
        ),
        "mps_driver_allocated_gib": (
            torch.mps.driver_allocated_memory() / 2**30
        ),
        "blocks": blocks,
    }

    write_profile(record)


@torch.no_grad()
def generate(
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
    tag = os.environ.get("PROFILE_TAG", "nocache")
    prompt_tokens = int(prompt.shape[1])
    original_steps = steps

    phase_seconds = defaultdict(float)
    phase_calls = defaultdict(int)
    blocks = []

    sync()
    wall_start = time.perf_counter()

    def setup():
        value = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        value[:, :prompt.shape[1]] = prompt.clone()
        return value

    x, elapsed = timed(setup)
    phase_seconds["setup"] += elapsed
    phase_calls["setup"] += 1

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    full_input_tokens = int(x.shape[1])

    for block_index in range(num_blocks):
        current_start = (
            prompt.shape[1] + block_index * block_length
        )
        current_end = current_start + block_length

        block_mask_index = (
            x[:, current_start:current_end] == mask_id
        )

        num_transfer_tokens, elapsed = timed(
            lambda: get_num_transfer_tokens(
                block_mask_index,
                steps_per_block,
            )
        )

        phase_seconds["transfer_schedule"] += elapsed
        phase_calls["transfer_schedule"] += 1

        block_profile = {
            "block": block_index,
            "prefix_tokens": current_start,
            "query_tokens": full_input_tokens,
            "full_forward_calls": 0,
            "full_forward_seconds": 0.0,
            "cached_forward_calls": 0,
            "cached_forward_seconds": 0.0,
            "transfer_seconds": 0.0,
        }

        local_step = 0

        while True:
            nfe += 1
            mask_index = x == mask_id

            logits, elapsed = timed(
                lambda: model(x).logits
            )

            phase_seconds["full_forward"] += elapsed
            phase_calls["full_forward"] += 1
            block_profile["full_forward_calls"] += 1
            block_profile["full_forward_seconds"] += elapsed

            mask_index[:, current_end:] = False

            def select_and_update():
                if factor is None:
                    x0, transfer_index = get_transfer_index(
                        logits,
                        temperature,
                        remasking,
                        mask_index,
                        x,
                        (
                            num_transfer_tokens[:, local_step]
                            if threshold is None
                            else None
                        ),
                        threshold,
                    )
                else:
                    x0, transfer_index = (
                        get_transfer_index_dynamic(
                            logits,
                            temperature,
                            remasking,
                            mask_index,
                            x,
                            None,
                            factor,
                        )
                    )

                x[transfer_index] = x0[transfer_index]

            _, elapsed = timed(select_and_update)

            phase_seconds["transfer_update"] += elapsed
            phase_calls["transfer_update"] += 1
            block_profile["transfer_seconds"] += elapsed

            local_step += 1
            del logits

            remaining = (
                x[:, current_start:current_end] == mask_id
            ).sum()

            if int(remaining.item()) == 0:
                break

        blocks.append(block_profile)

    sync()
    wall_seconds = time.perf_counter() - wall_start

    finalize_profile(
        mode="matched_nocache",
        tag=tag,
        prompt_tokens=prompt_tokens,
        gen_length=gen_length,
        block_length=block_length,
        original_steps=original_steps,
        nfe=nfe,
        wall_seconds=wall_seconds,
        phase_seconds=phase_seconds,
        phase_calls=phase_calls,
        blocks=blocks,
    )

    return x, nfe


@torch.no_grad()
def generate_with_prefix_cache(
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
    tag = os.environ.get("PROFILE_TAG", "prefix_cache")
    prompt_tokens = int(prompt.shape[1])
    original_steps = steps

    phase_seconds = defaultdict(float)
    phase_calls = defaultdict(int)
    blocks = []

    sync()
    wall_start = time.perf_counter()

    def setup():
        value = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        value[:, :prompt.shape[1]] = prompt.clone()
        return value

    x, elapsed = timed(setup)
    phase_seconds["setup"] += elapsed
    phase_calls["setup"] += 1

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    full_input_tokens = int(x.shape[1])

    for block_index in range(num_blocks):
        current_start = (
            prompt.shape[1] + block_index * block_length
        )
        current_end = current_start + block_length

        block_mask_index = (
            x[:, current_start:current_end] == mask_id
        )

        num_transfer_tokens, elapsed = timed(
            lambda: get_num_transfer_tokens(
                block_mask_index,
                steps_per_block,
            )
        )

        phase_seconds["transfer_schedule"] += elapsed
        phase_calls["transfer_schedule"] += 1

        block_profile = {
            "block": block_index,
            "prefix_tokens": current_start,
            "initial_full_query_tokens": full_input_tokens,
            "cached_query_tokens": int(
                x[:, current_start:].shape[1]
            ),
            "initial_full_forward_calls": 0,
            "initial_full_forward_seconds": 0.0,
            "cached_forward_calls": 0,
            "cached_forward_seconds": 0.0,
            "kv_crop_seconds": 0.0,
            "transfer_seconds": 0.0,
        }

        output, elapsed = timed(
            lambda: model(x, use_cache=True)
        )

        phase_seconds["initial_full_forward"] += elapsed
        phase_calls["initial_full_forward"] += 1
        block_profile["initial_full_forward_calls"] += 1
        block_profile["initial_full_forward_seconds"] += elapsed

        past_key_values = output.past_key_values

        mask_index = x == mask_id
        mask_index[:, current_end:] = False

        def first_transfer():
            if factor is None:
                x0, transfer_index = get_transfer_index(
                    output.logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    (
                        num_transfer_tokens[:, 0]
                        if threshold is None
                        else None
                    ),
                    threshold,
                )
            else:
                x0, transfer_index = (
                    get_transfer_index_dynamic(
                        output.logits,
                        temperature,
                        remasking,
                        mask_index,
                        x,
                        None,
                        factor,
                    )
                )

            x[transfer_index] = x0[transfer_index]

        _, elapsed = timed(first_transfer)

        phase_seconds["transfer_update"] += elapsed
        phase_calls["transfer_update"] += 1
        block_profile["transfer_seconds"] += elapsed

        def crop_prefix_cache():
            cropped = []

            for layer_cache in past_key_values:
                cropped_layer = tuple(
                    tensor[:, :, :current_start]
                    for tensor in layer_cache
                )
                cropped.append(cropped_layer)

            return cropped

        past_key_values, elapsed = timed(crop_prefix_cache)

        phase_seconds["kv_crop"] += elapsed
        phase_calls["kv_crop"] += 1
        block_profile["kv_crop_seconds"] += elapsed

        nfe += 1
        local_step = 1

        while True:
            remaining = (
                x[:, current_start:current_end] == mask_id
            ).sum()

            if int(remaining.item()) == 0:
                break

            nfe += 1

            suffix = x[:, current_start:]
            mask_index = suffix == mask_id
            mask_index[:, block_length:] = False

            logits, elapsed = timed(
                lambda: model(
                    suffix,
                    past_key_values=past_key_values,
                    use_cache=True,
                ).logits
            )

            phase_seconds["cached_forward"] += elapsed
            phase_calls["cached_forward"] += 1
            block_profile["cached_forward_calls"] += 1
            block_profile["cached_forward_seconds"] += elapsed

            def cached_transfer():
                if factor is None:
                    x0, transfer_index = get_transfer_index(
                        logits,
                        temperature,
                        remasking,
                        mask_index,
                        suffix,
                        (
                            num_transfer_tokens[:, local_step]
                            if threshold is None
                            else None
                        ),
                        threshold,
                    )
                else:
                    x0, transfer_index = (
                        get_transfer_index_dynamic(
                            logits,
                            temperature,
                            remasking,
                            mask_index,
                            suffix,
                            None,
                            factor,
                        )
                    )

                x[:, current_start:][transfer_index] = (
                    x0[transfer_index]
                )

            _, elapsed = timed(cached_transfer)

            phase_seconds["transfer_update"] += elapsed
            phase_calls["transfer_update"] += 1
            block_profile["transfer_seconds"] += elapsed

            local_step += 1
            del logits

        blocks.append(block_profile)

    sync()
    wall_seconds = time.perf_counter() - wall_start

    finalize_profile(
        mode="prefix_cache",
        tag=tag,
        prompt_tokens=prompt_tokens,
        gen_length=gen_length,
        block_length=block_length,
        original_steps=original_steps,
        nfe=nfe,
        wall_seconds=wall_seconds,
        phase_seconds=phase_seconds,
        phase_calls=phase_calls,
        blocks=blocks,
    )

    return x, nfe
