from __future__ import annotations

import argparse
import gc
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoConfig

import rotated_k_v2b_runtime_patch  # noqa: F401

from model.modeling_llada import LLaDAModelLM
from rotated_k_position_context import position_forward_scope


DEVICE = torch.device("mps")
DTYPE = torch.bfloat16
Event = torch.mps.Event


def sync() -> None:
    torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
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
        (1, query_len),
        mask_token_id,
        dtype=torch.long,
        device=DEVICE,
    )

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
        past_key_values.append((key, value))

    sync()

    block_start = past_len - query_len
    block_end = past_len

    records = defaultdict(list)
    original_forwards = []
    matched_counts = defaultdict(int)
    collecting = False

    def classify(name: str) -> str | None:
        if name == "":
            return "MODEL TOTAL"

        if name.endswith(".attn_norm"):
            return "RMSNorm: attention"

        if name.endswith(".q_proj"):
            return "Linear: Q"

        if name.endswith(".k_proj"):
            return "Linear: K"

        if name.endswith(".v_proj"):
            return "Linear: V"

        if name.endswith(".attention"):
            return "Attention + RoPE + O"

        if name.endswith(".ff_norm"):
            return "RMSNorm: FFN"

        if name.endswith(".ff_proj"):
            return "Linear: FF gate"

        if name.endswith(".up_proj"):
            return "Linear: FF up"

        if name.endswith(".act"):
            return "Activation"

        if name.endswith(".ff_out"):
            return "Linear: FF down"

        if (
            name.endswith(".ln_f")
            or name.endswith(".final_norm")
        ):
            return "Final norm"

        if (
            name.endswith(".lm_head")
            or name.endswith(".embed_out")
        ):
            return "Output head"

        return None

    def install_timer(
        module: torch.nn.Module,
        category: str,
    ) -> None:
        original_forward = module.forward

        def timed_forward(
            *forward_args,
            _original=original_forward,
            _category=category,
            **forward_kwargs,
        ):
            if not collecting:
                return _original(
                    *forward_args,
                    **forward_kwargs,
                )

            # 嵌套模块计时时，end event 必须在模块执行后创建。
            # 否则父模块提前创建的 end event 会和子模块事件乱序。
            start_event = Event(enable_timing=True)
            start_event.record()

            result = _original(
                *forward_args,
                **forward_kwargs,
            )

            end_event = Event(enable_timing=True)
            end_event.record()

            records[_category].append(
                (start_event, end_event)
            )

            return result

        original_forwards.append(
            (module, original_forward)
        )

        module.forward = timed_forward

    for module_name, module in model.named_modules():
        category = classify(module_name)

        # MPS Event 不做嵌套计时：
        # 本次只测 Q projection。
        if category != "Linear: Q":
            continue

        install_timer(module, category)
        matched_counts[category] += 1

    print()
    print("INSTRUMENTED MODULES")

    for category, count in sorted(
        matched_counts.items()
    ):
        print(f"{category:24s}: {count}")

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

    print("\nWarmup...", flush=True)

    for _ in range(args.warmups):
        output = forward()
        del output

    sync()

    records.clear()
    collecting = True

    wall_start = time.perf_counter()

    for _ in range(args.iterations):
        output = forward()
        del output

    sync()

    wall_ms = (
        time.perf_counter() - wall_start
    ) * 1000.0

    collecting = False

    totals = {}

    for category, pairs in records.items():
        durations = [
            start.elapsed_time(end)
            for start, end in pairs
        ]

        totals[category] = {
            "count": len(durations),
            "total_ms": sum(durations),
        }

    model_total_ms = totals.get(
        "MODEL TOTAL",
        {},
    ).get("total_ms", 0.0)

    child_total_ms = sum(
        item["total_ms"]
        for category, item in totals.items()
        if category != "MODEL TOTAL"
    )

    print()
    print("========== MPS MODULE BREAKDOWN ==========")

    print(
        f"{'Wall total':28s}"
        f"{wall_ms:12.3f} ms"
    )

    print(
        f"{'Wall mean / forward':28s}"
        f"{wall_ms / args.iterations:12.3f} ms"
    )

    print(
        f"{'Event model total':28s}"
        f"{model_total_ms:12.3f} ms"
    )

    print("------------------------------------------")

    print(
        f"{'Category':28s}"
        f"{'Calls':>8s}"
        f"{'Total ms':>13s}"
        f"{'Mean us':>13s}"
        f"{'Share':>10s}"
    )

    child_rows = [
        (category, item)
        for category, item in totals.items()
        if category != "MODEL TOTAL"
    ]

    child_rows.sort(
        key=lambda item: item[1]["total_ms"],
        reverse=True,
    )

    for category, item in child_rows:
        count = item["count"]
        total_ms = item["total_ms"]

        mean_us = (
            total_ms * 1000.0 / count
            if count
            else 0.0
        )

        share = (
            total_ms / model_total_ms * 100.0
            if model_total_ms
            else 0.0
        )

        print(
            f"{category:28s}"
            f"{count:8d}"
            f"{total_ms:13.3f}"
            f"{mean_us:13.3f}"
            f"{share:9.2f}%"
        )

    unaccounted_ms = max(
        0.0,
        model_total_ms - child_total_ms,
    )

    unaccounted_share = (
        unaccounted_ms / model_total_ms * 100.0
        if model_total_ms
        else 0.0
    )

    print("------------------------------------------")

    print(
        f"{'Residual / other':28s}"
        f"{'':8s}"
        f"{unaccounted_ms:13.3f}"
        f"{'':13s}"
        f"{unaccounted_share:9.2f}%"
    )

    print("==========================================")

    for module, original_forward in original_forwards:
        module.forward = original_forward

    records.clear()
    original_forwards.clear()
    gc.collect()


if __name__ == "__main__":
    main()
