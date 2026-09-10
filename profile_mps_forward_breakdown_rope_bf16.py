from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoConfig, AutoTokenizer

import rotated_k_v2b_runtime_patch  # noqa: F401

from generate import generate_with_dual_cache
from model.modeling_llada import LLaDAModelLM


DEVICE = torch.device("mps")
DTYPE = torch.bfloat16


def sync() -> None:
    torch.mps.synchronize()


def run_once(
    *,
    model,
    tokenizer,
    raw_prompt: str,
    gen_length: int,
    steps: int,
    block_length: int,
    threshold: float | None,
    mask_token_id: int,
):
    metrics = {}

    sync()
    e2e_start = time.perf_counter()

    start = time.perf_counter()
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    metrics["chat_template_cpu_ms"] = (
        time.perf_counter() - start
    ) * 1000.0

    start = time.perf_counter()
    token_ids = tokenizer(
        formatted_prompt,
        add_special_tokens=False,
    )["input_ids"]
    metrics["tokenize_cpu_ms"] = (
        time.perf_counter() - start
    ) * 1000.0

    start = time.perf_counter()
    prompt_cpu = torch.tensor(
        token_ids,
        dtype=torch.long,
    ).unsqueeze(0)
    metrics["tensor_build_cpu_ms"] = (
        time.perf_counter() - start
    ) * 1000.0

    sync()
    start = time.perf_counter()
    prompt = prompt_cpu.to(DEVICE)
    sync()
    metrics["host_to_mps_ms"] = (
        time.perf_counter() - start
    ) * 1000.0

    prompt_length = int(prompt.shape[1])

    sync()
    generation_start = time.perf_counter()

    with torch.inference_mode():
        output, nfe = generate_with_dual_cache(
            model=model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            remasking="low_confidence",
            mask_id=mask_token_id,
            threshold=threshold,
        )

    call_return = time.perf_counter()

    sync()
    generation_end = time.perf_counter()

    metrics["generation_call_ms"] = (
        call_return - generation_start
    ) * 1000.0

    metrics["generation_tail_sync_ms"] = (
        generation_end - call_return
    ) * 1000.0

    metrics["generation_total_ms"] = (
        generation_end - generation_start
    ) * 1000.0

    sync()
    start = time.perf_counter()

    generated_ids_cpu = output[
        :,
        prompt_length:,
    ].to("cpu")

    sync()
    metrics["mps_to_host_ms"] = (
        time.perf_counter() - start
    ) * 1000.0

    start = time.perf_counter()

    answer = tokenizer.batch_decode(
        generated_ids_cpu,
        skip_special_tokens=True,
    )[0]

    metrics["detokenize_cpu_ms"] = (
        time.perf_counter() - start
    ) * 1000.0

    sync()
    metrics["e2e_total_ms"] = (
        time.perf_counter() - e2e_start
    ) * 1000.0

    stages = [
        "chat_template_cpu_ms",
        "tokenize_cpu_ms",
        "tensor_build_cpu_ms",
        "host_to_mps_ms",
        "generation_total_ms",
        "mps_to_host_ms",
        "detokenize_cpu_ms",
    ]

    metrics["other_python_ms"] = max(
        0.0,
        metrics["e2e_total_ms"]
        - sum(metrics[name] for name in stages),
    )

    metrics["generation_ms_per_nfe"] = (
        metrics["generation_total_ms"]
        / max(int(nfe), 1)
    )

    return metrics, answer, int(nfe)


def print_e2e_report(reports, nfes) -> None:
    averaged = {
        key: mean(report[key] for report in reports)
        for key in reports[0]
    }

    e2e_ms = averaged["e2e_total_ms"]

    print()
    print("========== E2E BREAKDOWN ==========")

    stages = [
        "chat_template_cpu_ms",
        "tokenize_cpu_ms",
        "tensor_build_cpu_ms",
        "host_to_mps_ms",
        "generation_total_ms",
        "mps_to_host_ms",
        "detokenize_cpu_ms",
        "other_python_ms",
    ]

    for name in stages:
        value = averaged[name]
        percentage = value / e2e_ms * 100.0

        print(
            f"{name:30s}: "
            f"{value:11.3f} ms "
            f"({percentage:6.2f}%)"
        )

    print("-----------------------------------")

    for name in [
        "generation_call_ms",
        "generation_tail_sync_ms",
        "generation_ms_per_nfe",
        "e2e_total_ms",
    ]:
        print(
            f"{name:30s}: "
            f"{averaged[name]:11.3f} ms"
        )

    print(
        f"{'NFE per run':30s}: "
        f"{nfes}"
    )

    print("===================================")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        default=str(
            Path.home()
            / "models"
            / "LLaDA-8B-Instruct"
        ),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Explain why KV cache can accelerate "
            "large language model inference."
        ),
    )

    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gen-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    if args.gen_length % args.block_length != 0:
        raise ValueError(
            "gen_length must be divisible by block_length"
        )

    num_blocks = (
        args.gen_length // args.block_length
    )

    if args.steps % num_blocks != 0:
        raise ValueError(
            "steps must be divisible by number of blocks"
        )

    torch.set_grad_enabled(False)

    model_path = Path(
        args.model_path
    ).expanduser()

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

    mask_token_id = int(
        getattr(
            config,
            "mask_token_id",
            None,
        )
        or 126336
    )

    config.rope_full_precision = False
    print(
        "rope_full_precision=False",
        flush=True,
    )

    print("Loading tokenizer...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
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

    # Warmup 不进入 forward breakdown。
    for index in range(args.warmups):
        print(
            f"WARMUP {index + 1}/{args.warmups}",
            flush=True,
        )

        run_once(
            model=model,
            tokenizer=tokenizer,
            raw_prompt=args.prompt,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            threshold=args.threshold,
            mask_token_id=mask_token_id,
        )

    forward_total_ms = defaultdict(float)
    forward_call_count = defaultdict(int)

    forward_shape_ms = defaultdict(float)
    forward_shape_count = defaultdict(int)

    original_forward = model.forward

    def has_real_cache(past_key_values) -> bool:
        if past_key_values is None:
            return False

        try:
            return len(past_key_values) > 0
        except TypeError:
            return True

    def timed_forward(*forward_args, **forward_kwargs):
        input_ids = forward_kwargs.get(
            "input_ids",
            None,
        )

        if input_ids is None and forward_args:
            input_ids = forward_args[0]

        past_key_values = forward_kwargs.get(
            "past_key_values",
            None,
        )

        if has_real_cache(past_key_values):
            forward_kind = "cached_forward"
        else:
            forward_kind = "full_forward"

        if isinstance(input_ids, torch.Tensor):
            sequence_length = int(
                input_ids.shape[-1]
            )
        else:
            sequence_length = -1

        # 排除前一个阶段遗留的异步工作。
        sync()
        start = time.perf_counter()

        result = original_forward(
            *forward_args,
            **forward_kwargs,
        )

        # 等待本次 forward 真正执行完成。
        sync()
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        forward_total_ms[forward_kind] += elapsed_ms
        forward_call_count[forward_kind] += 1

        shape_key = (
            forward_kind,
            sequence_length,
        )

        forward_shape_ms[shape_key] += elapsed_ms
        forward_shape_count[shape_key] += 1

        return result

    model.forward = timed_forward

    reports = []
    nfes = []
    answer = ""

    try:
        for index in range(args.runs):
            print(
                f"MEASURED_RUN "
                f"{index + 1}/{args.runs}",
                flush=True,
            )

            report, answer, nfe = run_once(
                model=model,
                tokenizer=tokenizer,
                raw_prompt=args.prompt,
                gen_length=args.gen_length,
                steps=args.steps,
                block_length=args.block_length,
                threshold=args.threshold,
                mask_token_id=mask_token_id,
            )

            reports.append(report)
            nfes.append(nfe)
    finally:
        model.forward = original_forward

    print_e2e_report(reports, nfes)

    generation_mean_ms = mean(
        report["generation_total_ms"]
        for report in reports
    )

    measured_runs = max(args.runs, 1)

    full_forward_ms = (
        forward_total_ms["full_forward"]
        / measured_runs
    )

    cached_forward_ms = (
        forward_total_ms["cached_forward"]
        / measured_runs
    )

    full_forward_calls = (
        forward_call_count["full_forward"]
        / measured_runs
    )

    cached_forward_calls = (
        forward_call_count["cached_forward"]
        / measured_runs
    )

    model_forward_ms = (
        full_forward_ms
        + cached_forward_ms
    )

    non_model_generation_ms = max(
        0.0,
        generation_mean_ms
        - model_forward_ms,
    )

    print()
    print("======= GENERATION BREAKDOWN =======")

    print(
        f"{'full_forward_ms':30s}: "
        f"{full_forward_ms:11.3f} ms"
    )

    print(
        f"{'full_forward_calls':30s}: "
        f"{full_forward_calls:11.1f}"
    )

    if full_forward_calls > 0:
        print(
            f"{'full_forward_mean_ms':30s}: "
            f"{full_forward_ms / full_forward_calls:11.3f} ms"
        )

    print(
        f"{'cached_forward_ms':30s}: "
        f"{cached_forward_ms:11.3f} ms"
    )

    print(
        f"{'cached_forward_calls':30s}: "
        f"{cached_forward_calls:11.1f}"
    )

    if cached_forward_calls > 0:
        print(
            f"{'cached_forward_mean_ms':30s}: "
            f"{cached_forward_ms / cached_forward_calls:11.3f} ms"
        )

    print(
        f"{'model_forward_total_ms':30s}: "
        f"{model_forward_ms:11.3f} ms"
    )

    print(
        f"{'non_model_generation_ms':30s}: "
        f"{non_model_generation_ms:11.3f} ms"
    )

    print(
        f"{'non_model_generation_ratio':30s}: "
        f"{non_model_generation_ms / generation_mean_ms * 100:10.2f}%"
    )

    print("------------------------------------")
    print("FORWARD SHAPES")

    for shape_key in sorted(
        forward_shape_count,
        key=lambda item: (
            item[0],
            item[1],
        ),
    ):
        kind, sequence_length = shape_key

        total_ms = (
            forward_shape_ms[shape_key]
            / measured_runs
        )

        count = (
            forward_shape_count[shape_key]
            / measured_runs
        )

        mean_ms = total_ms / max(count, 1)

        print(
            f"{kind:16s} "
            f"seq_len={sequence_length:5d} "
            f"calls={count:6.1f} "
            f"total={total_ms:10.3f} ms "
            f"mean={mean_ms:9.3f} ms"
        )

    print("====================================")

    allocated_gib = (
        torch.mps.current_allocated_memory()
        / (1024 ** 3)
    )

    print(
        f"MPS allocated GiB: "
        f"{allocated_gib:.3f}"
    )

    print()
    print("ANSWER")
    print(answer)


if __name__ == "__main__":
    main()
