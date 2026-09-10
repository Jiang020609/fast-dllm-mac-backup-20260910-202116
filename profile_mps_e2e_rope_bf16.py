from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoConfig, AutoTokenizer

# 应用你现有的 Rotated-K v2b runtime patch。
import rotated_k_v2b_runtime_patch  # noqa: F401

from generate import generate_with_dual_cache
from model.modeling_llada import LLaDAModelLM


DEVICE = torch.device("mps")
DTYPE = torch.bfloat16


def sync() -> None:
    """等待当前 MPS 队列中的计算全部完成。"""
    torch.mps.synchronize()


def run_once(
    *,
    model: LLaDAModelLM,
    tokenizer: AutoTokenizer,
    raw_prompt: str,
    gen_length: int,
    steps: int,
    block_length: int,
    threshold: float | None,
    mask_token_id: int,
) -> tuple[dict[str, float], str, int]:
    metrics: dict[str, float] = {}

    # 清空前面遗留的异步 MPS 工作。
    sync()
    e2e_start = time.perf_counter()

    # --------------------------------------------------
    # 1. Chat template：CPU 字符串处理
    # --------------------------------------------------
    start = time.perf_counter()

    messages = [
        {
            "role": "user",
            "content": raw_prompt,
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    metrics["chat_template_cpu_ms"] = (
        time.perf_counter() - start
    ) * 1000.0

    # --------------------------------------------------
    # 2. Tokenizer：CPU
    # --------------------------------------------------
    start = time.perf_counter()

    token_ids = tokenizer(
        formatted_prompt,
        add_special_tokens=False,
    )["input_ids"]

    metrics["tokenize_cpu_ms"] = (
        time.perf_counter() - start
    ) * 1000.0

    # --------------------------------------------------
    # 3. 构造 CPU tensor
    # --------------------------------------------------
    start = time.perf_counter()

    prompt_cpu = torch.tensor(
        token_ids,
        dtype=torch.long,
    ).unsqueeze(0)

    metrics["tensor_build_cpu_ms"] = (
        time.perf_counter() - start
    ) * 1000.0

    # --------------------------------------------------
    # 4. CPU -> MPS
    # --------------------------------------------------
    sync()
    start = time.perf_counter()

    prompt = prompt_cpu.to(DEVICE)

    sync()
    metrics["host_to_mps_ms"] = (
        time.perf_counter() - start
    ) * 1000.0

    prompt_length = int(prompt.shape[1])

    # --------------------------------------------------
    # 5. 完整 Fast-dLLM 生成
    # --------------------------------------------------
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

    # Python 函数返回时刻。
    call_return = time.perf_counter()

    # 检查函数返回后，MPS 队列是否还有未完成计算。
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

    # --------------------------------------------------
    # 6. MPS -> CPU
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 7. Detokenizer：CPU
    # --------------------------------------------------
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

    measured_stages = [
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
        - sum(metrics[name] for name in measured_stages),
    )

    metrics["generation_ms_per_nfe"] = (
        metrics["generation_total_ms"] / max(nfe, 1)
    )

    return metrics, answer, nfe


def print_report(
    reports: list[dict[str, float]],
    nfes: list[int],
) -> None:
    averaged = {
        key: mean(report[key] for report in reports)
        for key in reports[0]
    }

    e2e_ms = averaged["e2e_total_ms"]

    print()
    print("========== E2E BREAKDOWN ==========")

    # 这些阶段互不重叠，可以看占端到端比例。
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

    # generation_call + generation_tail_sync
    # 是 generation_total 的内部拆分，不与上面的阶段相加。
    print(
        f"{'generation_call_ms':30s}: "
        f"{averaged['generation_call_ms']:11.3f} ms"
    )

    print(
        f"{'generation_tail_sync_ms':30s}: "
        f"{averaged['generation_tail_sync_ms']:11.3f} ms"
    )

    print(
        f"{'generation_ms_per_nfe':30s}: "
        f"{averaged['generation_ms_per_nfe']:11.3f} ms"
    )

    print(
        f"{'e2e_total_ms':30s}: "
        f"{e2e_ms:11.3f} ms"
    )

    print(
        f"{'NFE per run':30s}: "
        f"{nfes}"
    )

    print("===================================")
    print()


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
    print("rope_full_precision=False", flush=True)

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

    for index in range(args.warmups):
        print(
            f"WARMUP {index + 1}/{args.warmups}",
            flush=True,
        )

        _, _, _ = run_once(
            model=model,
            tokenizer=tokenizer,
            raw_prompt=args.prompt,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            threshold=args.threshold,
            mask_token_id=mask_token_id,
        )

    reports: list[dict[str, float]] = []
    nfes: list[int] = []
    answer = ""

    for index in range(args.runs):
        print(
            f"MEASURED_RUN {index + 1}/{args.runs}",
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

    print_report(reports, nfes)

    allocated_gib = (
        torch.mps.current_allocated_memory()
        / (1024 ** 3)
    )

    print(
        "MPS allocated GiB: "
        f"{allocated_gib:.3f}"
    )

    print()
    print("ANSWER")
    print(answer)


if __name__ == "__main__":
    main()
