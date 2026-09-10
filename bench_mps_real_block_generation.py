from __future__ import annotations

import argparse
import hashlib
import inspect
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

import rotated_k_v2b_runtime_patch  # noqa: F401

import generate as genmod
from model.modeling_llada import LLaDAModelLM


DEVICE = torch.device("mps")
DTYPE = torch.bfloat16


def sync() -> None:
    torch.mps.synchronize()


def split_blocks(
    transfer_counts: list[int],
    block_length: int,
) -> list[list[int]]:
    """按累计确认 token 数，将每轮 transfer 数量划分到各 block。"""
    blocks: list[list[int]] = []
    current: list[int] = []
    confirmed = 0

    for count in transfer_counts:
        current.append(count)
        confirmed += count

        if confirmed == block_length:
            blocks.append(current)
            current = []
            confirmed = 0
        elif confirmed > block_length:
            raise RuntimeError(
                f"一个 block 累计确认数超过 {block_length}: "
                f"{current}"
            )

    if current:
        raise RuntimeError(
            f"最后一个 block 未完成：累计 {confirmed} tokens"
        )

    return blocks


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--block-lengths",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
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
        "--threshold",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Explain why KV cache accelerates Transformer "
            "inference, focusing on attention computation."
        ),
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

    if not hasattr(config, "train_max_sequence_length"):
        config.train_max_sequence_length = int(
            config.max_sequence_length
        )

    mask_token_id = int(
        getattr(config, "mask_token_id", None)
        or 126336
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

    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )

    prompt_ids = tokenizer(
        formatted_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(DEVICE)

    prompt_length = int(prompt_ids.shape[1])

    # --------------------------------------------------
    # 包装 get_transfer_index：
    # 只保存 transfer_index 引用，不在生成循环里 .item()，
    # 避免额外 CPU-GPU 同步污染生成延迟。
    # --------------------------------------------------
    # generate_with_dual_cache 被 @torch.no_grad() 包装过，
    # 先 unwrap，再修改真正实现函数所使用的 globals。
    generation_impl = inspect.unwrap(
        genmod.generate_with_dual_cache
    )
    generation_globals = generation_impl.__globals__

    original_get_transfer_index = (
        generation_globals["get_transfer_index"]
    )

    capture = False
    captured_transfer_indices: list[torch.Tensor] = []

    def recorded_get_transfer_index(*call_args, **call_kwargs):
        result = original_get_transfer_index(
            *call_args,
            **call_kwargs,
        )

        x0, transfer_index = result

        if capture:
            captured_transfer_indices.append(
                transfer_index.detach().clone()
            )

        return x0, transfer_index

    generation_globals["get_transfer_index"] = recorded_get_transfer_index

    all_summaries: list[dict[str, float | int]] = []

    try:
        for block_length in args.block_lengths:
            if args.gen_length % block_length != 0:
                print(
                    f"SKIP block={block_length}: "
                    "gen_length 不能整除 block_length"
                )
                continue

            num_blocks = args.gen_length // block_length

            if args.steps % num_blocks != 0:
                print(
                    f"SKIP block={block_length}: "
                    "steps 不能整除 num_blocks"
                )
                continue

            print()
            print("=" * 76)
            print(
                f"BLOCK_LENGTH={block_length}  "
                f"THRESHOLD={args.threshold}"
            )
            print("=" * 76)

            # 每种 block shape 单独预热，排除 MPSGraph 首次编译。
            capture = False

            for warmup_index in range(args.warmups):
                print(
                    f"WARMUP "
                    f"{warmup_index + 1}/{args.warmups}",
                    flush=True,
                )

                with torch.inference_mode():
                    warm_output, _ = (
                        genmod.generate_with_dual_cache(
                            model=model,
                            prompt=prompt_ids,
                            steps=args.steps,
                            gen_length=args.gen_length,
                            block_length=block_length,
                            temperature=0.0,
                            remasking="low_confidence",
                            mask_id=mask_token_id,
                            threshold=args.threshold,
                        )
                    )

                sync()
                del warm_output

            run_times_ms: list[float] = []
            run_nfes: list[int] = []
            run_tok_per_nfe: list[float] = []
            run_tps: list[float] = []

            first_counts: list[int] = []
            first_blocks: list[list[int]] = []
            first_answer = ""
            first_hash = ""

            for run_index in range(args.runs):
                captured_transfer_indices.clear()
                capture = True

                sync()
                start = time.perf_counter()

                with torch.inference_mode():
                    output, nfe = (
                        genmod.generate_with_dual_cache(
                            model=model,
                            prompt=prompt_ids,
                            steps=args.steps,
                            gen_length=args.gen_length,
                            block_length=block_length,
                            temperature=0.0,
                            remasking="low_confidence",
                            mask_id=mask_token_id,
                            threshold=args.threshold,
                        )
                    )

                sync()
                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000.0

                capture = False

                # 这些统计全部放在计时窗口之后。
                transfer_counts = [
                    int(index.sum().item())
                    for index in captured_transfer_indices
                ]
                sync()

                print(
                    f"captured_transfer_calls="
                    f"{len(captured_transfer_indices)}, "
                    f"transfer_counts={transfer_counts}",
                    flush=True,
                )

                generated_cpu = output[
                    :,
                    prompt_length:,
                ].to("cpu")

                generated_ids = (
                    generated_cpu[0].tolist()
                )

                remaining_masks = sum(
                    token == mask_token_id
                    for token in generated_ids
                )

                if remaining_masks != 0:
                    raise RuntimeError(
                        f"仍剩 {remaining_masks} 个 mask token"
                    )

                transferred_total = sum(transfer_counts)

                if transferred_total != args.gen_length:
                    raise RuntimeError(
                        "transfer 总数不等于 gen_length："
                        f"{transferred_total} != "
                        f"{args.gen_length}"
                    )

                block_rounds = split_blocks(
                    transfer_counts,
                    block_length,
                )

                tokens_per_nfe = (
                    args.gen_length / int(nfe)
                )

                tokens_per_second = (
                    args.gen_length
                    / (elapsed_ms / 1000.0)
                )

                token_hash = hashlib.sha1(
                    bytes(
                        str(generated_ids),
                        encoding="utf-8",
                    )
                ).hexdigest()[:12]

                answer = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                run_times_ms.append(elapsed_ms)
                run_nfes.append(int(nfe))
                run_tok_per_nfe.append(tokens_per_nfe)
                run_tps.append(tokens_per_second)

                if run_index == 0:
                    first_counts = transfer_counts
                    first_blocks = block_rounds
                    first_answer = answer
                    first_hash = token_hash

                print(
                    f"RUN {run_index + 1}: "
                    f"time={elapsed_ms:.3f} ms, "
                    f"NFE={nfe}, "
                    f"tok/NFE={tokens_per_nfe:.3f}, "
                    f"generated_tok/s="
                    f"{tokens_per_second:.3f}, "
                    f"hash={token_hash}",
                    flush=True,
                )

            mean_ms = statistics.mean(run_times_ms)
            median_ms = statistics.median(run_times_ms)
            mean_nfe = statistics.mean(run_nfes)
            mean_tok_nfe = statistics.mean(
                run_tok_per_nfe
            )
            mean_tps = statistics.mean(run_tps)

            print()
            print("PER-BLOCK TRANSFERS (first measured run)")

            for block_index, counts in enumerate(
                first_blocks,
                start=1,
            ):
                print(
                    f"block {block_index:2d}: "
                    f"rounds={len(counts):2d}, "
                    f"transfers={counts}, "
                    f"mean={statistics.mean(counts):.2f}, "
                    f"max={max(counts)}"
                )

            print()
            print("SUMMARY")
            print(
                f"block_length       : {block_length}"
            )
            print(
                f"mean_time_ms       : {mean_ms:.3f}"
            )
            print(
                f"median_time_ms     : {median_ms:.3f}"
            )
            print(
                f"mean_NFE           : {mean_nfe:.3f}"
            )
            print(
                f"mean_tokens_per_NFE: {mean_tok_nfe:.3f}"
            )
            print(
                f"mean_generated_tps : {mean_tps:.3f}"
            )
            print(
                f"output_hash        : {first_hash}"
            )
            print(
                f"answer_preview     : "
                f"{first_answer[:180]!r}"
            )

            all_summaries.append(
                {
                    "block_length": block_length,
                    "mean_ms": mean_ms,
                    "median_ms": median_ms,
                    "mean_nfe": mean_nfe,
                    "tok_per_nfe": mean_tok_nfe,
                    "generated_tps": mean_tps,
                }
            )

            del output
            torch.mps.empty_cache()
            sync()

    finally:
        generation_globals["get_transfer_index"] = (
            original_get_transfer_index
        )

    print()
    print("=" * 92)
    print("FINAL COMPARISON")
    print("=" * 92)
    print(
        f"{'Block':>7s}"
        f"{'Mean ms':>13s}"
        f"{'Median ms':>13s}"
        f"{'NFE':>10s}"
        f"{'Tok/NFE':>12s}"
        f"{'Gen tok/s':>14s}"
    )

    for row in all_summaries:
        print(
            f"{int(row['block_length']):7d}"
            f"{float(row['mean_ms']):13.3f}"
            f"{float(row['median_ms']):13.3f}"
            f"{float(row['mean_nfe']):10.2f}"
            f"{float(row['tok_per_nfe']):12.3f}"
            f"{float(row['generated_tps']):14.3f}"
        )


if __name__ == "__main__":
    main()
