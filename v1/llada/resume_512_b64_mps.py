import argparse
import csv
import gc
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from bench_prefix_matrix_mps import (
    DEVICE,
    DTYPE,
    MODEL_PATH,
    PROMPTS,
    gib,
    percentile,
    sync,
)
from generate import generate_with_prefix_cache
from model.modeling_llada import LLaDAModelLM


LENGTH = 512
BLOCK_LENGTH = 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    summary_path = Path(args.summary)

    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    # 读取已有原始结果，确定哪些运行已经完成。
    with raw_path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file)
        raw_fields = reader.fieldnames
        all_existing_rows = list(reader)

    if raw_fields is None:
        raise RuntimeError("Raw CSV has no header")

    existing_target_rows = [
        row
        for row in all_existing_rows
        if int(row["length"]) == LENGTH
        and int(row["block_length"]) == BLOCK_LENGTH
    ]

    completed_keys = {
        (int(row["prompt_id"]), int(row["repeat"]))
        for row in existing_target_rows
    }

    planned_keys = {
        (prompt_id, repeat)
        for prompt_id in range(len(PROMPTS))
        for repeat in range(args.repeats)
    }

    missing_keys = sorted(planned_keys - completed_keys)

    print("===== Resume L=512, B=64 =====", flush=True)
    print("Already completed:", len(completed_keys), "/ 36", flush=True)
    print("Missing:", len(missing_keys), flush=True)
    print("Missing keys:", missing_keys, flush=True)

    if not missing_keys:
        print("Nothing to resume.", flush=True)
    else:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")

        print("\nLoading tokenizer...", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )

        print("Loading cache-enabled BF16 model...", flush=True)

        load_start = time.perf_counter()

        model = LLaDAModelLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        ).eval().to(DEVICE)

        sync()
        model_load_seconds = time.perf_counter() - load_start

        print("Model device:", next(model.parameters()).device, flush=True)
        print("Model dtype:", next(model.parameters()).dtype, flush=True)
        print("Model load time:", round(model_load_seconds, 3), "s", flush=True)

        prepared = []

        for prompt_id, (task, prompt_text) in enumerate(PROMPTS):
            chat_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True,
                tokenize=False,
            )

            input_ids = tokenizer(
                chat_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]

            prepared.append(
                {
                    "prompt_id": prompt_id,
                    "task": task,
                    "prompt": prompt_text,
                    "input_ids": input_ids,
                    "prompt_tokens": int(input_ids.shape[1]),
                }
            )

        # 只做一次 warm-up。
        print("\nWarm-up...", flush=True)
        warmup_ids = prepared[0]["input_ids"].to(DEVICE)

        with torch.inference_mode():
            warmup_output, warmup_nfe = generate_with_prefix_cache(
                model,
                warmup_ids,
                steps=LENGTH,
                gen_length=LENGTH,
                block_length=BLOCK_LENGTH,
                temperature=0.0,
                remasking="low_confidence",
                threshold=None,
                factor=None,
            )

        sync()
        print("Warm-up finished, NFE =", warmup_nfe, flush=True)

        del warmup_output
        del warmup_ids
        gc.collect()

        resume_start = time.perf_counter()
        newly_completed = 0

        # 追加写入；每完成一次立即 flush。
        with raw_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=raw_fields)

            for prompt_id, repeat in missing_keys:
                item = prepared[prompt_id]
                input_ids = item["input_ids"].to(DEVICE)

                sync()
                start = time.perf_counter()

                with torch.inference_mode():
                    output, nfe = generate_with_prefix_cache(
                        model,
                        input_ids,
                        steps=LENGTH,
                        gen_length=LENGTH,
                        block_length=BLOCK_LENGTH,
                        temperature=0.0,
                        remasking="low_confidence",
                        threshold=None,
                        factor=None,
                    )

                sync()
                latency = time.perf_counter() - start

                generated_ids = output[:, input_ids.shape[1]:]
                answer = tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]

                row = {
                    "length": LENGTH,
                    "block_length": BLOCK_LENGTH,
                    "steps": LENGTH,
                    "prompt_id": prompt_id,
                    "task": item["task"],
                    "prompt_tokens": item["prompt_tokens"],
                    "repeat": repeat,
                    "nfe": int(nfe),
                    "latency_seconds": latency,
                    "effective_slots_per_second": LENGTH / latency,
                    "mps_current_allocated_gib": gib(
                        torch.mps.current_allocated_memory()
                    ),
                    "mps_driver_allocated_gib": gib(
                        torch.mps.driver_allocated_memory()
                    ),
                    "model_load_seconds": model_load_seconds,
                    "answer": answer.replace("\n", "\\n"),
                }

                writer.writerow(row)
                file.flush()

                newly_completed += 1
                elapsed = time.perf_counter() - resume_start
                remaining = len(missing_keys) - newly_completed
                eta = elapsed / newly_completed * remaining

                print(
                    f"[{newly_completed:02d}/{len(missing_keys)}] "
                    f"prompt={prompt_id:02d} "
                    f"repeat={repeat} "
                    f"NFE={nfe} "
                    f"latency={latency:.3f}s "
                    f"slots/s={LENGTH / latency:.3f} "
                    f"ETA≈{eta / 60:.1f}min",
                    flush=True,
                )

                del output
                del generated_ids
                del input_ids
                gc.collect()

    # 重新读取，确认最终完整。
    with raw_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    target_rows = [
        row
        for row in rows
        if int(row["length"]) == LENGTH
        and int(row["block_length"]) == BLOCK_LENGTH
    ]

    unique_target_rows = {
        (int(row["prompt_id"]), int(row["repeat"])): row
        for row in target_rows
    }

    print("\nFinal unique rows:", len(unique_target_rows), "/ 36", flush=True)

    if len(unique_target_rows) != 36:
        raise RuntimeError(
            f"Resume incomplete: only {len(unique_target_rows)}/36 rows"
        )

    final_rows = list(unique_target_rows.values())

    latencies = [float(row["latency_seconds"]) for row in final_rows]
    throughputs = [
        float(row["effective_slots_per_second"])
        for row in final_rows
    ]
    nfes = [int(row["nfe"]) for row in final_rows]
    memories = [
        float(row["mps_driver_allocated_gib"])
        for row in final_rows
    ]

    summary_row = {
        "length": LENGTH,
        "block_length": BLOCK_LENGTH,
        "num_runs": len(final_rows),
        "latency_mean_s": statistics.mean(latencies),
        "latency_p50_s": statistics.median(latencies),
        "latency_p90_s": percentile(latencies, 0.90),
        "latency_min_s": min(latencies),
        "latency_max_s": max(latencies),
        "slots_per_second_mean": statistics.mean(throughputs),
        "slots_per_second_p50": statistics.median(throughputs),
        "nfe_mean": statistics.mean(nfes),
        "nfe_min": min(nfes),
        "nfe_max": max(nfes),
        "max_mps_driver_gib": max(memories),
    }

    # 保留前 11 组结果，替换或补上最后一组。
    existing_summary_rows = []
    summary_fields = list(summary_row.keys())

    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames:
                summary_fields = reader.fieldnames
            existing_summary_rows = [
                row
                for row in reader
                if not (
                    int(row["length"]) == LENGTH
                    and int(row["block_length"]) == BLOCK_LENGTH
                )
            ]

    temp_path = summary_path.with_suffix(".tmp")

    with temp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(existing_summary_rows)
        writer.writerow(summary_row)

    temp_path.replace(summary_path)

    print("\n===== Resume completed =====", flush=True)
    print("Mean latency:", round(summary_row["latency_mean_s"], 3), "s", flush=True)
    print("P50 latency:", round(summary_row["latency_p50_s"], 3), "s", flush=True)
    print("P90 latency:", round(summary_row["latency_p90_s"], 3), "s", flush=True)
    print(
        "Mean slots/s:",
        round(summary_row["slots_per_second_mean"], 3),
        flush=True,
    )
    print("Updated summary:", summary_path, flush=True)


if __name__ == "__main__":
    main()
