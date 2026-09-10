import argparse
import csv
import gc
import math
import statistics
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer

from generate import generate_with_prefix_cache
from model.modeling_llada import LLaDAModelLM


MODEL_PATH = "/Users/z33/models/LLaDA-8B-Instruct"
DEVICE = torch.device("mps")
DTYPE = torch.bfloat16

# 前两组优先，保证先得到和 B0/B1 可比的正式结果。
CONFIGS = [
    (64, 32),
    (128, 32),

    (64, 16),
    (64, 64),

    (128, 16),
    (128, 64),

    (256, 16),
    (256, 32),
    (256, 64),

    (512, 16),
    (512, 32),
    (512, 64),
]

PROMPTS = [
    (
        "gsm8k_style",
        "A bookstore sold 27 books on Monday and twice as many on Tuesday. "
        "It sold 19 books on Wednesday. How many books did it sell in total?",
    ),
    (
        "gsm8k_style",
        "A train travels 80 kilometers per hour for 3 hours, then "
        "60 kilometers per hour for 2 hours. How far does it travel?",
    ),
    (
        "gsm8k_style",
        "Mia has 144 stickers. She gives one quarter to her brother and "
        "then gives 18 stickers to her friend. How many stickers remain?",
    ),
    (
        "boolq_style",
        "Question: Can water remain liquid below 0 degrees Celsius under "
        "some physical conditions? Answer yes or no and briefly explain.",
    ),
    (
        "boolq_style",
        "Question: Is every square also a rectangle? "
        "Answer yes or no and briefly explain.",
    ),
    (
        "boolq_style",
        "Question: Does increasing the number of CPU cores always make a "
        "single-threaded program run faster? Answer yes or no and explain.",
    ),
    (
        "arc_style",
        "Which process most directly causes liquid water to become water vapor? "
        "A. Freezing B. Evaporation C. Condensation D. Precipitation. "
        "Give the letter and explanation.",
    ),
    (
        "arc_style",
        "A metal spoon feels colder than a wooden spoon in the same room "
        "mainly because metal: A. has a lower temperature B. conducts heat "
        "more quickly C. contains less energy D. reflects more light. "
        "Give the letter and explanation.",
    ),
    (
        "arc_style",
        "Which component stores instructions and data currently being used "
        "by a computer? A. RAM B. Keyboard C. Monitor D. Power supply. "
        "Give the letter and explanation.",
    ),
    (
        "hellaswag_style",
        "A person puts bread into a toaster and presses the lever. "
        "What is the most likely thing that happens next?",
    ),
    (
        "hellaswag_style",
        "A student finishes writing a report, checks it for errors, and "
        "attaches it to an email. What is the most likely next action?",
    ),
    (
        "hellaswag_style",
        "A runner approaches the finish line far ahead of the other runners. "
        "What is the most likely thing that happens next?",
    ),
]


def sync() -> None:
    torch.mps.synchronize()


def gib(value: int) -> float:
    return value / (1024 ** 3)


def percentile(values, probability: float) -> float:
    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_path = results_dir / f"mac_b2_matrix_{args.run_id}_raw.csv"
    summary_path = results_dir / f"mac_b2_matrix_{args.run_id}_summary.csv"
    error_path = results_dir / f"mac_b2_matrix_{args.run_id}_errors.csv"

    print("===== Mac-B2 Prefix KV Cache Matrix =====", flush=True)
    print("Configs:", CONFIGS, flush=True)
    print("Prompts:", len(PROMPTS), flush=True)
    print("Repeats:", args.repeats, flush=True)
    print("Total planned runs:",
          len(CONFIGS) * len(PROMPTS) * args.repeats,
          flush=True)

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

    raw_fields = [
        "length",
        "block_length",
        "steps",
        "prompt_id",
        "task",
        "prompt_tokens",
        "repeat",
        "nfe",
        "latency_seconds",
        "effective_slots_per_second",
        "mps_current_allocated_gib",
        "mps_driver_allocated_gib",
        "model_load_seconds",
        "answer",
    ]

    summary_fields = [
        "length",
        "block_length",
        "num_runs",
        "latency_mean_s",
        "latency_p50_s",
        "latency_p90_s",
        "latency_min_s",
        "latency_max_s",
        "slots_per_second_mean",
        "slots_per_second_p50",
        "nfe_mean",
        "nfe_min",
        "nfe_max",
        "max_mps_driver_gib",
    ]

    error_fields = [
        "length",
        "block_length",
        "error",
        "traceback",
    ]

    with (
        raw_path.open("w", newline="", encoding="utf-8") as raw_file,
        summary_path.open("w", newline="", encoding="utf-8") as summary_file,
        error_path.open("w", newline="", encoding="utf-8") as error_file,
    ):
        raw_writer = csv.DictWriter(raw_file, fieldnames=raw_fields)
        summary_writer = csv.DictWriter(
            summary_file,
            fieldnames=summary_fields,
        )
        error_writer = csv.DictWriter(error_file, fieldnames=error_fields)

        raw_writer.writeheader()
        summary_writer.writeheader()
        error_writer.writeheader()

        raw_file.flush()
        summary_file.flush()
        error_file.flush()

        completed_configs = 0
        failed_configs = 0
        overall_start = time.perf_counter()

        for config_index, (length, block_length) in enumerate(CONFIGS):
            print(
                f"\n===== Config {config_index + 1}/{len(CONFIGS)}: "
                f"length={length}, block={block_length} =====",
                flush=True,
            )

            try:
                if length % block_length != 0:
                    raise ValueError(
                        f"{length} is not divisible by {block_length}"
                    )

                print("Warm-up...", flush=True)

                warmup_ids = prepared[0]["input_ids"].to(DEVICE)

                for warmup_index in range(args.warmup):
                    with torch.inference_mode():
                        warmup_output, warmup_nfe = (
                            generate_with_prefix_cache(
                                model,
                                warmup_ids,
                                steps=length,
                                gen_length=length,
                                block_length=block_length,
                                temperature=0.0,
                                remasking="low_confidence",
                                threshold=None,
                                factor=None,
                            )
                        )

                    sync()
                    print(
                        f"Warm-up {warmup_index + 1}/{args.warmup} "
                        f"finished, NFE={warmup_nfe}",
                        flush=True,
                    )
                    del warmup_output

                del warmup_ids
                gc.collect()

                config_rows = []
                total_runs = len(prepared) * args.repeats
                completed_runs = 0
                config_start = time.perf_counter()

                for item in prepared:
                    input_ids = item["input_ids"].to(DEVICE)

                    for repeat in range(args.repeats):
                        sync()
                        start = time.perf_counter()

                        with torch.inference_mode():
                            output, nfe = generate_with_prefix_cache(
                                model,
                                input_ids,
                                steps=length,
                                gen_length=length,
                                block_length=block_length,
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

                        completed_runs += 1
                        elapsed = time.perf_counter() - config_start
                        eta_seconds = (
                            elapsed / completed_runs
                            * (total_runs - completed_runs)
                        )

                        row = {
                            "length": length,
                            "block_length": block_length,
                            "steps": length,
                            "prompt_id": item["prompt_id"],
                            "task": item["task"],
                            "prompt_tokens": item["prompt_tokens"],
                            "repeat": repeat,
                            "nfe": int(nfe),
                            "latency_seconds": latency,
                            "effective_slots_per_second": length / latency,
                            "mps_current_allocated_gib": gib(
                                torch.mps.current_allocated_memory()
                            ),
                            "mps_driver_allocated_gib": gib(
                                torch.mps.driver_allocated_memory()
                            ),
                            "model_load_seconds": model_load_seconds,
                            "answer": answer.replace("\n", "\\n"),
                        }

                        raw_writer.writerow(row)
                        raw_file.flush()
                        config_rows.append(row)

                        print(
                            f"[{completed_runs:03d}/{total_runs}] "
                            f"L={length} B={block_length} "
                            f"prompt={item['prompt_id']:02d} "
                            f"repeat={repeat} "
                            f"NFE={nfe} "
                            f"latency={latency:.3f}s "
                            f"slots/s={length / latency:.3f} "
                            f"ETA≈{eta_seconds / 60:.1f}min",
                            flush=True,
                        )

                        del output
                        del generated_ids

                    del input_ids
                    gc.collect()

                latencies = [
                    float(row["latency_seconds"])
                    for row in config_rows
                ]
                throughputs = [
                    float(row["effective_slots_per_second"])
                    for row in config_rows
                ]
                nfes = [int(row["nfe"]) for row in config_rows]
                memories = [
                    float(row["mps_driver_allocated_gib"])
                    for row in config_rows
                ]

                summary_row = {
                    "length": length,
                    "block_length": block_length,
                    "num_runs": len(config_rows),
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

                summary_writer.writerow(summary_row)
                summary_file.flush()

                completed_configs += 1

                print(
                    f"Config completed: mean={summary_row['latency_mean_s']:.3f}s, "
                    f"p50={summary_row['latency_p50_s']:.3f}s, "
                    f"slots/s={summary_row['slots_per_second_mean']:.3f}",
                    flush=True,
                )

            except Exception as error:
                failed_configs += 1

                error_writer.writerow(
                    {
                        "length": length,
                        "block_length": block_length,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                error_file.flush()

                print(
                    f"ERROR in length={length}, block={block_length}: "
                    f"{error!r}",
                    flush=True,
                )
                traceback.print_exc()

            gc.collect()

            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        total_hours = (
            time.perf_counter() - overall_start
        ) / 3600

        print("\n===== Entire matrix completed =====", flush=True)
        print("Completed configs:", completed_configs, flush=True)
        print("Failed configs:", failed_configs, flush=True)
        print("Total time:", round(total_hours, 2), "hours", flush=True)
        print("Raw CSV:", raw_path, flush=True)
        print("Summary CSV:", summary_path, flush=True)
        print("Errors CSV:", error_path, flush=True)


if __name__ == "__main__":
    main()
