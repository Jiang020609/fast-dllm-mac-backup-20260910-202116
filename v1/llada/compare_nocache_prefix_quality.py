import argparse
import csv
import difflib
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def restore_text(text: str) -> str:
    return text.replace("\\n", "\n")


def normalize_text(text: str) -> str:
    text = restore_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[`*_#$]", "", text)
    text = re.sub(r"\s*([=+\-×*/,:;.!?])\s*", r"\1", text)
    return text.strip()


def final_signature(text: str) -> str:
    """
    启发式抽取最终答案，只用于快速一致性检查，不等同于正式 benchmark 评分。
    """
    restored = restore_text(text)

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", restored)
    if boxed:
        return "boxed:" + compact_text(boxed[-1])

    choice = re.findall(
        r"(?i)\b(?:correct\s+answer|answer)\s*(?:is|:)?\s*\(?([A-D])\)?\b",
        restored,
    )
    if choice:
        return "choice:" + choice[-1].upper()

    stripped = restored.strip()
    first_token = re.match(r"(?i)^(yes|no)\b", stripped)
    if first_token:
        return "yesno:" + first_token.group(1).lower()

    numbers = re.findall(
        r"(?<![\w.])-?\d+(?:\.\d+)?",
        restored.replace(",", ""),
    )
    if numbers:
        return "number:" + numbers[-1]

    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", stripped)
        if part.strip()
    ]
    final_sentence = sentences[-1] if sentences else stripped
    return "text:" + compact_text(final_sentence)


def read_filtered(path: Path, lengths: set[int]) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    return [
        row
        for row in rows
        if int(row["length"]) in lengths
        and int(row["block_length"]) == 32
    ]


def most_common(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nocache", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    lengths = {64, 128}

    nocache_path = Path(args.nocache)
    prefix_path = Path(args.prefix)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_path = output_dir / "quality_pair_comparison.csv"
    summary_path = output_dir / "quality_summary.csv"
    mismatch_path = output_dir / "quality_mismatches.txt"

    nocache_rows = read_filtered(nocache_path, lengths)
    prefix_rows = read_filtered(prefix_path, lengths)

    nocache_index = {
        (
            int(row["length"]),
            int(row["prompt_id"]),
            int(row["repeat"]),
        ): row
        for row in nocache_rows
    }

    prefix_index = {
        (
            int(row["length"]),
            int(row["prompt_id"]),
            int(row["repeat"]),
        ): row
        for row in prefix_rows
    }

    common_keys = sorted(set(nocache_index) & set(prefix_index))
    expected_pairs = len(lengths) * 12 * 3

    print("No-cache rows:", len(nocache_rows))
    print("Prefix-cache rows:", len(prefix_rows))
    print("Matched pairs:", len(common_keys), "/", expected_pairs)

    if len(common_keys) != expected_pairs:
        missing_no = sorted(set(prefix_index) - set(nocache_index))
        missing_prefix = sorted(set(nocache_index) - set(prefix_index))
        raise RuntimeError(
            f"Pairing incomplete. "
            f"Missing no-cache={missing_no}, "
            f"missing prefix={missing_prefix}"
        )

    pair_fields = [
        "length",
        "prompt_id",
        "repeat",
        "task",
        "raw_exact_match",
        "normalized_exact_match",
        "final_signature_match",
        "text_similarity",
        "nocache_final_signature",
        "prefix_final_signature",
        "nocache_answer",
        "prefix_answer",
    ]

    pair_rows = []

    with pair_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=pair_fields)
        writer.writeheader()

        for key in common_keys:
            no_row = nocache_index[key]
            prefix_row = prefix_index[key]

            no_answer = restore_text(no_row["answer"])
            prefix_answer = restore_text(prefix_row["answer"])

            no_normalized = normalize_text(no_answer)
            prefix_normalized = normalize_text(prefix_answer)

            no_signature = final_signature(no_answer)
            prefix_signature = final_signature(prefix_answer)

            similarity = difflib.SequenceMatcher(
                None,
                no_normalized,
                prefix_normalized,
            ).ratio()

            result = {
                "length": key[0],
                "prompt_id": key[1],
                "repeat": key[2],
                "task": no_row["task"],
                "raw_exact_match": int(no_answer == prefix_answer),
                "normalized_exact_match": int(
                    no_normalized == prefix_normalized
                ),
                "final_signature_match": int(
                    no_signature == prefix_signature
                ),
                "text_similarity": similarity,
                "nocache_final_signature": no_signature,
                "prefix_final_signature": prefix_signature,
                "nocache_answer": no_answer.replace("\n", "\\n"),
                "prefix_answer": prefix_answer.replace("\n", "\\n"),
            }

            writer.writerow(result)
            pair_rows.append(result)

    summary_fields = [
        "length",
        "num_pairs",
        "raw_exact_match_rate",
        "normalized_exact_match_rate",
        "final_signature_match_rate",
        "mean_text_similarity",
        "nocache_deterministic_prompts",
        "prefix_deterministic_prompts",
        "num_prompts",
        "prompt_consensus_normalized_match_rate",
        "prompt_consensus_signature_match_rate",
    ]

    summary_rows = []

    for length in sorted(lengths):
        current_pairs = [
            row for row in pair_rows
            if int(row["length"]) == length
        ]

        no_by_prompt = defaultdict(list)
        prefix_by_prompt = defaultdict(list)

        for key, row in nocache_index.items():
            if key[0] == length:
                no_by_prompt[key[1]].append(
                    normalize_text(row["answer"])
                )

        for key, row in prefix_index.items():
            if key[0] == length:
                prefix_by_prompt[key[1]].append(
                    normalize_text(row["answer"])
                )

        prompt_ids = sorted(set(no_by_prompt) & set(prefix_by_prompt))

        no_deterministic = sum(
            len(set(no_by_prompt[prompt_id])) == 1
            for prompt_id in prompt_ids
        )

        prefix_deterministic = sum(
            len(set(prefix_by_prompt[prompt_id])) == 1
            for prompt_id in prompt_ids
        )

        consensus_normalized_matches = 0
        consensus_signature_matches = 0

        for prompt_id in prompt_ids:
            no_consensus = most_common(no_by_prompt[prompt_id])
            prefix_consensus = most_common(prefix_by_prompt[prompt_id])

            consensus_normalized_matches += int(
                no_consensus == prefix_consensus
            )

            consensus_signature_matches += int(
                final_signature(no_consensus)
                == final_signature(prefix_consensus)
            )

        summary = {
            "length": length,
            "num_pairs": len(current_pairs),
            "raw_exact_match_rate": statistics.mean(
                int(row["raw_exact_match"])
                for row in current_pairs
            ),
            "normalized_exact_match_rate": statistics.mean(
                int(row["normalized_exact_match"])
                for row in current_pairs
            ),
            "final_signature_match_rate": statistics.mean(
                int(row["final_signature_match"])
                for row in current_pairs
            ),
            "mean_text_similarity": statistics.mean(
                float(row["text_similarity"])
                for row in current_pairs
            ),
            "nocache_deterministic_prompts": no_deterministic,
            "prefix_deterministic_prompts": prefix_deterministic,
            "num_prompts": len(prompt_ids),
            "prompt_consensus_normalized_match_rate": (
                consensus_normalized_matches / len(prompt_ids)
            ),
            "prompt_consensus_signature_match_rate": (
                consensus_signature_matches / len(prompt_ids)
            ),
        }

        summary_rows.append(summary)

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    with mismatch_path.open("w", encoding="utf-8") as file:
        for row in pair_rows:
            if int(row["final_signature_match"]) == 1:
                continue

            file.write("=" * 80 + "\n")
            file.write(
                f"Length={row['length']} "
                f"Prompt={row['prompt_id']} "
                f"Repeat={row['repeat']} "
                f"Task={row['task']}\n"
            )
            file.write(
                f"No-cache signature: "
                f"{row['nocache_final_signature']}\n"
            )
            file.write(
                f"Prefix signature: "
                f"{row['prefix_final_signature']}\n"
            )
            file.write("\n[No-cache]\n")
            file.write(restore_text(row["nocache_answer"]) + "\n")
            file.write("\n[Prefix Cache]\n")
            file.write(restore_text(row["prefix_answer"]) + "\n\n")

    print("\n===== Quality consistency summary =====")

    for row in summary_rows:
        print(
            f"L={row['length']}: "
            f"normalized exact="
            f"{row['normalized_exact_match_rate']:.1%}, "
            f"final signature="
            f"{row['final_signature_match_rate']:.1%}, "
            f"mean similarity="
            f"{row['mean_text_similarity']:.3f}, "
            f"no-cache deterministic="
            f"{row['nocache_deterministic_prompts']}/"
            f"{row['num_prompts']}, "
            f"prefix deterministic="
            f"{row['prefix_deterministic_prompts']}/"
            f"{row['num_prompts']}"
        )

    print("\nPair CSV:", pair_path)
    print("Summary CSV:", summary_path)
    print("Mismatch report:", mismatch_path)


if __name__ == "__main__":
    main()
