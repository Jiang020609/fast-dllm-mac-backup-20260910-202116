from __future__ import annotations

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: summarize_rotated_k_ab.py LOGROOT OUTROOT"
    )

logroot = Path(sys.argv[1])
outroot = Path(sys.argv[2])


def last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text)
    return float(values[-1]) if values else None


def last_int(pattern: str, text: str) -> int | None:
    values = re.findall(pattern, text)
    return int(values[-1]) if values else None


def parse_metric(
    text: str,
    marker: str,
) -> float | None:
    for line in reversed(text.splitlines()):
        if marker not in line or "exact_match" not in line:
            continue

        fields = [field.strip() for field in line.split("|")]

        try:
            metric_index = fields.index("exact_match")
        except ValueError:
            continue

        for field in fields[metric_index + 1:]:
            cleaned = (
                field.replace("↑", "")
                .replace("↓", "")
                .strip()
            )

            try:
                return float(cleaned)
            except ValueError:
                continue

    return None


groups: dict[str, list[dict]] = defaultdict(list)

for path in sorted(logroot.glob("*.log")):
    match = re.fullmatch(
        r"(baseline|rotated_k)_r(\d+)\.log",
        path.name,
    )

    if not match:
        continue

    mode = match.group(1)
    repeat = int(match.group(2))

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    latency = last_float(
        r"Total time taken:\s*([0-9.]+)\s*seconds",
        text,
    )

    if latency is None:
        print(f"Skipping incomplete log: {path}")
        continue

    groups[mode].append({
        "repeat": repeat,
        "latency": latency,
        "nfe": last_int(
            r"Total NFE is\s*(\d+)",
            text,
        ),
        "tokens_per_second": last_float(
            r"Tokens per second:\s*([0-9.]+)",
            text,
        ),
        "flexible_em": parse_metric(
            text,
            "flexible-extract",
        ),
        "strict_em": parse_metric(
            text,
            "strict-match",
        ),
    })


rows = []

for mode in ("baseline", "rotated_k"):
    items = groups.get(mode, [])

    if not items:
        raise RuntimeError(
            f"No successful runs found for mode={mode}"
        )

    latencies = [
        item["latency"]
        for item in items
    ]

    nfes = [
        item["nfe"]
        for item in items
        if item["nfe"] is not None
    ]

    throughputs = [
        item["tokens_per_second"]
        for item in items
        if item["tokens_per_second"] is not None
    ]

    flexible_values = [
        item["flexible_em"]
        for item in items
        if item["flexible_em"] is not None
    ]

    strict_values = [
        item["strict_em"]
        for item in items
        if item["strict_em"] is not None
    ]

    rows.append({
        "mode": mode,
        "runs": len(items),
        "mean_latency_s": statistics.mean(latencies),
        "median_latency_s": statistics.median(latencies),
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
        "mean_nfe": (
            statistics.mean(nfes)
            if nfes
            else None
        ),
        "mean_tokens_per_second": (
            statistics.mean(throughputs)
            if throughputs
            else None
        ),
        "flexible_em_min": (
            min(flexible_values)
            if flexible_values
            else None
        ),
        "strict_em_min": (
            min(strict_values)
            if strict_values
            else None
        ),
    })


baseline_latency = next(
    row["mean_latency_s"]
    for row in rows
    if row["mode"] == "baseline"
)

for row in rows:
    row["speedup_vs_baseline"] = (
        baseline_latency / row["mean_latency_s"]
    )


outroot.mkdir(parents=True, exist_ok=True)

summary_path = (
    outroot / "rotated_k_ab_summary.csv"
)

fieldnames = list(rows[0].keys())

with summary_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(rows)


print()
print("=" * 112)
print("Rotated-K Cache A/B")
print("=" * 112)

print(
    f"{'Mode':<14}"
    f"{'Runs':>7}"
    f"{'Mean(s)':>12}"
    f"{'Median(s)':>12}"
    f"{'Min(s)':>12}"
    f"{'Max(s)':>12}"
    f"{'NFE':>10}"
    f"{'Tok/s':>12}"
    f"{'Speedup':>12}"
    f"{'Flex':>8}"
    f"{'Strict':>8}"
)

print("-" * 112)

for row in rows:
    mean_nfe = row["mean_nfe"]
    throughput = row["mean_tokens_per_second"]
    flex = row["flexible_em_min"]
    strict = row["strict_em_min"]

    print(
        f"{row['mode']:<14}"
        f"{row['runs']:>7}"
        f"{row['mean_latency_s']:>12.3f}"
        f"{row['median_latency_s']:>12.3f}"
        f"{row['min_latency_s']:>12.3f}"
        f"{row['max_latency_s']:>12.3f}"
        f"{mean_nfe if mean_nfe is not None else 0:>10.2f}"
        f"{throughput if throughput is not None else 0:>12.3f}"
        f"{row['speedup_vs_baseline']:>11.3f}x"
        f"{flex if flex is not None else -1:>8.1f}"
        f"{strict if strict is not None else -1:>8.1f}"
    )

print()
print("Per-run results:")

for mode in ("baseline", "rotated_k"):
    for item in sorted(
        groups[mode],
        key=lambda value: value["repeat"],
    ):
        print(
            f"  {mode:<10} "
            f"r{item['repeat']}: "
            f"latency={item['latency']:.3f}s, "
            f"NFE={item['nfe']}, "
            f"tokens/s={item['tokens_per_second']:.3f}, "
            f"flex={item['flexible_em']}, "
            f"strict={item['strict_em']}"
        )

print()
print("Summary:")
print(summary_path)
