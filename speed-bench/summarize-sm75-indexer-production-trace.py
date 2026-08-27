#!/usr/bin/env python3
"""Compare paired unsynchronized WMMA128/WMMA64 production traces."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


VARIANTS = {"wmma128": "indexer_scores_wmma128_kernel",
            "wmma64": "indexer_scores_wmma64_kernel"}


def die(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        die(f"missing or empty input: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in text.split("/"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def summarize_variant(root: Path, variant: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    variant_dir = root / variant
    operations = read_csv(variant_dir / "operation-attribution.csv")
    trace_rows = read_csv(variant_dir / "trace-summary.csv")
    benchmark = read_csv(variant_dir / "nsys" / "combined-benchmark.csv")
    if len(trace_rows) != 1:
        die(f"{variant} must contain exactly one trace-summary row")
    if len(benchmark) != 1 or int(benchmark[0]["ctx_tokens"]) != 32768:
        die(f"{variant} is not one genuine 32768-token frontier")

    expected = VARIANTS[variant]
    score_ops: list[dict[str, str]] = []
    wrong_score_names: set[str] = set()
    for row in operations:
        if row["kind"] != "kernel" or "indexer_scores_wmma" not in row["name"]:
            continue
        if expected not in row["name"]:
            wrong_score_names.add(row["name"])
        else:
            score_ops.append(row)
    if wrong_score_names:
        die(f"{variant} contains another score kernel: {sorted(wrong_score_names)[0]}")
    if not score_ops:
        die(f"{variant} contains no attributed {expected} launches")

    by_cell: dict[tuple[int, int, int], dict[str, int]] = defaultdict(
        lambda: {"duration_ns": 0, "launches": 0}
    )
    for row in score_ops:
        stage = parse_fields(row["stage_range"])
        if "stage" not in stage or "tier" not in stage:
            die(f"{variant} score launch is outside an attributed stage")
        key = (int(stage["stage"]), int(stage["tier"]), int(row["device"]))
        by_cell[key]["duration_ns"] += int(row["duration_ns"])
        by_cell[key]["launches"] += 1

    score_ns = sum(int(row["duration_ns"]) for row in score_ops)
    annotated_ns = round(float(trace_rows[0]["annotated_kernel_ms"]) * 1e6)
    pipeline_ms = float(trace_rows[0]["pipeline_gpu_span_ms"])
    cells = [
        {
            "variant": variant,
            "stage": stage,
            "tier": tier,
            "device": device,
            "score_ms": values["duration_ns"] / 1e6,
            "launches": values["launches"],
            "mean_us": values["duration_ns"] / values["launches"] / 1e3,
        }
        for (stage, tier, device), values in sorted(by_cell.items())
    ]
    result: dict[str, object] = {
        "variant": variant,
        "prefill_tokens": int(benchmark[0]["prefill_tokens"]),
        "prefill_tps": float(benchmark[0]["prefill_tps"]),
        "pipeline_gpu_span_ms": pipeline_ms,
        "annotated_kernel_ms": annotated_ns / 1e6,
        "score_ms": score_ns / 1e6,
        "score_share_pct": 100.0 * score_ns / annotated_ns if annotated_ns else 0.0,
        "score_launches": len(score_ops),
        "score_mean_us": score_ns / len(score_ops) / 1e3,
    }
    return result, cells


def main() -> int:
    if len(sys.argv) != 2:
        die("usage: summarize-sm75-indexer-production-trace.py OUTPUT_DIR")
    root = Path(sys.argv[1]).resolve()
    summaries: dict[str, dict[str, object]] = {}
    cells: list[dict[str, object]] = []
    for variant in VARIANTS:
        summary, variant_cells = summarize_variant(root, variant)
        summaries[variant] = summary
        cells.extend(variant_cells)

    if summaries["wmma128"]["prefill_tokens"] != summaries["wmma64"]["prefill_tokens"]:
        die("the paired traces contain different prefill work")
    if summaries["wmma128"]["score_launches"] != summaries["wmma64"]["score_launches"]:
        die("the paired traces contain different score-launch counts")

    write_csv(
        root / "trace-comparison.csv",
        [
            "variant", "prefill_tokens", "prefill_tps", "pipeline_gpu_span_ms",
            "annotated_kernel_ms", "score_ms", "score_share_pct",
            "score_launches", "score_mean_us",
        ],
        list(summaries.values()),
    )
    write_csv(
        root / "score-device-stage.csv",
        ["variant", "stage", "tier", "device", "score_ms", "launches", "mean_us"],
        cells,
    )

    base = summaries["wmma128"]
    candidate = summaries["wmma64"]
    ratios = {
        "score_kernel_speedup": float(base["score_ms"]) / float(candidate["score_ms"]),
        "pipeline_span_speedup": (
            float(base["pipeline_gpu_span_ms"]) /
            float(candidate["pipeline_gpu_span_ms"])
        ),
        "trace_prefill_tps_speedup": (
            float(candidate["prefill_tps"]) / float(base["prefill_tps"])
        ),
        "score_ms_saved": float(base["score_ms"]) - float(candidate["score_ms"]),
        "pipeline_ms_saved": (
            float(base["pipeline_gpu_span_ms"]) -
            float(candidate["pipeline_gpu_span_ms"])
        ),
    }
    (root / "comparison.json").write_text(
        json.dumps({"variants": summaries, "comparison": ratios}, indent=2) + "\n",
        encoding="utf-8",
    )

    score_transfer = ratios["score_kernel_speedup"] > 1.0
    pipeline_transfer = ratios["pipeline_span_speedup"] > 1.0
    if not score_transfer:
        interpretation = (
            "WMMA64 did not reduce score-kernel time across the captured "
            "production launch population."
        )
    elif not pipeline_transfer:
        interpretation = (
            "WMMA64 reduced production score-kernel work, but that work was "
            "not reflected in a shorter captured pipeline span."
        )
    else:
        interpretation = (
            "WMMA64 reduced production score-kernel work and shortened the "
            "captured pipeline span."
        )

    with (root / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# SM75 indexer production trace comparison\n\n")
        handle.write(
            "Both traces use a genuine 32K frontier, a 256K allocation, "
            "balanced T256 placement, mirrored-KV attention row split, and "
            "no cross-device synchronization.\n\n"
        )
        handle.write(
            "| Variant | Score GPU time | Score launches | Mean score launch | "
            "Pipeline GPU span | Trace prefill |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for variant in VARIANTS:
            row = summaries[variant]
            handle.write(
                f"| {variant} | {float(row['score_ms']):.3f} ms | "
                f"{int(row['score_launches'])} | "
                f"{float(row['score_mean_us']):.3f} us | "
                f"{float(row['pipeline_gpu_span_ms']):.3f} ms | "
                f"{float(row['prefill_tps']):.2f} tokens/s |\n"
            )
        handle.write(
            f"\nScore-kernel speedup: **{ratios['score_kernel_speedup']:.3f}x**. "
            f"Pipeline-span speedup: **{ratios['pipeline_span_speedup']:.3f}x**. "
            f"Trace-throughput speedup: **{ratios['trace_prefill_tps_speedup']:.3f}x**.\n\n"
        )
        handle.write(f"Interpretation: {interpretation}\n")
        handle.write(
            "This is a mechanism diagnostic, not a promotion gate; use the "
            "interleaved uninstrumented A/B for end-to-end significance.\n"
        )
    print((root / "summary.md").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
