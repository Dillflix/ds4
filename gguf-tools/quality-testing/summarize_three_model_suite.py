#!/usr/bin/env python3
"""Summarize the fixed Q2 / IQ2-IQ2-Q4 / Q4 quality-performance suite."""

from __future__ import annotations

import csv
import html
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LABELS = ("q2", "hybrid", "q4")
DISPLAY = {"q2": "Q2", "hybrid": "IQ2 / IQ2 / Q4", "q4": "Q4"}
COLORS = {"q2": "#2563eb", "hybrid": "#d97706", "q4": "#059669"}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp, delimiter="\t"))


def quality_summary(path: Path) -> dict[str, float]:
    rows = read_tsv(path)
    if not rows:
        raise ValueError(f"no quality rows in {path}")
    tokens = sum(int(row["target_tokens"]) for row in rows)

    def isum(key: str) -> int:
        return sum(int(float(row.get(key, 0) or 0)) for row in rows)

    def weighted(key: str, count: str) -> float:
        n = isum(count)
        if not n:
            return math.nan
        return sum(float(row[key]) * int(float(row[count])) for row in rows) / n

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else math.nan

    return {
        "cases": len(rows),
        "tokens": tokens,
        "avg_nll": sum(float(row["nll"]) for row in rows) / tokens,
        "first_match_rate": sum(int(row["first_match"]) for row in rows) / len(rows),
        "avg_lcp": sum(int(row["greedy_lcp"]) for row in rows) / len(rows),
        "api_target_mae": weighted("api_target_mae", "api_target_tokens"),
        "api_top1_rate": ratio(isum("api_top1_match"), isum("api_top1_count")),
        "api_topn_recall": ratio(isum("api_topn_hit"), isum("api_topn_ref")),
        "api_pair_rate": ratio(isum("api_pair_agree"), isum("api_pair_total")),
    }


def performance_summary(paths: list[Path]) -> dict[int, dict[str, float]]:
    by_context: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                ctx = int(row["ctx_tokens"])
                for key in ("prefill_tps", "gen_tps", "gen_first_ms", "gen_steady_tps"):
                    by_context[ctx][key].append(float(row[key]))
    return {
        ctx: {
            key: statistics.median(values)
            for key, values in metrics.items()
        }
        for ctx, metrics in sorted(by_context.items())
    }


def gpu_peaks(paths: list[Path]) -> dict[int, float]:
    peaks: dict[int, float] = defaultdict(float)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                index = int(row["index"].strip())
                used = float(row["memory_used_mib"].strip())
                peaks[index] = max(peaks[index], used)
    return dict(sorted(peaks.items()))


def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def write_svg(path: Path, perf: dict[str, dict[int, dict[str, float]]]) -> None:
    width, height = 1000, 620
    left, right, top = 78, 30, 55
    panel_height, gap = 205, 80
    plot_width = width - left - right
    contexts = sorted({ctx for model in perf.values() for ctx in model})
    if not contexts:
        return
    x_min, x_max = math.log2(min(contexts)), math.log2(max(contexts))

    panels = (
        ("prefill_tps", "Prompt processing (tokens/s)"),
        ("gen_steady_tps", "Steady generation (tokens/s)"),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#1f2937}.title{font-size:22px;font-weight:700}.axis{font-size:12px}.legend{font-size:13px;font-weight:600}</style>',
        '<text class="title" x="500" y="30" text-anchor="middle">DeepSeek V4 Flash: fixed performance comparison</text>',
    ]
    for panel_index, (metric, title) in enumerate(panels):
        y_top = top + panel_index * (panel_height + gap)
        values = [entry[metric] for model in perf.values() for entry in model.values()]
        y_max = max(values) * 1.08 if values else 1.0
        parts.append(f'<text x="{left}" y="{y_top - 12}" font-size="15" font-weight="700">{html.escape(title)}</text>')
        for tick_index in range(6):
            tick = y_max * tick_index / 5
            y = y_top + panel_height - panel_height * tick_index / 5
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            parts.append(f'<text class="axis" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{tick:.1f}</text>')
        for ctx in contexts:
            x = left + (math.log2(ctx) - x_min) / (x_max - x_min or 1) * plot_width
            parts.append(f'<line x1="{x:.1f}" y1="{y_top}" x2="{x:.1f}" y2="{y_top+panel_height}" stroke="#f3f4f6"/>')
            parts.append(f'<text class="axis" x="{x:.1f}" y="{y_top+panel_height+20}" text-anchor="middle">{ctx//1024}K</text>')
        for label in LABELS:
            points = []
            for ctx, entry in perf[label].items():
                x = left + (math.log2(ctx) - x_min) / (x_max - x_min or 1) * plot_width
                y = y_top + panel_height - entry[metric] / y_max * panel_height
                points.append((x, y))
            polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            color = COLORS[label]
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
    legend_y = height - 28
    legend_x = 250
    for index, label in enumerate(LABELS):
        x = legend_x + index * 230
        parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x+30}" y2="{legend_y}" stroke="{COLORS[label]}" stroke-width="4"/>')
        parts.append(f'<text class="legend" x="{x+38}" y="{legend_y+5}">{html.escape(DISPLAY[label])}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RESULTS_DIR", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    models = {row["label"]: row for row in read_tsv(root / "models.tsv")}
    quality = {label: quality_summary(root / "quality" / f"{label}.tsv") for label in LABELS}
    perf = {
        label: performance_summary(sorted((root / "performance").glob(f"{label}.run*.csv")))
        for label in LABELS
    }
    peaks = {
        label: gpu_peaks(sorted((root / "gpu").glob(f"*{label}*.csv")))
        for label in LABELS
    }
    contexts = sorted(set.intersection(*(set(perf[label]) for label in LABELS)))

    fields = [
        "label", "display", "bytes", "gib", "quality_cases", "quality_tokens",
        "avg_nll", "first_match_rate", "avg_lcp", "api_target_mae",
        "api_top1_rate", "api_topn_recall", "api_pair_rate", "peak_vram_mib_by_gpu",
    ]
    for ctx in contexts:
        fields.extend((f"prefill_tps_{ctx}", f"steady_gen_tps_{ctx}", f"first_token_ms_{ctx}"))
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for label in LABELS:
            q = quality[label]
            row: dict[str, object] = {
                "label": label,
                "display": DISPLAY[label],
                "bytes": models[label]["bytes"],
                "gib": int(models[label]["bytes"]) / 2**30,
                "quality_cases": q["cases"],
                "quality_tokens": q["tokens"],
                "avg_nll": q["avg_nll"],
                "first_match_rate": q["first_match_rate"],
                "avg_lcp": q["avg_lcp"],
                "api_target_mae": q["api_target_mae"],
                "api_top1_rate": q["api_top1_rate"],
                "api_topn_recall": q["api_topn_recall"],
                "api_pair_rate": q["api_pair_rate"],
                "peak_vram_mib_by_gpu": ";".join(f"{gpu}:{value:.0f}" for gpu, value in peaks[label].items()),
            }
            for ctx in contexts:
                row[f"prefill_tps_{ctx}"] = perf[label][ctx]["prefill_tps"]
                row[f"steady_gen_tps_{ctx}"] = perf[label][ctx]["gen_steady_tps"]
                row[f"first_token_ms_{ctx}"] = perf[label][ctx]["gen_first_ms"]
            writer.writerow(row)

    lines = [
        "# Q2 vs IQ2/IQ2/Q4 vs Q4",
        "",
        "This is an end-to-end artifact comparison against the same 100 official Flash continuations. "
        "If the published Q2/Q4 use a checkpoint other than 0731, quality deltas combine checkpoint and quantization effects; "
        "they do not isolate quantization error.",
        "",
        "## Quality",
        "",
        "Lower NLL and API target MAE are better; higher match, recall, and ordering rates are better.",
        "",
        "| Model | Size GiB | Avg NLL | First match | Avg LCP | API target MAE | API top-1 | API top-N recall | API pair order |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        q = quality[label]
        lines.append(
            f"| {DISPLAY[label]} | {int(models[label]['bytes']) / 2**30:.2f} | {q['avg_nll']:.6f} | "
            f"{q['first_match_rate']:.1%} | {q['avg_lcp']:.2f} | {fmt(q['api_target_mae'], 6)} | "
            f"{q['api_top1_rate']:.1%} | {q['api_topn_recall']:.1%} | {q['api_pair_rate']:.1%} |"
        )
    lines.extend(("", "## Performance", "", "Values are medians across repetitions; steady generation excludes the first generated token.", ""))
    header = "| Context | " + " | ".join(DISPLAY[label] for label in LABELS) + " |"
    lines.extend((header, "|---:" + "|---:" * len(LABELS) + "|"))
    for ctx in contexts:
        cells = []
        for label in LABELS:
            p = perf[label][ctx]
            cells.append(f"{p['prefill_tps']:.1f} prefill / {p['gen_steady_tps']:.2f} gen tok/s")
        lines.append(f"| {ctx // 1024}K | " + " | ".join(cells) + " |")
    lines.extend(("", "## Peak VRAM", "", "Peak used MiB per physical GPU across quality and performance sampling.", ""))
    lines.extend(("| Model | Per-GPU peak MiB |", "|---|---|"))
    for label in LABELS:
        peak_text = ", ".join(f"GPU {gpu}: {value:.0f}" for gpu, value in peaks[label].items())
        lines.append(f"| {DISPLAY[label]} | {peak_text} |")
    lines.extend((
        "",
        "## Method",
        "",
        "- Quality: deterministic target-token NLL, exact CUDA math (`--quality` internally), 4096-token allocation unless overridden.",
        "- Performance: identical prompt, 2K-to-64K doubling context sweep unless overridden, 128 greedy tokens per frontier.",
        "- Models run one at a time with the same GPU order and automatic VRAM budgets.",
        "- Performance repetitions use a balanced rotating order to reduce temperature/order bias.",
        "- Raw score tables, benchmark CSVs, logs, GPU telemetry, and pairwise quality reports remain alongside this summary.",
        "",
    ))
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_svg(root / "performance.svg", perf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
