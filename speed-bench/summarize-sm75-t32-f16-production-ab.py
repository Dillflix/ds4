#!/usr/bin/env python3
"""Summarize the dual-model production T32 FP16-output prefill A/B."""

from __future__ import annotations

import array
import csv
import json
import math
import statistics
import sys
from pathlib import Path


LAYOUTS = ("mixed15", "all43")
VARIANTS = ("control", "fused")
EXPECTED_CONTEXTS = (512, 4096, 32768)


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def load_prefill(path: Path) -> dict[int, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        result = {
            int(row["ctx_tokens"]): float(row["prefill_tps"])
            for row in rows
        }
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"invalid benchmark CSV {path}: {exc}")
    if tuple(sorted(result)) != EXPECTED_CONTEXTS:
        fail(f"unexpected contexts in {path}: {sorted(result)}")
    if any(not math.isfinite(value) or value <= 0 for value in result.values()):
        fail(f"invalid prefill throughput in {path}")
    return result


def load_logits(directory: Path, context: int) -> tuple[bytes, list[float]]:
    stem = f"frontier_{context:06d}.logits"
    raw_path = directory / f"{stem}.f32"
    json_path = directory / f"{stem}.json"
    if not raw_path.is_file() or not json_path.is_file():
        fail(f"missing frontier {context} in {directory}")
    raw = raw_path.read_bytes()
    if not raw or len(raw) % 4:
        fail(f"invalid raw logits file: {raw_path}")
    values = array.array("f")
    values.frombytes(raw)
    decoded = [float(value) for value in values]
    if not all(math.isfinite(value) for value in decoded):
        fail(f"non-finite raw logits: {raw_path}")
    payload = json.loads(json_path.read_text())
    if payload.get("frontier_tokens") != context:
        fail(f"frontier metadata mismatch: {json_path}")
    if payload.get("vocab") != len(decoded):
        fail(f"vocabulary size mismatch: {json_path}")
    return raw, decoded


def top_indices(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:count]


def compare_logits(reference: list[float], candidate: list[float]) -> dict[str, object]:
    if len(reference) != len(candidate):
        fail("logit vector length mismatch")
    deltas = [cand - ref for ref, cand in zip(reference, candidate)]
    ref_rms = math.sqrt(statistics.fmean(value * value for value in reference))
    error_rms = math.sqrt(statistics.fmean(value * value for value in deltas))
    ref_top = top_indices(reference, 10)
    candidate_top = top_indices(candidate, 10)
    return {
        "bit_exact": int(reference == candidate),
        "nrmse": error_rms / ref_rms if ref_rms else math.inf,
        "max_abs": max((abs(value) for value in deltas), default=0.0),
        "mean_abs": statistics.fmean(abs(value) for value in deltas),
        "top1_equal": int(ref_top[0] == candidate_top[0]),
        "top10_overlap": len(set(ref_top) & set(candidate_top)),
    }


def main() -> None:
    if len(sys.argv) != 3:
        fail("usage: summarize-sm75-t32-f16-production-ab.py RUNS.tsv OUTPUT_DIR")
    runs_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    with runs_path.open(newline="") as handle:
        runs = list(csv.DictReader(handle, delimiter="\t"))
    if not runs:
        fail("run manifest is empty")

    throughput: dict[tuple[str, int, str], dict[int, float]] = {}
    logits: dict[tuple[str, int, str, int], tuple[bytes, list[float]]] = {}
    repeats_by_layout: dict[str, set[int]] = {layout: set() for layout in LAYOUTS}
    for row in runs:
        try:
            layout = row["model_layout"]
            repeat = int(row["repeat"])
            variant = row["variant"]
            csv_path = Path(row["csv"])
            logits_dir = Path(row["logits"])
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"invalid run manifest row: {exc}")
        if layout not in LAYOUTS or variant not in VARIANTS or repeat <= 0:
            fail(f"invalid run identity: {layout} repeat={repeat} variant={variant}")
        key = (layout, repeat, variant)
        if key in throughput:
            fail(f"duplicate run: {key}")
        throughput[key] = load_prefill(csv_path)
        repeats_by_layout[layout].add(repeat)
        for context in EXPECTED_CONTEXTS:
            logits[(layout, repeat, variant, context)] = load_logits(logits_dir, context)

    for layout in LAYOUTS:
        repeats = sorted(repeats_by_layout[layout])
        if not repeats or repeats != list(range(1, repeats[-1] + 1)):
            fail(f"{layout} repeats are incomplete: {repeats}")
        for repeat in repeats:
            for variant in VARIANTS:
                if (layout, repeat, variant) not in throughput:
                    fail(f"missing {layout} repeat={repeat} variant={variant}")

    paired_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    determinism_rows: list[dict[str, object]] = []

    for layout in LAYOUTS:
        repeats = sorted(repeats_by_layout[layout])
        for repeat in repeats:
            for context in EXPECTED_CONTEXTS:
                control = throughput[(layout, repeat, "control")][context]
                fused = throughput[(layout, repeat, "fused")][context]
                paired_rows.append({
                    "model_layout": layout,
                    "repeat": repeat,
                    "ctx_tokens": context,
                    "control_tps": control,
                    "fused_tps": fused,
                    "paired_speedup": fused / control,
                })
                control_raw, control_values = logits[(layout, repeat, "control", context)]
                fused_raw, fused_values = logits[(layout, repeat, "fused", context)]
                metrics = compare_logits(control_values, fused_values)
                metrics["bit_exact"] = int(control_raw == fused_raw)
                comparison_rows.append({
                    "model_layout": layout,
                    "repeat": repeat,
                    "ctx_tokens": context,
                    **metrics,
                })

        for context in EXPECTED_CONTEXTS:
            context_rows = [
                row for row in paired_rows
                if row["model_layout"] == layout and row["ctx_tokens"] == context
            ]
            ratios = [float(row["paired_speedup"]) for row in context_rows]
            summary_rows.append({
                "model_layout": layout,
                "ctx_tokens": context,
                "control_median_tps": statistics.median(
                    float(row["control_tps"]) for row in context_rows
                ),
                "fused_median_tps": statistics.median(
                    float(row["fused_tps"]) for row in context_rows
                ),
                "paired_median_speedup": statistics.median(ratios),
                "paired_speedup_sd": statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
            })

        reference_repeat = repeats[0]
        for repeat in repeats[1:]:
            for variant in VARIANTS:
                for context in EXPECTED_CONTEXTS:
                    reference_raw = logits[(layout, reference_repeat, variant, context)][0]
                    current_raw = logits[(layout, repeat, variant, context)][0]
                    determinism_rows.append({
                        "model_layout": layout,
                        "variant": variant,
                        "reference_repeat": reference_repeat,
                        "repeat": repeat,
                        "ctx_tokens": context,
                        "bit_exact": int(reference_raw == current_raw),
                    })

    def write_csv(name: str, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        with (output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("paired-samples.csv", paired_rows)
    write_csv("summary.csv", summary_rows)
    write_csv("logit-comparison.csv", comparison_rows)
    write_csv("repeat-determinism.csv", determinism_rows)

    semantic_gate = all(int(row["top1_equal"]) == 1 for row in comparison_rows)
    determinism_gate = all(int(row["bit_exact"]) == 1 for row in determinism_rows)
    long_context = {
        row["model_layout"]: float(row["paired_median_speedup"])
        for row in summary_rows if row["ctx_tokens"] == 32768
    }
    performance_gate = (
        all(value > 1.0 for value in long_context.values())
        and all(float(row["paired_median_speedup"]) >= 0.995 for row in summary_rows)
    )
    screen_pass = semantic_gate and determinism_gate and performance_gate
    max_nrmse = max(float(row["nrmse"]) for row in comparison_rows)
    max_abs = max(float(row["max_abs"]) for row in comparison_rows)
    min_top10 = min(int(row["top10_overlap"]) for row in comparison_rows)

    gates = {
        "semantic_top1_gate": semantic_gate,
        "repeat_determinism_gate": determinism_gate,
        "performance_gate": performance_gate,
        "production_ab_screen_pass": screen_pass,
        "full_quality_gate_required_before_default": True,
    }
    (output_dir / "gates.json").write_text(json.dumps(gates, indent=2) + "\n")

    lines = [
        "# SM75 production T32 FP16-output prefill A/B",
        "",
        "The control and candidate use the same stage-aware 344/344 dense-Q8 plan; only the T32 projection-output boundary changes.",
        "",
        "| Model | Context | Control tok/s | Fused tok/s | Paired speedup | Change | SD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        speedup = float(row["paired_median_speedup"])
        lines.append(
            f"| {row['model_layout']} | {row['ctx_tokens']} | "
            f"{float(row['control_median_tps']):.3f} | "
            f"{float(row['fused_median_tps']):.3f} | {speedup:.6f}x | "
            f"{(speedup - 1.0) * 100.0:+.3f}% | "
            f"{float(row['paired_speedup_sd']):.6f} |"
        )
    lines.extend([
        "",
        "## Numerical screen",
        "",
        f"- Frontier top-1 agreement: {sum(int(row['top1_equal']) for row in comparison_rows)}/{len(comparison_rows)}",
        f"- Maximum NRMSE: {max_nrmse:.9g}",
        f"- Maximum absolute delta: {max_abs:.9g}",
        f"- Minimum top-10 overlap: {min_top10}/10",
        f"- Within-arm repeat determinism: {'pass' if determinism_gate else 'fail'}",
        f"- Performance gate (both 32K medians >1.0x; every frontier >=0.995x): {'pass' if performance_gate else 'fail'}",
        f"- Production A/B screen: {'pass' if screen_pass else 'fail'}",
        "",
        "Passing this screen does not by itself enable the candidate: because FP16 rounding is intentional, the existing multi-prompt quality gate must pass for both models before the default changes.",
    ])
    summary_text = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(summary_text)
    print(summary_text, end="")


if __name__ == "__main__":
    main()
