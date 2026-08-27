#!/usr/bin/env python3
"""Summarize Q4-32/Q3A4 production kernel shares and per-device stages."""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


Q3A4_DEFAULT = {6, 8, 10, 12, 14, 16, 18, 20, 30, 32, 34, 36, 38, 40, 42}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def layer_of(row: dict[str, str]) -> int | None:
    for field in ("layer_range", "stage_range"):
        match = re.search(r"(?:^|/)layer=(\d+)(?:/|$)", row.get(field, ""))
        if match:
            return int(match.group(1))
    return None


def family_of(row: dict[str, str]) -> str:
    name = row["name"]
    layer = layer_of(row)
    if "moe_gate_up_mid_sm75_q32_tile8_kernel" in name:
        return "q3a4_gate_up" if layer in Q3A4_DEFAULT else "q4_32_gate_up"
    if "moe_down_sm75_q4_32_tile_kernel" in name:
        return "q4_32_down"
    if "attention_indexed" in name:
        return "attention_indexed"
    if "attention_decode_mixed" in name:
        return "attention_mixed"
    if "indexer_scores" in name:
        return "indexer_score"
    if "indexer_topk" in name:
        return "indexer_select"
    if "attention" in name:
        return "attention_other"
    if "matmul_q8" in name or "gemm" in name.lower():
        return "dense_projection"
    if "moe_" in name or "routed" in name:
        return "routed_support"
    return "other"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    operations = read_csv(output / "operation-attribution.csv")
    kernels = [row for row in operations if row["kind"] == "kernel"]
    if not kernels:
        raise SystemExit("operation-attribution.csv contains no kernels")

    family_ms: dict[tuple[str, int], float] = defaultdict(float)
    total_ms = 0.0
    for row in kernels:
        duration_ms = float(row["duration_ns"]) / 1.0e6
        family_ms[(family_of(row), int(row["device"]))] += duration_ms
        total_ms += duration_ms
    rows = []
    for (family, device), duration_ms in sorted(family_ms.items()):
        rows.append(
            {
                "family": family,
                "device": device,
                "kernel_ms": f"{duration_ms:.6f}",
                "share_of_annotated_kernel_pct": f"{100.0*duration_ms/total_ms:.6f}",
            }
        )
    with (output / "kernel-family-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family", "device", "kernel_ms", "share_of_annotated_kernel_pct"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    stages = read_csv(output / "stage-device-summary.csv")
    traces = read_csv(output / "trace-summary.csv")
    with (output / "profile-summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# SM75 Q4-32/Q3A4 32K production profile\n\n")
        manifest = (output / "manifest.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        rowsplit = "attention_rowsplit=1" in manifest
        handle.write(
            f"Attention/indexer placement: **{'indexed-chain query-row split' if rowsplit else 'home-only'}**.\n\n"
        )
        handle.write(
            "The Nsight Systems run is one genuine 32K production frontier. "
            "Kernel percentages below are sums of attributed GPU kernel time; "
            "concurrent devices therefore make them work shares, not elapsed-time "
            "Amdahl shares.\n\n"
        )
        handle.write("## Trace\n\n")
        for row in traces:
            handle.write(
                f"- `{row['trace']}` devices `{row['devices']}`: "
                f"pipeline span {float(row['pipeline_gpu_span_ms']):.3f} ms, "
                f"annotated kernels {float(row['annotated_kernel_ms']):.3f} ms.\n"
            )
        handle.write("\n## Per-device stage timing\n\n")
        handle.write("| Stage | Role | Device | Layers | Kernel ms | ms/layer | Memcpy ms |\n")
        handle.write("|---:|---|---:|---|---:|---:|---:|\n")
        for row in stages:
            handle.write(
                f"| {row['stage']} | {row['role']} | {row['device']} | "
                f"{row['layers']} | {float(row['kernel_ms']):.3f} | "
                f"{float(row['kernel_ms_per_layer']):.3f} | "
                f"{float(row['memcpy_ms']):.3f} |\n"
            )
        handle.write("\n## Attributed kernel work\n\n")
        handle.write("| Family | Device | Kernel ms | Share |\n")
        handle.write("|---|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['family']} | {row['device']} | "
                f"{float(row['kernel_ms']):.3f} | "
                f"{float(row['share_of_annotated_kernel_pct']):.2f}% |\n"
            )
        handle.write(
            "\nTargeted Nsight Compute reports are under `ncu/`; the routed "
            "captures use bounded exact production-dispatch harnesses, while the "
            "attention/indexer captures reproduce the 32K indexed, raw-window "
            "mixed, WMMA score, and 8192-wide top-k shapes.\n"
        )
    print((output / "profile-summary.md").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
