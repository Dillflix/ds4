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
    lower = name.lower()
    layer = layer_of(row)
    if row.get("partner_range", ""):
        if row["kind"] == "memcpy":
            return "partner_t256_memcpy"
        if "f32_to_f16" in lower:
            return "partner_t256_activation_convert"
        return "partner_t256_cublas"
    if row["kind"] == "memcpy" and row.get("attention_rows_range", ""):
        return "attention_row_split_memcpy"
    if row["kind"] == "memcpy" and row.get("indexer_rows_range", ""):
        return "indexer_row_split_memcpy"
    if row["kind"] == "memcpy" and row.get("handoff_range", ""):
        return "stage_handoff_memcpy"
    if row["kind"] == "memcpy":
        return "other_memcpy"
    if "moe_gate_up_mid_sm75_q32_tile8_kernel" in name:
        return "q3a4_gate_up" if layer in Q3A4_DEFAULT else "q4_32_gate_up"
    if (
        "moe_down_sm75_q4_32_tile_kernel" in name
        or "moe_down_sm75_q4_32_tile16_compact_kernel" in name
    ):
        return "q4_32_down"
    if "attention_indexed" in lower:
        return "attention_indexed"
    if "attention_decode_mixed" in lower:
        return "attention_mixed"
    if "attention" in lower:
        return "attention_other"
    if "indexer_" in lower and "topk" not in lower:
        return "indexer_score"
    if "indexed_topk" in lower or "indexer_topk" in lower:
        return "indexer_topk"
    if "matmul_q8" in lower:
        return "dense_q8_native"
    if any(
        marker in lower
        for marker in (
            "turing_s", "cutlass::kernel2", "cublaslt::splitkreduce",
            "gemvx::kernel", "gemm",
        )
    ):
        return "local_fp16_gemm"
    if "moe_" in lower or "routed" in lower:
        return "routed_support"
    if any(marker in lower for marker in ("rms_norm", "hc_", "dsv4_qkv_rms")):
        return "norm_hyperconnection"
    if "compressor_" in lower:
        return "compressor"
    if "rope_" in lower:
        return "rope"
    if any(
        marker in lower
        for marker in (
            "f32_to_f16", "q8_k_quantize", "q8_k_pack",
            "fp8_kv_quantize", "quantize_q8_0",
        )
    ):
        return "format_quant_pack"
    if any(
        marker in lower
        for marker in (
            "add_kernel", "swiglu_kernel", "router_select_",
            "store_raw_kv_", "fill_f32_kernel", "embed_tokens_",
            "output_hc_weights_",
        )
    ):
        return "elementwise_control"
    return "other"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} OUTPUT_DIR")
    output = Path(sys.argv[1]).resolve()
    operations = read_csv(output / "operation-attribution.csv")
    kernels = [row for row in operations if row["kind"] == "kernel"]
    if not kernels:
        raise SystemExit("operation-attribution.csv contains no kernels")

    operation_groups: dict[tuple[str, str, int], dict[str, int]] = defaultdict(
        lambda: {"duration_ns": 0, "bytes": 0}
    )
    kind_duration: dict[str, int] = defaultdict(int)
    for row in operations:
        key = (row["kind"], family_of(row), int(row["device"]))
        operation_groups[key]["duration_ns"] += int(row["duration_ns"])
        operation_groups[key]["bytes"] += int(row["bytes"])
        kind_duration[row["kind"]] += int(row["duration_ns"])
    operation_rows = []
    for (kind, family, device), values in sorted(operation_groups.items()):
        operation_rows.append(
            {
                "kind": kind,
                "family": family,
                "device": device,
                "duration_ms": f"{values['duration_ns']/1.0e6:.6f}",
                "bytes": values["bytes"],
                "share_of_kind_pct": (
                    f"{100.0*values['duration_ns']/kind_duration[kind]:.6f}"
                ),
            }
        )
    with (output / "operation-family-summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kind", "family", "device", "duration_ms", "bytes",
                "share_of_kind_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(operation_rows)

    family_ms: dict[tuple[str, int], float] = defaultdict(float)
    total_ms = 0.0
    for row in kernels:
        duration_ms = float(row["duration_ns"]) / 1.0e6
        family_ms[(family_of(row), int(row["device"]))] += duration_ms
        total_ms += duration_ms
    rows = []
    family_totals: dict[str, float] = defaultdict(float)
    for (family, device), duration_ms in sorted(family_ms.items()):
        family_totals[family] += duration_ms
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

    total_rows = [
        {
            "family": family,
            "kernel_ms": f"{duration_ms:.6f}",
            "share_of_annotated_kernel_pct": f"{100.0*duration_ms/total_ms:.6f}",
        }
        for family, duration_ms in sorted(
            family_totals.items(), key=lambda item: item[1], reverse=True
        )
    ]
    with (output / "kernel-family-total.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["family", "kernel_ms", "share_of_annotated_kernel_pct"],
        )
        writer.writeheader()
        writer.writerows(total_rows)

    stages = read_csv(output / "stage-device-summary.csv")
    traces = read_csv(output / "trace-summary.csv")
    with (output / "profile-summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# SM75 Q4-32/Q3A4 32K production profile\n\n")
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
        handle.write("\n## Aggregate attributed kernel work\n\n")
        handle.write("| Family | Kernel ms | Share |\n")
        handle.write("|---|---:|---:|\n")
        for row in total_rows:
            handle.write(
                f"| {row['family']} | {float(row['kernel_ms']):.3f} | "
                f"{float(row['share_of_annotated_kernel_pct']):.2f}% |\n"
            )
        handle.write("\n## Attributed kernel work by device\n\n")
        handle.write("| Family | Device | Kernel ms | Share |\n")
        handle.write("|---|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['family']} | {row['device']} | "
                f"{float(row['kernel_ms']):.3f} | "
                f"{float(row['share_of_annotated_kernel_pct']):.2f}% |\n"
            )
        transfer_rows = [row for row in operation_rows if row["kind"] == "memcpy"]
        handle.write("\n## Attributed transfer work by device\n\n")
        handle.write("| Family | Device | Time ms | Bytes | Transfer-time share |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in transfer_rows:
            handle.write(
                f"| {row['family']} | {row['device']} | "
                f"{float(row['duration_ms']):.3f} | {row['bytes']} | "
                f"{float(row['share_of_kind_pct']):.2f}% |\n"
            )
        handle.write(
            "\nTargeted Nsight Compute reports are under `ncu/`; the routed "
            "captures use bounded exact production-dispatch harnesses, while the "
            "attention captures reproduce the 32K indexed and mixed-score shapes.\n"
        )
    print((output / "profile-summary.md").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
