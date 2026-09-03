#!/usr/bin/env python3
"""Consolidate genuine four-GPU prefill and decode production traces."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


MIXED15 = {6, 8, 10, 12, 14, 16, 18, 20, 30, 32, 34, 36, 38, 40, 42}


class SummaryError(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise SummaryError(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise SummaryError(f"missing TSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_csv(path: Path, fieldnames: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def layout_by_trace(rows: list[dict[str, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        label = row.get("label", "")
        layout = row.get("layout", "")
        if not label or layout not in ("mixed15", "all43"):
            raise SummaryError(f"invalid trace-map row: {row}")
        if label in result:
            raise SummaryError(f"duplicate trace label: {label}")
        result[label] = layout
    return result


def integer_field(text: str, key: str) -> int | None:
    match = re.search(rf"(?:^|/){re.escape(key)}=(\d+)(?:/|$)", text)
    return int(match.group(1)) if match else None


def operation_layer(row: dict[str, str]) -> int | None:
    value = row.get("layer", "")
    if value and value != "-1":
        try:
            return int(value)
        except ValueError:
            pass
    for key in ("layer_range", "stage_range"):
        value = integer_field(row.get(key, ""), "layer")
        if value is not None:
            return value
    return None


def partner_family(row: dict[str, str]) -> str:
    text = row.get("partner_range", "").lower()
    if "t32" in text or "q_b" in text or "qb" in text:
        return "partner_t32"
    if "t256" in text or "output_b" in text:
        return "partner_t256"
    return "partner_dense_projection"


def family_of(row: dict[str, str], layout: str) -> str:
    name = row.get("name", "")
    lower = name.lower()
    layer = operation_layer(row)
    kind = row.get("kind", "")
    if kind == "memcpy":
        if row.get("attention_rows_range", ""):
            return "attention_row_split_transfer"
        if row.get("indexer_rows_range", ""):
            return "indexer_row_split_transfer"
        if row.get("handoff_range", ""):
            return "pipeline_handoff_transfer"
        if row.get("partner_range", ""):
            return f"{partner_family(row)}_transfer"
        return "other_transfer"
    if row.get("partner_range", ""):
        if "f32_to_f16" in lower:
            return f"{partner_family(row)}_conversion"
        return f"{partner_family(row)}_compute"
    if "q8_k_quantize_sm75_native_kernel" in lower:
        return "direct_native_q8_quantize"
    if "q8_k_quantize" in lower or "quantize_q8" in lower:
        return "canonical_q8_quantize"
    if "matmul_f16_pair_compressor_store_ordered_chunks_kernel" in lower:
        return "compressor_projection_state_fused"
    if "moe_gate_up_mid_decode_sm75_q3a4" in lower:
        return "q3a4_gate_up"
    if "moe_gate_up_mid_decode_sm75_q4_32" in lower:
        return "q4_32_gate_up"
    if "moe_gate_up_mid_sm75_q32" in lower:
        q3 = layout == "all43" or layer in MIXED15
        return "q3a4_gate_up" if q3 else "q4_32_gate_up"
    if "moe_down_sm75_q4_32" in lower:
        return "q4_32_down"
    if "attention_indexed" in lower:
        return "attention_indexed"
    if "attention_decode_mixed" in lower or "attention_prefill_mixed" in lower:
        return "attention_mixed"
    if "attention" in lower:
        return "attention_other"
    if "indexer" in lower and "topk" in lower:
        return "indexer_topk"
    if "indexer" in lower:
        return "indexer_score"
    if "matmul_q8" in lower or ("q8_0" in lower and "matmul" in lower):
        return "dense_q8"
    if any(marker in lower for marker in (
        "cutlass::", "cublas", "gemm", "gemvx::", "turing_s",
    )):
        return "dense_fp16"
    if "compressor" in lower:
        return "compressor_other"
    if "rope" in lower:
        return "rope"
    if any(marker in lower for marker in (
        "f32_to_f16", "fp8_kv_quantize", "pack_", "dequant",
    )):
        return "format_convert_pack"
    if "moe" in lower or "routed" in lower or "router" in lower:
        return "routed_support"
    if any(marker in lower for marker in (
        "rms_norm", "swiglu", "add_kernel", "fill_", "embed_tokens",
        "output_hc", "hc_",
    )):
        return "elementwise_control"
    return "other"


def benchmark_row(path: str, phase: str) -> dict[str, float]:
    rows = read_csv(Path(path))
    if len(rows) != 1:
        raise SummaryError(f"{path}: expected one benchmark row, found {len(rows)}")
    row = rows[0]
    try:
        return {
            "context": float(row["ctx_tokens"]),
            "prefill_tps": float(row["prefill_tps"]),
            "decode_tps": float(row["gen_steady_tps"]),
        }
    except (KeyError, ValueError) as exc:
        raise SummaryError(f"{path}: malformed {phase} benchmark: {exc}") from exc


def aggregate_families(
    operations: list[dict[str, str]], layouts: dict[str, str], phase: str
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"duration_ns": 0, "bytes": 0, "calls": 0}
    )
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for row in operations:
        trace = row["trace"]
        if trace not in layouts:
            raise SummaryError(f"{phase}: operation has unknown trace {trace}")
        kind = row["kind"]
        family = family_of(row, layouts[trace])
        key = (trace, kind, family)
        duration = int(row["duration_ns"])
        groups[key]["duration_ns"] += duration
        groups[key]["bytes"] += int(row.get("bytes", "0") or 0)
        groups[key]["calls"] += 1
        totals[(trace, kind)] += duration
    result: list[dict[str, object]] = []
    for (trace, kind, family), value in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], -item[1]["duration_ns"])
    ):
        total = totals[(trace, kind)]
        result.append({
            "phase": phase,
            "trace": trace,
            "layout": layouts[trace],
            "kind": kind,
            "family": family,
            "calls": value["calls"],
            "duration_ms": value["duration_ns"] / 1.0e6,
            "bytes": value["bytes"],
            "share_of_kind_pct": 100.0 * value["duration_ns"] / total if total else 0.0,
        })
    return result


def stage_balance(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["trace"], int(row["stage"]))].append(row)
    per_stage: dict[tuple[str, int], float] = {}
    for key, values in grouped.items():
        per_stage[key] = max(float(row["sum_gpu_envelope_ms"]) for row in values)
    result = []
    for trace in sorted({key[0] for key in per_stage}):
        early = per_stage.get((trace, 0), 0.0)
        late = per_stage.get((trace, 1), 0.0)
        total = early + late
        result.append({
            "trace": trace,
            "stage0_envelope_ms": early,
            "stage1_envelope_ms": late,
            "stage1_over_stage0": late / early if early else 0.0,
            "absolute_skew_pct": 100.0 * abs(late - early) / total if total else 0.0,
            "heavier_stage": 0 if early >= late else 1,
        })
    return result


def phase3_rows(
    prefill: list[dict[str, object]], decode: list[dict[str, object]]
) -> list[dict[str, object]]:
    candidates = [
        ("pipeline-boundary", {"pipeline_handoff_transfer"}),
        ("quantize-moe-input-once", {"direct_native_q8_quantize", "canonical_q8_quantize"}),
        ("compact-owner-local-routed-slots", {"routed_support"}),
        ("all43-q3a4-tile16-k-streaming-prefill", {"q3a4_gate_up", "q4_32_down"}),
        ("t256-attention-output-fp16", {
            "partner_t256_compute", "partner_t256_conversion", "partner_t256_transfer",
            "partner_dense_projection_compute", "partner_dense_projection_conversion",
            "partner_dense_projection_transfer",
        }),
    ]
    rows = []
    for phase, families in (("prefill", prefill), ("decode", decode)):
        traces = sorted({str(row["trace"]) for row in families})
        kernel_total: dict[str, float] = defaultdict(float)
        transfer_total: dict[str, int] = defaultdict(int)
        for row in families:
            trace = str(row["trace"])
            if row["kind"] == "kernel":
                kernel_total[trace] += float(row["duration_ms"])
            elif row["kind"] == "memcpy":
                transfer_total[trace] += int(row["bytes"])
        for target, selected in candidates:
            by_trace: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
            for row in families:
                if row["family"] not in selected:
                    continue
                trace = str(row["trace"])
                ms, byte_count = by_trace[trace]
                if row["kind"] == "kernel":
                    ms += float(row["duration_ms"])
                else:
                    byte_count += int(row["bytes"])
                by_trace[trace] = (ms, byte_count)
            for trace in traces:
                ms, byte_count = by_trace[trace]
                rows.append({
                    "phase": phase,
                    "trace": trace,
                    "target": target,
                    "kernel_ms": ms,
                    "kernel_work_share_pct": 100.0 * ms / kernel_total[trace] if kernel_total[trace] else 0.0,
                    "transfer_bytes": byte_count,
                    "transfer_byte_share_pct": 100.0 * byte_count / transfer_total[trace] if transfer_total[trace] else 0.0,
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)

    prefill_map_rows = read_tsv(root / "prefill" / "trace-map.tsv")
    decode_map_rows = read_tsv(root / "decode" / "trace-map.tsv")
    prefill_layouts = layout_by_trace(prefill_map_rows)
    decode_layouts = layout_by_trace(decode_map_rows)
    if set(prefill_layouts.values()) != {"mixed15", "all43"} or \
       set(decode_layouts.values()) != {"mixed15", "all43"}:
        raise SummaryError("both phases must contain mixed15 and all43 traces")

    prefill_ops = read_csv(root / "prefill" / "operation-attribution.csv")
    decode_ops = read_csv(root / "decode" / "summary" / "operation-attribution.csv")
    prefill_families = aggregate_families(prefill_ops, prefill_layouts, "prefill")
    decode_families = aggregate_families(decode_ops, decode_layouts, "decode")
    family_fields = [
        "phase", "trace", "layout", "kind", "family", "calls",
        "duration_ms", "bytes", "share_of_kind_pct",
    ]
    write_csv(summary / "prefill-family-summary.csv", family_fields, prefill_families)
    write_csv(summary / "decode-family-summary.csv", family_fields, decode_families)

    balances = stage_balance(read_csv(root / "prefill" / "stage-device-summary.csv"))
    write_csv(summary / "prefill-stage-balance.csv", [
        "trace", "stage0_envelope_ms", "stage1_envelope_ms",
        "stage1_over_stage0", "absolute_skew_pct", "heavier_stage",
    ], balances)

    throughput_rows = []
    for phase, map_rows in (("prefill", prefill_map_rows), ("decode", decode_map_rows)):
        for row in map_rows:
            values = benchmark_row(row["benchmark"], phase)
            throughput_rows.append({
                "phase": phase,
                "trace": row["label"],
                "layout": row["layout"],
                "stage_split": row["stage_split"],
                "context": int(values["context"]),
                "prefill_tps": values["prefill_tps"],
                "decode_tps": values["decode_tps"],
            })
    write_csv(summary / "throughput.csv", [
        "phase", "trace", "layout", "stage_split", "context",
        "prefill_tps", "decode_tps",
    ], throughput_rows)

    targets = phase3_rows(prefill_families, decode_families)
    write_csv(summary / "phase3-target-evidence.csv", [
        "phase", "trace", "target", "kernel_ms", "kernel_work_share_pct",
        "transfer_bytes", "transfer_byte_share_pct",
    ], targets)

    prefill_trace = {row["trace"]: row for row in read_csv(root / "prefill" / "trace-summary.csv")}
    decode_trace = {row["trace"]: row for row in read_csv(root / "decode" / "summary" / "trace-summary.csv")}
    balance_by_trace = {str(row["trace"]): row for row in balances}
    lines = [
        "# SM75 Phase 2 consolidated production profile",
        "",
        "All measurements use the accepted production defaults and stable pair policy: pair-0 attention rows off, pair-0 indexer rows on, and both pair-1 row splits on. Kernel shares are aggregate GPU work across concurrent devices, not elapsed-time Amdahl shares.",
        "",
        "## End-to-end and stage balance",
        "",
        "| Phase | Layout | Throughput | GPU span | Stage 1 / stage 0 | Heavier stage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in throughput_rows:
        trace = str(row["trace"])
        if row["phase"] == "prefill":
            trace_row = prefill_trace[trace]
            balance = balance_by_trace[trace]
            value = float(row["prefill_tps"])
            span = float(trace_row["pipeline_gpu_span_ms"])
            ratio = float(balance["stage1_over_stage0"])
            heavier = int(balance["heavier_stage"])
        else:
            trace_row = decode_trace[trace]
            value = float(row["decode_tps"])
            span = float(trace_row["gpu_envelope_ms"])
            ratio = 0.0
            heavier = -1
        lines.append(
            f"| {row['phase']} | {row['layout']} | {value:.3f} tok/s | "
            f"{span:.3f} ms | {ratio:.4f}x | {heavier if heavier >= 0 else 'n/a'} |"
        )

    lines.extend(["", "## Largest kernel-work families", ""])
    for phase, families in (("prefill", prefill_families), ("decode", decode_families)):
        lines.extend([
            f"### {phase.capitalize()}", "",
            "| Layout | Family | Kernel ms | Work share |",
            "|---|---|---:|---:|",
        ])
        for trace in sorted({str(row["trace"]) for row in families}):
            selected = sorted(
                (row for row in families if row["trace"] == trace and row["kind"] == "kernel"),
                key=lambda row: float(row["duration_ms"]), reverse=True,
            )[:8]
            for row in selected:
                lines.append(
                    f"| {row['layout']} | {row['family']} | "
                    f"{float(row['duration_ms']):.3f} | "
                    f"{float(row['share_of_kind_pct']):.2f}% |"
                )
        lines.append("")

    lines.extend(["## Largest transfer families", ""])
    for phase, families in (("prefill", prefill_families), ("decode", decode_families)):
        lines.extend([
            f"### {phase.capitalize()}", "",
            "| Layout | Family | Transfer GiB | GPU copy ms | Copy-time share |",
            "|---|---|---:|---:|---:|",
        ])
        for trace in sorted({str(row["trace"]) for row in families}):
            selected = sorted(
                (row for row in families if row["trace"] == trace and row["kind"] == "memcpy"),
                key=lambda row: float(row["duration_ms"]), reverse=True,
            )[:8]
            for row in selected:
                lines.append(
                    f"| {row['layout']} | {row['family']} | "
                    f"{int(row['bytes']) / (1024.0 ** 3):.3f} | "
                    f"{float(row['duration_ms']):.3f} | "
                    f"{float(row['share_of_kind_pct']):.2f}% |"
                )
        lines.append("")

    ncu_path = root / "ncu-decode-weight" / "summary.csv"
    if ncu_path.is_file() and ncu_path.stat().st_size:
        ncu_rows = sorted(
            read_csv(ncu_path),
            key=lambda row: float(row["duration_us"]),
            reverse=True,
        )
        lines.extend([
            "## Bounded decode-kernel diagnosis", "",
            "| Scenario | Duration us | DRAM peak | L2 hit | Long scoreboard | Occupancy | Evidence |",
            "|---|---:|---:|---:|---:|---:|---|",
        ])
        for row in ncu_rows:
            lines.append(
                f"| {row['scenario']} | {float(row['duration_us']):.3f} | "
                f"{float(row['dram_peak_pct']):.2f}% | "
                f"{float(row['l2_hit_pct']):.2f}% | "
                f"{float(row['long_scoreboard_ratio']):.3f} | "
                f"{float(row['achieved_occupancy_pct']):.2f}% | "
                f"{row['evidence_class']} |"
            )
        lines.append("")

    lines.extend([
        "## Phase 3 evidence map",
        "",
        "The machine-readable target table reports the measured kernel-work and transfer-byte exposure for each proposed Phase 3 target. A zero means that the target is not directly represented by an attributed family, not that its implementation has zero possible benefit.",
        "",
        "See `phase3-target-evidence.csv`, `prefill-stage-balance.csv`, and the per-phase family summaries for target selection.",
        "",
    ])
    text = "\n".join(lines)
    (summary / "summary.md").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SummaryError as exc:
        raise SystemExit(f"error: {exc}")
