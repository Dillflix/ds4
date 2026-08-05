#!/usr/bin/env python3
"""Summarize DS4-owned NVTX ranges from two Nsight Systems SQLite exports."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NvtxRange:
    start: int
    end: int
    tid: int
    text: str


def table_exists(db: sqlite3.Connection, name: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in text.split("/"):
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def enclosing_range(
    ranges: list[NvtxRange], timestamp: int, prefix: str
) -> NvtxRange | None:
    matches = [
        item
        for item in ranges
        if item.text.startswith(prefix) and item.start <= timestamp <= item.end
    ]
    return min(matches, key=lambda item: item.end - item.start) if matches else None


def load_trace(label: str, sqlite_path: Path, devices: list[int]):
    db = sqlite3.connect(sqlite_path)
    if not table_exists(db, "NVTX_EVENTS"):
        raise RuntimeError(f"{sqlite_path} has no NVTX_EVENTS table")

    ranges_by_tid: dict[int, list[NvtxRange]] = defaultdict(list)
    rows = db.execute(
        """
        SELECT n.start, n.end, n.globalTid,
               COALESCE(n.text, s.value, '')
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id = n.textId
        WHERE n.end IS NOT NULL
          AND COALESCE(n.text, s.value, '') LIKE 'ds4/%'
        """
    ).fetchall()
    for start, end, tid, text in rows:
        if tid is not None:
            ranges_by_tid[int(tid)].append(
                NvtxRange(int(start), int(end), int(tid), str(text))
            )
    for items in ranges_by_tid.values():
        items.sort(key=lambda item: (item.start, -item.end))
    if not ranges_by_tid:
        raise RuntimeError(
            f"{sqlite_path} contains no DS4 timeline ranges; verify "
            "DS4_CUDA_CRITICAL_PATH_NVTX=1 and nvtx tracing"
        )

    runtime = {
        int(correlation): (int(tid), int(start))
        for start, _end, tid, correlation in db.execute(
            """
            SELECT start, end, globalTid, correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE globalTid IS NOT NULL AND correlationId IS NOT NULL
            """
        )
    }

    operations: list[dict[str, object]] = []

    def append_operation(
        kind: str,
        start: int,
        end: int,
        device: int,
        stream: int,
        correlation: int | None,
        name: str,
        byte_count: int = 0,
    ) -> None:
        if correlation is None or int(correlation) not in runtime:
            return
        tid, api_start = runtime[int(correlation)]
        host_ranges = ranges_by_tid.get(tid, [])
        stage = enclosing_range(host_ranges, api_start, "ds4/prefill/stage/")
        layer = enclosing_range(host_ranges, api_start, "ds4/prefill/layer/")
        handoff = enclosing_range(host_ranges, api_start, "ds4/prefill/handoff/")
        partner = enclosing_range(host_ranges, api_start, "ds4/q8/partner/")
        wave = enclosing_range(host_ranges, api_start, "ds4/prefill/wave/")
        embedding = enclosing_range(host_ranges, api_start, "ds4/prefill/embedding/")
        output = enclosing_range(host_ranges, api_start, "ds4/prefill/output/")
        if not any((stage, handoff, partner, wave, embedding, output)):
            return
        operations.append(
            {
                "trace": label,
                "kind": kind,
                "start_ns": int(start),
                "end_ns": int(end),
                "duration_ns": int(end) - int(start),
                "device": int(device),
                "stream": int(stream),
                "bytes": int(byte_count),
                "name": name,
                "stage_range": stage.text if stage else "",
                "layer_range": layer.text if layer else "",
                "handoff_range": handoff.text if handoff else "",
                "partner_range": partner.text if partner else "",
                "wave_range": wave.text if wave else "",
                "embedding_range": embedding.text if embedding else "",
                "output_range": output.text if output else "",
            }
        )

    for row in db.execute(
        """
        SELECT k.start, k.end, k.deviceId, k.streamId, k.correlationId,
               COALESCE(s.value, '')
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON s.id = k.demangledName
        """
    ):
        append_operation("kernel", *row)

    if table_exists(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for row in db.execute(
            """
            SELECT start, end, deviceId, streamId, correlationId, bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            """
        ):
            start, end, device, stream, correlation, byte_count = row
            append_operation(
                "memcpy",
                start,
                end,
                device,
                stream,
                correlation,
                "memcpy",
                byte_count,
            )
    db.close()

    stage_meta: dict[str, dict[str, str]] = {}
    for items in ranges_by_tid.values():
        for item in items:
            if item.text.startswith("ds4/prefill/stage/"):
                stage_meta[item.text] = parse_fields(item.text)

    annotated_kernel_ns = sum(
        int(op["duration_ns"]) for op in operations if op["kind"] == "kernel"
    )
    return {
        "label": label,
        "path": sqlite_path,
        "devices": devices,
        "ranges": [item for items in ranges_by_tid.values() for item in items],
        "operations": operations,
        "stage_meta": stage_meta,
        "annotated_kernel_ns": annotated_kernel_ns,
    }


def write_csv(path: Path, fieldnames: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def median_or_zero(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def read_trace_map(path: Path):
    traces = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            devices = [int(value) for value in row["devices"].split(",")]
            if len(devices) != 4:
                raise RuntimeError(f"{row['label']} must specify four devices")
            traces.append((row["label"], Path(row["sqlite"]), devices))
    if len(traces) != 2:
        raise RuntimeError("trace-map.tsv must contain exactly two traces")
    return traces


def read_harness(path: Path):
    samples = []
    if not path.exists():
        return samples
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            values: dict[str, str] = {}
            log_path = Path(row["log"])
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    values[key] = value
            if "timed_per_call_ms" not in values:
                raise RuntimeError(f"missing timed_per_call_ms in {log_path}")
            samples.append(
                {
                    "trial": int(row["trial"]),
                    "slot": int(row["slot"]),
                    "scenario": row["scenario"],
                    "device": int(row["device"]),
                    "timed_per_call_ms": float(values["timed_per_call_ms"]),
                    "log": str(log_path),
                }
            )
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    traces = [
        load_trace(label, sqlite_path, devices)
        for label, sqlite_path, devices in read_trace_map(output_dir / "trace-map.tsv")
    ]

    all_ops = [op for trace in traces for op in trace["operations"]]
    op_fields = [
        "trace", "kind", "start_ns", "end_ns", "duration_ns", "device",
        "stream", "bytes", "name", "stage_range", "layer_range",
        "handoff_range", "partner_range", "wave_range", "embedding_range",
        "output_range",
    ]
    write_csv(output_dir / "operation-attribution.csv", op_fields, all_ops)

    def summarize_named_ranges(range_key: str):
        groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
        for op in all_ops:
            range_name = str(op[range_key])
            if range_name:
                groups[(str(op["trace"]), range_name, int(op["device"]))].append(op)
        result = []
        for (trace_label, range_name, device), ops in sorted(groups.items()):
            fields = parse_fields(range_name)
            result.append(
                {
                    "trace": trace_label,
                    "range": range_name,
                    "device": device,
                    "stage": fields.get("stage", fields.get("from_stage", "")),
                    "microbatch": fields.get("mb", ""),
                    "tier": fields.get("tier", fields.get("home_tier", "")),
                    "layer": fields.get("layer", ""),
                    "label": fields.get("label", ""),
                    "kernel_ms": sum(
                        int(op["duration_ns"])
                        for op in ops if op["kind"] == "kernel"
                    ) / 1e6,
                    "kernel_count": sum(op["kind"] == "kernel" for op in ops),
                    "memcpy_ms": sum(
                        int(op["duration_ns"])
                        for op in ops if op["kind"] == "memcpy"
                    ) / 1e6,
                    "memcpy_count": sum(op["kind"] == "memcpy" for op in ops),
                    "memcpy_bytes": sum(
                        int(op["bytes"]) for op in ops if op["kind"] == "memcpy"
                    ),
                    "gpu_envelope_ms": (
                        max(int(op["end_ns"]) for op in ops)
                        - min(int(op["start_ns"]) for op in ops)
                    ) / 1e6,
                }
            )
        return result

    named_fields = [
        "trace", "range", "device", "stage", "microbatch", "tier",
        "layer", "label", "kernel_ms", "kernel_count", "memcpy_ms",
        "memcpy_count", "memcpy_bytes", "gpu_envelope_ms",
    ]
    write_csv(
        output_dir / "layer-device-summary.csv",
        named_fields,
        summarize_named_ranges("layer_range"),
    )
    write_csv(
        output_dir / "partner-projection-summary.csv",
        named_fields,
        summarize_named_ranges("partner_range"),
    )
    write_csv(
        output_dir / "handoff-device-summary.csv",
        named_fields,
        summarize_named_ranges("handoff_range"),
    )

    stage_mb_rows = []
    stage_rows = []
    stage_cells: dict[tuple[str, int, int], dict[str, float]] = {}
    trace_rows = []
    for trace in traces:
        devices = trace["devices"]
        stage_groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
        for op in trace["operations"]:
            if op["stage_range"]:
                stage_groups[(str(op["stage_range"]), int(op["device"]))].append(op)

        aggregate: dict[tuple[int, int], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        stage_names: dict[int, dict[str, str]] = {}
        for (stage_name, device), ops in sorted(stage_groups.items()):
            fields = parse_fields(stage_name)
            stage = int(fields["stage"])
            stage_names[stage] = fields
            start = min(int(op["start_ns"]) for op in ops)
            end = max(int(op["end_ns"]) for op in ops)
            kernel_ns = sum(
                int(op["duration_ns"]) for op in ops if op["kind"] == "kernel"
            )
            memcpy_ns = sum(
                int(op["duration_ns"]) for op in ops if op["kind"] == "memcpy"
            )
            memcpy_bytes = sum(
                int(op["bytes"]) for op in ops if op["kind"] == "memcpy"
            )
            stage_mb_rows.append(
                {
                    "trace": trace["label"],
                    "stage": stage,
                    "microbatch": int(fields["mb"]),
                    "tier": int(fields["tier"]),
                    "layers": fields["layers"],
                    "device": device,
                    "kernel_ms": kernel_ns / 1e6,
                    "memcpy_ms": memcpy_ns / 1e6,
                    "memcpy_bytes": memcpy_bytes,
                    "gpu_envelope_ms": (end - start) / 1e6,
                    "operation_count": len(ops),
                }
            )
            cell = aggregate[(stage, device)]
            cell["kernel_ns"] += kernel_ns
            cell["memcpy_ns"] += memcpy_ns
            cell["memcpy_bytes"] += memcpy_bytes
            cell["envelope_ns"] += end - start
            cell["microbatch_device_intervals"] += 1

        for stage, fields in sorted(stage_names.items()):
            tier = int(fields["tier"])
            first, end_layer = (int(value) for value in fields["layers"].split("-"))
            layer_count = end_layer - first
            home = devices[tier]
            partner_tier = tier + len(devices) // 2
            partner = devices[partner_tier] if partner_tier < len(devices) else -1
            for device in sorted({key[1] for key in aggregate if key[0] == stage}):
                cell = aggregate[(stage, device)]
                role = "home" if device == home else "partner" if device == partner else "other"
                kernel_ms = cell["kernel_ns"] / 1e6
                row = {
                    "trace": trace["label"],
                    "stage": stage,
                    "tier": tier,
                    "layers": fields["layers"],
                    "layer_count": layer_count,
                    "home_device": home,
                    "partner_device": partner,
                    "device": device,
                    "role": role,
                    "kernel_ms": kernel_ms,
                    "kernel_ms_per_layer": kernel_ms / layer_count,
                    "memcpy_ms": cell["memcpy_ns"] / 1e6,
                    "memcpy_bytes": int(cell["memcpy_bytes"]),
                    "sum_gpu_envelope_ms": cell["envelope_ns"] / 1e6,
                    "microbatch_device_intervals": int(cell["microbatch_device_intervals"]),
                }
                stage_rows.append(row)
                stage_cells[(str(trace["label"]), stage, device)] = row

        wave_ops = [op for op in trace["operations"] if op["wave_range"]]
        annotated_ops = trace["operations"]
        trace_rows.append(
            {
                "trace": trace["label"],
                "devices": ",".join(str(value) for value in devices),
                "ds4_nvtx_ranges": len(trace["ranges"]),
                "annotated_operations": len(annotated_ops),
                "annotated_kernel_ms": sum(
                    int(op["duration_ns"])
                    for op in annotated_ops
                    if op["kind"] == "kernel"
                ) / 1e6,
                "annotated_memcpy_ms": sum(
                    int(op["duration_ns"])
                    for op in annotated_ops
                    if op["kind"] == "memcpy"
                ) / 1e6,
                "pipeline_gpu_span_ms": (
                    (max(int(op["end_ns"]) for op in wave_ops)
                     - min(int(op["start_ns"]) for op in wave_ops)) / 1e6
                    if wave_ops else 0.0
                ),
            }
        )

    write_csv(
        output_dir / "stage-microbatch-device.csv",
        [
            "trace", "stage", "microbatch", "tier", "layers", "device",
            "kernel_ms", "memcpy_ms", "memcpy_bytes", "gpu_envelope_ms",
            "operation_count",
        ],
        stage_mb_rows,
    )
    write_csv(
        output_dir / "stage-device-summary.csv",
        [
            "trace", "stage", "tier", "layers", "layer_count", "home_device",
            "partner_device", "device", "role", "kernel_ms",
            "kernel_ms_per_layer", "memcpy_ms", "memcpy_bytes",
            "sum_gpu_envelope_ms", "microbatch_device_intervals",
        ],
        stage_rows,
    )
    write_csv(
        output_dir / "trace-summary.csv",
        [
            "trace", "devices", "ds4_nvtx_ranges", "annotated_operations",
            "annotated_kernel_ms", "annotated_memcpy_ms", "pipeline_gpu_span_ms",
        ],
        trace_rows,
    )

    harness = read_harness(output_dir / "harness-runs.tsv")
    write_csv(
        output_dir / "same-work-gpu-samples.csv",
        ["trial", "slot", "scenario", "device", "timed_per_call_ms", "log"],
        harness,
    )
    harness_groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for sample in harness:
        harness_groups[(sample["scenario"], sample["device"])].append(
            sample["timed_per_call_ms"]
        )
    harness_rows = []
    scenarios = sorted({key[0] for key in harness_groups})
    for scenario in scenarios:
        values0 = harness_groups.get((scenario, 0), [])
        values3 = harness_groups.get((scenario, 3), [])
        med0 = median_or_zero(values0)
        med3 = median_or_zero(values3)
        harness_rows.append(
            {
                "scenario": scenario,
                "gpu0_samples": len(values0),
                "gpu0_median_ms": med0,
                "gpu3_samples": len(values3),
                "gpu3_median_ms": med3,
                "gpu3_over_gpu0": med3 / med0 if med0 else 0.0,
            }
        )
    write_csv(
        output_dir / "same-work-gpu-summary.csv",
        [
            "scenario", "gpu0_samples", "gpu0_median_ms", "gpu3_samples",
            "gpu3_median_ms", "gpu3_over_gpu0",
        ],
        harness_rows,
    )

    # The two maps form a 2x2 experiment: each stage runs once on GPU 0 and
    # once on GPU 3.  Per-layer home-kernel time removes the 22/21 layer-count
    # difference before separating physical-GPU and stage effects.
    values: dict[tuple[int, int], float] = {}
    for row in stage_rows:
        if row["role"] == "home" and row["device"] in (0, 3):
            values[(int(row["stage"]), int(row["device"]))] = float(
                row["kernel_ms_per_layer"]
            )
    gpu_factor = stage_factor = 0.0
    if all((stage, device) in values for stage in (0, 1) for device in (0, 3)):
        gpu_factor = math.sqrt(
            (values[(0, 3)] / values[(0, 0)])
            * (values[(1, 3)] / values[(1, 0)])
        )
        stage_factor = math.sqrt(
            (values[(1, 0)] / values[(0, 0)])
            * (values[(1, 3)] / values[(0, 3)])
        )

    with (output_dir / "analysis.txt").open("w", encoding="utf-8") as handle:
        handle.write("DS4 SM75 critical-path / pair-attribution audit\n\n")
        for row in trace_rows:
            handle.write(
                f"{row['trace']}: devices={row['devices']} "
                f"pipeline_gpu_span={row['pipeline_gpu_span_ms']:.3f} ms "
                f"annotated_kernel={row['annotated_kernel_ms']:.3f} ms\n"
            )
        handle.write("\nPer-layer home-kernel 2x2 cells (ms):\n")
        for stage in (0, 1):
            handle.write(
                f"  stage {stage}: gpu0={values.get((stage, 0), 0.0):.6f} "
                f"gpu3={values.get((stage, 3), 0.0):.6f}\n"
            )
        handle.write(
            f"\nphysical_gpu3_over_gpu0_factor={gpu_factor:.6f}\n"
            f"late_stage_over_early_stage_factor={stage_factor:.6f}\n"
        )
        if harness_rows:
            handle.write("\nSame-work harness medians:\n")
            for row in harness_rows:
                handle.write(
                    f"  {row['scenario']}: gpu0={row['gpu0_median_ms']:.6f} ms "
                    f"gpu3={row['gpu3_median_ms']:.6f} ms "
                    f"ratio={row['gpu3_over_gpu0']:.6f}\n"
                )
        handle.write(
            "\nInterpretation boundary: the 2x2 factors describe attributed GPU "
            "work, not end-to-end speedup. Use trace-summary.csv for pipeline "
            "span and same-work-gpu-summary.csv to confirm or reject a physical "
            "GPU/clock explanation.\n"
        )
    print((output_dir / "analysis.txt").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
