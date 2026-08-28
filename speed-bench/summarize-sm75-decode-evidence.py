#!/usr/bin/env python3
"""Validate and summarize bounded SM75 production decode evidence."""

from __future__ import annotations

import argparse
import bisect
import csv
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


@dataclass(frozen=True)
class RangeIndex:
    items: list[NvtxRange]
    starts: list[int]
    prefix_max_ends: list[int]


PREFIXES = (
    "ds4/decode/token/",
    "ds4/decode/layer/",
    "ds4/decode/stage/",
    "ds4/decode/embedding/",
    "ds4/decode/output/",
)


class EvidenceError(RuntimeError):
    pass


def table_exists(db: sqlite3.Connection, name: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def fields(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in text.split("/"):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def build_index(items: list[NvtxRange], prefix: str) -> RangeIndex:
    selected = sorted(
        (item for item in items if item.text.startswith(prefix)),
        key=lambda item: (item.start, -item.end),
    )
    starts: list[int] = []
    max_ends: list[int] = []
    max_end = -1
    for item in selected:
        starts.append(item.start)
        max_end = max(max_end, item.end)
        max_ends.append(max_end)
    return RangeIndex(selected, starts, max_ends)


def enclosing(index: RangeIndex | None, timestamp: int) -> NvtxRange | None:
    if index is None or not index.items:
        return None
    position = bisect.bisect_right(index.starts, timestamp) - 1
    best: NvtxRange | None = None
    best_duration: int | None = None
    while position >= 0 and index.prefix_max_ends[position] >= timestamp:
        item = index.items[position]
        if item.end >= timestamp:
            duration = item.end - item.start
            if best_duration is None or duration <= best_duration:
                best = item
                best_duration = duration
        position -= 1
    return best


def write_csv(path: Path, fieldnames: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def one_benchmark_row(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise EvidenceError(f"missing benchmark CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise EvidenceError(f"{path}: expected one row, found {len(rows)}")
    return rows[0]


def summarize_ab(root: Path) -> tuple[list[dict[str, object]], dict[str, float]]:
    path = root / "ab" / "runs.csv"
    if not path.exists():
        return [], {}
    samples: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))
    if not runs:
        raise EvidenceError("threshold A/B inventory is empty")
    seen: set[tuple[int, str]] = set()
    for run in runs:
        repeat = int(run["repeat"])
        variant = run["variant"]
        if variant not in ("indexed1024", "dense4096"):
            raise EvidenceError(f"unknown A/B variant: {variant}")
        key = (repeat, variant)
        if key in seen:
            raise EvidenceError(f"duplicate A/B sample: {key}")
        seen.add(key)
        row = one_benchmark_row(Path(run["csv"]))
        pp = int(run["pp_tokens"])
        tg = int(run["tg_tokens"])
        if int(row["ctx_tokens"]) != pp or int(row["gen_tokens"]) != tg:
            raise EvidenceError(f"{run['csv']}: benchmark shape does not match inventory")
        samples.append(
            {
                "repeat": repeat,
                "slot": int(run["slot"]),
                "variant": variant,
                "threshold": int(run["threshold"]),
                "pp_tokens": pp,
                "tg_tokens": tg,
                "gen_tps": float(row["gen_tps"]),
                "gen_first_ms": float(row["gen_first_ms"]),
                "gen_steady_tps": float(row["gen_steady_tps"]),
                "csv": run["csv"],
                "log": run["log"],
            }
        )
    repeats = sorted({int(sample["repeat"]) for sample in samples})
    for repeat in repeats:
        variants = {
            str(sample["variant"])
            for sample in samples
            if int(sample["repeat"]) == repeat
        }
        if variants != {"indexed1024", "dense4096"}:
            raise EvidenceError(f"repeat {repeat} lacks a complete threshold pair")
    by_variant = {
        variant: [
            float(sample["gen_steady_tps"])
            for sample in samples
            if sample["variant"] == variant
        ]
        for variant in ("indexed1024", "dense4096")
    }
    paired = []
    for repeat in repeats:
        values = {
            str(sample["variant"]): float(sample["gen_steady_tps"])
            for sample in samples
            if int(sample["repeat"]) == repeat
        }
        paired.append(values["dense4096"] / values["indexed1024"])
    result = {
        "indexed_median_tps": statistics.median(by_variant["indexed1024"]),
        "dense_median_tps": statistics.median(by_variant["dense4096"]),
        "dense_over_indexed": statistics.median(paired),
        "paired_speedup_sd": statistics.stdev(paired) if len(paired) > 1 else 0.0,
    }
    return samples, result


def load_trace(
    label: str,
    pp: int,
    threshold: int,
    devices: list[int],
    sqlite_path: Path,
) -> tuple[list[dict[str, object]], int]:
    if not sqlite_path.is_file() or sqlite_path.stat().st_size == 0:
        raise EvidenceError(f"missing trace database: {sqlite_path}")
    db = sqlite3.connect(sqlite_path)
    if not table_exists(db, "NVTX_EVENTS"):
        raise EvidenceError(f"{sqlite_path} has no NVTX_EVENTS table")
    ranges_by_tid: dict[int, list[NvtxRange]] = defaultdict(list)
    for start, end, tid, text in db.execute(
        """
        SELECT n.start, n.end, n.globalTid, COALESCE(n.text, s.value, '')
        FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON s.id = n.textId
        WHERE n.end IS NOT NULL
          AND COALESCE(n.text, s.value, '') LIKE 'ds4/decode/%'
        """
    ):
        if tid is not None:
            ranges_by_tid[int(tid)].append(
                NvtxRange(int(start), int(end), int(tid), str(text))
            )
    if not ranges_by_tid:
        raise EvidenceError(f"{sqlite_path} contains no DS4 decode NVTX ranges")
    indexes = {
        tid: {prefix: build_index(items, prefix) for prefix in PREFIXES}
        for tid, items in ranges_by_tid.items()
    }
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

    def append(
        kind: str,
        start: int,
        end: int,
        device: int,
        stream: int,
        correlation: int | None,
        name: str,
        byte_count: int = 0,
    ) -> None:
        runtime_entry = (
            runtime.get(int(correlation)) if correlation is not None else None
        )
        tid, api_start = runtime_entry if runtime_entry else (-1, -1)
        by_prefix = indexes.get(tid, {})
        selected = {
            prefix: enclosing(by_prefix.get(prefix), api_start)
            if runtime_entry else None
            for prefix in PREFIXES
        }
        token_range = selected["ds4/decode/token/"]
        layer_range = selected["ds4/decode/layer/"]
        stage_range = selected["ds4/decode/stage/"]
        embedding_range = selected["ds4/decode/embedding/"]
        output_range = selected["ds4/decode/output/"]
        token_meta = fields(token_range.text) if token_range else {}
        layer_meta = fields(layer_range.text) if layer_range else {}
        stage_meta = fields(stage_range.text) if stage_range else {}
        if not runtime_entry:
            category = "unattributed-runtime"
        elif token_range is None:
            category = "unattributed-thread"
        elif output_range:
            category = "output"
        elif embedding_range:
            category = "embedding"
        elif stage_meta.get("name"):
            category = stage_meta["name"]
        elif layer_range:
            category = "layer-other"
        else:
            category = "token-other"
        operations.append(
            {
                "trace": label,
                "pp_tokens": pp,
                "threshold": threshold,
                "kind": kind,
                "start_ns": int(start),
                "end_ns": int(end),
                "duration_ns": int(end) - int(start),
                "device": int(device),
                "stream": int(stream),
                "bytes": int(byte_count),
                "name": name,
                "category": category,
                "token_pos": int(token_meta.get("pos", "-1")),
                "layer": int(layer_meta.get("layer", "-1")),
                "home_tier": int(layer_meta.get("tier", "-1")),
                "token_range": token_range.text if token_range else "",
                "layer_range": layer_range.text if layer_range else "",
                "stage_range": stage_range.text if stage_range else "",
            }
        )

    if not table_exists(db, "CUPTI_ACTIVITY_KIND_KERNEL"):
        raise EvidenceError(f"{sqlite_path} has no CUDA kernel table")
    for row in db.execute(
        """
        SELECT k.start, k.end, k.deviceId, k.streamId, k.correlationId,
               COALESCE(s.value, '')
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON s.id = k.demangledName
        """
    ):
        append("kernel", *row)
    if table_exists(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for start, end, device, stream, correlation, byte_count in db.execute(
            """
            SELECT start, end, deviceId, streamId, correlationId, bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            """
        ):
            append(
                "memcpy", start, end, device, stream, correlation,
                "memcpy", byte_count
            )
    db.close()
    token_positions = {
        int(fields(item.text).get("pos", "-1"))
        for items in ranges_by_tid.values()
        for item in items
        if item.text.startswith("ds4/decode/token/")
    }
    if not operations:
        raise EvidenceError(f"{sqlite_path} has no CUDA operations")
    if not token_positions or -1 in token_positions:
        raise EvidenceError(f"{sqlite_path} has invalid decode token ranges")
    unknown_devices = {int(op["device"]) for op in operations} - set(devices)
    if unknown_devices:
        raise EvidenceError(f"{label}: unexpected physical devices {sorted(unknown_devices)}")
    return operations, len(token_positions)


def summarize_traces(root: Path) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    path = root / "trace-map.tsv"
    if not path.exists():
        return [], [], [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        trace_rows = list(csv.DictReader(handle, delimiter="\t"))
    if not trace_rows:
        return [], [], [], [], []
    all_ops: list[dict[str, object]] = []
    token_counts: dict[str, int] = {}
    trace_meta: dict[str, tuple[int, int, list[int]]] = {}
    for row in trace_rows:
        label = row["label"]
        devices = [int(value) for value in row["devices"].split(",")]
        if len(devices) != 4:
            raise EvidenceError(f"{label}: expected four devices")
        pp = int(row["pp_tokens"])
        threshold = int(row["threshold"])
        expected_tokens = int(row["captured_tokens"])
        ops, token_count = load_trace(
            label, pp, threshold, devices, Path(row["sqlite"])
        )
        if token_count != expected_tokens:
            raise EvidenceError(
                f"{label}: captured {token_count} decode token ranges, "
                f"expected {expected_tokens}"
            )
        all_ops.extend(ops)
        token_counts[label] = token_count
        trace_meta[label] = (pp, threshold, devices)

    stage_groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    layer_groups: dict[tuple[str, int, int, int], list[dict[str, object]]] = defaultdict(list)
    kernel_groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for op in all_ops:
        stage_groups[(str(op["trace"]), str(op["category"]), int(op["device"]))].append(op)
        if int(op["layer"]) >= 0:
            layer_groups[(
                str(op["trace"]), int(op["layer"]), int(op["home_tier"]),
                int(op["device"])
            )].append(op)
        if op["kind"] == "kernel":
            kernel_groups[(str(op["trace"]), str(op["name"]), int(op["device"]))].append(op)

    stage_rows: list[dict[str, object]] = []
    for (label, category, device), ops in sorted(stage_groups.items()):
        kernels = [op for op in ops if op["kind"] == "kernel"]
        copies = [op for op in ops if op["kind"] == "memcpy"]
        stage_rows.append(
            {
                "trace": label,
                "category": category,
                "device": device,
                "kernel_ms": sum(int(op["duration_ns"]) for op in kernels) / 1e6,
                "kernel_calls": len(kernels),
                "memcpy_ms": sum(int(op["duration_ns"]) for op in copies) / 1e6,
                "memcpy_calls": len(copies),
                "memcpy_bytes": sum(int(op["bytes"]) for op in copies),
            }
        )
    layer_rows: list[dict[str, object]] = []
    for (label, layer, home_tier, device), ops in sorted(layer_groups.items()):
        layer_rows.append(
            {
                "trace": label,
                "layer": layer,
                "home_tier": home_tier,
                "device": device,
                "kernel_ms": sum(
                    int(op["duration_ns"]) for op in ops if op["kind"] == "kernel"
                ) / 1e6,
                "memcpy_ms": sum(
                    int(op["duration_ns"]) for op in ops if op["kind"] == "memcpy"
                ) / 1e6,
                "memcpy_bytes": sum(
                    int(op["bytes"]) for op in ops if op["kind"] == "memcpy"
                ),
            }
        )
    kernel_rows: list[dict[str, object]] = []
    for (label, name, device), ops in sorted(kernel_groups.items()):
        kernel_rows.append(
            {
                "trace": label,
                "device": device,
                "kernel": name,
                "calls": len(ops),
                "kernel_ms": sum(int(op["duration_ns"]) for op in ops) / 1e6,
            }
        )
    trace_summary: list[dict[str, object]] = []
    for label, (pp, threshold, devices) in trace_meta.items():
        ops = [op for op in all_ops if op["trace"] == label]
        kernel_ns = sum(
            int(op["duration_ns"]) for op in ops if op["kind"] == "kernel"
        )
        attributed_kernel_ns = sum(
            int(op["duration_ns"])
            for op in ops
            if op["kind"] == "kernel"
            and not str(op["category"]).startswith("unattributed-")
        )
        copy_ns = sum(
            int(op["duration_ns"]) for op in ops if op["kind"] == "memcpy"
        )
        trace_summary.append(
            {
                "trace": label,
                "pp_tokens": pp,
                "threshold": threshold,
                "captured_tokens": token_counts[label],
                "devices": ",".join(str(device) for device in devices),
                "operation_count": len(ops),
                "aggregate_kernel_ms": kernel_ns / 1e6,
                "attributed_kernel_ms": attributed_kernel_ns / 1e6,
                "attribution_pct": (
                    100.0 * attributed_kernel_ns / kernel_ns
                    if kernel_ns else 0.0
                ),
                "aggregate_memcpy_ms": copy_ns / 1e6,
                "memcpy_bytes": sum(int(op["bytes"]) for op in ops),
                "gpu_envelope_ms": (
                    max(int(op["end_ns"]) for op in ops)
                    - min(int(op["start_ns"]) for op in ops)
                ) / 1e6,
            }
        )
    return all_ops, stage_rows, layer_rows, kernel_rows, trace_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    ab_samples, ab_result = summarize_ab(root)
    if ab_samples:
        write_csv(
            summary_dir / "threshold-ab-samples.csv",
            [
                "repeat", "slot", "variant", "threshold", "pp_tokens",
                "tg_tokens", "gen_tps", "gen_first_ms", "gen_steady_tps",
                "csv", "log",
            ],
            ab_samples,
        )

    all_ops, stage_rows, layer_rows, kernel_rows, trace_rows = summarize_traces(root)
    if all_ops:
        write_csv(
            summary_dir / "operation-attribution.csv",
            [
                "trace", "pp_tokens", "threshold", "kind", "start_ns",
                "end_ns", "duration_ns", "device", "stream", "bytes",
                "name", "category", "token_pos", "layer", "home_tier",
                "token_range", "layer_range", "stage_range",
            ],
            all_ops,
        )
        write_csv(
            summary_dir / "stage-device-summary.csv",
            [
                "trace", "category", "device", "kernel_ms", "kernel_calls",
                "memcpy_ms", "memcpy_calls", "memcpy_bytes",
            ],
            stage_rows,
        )
        write_csv(
            summary_dir / "layer-device-summary.csv",
            [
                "trace", "layer", "home_tier", "device", "kernel_ms",
                "memcpy_ms", "memcpy_bytes",
            ],
            layer_rows,
        )
        write_csv(
            summary_dir / "kernel-device-summary.csv",
            ["trace", "device", "kernel", "calls", "kernel_ms"],
            sorted(kernel_rows, key=lambda row: (str(row["trace"]), -float(row["kernel_ms"]))),
        )
        write_csv(
            summary_dir / "trace-summary.csv",
            [
                "trace", "pp_tokens", "threshold", "captured_tokens",
                "devices", "operation_count", "aggregate_kernel_ms",
                "attributed_kernel_ms", "attribution_pct",
                "aggregate_memcpy_ms", "memcpy_bytes", "gpu_envelope_ms",
            ],
            trace_rows,
        )

    exact_path = root / "exact" / "verification.txt"
    exact = exact_path.is_file() and "bit_exact=true" in exact_path.read_text(
        encoding="utf-8", errors="replace"
    )
    lines = ["# SM75 production decode evidence", ""]
    if ab_result:
        change = (ab_result["dense_over_indexed"] - 1.0) * 100.0
        lines.extend(
            [
                "## PP4096 indexer threshold A/B",
                "",
                "| Indexed 1024 tok/s | Forced dense 4096 tok/s | Dense / indexed | Change | Paired speedup SD |",
                "| ---: | ---: | ---: | ---: | ---: |",
                (
                    f"| {ab_result['indexed_median_tps']:.3f} | "
                    f"{ab_result['dense_median_tps']:.3f} | "
                    f"{ab_result['dense_over_indexed']:.6f}x | "
                    f"{change:+.3f}% | {ab_result['paired_speedup_sd']:.6f} |"
                ),
                "",
            ]
        )
    if exact_path.exists():
        lines.extend(
            [
                "## Exactness",
                "",
                f"Per-token full-vocabulary FP32 logits: **{'bit-exact' if exact else 'FAILED'}**.",
                "",
            ]
        )
    trace_csv = summary_dir / "trace-summary.csv"
    if trace_csv.exists():
        with trace_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        lines.extend(
            [
                "## Bounded steady-decode traces",
                "",
                "Aggregate GPU work is a sum across devices; GPU envelope is wall-clock span.",
                "",
                "| Trace | Captured tokens | Aggregate kernel ms | Attributed | GPU envelope ms | Memcpy MiB |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['trace']} | {row['captured_tokens']} | "
                f"{float(row['aggregate_kernel_ms']):.3f} | "
                f"{float(row['attribution_pct']):.2f}% | "
                f"{float(row['gpu_envelope_ms']):.3f} | "
                f"{int(row['memcpy_bytes']) / 1048576.0:.3f} |"
            )
        lines.append("")
    if not ab_result and not exact_path.exists() and not all_ops:
        raise EvidenceError("no completed evidence phase was found")
    summary = "\n".join(lines) + "\n"
    (summary_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as exc:
        raise SystemExit(f"error: {exc}")
