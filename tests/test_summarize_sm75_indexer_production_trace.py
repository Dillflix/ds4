#!/usr/bin/env python3
"""Synthetic regression for the paired indexer production-trace summary."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench" / "summarize-sm75-indexer-production-trace.py"


def write_csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def make_variant(root: Path, variant: str, score_ns: int, span_ms: float,
                 tps: float) -> None:
    kernel = f"indexer_scores_{variant}_kernel(float *)"
    variant_dir = root / variant
    operation_fields = [
        "trace", "kind", "start_ns", "end_ns", "duration_ns", "device",
        "stream", "bytes", "name", "stage_range", "layer_range",
        "handoff_range", "partner_range", "attention_rows_range",
        "wave_range", "embedding_range", "output_range",
    ]
    half = score_ns // 2
    write_csv(
        variant_dir / "operation-attribution.csv",
        operation_fields,
        [
            [variant, "kernel", 0, half, half, 0, 1, 0, kernel,
             "ds4/prefill/stage/stage=0/mb=0/tier=0/layers=0-22/pos=32256/tokens=512",
             "ds4/prefill/layer/stage=0/mb=0/tier=0/layer=2/pos=32256/tokens=512",
             "", "", "", "", "", ""],
            [variant, "kernel", half, score_ns, score_ns - half, 3, 1, 0,
             kernel, "ds4/prefill/stage/stage=1/mb=0/tier=1/layers=22-43/pos=32256/tokens=512",
             "ds4/prefill/layer/stage=1/mb=0/tier=1/layer=22/pos=32256/tokens=512",
             "", "", "", "", "", ""],
        ],
    )
    write_csv(
        variant_dir / "trace-summary.csv",
        ["trace", "devices", "ds4_nvtx_ranges", "annotated_operations",
         "annotated_kernel_ms", "annotated_memcpy_ms", "pipeline_gpu_span_ms"],
        [[variant, "0,3,1,2", 20, 2, 20.0, 0.0, span_ms]],
    )
    write_csv(
        variant_dir / "nsys" / "combined-benchmark.csv",
        ["ctx_tokens", "prefill_tokens", "prefill_tps"],
        [[32768, 32768, tps]],
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ds4-indexer-trace-test-") as temp:
        root = Path(temp)
        make_variant(root, "wmma128", 10_000_000, 100.0, 500.0)
        make_variant(root, "wmma64", 7_500_000, 100.5, 502.0)
        subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
        result = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
        comparison = result["comparison"]
        assert abs(comparison["score_kernel_speedup"] - 4 / 3) < 1e-9
        assert abs(comparison["pipeline_span_speedup"] - 100 / 100.5) < 1e-9
        assert abs(comparison["trace_prefill_tps_speedup"] - 502 / 500) < 1e-9
        cells = list(csv.DictReader((root / "score-device-stage.csv").open()))
        assert len(cells) == 4
        assert {row["stage"] for row in cells} == {"0", "1"}
        positions = list(csv.DictReader((root / "score-position-stage.csv").open()))
        assert len(positions) == 2
        assert {row["pos"] for row in positions} == {"32256"}
        assert all(abs(float(row["wmma64_speedup"]) - 4 / 3) < 1e-9
                   for row in positions)
        assert comparison["wmma64_faster_position_stage_cells"] == 2
        assert comparison["final_position"] == 32256
        assert "not reflected" in (root / "summary.md").read_text()
    print("SM75 indexer production trace summarizer test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
