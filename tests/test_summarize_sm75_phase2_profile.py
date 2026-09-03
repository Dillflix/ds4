#!/usr/bin/env python3

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "speed-bench" / "summarize-sm75-phase2-profile.py"


def write_rows(path: Path, fieldnames: list[str], rows, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


class Phase2SummaryTest(unittest.TestCase):
    def test_consolidates_layout_aware_prefill_and_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_fields = [
                "label", "devices", "sqlite", "layout", "stage_split",
                "benchmark", "log",
            ]
            decode_map_fields = [
                "label", "pp_tokens", "threshold", "captured_tokens",
                "devices", "sqlite", "layout", "stage_split", "benchmark", "log",
            ]
            prefill_map = []
            decode_map = []
            for layout in ("mixed15", "all43"):
                prefill_bench = root / "input" / f"{layout}-prefill.csv"
                decode_bench = root / "input" / f"{layout}-decode.csv"
                benchmark_fields = [
                    "ctx_tokens", "unused", "prefill_tps", "gen_tokens",
                    "u5", "u6", "u7", "gen_steady_tps",
                ]
                write_rows(prefill_bench, benchmark_fields, [{
                    "ctx_tokens": 32768, "unused": 0, "prefill_tps": 500,
                    "gen_tokens": 0, "u5": 0, "u6": 0, "u7": 0,
                    "gen_steady_tps": 0,
                }])
                write_rows(decode_bench, benchmark_fields, [{
                    "ctx_tokens": 32768, "unused": 0, "prefill_tps": 500,
                    "gen_tokens": 32, "u5": 0, "u6": 0, "u7": 0,
                    "gen_steady_tps": 17,
                }])
                prefill_map.append({
                    "label": f"{layout}-prefill-32k", "devices": "0,3,1,2",
                    "sqlite": "unused", "layout": layout, "stage_split": "22/21",
                    "benchmark": str(prefill_bench), "log": "unused",
                })
                decode_map.append({
                    "label": f"{layout}-decode-32k", "pp_tokens": 32768,
                    "threshold": 1024, "captured_tokens": 16,
                    "devices": "0,3,1,2", "sqlite": "unused", "layout": layout,
                    "stage_split": "22/21", "benchmark": str(decode_bench),
                    "log": "unused",
                })
            write_rows(root / "prefill" / "trace-map.tsv", map_fields, prefill_map, "\t")
            write_rows(root / "decode" / "trace-map.tsv", decode_map_fields, decode_map, "\t")

            prefill_op_fields = [
                "trace", "kind", "duration_ns", "device", "bytes", "name",
                "stage_range", "layer_range", "handoff_range", "partner_range",
                "attention_rows_range", "indexer_rows_range",
            ]
            decode_op_fields = [
                "trace", "kind", "duration_ns", "device", "bytes", "name",
                "category", "layer", "stage_range", "layer_range",
            ]
            prefill_ops = []
            decode_ops = []
            for layout in ("mixed15", "all43"):
                plabel = f"{layout}-prefill-32k"
                dlabel = f"{layout}-decode-32k"
                prefill_ops.extend([
                    {
                        "trace": plabel, "kind": "kernel", "duration_ns": 6000000,
                        "device": 0, "bytes": 0,
                        "name": "moe_gate_up_mid_sm75_q32_tile8_kernel",
                        "stage_range": "ds4/prefill/stage/stage=0/mb=0/tier=0/layers=0-22",
                        "layer_range": "ds4/prefill/layer/layer=0", "handoff_range": "",
                        "partner_range": "", "attention_rows_range": "",
                        "indexer_rows_range": "",
                    },
                    {
                        "trace": plabel, "kind": "memcpy", "duration_ns": 1000000,
                        "device": 3, "bytes": 1048576, "name": "memcpy",
                        "stage_range": "", "layer_range": "", "handoff_range": "",
                        "partner_range": "", "attention_rows_range": "",
                        "indexer_rows_range": "ds4/prefill/indexer-rows/tier=0",
                    },
                ])
                decode_ops.extend([
                    {
                        "trace": dlabel, "kind": "kernel", "duration_ns": 2000000,
                        "device": 0, "bytes": 0,
                        "name": "q8_K_quantize_sm75_native_kernel", "category": "ffn",
                        "layer": 0, "stage_range": "", "layer_range": "",
                    },
                    {
                        "trace": dlabel, "kind": "kernel", "duration_ns": 3000000,
                        "device": 0, "bytes": 0,
                        "name": "matmul_f16_pair_compressor_store_ordered_chunks_kernel",
                        "category": "attention", "layer": 0,
                        "stage_range": "", "layer_range": "",
                    },
                    {
                        "trace": dlabel, "kind": "kernel", "duration_ns": 4000000,
                        "device": 0, "bytes": 0,
                        "name": (
                            "moe_gate_up_mid_decode_sm75_q3a4_tile32_k4_"
                            "prefetch_owned_kernel" if layout == "all43" else
                            "moe_gate_up_mid_decode_sm75_q4_32_tile32_owned_kernel"
                        ),
                        "category": "ffn", "layer": 0,
                        "stage_range": "", "layer_range": "",
                    },
                ])
            write_rows(root / "prefill" / "operation-attribution.csv", prefill_op_fields, prefill_ops)
            write_rows(root / "decode" / "summary" / "operation-attribution.csv", decode_op_fields, decode_ops)

            stage_fields = ["trace", "stage", "sum_gpu_envelope_ms"]
            stages = []
            for layout in ("mixed15", "all43"):
                trace = f"{layout}-prefill-32k"
                stages.extend([
                    {"trace": trace, "stage": 0, "sum_gpu_envelope_ms": 100},
                    {"trace": trace, "stage": 1, "sum_gpu_envelope_ms": 120},
                ])
            write_rows(root / "prefill" / "stage-device-summary.csv", stage_fields, stages)
            write_rows(root / "prefill" / "trace-summary.csv", [
                "trace", "pipeline_gpu_span_ms",
            ], [
                {"trace": f"{layout}-prefill-32k", "pipeline_gpu_span_ms": 200}
                for layout in ("mixed15", "all43")
            ])
            write_rows(root / "decode" / "summary" / "trace-summary.csv", [
                "trace", "gpu_envelope_ms",
            ], [
                {"trace": f"{layout}-decode-32k", "gpu_envelope_ms": 900}
                for layout in ("mixed15", "all43")
            ])
            write_rows(root / "ncu-decode-weight" / "summary.csv", [
                "scenario", "duration_us", "dram_peak_pct", "l2_hit_pct",
                "long_scoreboard_ratio", "achieved_occupancy_pct",
                "evidence_class",
            ], [{
                "scenario": "q8-native-quantize", "duration_us": 4.5,
                "dram_peak_pct": 12.0, "l2_hit_pct": 80.0,
                "long_scoreboard_ratio": 1.25,
                "achieved_occupancy_pct": 50.0,
                "evidence_class": "not DRAM-saturated",
            }])

            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(root)],
                check=False, text=True, capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (root / "summary" / "prefill-family-summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                family_rows = list(csv.DictReader(handle))
            by_trace = {(row["trace"], row["family"]) for row in family_rows}
            self.assertIn(("mixed15-prefill-32k", "q4_32_gate_up"), by_trace)
            self.assertIn(("all43-prefill-32k", "q3a4_gate_up"), by_trace)
            self.assertIn(
                ("mixed15-prefill-32k", "indexer_row_split_transfer"), by_trace
            )
            with (root / "summary" / "decode-family-summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                decode_rows = list(csv.DictReader(handle))
            decode_families = {row["family"] for row in decode_rows}
            self.assertIn("direct_native_q8_quantize", decode_families)
            self.assertIn("compressor_projection_state_fused", decode_families)
            decode_by_trace = {
                (row["trace"], row["family"]) for row in decode_rows
            }
            self.assertIn(("mixed15-decode-32k", "q4_32_gate_up"), decode_by_trace)
            self.assertIn(("all43-decode-32k", "q3a4_gate_up"), decode_by_trace)
            with (root / "summary" / "phase3-target-evidence.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                target_rows = list(csv.DictReader(handle))
            self.assertEqual(len(target_rows), 2 * 2 * 5)
            compact_rows = [
                row for row in target_rows
                if row["target"] == "compact-owner-local-routed-slots"
            ]
            self.assertTrue(compact_rows)
            self.assertTrue(all(float(row["kernel_ms"]) == 0.0 for row in compact_rows))
            self.assertIn("pair-0 attention rows off", completed.stdout)
            self.assertIn("Bounded decode-kernel diagnosis", completed.stdout)
            self.assertIn("q8-native-quantize", completed.stdout)


if __name__ == "__main__":
    unittest.main()
