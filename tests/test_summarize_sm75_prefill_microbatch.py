#!/usr/bin/env python3
"""Tests for the paired SM75 prefill-microbatch summarizer."""

from __future__ import annotations

import csv
import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "speed-bench" / "summarize-sm75-prefill-microbatch.py"
SPEC = importlib.util.spec_from_file_location("prefill_microbatch_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SummaryTests(unittest.TestCase):
    def write_bench(self, root: pathlib.Path, name: str, values: dict[int, float]) -> pathlib.Path:
        path = root / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["ctx_tokens", "prefill_tps"])
            writer.writeheader()
            for ctx, tps in values.items():
                writer.writerow({"ctx_tokens": ctx, "prefill_tps": tps})
        return path

    def write_runs(self, root: pathlib.Path, rows: list[dict[str, str]]) -> pathlib.Path:
        path = root / "runs.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["repeat", "variant", "csv"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_paired_ratios_and_medians(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            rows = []
            values = {
                (1, "mb512"): {2048: 100.0, 4096: 80.0},
                (1, "mb1024"): {2048: 120.0, 4096: 84.0},
                (2, "mb512"): {2048: 110.0, 4096: 82.0},
                (2, "mb1024"): {2048: 121.0, 4096: 86.1},
                (3, "mb512"): {2048: 90.0, 4096: 78.0},
                (3, "mb1024"): {2048: 108.0, 4096: 81.9},
            }
            for (repeat, variant), samples in values.items():
                bench = self.write_bench(root, f"r{repeat}-{variant}", samples)
                rows.append({"repeat": str(repeat), "variant": variant, "csv": str(bench)})
            summary = MODULE.summarize(self.write_runs(root, rows))
            self.assertEqual(summary[0]["ctx_tokens"], "2048")
            self.assertEqual(summary[0]["mb512_median_tps"], "100.000")
            self.assertEqual(summary[0]["mb1024_median_tps"], "120.000")
            self.assertEqual(summary[0]["paired_median_speedup"], "1.200000")
            self.assertEqual(summary[1]["paired_median_speedup"], "1.050000")
            self.assertEqual(summary[1]["logits"], "bit-exact")

    def test_rejects_missing_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            bench = self.write_bench(root, "only", {2048: 100.0})
            runs = self.write_runs(
                root,
                [{"repeat": "1", "variant": "mb512", "csv": str(bench)}],
            )
            with self.assertRaisesRegex(ValueError, "incomplete run inventory"):
                MODULE.summarize(runs)

    def test_rejects_different_frontiers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            a = self.write_bench(root, "a", {2048: 100.0})
            b = self.write_bench(root, "b", {4096: 100.0})
            runs = self.write_runs(
                root,
                [
                    {"repeat": "1", "variant": "mb512", "csv": str(a)},
                    {"repeat": "1", "variant": "mb1024", "csv": str(b)},
                ],
            )
            with self.assertRaisesRegex(ValueError, "frontier inventory differs"):
                MODULE.summarize(runs)


if __name__ == "__main__":
    unittest.main()
