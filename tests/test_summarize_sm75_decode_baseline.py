#!/usr/bin/env python3
"""Tests for the canonical SM75 decode-baseline summarizer."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "speed-bench" / "summarize-sm75-decode-baseline.py"
HEADER = (
    "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,"
    "gen_steady_tokens,gen_steady_tps,kvcache_bytes\n"
)


def write_case(path: Path, pp: int, tg: int, prefill: float, gen: float, first: float, steady: float) -> None:
    path.write_text(
        HEADER
        + f"{pp},{pp},{prefill},{tg},{gen},{first},{tg - 1},{steady},0\n"
    )


class DecodeSummaryTests(unittest.TestCase):
    def run_summary(self, runs: Path, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(runs), str(output)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_subset_two_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for repeat, delta in ((1, 0.0), (2, 2.0)):
                for slot, (case, pp, tg) in enumerate(
                    (("pp512-tg256", 512, 256), ("pp32768-tg512", 32768, 512)), 1
                ):
                    source = root / f"r{repeat}-{case}.csv"
                    write_case(source, pp, tg, 600 + delta, 10 + delta, 100 + delta, 11 + delta)
                    records.append({
                        "repeat": repeat,
                        "slot": slot,
                        "case_id": case,
                        "pp_tokens": pp,
                        "tg_tokens": tg,
                        "ctx_alloc": pp + tg + 1,
                        "csv": source,
                        "log": root / f"r{repeat}-{case}.log",
                    })
            runs = root / "runs.csv"
            with runs.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            output = root / "summary"
            result = self.run_summary(runs, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["case_id"] for row in rows], ["pp512-tg256", "pp32768-tg512"])
            self.assertEqual(rows[0]["median_gen_tps"], "11.000000")
            self.assertEqual(rows[1]["median_gen_steady_tps"], "12.000000")
            self.assertEqual(rows[0]["samples"], "2")
            self.assertIn("SM75 production decode baseline", (output / "summary.md").read_text())

    def test_rejects_wrong_generated_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "bad.csv"
            source.write_text(HEADER + "512,512,600,255,10,100,254,11,0\n")
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--validate-case", str(source),
                    "--case-id", "pp512-tg256", "--pp", "512", "--tg", "256",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("gen_tokens", result.stderr)

    def test_rejects_noncanonical_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "case.csv"
            write_case(source, 1024, 256, 600, 10, 100, 11)
            result = subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--validate-case", str(source),
                    "--case-id", "pp512-tg256", "--pp", "1024", "--tg", "256",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not canonical", result.stderr)


if __name__ == "__main__":
    unittest.main()
