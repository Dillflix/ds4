#!/usr/bin/env python3

import csv
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
SUMMARIZER = REPO / "speed-bench/summarize-sm75-small-bar1-pair-isolation.py"


class SummarizeSmallBar1PairIsolationTest(unittest.TestCase):
    def test_identifies_partner_execution_and_preserved_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            nvlink = root / "nvlink"
            production.mkdir()
            nvlink.mkdir()
            cases = {
                "partner-off": ("passed", "decode", "frontier-complete"),
                "indexer-off": ("failed", "decode", "token-start"),
                "both-off": ("passed", "decode", "frontier-complete"),
                "admission-off": ("passed", "decode", "frontier-complete"),
                "production": ("failed", "measured-prefill", "chunk-complete"),
            }
            for slot, (variant, (status, phase, event)) in enumerate(cases.items(), 1):
                stem = f"r1-s{slot}-{variant}"
                (production / f"{stem}.result").write_text(
                    f"variant={variant}\nstatus={status}\nexit_status="
                    f"{0 if status == 'passed' else 1}\nlast_phase={phase}\n"
                    f"last_event={event}\nlast_current=16\nlast_total=256\n"
                )
                (production / f"{stem}.log").write_text(
                    "ds4: CUDA q8 partner transfer audit event=begin "
                    "home_tier=0 partner_tier=2 calls=64 bytes=4096 "
                    "activation_bytes=32 result_bytes=32 tokens=1\n"
                    "ds4: CUDA decode indexer row audit event=begin layer=2 "
                    "home_tier=0 partner_tier=2 n_comp=1024 transfer_bytes=256\n"
                )
                (nvlink / f"{stem}.log").write_text(
                    "snapshot_utc=2026-08-30T00:00:00Z\nRx0: 1 KiB\n"
                )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            production_row = next(row for row in rows
                                  if row["variant"] == "production")
            self.assertEqual(production_row["last_phase"], "measured-prefill")
            self.assertEqual(production_row["pair0_q8_begin_checkpoint_bytes"], "4096")
            self.assertEqual(production_row["pair0_indexer_begin_bytes"], "256")
            report = (root / "summary.md").read_text()
            self.assertIn("partner execution is necessary", report)
            self.assertIn("peer-cache admission alone is not sufficient", report)

    def test_distinguishes_admission_from_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            (root / "nvlink").mkdir()
            production.mkdir()
            statuses = {
                "partner-off": "failed",
                "indexer-off": "failed",
                "both-off": "failed",
                "admission-off": "passed",
                "production": "failed",
            }
            for slot, (variant, status) in enumerate(statuses.items(), 1):
                stem = f"r1-s{slot}-{variant}"
                (production / f"{stem}.result").write_text(
                    f"variant={variant}\nstatus={status}\nexit_status=1\n"
                    "last_phase=measured-prefill\nlast_event=chunk-start\n"
                    "last_current=2048\nlast_total=32768\n"
                )
                (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("peer cache/scratch admission", report)
            self.assertIn("not decode-indexer rows", report)


if __name__ == "__main__":
    unittest.main()
