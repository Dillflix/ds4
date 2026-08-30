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
    def test_identifies_prefill_attention_rows_as_necessary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            (root / "nvlink").mkdir()
            production.mkdir()
            statuses = {"attention-off": "passed", "production": "failed"}
            for slot, (variant, status) in enumerate(statuses.items(), 1):
                stem = f"r1-s{slot}-{variant}"
                (production / f"{stem}.result").write_text(
                    f"variant={variant}\nstatus={status}\nexit_status="
                    f"{0 if status == 'passed' else 1}\n"
                    "last_phase=measured-prefill\nlast_event=chunk-start\n"
                )
                (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("prefill attention row split disabled", report)
            self.assertIn("necessary condition", report)

    def test_rejects_power_limit_drift_as_causal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            telemetry = root / "telemetry"
            (root / "nvlink").mkdir()
            production.mkdir()
            telemetry.mkdir()
            stem = "r1-s1-attention-off"
            (production / f"{stem}.result").write_text(
                "variant=attention-off\nstatus=failed\nexit_status=126\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")
            (telemetry / f"{stem}-watch-event.txt").write_text(
                "status=power-limit-drift\nrequired_power_limit_w=250\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("invalid for causal comparison", report)
            self.assertIn("external power-limit writer", report)

    def test_identifies_direct_or_overlap_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            nvlink = root / "nvlink"
            production.mkdir()
            nvlink.mkdir()
            cases = {
                "partner-bounce": ("passed", "decode", "frontier-complete"),
                "bounce-indexer-off": ("failed", "decode", "token-start"),
                "partner-serialized": ("passed", "decode", "frontier-complete"),
                "indexer-off": ("failed", "decode", "token-start"),
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
                    "activation_bytes=32 result_bytes=32 tokens=1 "
                    f"transport={'host-bounce' if 'bounce' in variant else 'peer'} "
                    f"serialized={'yes' if variant == 'partner-serialized' else 'no'}\n"
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
            bounce_row = next(row for row in rows
                              if row["variant"] == "partner-bounce")
            self.assertEqual(bounce_row["pair0_q8_transport"], "host-bounce")
            serialized_row = next(row for row in rows
                                  if row["variant"] == "partner-serialized")
            self.assertEqual(serialized_row["pair0_q8_serialized"], "yes")
            report = (root / "summary.md").read_text()
            self.assertIn("direct P2P/BAR1 traffic", report)
            self.assertIn("instantaneous load", report)

    def test_identifies_indexer_requirement_across_transports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            (root / "nvlink").mkdir()
            production.mkdir()
            statuses = {
                "partner-bounce": "failed",
                "bounce-indexer-off": "passed",
                "partner-serialized": "failed",
                "indexer-off": "passed",
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
            self.assertIn("under both direct-peer and host-bounce", report)
            self.assertIn("indexer path or its interaction", report)

    def test_requires_final_production_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            (root / "nvlink").mkdir()
            production.mkdir()
            stem = "r1-s1-partner-bounce"
            (production / f"{stem}.result").write_text(
                "variant=partner-bounce\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("production control has no durable outcome yet", report)
            self.assertIn("requires the final control", report)


if __name__ == "__main__":
    unittest.main()
