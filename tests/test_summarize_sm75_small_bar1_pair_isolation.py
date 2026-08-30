#!/usr/bin/env python3

import csv
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
SUMMARIZER = REPO / "speed-bench/summarize-sm75-small-bar1-pair-isolation.py"


def write_healthy_post(root: pathlib.Path, stem: str) -> None:
    health = root / "health"
    health.mkdir(exist_ok=True)
    (health / f"{stem}-post.log").write_text(
        "GPU 0: ok\nGPU 1: ok\nGPU 2: ok\nGPU 3: ok\n"
    )


class SummarizeSmallBar1PairIsolationTest(unittest.TestCase):
    def test_identifies_prefill_attention_rows_as_necessary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
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
                if status == "passed":
                    write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("prefill attention row split disabled", report)
            self.assertIn("pair-0 decode-indexer splitting", report)
            self.assertIn("Pair-0 prefill indexer splitting also stayed home", report)
            self.assertIn("necessary condition", report)

    def test_rejects_power_limit_drift_as_causal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            telemetry = root / "telemetry"
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

    def test_recovers_completed_attention_off_without_result_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-off"
            (production / f"{stem}.started").write_text(
                "variant=attention-off\nrepeat=1\n"
            )
            (production / f"{stem}.log").write_text("")
            (production / f"{stem}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,487.24,16.89\n"
            )
            (production / f"{stem}-progress.csv").write_text(
                "realtime_sec,realtime_nsec,phase,event,current,total\n"
                "1,0,decode,frontier-complete,256,256\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("completed-no-result", report)
            self.assertIn("older wrapper omitted its final result record", report)
            self.assertIn("mixed prefill split-attention", report)

    def test_validation_failure_is_not_reported_as_device_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-phase-audit\nstatus=validation-failed\n"
                "exit_status=127\nlast_phase=decode\n"
                "last_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("failed its production-path validation", report)
            self.assertIn("invalid for causal inference", report)
            self.assertNotIn("observation boundary for the device loss", report)

    def test_completed_no_result_does_not_override_failed_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            health = root / "health"
            production.mkdir()
            health.mkdir()
            completed = "r1-s1-attention-off"
            (production / f"{completed}.started").write_text(
                "variant=attention-off\nrepeat=1\n"
            )
            (production / f"{completed}.log").write_text("")
            (production / f"{completed}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,487.24,16.89\n"
            )
            (production / f"{completed}-progress.csv").write_text(
                "realtime_sec,realtime_nsec,phase,event,current,total\n"
                "1,0,decode,frontier-complete,256,256\n"
            )
            (health / f"{completed}-post.log").write_text(
                "GPU 0: ok\nGPU 1: ok\nGPU 2: ok\nGPU 3: ok\n"
            )
            failed = "r2-s2-attention-off"
            (production / f"{failed}.result").write_text(
                "variant=attention-off\nstatus=failed\nexit_status=1\n"
            )
            (production / f"{failed}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("Pair 0 still failed", report)
            self.assertNotIn("Pair 0 survived", report)

    def test_reports_targeted_attention_phase_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-phase-audit\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row phase audit event=complete "
                "phase=result-gather kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("All production row-split operations and traffic survived", report)
            self.assertIn("timing/power/traffic envelope", report)
            self.assertIn(
                "complete:result-gather:mixed:layer17:pos512:tokens512:home0:partner2",
                report,
            )

    def test_passed_result_without_post_health_is_not_survival(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-phase-audit\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("passed-unverified-health", report)
            self.assertIn("has no validated final outcome", report)
            self.assertNotIn(
                "All production row-split operations and traffic survived",
                report,
            )

    def test_recovers_started_only_phase_audit_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-phase-audit"
            (production / f"{stem}.started").write_text(
                "variant=attention-phase-audit\nrepeat=1\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row phase audit event=begin "
                "phase=partner-attention kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            (production / f"{stem}-progress.csv").write_text(
                "realtime_sec,realtime_nsec,phase,event,current,total\n"
                "1,0,measured-prefill,chunk-start,512,32768\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("interrupted-no-result", report)
            self.assertIn(
                "begin:partner-attention:mixed:layer17:pos512:tokens512:home0:partner2",
                report,
            )
            self.assertIn("has no validated final outcome", report)

    def test_identifies_direct_or_overlap_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
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
                if status == "passed":
                    write_healthy_post(root, stem)
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
                if status == "passed":
                    write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("under both direct-peer and host-bounce", report)
            self.assertIn("indexer path or its interaction", report)

    def test_requires_final_production_control(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-partner-bounce"
            (production / f"{stem}.result").write_text(
                "variant=partner-bounce\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text("")
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("production control has no durable outcome yet", report)
            self.assertIn("requires the final control", report)


if __name__ == "__main__":
    unittest.main()
