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


def write_unhealthy_post(root: pathlib.Path, stem: str, device: int = 1) -> None:
    health = root / "health"
    health.mkdir(exist_ok=True)
    (health / f"{stem}-post.log").write_text(
        "".join(
            f"GPU {index}: {'GPU is lost' if index == device else 'ok'}\n"
            for index in range(4)
        )
    )


def write_lost_watch(root: pathlib.Path, stem: str, lost_devices: str) -> None:
    telemetry = root / "telemetry"
    telemetry.mkdir(exist_ok=True)
    (telemetry / f"{stem}-watch-event.txt").write_text(
        "status=lost-device-detected\n"
        f"lost_devices={lost_devices}\n"
    )


def complete_row_boundary_log() -> str:
    end_identity = "kind=mixed layer=17 pos=512 tokens=512 home=0 partner=2"
    entry_identity = "kind=mixed layer=18 pos=512 tokens=512 home=0 partner=2"
    lines = [
        f"ds4: CUDA prefill attention row end fence event=begin target=pair {end_identity}",
        f"ds4: CUDA prefill attention row end fence event=complete target=partner {end_identity}",
        f"ds4: CUDA prefill attention row end fence event=complete target=home {end_identity}",
        f"ds4: CUDA prefill attention row end fence event=complete target=pair {end_identity}",
        f"ds4: CUDA prefill attention row audit dispatch=split {end_identity} q_bytes=1 result_bytes=1",
        f"ds4: CUDA prefill attention row entry fence event=begin target=pair {entry_identity}",
        f"ds4: CUDA prefill attention row entry fence event=complete target=partner {entry_identity}",
        f"ds4: CUDA prefill attention row entry fence event=complete target=home {entry_identity}",
        f"ds4: CUDA prefill attention row entry fence event=complete target=pair {entry_identity}",
    ]
    for phase in (
            "query-copy", "partner-attention", "home-attention", "result-gather"):
        lines.extend([
            f"ds4: CUDA prefill attention row phase audit event=begin phase={phase} "
            f"{entry_identity}",
            f"ds4: CUDA prefill attention row phase audit event=complete phase={phase} "
            f"{entry_identity}",
        ])
    lines.append(
        f"ds4: CUDA prefill attention row audit dispatch=split {entry_identity} "
        "q_bytes=1 result_bytes=1"
    )
    return "\n".join(lines) + "\n"


def q8_phase_marker(
        sequence: int, event: str, stage: str, *, home_tier: int = 0,
        home_device: int = 0, partner_tier: int = 2,
        partner_device: int = 1, cuda_error: str = "none") -> str:
    return (
        "ds4: CUDA q8 partner phase audit "
        f"sequence={sequence} event={event} stage={stage} "
        "binding_label=tensor:blk.14.attn_output_b.weight "
        "passed_label=attn_output_b weight_offset=144000 "
        f"home_tier={home_tier} home_device={home_device} "
        f"partner_tier={partner_tier} partner_device={partner_device} "
        "tokens=512 in=8192 out=4096 transfer_bytes=1048576 "
        f"result_bytes=8388608 cuda_error={cuda_error}\n"
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
                else:
                    write_unhealthy_post(root, stem)

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

    def test_interrupted_prior_run_without_device_loss_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=interrupted-prior-run\n"
                "exit_status=125\nlast_phase=measured-prefill\n"
                "last_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "interrupted-prior-run")
            report = (root / "summary.md").read_text()
            self.assertIn("production control has no verified outcome", report)
            self.assertIn("fresh directory", report)

    def test_generic_failed_result_without_device_loss_is_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "failed-unverified")
            report = (root / "summary.md").read_text()
            self.assertIn("production control has no verified outcome", report)

    def test_lost_device_watch_corroborates_interrupted_prior_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            telemetry = root / "telemetry"
            production.mkdir()
            telemetry.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=interrupted-prior-run\n"
                "exit_status=125\nlast_phase=measured-prefill\n"
                "last_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")
            (telemetry / f"{stem}-watch-event.txt").write_text(
                "status=lost-device-detected\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["status"], "interrupted-prior-run-device-loss"
            )

    def test_lost_device_watch_corroborates_started_only_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            telemetry = root / "telemetry"
            production.mkdir()
            telemetry.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.started").write_text(
                "variant=production\nrepeat=1\n"
            )
            (production / f"{stem}.log").write_text("")
            (telemetry / f"{stem}-watch-event.txt").write_text(
                "status=lost-device-detected\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["status"], "interrupted-no-result-device-loss"
            )

    def test_unhealthy_post_corroborates_interrupted_prior_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            health = root / "health"
            production.mkdir()
            health.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=interrupted-prior-run\n"
                "exit_status=125\nlast_phase=measured-prefill\n"
                "last_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")
            (health / f"{stem}-post.log").write_text(
                "GPU 0: ok\nGPU 1: GPU is lost\nGPU 2: ok\nGPU 3: ok\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["status"], "interrupted-prior-run-device-loss"
            )
            self.assertEqual(row["post_health"], "unhealthy")

    def test_empty_post_snapshot_does_not_imply_device_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            health = root / "health"
            production.mkdir()
            health.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=interrupted-prior-run\n"
                "exit_status=125\nlast_phase=measured-prefill\n"
                "last_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")
            (health / f"{stem}-post.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "interrupted-prior-run")
            self.assertEqual(row["post_health"], "unverified")

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
            write_unhealthy_post(root, failed)

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

    def test_watch_marker_invalidates_passed_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            telemetry = root / "telemetry"
            production.mkdir()
            telemetry.mkdir()
            stem = "r1-s1-attention-query-dst"
            (production / f"{stem}.result").write_text(
                "variant=attention-query-dst\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text("")
            write_healthy_post(root, stem)
            (telemetry / f"{stem}-watch-event.txt").write_text(
                "status=foreign-compute-process\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "passed-invalidated-watch")
            report = (root / "summary.md").read_text()
            self.assertIn("unexpected GPU compute process", report)
            self.assertIn("invalid for causal comparison", report)

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

    def test_reports_passed_attention_end_fence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-end-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-end-fence\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=complete "
                "target=pair kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("full production path survived", report)
            self.assertIn("not a proposal to disable row splitting", report)
            self.assertIn(
                "complete:pair:mixed:layer21:pos512:tokens512:home0:partner2",
                report,
            )

    def test_recovers_started_only_attention_end_fence_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-end-fence"
            (production / f"{stem}.started").write_text(
                "variant=attention-end-fence\nrepeat=1\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=begin "
                "target=pair kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row end fence event=complete "
                "target=partner kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row end fence event=sync-failed "
                "target=home kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row end fence event=failed "
                "target=home kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)
            (production / f"{stem}-progress.csv").write_text(
                "realtime_sec,realtime_nsec,phase,event,current,total\n"
                "1,0,measured-prefill,prefill_chunk,0,32768\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("interrupted-no-result", report)
            self.assertIn(
                "failed:home:mixed:layer21:pos512:tokens512:home0:partner2",
                report,
            )
            self.assertIn("partner end-fence completed", report)
            self.assertIn("following home boundary failed", report)

    def test_distinguishes_failure_after_completed_attention_end_fence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-end-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-end-fence\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=complete "
                "target=pair kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("completed on both partner and home", report)
            self.assertIn("host-confirmed complete", report)
            self.assertIn("first subsequent CUDA observation lies", report)
            self.assertIn("delayed physical fault", report)

    def test_distinguishes_partner_attention_end_fence_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-end-fence"
            (production / f"{stem}.result").write_text(
                "variant=attention-end-fence\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=sync-failed "
                "target=partner kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row end fence event=failed "
                "target=partner kind=mixed layer=21 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("partner synchronization was the first failing", report)
            self.assertIn("Home synchronization was deliberately not attempted", report)

    def test_layer_boundary_audit_localizes_failure_before_l18_query_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-row-boundary-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
                "last_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=complete "
                "target=pair kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row entry fence event=begin "
                "target=pair kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row entry fence event=sync-failed "
                "target=partner kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row entry fence event=failed "
                "target=partner kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn(
                "failed:partner:mixed:layer18:pos512:tokens512:home0:partner2",
                report,
            )
            self.assertIn("before layer-18 query copy was submitted", report)
            self.assertIn("layer-17 tail/MoE/Q8 work", report)
            self.assertIn("does not prove a layer-18 kernel bug", report)

    def test_layer_boundary_audit_reports_failure_inside_l18_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-row-boundary-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
                "last_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=complete "
                "target=pair kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row entry fence event=complete "
                "target=pair kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row phase audit event=sync-failed "
                "phase=partner-attention kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn(
                "sync-failed:partner-attention:mixed:layer18:pos512:tokens512:"
                "home0:partner2",
                report,
            )
            self.assertIn("layer-18 row-launch/pre-query fence completed", report)
            self.assertIn("first failure-class marker is inside", report)
            self.assertIn("not by itself proof of a layer-specific", report)

    def test_reports_passed_layer_boundary_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-row-boundary-audit\nstatus=passed\n"
                "exit_status=0\nlast_phase=decode\n"
                "last_event=frontier-complete\n"
            )
            (production / f"{stem}.log").write_text(complete_row_boundary_log())
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["attention_entry_fence_last"],
                "complete:pair:mixed:layer18:pos512:tokens512:home0:partner2",
            )
            self.assertEqual(row["attention_row_boundary_marker_state"], "complete")
            report = (root / "summary.md").read_text()
            self.assertIn("healthy post-run snapshot passed", report)
            self.assertIn("immediately before layer 18 query copy", report)
            self.assertIn("does not identify a layer-specific software defect", report)

    def test_recovers_completed_boundary_audit_without_result_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.started").write_text(
                "variant=attention-row-boundary-audit\n"
            )
            (production / f"{stem}.log").write_text(complete_row_boundary_log())
            (production / f"{stem}.csv").write_text(
                "pp_tokens,prefill_tps,gen_tps\n32768,600.0,16.0\n"
            )
            (production / f"{stem}-progress.csv").write_text(
                "realtime_sec,realtime_nsec,phase,event,current,total\n"
                "1,0,decode,frontier-complete,256,256\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["status"], "completed-no-result")
            report = (root / "summary.md").read_text()
            self.assertIn("wrapper omitted its final result record", report)
            self.assertIn("healthy post-run snapshot passed", report)

    def test_ctrl_c_after_complete_phases_is_not_called_gpu_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-row-boundary-audit\nstatus=failed\n"
                "exit_status=130\nlast_phase=interrupted\nlast_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(complete_row_boundary_log())

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("no lost-device watcher record", report)
            self.assertIn("not evidence of a GPU failure", report)
            self.assertNotIn("first observed error is downstream", report)

    def test_lost_device_after_complete_phases_is_downstream_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            telemetry = root / "telemetry"
            telemetry.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-row-boundary-audit\nstatus=failed\n"
                "exit_status=143\nlast_phase=measured-prefill\n"
                "last_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(complete_row_boundary_log())
            (telemetry / f"{stem}-watch-event.txt").write_text(
                "status=lost-device-detected\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("independently detected device loss", report)
            self.assertIn("downstream of layer-18 attention completion", report)

    def test_validates_exact_row_boundary_marker_state_machine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = pathlib.Path(temporary) / "boundary.log"
            complete = complete_row_boundary_log()
            command = [
                sys.executable,
                str(SUMMARIZER),
                "--validate-attention-row-boundary-log",
                str(log_path),
                "17",
                "18",
                "512",
            ]
            log_path.write_text(complete)
            subprocess.run(command, check=True)

            mutations = {
                "wrong-kind": complete.replace(
                    "phase audit event=begin phase=query-copy kind=mixed",
                    "phase audit event=begin phase=query-copy kind=indexed",
                    1,
                ),
                "wrong-position": complete.replace(
                    "entry fence event=begin target=pair kind=mixed layer=18 pos=512",
                    "entry fence event=begin target=pair kind=mixed layer=18 pos=1024",
                    1,
                ),
                "unexpected-l17-entry": complete.replace(
                    "prefill attention row audit dispatch=split kind=mixed layer=17",
                    "prefill attention row entry fence event=begin target=pair "
                    "kind=mixed layer=17 pos=512 tokens=512 home=0 partner=2\n"
                    "ds4: CUDA prefill attention row audit dispatch=split "
                    "kind=mixed layer=17",
                    1,
                ),
                "missing-l18-dispatch": complete.rsplit("\n", 2)[0] + "\n",
                "duplicate-phase": complete.replace(
                    "ds4: CUDA prefill attention row phase audit event=complete "
                    "phase=query-copy",
                    "ds4: CUDA prefill attention row phase audit event=complete "
                    "phase=query-copy kind=mixed layer=18 pos=512 tokens=512 "
                    "home=0 partner=2\n"
                    "ds4: CUDA prefill attention row phase audit event=complete "
                    "phase=query-copy",
                    1,
                ),
            }
            for name, mutated in mutations.items():
                with self.subTest(name=name):
                    log_path.write_text(mutated)
                    result = subprocess.run(
                        command, text=True, capture_output=True, check=False
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("marker sequence is", result.stderr)

    def test_first_phase_failure_survives_secondary_cleanup_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-row-boundary-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-row-boundary-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
                "last_event=prefill_chunk\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row end fence event=complete "
                "target=pair kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row entry fence event=complete "
                "target=pair kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row phase audit event=sync-failed "
                "phase=query-copy kind=mixed layer=18 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row phase audit "
                "event=device-switch-failed phase=home-attention kind=mixed "
                "layer=18 pos=512 tokens=512 home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["attention_audit_first_failure"],
                "phase-audit:sync-failed:query-copy:mixed:layer18:pos512:"
                "tokens512:home0:partner2",
            )
            self.assertEqual(
                row["attention_phase_audit_last"],
                "device-switch-failed:home-attention:mixed:layer18:pos512:"
                "tokens512:home0:partner2",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("phase-audit:sync-failed:query-copy", report)
            self.assertIn("secondary cleanup failure cannot overwrite", report)

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
                else:
                    write_unhealthy_post(root, stem)
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
                else:
                    write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("under both direct-peer and host-bounce", report)
            self.assertIn("indexer path or its interaction", report)

    def test_reports_query_destination_stream_survival(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            variants = (
                ("attention-query-dst", "passed"),
                ("attention-gather-dst", "failed"),
                ("attention-both-dst", "failed"),
                ("production", "failed"),
            )
            for slot, (variant, status) in enumerate(variants, 1):
                stem = f"r1-s{slot}-{variant}"
                (production / f"{stem}.result").write_text(
                    f"variant={variant}\nstatus={status}\nexit_status="
                    f"{'0' if status == 'passed' else '1'}\n"
                    "last_phase=decode\nlast_event=frontier-complete\n"
                )
                (production / f"{stem}.log").write_text(
                    "ds4: CUDA prefill attention row audit dispatch=split "
                    "kind=mixed layer=17 pos=512 tokens=512 home=0 partner=2 "
                    "q_bytes=65536 result_bytes=65536 "
                    "query_copy_stream=destination gather_copy_stream=source\n"
                    if variant == "attention-query-dst" else ""
                )
                if status == "passed":
                    write_healthy_post(root, stem)
                else:
                    write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("Query-copy destination scheduling passed", report)
            self.assertIn("scheduling/peer-access axis", report)
            self.assertIn("not identification of a physical copy engine", report)
            with (root / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            query_row = next(
                row for row in rows
                if row["variant"] == "attention-query-dst"
            )
            self.assertEqual(
                query_row["pair0_attention_query_copy_schedule"], "destination"
            )
            self.assertEqual(
                query_row["pair0_attention_gather_copy_schedule"], "source"
            )

    def test_reports_attention_host_bounce_transport_and_survival(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-host-bounce\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,575.00,15.00\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=raw event=complete "
                "bytes=33554432\n"
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row audit dispatch=split "
                "kind=indexed layer=27 pos=16384 tokens=512 home=0 partner=2 "
                "q_bytes=33554432 result_bytes=33554432 "
                "query_copy_stream=source gather_copy_stream=source "
                "query_copy_transport=host-bounce "
                "gather_copy_transport=host-bounce "
                "topk_copy_transport=partner-local\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["pair0_attention_query_copy_schedule"], "source")
            self.assertEqual(row["pair0_attention_gather_copy_schedule"], "source")
            self.assertEqual(
                row["pair0_attention_query_copy_transport"], "host-bounce"
            )
            self.assertEqual(
                row["pair0_attention_gather_copy_transport"], "host-bounce"
            )
            self.assertEqual(
                row["pair0_attention_cache_copy_transport"], "host-bounce"
            )
            self.assertEqual(
                row["pair0_attention_topk_copy_transport"], "partner-local"
            )
            self.assertEqual(
                row["attention_host_bounce_checkpoint"],
                "complete:mixed:layer17:pos512:tokens512:home0:partner2",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("attention-owned query, mirrored-cache", report)
            self.assertIn("top-k remained partner-local", report)
            self.assertIn("no cross-device top-k payload transfer", report)
            self.assertIn("direct attention P2P/BAR1 path", report)
            self.assertIn("not by itself a power-matched proof", report)

    def test_reports_attention_host_bounce_device_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-host-bounce\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=mixed layer=0 pos=0 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("no durable measured pos=512", report)
            self.assertIn("real failure but inconclusive", report)
            self.assertNotIn("not a necessary trigger", report)

    def test_reports_attention_host_bounce_device_loss_after_checkpoint(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-host-bounce\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=begin kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["attention_host_bounce_checkpoint"],
                "begin:mixed:layer17:pos512:tokens512:home0:partner2",
            )
            self.assertEqual(row["lost_devices"], "1@00000000:03:00.0")
            report = (root / "summary.md").read_text()
            self.assertIn("durable measured pos=512 begin/failed checkpoint", report)
            self.assertIn("forced-host-bounce submission boundary", report)
            self.assertIn(
                "does not claim that the pos=512 attention operation completed",
                report,
            )
            self.assertIn("No top-k transfer was observed", report)
            self.assertIn("not proof that none occurred", report)
            self.assertIn(
                "not necessary for the observed pair-0 device loss", report
            )
            self.assertIn("partner attention execution", report)
            self.assertIn("pair-1 direct traffic", report)

    def test_reports_completed_attention_host_bounce_before_device_loss(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-host-bounce\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=mixed layer=2 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("event=complete checkpoint", report)
            self.assertIn(
                "measured layer at pos=512 completed end-to-end", report
            )
            self.assertNotIn(
                "does not claim that the pos=512 attention operation completed",
                report,
            )
            self.assertIn("No top-k transfer was observed", report)
            self.assertIn("not proof that none occurred", report)

    def test_combined_host_bounce_failure_requires_both_measured_routes(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-host-bounce\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA q8 partner transfer audit event=begin "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096 "
                "activation_bytes=32 result_bytes=32 tokens=512 "
                "transport=host-bounce serialized=no\n"
                "ds4: CUDA q8 partner transfer audit event=complete "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=raw event=complete bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=attn-comp event=complete bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=index event=complete bytes=4096\n"
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=mixed layer=2 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row audit dispatch=split "
                "kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2 "
                "query_copy_stream=source gather_copy_stream=source "
                "query_copy_transport=host-bounce "
                "gather_copy_transport=host-bounce "
                "topk_copy_transport=none\n"
                "ds4: CUDA default-stream bounce h2d failed: "
                "unspecified launch failure\n"
                "ds4: CUDA prefill attention host-bounce failure "
                "class=result-gather stage=copy kind=mixed layer=16 pos=512 "
                "tokens=512 source_tier=2 source_device=1 "
                "destination_tier=0 destination_device=0 bytes=33554432\n"
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["variant"], "attention-q8-host-bounce")
            self.assertEqual(
                row["pair0_q8_complete_checkpoint_calls"], "128"
            )
            self.assertEqual(
                row["pair0_attention_cache_host_bounce_classes"],
                "attn-comp,index,raw",
            )
            self.assertEqual(
                row["host_bounce_failure_context"],
                "cuda_phase=h2d class=result-gather stage=copy kind=mixed "
                "layer=16 pos=512 "
                "tokens=512 source_tier=2 source_device=1 "
                "destination_tier=0 destination_device=0 bytes=33554432",
            )
            report = (root / "summary.md").read_text()
            self.assertIn(
                "direct pair-0 production attention-owned and Q8-partner payload "
                "transport is unnecessary",
                report,
            )
            self.assertIn("Pair-1 traffic remained direct peer", report)
            self.assertIn("was not removed", report)

    def test_combined_host_bounce_warmup_only_is_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-host-bounce\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA q8 partner transfer audit event=begin "
                "home_tier=0 partner_tier=2 calls=64 bytes=2048 "
                "activation_bytes=32 result_bytes=32 tokens=512 "
                "transport=host-bounce serialized=no\n"
                "ds4: CUDA q8 partner transfer audit event=complete "
                "home_tier=0 partner_tier=2 calls=64 bytes=2048\n"
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=mixed layer=0 pos=0 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("warmup evidence only", report)
            self.assertIn(
                "Do not infer that the cut pair-0 attention/Q8 payload routes were "
                "unnecessary",
                report,
            )
            self.assertIn("Pair-1 traffic retained direct peer transport", report)
            self.assertIn("no pair-0 indexer dispatch was durably logged", report)
            self.assertNotIn("those direct indexer routes", report)

    def test_combined_host_bounce_underloaded_completion_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-host-bounce\n"
                "status=inconclusive-underloaded\nexit_status=128\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,310.00,15.00\n"
            )
            (production / f"{stem}.log").write_text("")
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("inconclusive-underloaded", report)
            self.assertIn("below the required 500 prefill tok/s", report)
            self.assertIn("cannot show", report)
            self.assertIn("Pair-1 traffic remained direct", report)

    def test_combined_host_bounce_verified_full_load_pass_is_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-host-bounce\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,575.00,15.00\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA q8 partner transfer audit event=begin "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096 "
                "activation_bytes=32 result_bytes=32 tokens=512 "
                "transport=host-bounce serialized=no\n"
                "ds4: CUDA q8 partner transfer audit event=complete "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=raw event=complete bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=attn-comp event=complete "
                "bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=index event=complete bytes=4096\n"
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=indexed layer=27 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row audit dispatch=split "
                "kind=indexed layer=27 pos=512 tokens=512 home=0 partner=2 "
                "query_copy_stream=source gather_copy_stream=source "
                "query_copy_transport=host-bounce "
                "gather_copy_transport=host-bounce "
                "topk_copy_transport=partner-local\n"
                "ds4: CUDA decode indexer row audit event=begin layer=27 "
                "home_tier=0 partner_tier=2 n_comp=7936 "
                "transfer_bytes=50176\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("durable measured completion evidence", report)
            self.assertIn("indexer transport retained its direct-peer configuration", report)
            self.assertIn("durably logged 1 begun dispatch", report)
            self.assertIn("not by itself a", report)

    def test_combined_pass_markers_below_load_floor_are_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-host-bounce\nstatus=passed\nexit_status=0\n"
                "last_phase=decode\nlast_event=frontier-complete\n"
            )
            (production / f"{stem}.csv").write_text(
                "ctx_tokens,prefill_tps,gen_tps\n32768,499.00,15.00\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA q8 partner transfer audit event=begin "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096 "
                "activation_bytes=32 result_bytes=32 tokens=512 "
                "transport=host-bounce serialized=no\n"
                "ds4: CUDA q8 partner transfer audit event=complete "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=raw event=complete bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=attn-comp event=complete "
                "bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=index event=complete bytes=4096\n"
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=indexed layer=27 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row audit dispatch=split "
                "kind=indexed layer=27 pos=512 tokens=512 home=0 partner=2 "
                "query_copy_stream=source gather_copy_stream=source "
                "query_copy_transport=host-bounce "
                "gather_copy_transport=host-bounce "
                "topk_copy_transport=partner-local\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("recorded as passed", report)
            self.assertIn("required >=500 prefill tok/s load", report)
            self.assertIn("Treat it as inconclusive", report)
            self.assertNotIn("This implicates one of the removed", report)

    def test_combined_host_bounce_pair1_only_loss_is_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-host-bounce\nstatus=failed-device-loss\n"
                "exit_status=1\nlast_phase=measured-prefill\n"
                "last_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA q8 partner transfer audit event=begin "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096 "
                "activation_bytes=32 result_bytes=32 tokens=512 "
                "transport=host-bounce serialized=no\n"
                "ds4: CUDA q8 partner transfer audit event=complete "
                "home_tier=0 partner_tier=2 calls=128 bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=raw event=complete bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=attn-comp event=complete "
                "bytes=4096\n"
                "ds4: CUDA prefill attention cache mirror transport=host-bounce "
                "home_tier=0 partner_tier=2 class=index event=complete bytes=4096\n"
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=complete kind=mixed layer=2 pos=512 tokens=512 "
                "home=0 partner=2\n"
                "ds4: CUDA prefill attention row audit dispatch=split "
                "kind=mixed layer=2 pos=512 tokens=512 home=0 partner=2 "
                "query_copy_stream=source gather_copy_stream=source "
                "query_copy_transport=host-bounce "
                "gather_copy_transport=host-bounce topk_copy_transport=none\n"
            )
            write_unhealthy_post(root, stem, device=2)
            write_lost_watch(root, stem, "2@00000000:81:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("does not identify physical GPU 0 or 1", report)
            self.assertIn("real failure evidence", report)
            self.assertIn("cannot show", report)

    def test_q8_phase_audit_compute_sync_failure_is_partner_side(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=untimed-warmup\n"
            )
            (production / f"{stem}.log").write_text(
                q8_phase_marker(41, "begin", "activation-prepare")
                + q8_phase_marker(41, "activation-complete", "activation-copy")
                + q8_phase_marker(41, "compute-submitted", "compute")
                + q8_phase_marker(
                    41, "compute-sync-failed", "compute-sync",
                    cuda_error="unspecified launch failure",
                )
                + q8_phase_marker(
                    41, "home-restored-after-failure", "recovery"
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "0@00000000:02:00.0,1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertIn(
                "sequence=41 event=compute-sync-failed stage=compute-sync",
                row["q8_phase_audit_last_checkpoint"],
            )
            self.assertEqual(
                row["q8_phase_audit_last_checkpoint"],
                row["q8_phase_audit_first_failure"],
            )
            self.assertEqual(
                row["q8_phase_audit_classification"],
                "partner-compute-or-earlier-async-partner-work",
            )
            self.assertIn(
                "cuda_error=unspecified launch failure",
                row["q8_phase_audit_first_failure"],
            )
            report = (root / "summary.md").read_text()
            self.assertIn(
                "partner compute or earlier asynchronous partner-side work", report
            )
            self.assertIn("before the result gather was attempted", report)
            self.assertIn("not in the result D2H host-bounce copy", report)
            self.assertIn("no pair-0 indexer dispatch was durably logged", report)

    def test_q8_phase_audit_compute_complete_then_d2h_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=untimed-warmup\n"
            )
            (production / f"{stem}.log").write_text(
                q8_phase_marker(42, "begin", "activation-prepare")
                + q8_phase_marker(42, "activation-complete", "activation-copy")
                + q8_phase_marker(42, "compute-submitted", "compute")
                + q8_phase_marker(42, "compute-complete", "compute-sync")
                + "ds4: CUDA default-stream bounce d2h failed: "
                  "unspecified launch failure\n"
                + q8_phase_marker(
                    42, "result-copy-failed", "result-gather",
                    cuda_error="unspecified launch failure",
                )
                + "ds4: CUDA q8 partner host-bounce failure "
                  "class=result-gather stage=copy source_tier=2 "
                  "source_device=1 destination_tier=0 destination_device=0 "
                  "bytes=8388608 tokens=512 in=8192 out=4096 "
                  "label=attn_output_b\n"
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "0@00000000:02:00.0,1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertIn(
                "sequence=42 event=result-copy-failed stage=result-gather",
                row["q8_phase_audit_last_checkpoint"],
            )
            self.assertEqual(
                row["q8_phase_audit_last_checkpoint"],
                row["q8_phase_audit_first_failure"],
            )
            self.assertIn(
                "cuda_phase=d2h", row["q8_phase_audit_first_failure"]
            )
            self.assertEqual(
                row["q8_phase_audit_classification"],
                "result-d2h-pcie-host-bounce-transfer",
            )
            report = (root / "summary.md").read_text()
            self.assertIn("completed the partner compute synchronization", report)
            self.assertIn("partner-to-host PCIe leg", report)
            self.assertIn("rather than to the audited partner compute", report)

    def test_q8_phase_audit_without_same_sequence_compute_complete_is_inconclusive(
            self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-q8-phase-audit"
            (production / f"{stem}.result").write_text(
                "variant=attention-q8-phase-audit\nstatus=failed\n"
                "exit_status=1\nlast_phase=untimed-warmup\n"
            )
            (production / f"{stem}.log").write_text(
                q8_phase_marker(8, "compute-complete", "compute-sync")
                + q8_phase_marker(
                    9, "result-copy-failed", "result-gather",
                    cuda_error="unspecified launch failure",
                )
            )
            write_unhealthy_post(root, stem)
            write_lost_watch(root, stem, "1@00000000:03:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["q8_phase_audit_classification"], "inconclusive")
            report = (root / "summary.md").read_text()
            self.assertIn(
                "without a marker sequence that separates partner compute", report
            )
            self.assertNotIn("partner-to-host PCIe leg", report)

    def test_q8_phase_audit_variant_sorts_after_combined_host_bounce(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            for slot, variant in enumerate((
                    "attention-q8-phase-audit", "attention-q8-host-bounce"), 1):
                stem = f"r1-s{slot}-{variant}"
                (production / f"{stem}.result").write_text(
                    f"variant={variant}\nstatus=validation-failed\nexit_status=127\n"
                )
                (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["variant"] for row in rows],
                ["attention-q8-host-bounce", "attention-q8-phase-audit"],
            )

    def test_invalid_outcome_outranks_underloaded_across_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            for repeat, status in ((1, "inconclusive-underloaded"),
                                   (2, "validation-failed")):
                stem = f"r{repeat}-s{repeat}-attention-q8-host-bounce"
                (production / f"{stem}.result").write_text(
                    "variant=attention-q8-host-bounce\n"
                    f"status={status}\nexit_status=127\n"
                )
                (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("failed its production-path", report)
            self.assertNotIn("inconclusive-underloaded: survival", report)

    def test_parses_q8_activation_host_bounce_failure_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=run-failed-unverified\n"
                "exit_status=1\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA default-stream bounce d2h failed: "
                "unspecified launch failure\n"
                "ds4: CUDA q8 partner host-bounce failure "
                "class=activation stage=copy source_tier=0 source_device=0 "
                "destination_tier=2 destination_device=1 bytes=1048576 "
                "tokens=512 in=512 out=32768 label=q8_0\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["host_bounce_failure_context"],
                "cuda_phase=d2h class=activation stage=copy source_tier=0 "
                "source_device=0 destination_tier=2 destination_device=1 "
                "bytes=1048576 tokens=512 in=512 out=32768 label=q8_0",
            )

    def test_parses_attention_cache_host_bounce_failure_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=run-failed-unverified\n"
                "exit_status=1\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA default-stream bounce h2d failed: "
                "unspecified launch failure\n"
                "ds4: CUDA prefill attention host-bounce failure "
                "class=cache-raw stage=copy layer=16 row=512 "
                "source_tier=0 source_device=0 destination_tier=2 "
                "destination_device=1 bytes=1048576\n"
            )

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["host_bounce_failure_context"],
                "cuda_phase=h2d class=cache-raw stage=copy layer=16 row=512 "
                "source_tier=0 source_device=0 destination_tier=2 "
                "destination_device=1 bytes=1048576",
            )

    def test_does_not_attribute_pair1_loss_to_pair0_host_bounce(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-host-bounce"
            (production / f"{stem}.result").write_text(
                "variant=attention-host-bounce\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention host-bounce checkpoint "
                "event=begin kind=mixed layer=17 pos=512 tokens=512 "
                "home=0 partner=2\n"
            )
            write_unhealthy_post(root, stem, device=2)
            write_lost_watch(root, stem, "2@00000000:81:00.0")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("does not identify physical GPU 0 or 1", report)
            self.assertIn("pair 1 retained direct peer traffic", report)
            self.assertIn("real but inconclusive", report)
            self.assertNotIn("not a necessary trigger", report)

    def test_reports_all_destination_stream_arms_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            for slot, variant in enumerate((
                    "attention-query-dst", "attention-gather-dst",
                    "attention-both-dst"), 1):
                stem = f"r1-s{slot}-{variant}"
                (production / f"{stem}.result").write_text(
                    f"variant={variant}\nstatus=failed\nexit_status=1\n"
                    "last_phase=measured-prefill\nlast_event=chunk-start\n"
                )
                (production / f"{stem}.log").write_text("")
                write_unhealthy_post(root, stem)
            stem = "r1-s4-production"
            (production / f"{stem}.result").write_text(
                "variant=production\nstatus=failed\nexit_status=1\n"
                "last_phase=measured-prefill\nlast_event=chunk-start\n"
            )
            (production / f"{stem}.log").write_text("")
            write_unhealthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("all recorded failures", report)
            self.assertIn("was not sufficient to prevent the fault", report)
            self.assertIn("does not identify a physical copy engine", report)

    def test_reports_unverified_ownership_matrix_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            stem = "r1-s1-attention-query-dst"
            (production / f"{stem}.started").write_text(
                "variant=attention-query-dst\n"
            )
            (production / f"{stem}.log").write_text("")

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            report = (root / "summary.md").read_text()
            self.assertIn("has no verified outcome", report)
            self.assertIn("not a failed arm and is not a safe pass", report)
            self.assertIn("fresh full matrix", report)

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
            (production / f"{stem}.log").write_text(
                "ds4: CUDA prefill attention row audit dispatch=split "
                "kind=mixed layer=17 pos=512 tokens=512 home=0 partner=2 "
                "q_bytes=1 result_bytes=1 query_copy_stream=destination "
                "gather_copy_stream=source\n"
            )
            write_healthy_post(root, stem)

            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (root / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(
                row["pair0_attention_query_copy_schedule"], "destination"
            )
            self.assertEqual(
                row["pair0_attention_gather_copy_schedule"], "source"
            )
            report = (root / "summary.md").read_text()
            self.assertIn("production control has not run yet", report)
            self.assertIn("requires the final control", report)


if __name__ == "__main__":
    unittest.main()
