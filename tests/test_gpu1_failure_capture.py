#!/usr/bin/env python3
"""CPU-only collector tests. Never invoke a real NVIDIA/system service tool."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "speed-bench/capture-sm75-gpu1-failure.py"
spec = importlib.util.spec_from_file_location("capture_gpu1", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def kernel_line(bdf="0000:81:00", code=79):
    return json.dumps({"MESSAGE": f"NVRM: Xid (PCI:{bdf}): {code}, GPU has fallen off the bus.",
                       "__MONOTONIC_TIMESTAMP": "7600000000",
                       "__REALTIME_TIMESTAMP": "1788812244000000", "_BOOT_ID": "test-boot"})


class BaseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="gpu1-capture-cpu-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name) / "capture"
        self.capture = module.Capture(self.folder, sys.executable, [sys.executable, "-c", "pass"], 1)
        if sys.platform != "linux":
            # Windows can run the CPU children, but cannot exercise POSIX
            # process_group. The exact POSIX launch contract is tested below.
            self.capture.launch_collector = lambda command, **kwargs: subprocess.Popen(command, **kwargs)


class PreflightTests(BaseTest):
    def setUp(self):
        super().setUp()
        self.capture.output.mkdir()
        self.responses = {
            "pre/sudo": "",
            "pre/report-tool": "/usr/bin/nvidia-bug-report.sh\n",
            "pre/dcgm": "Service Startup Current Notes\ndcgm.nv-hostengine disabled inactive -\n"
                        "dcgm.dcgm-exporter disabled inactive -\n",
            "pre/retrain-unit": "[Service]\nType=oneshot\nRemainAfterExit=yes\n",
            "pre/retrain-state": "Type=oneshot\nActiveState=active\nSubState=exited\nMainPID=0\n"
                                 "Result=success\nExecMainStatus=0\n",
            "pre/processes": " PID PPID STARTED COMMAND\n 12 1 today sshd\n",
            "pre/gpus": "\n".join(
                f"{i}, {bdf}, {module.GPU1_UUID if i == 1 else 'GPU-test'}, "
                f"595.84, P8, 35, 30, 260, 1, 49151"
                for i, bdf in enumerate(module.ENDPOINTS)),
            "pre/compute-processes": "",
            "pre/gpu-details": "BAR1 Memory Usage (mock)\n",
            "kernel-baseline": '{"MESSAGE":"boot healthy"}\n-- cursor: cursor-123\n',
            "pre/retrain-history": "service finished before workload\n",
        }
        self.capture.required = Mock(side_effect=lambda name, *a, **kw: self.responses[name])
        self.capture.pci_snapshot = Mock()
        self.capture.start_journal = Mock()
        self.capture.execute = Mock()

    def preflight(self):
        original = Path.read_text

        def read(path, *args, **kwargs):
            if str(path).replace("\\", "/").endswith("/proc/sys/kernel/random/boot_id"):
                return "test-boot\n"
            if str(path).replace("\\", "/").endswith("/proc/cmdline"):
                return "pcie_aspm=off\n"
            return original(path, *args, **kwargs)
        with patch.object(module.sys, "platform", "linux"), \
             patch.object(module, "sha256", return_value=module.EXPECTED_SHA256), \
             patch.object(module.os, "uname", return_value=("Linux", "test"), create=True), \
             patch.object(Path, "read_text", read), patch.dict(module.os.environ, {}, clear=True):
            self.capture.preflight()

    def test_valid_preflight_requires_completed_oneshot_not_inactive(self):
        self.preflight()
        self.capture.start_journal.assert_called_once_with("cursor-123")
        self.capture.execute.assert_not_called()

    def test_dcgm_active_rejected(self):
        self.responses["pre/dcgm"] = "dcgm.nv-hostengine enabled active -"
        with self.assertRaisesRegex(RuntimeError, "disabled and inactive"):
            self.preflight()
        self.capture.execute.assert_not_called()

    def test_retrain_running_rejected(self):
        self.responses["pre/retrain-state"] = self.responses["pre/retrain-state"].replace("MainPID=0", "MainPID=42")
        with self.assertRaisesRegex(RuntimeError, "retraining"):
            self.preflight()

    def test_retrain_failed_rejected(self):
        self.responses["pre/retrain-state"] = self.responses["pre/retrain-state"].replace("Result=success", "Result=timeout")
        with self.assertRaisesRegex(RuntimeError, "retraining"):
            self.preflight()

    def test_background_nvbandwidth_rejected(self):
        self.responses["pre/processes"] += " 123 1 today nvbandwidth\n"
        with self.assertRaisesRegex(RuntimeError, "background"):
            self.preflight()

    def test_gpu2_absent_rejected(self):
        self.responses["pre/gpus"] = "\n".join(self.responses["pre/gpus"].splitlines()[:2])
        with self.assertRaisesRegex(RuntimeError, "four accessible"):
            self.preflight()

    def test_gpu1_identity_rejected(self):
        self.responses["pre/gpus"] = self.responses["pre/gpus"].replace(module.GPU1_UUID, "GPU-other")
        with self.assertRaisesRegex(RuntimeError, "UUID mismatch"):
            self.preflight()

    def test_active_cuda_client_rejected(self):
        self.responses["pre/compute-processes"] = "123, GPU-test, benchmark"
        with self.assertRaisesRegex(RuntimeError, "compute client"):
            self.preflight()

    def test_old_boot_fault_rejected_not_replayed(self):
        self.responses["kernel-baseline"] = kernel_line() + "\n-- cursor: after-fault\n"
        with self.assertRaisesRegex(RuntimeError, "already contains"):
            self.preflight()
        self.capture.start_journal.assert_not_called()

    def test_missing_journal_cursor_rejected(self):
        self.responses["kernel-baseline"] = '{}\n'
        with self.assertRaisesRegex(RuntimeError, "cursor unavailable"):
            self.preflight()

    def test_no_sudo_rejected(self):
        self.capture.required.side_effect = RuntimeError("sudo not available")
        with self.assertRaisesRegex(RuntimeError, "sudo"):
            self.preflight()
        self.capture.start_journal.assert_not_called()

    def test_changed_executable_refused_before_any_host_query(self):
        with patch.object(module.sys, "platform", "linux"), patch.object(module, "sha256", return_value="bad"):
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                self.capture.preflight()
        self.capture.required.assert_not_called()

    def test_old_python_refused_before_any_host_query(self):
        with patch.object(module.sys, "platform", "linux"), \
             patch.object(module.sys, "version_info", (3, 10)):
            with self.assertRaisesRegex(RuntimeError, "Python 3.11"):
                self.capture.preflight()
        self.capture.required.assert_not_called()

    def test_instrumentation_refused(self):
        with patch.object(module.sys, "platform", "linux"), \
             patch.object(module, "sha256", return_value=module.EXPECTED_SHA256), \
             patch.dict(module.os.environ, {"LD_PRELOAD": "instrumentation.so"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "inherited instrumentation"):
                self.capture.preflight()
        self.capture.required.assert_not_called()


class ExecutionTests(BaseTest):
    def setUp(self):
        super().setUp()
        # Real CPU-only child, synthetic journal. No GPU or OS collectors run.
        self.capture.preflight = Mock()
        self.capture.postmortem = Mock()
        self.capture.process_snapshot = Mock(return_value={"threads": {}})
        self.capture.journal_process = Mock()
        self.capture.journal_process.poll.return_value = None

        def stop(process):
            if process is self.capture.journal_process:
                process.poll.return_value = 0
                return True
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=3)
            return True
        self.capture.stop = Mock(side_effect=stop)

    def run_capture(self, program):
        self.capture.command = [sys.executable, "-u", "-c", program]
        output = Mock(buffer=io.BytesIO())
        with patch.object(module.sys, "stdout", output):
            return module.Capture.capture(self.capture)

    def test_success_one_launch_and_timestamped_progress(self):
        code = self.run_capture("print('production103_replay_phase=burnin,event=complete,calls=10,total=1024')")
        self.assertEqual(code, 0)
        timeline = [json.loads(line) for line in (self.folder / "application-timeline.jsonl").read_text().splitlines()]
        self.assertEqual(len(timeline), 1)
        self.assertGreater(timeline[0]["utc_ns"], 0)
        self.assertGreater(timeline[0]["monotonic_ns"], 0)
        self.capture.postmortem.assert_called_once()

    def test_workload_134_separate_from_successful_capture(self):
        code = self.run_capture("import sys; print('partial evidence'); sys.exit(134)")
        self.assertEqual(code, 134)
        saved = json.loads((self.folder / "summary.json").read_text())
        self.assertEqual(saved["collection"], "complete")
        self.assertEqual(saved["workload_returncode"], 134)

    def test_timeout_retains_partial_log(self):
        started = time.monotonic()
        code = self.run_capture("import time; print('before stall'); time.sleep(30)")
        self.assertEqual(code, 124)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIn("before stall", (self.folder / "application-timeline.jsonl").read_text())
        self.capture.postmortem.assert_called_once()

    def test_gpu2_fault_stops_gpu1_workload_and_preserves_identity(self):
        timer = threading.Timer(0.3, lambda: self.capture.observe_kernel(kernel_line()))
        timer.start()
        self.addCleanup(timer.cancel)
        self.assertEqual(self.run_capture("import time; time.sleep(30)"), 86)
        self.assertEqual(self.capture.summary["first_fault"]["bdf"], "0000:81:00")
        self.assertEqual(self.capture.summary["first_fault"]["xid"], 79)
        self.assertEqual(self.capture.summary["workload"], "stopped-on-kernel-fault")

    def test_interrupt_still_collects(self):
        timer = threading.Timer(0.3, self.capture.interrupted.set)
        timer.start()
        self.addCleanup(timer.cancel)
        self.assertEqual(self.run_capture("import time; time.sleep(30)"), 130)
        self.capture.postmortem.assert_called_once()

    def test_dead_journal_stops_workload(self):
        self.capture.journal_process.poll.return_value = 1
        self.assertNotEqual(self.run_capture("import time; time.sleep(30)"), 0)
        self.assertEqual(self.capture.summary["workload"], "stopped-on-collector-failure")
        self.assertEqual(self.capture.summary["collection"], "partial")

    def test_failed_postmortem_cannot_hide_workload_failure(self):
        self.capture.postmortem.side_effect = RuntimeError("collector unavailable")
        self.assertEqual(self.run_capture("import sys; sys.exit(134)"), 134)
        self.assertEqual(self.capture.summary["collection"], "partial")

    def test_failed_preflight_never_launches_workload(self):
        self.capture.preflight.side_effect = RuntimeError("DCGM still active")
        self.assertEqual(self.run_capture("raise AssertionError('must not run')"), 2)
        self.capture.postmortem.assert_not_called()
        self.assertIsNone(self.capture.workload_process)
        self.assertTrue((self.folder / "summary.json").exists())

    def test_sudo_refresh_runs_only_during_owned_workload(self):
        self.capture.sudo_refresh_seconds = 0.1
        self.capture.run = Mock(return_value=({"returncode": 0, "timeout": False}, ""))
        self.assertEqual(self.run_capture("import time; time.sleep(0.3)"), 0)
        self.assertGreaterEqual(self.capture.run.call_count, 1)
        for call in self.capture.run.call_args_list:
            self.assertEqual(call.args[1], ["sudo", "-n", "-v"])
            self.assertEqual(call.kwargs["seconds"], 5)

    def test_failed_sudo_refresh_stops_before_more_work(self):
        self.capture.sudo_refresh_seconds = 0.1
        self.capture.run = Mock(return_value=({"returncode": 1, "timeout": False}, "password required"))
        self.assertNotEqual(self.run_capture("import time; time.sleep(30)"), 0)
        self.assertEqual(self.capture.summary["workload"], "stopped-on-collector-failure")
        self.assertEqual(self.capture.summary["collection"], "partial")
        self.capture.postmortem.assert_called_once()

    def test_workload_launch_remains_unprivileged_and_separate_session(self):
        original = subprocess.Popen
        with patch.object(module.subprocess, "Popen", side_effect=original) as launch:
            self.assertEqual(self.run_capture("pass"), 0)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(launch.call_args.args[0], self.capture.command)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])


class CollectorTests(BaseTest):
    def setUp(self):
        super().setUp()
        self.folder.mkdir()

    def test_command_failure_saves_stderr_and_status(self):
        result, output = self.capture.run("failed", [sys.executable, "-c",
            "import sys; print('collector evidence', file=sys.stderr); sys.exit(3)"])
        self.assertEqual(result["returncode"], 3)
        self.assertIn("collector evidence", output)

    def test_sudo_error_visible_without_opening_the_archive(self):
        self.capture.run = Mock(return_value=({"returncode": 1, "timeout": False},
                                             "sudo: a password is required\n"))
        with self.assertRaisesRegex(RuntimeError, "pre/sudo: sudo: a password is required"):
            self.capture.required("pre/sudo", ["sudo", "-n", "true"])

    def test_collector_keeps_terminal_session_but_owns_process_group(self):
        with patch.object(module.subprocess, "Popen") as launch:
            module.Capture.launch_collector(["sudo", "-n", "true"], stdout=subprocess.PIPE)
        self.assertFalse(launch.call_args.kwargs["start_new_session"])
        self.assertEqual(launch.call_args.kwargs["process_group"], 0)
        self.assertNotIn("preexec_fn", launch.call_args.kwargs)

    def test_journal_uses_same_session_collector_launcher(self):
        process = Mock(stdout=io.BytesIO(b'{"MESSAGE":"healthy"}\n'))
        process.poll.return_value = None
        self.capture.boot_id = "test-boot"
        self.capture.launch_collector = Mock(return_value=process)
        self.capture.stop = Mock(return_value=True)
        try:
            self.capture.start_journal("test-cursor")
            self.capture.journal_reader.join(timeout=1)
            call = self.capture.launch_collector.call_args
            self.assertEqual(call.args[0][:2], ["sudo", "-n"])
            self.assertIn("--after-cursor=test-cursor", call.args[0])
            self.assertNotIn("start_new_session", call.kwargs)
        finally:
            self.capture.finish()

    def test_command_timeout_saves_partial_output(self):
        def stop(process):
            process.terminate()
            process.wait(timeout=3)
            return True
        with patch.object(self.capture, "stop", side_effect=stop):
            result, output = self.capture.run("slow", [sys.executable, "-u", "-c",
                "import time; print('partial'); time.sleep(30)"], seconds=0.2)
        self.assertTrue(result["timeout"])
        self.assertIn("partial", output)

    def test_missing_collector_is_recorded(self):
        result, _ = self.capture.run("missing", [str(self.folder / "does-not-exist")])
        self.assertIsNone(result["returncode"])
        self.assertTrue(self.capture.summary["issues"])

    def test_stop_only_signals_owned_process_group(self):
        process = Mock(pid=12345)
        process.poll.return_value = None
        process.wait.return_value = 0
        with patch.object(module.os, "killpg", create=True) as kill, \
             patch.object(module.signal, "SIGKILL", 9, create=True):
            self.assertTrue(self.capture.stop(process))
        kill.assert_called_once_with(12345, module.signal.SIGTERM)

    def test_uninterruptible_process_does_not_block_forever(self):
        process = Mock(pid=12345)
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("owned-child", 2)
        with patch.object(module.os, "killpg", create=True) as kill, \
             patch.object(module.signal, "SIGKILL", 9, create=True):
            self.assertFalse(self.capture.stop(process))
        self.assertEqual(kill.call_count, 2)

    def test_first_fault_not_overwritten_by_followup_xid154(self):
        self.capture.observe_kernel(kernel_line(code=79))
        self.capture.observe_kernel(kernel_line(code=154))
        self.assertEqual(self.capture.summary["first_fault"]["xid"], 79)

    def test_correctable_aer_is_recorded_but_not_fatal(self):
        line = json.dumps({"MESSAGE": "pcieport 0000:80:02.0: AER: Corrected error received"})
        self.assertIsNone(module.fault_event(line))

    def test_fatal_aer_detected(self):
        line = json.dumps({"MESSAGE": "pcieport 0000:80:02.0: AER: Uncorrected (Fatal) error received"})
        self.assertEqual(module.fault_event(line)["bdf"], "0000:80:02.0")

    def test_malformed_journal_not_misclassified(self):
        self.assertIsNone(module.fault_event("-- cursor: example"))
        self.assertIsNone(module.fault_event('{"MESSAGE": [65, 66]}'))

    def test_report_after_failure_once_with_safe_flags(self):
        self.capture.boot_id = "test-boot"
        self.capture.summary["workload"] = "failed"
        self.capture.run = Mock(return_value=({"returncode": 0}, ""))
        self.capture.pci_snapshot = Mock()
        self.capture.postmortem()
        reports = [call for call in self.capture.run.call_args_list if call.args[0] == "nvidia-report/collector"]
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].args[1], ["nvidia-bug-report.sh", "--safe-mode", "--extra-system-data"])
        self.assertEqual(reports[0].kwargs["seconds"], 120)

    def test_success_does_not_launch_bug_report(self):
        self.capture.boot_id = "test-boot"
        self.capture.summary["workload"] = "exited-zero"
        self.capture.run = Mock(return_value=({"returncode": 0}, ""))
        self.capture.pci_snapshot = Mock()
        self.capture.postmortem()
        self.assertFalse(any(c.args[0] == "nvidia-report/collector" for c in self.capture.run.call_args_list))

    def test_late_fault_in_post_journal_triggers_report(self):
        self.capture.boot_id = "test-boot"
        self.capture.summary["workload"] = "exited-zero"
        self.capture.summary["workload_returncode"] = 0
        self.capture.run = Mock(side_effect=lambda name, *a, **kw:
                                ({"returncode": 0}, kernel_line() if name == "post/kernel" else ""))
        self.capture.pci_snapshot = Mock()
        self.capture.postmortem()
        self.assertTrue(self.capture.fault.is_set())
        self.assertEqual(self.capture.result(), 86)


if __name__ == "__main__":
    unittest.main()
