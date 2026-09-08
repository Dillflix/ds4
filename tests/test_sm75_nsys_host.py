"""CPU-only unit tests; native Nsight process behavior is a separate host gate."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "speed-bench/qualify-sm75-nsys-host.py"
spec = importlib.util.spec_from_file_location("nsys_host", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def info(pid, parent=0, start=1, uid=1000):
    return {"pid": pid, "ppid": parent, "starttime": start, "uid": uid, "state": "S"}


class HostTests(unittest.TestCase):
    def test_command_has_no_cuda_or_gpu_trace(self):
        argv, target = m.host_command(Path("/opt/nsys"), Path("/tmp/case"),
                                      Path("/tmp/fixture.py"), "123abc", "normal")
        self.assertIn("--trace=none", argv)
        self.assertIn("--gpu-metrics-devices=none", argv)
        self.assertIn("--gpu-video-devices=none", argv)
        self.assertIn("--gpuctxsw=false", argv)
        self.assertIn("--sample=none", argv)
        self.assertIn("--cpuctxsw=none", argv)
        self.assertNotIn("cuda", ",".join(argv))
        self.assertEqual(target, [sys.executable, "-I", str(Path("/tmp/fixture.py")),
                                  str(Path("/tmp/case")), "123abc", "normal"])
        self.assertEqual(argv[-len(target):], target)

    def test_profiler_cannot_kill_other_process_groups(self):
        self.assertIn("--kill=none", m.HOST_OPTIONS)
        self.assertIn("--wait=all", m.HOST_OPTIONS)
        self.assertIn("--trace-fork-before-exec=false", m.HOST_OPTIONS)
        self.assertIn("--force-overwrite=false", m.HOST_OPTIONS)
        self.assertIn("--export=none", m.HOST_OPTIONS)

    def test_known_help(self):
        names = [option.split("=", 1)[0] for option in m.HOST_OPTIONS]
        names += ["--session-new", "--output", "--cuda-trace-all-apis", "--cuda-flush-interval",
                  "--cuda-event-trace", "--cuda-memory-usage", "--cuda-trace-scope"]
        m.validate_help("\n".join(name + "=\n value" for name in names))

    def test_missing_help_is_not_assumed_supported(self):
        with self.assertRaisesRegex(ValueError, "required options"):
            m.validate_help("--trace=none")

    def test_help_does_not_match_option_prefix(self):
        with self.assertRaises(ValueError):
            m.validate_help(" ".join(name.split("=")[0] + "-unrelated=" for name in m.HOST_OPTIONS))

    def test_reused_pid_rejected(self):
        self.assertFalse(m.same_process(info(10), info(10, start=2)))

    def test_changed_owner_rejected(self):
        self.assertFalse(m.same_process(info(10), info(10, uid=2000)))

    def test_state_changes_do_not_change_identity(self):
        new = info(10)
        new["state"] = "Z"
        self.assertTrue(m.same_process(info(10), new))

    def test_proc_stat_with_spaces_and_parentheses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "10").mkdir()
            # Field 3 is state, field 4 parent, field 22 start time.
            (root / "10/stat").write_text("10 (a name ) with parens) S 3 " + "0 " * 17 + "999 0")
            value = m.process_info(10, root)
            self.assertEqual(value["ppid"], 3)
            self.assertEqual(value["starttime"], 999)

    def test_pidfd_open_race_closes_fd(self):
        tree = m.OwnedTree.__new__(m.OwnedTree)
        tree.members = {}
        with patch.object(m.os, "getuid", return_value=1000, create=True), \
             patch.object(m.os, "pidfd_open", return_value=55, create=True), \
             patch.object(m, "process_info", return_value=info(10, start=2)), \
             patch.object(m.os, "close") as close:
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                tree.add(info(10))
            close.assert_called_once_with(55)
        self.assertEqual(tree.members, {})

    def test_pidfd_add_success(self):
        tree = m.OwnedTree.__new__(m.OwnedTree)
        tree.members = {}
        with patch.object(m.os, "getuid", return_value=1000, create=True), \
             patch.object(m.os, "pidfd_open", return_value=55, create=True), \
             patch.object(m, "process_info", return_value=info(10)):
            self.assertTrue(tree.add(info(10)))
            self.assertFalse(tree.add(info(10, start=2)))
        self.assertEqual(tree.members[10], (info(10), 55))

    def test_changed_user_not_signaled(self):
        tree = m.OwnedTree.__new__(m.OwnedTree)
        tree.members = {}
        with patch.object(m.os, "getuid", return_value=1000, create=True):
            with self.assertRaisesRegex(RuntimeError, "changed user"):
                tree.add(info(10, uid=0))

    def test_signal_uses_pidfd_not_numeric_pid(self):
        tree = m.OwnedTree.__new__(m.OwnedTree)
        tree.members = {10: (info(10), 55)}
        with patch.object(tree, "alive", return_value=True), \
             patch.object(m.signal, "pidfd_send_signal", create=True) as send:
            tree.send(10, signal.SIGTERM)
            send.assert_called_once_with(55, signal.SIGTERM)

    def test_dead_target_not_signaled(self):
        tree = m.OwnedTree.__new__(m.OwnedTree)
        tree.members = {10: (info(10), 55)}
        with patch.object(tree, "alive", return_value=False), \
             patch.object(m.signal, "pidfd_send_signal", create=True) as send:
            tree.send(10, signal.SIGTERM)
            send.assert_not_called()

    def test_exit_while_signaling_is_harmless(self):
        tree = m.OwnedTree.__new__(m.OwnedTree)
        tree.members = {10: (info(10), 55)}
        with patch.object(tree, "alive", return_value=True), \
             patch.object(m.signal, "pidfd_send_signal", side_effect=ProcessLookupError, create=True):
            tree.send(10, signal.SIGTERM)

    def test_fd_cleanup_is_idempotent(self):
        tree = m.OwnedTree.__new__(m.OwnedTree)
        tree.members = {10: (info(10), 55)}
        tree.closed = False
        with patch.object(m.os, "close") as close:
            tree.close()
            tree.close()
            close.assert_called_once_with(55)

    def scan_tree(self, records, root=None):
        tree = m.OwnedTree.__new__(m.OwnedTree)
        tree.members = {10: (root or info(10), 55)}
        def add(value):
            tree.members[value["pid"]] = (value, value["pid"] + 100)
            return True
        with patch.object(m.Path, "iterdir", return_value=[Path("/proc") / str(pid) for pid in records]), \
             patch.object(m, "process_info", side_effect=records.__getitem__), \
             patch.object(tree, "add", side_effect=add):
            tree.scan()
        return tree

    def test_descendant_discovery_is_transitive(self):
        tree = self.scan_tree({12: info(12, 11), 11: info(11, 10), 10: info(10), 99: info(99, 1)})
        self.assertEqual(set(tree.members), {10, 11, 12})

    def test_pid_reuse_does_not_adopt_unrelated_children(self):
        tree = self.scan_tree({11: info(11, 10), 10: info(10, start=2)})
        self.assertEqual(set(tree.members), {10})

    def test_missing_parent_does_not_establish_ownership(self):
        tree = self.scan_tree({11: info(11, 10)})
        self.assertEqual(set(tree.members), {10})

    def test_supervisor_cleans_up_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = Mock(pid=10, stdout=io.BytesIO(b"ok\n"), returncode=0)
            process.poll.return_value = 0
            tree = Mock(members={10: (info(10), 55)}, cleanup_issues=[])
            tree.live_pids.return_value = []
            tree.stop.return_value = []
            with patch.object(m.subprocess, "Popen", return_value=process), \
                 patch.object(m, "OwnedTree", return_value=tree):
                result = m.run_owned(["host-only"], Path(tmp))
            self.assertEqual(result["reason"], "processes-exited")
            self.assertTrue(result["reader_finished"])
            self.assertEqual((Path(tmp) / "console.log").read_bytes(), b"ok\n")
            tree.stop.assert_called_once()
            tree.close.assert_called_once()

    def test_supervisor_bounds_nonexiting_profiler(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = Mock(pid=10, stdout=io.BytesIO(b""), returncode=None)
            process.poll.return_value = None
            tree = Mock(members={10: (info(10), 55)}, cleanup_issues=[])
            tree.live_pids.return_value = [10]
            tree.stop.return_value = []
            with patch.object(m.subprocess, "Popen", return_value=process), \
                 patch.object(m, "OwnedTree", return_value=tree):
                result = m.run_owned(["host-only"], Path(tmp), seconds=0)
            self.assertEqual(result["reason"], "supervisor-timeout")
            tree.stop.assert_called_once()

    def test_log_flood_is_bounded_and_stops_owned_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = Mock(pid=10, stdout=io.BytesIO(b"0123456789"), returncode=0)
            process.poll.side_effect = lambda: (time.sleep(0.03) or 0)
            tree = Mock(members={10: (info(10), 55)}, cleanup_issues=[])
            tree.live_pids.return_value = []
            tree.stop.return_value = []
            with patch.object(m.subprocess, "Popen", return_value=process), \
                 patch.object(m, "OwnedTree", return_value=tree), patch.object(m, "LOG_LIMIT", 4):
                result = m.run_owned(["host-only"], Path(tmp))
            self.assertEqual(result["reason"], "log-capture-failed")
            self.assertTrue(result["log_overflow"])
            self.assertEqual((Path(tmp) / "console.log").read_bytes(), b"0123")

    def test_exception_still_stops_owned_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = Mock(pid=10, stdout=io.BytesIO(b""), returncode=None)
            process.poll.return_value = None
            tree = Mock(members={}, cleanup_issues=[])
            tree.scan.side_effect = RuntimeError("lost ancestry")
            tree.stop.return_value = []
            with patch.object(m.subprocess, "Popen", return_value=process), \
                 patch.object(m, "OwnedTree", return_value=tree):
                with self.assertRaisesRegex(RuntimeError, "lost ancestry"):
                    m.run_owned(["host-only"], Path(tmp))
            tree.stop.assert_called_once()
            tree.close.assert_called_once()
            self.assertTrue((Path(tmp) / "result.json").exists())

    def test_gpu_library_patterns(self):
        for lib in ("libcuda.so.595", "libcudart.so.13", "libcublas.so.13", "libcupti.so.13", "libnvidia-foo.so"):
            self.assertIsNotNone(m.GPU_LIB.search("0000 r-xp /usr/lib/" + lib))
        self.assertIsNone(m.GPU_LIB.search("/usr/lib/libc.so.6"))

    def test_readiness_rejects_unowned_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ready.json"
            path.write_text(json.dumps({"pid": 99}))
            tree = Mock(members={})
            with self.assertRaisesRegex(ValueError, "observed descendant"):
                m.validate_ready(path, tree, [], "abc", "normal")

    def test_readiness_rejects_boolean_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ready.json"
            path.write_text('{"pid":true}')
            tree = Mock(members={1: (info(1), 5)})
            with self.assertRaises(ValueError):
                m.validate_ready(path, tree, [], "abc", "normal")

    def test_oversize_readiness_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ready.json"
            path.write_text("x" * 10)
            with patch.object(m, "LOG_LIMIT", 4):
                with self.assertRaisesRegex(ValueError, "invalid fixture"):
                    m.validate_ready(path, Mock(), [], "abc", "normal")

    def test_safe_artifact_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").write_bytes(b"1234")
            (root / "b").write_bytes(b"12345")
            with patch.object(m, "ARTIFACT_LIMIT", 4):
                files, excluded = m.safe_files(root)
            self.assertEqual([p.name for p in files], ["a"])
            self.assertEqual(excluded, ["b"])

    def test_safe_total_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a").write_bytes(b"1234")
            (root / "b").write_bytes(b"1234")
            with patch.object(m, "TOTAL_LIMIT", 5):
                files, excluded = m.safe_files(root)
            self.assertEqual([p.name for p in files], ["a"])
            self.assertEqual(excluded, ["b"])

    def test_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a"
            path.write_bytes(b"abc")
            self.assertEqual(m.digest(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_forbidden_env_archives_failure_without_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            output.mkdir()
            with patch.dict(os.environ, {"LD_PRELOAD": "bad.so"}), patch.object(m, "run_owned") as launch:
                self.assertEqual(m.run(output, Path("/missing/nsys")), 2)
            launch.assert_not_called()
            summary = json.loads((output / "summary.json").read_text())
            self.assertFalse(summary["gpu_capture_qualified"])
            self.assertTrue(Path(str(output) + ".tar.gz").is_file())

    def test_complete_mock_gate_never_qualifies_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            nsys = root / "nsys"
            nsys.write_bytes(b"a fake host tool, never executed")
            def fake_run(argv, directory, **kwargs):
                if directory.name == "profile-help":
                    text = "\n".join(option.split("=", 1)[0] + "=" for option in m.HOST_OPTIONS)
                    text += "\n--session-new=\n--output=\n--cuda-trace-all-apis=\n--cuda-flush-interval="
                    text += "\n--cuda-event-trace=\n--cuda-memory-usage=\n--cuda-trace-scope="
                else:
                    text = "cpu_fixture_complete=normal"
                (directory / "console.log").write_text(text)
                return {"returncode": 0, "reason": "processes-exited", "reader_finished": True,
                        "log_overflow": False, "reader_errors": [], "cleanup_issues": [],
                        "surviving_owned_pids": [], "target": {"pid": 10}, "target_kill_sent": True}
            with patch.dict(os.environ, {}, clear=True), patch.object(m, "run_owned", side_effect=fake_run) as launch:
                self.assertEqual(m.run(output, nsys), 0)
            self.assertEqual(launch.call_count, 6)
            result = json.loads((output / "summary.json").read_text())
            self.assertEqual(result["host_qualification"], "passed")
            self.assertFalse(result["gpu_capture_qualified"])
            self.assertFalse(result["gpu_workload_executed"])
            self.assertEqual(len(result["cases"]), 3)
            self.assertTrue(Path(str(output) + ".tar.gz").exists())

    def test_output_substitution_is_rejected_without_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output%p"
            output.mkdir()
            with patch.object(m, "run_owned") as launch:
                self.assertEqual(m.run(output, Path("missing")), 2)
            launch.assert_not_called()

    def test_fixture_has_no_workload_imports(self):
        import ast
        path = SCRIPT.parent.parent / "tests/fixtures/sm75-nsys/cpu_target.py"
        node = ast.parse(path.read_text())
        imports = []
        for value in ast.walk(node):
            if isinstance(value, ast.Import):
                imports += [item.name for item in value.names]
            elif isinstance(value, ast.ImportFrom):
                imports.append(value.module)
        self.assertEqual(set(imports), {"json", "os", "pathlib", "signal", "sys", "time"})


if __name__ == "__main__":
    unittest.main()
