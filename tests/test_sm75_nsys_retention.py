"""Portable CPU checks. Native Linux/Nsight qualification remains separate."""
import importlib.util
from contextlib import closing, ExitStack
import io
import json
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "speed-bench/qualify-sm75-nsys-retention.py"
spec = importlib.util.spec_from_file_location("nsys_retention", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "live"
        self.source.mkdir()
        self.destination = self.root / "retained"
        # Windows CPU test only: no Linux open flags, native test exercises them.
        for name in ("O_NOFOLLOW", "O_NONBLOCK"):
            p = patch.object(m.os, name, getattr(os, name, 0), create=True)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(m.signal, "SIGKILL", getattr(m.signal, "SIGKILL", 9), create=True)
        p.start()
        self.addCleanup(p.stop)

    def store(self):
        return m.PrefixStore(self.source, self.destination)

    def test_private_temp_controls_do_not_mutate_parent_environment(self):
        source = SCRIPT.read_text()
        self.assertIn('environment = os.environ.copy()', source)
        self.assertIn('"NSYS_TMPDIR": str(tmp), "TMPDIR": str(tmp)', source)
        self.assertNotIn('os.environ.update', source)

    def test_cpu_profile_policy_is_preserved(self):
        argv, target = m.gated_command(Path("/opt/nsys"), Path("/tmp/case"),
            Path("/tmp/gate.py"), Path("/tmp/config.json"), "abc")
        for option in m.host.HOST_OPTIONS:
            self.assertIn(option, argv)
        self.assertIn("--trace=none", argv)
        self.assertIn("--kill=none", argv)
        self.assertEqual(argv[-len(target):], target)
        self.assertEqual(target[1:], ["-I", str(Path("/tmp/gate.py")), str(Path("/tmp/config.json"))])

    def test_cli_has_no_workload_or_cuda_option(self):
        with patch.object(m.sys, "argv", [str(SCRIPT), "--executable", "gpu-test"]):
            with self.assertRaises(SystemExit) as error:
                m.main()
            self.assertEqual(error.exception.code, 2)

    def test_non_linux_rejected_before_any_subprocess(self):
        with patch.object(m.sys, "argv", [str(SCRIPT)]), \
             patch.object(m.platform, "system", return_value="Windows"), \
             patch.object(m.subprocess, "Popen") as popen:
            with self.assertRaises(SystemExit):
                m.main()
            popen.assert_not_called()

    def test_prefix_survives_original_deletion(self):
        path = self.source / "report.qdstrm"
        path.write_bytes(b"prefix")
        store = self.store()
        store.capture()
        path.unlink()
        store.capture()
        record = store.records["report.qdstrm"]
        self.assertEqual((self.destination / record["prefix_file"]).read_bytes(), b"prefix")
        self.assertEqual(record["completeness"], "unknown-nontransactional-prefix")
        self.assertEqual(record["captures"], 1)

    def test_replacement_and_truncation_not_assumed_append_only(self):
        path = self.source / "report.qdstrm"
        path.write_bytes(b"long content")
        store = self.store()
        store.capture()
        path.unlink()
        path.write_bytes(b"new")
        store.capture()
        record = store.records["report.qdstrm"]
        self.assertEqual((self.destination / record["prefix_file"]).read_bytes(), b"new")
        self.assertEqual(record["captures"], 2)

    def test_per_file_limit_marks_truncated(self):
        (self.source / "a").write_bytes(b"0123456789")
        with patch.object(m, "PREFIX_LIMIT", 4):
            store = self.store()
            store.capture()
        self.assertEqual(store.records["a"]["retained_bytes"], 4)
        self.assertTrue(store.records["a"]["truncated"])

    def test_total_limit(self):
        (self.source / "a").write_bytes(b"aaaa")
        (self.source / "b").write_bytes(b"bbbb")
        with patch.object(m, "PREFIX_TOTAL", 5):
            store = self.store()
            store.capture()
        self.assertEqual(sum(x["retained_bytes"] for x in store.records.values()), 5)
        self.assertTrue(store.records["b"]["truncated"])

    def test_count_limit(self):
        (self.source / "a").touch()
        (self.source / "b").touch()
        with patch.object(m, "FILE_COUNT_LIMIT", 1):
            with self.assertRaisesRegex(ValueError, "file count"):
                self.store().capture()

    def test_nested_file_is_preserved_with_relative_name(self):
        (self.source / "nested").mkdir()
        (self.source / "nested/a").write_bytes(b"hello")
        store = self.store()
        store.capture()
        self.assertIn(str(Path("nested/a")), store.records)
        self.assertTrue((self.destination / "manifest.json").is_file())

    def test_nonordinary_file_rejected(self):
        (self.source / "a").write_bytes(b"hello")
        bad = Mock(st_mode=stat.S_IFIFO, st_nlink=1)
        with patch.object(m.os, "fstat", return_value=bad):
            with self.assertRaisesRegex(ValueError, "nonordinary"):
                self.store().capture()

    def test_hardlink_rejected(self):
        (self.source / "a").write_bytes(b"hello")
        bad = Mock(st_mode=stat.S_IFREG, st_nlink=2)
        with patch.object(m.os, "fstat", return_value=bad):
            with self.assertRaisesRegex(ValueError, "nonordinary"):
                self.store().capture()

    def test_disappearing_source_is_not_a_fake_retained_record(self):
        (self.source / "a").write_bytes(b"hello")
        with patch.object(m.os, "open", side_effect=FileNotFoundError):
            store = self.store()
            store.capture()
        self.assertEqual(store.records, {})

    def test_source_changed_while_copying_is_recorded(self):
        path = self.source / "a"
        path.write_bytes(b"hello")
        actual = path.stat()
        after = Mock(st_size=999, st_mtime_ns=actual.st_mtime_ns + 1)
        with patch.object(m.os, "fstat", side_effect=[actual, after]):
            store = self.store()
            store.capture()
        self.assertTrue(store.records["a"]["source_changed_during_copy"])

    def test_private_read_rejects_mode_or_owner_or_size(self):
        path = self.source / "config"
        path.write_bytes(b"{}")
        for fields in ({"st_mode": stat.S_IFREG | 0o644}, {"st_mode": stat.S_IFREG | 0o666}, {"st_uid": 2},
                       {"st_size": 999999}, {"st_nlink": 2}, {"st_mode": stat.S_IFIFO}):
            properties = dict(st_mode=stat.S_IFREG | 0o600, st_nlink=1, st_uid=1, st_size=2)
            properties.update(fields)
            with self.subTest(fields=fields), patch.object(m.os, "getuid", return_value=1, create=True), \
                 patch.object(m.os, "fstat", return_value=Mock(**properties)):
                with self.assertRaisesRegex(ValueError, "private gate"):
                    m.gate.private_read(path)

    def test_private_read_success(self):
        path = self.source / "config"
        path.write_bytes(b"{}")
        info = Mock(st_mode=stat.S_IFREG | 0o600, st_nlink=1, st_uid=1, st_size=2)
        with patch.object(m.os, "getuid", return_value=1, create=True), \
             patch.object(m.os, "fstat", return_value=info):
            self.assertEqual(m.gate.private_read(path), "{}")

    def test_private_json_requests_0600_exclusively_without_changing_umask(self):
        path = self.source / "ready.json"
        real_open = os.open
        with patch.object(m.gate.os, "open", wraps=real_open) as opened, \
             patch.object(m.gate.os, "umask") as umask:
            m.gate.private_json(path, {"state": "waiting-before-exec"})
        opened.assert_called_once_with(path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        umask.assert_not_called()
        self.assertEqual(json.loads(path.read_text()), {"state": "waiting-before-exec"})

    def test_private_json_cannot_overwrite_existing_evidence(self):
        path = self.source / "ready.json"
        path.write_text("original evidence")
        with self.assertRaises(FileExistsError):
            m.gate.private_json(path, {"replacement": True})
        self.assertEqual(path.read_text(), "original evidence")

    def test_archived_0666_readiness_rejected_with_actionable_details(self):
        # Native hjb8qrpk archive: the gate's open('x') inherited a permissive
        # mask; collector config was 0600 but gate-ready.json was 0666.
        path = self.source / "gate-ready.json"
        path.write_text('{}')
        info = Mock(st_mode=stat.S_IFREG | 0o666, st_nlink=1, st_uid=1000, st_size=126)
        with patch.object(m.os, "getuid", return_value=1000, create=True), \
             patch.object(m.os, "fstat", return_value=info):
            with self.assertRaisesRegex(ValueError, "mode=0666 uid=1000"):
                m.gate.private_read(path)

    def test_both_gate_records_use_explicit_private_writer_before_exec(self):
        executable = Path(m.sys.executable).resolve()
        config_path = self.source / "gate-config.json"
        config = {"executable": str(executable), "sha256": "expected", "nonce": "abc", "arguments": []}
        (self.source / "gate-release").touch()
        class ExecReached(Exception):
            pass
        with patch.object(m.gate.sys, "argv", ["gate", str(config_path)]), \
             patch.object(m.gate, "private_read", side_effect=[json.dumps(config), "abc"]), \
             patch.object(m.gate.Path, "read_text", return_value="10 (gate) S 9 " + "0 " * 17 + "123 0"), \
             patch.object(m.gate, "fingerprint", return_value="expected"), \
             patch.object(m.gate, "private_json") as write, \
             patch.object(m.gate.os, "execv", side_effect=ExecReached) as execute:
            with self.assertRaises(ExecReached):
                m.gate.main()
        self.assertEqual([c.args[0].name for c in write.call_args_list], ["gate-ready.json", "exec-attempt.json"])
        self.assertEqual(write.call_args_list[0].args[1]["starttime"], 123)
        self.assertEqual(write.call_args_list[1].args[1]["starttime"], 123)
        execute.assert_called_once_with(str(executable), [str(executable)])

    def test_console_validation_exposes_saved_session_error(self):
        case = self.clean_case()
        case["issues"] = ["invalid private gate file: mode=0666"]
        with self.assertRaisesRegex(RuntimeError, "mode=0666"):
            m.validate_case(case, "")

    def test_gate_rejects_unowned_or_dead_or_noninteger_pid(self):
        tree = Mock(members={10: ({}, 55)})
        for pid, alive in ((11, True), (True, True), (10, False)):
            tree.alive.return_value = alive
            with self.subTest(pid=pid, alive=alive), \
                 patch.object(m.gate, "private_read", return_value=json.dumps({"pid": pid})):
                with self.assertRaisesRegex(ValueError, "live owned"):
                    m.admitted_gate(Path("ready"), tree, [], "abc")

    def test_gate_rejects_wrong_nonce_or_process_generation(self):
        info = dict(pid=10, ppid=1, starttime=2, uid=3)
        tree = Mock(members={10: (info, 55)})
        tree.alive.return_value = True
        for values in ({"nonce": "wrong"}, {"starttime": 99}, {"state": "already-exec"}):
            ready = dict(info, nonce="abc", state="waiting-before-exec")
            ready.update(values)
            with self.subTest(values=values), patch.object(m.gate, "private_read", return_value=json.dumps(ready)), \
                 patch.object(m.host, "process_info", return_value=info):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    m.admitted_gate(Path("ready"), tree, [], "abc")

    def clean_case(self, mode="normal"):
        return dict(mode=mode, reason="processes-exited", issues=[], exec_identity_verified=True,
                    target_exit_observed=True, surviving_owned_pids=[], reader_finished=True,
                    log_overflow=False, profiler_returncode=0)

    def test_clean_case_passes(self):
        m.validate_case(self.clean_case(), "cpu_fixture_complete=normal")

    def test_partial_supervision_cannot_pass(self):
        for field, value in (("exec_identity_verified", False), ("target_exit_observed", False),
                             ("issues", ["failed"]), ("surviving_owned_pids", [99]),
                             ("reader_finished", False), ("log_overflow", True)):
            case = self.clean_case()
            case[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                m.validate_case(case, "cpu_fixture_complete=normal")

    def test_profiler_zero_alone_is_not_target_success(self):
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            m.validate_case(self.clean_case(), "unrelated")

    def test_normal_nonzero_rejected(self):
        case = self.clean_case()
        case["profiler_returncode"] = 1
        with self.assertRaises(RuntimeError):
            m.validate_case(case, "cpu_fixture_complete=normal")

    def test_abrupt_must_not_report_success(self):
        with self.assertRaisesRegex(RuntimeError, "loss was not reflected"):
            m.validate_case(self.clean_case("abrupt"), "")

    def test_timeout_requires_verified_hard_stop(self):
        case = self.clean_case("timeout")
        with self.assertRaisesRegex(RuntimeError, "hard stop"):
            m.validate_case(case, "")
        case["target_kill_sent"] = True
        m.validate_case(case, "")

    def test_absent_report_is_not_export_success(self):
        with patch.object(m.host, "run_owned") as run:
            result = m.export_report(Path("nsys"), self.root)
            run.assert_not_called()
        self.assertEqual(result["status"], "report-absent")

    def test_export_uses_copy_and_timeout(self):
        original = self.root / "report.nsys-rep"
        original.write_bytes(b"report")
        with patch.object(m.host, "run_owned", return_value={"returncode": 124}) as run:
            result = m.export_report(Path("nsys"), self.root)
        self.assertEqual(run.call_args.kwargs["seconds"], 60)
        self.assertEqual(run.call_args.args[0][-1], str(self.root / "export/retained.nsys-rep"))
        self.assertEqual(original.read_bytes(), b"report")
        self.assertEqual(result["status"], "export-failed-or-incomplete")

    def test_export_failure_never_fabricates_schema(self):
        (self.root / "report.nsys-rep").write_bytes(b"report")
        with patch.object(m.host, "run_owned", return_value={"returncode": 1}):
            result = m.export_report(Path("nsys"), self.root)
        self.assertNotIn("schema", result)
        self.assertIn("unknown", result["intermediate_importability"])

    def test_sqlite_schema_inventory_does_not_assume_cupti_tables(self):
        db = self.root / "report.sqlite"
        with closing(sqlite3.connect(db)) as connection:
            connection.execute('CREATE TABLE "odd""name" (x INTEGER, y TEXT)')
        self.assertEqual(m.sqlite_inventory(db), [{"table": 'odd"name', "columns": ["x", "y"]}])

    def simulated_case(self, *, ready_pid=11, poll_error=None, prefix_error=None):
        directory = self.root / "case"
        directory.mkdir()
        (directory / "gate-ready.json").write_text("{}")
        (directory / "ready.json").write_text("{}")
        process = Mock(pid=10, stdout=io.BytesIO(b"cpu_fixture_complete=normal\n"), returncode=None)
        polls = [0]
        def poll():
            polls[0] += 1
            if poll_error:
                raise poll_error
            if polls[0] >= 2:
                process.returncode = 0
            return process.returncode
        process.poll.side_effect = poll
        tree = Mock(members={10: ({"pid": 10}, 55), 11: ({"pid": 11}, 56)}, cleanup_issues=[])
        tree.live_pids.side_effect = lambda: [10, 11] if process.returncode is None else []
        tree.alive.side_effect = lambda pid: process.returncode is None
        tree.stop.return_value = []
        prefixes = Mock(records={"x": {}}, capture=Mock(side_effect=prefix_error))
        ready = dict(pid=ready_pid, starttime=123, nonce="abc", mode="normal")
        with ExitStack() as stack:
            for target, attr, value in (
                (m.subprocess, "Popen", Mock(return_value=process)),
                (m.host, "OwnedTree", Mock(return_value=tree)),
                (m, "admitted_gate", Mock(return_value=dict(ready, pid=11))),
                (m.host, "validate_ready", Mock(return_value=ready)),
                (m.gate, "private_read", lambda path: Path(path).read_text()),
                (m, "component_paths", Mock(return_value=set())),
                (m, "PrefixStore", Mock(return_value=prefixes)),
                (m.time, "sleep", Mock()),
            ):
                stack.enter_context(patch.object(target, attr, value))
            result = m.run_case(Path("nsys"), directory, Path("gate.py"), Path("fixture.py"), "normal")
        return result, tree, process

    def test_session_keeps_profiler_status_separate_from_target_exit_code(self):
        result, tree, process = self.simulated_case()
        self.assertEqual(result["profiler_returncode"], 0)
        self.assertIsNone(result["target_exit_code"])
        self.assertEqual(result["target_exit_code_source"], "unavailable-grandchild-not-waited")
        self.assertTrue(result["exec_identity_verified"])
        self.assertTrue(result["target_exit_observed"])
        self.assertTrue(result["reader_finished"])
        self.assertEqual(result["issues"], [])
        tree.close.assert_called_once()

    def test_changed_pid_across_exec_fails_session_and_stops_owned_tree(self):
        result, tree, process = self.simulated_case(ready_pid=99)
        self.assertEqual(result["reason"], "supervision-failed")
        self.assertIn("exec did not preserve", result["issues"][0])
        tree.stop.assert_called_once()
        self.assertFalse(result.get("exec_identity_verified", False))

    def test_prefix_failure_does_not_skip_owned_tree_cleanup(self):
        result, tree, process = self.simulated_case(prefix_error=ValueError("bad prefix"))
        self.assertEqual(result["reason"], "supervision-failed")
        tree.stop.assert_called_once()
        tree.close.assert_called_once()
        self.assertTrue(any("bad prefix" in issue for issue in result["issues"]))

    def test_inherited_instrumentation_rejected_before_profiler_launch(self):
        with patch.dict(m.os.environ, {"NSYS_FOO": "1"}), \
             patch.object(m.host, "run_owned") as run:
            result = m.run(self.root, Path("nsys"))
        run.assert_not_called()
        self.assertEqual(result, 2)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertFalse(summary["gpu_capture_qualified"])
        self.assertFalse(summary["gpu_workload_executed"])
        Path(str(self.root) + ".tar.gz").unlink()


if __name__ == "__main__":
    unittest.main()
