#!/usr/bin/env python3
"""CPU-only offline archive analysis tests; no CUDA or external commands."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "speed-bench/analyze-sm75-failure-archive.py"
SPEC = importlib.util.spec_from_file_location("failure_archive", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
BOOT = "9c9247806852474889f012872d1656e6"
ROOT = "run/failure-context/"


def event(code, bdf="0000:03:00", time=2000, cursor="one", boot=BOOT):
    return {"MESSAGE": "NVRM: Xid (PCI:%s): %d, test" % (bdf, code),
            "_BOOT_ID": boot, "_SOURCE_MONOTONIC_TIMESTAMP": str(time),
            "__MONOTONIC_TIMESTAMP": str(time + 400),
            "__REALTIME_TIMESTAMP": "1788827774170793", "__CURSOR": cursor}


def jsonl(events):
    return "".join(json.dumps(item) + "\n" for item in events)


def summary(**overrides):
    result = {"boot_id": BOOT, "executable_sha256": analysis.EXPECTED_SHA256,
              "workload_start": {"monotonic_ns": 1000000},
              "finished": {"monotonic_ns": 30000000}, "commands": [],
              "collection": "partial", "workload_terminated": False,
              "issues": ["owned workload would not terminate; possible uninterruptible task"],
              "workload_returncode": None}
    result.update(overrides)
    return json.dumps(result)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "evidence.tar.gz"

    def archive(self, files=None, extra=()):
        members = {"summary.json": summary()}
        members.update(files or {})
        with tarfile.open(self.path, "w:gz") as output:
            for name, value in members.items():
                data = value.encode()
                member = tarfile.TarInfo(ROOT + name)
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))
            for member, data in extra:
                output.addfile(member, io.BytesIO(data))
        return self.path

    def test_latest_signal_order_dedup_and_partial_limits(self):
        gpu1 = event(79)
        gpu0 = event(175, bdf="0000:02:00", time=12000, cursor="two")
        path = self.archive({"kernel-live.jsonl": jsonl([gpu0, gpu1]),
                             "post/kernel.log": jsonl([gpu1]),
                             "application-timeline.jsonl": jsonl([
                                 {"monotonic_ns": 9000000, "line": "algo103-half0-submit"},
                                 {"monotonic_ns": 9500000, "line": "CUDA synchronize failed"}])})
        result = analysis.analyze(path)
        self.assertEqual([item["xid"] for item in result["ordered_capture_xids"]], [79, 175])
        self.assertEqual(result["excluded_xid_records"]["duplicates"], 1)
        self.assertEqual(result["capture"]["collection"], "partial")
        self.assertIsNone(result["capture"]["workload_returncode"])
        self.assertTrue(result["recorded_executable_matches_pinned_reproducer"])
        self.assertIn("indefinitely", " ".join(result["interpretation_limits"]))
        self.assertIn("Buffered output", " ".join(result["interpretation_limits"]))
        self.assertEqual(result["application_tail_observation_order"][0]["observer_monotonic_ns"], 9000000)

    def test_prior_boot_baseline_and_outside_window_not_current_faults(self):
        baseline = event(79, time=10, cursor="baseline")
        other_boot = event(79, cursor="other", boot="0" * 32)
        pre = event(31, time=999, cursor="pre")
        post = event(31, time=30001, cursor="later")
        current = event(79, cursor="current")
        result = analysis.analyze(self.archive({
            "kernel-baseline.log": jsonl([baseline]) + "-- cursor: baseline\n",
            "post/kernel.log": jsonl([baseline, other_boot, pre, post, current])}))
        self.assertEqual(len(result["ordered_capture_xids"]), 1)
        counts = result["excluded_xid_records"]
        self.assertEqual(counts["baseline"], 1)
        self.assertEqual(counts["other_or_unknown_boot"], 1)
        self.assertEqual(counts["outside_capture_window"], 2)

    def test_record_without_clock_does_not_invent_order(self):
        value = event(79)
        del value["_SOURCE_MONOTONIC_TIMESTAMP"]
        del value["__MONOTONIC_TIMESTAMP"]
        result = analysis.analyze(self.archive({"kernel-live.jsonl": jsonl([value])}))
        self.assertFalse(result["ordered_capture_xids"])
        self.assertEqual(result["excluded_xid_records"]["missing_clock_or_window"], 1)

    def test_receipt_fallback_and_no_causality_claim(self):
        value = event(79)
        del value["_SOURCE_MONOTONIC_TIMESTAMP"]
        result = analysis.analyze(self.archive({"kernel-live.jsonl": jsonl([value])}))
        self.assertEqual(result["ordered_capture_xids"][0]["ordering_clock"], "journal-receipt-fallback")
        self.assertIn("generation order", " ".join(result["interpretation_limits"]))

    def test_source_wins_over_receipt_clock_for_order(self):
        earlier = event(79, time=2000)
        earlier["__MONOTONIC_TIMESTAMP"] = "7000"
        later = event(175, time=3000, cursor="two")
        result = analysis.analyze(self.archive({"kernel-live.jsonl": jsonl([later, earlier])}))
        self.assertEqual([x["xid"] for x in result["ordered_capture_xids"]], [79, 175])

    def test_sysfs_and_pci_status_diff_preserve_missing_not_clear(self):
        result = analysis.analyze(self.archive({
            "pre/pci-0000_00_03.0.log": "  UESta: DLP- SDES- TLP-\n  DevSta: CorrErr- FatalErr-\n  RootCmd: CERptEn- FERptEn-\n",
            "post/pci-0000_00_03.0.log": "  UESta: DLP- SDES+ TLP-\n  DevSta: CorrErr- FatalErr+\n  RootCmd: CERptEn- FERptEn-\n",
            "pre/pci-0000_03_00.0.log": "  LnkSta: Speed 2.5GT/s, Width x16\n",
            "post/pci-0000_03_00.0.log": "03:00.0 Unknown header type 7f\n",
            "pre-sysfs.json": json.dumps({"devices": {"0000:03:00.0": {"current_link_width": "16\n"}}}),
            "post-sysfs.json": json.dumps({"devices": {"0000:03:00.0": {"current_link_width": "255\n"}}})}))
        port, endpoint = result["pci_comparison"]
        self.assertEqual(port["field_changes"]["UESta"]["after"], ["DLP- SDES+ TLP-"])
        self.assertNotIn("RootCmd", port["field_changes"])
        self.assertIsNone(endpoint["field_changes"]["LnkSta"]["after"])
        self.assertEqual(result["sysfs_changes"]["0000:03:00.0"]["current_link_width"]["after"], "255\n")

    def test_report_text_and_nested_archive_not_searched_for_events(self):
        misleading = tarfile.TarInfo(ROOT + "nvidia-report/old-history.txt")
        data = jsonl([event(79, cursor="history")]).encode()
        misleading.size = len(data)
        result = analysis.analyze(self.archive(extra=[(misleading, data)]))
        self.assertEqual(result["ordered_capture_xids"], [])

    def test_invalid_json_line_reported_not_silently_success(self):
        result = analysis.analyze(self.archive({"kernel-live.jsonl": "{broken\n" + jsonl([event(79)])}))
        self.assertEqual(len(result["parse_issues"]), 1)
        self.assertEqual(len(result["ordered_capture_xids"]), 1)

    def test_hash_recomputed_and_result_deterministic(self):
        path = self.archive()
        first = analysis.analyze(path)
        self.assertEqual(first, analysis.analyze(path))
        self.assertEqual(first["archive_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_wrong_executable_hash_is_recorded_not_overridden(self):
        result = analysis.analyze(self.archive({"summary.json": summary(executable_sha256="0" * 64)}))
        self.assertFalse(result["recorded_executable_matches_pinned_reproducer"])

    def test_duplicate_member_rejected(self):
        duplicate = tarfile.TarInfo(ROOT + "summary.json")
        duplicate.size = 2
        with self.assertRaisesRegex(analysis.EvidenceError, "duplicate"):
            analysis.analyze(self.archive(extra=[(duplicate, b"{}")]))

    def test_traversal_absolute_windows_and_backslash_names_rejected(self):
        for name in ("../outside", "/etc/passwd", "C:/outside", "C:\\outside", "run/../../outside"):
            with self.subTest(name=name):
                member = tarfile.TarInfo(name)
                with self.assertRaisesRegex(analysis.EvidenceError, "unsafe"):
                    analysis.analyze(self.archive(extra=[(member, b"")]))

    def test_symlink_hardlink_and_device_rejected_even_unselected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE):
            with self.subTest(kind=kind):
                member = tarfile.TarInfo("unselected")
                member.type = kind
                member.linkname = "/etc/passwd"
                with self.assertRaisesRegex(analysis.EvidenceError, "special archive"):
                    analysis.analyze(self.archive(extra=[(member, b"")]))

    def test_all_size_and_count_limits_enforced(self):
        path = self.archive({"kernel-live.jsonl": jsonl([event(79)])})
        for constant in ("MAX_MEMBERS", "MAX_MEMBER_BYTES", "MAX_TOTAL_BYTES", "MAX_LINE_BYTES"):
            with self.subTest(limit=constant):
                if constant == "MAX_LINE_BYTES":
                    path = self.archive({"kernel-live.jsonl": jsonl([event(79)])})
                with patch.object(analysis, constant, 1), self.assertRaises(analysis.EvidenceError):
                    analysis.analyze(path)

    def test_extra_capture_root_rejected(self):
        member = tarfile.TarInfo("another/failure-context/summary.json")
        member.size = 2
        with self.assertRaisesRegex(analysis.EvidenceError, "exactly one"):
            analysis.analyze(self.archive(extra=[(member, b"{}")]))

    def test_preflight_has_no_workload_window_or_claimed_xids(self):
        result = analysis.analyze(self.archive({"summary.json": summary(workload_start={},
            workload="not-started", collection="complete", preflight="passed", issues=[])}))
        self.assertEqual(result["capture"]["workload"], "not-started")
        self.assertFalse(result["ordered_capture_xids"])

    def test_collector_failures_distinct_from_workload_failure(self):
        result = analysis.analyze(self.archive({"summary.json": summary(commands=[
            {"name": "pre/sudo", "returncode": 0, "timeout": False},
            {"name": "post/kernel", "returncode": 124, "timeout": True}])}))
        self.assertEqual(result["failed_collector_commands"], ["post/kernel"])

    def test_later_zombie_does_not_mean_permanent_uninterruptible_process(self):
        result = analysis.analyze(self.archive({
            "summary.json": summary(workload_pid=4312),
            "post/processes.log": "    PID PPID STARTED STAT WCHAN COMMAND\n"
                " 4312 4221 Tue Sep  8 00:36:11 2026 Zs - cuda_sm75_token\n"}))
        record = result["post_workload_process"]["matching_pid_record"]
        self.assertEqual(record["state"], "Zs")
        self.assertEqual(record["parent_pid"], 4221)
        self.assertIn("exited, awaiting reaping", " ".join(result["interpretation_limits"]))
        self.assertIsNone(result["capture"]["workload_returncode"])

    def test_missing_post_process_is_not_reported_as_exit(self):
        result = analysis.analyze(self.archive({"summary.json": summary(workload_pid=4312)}))
        self.assertFalse(result["post_workload_process"]["snapshot_present"])
        self.assertIsNone(result["post_workload_process"]["matching_pid_record"])

    def test_missing_capture_rejected(self):
        with tarfile.open(self.path, "w:gz"):
            pass
        with self.assertRaises(analysis.EvidenceError):
            analysis.analyze(self.path)


if __name__ == "__main__":
    unittest.main()
