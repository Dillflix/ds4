#!/usr/bin/env python3
"""Offline, read-only JSON summary of a capture-sm75-gpu1-failure archive.

Never extracts files, executes archive contents, or contacts GPUs/the network.
The input is evidence, not instructions. stdout is deterministic JSON; this is
not a root-cause classifier or a replacement for the original support archive.
"""
import argparse
import hashlib
import json
from pathlib import PurePosixPath
import re
import sys
import tarfile


MAX_MEMBERS = 2048
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
EXPECTED_SHA256 = "5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414"
XID = re.compile(r"NVRM: Xid\s*\(PCI:([0-9a-f:.]+)\)\s*:\s*(\d+)", re.I)
PCI_MEMBER = re.compile(r"^(pre|post)/pci-([0-9a-f_]+\.[0-7])\.log$")
PCI_FIELD = re.compile(r"^(?:Lnk(?:Cap|Sta|Ctl)2?|Dev(?:Sta|Ctl)|UESta|UEMsk|UESvrt|CESta|CEMsk|RootCmd|RootSta|Status|Secondary status):")
SELECTED = {"summary.json", "kernel-baseline.log", "kernel-live.jsonl", "post/kernel.log",
            "application-timeline.jsonl", "pre-sysfs.json", "post-sysfs.json", "post/processes.log"}


class EvidenceError(ValueError):
    pass


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or len(name) > 4096 or "\\" in name or ":" in name or
            path.is_absolute() or ".." in path.parts):
        raise EvidenceError("unsafe archive member name")
    return path.as_posix()


def read_archive(path):
    """Stream with limits; validate every member, even ones we do not inspect."""
    selected = {}
    seen = set()
    total = 0
    count = 0
    with open(path, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        with tarfile.open(fileobj=source, mode="r|*") as archive:
            for member in archive:
                count += 1
                if count > MAX_MEMBERS:
                    raise EvidenceError("archive member-count limit exceeded")
                name = safe_name(member.name)
                if name in seen:
                    raise EvidenceError("duplicate archive member: " + name)
                seen.add(name)
                if not (member.isfile() or member.isdir()) or member.issparse():
                    raise EvidenceError("links, sparse files and special archive members are not supported")
                if not 0 <= member.size <= MAX_MEMBER_BYTES:
                    raise EvidenceError("archive member-size limit exceeded")
                total += member.size
                if total > MAX_TOTAL_BYTES:
                    raise EvidenceError("archive expanded-size limit exceeded")
                parts = PurePosixPath(name).parts
                if "failure-context" not in parts or not member.isfile():
                    continue
                position = parts.index("failure-context")
                relative = "/".join(parts[position + 1:])
                if relative not in SELECTED and not PCI_MEMBER.fullmatch(relative):
                    continue
                stream = archive.extractfile(member)  # Reads bytes only; never extracts to disk.
                content = stream.read(MAX_MEMBER_BYTES + 1)
                if len(content) != member.size:
                    raise EvidenceError("truncated or oversized member: " + name)
                selected[name] = content.decode("utf-8", errors="strict")
    summaries = [name for name in selected if name.endswith("/failure-context/summary.json")]
    if len(summaries) != 1:
        raise EvidenceError("exactly one failure-context/summary.json is required")
    prefix = summaries[0][:-len("summary.json")]
    if any(not name.startswith(prefix) for name in selected):
        raise EvidenceError("multiple capture roots are not supported")
    return {name[len(prefix):]: value for name, value in selected.items()}, digest


def object_json(text, name):
    try:
        value = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise EvidenceError("invalid JSON: " + name) from error
    if not isinstance(value, dict):
        raise EvidenceError("expected JSON object: " + name)
    return value


def records(text, name, issues):
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("-- cursor: "):
            continue
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            raise EvidenceError("JSON line-size limit exceeded: " + name)
        try:
            value = object_json(line, name)
        except EvidenceError:
            issues.append("unparsed line %d in %s" % (number, name))
            continue
        yield value


def integer(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,20}", value):
        return int(value)
    return None


def boot_id(value):
    if not isinstance(value, str):
        return None
    canonical = value.replace("-", "").lower()
    return canonical if re.fullmatch(r"[0-9a-f]{32}", canonical) else None


def event_key(event):
    # A cursor identifies the same journal record repeated in live/post copies.
    return event.get("__CURSOR") or json.dumps(
        [event.get(key) for key in ("_BOOT_ID", "_SOURCE_MONOTONIC_TIMESTAMP",
                                   "__MONOTONIC_TIMESTAMP", "MESSAGE")], sort_keys=True)


def canonical_xids(files, summary, issues):
    baseline = {event_key(event) for event in records(files.get("kernel-baseline.log", ""),
                                                    "kernel-baseline.log", issues)}
    expected_boot = boot_id(summary.get("boot_id"))
    start = integer(summary.get("workload_start", {}).get("monotonic_ns"))
    finish = integer(summary.get("finished", {}).get("monotonic_ns"))
    ignored = {"baseline": 0, "other_or_unknown_boot": 0, "outside_capture_window": 0,
               "missing_clock_or_window": 0, "duplicates": 0}
    events = {}
    for name in ("kernel-live.jsonl", "post/kernel.log"):
        for event in records(files.get(name, ""), name, issues):
            message = event.get("MESSAGE")
            match = XID.search(message) if isinstance(message, str) else None
            if not match:
                continue
            key = event_key(event)
            if key in baseline:
                ignored["baseline"] += 1
                continue
            if not expected_boot or boot_id(event.get("_BOOT_ID")) != expected_boot:
                ignored["other_or_unknown_boot"] += 1
                continue
            source_us = integer(event.get("_SOURCE_MONOTONIC_TIMESTAMP"))
            receipt_us = integer(event.get("__MONOTONIC_TIMESTAMP"))
            time_us = source_us if source_us is not None else receipt_us
            if time_us is None or start is None or finish is None:
                ignored["missing_clock_or_window"] += 1
                continue
            if not start <= time_us * 1000 <= finish:
                ignored["outside_capture_window"] += 1
                continue
            if key in events:
                ignored["duplicates"] += 1
                events[key]["evidence_members"].append(name)
                continue
            events[key] = {"xid": int(match.group(2)), "pci_device": match.group(1).lower(),
                           "message": message, "boot_id": expected_boot,
                           "source_monotonic_us": source_us,
                           "journal_receipt_monotonic_us": receipt_us,
                           "journal_receipt_realtime_us": integer(event.get("__REALTIME_TIMESTAMP")),
                           "ordering_clock": "source" if source_us is not None else "journal-receipt-fallback",
                           "order_monotonic_us": time_us, "evidence_members": [name]}
    return sorted(events.values(), key=lambda event: (event["order_monotonic_us"],
                  event["pci_device"], event["xid"], event["message"])), ignored


def pci_fields(text):
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if PCI_FIELD.match(line):
            label, value = line.split(":", 1)
            result.setdefault(label, []).append(value.strip())
    return result


def compare_pci(files):
    snapshots = {}
    for name, text in files.items():
        match = PCI_MEMBER.fullmatch(name)
        if match:
            phase, encoded_bdf = match.groups()
            snapshots.setdefault(encoded_bdf.replace("_", ":"), {})[phase] = pci_fields(text)
    result = []
    for bdf, pair in sorted(snapshots.items()):
        before, after = pair.get("pre", {}), pair.get("post", {})
        changes = {field: {"before": before.get(field), "after": after.get(field)}
                   for field in sorted(before.keys() | after.keys())
                   if before.get(field) != after.get(field)}
        result.append({"pci_device": bdf, "pre_present": "pre" in pair,
                       "post_present": "post" in pair, "field_changes": changes,
                       "pre_fields": before, "post_fields": after})
    return result


def post_process(files, summary):
    """Read the recorded ps snapshot, never query or signal a live process."""
    pid = integer(summary.get("workload_pid"))
    result = {"snapshot_present": "post/processes.log" in files,
              "workload_pid": pid, "matching_pid_record": None}
    for line in files.get("post/processes.log", "").splitlines():
        columns = line.split()
        # ps -eo pid,ppid,lstart,stat,wchan:32,comm: lstart occupies five fields.
        if len(columns) >= 10 and integer(columns[0]) == pid and pid is not None:
            result["matching_pid_record"] = {
                "parent_pid": integer(columns[1]), "recorded_start": " ".join(columns[2:7]),
                "state": columns[7], "wait_channel": columns[8], "command": " ".join(columns[9:])}
            break
    return result


def analyze(path):
    files, archive_hash = read_archive(path)
    summary = object_json(files["summary.json"], "summary.json")
    issues = []
    xids, ignored = canonical_xids(files, summary, issues)
    applications = list(records(files.get("application-timeline.jsonl", ""),
                                "application-timeline.jsonl", issues))
    failed_commands = [record.get("name") for record in summary.get("commands", [])
                       if record.get("returncode") != 0 or record.get("timeout")]
    # Retain observation order, not a causally misleading merge with kernel clocks.
    application_tail = [{"observer_monotonic_ns": record.get("monotonic_ns"),
                         "observer_utc_ns": record.get("utc_ns"), "line": record.get("line")}
                        for record in applications[-20:]]
    sysfs_changes = {}
    if "pre-sysfs.json" in files and "post-sysfs.json" in files:
        pre = object_json(files["pre-sysfs.json"], "pre-sysfs.json").get("devices", {})
        post = object_json(files["post-sysfs.json"], "post-sysfs.json").get("devices", {})
        for bdf in sorted(pre.keys() | post.keys()):
            before, after = pre.get(bdf, {}), post.get(bdf, {})
            sysfs_changes[bdf] = {key: {"before": before.get(key), "after": after.get(key)}
                                  for key in sorted(before.keys() | after.keys())
                                  if before.get(key) != after.get(key)}
    observed_first = summary.get("first_fault") or {}
    post_workload_process = post_process(files, summary)
    warnings = [
        "Application timestamps mark collector receipt, not CUDA submission/execution. Buffered output cannot establish the causal instruction.",
        "Kernel source, journal receipt and collector observation clocks are preserved separately; ordered Xids do not establish root cause.",
        "PCI snapshots are sequential, not atomic. Missing post fields indicate unavailable evidence, not cleared error bits.",
        "Recorded executable/library hashes are claims in the capture; the executable is not present for independent rehashing. Only archive SHA256 is independently calculated.",
        "This offline utility executes nothing and does not qualify hardware, GPU stability, or a future test."]
    if summary.get("workload_terminated") is False:
        warnings.append("Termination was not confirmed within the collector stop bound; this does not prove the process remained alive indefinitely.")
    post_record = post_workload_process["matching_pid_record"]
    if post_record and post_record["state"].startswith("Z"):
        warnings.append("The later ps snapshot records the workload PID as a zombie (exited, awaiting reaping), not a still-running or uninterruptible workload.")
    if any(event["source_monotonic_us"] is None for event in xids):
        warnings.append("Some Xids lack source timestamps; fallback journal-receipt ordering may not reflect generation order.")
    return {"schema_version": 1, "archive_sha256": archive_hash,
            "recorded_boot_id": summary.get("boot_id"),
            "recorded_executable_sha256": summary.get("executable_sha256"),
            "recorded_executable_matches_pinned_reproducer": summary.get("executable_sha256") == EXPECTED_SHA256,
            "recorded_loaded_library_sha256": summary.get("loaded_library_sha256", {}),
            "capture": {key: summary.get(key) for key in ("preflight", "collection", "workload",
                        "workload_returncode", "workload_terminated", "issues", "workload_start",
                        "stop_requested", "workload_end", "workload_final_poll", "late_exit_observed", "finished")},
            "failed_collector_commands": failed_commands,
            "post_workload_process": post_workload_process,
            "first_fault_observer_monotonic_ns": observed_first.get("monotonic_ns"),
            "xid_window": "workload launch through collector finish, including postmortem",
            "ordered_capture_xids": xids, "excluded_xid_records": ignored,
            "application_tail_observation_order": application_tail,
            "pci_comparison": compare_pci(files), "sysfs_changes": sysfs_changes,
            "parse_issues": issues, "interpretation_limits": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="capture .tar.gz (read-only)")
    args = parser.parse_args(argv)
    try:
        result = analyze(args.archive)
    except (EvidenceError, OSError, tarfile.TarError, UnicodeError, TypeError, AttributeError) as error:
        print("archive analysis failed: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
