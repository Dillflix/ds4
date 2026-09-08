#!/usr/bin/env python3
"""CPU-only qualification of exec admission, temporary prefixes and report export.

No option selects a CUDA executable. This tests the capture plumbing before it
can be integrated with the frozen GPU1 reproducer. It is not GPU qualification.
"""
import argparse
from contextlib import closing
import importlib.util
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time


def load_sibling(name):
    path = Path(__file__).resolve().with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


host = load_sibling("qualify-sm75-nsys-host")
gate = load_sibling("sm75-nsys-launch-gate")
PREFIX_LIMIT = 16 * 1024 * 1024
PREFIX_TOTAL = 128 * 1024 * 1024
FILE_COUNT_LIMIT = 64


class PrefixStore:
    """Bounded, explicitly nontransactional prefixes; never follow a symlink.

    One latest prefix per pathname is kept, even when Nsight removes its source.
    Rewrites/replacements are recorded, not assumed append-only. Caps are not a
    quota on Nsight itself; the supervisor separately checks its live files.
    """
    def __init__(self, source, destination):
        self.source, self.destination = Path(source), Path(destination)
        self.destination.mkdir()
        self.records = {}
        self.issues = []

    def capture(self):
        for directory, folders, files in os.walk(self.source, followlinks=False):
            for name in list(folders):
                if (Path(directory) / name).is_symlink():
                    raise ValueError("symlink in Nsight temporary directory")
            for name in sorted(files):
                path = Path(directory) / name
                relative = str(path.relative_to(self.source))
                if relative not in self.records and len(self.records) >= FILE_COUNT_LIMIT:
                    raise ValueError("Nsight temporary file count limit")
                # Linux O_NOFOLLOW also checks final component after directory
                # traversal. Directory ancestors live under our private root.
                if any(p.is_symlink() for p in path.parents if p != self.source.parent):
                    raise ValueError("symlink ancestor of Nsight temporary file")
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                except FileNotFoundError:
                    continue
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                        raise ValueError("nonordinary Nsight temporary file")
                    old = self.records.get(relative)
                    signature = [before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns]
                    if old and old["observed_signature"] == signature:
                        continue
                    used = sum(r["retained_bytes"] for r in self.records.values())
                    budget = PREFIX_TOTAL - used + (old["retained_bytes"] if old else 0)
                    amount = min(before.st_size, PREFIX_LIMIT, budget)
                    data = stream.read(amount)
                    after = os.fstat(stream.fileno())
                slot = old["prefix_file"] if old else "prefix-%03d.bin" % len(self.records)
                temporary = self.destination / (slot + ".next")
                with temporary.open("xb") as output:
                    output.write(data)
                    output.flush()
                os.replace(temporary, self.destination / slot)
                self.records[relative] = {
                    "source": relative, "prefix_file": slot,
                    "observed_signature": signature, "observed_bytes": before.st_size,
                    "retained_bytes": len(data), "sha256": host.digest(self.destination / slot),
                    "captures": (old["captures"] if old else 0) + 1,
                    "source_changed_during_copy": before.st_size != after.st_size or
                        before.st_mtime_ns != after.st_mtime_ns,
                    "truncated": len(data) < before.st_size,
                    "completeness": "unknown-nontransactional-prefix",
                    "monotonic_ns": time.monotonic_ns()}
        host.write_json(self.destination / "manifest.json", list(self.records.values()))


def gated_command(nsys, directory, gate_path, config, nonce):
    target = [sys.executable, "-I", str(gate_path), str(config)]
    return [str(nsys), "profile", *host.HOST_OPTIONS,
            "--session-new=sm75_retention_" + nonce,
            "--output=" + str(directory / "report"), *target], target


def admitted_gate(path, tree, argv, nonce):
    ready = json.loads(gate.private_read(path))
    pid = ready.get("pid")
    if type(pid) is not int or pid not in tree.members or not tree.alive(pid):
        raise ValueError("launch gate is not a live owned descendant")
    info = host.process_info(pid)
    if (ready.get("nonce") != nonce or ready.get("state") != "waiting-before-exec" or
            ready.get("starttime") != info["starttime"] or
            not host.same_process(tree.members[pid][0], info)):
        raise ValueError("launch gate identity mismatch")
    base = Path("/proc") / str(pid)
    actual = [os.fsdecode(x) for x in (base / "cmdline").read_bytes().split(b"\0") if x]
    if (actual != argv or (base / "exe").resolve() != Path(argv[0]).resolve()):
        raise ValueError("launch gate executable or argv mismatch")
    if host.GPU_LIB.search((base / "maps").read_text()):
        raise ValueError("CPU launch gate mapped a GPU library")
    return ready


def component_paths(pids):
    """Observed profiler executables/libraries, not a claim of complete coverage."""
    paths = set()
    for pid in pids:
        try:
            base = Path("/proc") / str(pid)
            paths.add(str((base / "exe").resolve(strict=True)))
            for line in (base / "maps").read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[5].startswith("/"):
                    path = fields[5]
                    if any(x in path.lower() for x in ("nsight", "nsys", "cupti", "toolsinjection")):
                        paths.add(path)
        except (OSError, ValueError):
            pass  # Short-lived processes may exit before their mappings are read.
    return paths


def run_case(nsys, directory, gate_path, fixture, mode):
    nonce = secrets.token_hex(16)
    executable = Path(sys.executable).resolve()
    arguments = ["-I", str(fixture), str(directory), nonce, mode]
    config = directory / "gate-config.json"
    host.write_json(config, {"nonce": nonce, "executable": str(executable),
                            "sha256": host.digest(executable), "arguments": arguments})
    argv, gate_argv = gated_command(nsys, directory, gate_path, config, nonce)
    tmp = directory / "live-tmp"
    tmp.mkdir()
    prefixes = PrefixStore(tmp, directory / "retained-prefixes")
    environment = os.environ.copy()
    environment.update({"NSYS_TMPDIR": str(tmp), "TMPDIR": str(tmp)})
    result = {"mode": mode, "argv": argv, "profiler_returncode": None,
              "target_exit_observed": False, "target_exit_code": None,
              "target_exit_code_source": "unavailable-grandchild-not-waited",
              "gpu_workload_executed": False, "issues": [], "observed_component_paths": []}
    overflow = threading.Event()
    errors = []
    tree, process, reader = None, None, None
    started = time.monotonic()
    components = set()
    try:
        process = subprocess.Popen(argv, cwd=directory, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        result["profiler_pid"] = process.pid
        tree = host.OwnedTree(process.pid)

        def drain():
            size = 0
            try:
                with process.stdout, (directory / "console.log").open("xb") as stream:
                    while data := process.stdout.read1(16384):
                        stream.write(data[:max(0, host.LOG_LIMIT - size)])
                        stream.flush()
                        size += len(data)
                        if size > host.LOG_LIMIT:
                            overflow.set()
            except Exception as error:
                errors.append(str(error))

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        admitted, ready, ready_at, term_at = None, None, None, None
        last_copy = -1
        with (directory / "processes.jsonl").open("x") as evidence:
            while True:
                now = time.monotonic()
                tree.scan()
                process.poll()
                live = tree.live_pids()
                evidence.write(json.dumps({"monotonic": now, "live": live,
                    "owned": [i for i, _ in tree.members.values()]}) + "\n")
                evidence.flush()
                components.update(component_paths(live))
                if admitted is None and (directory / "gate-ready.json").exists():
                    try:
                        admitted = admitted_gate(directory / "gate-ready.json", tree, gate_argv, nonce)
                    except json.JSONDecodeError:
                        pass
                    if admitted:
                        # Release only after independently verifying the CPU
                        # target executable; launcher checks it again before exec.
                        if host.digest(executable) != json.loads(gate.private_read(config))["sha256"]:
                            raise ValueError("target executable changed before gate release")
                        result["admitted_target"] = admitted
                        with (directory / "gate-release").open("x") as release:
                            release.write(nonce)
                if ready is None and (directory / "ready.json").exists():
                    try:
                        ready = host.validate_ready(directory / "ready.json", tree,
                            [str(executable), *arguments], nonce, mode)
                    except json.JSONDecodeError:
                        pass
                    if ready:
                        if (not admitted or ready["pid"] != admitted["pid"] or
                                ready["starttime"] != admitted["starttime"]):
                            raise ValueError("exec did not preserve the admitted process identity")
                        result["exec_identity_verified"] = True
                        result["target"] = {k: v for k, v in ready.items() if k != "maps"}
                        ready_at = now
                        if mode != "timeout":
                            with (directory / "release").open("x") as release:
                                release.write(nonce)
                if mode == "timeout" and ready_at and now - ready_at >= 1:
                    if term_at is None:
                        tree.send(ready["pid"], signal.SIGTERM)
                        term_at = now
                        result["target_term_sent"] = True
                    elif now - term_at >= 1 and tree.alive(ready["pid"]):
                        tree.send(ready["pid"], signal.SIGKILL)
                        result["target_kill_sent"] = True
                if admitted and not tree.alive(admitted["pid"]):
                    result["target_exit_observed"] = True
                if now - last_copy >= 0.1:
                    prefixes.capture()
                    last_copy = now
                    _, excluded = host.safe_files(directory)
                    if excluded:
                        raise RuntimeError("artifact size/type limit: " + repr(excluded))
                if overflow.is_set() or errors:
                    raise RuntimeError("profiler console retention failed")
                if process.returncode is not None and not tree.live_pids():
                    result["reason"] = "processes-exited"
                    break
                if now - started >= 60:
                    raise TimeoutError("CPU supervisor deadline; no automatic retry")
                time.sleep(0.05)
    except BaseException as error:
        result["issues"].append(str(error))
        result["reason"] = "supervision-failed"
        # Target first; give the profiler a bounded opportunity to finalize.
        admitted = result.get("admitted_target")
        if tree and admitted and tree.alive(admitted["pid"]):
            tree.send(admitted["pid"], signal.SIGTERM)
            time.sleep(0.1)
            tree.send(admitted["pid"], signal.SIGKILL)
            until = time.monotonic() + 5
            while process.poll() is None and time.monotonic() < until:
                time.sleep(0.05)
    finally:
        # Capture before stopping the profiler too: it may remove temporaries.
        try:
            prefixes.capture()
        except Exception as error:
            result["issues"].append("prefix retention: " + str(error))
        if tree:
            result["surviving_owned_pids"] = tree.stop()
            result["issues"].extend(tree.cleanup_issues)
            admitted = result.get("admitted_target")
            if admitted:
                result["target_exit_observed"] = not tree.alive(admitted["pid"])
            tree.close()
        elif process:
            process.kill()  # Unreaped Popen child, no arbitrary PID or name.
        if process:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                result["issues"].append("profiler not reaped")
            result["profiler_returncode"] = process.poll()
        if reader:
            reader.join(timeout=2)
        result["reader_finished"] = reader is not None and not reader.is_alive()
        result["issues"].extend(errors)
        result["log_overflow"] = overflow.is_set()
        result["elapsed_seconds"] = time.monotonic() - started
        result["observed_component_paths"] = sorted(components)
        result["temporary_prefix_count"] = len(prefixes.records)
        result["qdstrm_prefix_count"] = sum(
            r.get("source", "").endswith(".qdstrm") and r.get("retained_bytes", 0) > 0
            for r in prefixes.records.values())
        host.write_json(directory / "session.json", result)
    return result


def sqlite_inventory(path):
    # Export has finished. Read metadata only; no undocumented table assumptions.
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        deadline = time.monotonic() + 5
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchmany(1001)
        if len(tables) > 1000:
            raise ValueError("SQLite table count limit")
        return [{"table": name, "columns": [row[1] for row in connection.execute(
            'PRAGMA table_info("' + name.replace('"', '""') + '")').fetchmany(1001)]}
            for (name,) in tables]


def export_report(nsys, directory):
    report = directory / "report.nsys-rep"
    result = {"device_record_completeness": "not-tested-cpu-only",
              "intermediate_importability": "unknown-prefixes-not-imported"}
    if not report.is_file() or report.is_symlink():
        result["status"] = "report-absent"
        return result
    export = directory / "export"
    export.mkdir()
    copy = export / "retained.nsys-rep"
    shutil.copyfile(report, copy)
    db = export / "report.sqlite"
    result["report_sha256"] = host.digest(copy)
    run = host.run_owned([str(nsys), "export", "--type=sqlite", "--output=" + str(db),
                          str(copy)], export, seconds=60)
    result["process"] = run
    if (run["returncode"] == 0 and run["reason"] == "processes-exited" and
            run["reader_finished"] and not run["log_overflow"] and not run["reader_errors"] and
            not run["cleanup_issues"] and not run["surviving_owned_pids"] and db.is_file()):
        result["schema"] = sqlite_inventory(db)
        result["sqlite_sha256"] = host.digest(db)
        result["status"] = "exported"
    else:
        result["status"] = "export-failed-or-incomplete"
    return result


def validate_case(result, console):
    if (result.get("reason") != "processes-exited" or result.get("issues") or
            not result.get("exec_identity_verified") or not result.get("target_exit_observed") or
            result.get("surviving_owned_pids") or not result.get("reader_finished") or
            result.get("log_overflow")):
        raise RuntimeError("CPU exec/retention supervision failed")
    if result["mode"] == "normal" and (result["profiler_returncode"] != 0 or
            "cpu_fixture_complete=normal" not in console):
        raise RuntimeError("normal CPU fixture did not complete")
    if result["mode"] == "abrupt" and result["profiler_returncode"] in (None, 0):
        raise RuntimeError("abrupt fixture loss was not reflected by profiler")
    if result["mode"] == "timeout" and not result.get("target_kill_sent"):
        raise RuntimeError("timeout did not exercise target hard stop")


def run(output, nsys):
    summary = {"retention_qualification": "in-progress", "gpu_capture_qualified": False,
               "gpu_workload_executed": False, "cases": [], "issues": []}
    try:
        if "%" in str(output):
            raise ValueError("Nsight percent substitutions not permitted in output path")
        for key, value in os.environ.items():
            if value and (key in host.FORBIDDEN_ENV or key.startswith(("CUPTI_", "NSYS_", "CUBLAS_LOG"))):
                raise ValueError("unexpected inherited instrumentation control: " + key)
        sources = [Path(__file__).resolve(), Path(host.__file__), Path(gate.__file__),
            Path(__file__).resolve().parent.parent / "tests/fixtures/sm75-nsys/cpu_target.py"]
        summary["source_sha256"] = {}
        for source in sources:
            shutil.copyfile(source, output / source.name)
            summary["source_sha256"][source.name] = host.digest(source)
        summary["nsys"] = {"path": str(nsys), "sha256": host.digest(nsys)}
        summary["python"] = {"path": sys.executable, "sha256": host.digest(sys.executable),
                             "version": sys.version}
        for label, opts in (("version", ["--version"]), ("profile-help", ["profile", "--help"]),
                            ("export-help", ["export", "--help"])):
            directory = output / label
            directory.mkdir()
            result = host.run_owned([str(nsys), *opts], directory, seconds=20)
            if (result["returncode"] != 0 or result["reason"] != "processes-exited" or
                    result["surviving_owned_pids"] or result["cleanup_issues"] or
                    not result["reader_finished"] or result["reader_errors"] or result["log_overflow"]):
                raise RuntimeError("Nsight metadata query failed: " + label)
        host.validate_help((output / "profile-help/console.log").read_text(errors="replace"))
        for mode in ("normal", "abrupt", "timeout"):
            print("CPU-only Nsight exec/retention qualification: " + mode, flush=True)
            directory = output / mode
            directory.mkdir()
            result = run_case(nsys, directory, output / "sm75-nsys-launch-gate.py",
                              output / "cpu_target.py", mode)
            summary["cases"].append(result)
            validate_case(result, (directory / "console.log").read_text(errors="replace"))
            result["export"] = export_report(nsys, directory)
            if result["export"]["status"] != "exported":
                raise RuntimeError("CPU report export not qualified: " + mode)
        if not any(c["qdstrm_prefix_count"] for c in summary["cases"]):
            raise RuntimeError("no nonempty .qdstrm prefix retained; retention remains unqualified")
        summary["observed_components"] = []
        paths = set().union(*(set(c["observed_component_paths"]) for c in summary["cases"]))
        for path in sorted(paths):
            try:
                summary["observed_components"].append({"path": path, "sha256": host.digest(path)})
            except OSError as error:
                summary["observed_components"].append({"path": path, "error": str(error)})
        summary["component_coverage"] = "observed-only; short-lived and CUDA injection components may be absent"
        if host.digest(nsys) != summary["nsys"]["sha256"]:
            raise RuntimeError("Nsight entry point changed during qualification")
        summary["retention_qualification"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        summary["retention_qualification"] = "failed"
        summary["issues"].append(str(error))
        print("Retention qualification failed: " + str(error), file=sys.stderr, flush=True)
    finally:
        files, excluded = host.safe_files(output)
        summary["excluded_from_archive"] = excluded
        summary["artifacts"] = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size,
                                 "sha256": host.digest(p)} for p in files]
        if excluded:
            summary["retention_qualification"] = "failed"
            summary["issues"].append("unsafe/oversized evidence excluded; originals kept")
        host.write_json(output / "summary.json", summary)
        with tarfile.open(str(output) + ".tar.gz", "x:gz") as archive:
            for path in [*files, output / "summary.json"]:
                archive.add(path, arcname=str(Path(output.name) / path.relative_to(output)), recursive=False)
        print("Archive to return: " + str(output) + ".tar.gz", flush=True)
    return 0 if summary["retention_qualification"] == "passed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys", default="nsys")
    args = parser.parse_args()
    if (platform.system() != "Linux" or platform.machine() != "x86_64" or
            not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal")):
        parser.error("Linux x86-64 with Python pidfds required; nothing started")
    nsys = shutil.which(args.nsys)
    if not nsys:
        parser.error("installed nsys not found; nothing installed")
    os.umask(0o077)
    def interrupted(*_):
        raise KeyboardInterrupt("CPU retention qualification interrupted")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    output = Path(tempfile.mkdtemp(prefix="sm75-nsys-retention-", dir=Path.cwd())).resolve()
    print("CPU-only qualification directory: " + str(output), flush=True)
    return run(output, Path(nsys).resolve())


if __name__ == "__main__":
    sys.exit(main())
