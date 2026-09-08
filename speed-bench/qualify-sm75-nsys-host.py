#!/usr/bin/env python3
"""Qualify Nsight process supervision using CPU fixtures, never a GPU workload.

This deliberately has NO option to execute the frozen CUDA reproducer. Native
results must be reviewed before adding a profiler to the failure collector.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import select
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

LOG_LIMIT = 4 * 1024 * 1024
ARTIFACT_LIMIT = 256 * 1024 * 1024
TOTAL_LIMIT = 512 * 1024 * 1024
FORBIDDEN_ENV = ("LD_PRELOAD", "LD_AUDIT", "CUDA_INJECTION64_PATH",
                 "CUDA_INJECTION32_PATH", "NVTX_INJECTION64_PATH")
HOST_OPTIONS = (
    "--trace=none", "--sample=none", "--cpuctxsw=none", "--event-sample=none",
    "--gpu-metrics-devices=none", "--gpu-video-devices=none", "--gpuctxsw=false",
    "--trace-fork-before-exec=false", "--wait=all", "--kill=none",
    "--stop-on-exit=true", "--force-overwrite=false", "--export=none",
    "--stats=false", "--show-output=true",
)
GPU_LIB = re.compile(r"/(?:libcuda[.]|libcudart[.]|libcublas|libnvidia|libcupti)")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    # Only our freshly created private output directory is written.
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def process_info(pid, proc=Path("/proc")):
    base = proc / str(pid)
    text = (base / "stat").read_text()
    fields = text.rsplit(")", 1)[1].split()
    return {"pid": pid, "ppid": int(fields[1]), "starttime": int(fields[19]),
            "state": fields[0], "uid": base.stat().st_uid}


def same_process(before, after):
    return (before["pid"], before["starttime"], before["uid"]) == (
        after["pid"], after["starttime"], after["uid"])


class OwnedTree:
    """Only observed descendants of our child; pidfds prevent PID-reuse signals."""
    def __init__(self, pid):
        self.members = {}
        self.closed = False
        self.cleanup_issues = []
        self.add(process_info(pid))

    def add(self, info):
        pid = info["pid"]
        if pid in self.members:
            return same_process(self.members[pid][0], info)
        if info["uid"] != os.getuid():
            raise RuntimeError("profiler child changed user; refusing ownership")
        fd = os.pidfd_open(pid)
        try:
            if not same_process(info, process_info(pid)):
                raise RuntimeError("PID identity changed while opening pidfd")
        except BaseException:
            os.close(fd)
            raise
        self.members[pid] = (info, fd)
        return True

    def scan(self):
        records = {}
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                try:
                    info = process_info(int(entry.name))
                    records[info["pid"]] = info
                except (OSError, ValueError, IndexError):
                    pass
        # Anchor each ancestry edge to the parent's recorded process generation.
        changed = True
        while changed:
            changed = False
            for pid, info in records.items():
                parent = self.members.get(info["ppid"])
                live_parent = records.get(info["ppid"])
                if (pid not in self.members and parent and live_parent and
                        same_process(parent[0], live_parent)):
                    try:
                        changed = self.add(info) or changed
                    except ProcessLookupError:
                        pass
        return records

    def alive(self, pid):
        return pid in self.members and not select.select([self.members[pid][1]], [], [], 0)[0]

    def live_pids(self):
        return [pid for pid in self.members if self.alive(pid)]

    def send(self, pid, sig):
        if self.alive(pid):
            try:
                signal.pidfd_send_signal(self.members[pid][1], sig)
            except ProcessLookupError:
                pass

    def stop(self):
        try:
            self.scan()
        except (OSError, RuntimeError) as error:
            self.cleanup_issues.append(str(error))
        # Even if discovery fails, stop previously verified pidfds.
        for sig, seconds in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 2.0)):
            for pid in reversed(list(self.members)):
                self.send(pid, sig)
            deadline = time.monotonic() + seconds
            while self.live_pids() and time.monotonic() < deadline:
                time.sleep(0.02)
        return self.live_pids()

    def close(self):
        if not self.closed:
            for _, fd in self.members.values():
                os.close(fd)
            self.closed = True


def validate_ready(path, tree, target_argv, nonce, mode):
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > LOG_LIMIT:
        raise ValueError("invalid fixture readiness file")
    value = json.loads(path.read_text())
    pid = value.get("pid")
    if type(pid) is not int or pid not in tree.members or not tree.alive(pid):
        raise ValueError("fixture is not a live observed descendant of our profiler")
    live = process_info(pid)
    if (value.get("nonce") != nonce or value.get("mode") != mode or
            value.get("starttime") != live["starttime"] or
            not same_process(tree.members[pid][0], live)):
        raise ValueError("fixture readiness identity mismatch")
    actual = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    actual = [os.fsdecode(part) for part in actual if part]
    if (not actual or Path(actual[0]).resolve() != Path(target_argv[0]).resolve() or
            actual[1:] != target_argv[1:] or
            (Path("/proc") / str(pid) / "exe").resolve() != Path(target_argv[0]).resolve()):
        raise ValueError("fixture command line differs from the launched CPU fixture")
    maps = (Path("/proc") / str(pid) / "maps").read_text()
    if GPU_LIB.search(maps) or GPU_LIB.search(value.get("maps", "")):
        raise ValueError("CPU fixture unexpectedly mapped a GPU library")
    return value


def host_command(nsys, directory, fixture, nonce, mode):
    target = [sys.executable, "-I", str(fixture), str(directory), nonce, mode]
    argv = [str(nsys), "profile", *HOST_OPTIONS,
            "--session-new=sm75_host_" + nonce, "--output=" + str(directory / "report"), *target]
    return argv, target


def validate_help(text):
    required = [option.split("=", 1)[0] for option in HOST_OPTIONS]
    required += ["--session-new", "--output", "--cuda-trace-all-apis", "--cuda-flush-interval",
                 "--cuda-event-trace", "--cuda-memory-usage", "--cuda-trace-scope"]
    missing = [name for name in required if not re.search(re.escape(name) + r"(?=[=\s<])", text)]
    if missing:
        raise ValueError("installed Nsight help lacks required options: " + ", ".join(missing))


def run_owned(argv, directory, *, target=None, nonce=None, mode=None, seconds=60):
    """Bounded CPU-only subprocess with streamed log and retained process evidence."""
    started = time.monotonic()
    result = {"argv": argv, "started_monotonic": started, "mode": mode}
    overflow = threading.Event()
    reader_failed = []
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True, cwd=directory)
    tree = None
    reader = None
    try:
        tree = OwnedTree(process.pid)
        result["profiler_pid"] = process.pid

        def drain():
            size = 0
            try:
                with process.stdout, (directory / "console.log").open("xb") as stream:
                    while True:
                        data = process.stdout.read1(16384)
                        if not data:
                            break
                        remaining = max(0, LOG_LIMIT - size)
                        stream.write(data[:remaining])
                        stream.flush()
                        size += len(data)
                        if size > LOG_LIMIT:
                            overflow.set()
            except Exception as error:
                reader_failed.append(str(error))

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        ready = None
        ready_at = None
        deadline = started + seconds
        reason = None
        with (directory / "processes.jsonl").open("x") as stream:
            while True:
                tree.scan()
                process.poll()
                stream.write(json.dumps({"monotonic": time.monotonic(),
                    "owned": [info for info, _ in tree.members.values()],
                    "live_pids": tree.live_pids()}) + "\n")
                stream.flush()
                if target and ready is None and (directory / "ready.json").exists():
                    # Fixture fsyncs its ready file before printing, but a read
                    # can still precede completion. Retry only JSON decoding.
                    try:
                        ready = validate_ready(directory / "ready.json", tree, target, nonce, mode)
                    except json.JSONDecodeError:
                        ready = None
                    if ready is not None:
                        ready_at = time.monotonic()
                        result["target"] = {key: value for key, value in ready.items() if key != "maps"}
                        if mode != "timeout":
                            (directory / "release").write_text(nonce)
                if mode == "timeout" and ready_at and time.monotonic() - ready_at >= 1.0:
                    if "target_timeout" not in result:
                        result["target_timeout"] = True
                        # Signal only the verified CPU target, allowing the
                        # profiler time to notice exit and finish its report.
                        tree.send(ready["pid"], signal.SIGTERM)
                        result["term_sent_at"] = time.monotonic()
                    elif time.monotonic() - result["term_sent_at"] >= 1.0 and tree.alive(ready["pid"]):
                        tree.send(ready["pid"], signal.SIGKILL)
                        result["target_kill_sent"] = True
                if overflow.is_set() or reader_failed:
                    reason = "log-capture-failed"
                    break
                _, excluded = safe_files(directory)
                if excluded:
                    reason = "artifact-bound-or-type-failed"
                    break
                if process.returncode is not None and not tree.live_pids():
                    break
                if time.monotonic() >= deadline:
                    reason = "supervisor-timeout"
                    break
                time.sleep(0.05)
        result["reason"] = reason or "processes-exited"
    finally:
        if tree:
            result["surviving_owned_pids"] = tree.stop()
            result["cleanup_issues"] = tree.cleanup_issues
            tree.close()
        else:
            # Popen still owns this unreaped direct child; never signal a name.
            process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        if reader:
            reader.join(timeout=2)
        result.update({"returncode": process.poll(), "elapsed_seconds": time.monotonic() - started,
                       "log_overflow": overflow.is_set(), "reader_errors": reader_failed,
                       "reader_finished": reader is not None and not reader.is_alive()})
        write_json(directory / "result.json", result)
    return result


def safe_files(directory):
    """Confined, non-symlink evidence only; preserve rejected files on disk."""
    files, excluded, total = [], [], 0
    for path in sorted(directory.rglob("*")):
        try:
            info = path.lstat()
        except FileNotFoundError:
            # Nsight may replace intermediate files while building its report.
            # Final inventory runs again after supervision/cleanup.
            continue
        if stat.S_ISDIR(info.st_mode):
            continue
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                info.st_size > ARTIFACT_LIMIT or total + info.st_size > TOTAL_LIMIT):
            excluded.append(str(path.relative_to(directory)))
            continue
        total += info.st_size
        files.append(path)
    return files, excluded


def run(output, nsys):
    summary = {"gpu_workload_executed": False, "gpu_capture_qualified": False,
               "host_qualification": "in-progress", "issues": [], "cases": []}
    try:
        if "%" in str(output):
            raise ValueError("Nsight output path must not contain percent substitutions")
        for key, value in os.environ.items():
            if value and (key in FORBIDDEN_ENV or key.startswith(("CUPTI_", "NSYS_", "CUBLAS_LOG"))):
                raise ValueError("start host qualification without instrumentation control: " + key)
        source = Path(__file__).resolve()
        fixture = source.parent.parent / "tests/fixtures/sm75-nsys/cpu_target.py"
        copied = output / "cpu_target.py"
        shutil.copyfile(fixture, copied)
        shutil.copyfile(source, output / source.name)
        summary["source_sha256"] = {source.name: digest(source), "cpu_target.py": digest(copied)}
        summary["nsys"] = {"path": str(nsys), "sha256": digest(nsys)}
        summary["python"] = {"path": sys.executable, "version": sys.version}
        for label, options in (("version", ["--version"]), ("profile-help", ["profile", "--help"]),
                               ("export-help", ["export", "--help"])):
            directory = output / label
            directory.mkdir()
            record = run_owned([str(nsys), *options], directory, seconds=20)
            if (record["returncode"] != 0 or record["reason"] != "processes-exited" or
                    not record["reader_finished"] or record["log_overflow"] or
                    record["cleanup_issues"] or record["surviving_owned_pids"]):
                raise RuntimeError("Nsight metadata check failed: " + label)
        validate_help((output / "profile-help/console.log").read_text(errors="replace"))
        for mode in ("normal", "abrupt", "timeout"):
            print("Host-only Nsight qualification: " + mode, flush=True)
            directory = output / mode
            directory.mkdir()
            nonce = secrets.token_hex(12)
            argv, target = host_command(nsys, directory, copied, nonce, mode)
            result = run_owned(argv, directory, target=target, nonce=nonce, mode=mode)
            summary["cases"].append(result)
            if (not result.get("target") or result["reason"] != "processes-exited" or
                    result["surviving_owned_pids"] or not result["reader_finished"] or
                    result["log_overflow"] or result["reader_errors"] or result["cleanup_issues"]):
                raise RuntimeError("CPU fixture supervision failed: " + mode)
            if mode == "normal" and (result["returncode"] != 0 or
                    "cpu_fixture_complete=normal" not in (directory / "console.log").read_text()):
                raise RuntimeError("normal CPU fixture did not complete")
            if mode == "timeout" and not result.get("target_kill_sent"):
                raise RuntimeError("CPU timeout did not exercise verified target hard-stop")
        if digest(nsys) != summary["nsys"]["sha256"]:
            raise RuntimeError("Nsight entry point changed during qualification")
        summary["host_qualification"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        summary["host_qualification"] = "failed"
        summary["issues"].append(str(error))
        print("Host qualification: " + str(error), file=sys.stderr)
    finally:
        # Nsight with trace=none can legitimately emit no activity report.
        # Inventory what actually survived, never synthesize a successful one.
        files, excluded = safe_files(output)
        summary["excluded_from_archive"] = excluded
        summary["artifacts"] = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size,
                                 "sha256": digest(p)} for p in files]
        if excluded:
            summary["host_qualification"] = "failed"
            summary["issues"].append("unsafe or oversized evidence excluded; originals retained")
        write_json(output / "summary.json", summary)
        with tarfile.open(str(output) + ".tar.gz", "x:gz") as archive:
            for path in [*files, output / "summary.json"]:
                archive.add(path, arcname=str(Path(output.name) / path.relative_to(output)), recursive=False)
        print("Archive to return: " + str(output) + ".tar.gz", flush=True)
    return 0 if summary["host_qualification"] == "passed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys", default="nsys", help="installed Nsight CLI; no installation performed")
    args = parser.parse_args()
    if (platform.system() != "Linux" or platform.machine() != "x86_64" or
            not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal")):
        parser.error("Linux x86-64 with Python pidfd support required; no workload started")
    tool = shutil.which(args.nsys)
    if not tool:
        parser.error("installed nsys not found; nothing installed or executed")
    os.umask(0o077)
    def interrupted(*_):
        raise KeyboardInterrupt("host qualification interrupted; stopping owned CPU processes")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    output = Path(tempfile.mkdtemp(prefix="sm75-nsys-host-", dir=Path.cwd())).resolve()
    print("Host-only qualification directory: " + str(output), flush=True)
    return run(output, Path(tool).resolve())


if __name__ == "__main__":
    sys.exit(main())
