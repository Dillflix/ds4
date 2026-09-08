#!/usr/bin/env python3
"""Host-only evidence capture for ONE fingerprinted GPU1 reproducer invocation.

No GPU work, configuration writes, reset, retraining, or automatic retry here.
The caller retains the existing workload validation and outer archive trap.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import shutil
import stat
import subprocess
import sys
import threading
import time


EXPECTED_SHA256 = "5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414"
GPU1_UUID = "GPU-ba2c2d0b-6320-580f-208c-59e548ac9227"
ENDPOINTS = ["0000:02:00.0", "0000:03:00.0", "0000:81:00.0", "0000:82:00.0"]
ROOT_PORTS = ["0000:00:02.0", "0000:00:03.0", "0000:80:02.0", "0000:80:03.0"]
RETRAIN_UNIT = "retrain-gpu2-rootport.service"
TRACE_LIMIT = 64 * 1024 * 1024
TRACE_ENV = "DS4_RUNTIME_TRACE_LOG"
RUNTIME_CONTROLS = ("CUDA_MODULE_LOADING", "CUDA_FORCE_PTX_JIT", "CUDA_DISABLE_PTX_JIT",
                    "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES",
                    "CUDA_LAUNCH_BLOCKING", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
                    "LD_LIBRARY_PATH")


def valid_preload_path(path):
    return not re.search(r"[\s:]", str(path))


def trace_child_command(command, executable, inherited):
    """Decode only our runner's env form; never preload a shell/env/profiler.

    The original pinned ELF remains the directly owned PID. Reject alternate workloads,
    arguments and selectors instead of tracing an accidental reduced test.
    """
    if not command or command[0] not in ("env", "/usr/bin/env", "/bin/env"):
        raise ValueError("runtime trace requires the runner's explicit env command")
    child_env = dict(inherited)
    index = 1
    while index < len(command):
        item = command[index]
        if item == "-u":
            if index + 1 >= len(command) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", command[index + 1]):
                raise ValueError("invalid env unset in runtime trace command")
            child_env.pop(command[index + 1], None)
            index += 2
        elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", item):
            key, value = item.split("=", 1)
            if key in ("LD_PRELOAD", "LD_AUDIT", "CUDA_INJECTION64_PATH", "CUDA_ENABLE_COREDUMP_ON_EXCEPTION") or key.startswith(
                    ("DS4_RUNTIME_TRACE", "CUBLAS_LOG", "CUPTI_", "NV_COMPUTE_SANITIZER")):
                raise ValueError("runtime trace command contains unrelated instrumentation")
            child_env[key] = value
            index += 1
        else:
            break
    if len(command[index:]) != 1 or Path(command[index]).resolve() != Path(executable).resolve():
        raise ValueError("runtime trace must directly execute the pinned ELF without arguments")
    expected = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1",
        "DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_NO_ROW_OWNED": "1",
        "DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_CALLS": "1024",
        "DS4_TOKEN_ROW_ARITHMETIC_OUTPUT_B_PRODUCTION103_BATCH": "10",
        "B_TIMING_ROUNDS": "7", "B_TIMING_REPEATS": "10", "B_TIMING_WARMUPS": "3",
    }
    for key, value in expected.items():
        if child_env.get(key) != value:
            raise ValueError("runtime trace workload contract mismatch: " + key)
    if any(key.startswith("DS4_") and key not in expected for key in child_env):
        raise ValueError("runtime trace workload contains an extra DS4 selector")
    if child_env.get("CUDA_LAUNCH_BLOCKING"):
        raise ValueError("runtime trace must not add launch blocking")
    # Physical identity was established by the same preflight inventory. UUID
    # selection avoids another ordinal namespace in the interposer's observations.
    child_env["CUDA_VISIBLE_DEVICES"] = GPU1_UUID
    return [str(Path(executable).resolve())], child_env


def normalize_boot_id(value):
    value = value.strip()
    if not re.fullmatch(r"(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", value, re.I):
        raise ValueError("invalid kernel boot ID: " + value)
    return value.replace("-", "").lower()


def stamp():
    return {"utc_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fault_event(line):
    """Preserve the journal's own clocks/BDF, not just the observer's timestamp."""
    try:
        event = json.loads(line)
    except (ValueError, TypeError):
        return None
    message = event.get("MESSAGE", "")
    if not isinstance(message, str):
        return None
    if not re.search(r"NVRM: Xid|fallen off the bus|GPU Unavailable|"
                     r"AER:.*(?:Uncorrected|Fatal)|PCIe Bus Error: severity=Uncorrected|"
                     r"Machine check events logged|Hardware Error.*(?:fatal|uncorrected)",
                     message, re.I):
        return None
    bdf = re.search(r"(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}(?:\.[0-7])?", message, re.I)
    xid = re.search(r"Xid\s*\([^)]*\)\s*:\s*(\d+)", message)
    return {**stamp(), "journal": event, "bdf": bdf.group(0) if bdf else None,
            "xid": int(xid.group(1)) if xid else None}


class Capture:
    def __init__(self, output, executable, command, case_timeout, preflight_only=False,
                 runtime_trace_library=None, runtime_trace_sha256=None):
        self.output = Path(output)
        self.executable = Path(executable).resolve()
        self.command = command
        self.case_timeout = case_timeout
        self.preflight_only = preflight_only
        self.runtime_trace_library = Path(runtime_trace_library) if runtime_trace_library else None
        self.runtime_trace_sha256 = runtime_trace_sha256
        self.trace_command = None
        self.trace_environment = None
        self.summary = {"workload": "not-started", "workload_returncode": None,
                        "collection": "in-progress", "started": stamp(),
                        "first_fault": None, "commands": [], "issues": []}
        self.interrupted = threading.Event()
        self.fault = threading.Event()
        self.stream_error = threading.Event()
        self.journal_process = None
        self.journal_reader = None
        self.journal_error = None
        self.workload_process = None
        self.boot_id = None
        self.sudo_refresh_seconds = 60

    def prepare_runtime_trace(self):
        if self.runtime_trace_library is None:
            if self.runtime_trace_sha256:
                raise ValueError("runtime trace SHA requires an explicit library")
            return
        if not re.fullmatch(r"[0-9a-f]{64}", self.runtime_trace_sha256 or ""):
            raise ValueError("runtime trace requires an explicit lowercase SHA256")
        source = self.runtime_trace_library
        if source.is_symlink():
            raise ValueError("runtime trace library must not be a symlink")
        source = source.resolve(strict=True)
        info = source.stat()
        if (not stat.S_ISREG(info.st_mode) or not 64 <= info.st_size <= TRACE_LIMIT
                or info.st_mode & 0o022 or info.st_uid not in (0, os.getuid())):
            raise ValueError("runtime trace library must be a bounded, trusted-owner, non-group/world-writable file")
        with source.open("rb") as stream:
            header = stream.read(20)
        if (header[:6] != b"\x7fELF\x02\x01" or header[16:20] != b"\x03\x00\x3e\x00"):
            raise ValueError("runtime trace library must be an ELF64 x86-64 shared object")
        if sha256(source) != self.runtime_trace_sha256:
            raise ValueError("runtime trace library fingerprint mismatch")
        argv, environment = trace_child_command(self.command, self.executable, os.environ)
        directory = self.output / "runtime-trace"
        directory.mkdir(mode=0o700, exist_ok=False)
        copied = directory / "sm75-runtime-contract-trace.so"
        with source.open("rb") as src, copied.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        copied.chmod(0o500)
        if sha256(copied) != self.runtime_trace_sha256:
            raise ValueError("runtime trace copied-library fingerprint mismatch")
        # Dynamic loader treats spaces/colons as LD_PRELOAD list delimiters.
        if not valid_preload_path(copied.resolve()):
            raise ValueError("runtime trace output path cannot contain whitespace or colons")
        log_path = directory / "runtime-contract.jsonl"
        environment["LD_PRELOAD"] = str(copied.resolve())
        environment[TRACE_ENV] = str(log_path.resolve())
        self.trace_command, self.trace_environment = argv, environment
        self.summary["runtime_trace"] = {
            "mode": "instrumented-runtime-contract", "library_source": str(source),
            "library_copy": str(copied.resolve()), "library_sha256": self.runtime_trace_sha256,
            "log": str(log_path.resolve()), "direct_workload_pid": True,
            "added_device_synchronization": False,
            "claims": "application API observations; not all internal library activity or a general race detector",
        }
        self.summary["execution_mode"] = "instrumented-runtime-contract"
        self.summary["effective_runtime_controls"] = {
            key: {"set": key in environment, "value": environment.get(key)} for key in RUNTIME_CONTROLS}
        self.save()

    def finalize_runtime_trace(self):
        if self.trace_command is None or self.workload_process is None:
            return
        path = self.output / "runtime-trace/runtime-contract.jsonl"
        if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= TRACE_LIMIT:
            self.issue("runtime contract trace missing, invalid or oversized; no trace clearance")
            return
        detail = self.summary["runtime_trace"]
        # Freeze exactly a bounded prefix before hashing/parsing. A faulted child
        # may still be alive after stop bounds; never label a moving log immutable.
        size = path.stat().st_size
        if not 0 < size <= TRACE_LIMIT:
            self.issue("runtime contract trace changed beyond its size bound")
            return
        snapshot = path.with_name("runtime-contract.snapshot.jsonl")
        state = self.workload_process.poll()
        detail.update({"snapshot_start": stamp(), "raw_bytes_at_snapshot_start": size,
                       "child_returncode_at_snapshot": state, "snapshot_of_live_child": state is None})
        try:
            with path.open("rb") as source, snapshot.open("xb") as frozen:
                remaining = size
                while remaining:
                    block = source.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise RuntimeError("runtime trace shrank during snapshot")
                    frozen.write(block)
                    remaining -= len(block)
            detail.update({"trace_bytes": size, "trace_sha256": sha256(snapshot),
                           "analyzed_snapshot": str(snapshot), "snapshot_end": stamp()})
        except Exception as error:
            self.issue("runtime trace snapshot unavailable: " + str(error))
            return
        analyzer = Path(__file__).with_name("analyze-sm75-runtime-contract.py")
        try:
            spec = importlib.util.spec_from_file_location("sm75_runtime_contract_analysis", analyzer)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            analysis = module.analyze_file(snapshot)
            (path.parent / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
            detail["analysis"] = "runtime-trace/analysis.json"
            detail["analysis_status"] = analysis.get("status")
            if analysis.get("process_ids") != [self.workload_process.pid]:
                self.issue("runtime trace PID does not match the directly owned frozen ELF")
            if not analysis.get("required_capture_present") or not analysis.get("trace_complete"):
                self.issue("runtime trace coverage/envelope incomplete; retained as partial evidence")
            if not analysis.get("expected_transition", {}).get("expected_transition_required_present"):
                self.issue("runtime trace did not establish the required DEFAULT512-to-103/256 transition")
            if not analysis.get("expected_transition", {}).get("postburnin_transition_required_present"):
                self.issue("runtime trace did not establish the qualified post-burn-in DEFAULT512-to-103/256 transition")
            if analysis.get("violations"):
                self.issue("runtime trace recorded contract violations; inspect analysis.json")
        except Exception as error:
            self.issue("runtime trace analysis unavailable: " + str(error))
        self.save()

    def journal_command(self, *arguments):
        # -b has an optional argument: an unrecognized following UUID can be
        # parsed as a positional match. Use one explicit option with ID128 hex.
        return ["journalctl", "--boot=" + normalize_boot_id(self.boot_id), *arguments]

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        temporary = self.output / "summary.json.tmp"
        temporary.write_text(json.dumps(self.summary, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output / "summary.json")

    def issue(self, text):
        self.summary["issues"].append(text)
        self.save()

    @staticmethod
    def launch_collector(command, **kwargs):
        # Own a process group for bounded cleanup WITHOUT setsid(): sudo's tty
        # ticket belongs to the authenticated terminal session. Detaching the
        # collector loses that ticket even immediately after `sudo -v`.
        return subprocess.Popen(command, start_new_session=False, process_group=0, **kwargs)

    @staticmethod
    def stop(process):
        """Only signal this helper's freshly created child process group."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if process.poll() is not None:
                return True
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                return True
            except PermissionError:
                # sudo forwards signals; privileged children also have their
                # own GNU timeout. Do not broaden this to a system-wide kill.
                try:
                    process.send_signal(sig)
                except ProcessLookupError:
                    return True
            try:
                process.wait(timeout=2)
                return True
            except subprocess.TimeoutExpired:
                pass
        return process.poll() is not None

    def run(self, name, command, seconds=20, root=False, cwd=None):
        """Preserve partial output, errors and timeout status of every query."""
        path = self.output / (name + ".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        argv = (["sudo", "-n", "timeout", "--signal=TERM", "--kill-after=2",
                 str(seconds)] if root else []) + command
        record = {"name": name, "argv": argv, "start": stamp(), "timeout": False}
        with path.open("wb") as log:
            try:
                process = self.launch_collector(argv, stdout=log, stderr=subprocess.STDOUT, cwd=cwd)
                try:
                    record["returncode"] = process.wait(timeout=seconds + (4 if root else 0))
                    record["timeout"] = record["returncode"] == 124
                except subprocess.TimeoutExpired:
                    record["timeout"] = True
                    record["terminated"] = self.stop(process)
                    record["returncode"] = process.poll()
            except OSError as error:
                log.write(str(error).encode())
                record["returncode"] = None
        record["end"] = stamp()
        self.summary["commands"].append(record)
        if record["returncode"] != 0 or record["timeout"]:
            self.issue("collector failed or timed out: " + name)
        self.save()
        return record, path.read_text(encoding="utf-8", errors="replace")

    def required(self, name, command, **kwargs):
        record, text = self.run(name, command, **kwargs)
        if record["returncode"] != 0 or record["timeout"]:
            detail = text.strip()[-1000:] or "no collector output"
            raise RuntimeError("required preflight collector failed: " + name + ": " + detail)
        return text

    def pci_snapshot(self, phase):
        data = {"time": stamp(), "devices": {}}
        for bdf in ENDPOINTS + ROOT_PORTS:
            base = Path("/sys/bus/pci/devices") / bdf
            values = {"resolved_sysfs_path": str(base.resolve())}
            for item in [base / "current_link_speed", base / "current_link_width",
                         base / "resource", *sorted(base.glob("aer_*"))]:
                try:
                    values[item.name] = item.read_text()
                except OSError as error:
                    values[item.name] = {"unavailable": str(error)}
            values["aer_counters_present"] = bool(list(base.glob("aer_*")))
            data["devices"][bdf] = values
            self.run(phase + "/pci-" + bdf.replace(":", "_"),
                     ["lspci", "-D", "-s", bdf, "-vv"], root=True, seconds=5)
        (self.output / (phase + "-sysfs.json")).write_text(json.dumps(data, indent=2) + "\n")

    def preflight(self):
        if sys.platform != "linux":
            raise RuntimeError("capture is Linux-only; do not launch a GPU workload here")
        if sys.version_info < (3, 11):
            raise RuntimeError("capture requires Python 3.11+ for same-session process groups")
        actual = sha256(self.executable)
        self.summary.update({"executable": str(self.executable), "executable_sha256": actual,
                             "command": self.command})
        self.save()
        if actual != EXPECTED_SHA256:
            raise RuntimeError("executable fingerprint changed; do not rebuild or substitute")
        # These alter execution/tooling and do not belong in this uninstrumented
        # comparison. Never dump the whole environment (which can hold secrets).
        forbidden = [key for key in os.environ if
                     key in ("LD_PRELOAD", "LD_AUDIT", "CUDA_INJECTION64_PATH", "CUDA_ENABLE_COREDUMP_ON_EXCEPTION")
                     or key.startswith(("CUBLAS_LOG", "CUPTI_", "NV_COMPUTE_SANITIZER"))]
        if any(os.environ[key] for key in forbidden):
            raise RuntimeError("remove inherited instrumentation: " + ", ".join(forbidden))
        if any(key.startswith("DS4_RUNTIME_TRACE") and value for key, value in os.environ.items()):
            raise RuntimeError("remove inherited runtime trace controls; use the explicit collector option")
        self.summary["execution_mode"] = "uninstrumented-frozen-reproducer"
        self.prepare_runtime_trace()
        self.required("pre/sudo", ["sudo", "-n", "true"], seconds=5)
        self.required("pre/report-tool", ["sh", "-c", "command -v nvidia-bug-report.sh"], root=True)
        services = self.required("pre/dcgm", ["snap", "services", "dcgm"])
        for service in ("dcgm.nv-hostengine", "dcgm.dcgm-exporter"):
            if not any(row[:3] == [service, "disabled", "inactive"]
                       for row in (line.split() for line in services.splitlines())):
                raise RuntimeError(service + " must be disabled and inactive")
        self.required("pre/retrain-unit", ["systemctl", "cat", RETRAIN_UNIT])
        state = self.required("pre/retrain-state", ["systemctl", "show", RETRAIN_UNIT,
            "--property=Type,ActiveState,SubState,MainPID,Result,ExecMainStatus,"
            "ExecMainStartTimestamp,ExecMainExitTimestamp,ActiveEnterTimestamp,"
            "TimeoutStartUSec,Before,After"])
        properties = dict(line.split("=", 1) for line in state.splitlines() if "=" in line)
        for key, expected in {"Type": "oneshot", "ActiveState": "active", "SubState": "exited",
                              "MainPID": "0", "Result": "success", "ExecMainStatus": "0"}.items():
            if properties.get(key) != expected:
                raise RuntimeError("startup retraining has not completed successfully: " + key)
        processes = self.required("pre/processes", ["ps", "-eo", "pid,ppid,lstart,comm"])
        if re.search(r"\b(?:nv-hostengine|nvbandwidth|retrain-gpu2-ro\S*)\s*$", processes, re.M):
            raise RuntimeError("background hostengine/bandwidth/retraining process present")
        inventory = self.required("pre/gpus", ["nvidia-smi", "--query-gpu=index,pci.bus_id,uuid,"
            "driver_version,pstate,temperature.gpu,power.draw,power.limit,memory.used,memory.free",
            "--format=csv,noheader,nounits"])
        rows = list(csv.reader(inventory.splitlines(), skipinitialspace=True))
        if len(rows) != 4 or any(len(row) != 10 for row in rows):
            raise RuntimeError("expected four accessible GPUs")
        for index, bdf in enumerate(ENDPOINTS):
            matches = [r for r in rows if r[0] == str(index)]
            if (len(matches) != 1 or matches[0][1].lower().replace("00000000:", "0000:") != bdf
                    or re.search(r"ERR!|Unknown Error|N/A", ",".join(matches[0]), re.I)):
                raise RuntimeError("GPU inventory/health mismatch: " + bdf)
            if index == 1 and matches[0][2] != GPU1_UUID:
                raise RuntimeError("physical GPU1 UUID mismatch")
            if index == 1 and abs(float(matches[0][7]) - 260.0) > 0.5:
                raise RuntimeError("GPU1 power limit is not the established 260 W")
        clients = self.required("pre/compute-processes", ["nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name", "--format=csv,noheader,nounits"])
        if clients.strip():
            raise RuntimeError("another CUDA compute client is present")
        self.required("pre/gpu-details", ["nvidia-smi", "-q"])
        raw_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        self.boot_id = normalize_boot_id(raw_boot_id)
        self.summary["kernel_boot_id_raw"] = raw_boot_id
        self.summary["boot_id"] = self.boot_id
        self.summary["host"] = {"uname": list(os.uname()),
                                 "cmdline": Path("/proc/cmdline").read_text().strip()}
        baseline = self.required("kernel-baseline", self.journal_command("-k",
            "-o", "json", "--no-pager", "--show-cursor"), root=True)
        cursors = re.findall(r"^-- cursor: (.+)$", baseline, re.M)
        if not cursors:
            raise RuntimeError("journal cursor unavailable; refusing an unobserved run")
        if any(fault_event(line) for line in baseline.splitlines()):
            raise RuntimeError("current boot already contains a GPU/PCIe/hardware fault")
        self.pci_snapshot("pre")
        if self.summary["issues"]:
            raise RuntimeError("baseline collection incomplete; do not spend a failure run without it")
        self.required("pre/retrain-history", self.journal_command("--no-pager",
            "-u", RETRAIN_UNIT, "-o", "short-monotonic"), root=True)
        self.start_journal(cursors[-1])

    def observe_kernel(self, line):
        event = fault_event(line)
        if event and not self.fault.is_set():
            self.summary["first_fault"] = event
            self.fault.set()

    def start_journal(self, cursor):
        # All-kernel follow; no CUDA/NVML polling. sudo's timeout bounds even an
        # orphaned privileged follower. Normal shutdown signals its owned sudo.
        self.journal_error = (self.output / "kernel-live.stderr.log").open("wb")
        self.journal_process = self.launch_collector(["sudo", "-n", "timeout", "--signal=TERM",
            "--kill-after=2", str(self.case_timeout + 480), *self.journal_command(
            "-k", "--after-cursor=" + cursor, "-f", "-o", "json", "--no-pager")],
            stdout=subprocess.PIPE, stderr=self.journal_error)

        def read():
            try:
                with self.journal_process.stdout, (self.output / "kernel-live.jsonl").open("wb") as output:
                    for line in self.journal_process.stdout:
                        output.write(line)
                        output.flush()
                        self.observe_kernel(line)
            except Exception:
                self.stream_error.set()
        self.journal_reader = threading.Thread(target=read, daemon=True)
        self.journal_reader.start()
        time.sleep(0.2)
        if self.journal_process.poll() is not None:
            raise RuntimeError("live kernel journal exited before workload launch")

    def process_snapshot(self, pid):
        base = Path("/proc") / str(pid)
        record = {**stamp(), "pid": pid, "threads": {}}
        for task in sorted((base / "task").glob("*")):
            record["threads"][task.name] = {}
            for name in ("status", "wchan", "syscall"):
                try:
                    record["threads"][task.name][name] = (task / name).read_text()
                except OSError as error:
                    record["threads"][task.name][name] = {"unavailable": str(error)}
        try:
            maps = (base / "maps").read_text()
            selected = [line for line in maps.splitlines()
                        if re.search(r"lib(?:cuda|cublas|cudart|nvidia)", line)]
            record["cuda_library_mappings"] = selected
            # Save observed loaded library paths, not ldd's hypothetical choices.
            self.summary["loaded_library_paths"] = sorted(set(
                self.summary.get("loaded_library_paths", []) +
                [line.split(None, 5)[5] for line in selected if len(line.split(None, 5)) == 6]))
        except OSError as error:
            record["maps_unavailable"] = str(error)
        return record

    def execute(self):
        if self.fault.is_set() or self.interrupted.is_set():
            raise RuntimeError("fault or interrupt arrived before workload launch")
        if self.trace_command is not None:
            # Detect changes between preflight and launch without executing a
            # preliminary CUDA check or preloading any host collector subprocess.
            if (sha256(self.executable) != EXPECTED_SHA256 or
                sha256(self.summary["runtime_trace"]["library_copy"]) != self.runtime_trace_sha256):
                raise RuntimeError("runtime trace artifacts changed after preflight")
            process = subprocess.Popen(self.trace_command, env=self.trace_environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        else:
            process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        self.workload_process = process
        self.summary.update({"workload": "running", "workload_pid": process.pid,
                             "workload_start": stamp()})
        self.save()

        def read():
            try:
                with process.stdout, (self.output / "application-timeline.jsonl").open("w", encoding="utf-8") as log:
                    for line in process.stdout:
                        event = {**stamp(), "line": line.decode("utf-8", errors="replace").rstrip("\n")}
                        log.write(json.dumps(event) + "\n")
                        log.flush()
                        sys.stdout.buffer.write(line)
                        sys.stdout.buffer.flush()
            except Exception:
                self.stream_error.set()
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        deadline = time.monotonic() + self.case_timeout
        next_snapshot = 0
        next_sudo_refresh = time.monotonic() + self.sudo_refresh_seconds
        sudo_refresh_count = 0
        reason = None
        with (self.output / "process-timeline.jsonl").open("w", encoding="utf-8") as log:
            while process.poll() is None:
                if self.interrupted.is_set():
                    reason = "interrupted"
                elif self.fault.is_set():
                    reason = "stopped-on-kernel-fault"
                elif (self.journal_process.poll() is not None or self.stream_error.is_set()
                      or (self.journal_reader and not self.journal_reader.is_alive())):
                    reason = "stopped-on-collector-failure"
                    self.issue("live journal or application capture stopped during workload")
                elif time.monotonic() >= deadline:
                    reason = "timeout"
                if reason:
                    self.summary["stop_requested"] = stamp()
                    self.summary["workload_terminated"] = self.stop(process)
                    if not self.summary["workload_terminated"]:
                        self.issue("owned workload would not terminate; possible uninterruptible task")
                    break
                if time.monotonic() >= next_sudo_refresh:
                    sudo_refresh_count += 1
                    record, _ = self.run("sudo-refresh-" + str(sudo_refresh_count),
                                         ["sudo", "-n", "-v"], seconds=5)
                    if record["returncode"] != 0 or record["timeout"]:
                        reason = "stopped-on-collector-failure"
                        self.summary["stop_requested"] = stamp()
                        self.summary["workload_terminated"] = self.stop(process)
                        self.issue("cached sudo authorization could not be refreshed noninteractively")
                        break
                    next_sudo_refresh = time.monotonic() + self.sudo_refresh_seconds
                if time.monotonic() >= next_snapshot:
                    log.write(json.dumps(self.process_snapshot(process.pid)) + "\n")
                    log.flush()
                    next_snapshot = time.monotonic() + 1
                time.sleep(0.05)
        reader.join(timeout=2)
        self.summary.update({"workload": reason or ("exited-zero" if process.returncode == 0 else "failed"),
                             "workload_returncode": process.poll(), "workload_end": stamp()})
        self.save()
        if reader.is_alive():
            self.issue("application output reader did not finish; partial output retained")

    def postmortem(self):
        _, kernel = self.run("post/kernel", self.journal_command("-k", "-o", "json",
                                            "--no-pager"), root=True)
        # Baseline was required fault-free. This also catches an event delivered
        # just as the binary exits, before the live reader has processed it.
        for line in kernel.splitlines():
            self.observe_kernel(line)
        self.pci_snapshot("post")
        self.run("post/gpus", ["nvidia-smi", "-q"], seconds=15)
        self.run("post/processes", ["ps", "-eo", "pid,ppid,lstart,stat,wchan:32,comm"])
        if self.summary["workload"] != "exited-zero" or self.fault.is_set():
            report_dir = self.output / "nvidia-report"
            report_dir.mkdir(exist_ok=True)
            self.run("nvidia-report/collector", ["nvidia-bug-report.sh", "--safe-mode",
                "--extra-system-data"], seconds=120, root=True, cwd=report_dir)
            self.summary["nvidia_report_files"] = [p.name for p in report_dir.iterdir()]
            if not any(p.name.startswith("nvidia-bug-report") and p.stat().st_size
                       for p in report_dir.iterdir() if p.is_file()):
                self.issue("NVIDIA report produced no nonempty report; collector log retained")
        libraries = {}
        for name in self.summary.get("loaded_library_paths", []):
            try:
                libraries[name] = sha256(name)
            except OSError as error:
                libraries[name] = {"unavailable": str(error)}
        self.summary["loaded_library_sha256"] = libraries

    def observe_final_workload_status(self):
        if self.workload_process is None:
            return
        # Popen.poll() performs only a nonblocking wait on this owned child.
        # A child that outlived the initial stop bound can have exited during
        # postmortem collection. Record that observation without another wait,
        # signal, relaunch, or claiming the observation time is its exit time.
        code = self.workload_process.poll()
        observation = {**stamp(), "returncode": code}
        self.summary["workload_final_poll"] = observation
        if code is not None and self.summary["workload_returncode"] is None:
            self.summary["late_exit_observed"] = observation.copy()
            self.summary["workload_returncode"] = code
        # Preserve workload_terminated=False (the initial bounded stop result),
        # the original workload/fault reason and all partial-collection issues.

    def finish(self):
        if self.workload_process and self.workload_process.poll() is None:
            if not self.stop(self.workload_process):
                self.issue("owned workload still alive at collector exit")
        if self.journal_process:
            if not self.stop(self.journal_process):
                self.issue("owned journal follower did not terminate")
        if self.journal_reader:
            self.journal_reader.join(timeout=2)
            if self.journal_reader.is_alive():
                self.issue("journal reader did not finish; partial journal retained")
        if self.journal_error:
            self.journal_error.close()
        if self.stream_error.is_set():
            self.issue("a live capture stream failed; partial evidence retained")
        self.observe_final_workload_status()
        self.finalize_runtime_trace()
        self.summary["collection"] = "partial" if self.summary["issues"] else "complete"
        self.summary["finished"] = stamp()
        self.save()

    def result(self):
        if self.summary["workload"] == "timeout":
            return 124
        if self.interrupted.is_set():
            return 130
        if self.summary["workload"] == "not-started":
            if (self.preflight_only and self.summary.get("preflight") == "passed"
                    and not self.summary["issues"] and not self.fault.is_set()):
                return 0
            return 2
        if self.fault.is_set():
            return 86
        if self.summary["workload_returncode"]:
            code = self.summary["workload_returncode"]
            return code if code > 0 else 128 - code
        if self.summary["issues"] or self.summary["workload"] != "exited-zero":
            return 87
        return 0

    def capture(self):
        self.output.mkdir(parents=True, exist_ok=False)
        self.save()
        try:
            self.preflight()
            self.summary["preflight"] = "passed"
            if not self.preflight_only:
                self.execute()
        except Exception as error:
            self.issue(str(error))
            print("failure capture: " + str(error), file=sys.stderr, flush=True)
            if self.workload_process:
                self.stop(self.workload_process)
        finally:
            if self.workload_process:
                try:
                    self.postmortem()
                except Exception as error:
                    self.issue("postmortem incomplete: " + str(error))
            self.finish()
        return self.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--case-timeout", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate host collectors and exit without launching the CUDA executable")
    parser.add_argument("--runtime-trace-library", help="explicit opt-in to one instrumented frozen-ELF run")
    parser.add_argument("--runtime-trace-sha256", help="required SHA256 of the separately built host interposer")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (not command and not args.preflight_only) or not 1 <= args.case_timeout <= 600:
        parser.error("one command and a timeout of 1..600 seconds are required")
    capture = Capture(args.output, args.executable, command, args.case_timeout, args.preflight_only,
                      args.runtime_trace_library, args.runtime_trace_sha256)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: capture.interrupted.set())
    return capture.capture()


if __name__ == "__main__":
    sys.exit(main())
