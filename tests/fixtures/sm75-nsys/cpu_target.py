#!/usr/bin/env python3
"""CPU-only Nsight ownership fixture. No CUDA imports, calls or subprocesses."""
import json
import os
from pathlib import Path
import signal
import sys
import time


def main():
    directory, nonce, mode = sys.argv[1:]
    if mode not in ("normal", "abrupt", "timeout"):
        raise ValueError("unknown CPU fixture mode")
    directory = Path(directory)
    # All modes ignore TERM so the timeout case exercises a bounded hard stop.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    record = {"pid": os.getpid(), "ppid": os.getppid(), "starttime": int(stat[19]),
              "nonce": nonce, "mode": mode, "gpu_workload_executed": False,
              "maps": Path("/proc/self/maps").read_text()}
    # Exclusive private file; the parent checks process ancestry and the live
    # command line rather than trusting a PID supplied in a file alone.
    with (directory / "ready.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream)
        stream.flush()
        os.fsync(stream.fileno())
    print("cpu_fixture_ready=" + mode, flush=True)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        release = directory / "release"
        if release.exists() and release.read_text() == nonce and mode != "timeout":
            if mode == "abrupt":
                # Sudden process loss without a core dump or GPU involvement.
                os.kill(os.getpid(), signal.SIGKILL)
            print("cpu_fixture_complete=normal", flush=True)
            return 0
        time.sleep(0.02)
    return 120


if __name__ == "__main__":
    sys.exit(main())
