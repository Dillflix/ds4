#!/usr/bin/env python3
"""CPU launch barrier; exec preserves the PID already verified by the collector."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

def private_read(path, limit=32768):
    """Only bounded ordinary files in the collector's private directory."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > limit):
            raise ValueError("invalid private gate file: path=%s mode=%04o uid=%s links=%s bytes=%s" %
                             (path, stat.S_IMODE(info.st_mode), info.st_uid,
                              info.st_nlink, info.st_size))
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("oversized gate file")
        return data.decode("utf-8")


def private_json(path, value):
    """Create private from the first instant, even if the profiler resets umask.

    Do not change umask: exec must preserve the target's inherited process state.
    O_EXCL also rejects an existing file/symlink instead of replacing evidence.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    config_path = Path(sys.argv[1])
    config = json.loads(private_read(config_path))
    directory = config_path.parent
    executable = Path(config["executable"])
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    ready = {"pid": os.getpid(), "ppid": os.getppid(), "starttime": int(fields[19]),
             "nonce": config["nonce"], "state": "waiting-before-exec"}
    private_json(directory / "gate-ready.json", ready)
    # No CUDA call, library loading, model access or workload initialization.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        release = directory / "gate-release"
        if release.exists():
            if private_read(release) != config["nonce"]:
                raise ValueError("invalid gate release")
            if (not executable.is_absolute() or not stat.S_ISREG(executable.stat().st_mode)
                    or fingerprint(executable) != config["sha256"]):
                raise ValueError("executable changed before exec")
            private_json(directory / "exec-attempt.json",
                         {"pid": os.getpid(), "starttime": int(fields[19]),
                          "monotonic_ns": time.monotonic_ns(), "sha256": config["sha256"]})
            os.execv(str(executable), [str(executable), *config["arguments"]])
        time.sleep(0.02)
    raise TimeoutError("gate was not released; workload not executed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("launch gate failed: " + str(error), file=sys.stderr, flush=True)
        sys.exit(125)
