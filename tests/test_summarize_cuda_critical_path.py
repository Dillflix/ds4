#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench" / "summarize-cuda-critical-path.py"


def create_trace(path: Path, stage_devices: tuple[int, int], durations: tuple[int, int]):
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE NVTX_EVENTS(
            start INTEGER NOT NULL, end INTEGER, globalTid INTEGER,
            text TEXT, textId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            start INTEGER NOT NULL, end INTEGER NOT NULL, globalTid INTEGER,
            correlationId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
            start INTEGER NOT NULL, end INTEGER NOT NULL,
            deviceId INTEGER NOT NULL, streamId INTEGER NOT NULL,
            correlationId INTEGER, demangledName INTEGER NOT NULL
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
            start INTEGER NOT NULL, end INTEGER NOT NULL,
            deviceId INTEGER NOT NULL, streamId INTEGER NOT NULL,
            correlationId INTEGER, bytes INTEGER NOT NULL
        );
        INSERT INTO StringIds VALUES(1, 'synthetic_kernel');
        """
    )
    for stage, (device, duration) in enumerate(zip(stage_devices, durations)):
        base = 1000 + stage * 1000
        first = 0 if stage == 0 else 22
        end_layer = 22 if stage == 0 else 43
        ranges = [
            (base, base + 800, 7, f"ds4/prefill/wave/wave={stage}", None),
            (
                base + 10,
                base + 700,
                7,
                f"ds4/prefill/stage/stage={stage}/mb=0/tier={stage}/"
                f"layers={first}-{end_layer}/pos=0/tokens=512",
                None,
            ),
            (
                base + 20,
                base + 650,
                7,
                f"ds4/prefill/layer/stage={stage}/mb=0/tier={stage}/"
                f"layer={first}/pos=0/tokens=512",
                None,
            ),
        ]
        if stage == 0:
            ranges.append(
                (
                    base + 30,
                    base + 600,
                    7,
                    "ds4/q8/partner/label=attn_output_b/home_tier=0/"
                    "home_device=0/partner_tier=2/partner_device=1/"
                    "tokens=512/in=8192/out=4096/result=f32",
                    None,
                )
            )
            ranges.append(
                (
                    base + 35,
                    base + 590,
                    7,
                    "ds4/prefill/attention-rows/kind=indexed/layer=0/"
                    "pos=0/tokens=512/home_tier=0/partner_tier=2/"
                    "home_rows=256/partner_rows=256",
                    None,
                )
            )
        db.executemany("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?)", ranges)
        correlation = stage + 1
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)",
            (base + 40, base + 41, 7, correlation),
        )
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)",
            (base + 100, base + 100 + duration, device, 2, correlation, 1),
        )
    db.commit()
    db.close()


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ds4-critical-path-test-") as temp:
        out = Path(temp)
        current = out / "current.sqlite"
        swapped = out / "swapped.sqlite"
        create_trace(current, (0, 3), (50, 100))
        create_trace(swapped, (3, 0), (60, 90))
        (out / "trace-map.tsv").write_text(
            "label\tdevices\tsqlite\n"
            f"current\t0,3,1,2\t{current}\n"
            f"swapped\t3,0,2,1\t{swapped}\n",
            encoding="utf-8",
        )
        harness_lines = ["trial\tslot\tscenario\tdevice\tlog\n"]
        for slot, (device, elapsed) in enumerate(((0, 1.0), (3, 1.1)), 1):
            log = out / f"harness-gpu{device}.log"
            log.write_text(
                f"scenario=q4-early\ntimed_per_call_ms={elapsed}\n"
                "harness_status=ok\n",
                encoding="utf-8",
            )
            harness_lines.append(f"1\t{slot}\tq4-early\t{device}\t{log}\n")
        (out / "harness-runs.tsv").write_text(
            "".join(harness_lines), encoding="utf-8"
        )

        subprocess.run([sys.executable, str(SUMMARIZER), str(out)], check=True)
        stage_rows = read_rows(out / "stage-device-summary.csv")
        assert len(stage_rows) == 4, stage_rows
        assert {row["role"] for row in stage_rows} == {"home"}
        assert len(read_rows(out / "layer-device-summary.csv")) == 4
        assert len(read_rows(out / "partner-projection-summary.csv")) == 2
        assert len(read_rows(out / "attention-row-split-summary.csv")) == 2
        harness = read_rows(out / "same-work-gpu-summary.csv")
        assert len(harness) == 1
        assert math.isclose(float(harness[0]["gpu3_over_gpu0"]), 1.1)

        analysis = (out / "analysis.txt").read_text(encoding="utf-8")
        expected_gpu_factor = math.sqrt((60 / 50) * (100 / 90))
        marker = "physical_gpu3_over_gpu0_factor="
        actual = float(
            next(line[len(marker):] for line in analysis.splitlines()
                 if line.startswith(marker))
        )
        assert math.isclose(actual, expected_gpu_factor, rel_tol=1e-6)
        assert "late_stage_over_early_stage_factor=" in analysis

        # A bounded production profile may intentionally collect one trace to
        # avoid loading a 153 GiB NFS-backed model twice.  The same attribution
        # outputs must remain available without inventing 2x2 factors.
        (out / "trace-map.tsv").write_text(
            "label\tdevices\tsqlite\n"
            f"current\t0,3,1,2\t{current}\n",
            encoding="utf-8",
        )
        subprocess.run([sys.executable, str(SUMMARIZER), str(out)], check=True)
        single = (out / "analysis.txt").read_text(encoding="utf-8")
        assert "single_trace=true" in single
        assert "pair_attribution=not_requested" in single
        assert "physical_gpu3_over_gpu0_factor=" not in single
    print("critical-path summarizer synthetic 2x2 test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
