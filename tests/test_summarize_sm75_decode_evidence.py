#!/usr/bin/env python3

import csv
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "speed-bench" / "summarize-sm75-decode-evidence.py"
BENCH_HEADER = (
    "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,"
    "gen_steady_tokens,gen_steady_tps,kvcache_bytes\n"
)


def write_benchmark(path: Path, steady: float) -> None:
    path.write_text(
        BENCH_HEADER + f"4096,4096,600,256,18,60,255,{steady},0\n",
        encoding="utf-8",
    )


def make_trace(path: Path) -> None:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT)")
    db.execute(
        "CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, "
        "text TEXT, textId INTEGER)"
    )
    ranges = [
        (100, 1000, 7, "ds4/decode/token/pos=4096/token=1", None),
        (110, 800, 7, "ds4/decode/layer/layer=3/pos=4096/tier=0", None),
        (120, 400, 7, "ds4/decode/stage/name=attention/layer=3/pos=4096/tier=0", None),
        (410, 790, 7, "ds4/decode/stage/name=ffn/layer=3/pos=4096/tier=0", None),
        (810, 900, 7, "ds4/decode/output/pos=4096/tier=1", None),
    ]
    db.executemany("INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?)", ranges)
    db.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, "
        "globalTid INTEGER, correlationId INTEGER)"
    )
    db.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?)",
        [
            (140, 145, 7, 1),
            (450, 455, 7, 2),
            (830, 835, 7, 3),
            (460, 465, 7, 4),
            (500, 505, 8, 5),
        ],
    )
    db.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, "
        "deviceId INTEGER, streamId INTEGER, correlationId INTEGER, demangledName INTEGER)"
    )
    db.executemany(
        "INSERT INTO StringIds VALUES(?,?)",
        [
            (1, "attention_kernel"),
            (2, "moe_kernel"),
            (3, "output_kernel"),
            (4, "helper_thread_kernel"),
        ],
    )
    db.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)",
        [
            (200, 260, 0, 1, 1, 1),
            (500, 580, 1, 1, 2, 2),
            (840, 890, 2, 1, 3, 3),
            (520, 590, 3, 1, 5, 4),
        ],
    )
    db.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, "
        "deviceId INTEGER, streamId INTEGER, correlationId INTEGER, bytes INTEGER)"
    )
    db.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(470,480,1,2,4,4096)"
    )
    db.commit()
    db.close()


class DecodeEvidenceSummaryTests(unittest.TestCase):
    def test_threshold_and_trace_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ab").mkdir()
            (root / "exact").mkdir()
            (root / "summary").mkdir()
            records = []
            for repeat, (indexed, dense) in enumerate(((17.0, 18.0), (17.2, 18.1)), 1):
                for slot, (variant, threshold, steady) in enumerate(
                    (("indexed1024", 1024, indexed), ("dense4096", 4096, dense)), 1
                ):
                    bench = root / "ab" / f"r{repeat}-{variant}.csv"
                    write_benchmark(bench, steady)
                    records.append(
                        {
                            "repeat": repeat,
                            "slot": slot,
                            "variant": variant,
                            "threshold": threshold,
                            "pp_tokens": 4096,
                            "tg_tokens": 256,
                            "csv": bench,
                            "log": root / "ab" / f"r{repeat}-{variant}.log",
                        }
                    )
            with (root / "ab" / "runs.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            (root / "exact" / "verification.txt").write_text("bit_exact=true\n")
            trace = root / "trace.sqlite"
            make_trace(trace)
            (root / "trace-map.tsv").write_text(
                "label\tpp_tokens\tthreshold\tcaptured_tokens\tdevices\tsqlite\n"
                f"pp4096-indexed1024\t4096\t1024\t1\t0,3,1,2\t{trace}\n"
            )

            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(root)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = (root / "summary" / "summary.md").read_text()
            self.assertIn("bit-exact", summary)
            self.assertIn("PP4096 indexer threshold A/B", summary)
            with (root / "summary" / "stage-device-summary.csv").open(newline="") as handle:
                categories = {row["category"] for row in csv.DictReader(handle)}
            self.assertEqual(
                categories,
                {"attention", "ffn", "output", "unattributed-thread"},
            )
            with (root / "summary" / "trace-summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["captured_tokens"], "1")
            self.assertEqual(rows[0]["memcpy_bytes"], "4096")
            self.assertAlmostEqual(
                float(rows[0]["attribution_pct"]), 73.076923, places=6
            )


if __name__ == "__main__":
    unittest.main()
