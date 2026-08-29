#!/usr/bin/env python3

import csv
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
SUMMARIZER = REPO / "speed-bench/summarize-sm75-decode-crash-isolation.py"


class SummarizeSm75DecodeCrashIsolationTest(unittest.TestCase):
    def test_summarizes_graduated_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            fields = (
                "repeat", "slot", "pp_tokens", "tg_tokens", "prefill_tps",
                "gen_tps", "first_ms", "steady_tps",
                "steady_ms_per_token", "csv", "log", "telemetry", "plan",
                "bindings",
            )
            with (production / "runs.tsv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                for repeat in (1, 2):
                    for slot, tg in enumerate((16, 64, 256), 1):
                        writer.writerow(
                            {
                                "repeat": repeat,
                                "slot": slot,
                                "pp_tokens": 32768,
                                "tg_tokens": tg,
                                "prefill_tps": 500 + repeat,
                                "gen_tps": 15 + repeat,
                                "first_ms": 80 + repeat,
                                "steady_tps": 16 + repeat,
                                "steady_ms_per_token": 1000 / (16 + repeat),
                                "csv": "case.csv",
                                "log": "case.log",
                                "telemetry": "telemetry.csv",
                                "plan": "plan.csv",
                                "bindings": "bindings.csv",
                            }
                        )
            subprocess.run([sys.executable, str(SUMMARIZER), str(root)], check=True)
            with (production / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["tg_tokens"] for row in rows], ["16", "64", "256"])
            self.assertEqual(rows[0]["decode_median_tps"], "16.500000")
            self.assertEqual(rows[0]["decode_graph"], "disabled")
            self.assertEqual(rows[0]["gpu_health"], "passed-pre-and-post")


if __name__ == "__main__":
    unittest.main()
