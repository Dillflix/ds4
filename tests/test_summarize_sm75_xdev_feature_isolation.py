#!/usr/bin/env python3

import array
import csv
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
SUMMARIZER = REPO / "speed-bench/summarize-sm75-xdev-feature-isolation.py"


class SummarizeSm75XdevFeatureIsolationTest(unittest.TestCase):
    def test_reports_semantics_and_plan_confound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            production = root / "production"
            production.mkdir()
            run_rows = []
            variants = ("neither", "partner-only", "rows-only", "both")
            planned = {"neither": 120, "partner-only": 120,
                       "rows-only": 114, "both": 114}
            for slot, variant in enumerate(variants, 1):
                bench = production / f"{variant}.csv"
                with bench.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("ctx_tokens", "prefill_tps"))
                    writer.writerow((2, 100 + slot))
                    writer.writerow((4, 90 + slot))
                logits = production / f"{variant}-logits"
                logits.mkdir()
                offset = {"neither": 0.0, "partner-only": 0.0,
                          "rows-only": 0.25, "both": 0.5}[variant]
                for context in (2, 4):
                    values = array.array("f", [float(i) + offset for i in range(12)])
                    with (logits / f"frontier_{context:06d}.logits.f32").open("wb") as handle:
                        values.tofile(handle)
                plan = production / f"{variant}-plan.csv"
                with plan.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("label", "consumer_device", "fallback_device",
                                     "target_device", "placement_locked",
                                     "resident_device", "status"))
                    writer.writerow(("tensor:x", 0, 1,
                                     1 if variant in ("rows-only", "both") else 0,
                                     1, 1 if variant in ("rows-only", "both") else 0,
                                     "partner"))
                run_rows.append(
                    {
                        "repeat": 1,
                        "slot": slot,
                        "variant": variant,
                        "partner": int(variant in ("partner-only", "both")),
                        "rows": int(variant in ("rows-only", "both")),
                        "planned_partner": planned[variant],
                        "csv": bench,
                        "log": production / f"{variant}.log",
                        "logits": logits,
                        "telemetry": root / f"{variant}-telemetry.csv",
                        "plan": plan,
                    }
                )
            with (production / "runs.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
                writer.writeheader()
                writer.writerows(run_rows)

            subprocess.run(
                [sys.executable, str(SUMMARIZER), str(root), "2", "4"],
                check=True,
            )
            with (production / "logit-comparisons.csv").open(newline="") as handle:
                comparisons = list(csv.DictReader(handle))
            self.assertEqual(len(comparisons), 6)
            rows_only = next(
                row for row in comparisons
                if row["comparison"] == "row-split-no-partner-execution"
                and row["ctx_tokens"] == "2"
            )
            self.assertEqual(rows_only["bit_exact"], "no")
            self.assertEqual(rows_only["top1_equal"], "yes")
            self.assertEqual(rows_only["top10_overlap"], "10")
            self.assertEqual(rows_only["placement_plan_equal"], "no")
            self.assertEqual(rows_only["reference_planned_partner"], "120")
            self.assertEqual(rows_only["candidate_planned_partner"], "114")
            with (production / "summary.csv").open(newline="") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual([row["ctx_tokens"] for row in summary], ["2", "4"])


if __name__ == "__main__":
    unittest.main()
