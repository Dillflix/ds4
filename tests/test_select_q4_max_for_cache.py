#!/usr/bin/env python3

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_q4_max_for_cache",
    ROOT / "speed-bench" / "select-q4-max-for-cache.py",
)
SELECTOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SELECTOR)


class SelectQ4MaxForCacheTests(unittest.TestCase):
    def write_plan(self, path: Path, rows: list[list[str]]) -> None:
        fields = [
            "sequence",
            "label",
            "consumer_device",
            "fallback_device",
            "resident_device",
            "weight_offset",
            "weight_bytes",
            "in_dim",
            "out_dim",
            "fp16_bytes",
            "status",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_required_reclaim_models_fixed_and_partner_flexible_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            self.write_plan(
                path,
                [
                    ["0", "a", "0", "1", "-1", "100", "20", "4", "8", "64", "unadmitted"],
                    ["1", "a", "0", "1", "-1", "100", "20", "4", "8", "64", "unadmitted"],
                    ["2", "b", "1", "0", "1", "200", "20", "4", "8", "128", "home"],
                    ["3", "c", "3", "-1", "-1", "300", "20", "4", "8", "256", "unadmitted"],
                ],
            )
            pairs = SELECTOR.device_pairs([0, 3, 1, 2])
            required, detail = SELECTOR.required_reclaim_by_stage(
                path, pairs, 32
            )
            self.assertEqual(
                required,
                {0: 96, 1: 288},
            )
            self.assertEqual(detail[0][1], [64])
            self.assertEqual(detail[1][0][3], 256)

    def test_exact_objective_uses_iq2_pair_when_it_preserves_more_q4(self):
        q2, iq2, reclaim = SELECTOR.select_stage_recipe(
            required_bytes=500 * SELECTOR.MIB,
            q2_candidates=list(range(3, 22)),
            iq2_candidates=list(range(3, 22)),
            q2_saving_bytes_per_device=240 * SELECTOR.MIB,
            iq2_saving_bytes_per_device=624 * SELECTOR.MIB,
        )
        self.assertEqual(q2, [])
        self.assertEqual(iq2, [3])
        self.assertEqual(reclaim, 624 * SELECTOR.MIB)

    def test_preserve_down_breaks_equal_q4_count_tie(self):
        q2, iq2, reclaim = SELECTOR.select_stage_recipe(
            required_bytes=650 * SELECTOR.MIB,
            q2_candidates=list(range(3, 22)),
            iq2_candidates=list(range(3, 22)),
            q2_saving_bytes_per_device=240 * SELECTOR.MIB,
            iq2_saving_bytes_per_device=624 * SELECTOR.MIB,
            tie_break="preserve-down",
        )
        self.assertEqual(q2, [3])
        self.assertEqual(iq2, [3])
        self.assertEqual(reclaim, 864 * SELECTOR.MIB)

    def test_least_overshoot_breaks_equal_q4_count_tie(self):
        q2, iq2, reclaim = SELECTOR.select_stage_recipe(
            required_bytes=700 * SELECTOR.MIB,
            q2_candidates=list(range(3, 22)),
            iq2_candidates=list(range(3, 22)),
            q2_saving_bytes_per_device=240 * SELECTOR.MIB,
            iq2_saving_bytes_per_device=624 * SELECTOR.MIB,
            tie_break="least-overshoot",
        )
        self.assertEqual(q2, [3, 4, 5])
        self.assertEqual(iq2, [])
        self.assertEqual(reclaim, 720 * SELECTOR.MIB)

    def test_verify_rejects_any_unadmitted_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            self.write_plan(
                path,
                [["0", "a", "0", "1", "-1", "100", "20", "4", "8", "64", "unadmitted"]],
            )
            with self.assertRaisesRegex(SystemExit, "not fully resident"):
                SELECTOR.verify_plan(path)


if __name__ == "__main__":
    unittest.main()
