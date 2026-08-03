#!/usr/bin/env python3

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_q2_down_for_cache",
    ROOT / "speed-bench" / "select-q2-down-for-cache.py",
)
SELECTOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SELECTOR)


class SelectQ2DownForCacheTests(unittest.TestCase):
    def test_cache_missing_deduplicates_calls_and_keeps_resident_weights(self):
        fields = [
            "physical_device",
            "weight_offset",
            "weight_bytes",
            "result",
            "reason",
            "fp16_bytes",
        ]
        rows = [
            ["0", "100", "20", "native_q8", "budget_or_limit", "64"],
            ["0", "100", "20", "native_q8", "budget_or_limit", "64"],
            ["0", "200", "20", "native_q8", "budget_or_limit", "128"],
            ["0", "200", "20", "f16_hit", "resident", "128"],
            ["2", "300", "20", "native_q8", "budget_or_limit", "256"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            self.assertEqual(SELECTOR.cache_missing_by_device(path), {0: 64, 2: 256})

    def test_layout_and_preference_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.log"
            path.write_text(
                "multi-GPU layout:\n"
                "  GPU0: layers 0-21 + embedding\n"
                "  GPU1: layers 22-42 + output head\n",
                encoding="utf-8",
            )
            self.assertEqual(SELECTOR.parse_layout(path), [(0, 21), (22, 42)])
        self.assertEqual(
            SELECTOR.layer_order(3, 42, "36,3-5,36,100"),
            [36, 3, 4, 5],
        )

    def test_down_only_plan_when_it_fits(self):
        q2, iq2, reclaim = SELECTOR.select_stage_recipe(
            required_bytes=721 * SELECTOR.MIB,
            q2_candidates=[3, 4, 5, 6],
            iq2_candidates=[3, 4, 5, 6],
            q2_saving_bytes=240 * SELECTOR.MIB,
            iq2_pair_saving_bytes=624 * SELECTOR.MIB,
        )
        self.assertEqual(q2, [3, 4, 5, 6])
        self.assertEqual(iq2, [])
        self.assertEqual(reclaim, 960 * SELECTOR.MIB)

    def test_minimum_iq2_pairs_then_minimum_q2_down(self):
        q2, iq2, reclaim = SELECTOR.select_stage_recipe(
            required_bytes=6001 * SELECTOR.MIB,
            q2_candidates=list(range(3, 22)),
            iq2_candidates=list(range(10, 22)) + list(range(3, 10)),
            q2_saving_bytes=240 * SELECTOR.MIB,
            iq2_pair_saving_bytes=624 * SELECTOR.MIB,
        )
        self.assertEqual(len(iq2), 3)
        self.assertEqual(iq2, [10, 11, 12])
        self.assertEqual(len(q2), 18)
        self.assertGreaterEqual(reclaim, 6001 * SELECTOR.MIB)
        self.assertLess(reclaim, (6001 + 240) * SELECTOR.MIB)

    def test_infeasible_even_with_every_iq2_pair(self):
        with self.assertRaisesRegex(SystemExit, "maximum=1728 MiB"):
            SELECTOR.select_stage_recipe(
                required_bytes=2000 * SELECTOR.MIB,
                q2_candidates=[3, 4],
                iq2_candidates=[3, 4],
                q2_saving_bytes=240 * SELECTOR.MIB,
                iq2_pair_saving_bytes=624 * SELECTOR.MIB,
            )


if __name__ == "__main__":
    unittest.main()
