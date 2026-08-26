#!/usr/bin/env python3

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_q4_iq2_full_f16",
    ROOT / "speed-bench" / "select-q4-iq2-full-f16.py",
)
SELECTOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SELECTOR)


class SelectQ4Iq2FullF16Tests(unittest.TestCase):
    fields = [
        "sequence",
        "label",
        "consumer_device",
        "fallback_device",
        "target_device",
        "placement_locked",
        "resident_device",
        "weight_offset",
        "weight_bytes",
        "in_dim",
        "out_dim",
        "fp16_bytes",
        "status",
    ]

    def write_plan(self, path: Path, rows: list[list[str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.fields)
            writer.writerows(rows)

    def write_memory(self, path: Path, free_mib: dict[int, int]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "logical_tier",
                    "physical_device",
                    "free_bytes",
                    "total_bytes",
                    "q8_fp16_cached_bytes",
                    "q8_fp16_reserve_bytes",
                ]
            )
            for tier, device in enumerate([0, 3, 1, 2]):
                writer.writerow(
                    [
                        tier,
                        device,
                        free_mib[device] * SELECTOR.MIB,
                        48 * 1024 * SELECTOR.MIB,
                        2048 * SELECTOR.MIB,
                        768 * SELECTOR.MIB,
                    ]
                )

    def production_rows(self) -> list[list[str]]:
        rows: list[list[str]] = []
        sequence = 0
        for suffixes in SELECTOR.EXPECTED_PLAN_SUFFIXES.values():
            for layer in range(SELECTOR.N_LAYERS):
                for suffix in suffixes:
                    rows.append(
                        [
                            str(sequence),
                            f"tensor:blk.{layer}.{suffix}",
                            "0",
                            "-1",
                            "0",
                            "0",
                            "0",
                            str(1000 + sequence),
                            "20",
                            "4",
                            "8",
                            "64",
                            "home",
                        ]
                    )
                    sequence += 1
        return rows

    def test_locked_partner_and_fixed_home_are_charged_to_exact_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            self.write_plan(
                path,
                [
                    ["0", "tensor:blk.3.attn_output_b.weight", "0", "1", "1", "1", "-1", "100", "20", "4", "8", "128", "unadmitted"],
                    ["1", "tensor:blk.3.ffn_gate_shexp.weight", "0", "-1", "0", "0", "-1", "200", "20", "4", "8", "64", "unadmitted"],
                ],
            )
            required, detail = SELECTOR.required_reclaim_by_stage(
                path, SELECTOR.device_pairs([0, 3, 1, 2]), 32
            )
            self.assertEqual(required, {0: 160})
            self.assertEqual(detail[0][0], {0: 64, 1: 128})

    def test_flexible_candidates_use_exact_indivisible_subset_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            self.write_plan(
                path,
                [
                    ["0", "a", "0", "1", "0", "0", "-1", "100", "20", "4", "8", "64", "unadmitted"],
                    ["1", "b", "0", "1", "0", "0", "-1", "200", "20", "4", "8", "128", "unadmitted"],
                ],
            )
            required, detail = SELECTOR.required_reclaim_by_stage(
                path, SELECTOR.device_pairs([0, 3, 1, 2]), 0
            )
            self.assertEqual(required, {0: 128})
            self.assertEqual(detail[0][1], [64, 128])

    def test_selector_uses_only_matched_iq2_gate_up_pairs(self):
        layers, reclaim = SELECTOR.select_iq2_layers(
            1250 * SELECTOR.MIB,
            [12, 13, 14, 15],
            624 * SELECTOR.MIB,
        )
        self.assertEqual(layers, [12, 13, 14])
        self.assertEqual(reclaim, 1872 * SELECTOR.MIB)

    def test_all_iq2_calibration_promotes_only_pair_safe_q4_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.csv"
            # After protecting the 768 MiB cache reserve and 512 MiB extra,
            # stage 0 can promote two pairs; stage 1 can promote three.
            self.write_memory(
                path,
                {
                    # Stage 0 first pays for the three mandatory Q4 pairs in
                    # layers 0-2, then has room for two optional promotions.
                    0: 768 + 512 + 3 * 624 + 2 * 624 + 200,
                    1: 768 + 512 + 3 * 624 + 2 * 624,
                    3: 768 + 512 + 4 * 624,
                    2: 768 + 512 + 3 * 624,
                },
            )
            selected = SELECTOR.select_from_all_iq2_calibration(
                [(0, 21), (22, 42)],
                SELECTOR.device_pairs([0, 3, 1, 2]),
                SELECTOR.read_device_memory(path),
                list(range(3, 43)),
                3,
                42,
                624 * SELECTOR.MIB,
                512 * SELECTOR.MIB,
            )
            self.assertEqual(selected, list(range(3, 20)) + list(range(22, 40)))

    def test_all_iq2_calibration_charges_mandatory_q4_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.csv"
            # This would have appeared to allow two stage-0 promotions under
            # the old accounting, but it cannot even fund layers 0-2 becoming
            # Q4 while preserving reserve + headroom.
            self.write_memory(
                path,
                {
                    0: 768 + 512 + 2 * 624,
                    1: 768 + 512 + 2 * 624,
                    3: 768 + 512 + 4 * 624,
                    2: 768 + 512 + 3 * 624,
                },
            )
            with self.assertRaisesRegex(SystemExit, "mandatory Q4 gate/up"):
                SELECTOR.select_from_all_iq2_calibration(
                    [(0, 21), (22, 42)],
                    SELECTOR.device_pairs([0, 3, 1, 2]),
                    SELECTOR.read_device_memory(path),
                    list(range(3, 43)),
                    3,
                    42,
                    624 * SELECTOR.MIB,
                    512 * SELECTOR.MIB,
                )

    def test_all_iq2_calibration_requires_a_complete_layer_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.csv"
            self.write_memory(path, {0: 4096, 1: 4096, 2: 4096, 3: 4096})
            with self.assertRaisesRegex(SystemExit, "must order every routed layer"):
                SELECTOR.select_from_all_iq2_calibration(
                    [(0, 21), (22, 42)],
                    SELECTOR.device_pairs([0, 3, 1, 2]),
                    SELECTOR.read_device_memory(path),
                    list(range(3, 42)),
                    3,
                    42,
                    624 * SELECTOR.MIB,
                    512 * SELECTOR.MIB,
                )

    def test_verify_accepts_exact_344_row_production_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            rows = self.production_rows()
            self.assertEqual(len(rows), 344)
            self.write_plan(path, rows)
            SELECTOR.verify_plan(path, SELECTOR.DEFAULT_EXPECTED_CANDIDATES)

    def test_verify_rejects_stale_473_row_split_head_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            rows = self.production_rows()
            for index in range(129):
                row = rows[index].copy()
                row[0] = str(len(rows))
                row[1] += ".dead-half"
                rows.append(row)
            self.assertEqual(len(rows), 473)
            self.write_plan(path, rows)
            with self.assertRaisesRegex(SystemExit, "has 473 candidates; expected 344"):
                SELECTOR.verify_plan(path, SELECTOR.DEFAULT_EXPECTED_CANDIDATES)

    def test_verify_rejects_stale_473_expected_count_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            self.write_plan(path, self.production_rows())
            with self.assertRaisesRegex(
                SystemExit, "expected-candidates must be 344"
            ):
                SELECTOR.verify_plan(path, 473)

    def test_verify_rejects_dead_alternate_even_at_344_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            rows = self.production_rows()
            rows[0][1] = "tensor:blk.0.attn_output_a.weight.dead-full-fallback"
            self.write_plan(path, rows)
            with self.assertRaisesRegex(SystemExit, "unknown/dead alternate"):
                SELECTOR.verify_plan(path, SELECTOR.DEFAULT_EXPECTED_CANDIDATES)

    def test_verify_rejects_any_unadmitted_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            rows = self.production_rows()
            rows[0][12] = "unadmitted"
            rows[0][6] = "-1"
            self.write_plan(path, rows)
            with self.assertRaisesRegex(SystemExit, "incomplete"):
                SELECTOR.verify_plan(path, SELECTOR.DEFAULT_EXPECTED_CANDIDATES)


if __name__ == "__main__":
    unittest.main()
