#!/usr/bin/env python3

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "speed-bench" / "validate-sm75-stage-aware-plan.py"
PLAN_FIELDS = [
    "sequence", "label", "consumer_device", "fallback_device",
    "target_device", "placement_locked", "resident_device", "weight_offset",
    "weight_bytes", "in_dim", "out_dim", "fp16_bytes", "status",
]
BINDING_FIELDS = [
    "consumer_device", "resident_device", "partner_offload", "weight_offset",
    "weight_bytes", "in_dim", "out_dim", "fp16_bytes",
    "partner_scratch_tokens", "resident_weight_bytes", "partner_arithmetic",
    "label", "allocation_id", "used_calls", "live",
]


def write(path: Path, fields: list[str], items: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)


class StageAwarePlanTests(unittest.TestCase):
    def test_fixed_pair_confinement_and_live_inventory(self) -> None:
        devices = [0, 3, 1, 2]
        plan: list[dict[str, object]] = []
        bindings: list[dict[str, object]] = []
        for sequence in range(344):
            layer = sequence % 43
            stage = 0 if layer < 22 else 1
            home, partner = devices[stage], devices[stage + 2]
            resident = partner if sequence % 3 == 0 else home
            label = f"tensor:blk.{layer}.attn_q_b.weight:{sequence}"
            common = {
                "label": label, "weight_offset": str(sequence * 4096),
                "weight_bytes": "1024", "in_dim": "32", "out_dim": "64",
                "fp16_bytes": "4096",
            }
            plan.append(
                {
                    "sequence": sequence, **common,
                    "consumer_device": home, "fallback_device": partner,
                    "target_device": resident,
                    "placement_locked": int(resident == partner),
                    "resident_device": resident,
                    "status": "partner" if resident == partner else "home",
                }
            )
            bindings.append(
                {
                    **common, "consumer_device": home,
                    "resident_device": resident,
                    "partner_offload": int(resident == partner),
                    "partner_scratch_tokens": "2048",
                    "resident_weight_bytes": "4096",
                    "partner_arithmetic": "f16", "allocation_id": sequence + 1,
                    "used_calls": "1", "live": "1",
                }
            )
        with tempfile.TemporaryDirectory(prefix="ds4-stage-plan-") as tmp:
            root = Path(tmp)
            write(root / "plan.csv", PLAN_FIELDS, plan)
            write(root / "bindings.csv", BINDING_FIELDS, bindings)
            result = subprocess.run(
                [sys.executable, str(VALIDATOR), str(root / "plan.csv"),
                 str(root / "bindings.csv"), "0,3,1,2", str(root / "out.csv")],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("344/344 live bindings", result.stdout)
            self.assertGreater((root / "out.csv").stat().st_size, 0)

            plan[22]["resident_device"] = 1
            plan[22]["target_device"] = 1
            write(root / "bad.csv", PLAN_FIELDS, plan)
            bad = subprocess.run(
                [sys.executable, str(VALIDATOR), str(root / "bad.csv"),
                 str(root / "bindings.csv"), "0,3,1,2", str(root / "bad-out.csv")],
                text=True, capture_output=True,
            )
            self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
