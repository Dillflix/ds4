#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench" / "summarize-q8-partner-quality-isolation.py"


def write_table(path: Path, fields: list[str], rows: list[dict[str, object]],
                delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def quality_rows(delta: float = 0.0) -> list[dict[str, object]]:
    return [
        {
            "id": f"case_{index:03d}", "target_tokens": 4,
            "nll": 4.0 * (1.0 + delta), "avg_nll": 1.0 + delta,
            "first_match": 1, "greedy_lcp": 4,
        }
        for index in range(100)
    ]


def binding(label: str, in_dim: int, out_dim: int, partner: bool = False,
            offset: int = 100) -> dict[str, object]:
    return {
        "consumer_device": 0, "resident_device": 1 if partner else 0,
        "partner_offload": 1 if partner else 0, "weight_offset": offset,
        "weight_bytes": 10, "in_dim": in_dim, "out_dim": out_dim,
        "fp16_bytes": 20, "partner_scratch_tokens": 2048 if partner else 0,
        "label": label,
    }


def make_fixture(root: Path, t256_delta: float = 0.0) -> None:
    (root / "manifest.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "manifest.txt").write_text(
        "gpu_devices=0,3,1,2\ngpu_vram=auto\nstage_split=22/21\n"
        "quality_ctx=32769\nhome_plan=frozen-for-candidates\n"
        "variants=local,t256,t32\n",
        encoding="utf-8",
    )
    qfields = ["id", "target_tokens", "nll", "avg_nll", "first_match", "greedy_lcp"]
    write_table(root / "quality/local.tsv", qfields, quality_rows(), "\t")
    write_table(root / "quality/t256.tsv", qfields, quality_rows(t256_delta), "\t")
    write_table(root / "quality/t32.tsv", qfields, quality_rows(), "\t")
    bfields = [
        "consumer_device", "resident_device", "partner_offload", "weight_offset",
        "weight_bytes", "in_dim", "out_dim", "fp16_bytes",
        "partner_scratch_tokens", "label",
    ]
    home = binding("tensor:blk.0.attn_q_b.weight", 1024, 32768)
    write_table(root / "quality/local.bindings.csv", bfields, [home])
    write_table(
        root / "quality/t256.bindings.csv", bfields,
        [home, binding("tensor:blk.15.attn_output_b.weight", 8192, 4096, True, 200)],
    )
    write_table(
        root / "quality/t32.bindings.csv", bfields,
        [home, binding("tensor:blk.16.attn_q_b.weight", 1024, 32768, True, 300)],
    )
    afields = [
        "module", "label", "physical_device", "in_dim", "out_dim",
        "result", "reason",
    ]
    write_table(root / "quality/local.q8-audit.csv", afields, [])
    write_table(root / "quality/t256.q8-audit.csv", afields, [{
        "module": "attention_output", "label": "attn_output_b",
        "physical_device": 1, "in_dim": 8192, "out_dim": 4096,
        "result": "f16_partner_hit", "reason": "nvlink_offload",
    }])
    write_table(root / "quality/t32.q8-audit.csv", afields, [{
        "module": "attn_q_b", "label": "attn_q_b", "physical_device": 1,
        "in_dim": 1024, "out_dim": 32768, "result": "f16_partner_hit",
        "reason": "nvlink_offload",
    }])
    (root / "quality/local.log").write_text(
        "score_official: runtime_path=production\n"
        "ds4: CUDA EP forced pipeline split 22/21\n",
        encoding="utf-8",
    )
    for variant in ("t256", "t32"):
        (root / f"quality/{variant}.log").write_text(
            "score_official: runtime_path=production\n"
            "ds4: CUDA EP forced pipeline split 22/21\n"
            f"partner-classes={variant} home-order=frozen\n"
            "ds4: CUDA q8 fp16 partner summary: calls=1\n",
            encoding="utf-8",
        )


class QualityIsolationTests(unittest.TestCase):
    def run_it(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SUMMARIZER), str(root)],
            capture_output=True, text=True, check=False,
        )

    def test_additive_class_isolation_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-q8-isolation-") as raw:
            root = Path(raw)
            make_fixture(root)
            result = self.run_it(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "quality-isolation.json").read_text())
            self.assertTrue(payload["experiment_integrity"])
            self.assertEqual(payload["variants"]["t256"]["partner_layers"], [15])
            self.assertEqual(payload["variants"]["t32"]["partner_layers"], [16])
            self.assertTrue(payload["variants"]["t256"]["quality_pass"])

    def test_home_reshuffle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-q8-reshuffle-") as raw:
            root = Path(raw)
            make_fixture(root)
            path = root / "quality/t256.bindings.csv"
            rows = read_rows(path)
            rows[0]["resident_device"] = "1"
            write_table(path, list(rows[0]), rows)
            result = self.run_it(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("changed the primary/home binding set", result.stderr)

    def test_quality_failure_is_reported_without_invalidating_experiment(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds4-q8-quality-fail-") as raw:
            root = Path(raw)
            make_fixture(root, t256_delta=0.1)
            result = self.run_it(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((root / "quality-isolation.json").read_text())
            self.assertFalse(payload["variants"]["t256"]["quality_pass"])
            self.assertIn("| t256 | 15 |", (root / "summary.md").read_text())


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
