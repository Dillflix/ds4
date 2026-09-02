import array
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "speed-bench/summarize-sm75-t32-f16-production-ab.py"


class T32ProductionSummaryTests(unittest.TestCase):
    def make_fixture(self, root: Path, *, top1_change: bool = False) -> tuple[Path, Path]:
        runs = root / "runs.tsv"
        output = root / "summary"
        rows = []
        for layout_index, layout in enumerate(("mixed15", "all43")):
            for repeat in (1, 2, 3):
                for variant in ("control", "fused"):
                    run_dir = root / f"{layout}-r{repeat}-{variant}"
                    logits_dir = run_dir / "logits"
                    logits_dir.mkdir(parents=True)
                    csv_path = run_dir / "bench.csv"
                    with csv_path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=["ctx_tokens", "prefill_tps"])
                        writer.writeheader()
                        for context in (512, 4096, 32768):
                            base = 400.0 - layout_index * 20 - context / 10000
                            value = base * (1.01 if variant == "fused" else 1.0)
                            writer.writerow({"ctx_tokens": context, "prefill_tps": value})
                    for context in (512, 4096, 32768):
                        values = [0.25, 2.0, -1.0, 0.5]
                        if variant == "fused":
                            values[0] += 0.01
                            if top1_change and layout == "all43" and context == 4096:
                                values[0] = 3.0
                        raw = array.array("f", values)
                        stem = logits_dir / f"frontier_{context:06d}.logits"
                        stem.with_suffix(".logits.f32").write_bytes(raw.tobytes())
                        stem.with_suffix(".logits.json").write_text(json.dumps({
                            "frontier_tokens": context,
                            "vocab": len(values),
                            "logits": values,
                        }))
                    rows.append({
                        "model_layout": layout,
                        "repeat": repeat,
                        "slot": 1 if variant == "control" else 2,
                        "variant": variant,
                        "csv": csv_path,
                        "log": run_dir / "bench.log",
                        "logits": logits_dir,
                    })
        with runs.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return runs, output

    def run_summary(self, runs: Path, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(runs), str(output)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_dual_model_screen_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs, output = self.make_fixture(Path(tmp))
            result = self.run_summary(runs, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            gates = json.loads((output / "gates.json").read_text())
            self.assertTrue(gates["production_ab_screen_pass"])
            self.assertTrue(gates["full_quality_gate_required_before_default"])
            self.assertIn("mixed15", (output / "summary.md").read_text())
            self.assertIn("all43", (output / "summary.md").read_text())

    def test_top1_change_blocks_screen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs, output = self.make_fixture(Path(tmp), top1_change=True)
            result = self.run_summary(runs, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            gates = json.loads((output / "gates.json").read_text())
            self.assertFalse(gates["semantic_top1_gate"])
            self.assertFalse(gates["production_ab_screen_pass"])


if __name__ == "__main__":
    unittest.main()
