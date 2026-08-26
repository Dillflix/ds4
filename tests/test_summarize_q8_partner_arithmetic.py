#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "speed-bench/summarize-q8-partner-arithmetic.py"
ARMS = (
    "local", "f16", "w16-x16-sgemm", "w16-x32-sgemm",
    "w32-x32-sgemm", "w32-xq8-sgemm",
)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]],
              delimiter: str = ",") -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(root: Path) -> None:
    quality = root / "quality"
    quality.mkdir(parents=True)
    (root / "manifest.txt").write_text(
        "variants=" + ",".join(ARMS) + "\n"
        "gpu_devices=0,3,1,2\n"
        "gpu_vram=auto\n"
        "stage_split=22/21\n"
        "quality_ctx=32769\n"
        "t256_layers=15\n"
        "home_plan=frozen-for-all-arms\n",
        encoding="utf-8",
    )
    binding_fields = [
        "consumer_device", "resident_device", "partner_offload",
        "weight_offset", "weight_bytes", "in_dim", "out_dim", "fp16_bytes",
        "partner_scratch_tokens", "resident_weight_bytes",
        "partner_arithmetic", "label",
    ]
    home = {
        "consumer_device": 3, "resident_device": 3, "partner_offload": 0,
        "weight_offset": 100, "weight_bytes": 200, "in_dim": 2048,
        "out_dim": 4096, "fp16_bytes": 300, "partner_scratch_tokens": 0,
        "resident_weight_bytes": 300, "partner_arithmetic": "f16",
        "label": "tensor:blk.8.ffn_down_shexp.weight",
    }
    write_csv(quality / "local.bindings.csv", binding_fields, [home])
    audit_fields = [
        "module", "label", "physical_device", "in_dim", "out_dim",
        "result", "reason",
    ]
    write_csv(quality / "local.q8-audit.csv", audit_fields, [])
    for arm in ARMS[1:]:
        partner = {
            "consumer_device": 3, "resident_device": 2, "partner_offload": 1,
            "weight_offset": 400, "weight_bytes": 500, "in_dim": 8192,
            "out_dim": 4096, "fp16_bytes": 67108864,
            "partner_scratch_tokens": 2048,
            "resident_weight_bytes": 67108864 if arm == "f16" else 134217728,
            "partner_arithmetic": arm,
            "label": "tensor:blk.15.attn_output_b.weight",
        }
        write_csv(quality / f"{arm}.bindings.csv", binding_fields,
                  [home, partner])
        write_csv(quality / f"{arm}.q8-audit.csv", audit_fields, [{
            "module": "attention", "label": "attn_output_b",
            "physical_device": 2, "in_dim": 8192, "out_dim": 4096,
            "result": "f16_partner_hit" if arm == "f16" else "f32_partner_hit",
            "reason": "nvlink_offload" if arm == "f16" else arm,
        }])

    score_fields = [
        "id", "target_tokens", "nll", "avg_nll", "first_match",
        "greedy_lcp", "first_target_id", "first_greedy_id",
        "first_target_margin", "first_greedy_margin",
    ]
    for arm_index, arm in enumerate(ARMS):
        rows = []
        for case_index in range(2):
            greedy = 10 + (1 if arm_index >= 3 and case_index == 0 else 0)
            rows.append({
                "id": f"case_{case_index:03d}", "target_tokens": 4,
                "nll": 1.0 + arm_index * 0.01,
                "avg_nll": 0.25 + arm_index * 0.0025,
                "first_match": int(greedy == 10), "greedy_lcp": 2,
                "first_target_id": 10, "first_greedy_id": greedy,
                "first_target_margin": 0.1 - arm_index * 0.01,
                "first_greedy_margin": 0.1,
            })
        write_csv(quality / f"{arm}.tsv", score_fields, rows, "\t")


def run(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    make_fixture(root)
    result = run(root)
    assert result.returncode == 0, result.stderr
    assert (root / "arithmetic-isolation.json").is_file()
    assert "Experiment integrity: **PASS**" in result.stdout

    path = root / "quality/w32-x32-sgemm.bindings.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    rows[1]["weight_offset"] = "401"
    write_csv(path, list(rows[0]), rows)
    result = run(root)
    assert result.returncode != 0
    assert "same additive tensor set" in result.stderr

print("test_summarize_q8_partner_arithmetic: PASS")
