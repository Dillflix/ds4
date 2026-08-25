#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench" / "summarize-q8-partner-offload-ab.py"
AUDIT_HEADER = [
    "sequence", "module", "label", "layer", "token_offset",
    "physical_device", "weight_offset", "weight_bytes", "in_dim",
    "out_dim", "fp16_bytes", "result", "reason", "cache_bytes_after",
]


def audit_rows(variant: str) -> list[list[object]]:
    shapes = {
        "t32": ("attn_q_b", 1024, 32768),
        "t256": ("attn_output_b", 8192, 4096),
        "shared_down": ("shared_down", 2048, 4096),
    }
    names = [variant] if variant in shapes else (["t32", "t256"] if variant == "legacy" else [])
    rows = []
    for name_index, name in enumerate(names):
        module, in_dim, out_dim = shapes[name]
        for pair_index, physical_device in enumerate((1, 2)):
            sequence = name_index * 2 + pair_index
            rows.append([
                sequence, module, "q8_0", 3 if pair_index == 0 else 36,
                0, physical_device, sequence * 4096, 1024,
                in_dim, out_dim, in_dim * out_dim * 2,
                "f16_partner_hit", "nvlink_offload", 1024,
            ])
    return rows


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        out = tmp / "out"
        out.mkdir()
        records = []
        variants = ["local", "t32", "t256", "shared_down", "legacy"]
        for repeat in (1, 2):
            for variant_index, variant in enumerate(variants):
                stem = tmp / f"{variant}-r{repeat}"
                csv_path = stem.with_suffix(".csv")
                log_path = stem.with_suffix(".log")
                audit_path = stem.with_suffix(".audit.csv")
                logits_dir = tmp / f"{variant}-r{repeat}-logits"
                logits_dir.mkdir()
                with csv_path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["ctx_tokens", "prefill_tps"])
                    writer.writerow([2048, 400 + variant_index * 10 + repeat])
                    writer.writerow([8192, 300 + variant_index * 10 + repeat])
                rows = audit_rows(variant)
                calls = len(rows)
                runtime_calls = 5 if variant == "t32" and repeat == 1 else calls
                log_path.write_text(
                    "" if calls == 0 else
                    f"ds4: CUDA q8 fp16 partner summary: calls={runtime_calls} "
                    "activation=0.01 GiB result=0.02 GiB\n"
                )
                with audit_path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(AUDIT_HEADER)
                    writer.writerows(rows)
                for frontier in (2048, 8192):
                    values = [
                        0.25 + 0.001 * variant_index,
                        -0.5,
                        1.0,
                    ]
                    (logits_dir / f"frontier_{frontier:06d}.logits.json").write_text(
                        json.dumps({"ctx_tokens": frontier, "logits": values})
                    )
                records.append({
                    "repeat": repeat,
                    "slot": variant_index + 1,
                    "variant": variant,
                    "csv": csv_path,
                    "log": log_path,
                    "audit": audit_path,
                    "cache_before": "unused",
                    "cache_after": "unused",
                    "logits": logits_dir,
                })
        runs = tmp / "runs.tsv"
        with runs.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(records)

        subprocess.run(
            [sys.executable, str(SUMMARIZER), str(runs), str(out)],
            check=True,
            capture_output=True,
            text=True,
        )
        paired = list(csv.DictReader((out / "paired-samples.csv").open()))
        evidence = list(csv.DictReader((out / "class-evidence.csv").open()))
        logits = list(csv.DictReader((out / "logit-comparison.csv").open()))
        determinism = list(csv.DictReader((out / "logit-determinism.csv").open()))
        assert len(paired) == 2 * 4 * 2
        assert len(evidence) == 2 * 4
        assert len(logits) == 2 * 4 * 2
        assert len(determinism) == 5 * 2
        assert all(row["exact"] == "1" for row in determinism)
        t32 = next(row for row in evidence if row["variant"] == "t32")
        assert t32["evidence_status"] == "ok"
        assert (t32["process_total_calls"] == "5" and
                t32["audit_sample_calls"] == "2")
        assert (t32["audit_t32_calls"] == "2" and
                t32["audit_t256_calls"] == "0")
        assert t32["audit_partner_devices"] == "1:1;2:1"
        legacy = next(row for row in evidence if row["variant"] == "legacy")
        assert (legacy["audit_t32_calls"] == "2" and
                legacy["audit_t256_calls"] == "2")
        summary = (out / "summary.txt").read_text()
        assert "[shared_down]" in summary
        assert "median_candidate_tps=" in summary
        assert "median_local_tps=" in summary
        assert "median_candidate/local=" in summary
        assert "process_total_calls median=" in summary

        # Invalid class evidence must remain summarizable so the shell driver
        # can finish all bounded Nsight captures before rejecting the archive.
        t32_audit = tmp / "t32-r1.audit.csv"
        with t32_audit.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(audit_rows("shared_down"))
        out_bad = tmp / "out-bad"
        out_bad.mkdir()
        subprocess.run(
            [sys.executable, str(SUMMARIZER), str(runs), str(out_bad)],
            check=True,
            capture_output=True,
            text=True,
        )
        bad_evidence = list(csv.DictReader((out_bad / "class-evidence.csv").open()))
        bad_t32 = next(
            row for row in bad_evidence
            if row["repeat"] == "1" and row["variant"] == "t32"
        )
        assert "contaminated" in bad_t32["evidence_status"]


if __name__ == "__main__":
    main()
