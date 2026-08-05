#!/usr/bin/env python3
import csv
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
    for sequence, name in enumerate(names):
        module, in_dim, out_dim = shapes[name]
        rows.append([
            sequence, module, "q8_0", 3, 0, 1, sequence * 4096,
            1024, in_dim, out_dim, in_dim * out_dim * 2,
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
                with csv_path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["ctx_tokens", "prefill_tps"])
                    writer.writerow([2048, 400 + variant_index * 10 + repeat])
                    writer.writerow([8192, 300 + variant_index * 10 + repeat])
                rows = audit_rows(variant)
                calls = len(rows)
                log_path.write_text(
                    "" if calls == 0 else
                    f"ds4: CUDA q8 fp16 partner summary: calls={calls} "
                    "activation=0.01 GiB result=0.02 GiB\n"
                )
                with audit_path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(AUDIT_HEADER)
                    writer.writerows(rows)
                records.append({
                    "repeat": repeat,
                    "slot": variant_index + 1,
                    "variant": variant,
                    "csv": csv_path,
                    "log": log_path,
                    "audit": audit_path,
                    "cache_before": "unused",
                    "cache_after": "unused",
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
        assert len(paired) == 2 * 4 * 2
        assert len(evidence) == 2 * 4
        t32 = next(row for row in evidence if row["variant"] == "t32")
        assert t32["evidence_status"] == "ok"
        assert t32["t32_calls"] == "1" and t32["t256_calls"] == "0"
        legacy = next(row for row in evidence if row["variant"] == "legacy")
        assert legacy["t32_calls"] == "1" and legacy["t256_calls"] == "1"
        summary = (out / "summary.txt").read_text()
        assert "[shared_down]" in summary and "median_candidate/local=" in summary

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
