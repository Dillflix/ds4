#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench" / "summarize-q8-t32-fused-ab.py"
VARIANTS = ("old_local", "new_local", "old_partner", "new_partner")


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        out = tmp / "out"
        out.mkdir()
        records = []
        for repeat in (1, 2):
            for index, variant in enumerate(VARIANTS):
                csv_path = tmp / f"{variant}-r{repeat}.csv"
                logits_dir = tmp / f"{variant}-r{repeat}-logits"
                logits_dir.mkdir()
                with csv_path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["ctx_tokens", "prefill_tps"])
                    writer.writerow([2048, 400.0 + 10.0 * index + repeat])
                    writer.writerow([8192, 300.0 + 10.0 * index + repeat])
                for frontier in (2048, 8192):
                    old = [0.25, -0.5, 1.0]
                    new = [0.25, -0.499, 1.0]
                    values = old if variant in ("old_local", "old_partner") else new
                    (logits_dir / f"frontier_{frontier:06d}.logits.json").write_text(
                        json.dumps({"ctx_tokens": frontier, "logits": values})
                    )
                records.append({
                    "repeat": repeat,
                    "slot": index + 1,
                    "variant": variant,
                    "csv": csv_path,
                    "log": "unused",
                    "audit": "unused",
                    "cache_before": "unused",
                    "cache_after": "unused",
                    "logits": logits_dir,
                })
        runs = tmp / "runs.tsv"
        with runs.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(records[0]), delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(records)

        subprocess.run(
            [sys.executable, str(SUMMARIZER), str(runs), str(out)],
            check=True,
            capture_output=True,
            text=True,
        )
        paired = list(csv.DictReader((out / "paired-samples.csv").open()))
        logits = list(csv.DictReader((out / "logit-comparison.csv").open()))
        assert len(paired) == 2 * 5 * 2
        # Four cross-policy comparisons per repeat/frontier, plus one
        # repeat-determinism comparison per variant/frontier.
        assert len(logits) == (2 * 4 * 2) + (4 * 2)
        required = [row for row in logits if row["required_exact"] == "1"]
        assert required and all(row["exact"] == "1" for row in required)
        assert all("_r2_vs_r1" in row["comparison"] for row in required)
        cross_policy = [
            row for row in logits if "_r2_vs_r1" not in row["comparison"]
        ]
        assert cross_policy and all(
            row["required_exact"] == "0" for row in cross_policy
        )
        drift = [
            row for row in logits
            if row["comparison"] == "new_local_vs_old_local"
        ]
        assert drift and all(row["exact"] == "0" for row in drift)
        assert all(float(row["max_abs"]) > 0.0 for row in drift)
        summary = (out / "summary.txt").read_text()
        assert "[new_local/old_local]" in summary
        assert "[new_partner/old_partner]" in summary
        assert "top1_equal=4/4" in summary


if __name__ == "__main__":
    main()
