#!/usr/bin/env python3
import csv
import re
import statistics
import sys
from collections import Counter
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def classify(row: dict[str, str]) -> str:
    text = f"{row.get('module', '')} {row.get('label', '')}"
    if "attn_q_b" in text or (
        row.get("in_dim") == "1024" and row.get("out_dim") == "32768"
    ):
        return "t32"
    if "attn_output_b" in text or (
        row.get("in_dim") == "8192" and row.get("out_dim") == "4096"
    ):
        return "t256"
    if "shared_down" in text or "ffn_down_shexp" in text or (
        row.get("in_dim") == "2048" and row.get("out_dim") == "4096"
    ):
        return "shared_down"
    return "other"


if len(sys.argv) != 3:
    fail("usage: summarize-q8-partner-offload-ab.py RUNS.tsv OUTPUT_DIR")

runs_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
samples: dict[tuple[int, str], dict[int, float]] = {}
evidence: dict[tuple[int, str], dict[str, object]] = {}
with runs_path.open(newline="") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        repeat = int(row["repeat"])
        variant = row["variant"]
        with Path(row["csv"]).open(newline="") as csv_file:
            values = {
                int(item["ctx_tokens"]): float(item["prefill_tps"])
                for item in csv.DictReader(csv_file)
            }
        samples[(repeat, variant)] = values

        calls = 0
        activation_gib = 0.0
        result_gib = 0.0
        text = Path(row["log"]).read_text(errors="replace")
        match = re.search(
            r"q8 fp16 partner summary: calls=(\d+) "
            r"activation=([0-9.]+) GiB result=([0-9.]+) GiB",
            text,
        )
        if match:
            calls = int(match.group(1))
            activation_gib = float(match.group(2))
            result_gib = float(match.group(3))
        with Path(row["audit"]).open(newline="") as audit_file:
            offload_rows = [
                item
                for item in csv.DictReader(audit_file)
                if item["result"] == "f16_partner_hit"
                and item["reason"] == "nvlink_offload"
            ]
        by_class = Counter(classify(item) for item in offload_rows)
        evidence[(repeat, variant)] = {
            "summary_calls": calls,
            "audit_calls": len(offload_rows),
            "activation_gib": activation_gib,
            "result_gib": result_gib,
            "classes": by_class,
        }

repeats = sorted({repeat for repeat, _ in samples})
variants = sorted({variant for _, variant in samples if variant != "local"})
paired: list[dict[str, object]] = []
class_rows: list[dict[str, object]] = []
for repeat in repeats:
    local = samples.get((repeat, "local"))
    if not local:
        fail(f"repeat {repeat} lacks a local control")
    local_evidence = evidence[(repeat, "local")]
    if local_evidence["summary_calls"] != 0 or local_evidence["audit_calls"] != 0:
        fail(f"repeat {repeat} local control used partner execution")
    for variant in variants:
        candidate = samples.get((repeat, variant))
        if not candidate or set(candidate) != set(local):
            fail(f"repeat {repeat} lacks matched {variant}/local frontiers")
        item = evidence[(repeat, variant)]
        classes = item["classes"]
        assert isinstance(classes, Counter)
        total = sum(classes.values())
        valid = False
        status_reasons: list[str] = []
        if variant == "t32":
            valid = classes["t32"] > 0 and total == classes["t32"]
        elif variant == "t256":
            valid = classes["t256"] > 0 and total == classes["t256"]
        elif variant == "shared_down":
            valid = classes["shared_down"] > 0 and total == classes["shared_down"]
        elif variant == "legacy":
            valid = total > 0 and total == classes["t32"] + classes["t256"]
        if total == 0:
            status_reasons.append("not_exercised")
        elif not valid:
            status_reasons.append("contaminated")
        if item["summary_calls"] != item["audit_calls"]:
            status_reasons.append("count_mismatch")
        status = "+".join(status_reasons) if status_reasons else "ok"
        class_rows.append(
            {
                "repeat": repeat,
                "variant": variant,
                "evidence_status": status,
                "summary_calls": item["summary_calls"],
                "audit_calls": item["audit_calls"],
                "t32_calls": classes["t32"],
                "t256_calls": classes["t256"],
                "shared_down_calls": classes["shared_down"],
                "other_calls": classes["other"],
                "activation_gib": item["activation_gib"],
                "result_gib": item["result_gib"],
            }
        )
        for context in sorted(candidate):
            paired.append(
                {
                    "repeat": repeat,
                    "variant": variant,
                    "ctx_tokens": context,
                    "candidate_tps": candidate[context],
                    "local_tps": local[context],
                    "candidate_over_local": candidate[context] / local[context],
                }
            )

with (out_dir / "paired-samples.csv").open("w", newline="") as handle:
    fields = list(paired[0]) if paired else []
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(paired)

with (out_dir / "class-evidence.csv").open("w", newline="") as handle:
    fields = list(class_rows[0]) if class_rows else []
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(class_rows)

lines: list[str] = []
for variant in variants:
    lines.append(f"[{variant}]")
    for context in sorted({int(row["ctx_tokens"]) for row in paired}):
        ratios = [
            float(row["candidate_over_local"])
            for row in paired
            if row["variant"] == variant and row["ctx_tokens"] == context
        ]
        if not ratios:
            continue
        lines.append(
            f"ctx={context} median_candidate/local={statistics.median(ratios):.5f} "
            f"min={min(ratios):.5f} max={max(ratios):.5f} n={len(ratios)}"
        )
    selected = [row for row in class_rows if row["variant"] == variant]
    for class_name in ("t32", "t256", "shared_down", "other"):
        values = [int(row[f"{class_name}_calls"]) for row in selected]
        lines.append(
            f"{class_name}_calls median={statistics.median(values):.1f} "
            f"min={min(values)} max={max(values)}"
        )
    lines.append("")

(out_dir / "summary.txt").write_text("\n".join(lines).rstrip() + "\n")
print("\n".join(lines).rstrip())
