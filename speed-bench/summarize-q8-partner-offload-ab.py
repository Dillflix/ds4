#!/usr/bin/env python3
import csv
import json
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
logits_dirs: dict[tuple[int, str], Path] = {}
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
        logits_dirs[(repeat, variant)] = Path(row["logits"])

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
        if item["summary_calls"] <= 0:
            status_reasons.append("missing_runtime_count")
        if item["audit_calls"] > item["summary_calls"]:
            status_reasons.append("audit_exceeds_runtime")
        status = "+".join(status_reasons) if status_reasons else "ok"
        class_rows.append(
            {
                "repeat": repeat,
                "variant": variant,
                "evidence_status": status,
                "summary_calls": item["summary_calls"],
                "audit_calls": item["audit_calls"],
                "audit_coverage": (
                    float(item["audit_calls"]) / int(item["summary_calls"])
                    if int(item["summary_calls"]) > 0 else 1.0
                ),
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


def load_logits(directory: Path) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for path in sorted(directory.glob("frontier_*.logits.json")):
        payload = json.loads(path.read_text())
        values = payload.get("logits")
        if not isinstance(values, list):
            fail(f"missing logits array: {path}")
        result[path.name] = [float(value) for value in values]
    if not result:
        fail(f"no frontier logits in {directory}")
    return result


loaded_logits = {
    key: load_logits(directory) for key, directory in logits_dirs.items()
}
logit_rows: list[dict[str, object]] = []
for repeat in repeats:
    local = loaded_logits[(repeat, "local")]
    for variant in variants:
        candidate = loaded_logits[(repeat, variant)]
        if set(candidate) != set(local):
            fail(f"repeat {repeat} has mismatched {variant}/local logit frontiers")
        for filename in sorted(local):
            lhs = candidate[filename]
            rhs = local[filename]
            if len(lhs) != len(rhs):
                fail(f"logit length mismatch: repeat {repeat} {variant} {filename}")
            deltas = [abs(a - b) for a, b in zip(lhs, rhs)]
            logit_rows.append({
                "repeat": repeat,
                "variant": variant,
                "frontier": filename,
                "exact": int(lhs == rhs),
                "max_abs": max(deltas, default=0.0),
                "mean_abs": statistics.fmean(deltas) if deltas else 0.0,
                "top1_equal": int(
                    max(range(len(lhs)), key=lhs.__getitem__) ==
                    max(range(len(rhs)), key=rhs.__getitem__)
                ),
            })

determinism_rows: list[dict[str, object]] = []
reference_repeat = repeats[0]
for repeat in repeats[1:]:
    for variant in ("local", *variants):
        reference = loaded_logits[(reference_repeat, variant)]
        current = loaded_logits[(repeat, variant)]
        if set(current) != set(reference):
            fail(f"repeat {repeat} has mismatched {variant} logit frontiers")
        for filename in sorted(reference):
            exact = current[filename] == reference[filename]
            determinism_rows.append({
                "repeat": repeat,
                "variant": variant,
                "frontier": filename,
                "exact": int(exact),
            })

with (out_dir / "logit-comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(logit_rows[0]))
    writer.writeheader()
    writer.writerows(logit_rows)
with (out_dir / "logit-determinism.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(determinism_rows[0]))
    writer.writeheader()
    writer.writerows(determinism_rows)

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
    drift = [row for row in logit_rows if row["variant"] == variant]
    lines.append(
        f"logits max_abs={max(float(row['max_abs']) for row in drift):.9g} "
        f"max_mean_abs={max(float(row['mean_abs']) for row in drift):.9g} "
        f"top1_equal={sum(int(row['top1_equal']) for row in drift)}/{len(drift)}"
    )
    lines.append("")

(out_dir / "summary.txt").write_text("\n".join(lines).rstrip() + "\n")
print("\n".join(lines).rstrip())
