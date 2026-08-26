#!/usr/bin/env python3
"""Summarize and gate the production-default T256 validation."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from q8_partner_audit import REQUIRED_COLUMNS, class_evidence_valid, collect


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_audit(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            fail(f"{path} lacks audit columns: {','.join(sorted(missing))}")
        return list(reader)


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key not in values:
            values[key] = value
    return values


def quality_summary(rows: dict[str, dict[str, str]]) -> dict[str, float]:
    tokens = sum(int(row["target_tokens"]) for row in rows.values())
    if not rows or tokens <= 0:
        fail("quality table is empty")
    return {
        "cases": float(len(rows)),
        "tokens": float(tokens),
        "avg_nll": sum(float(row["nll"]) for row in rows.values()) / tokens,
        "first_matches": float(sum(int(row["first_match"]) for row in rows.values())),
        "avg_lcp": statistics.fmean(float(row["greedy_lcp"]) for row in rows.values()),
    }


def bootstrap_upper_delta(
    local: list[tuple[float, int]], candidate: list[tuple[float, int]],
    draws: int = 10000,
) -> float:
    rng = random.Random(0x75_256)
    values: list[float] = []
    n = len(local)
    for _ in range(draws):
        delta = 0.0
        tokens = 0
        for _ in range(n):
            index = rng.randrange(n)
            delta += candidate[index][0] - local[index][0]
            tokens += local[index][1]
        values.append(delta / tokens)
    values.sort()
    return values[math.ceil(0.95 * len(values)) - 1]


if len(sys.argv) != 2:
    fail("usage: summarize-q8-partner-production.py VALIDATION_DIR")
root = Path(sys.argv[1]).resolve()
manifest = read_manifest(root / "manifest.txt")
if (manifest.get("gpu_devices") != "0,3,1,2" or
        manifest.get("gpu_vram") != "auto" or
        manifest.get("stage_split") != "22/21" or
        manifest.get("ctx_start") != "16384" or
        manifest.get("ctx_max") != "65536" or
        manifest.get("step_mul") != "2" or
        manifest.get("quality_ctx") != "65537" or
        manifest.get("gpu_exactness_test") != "required-and-enabled"):
    fail("manifest does not describe the fixed production validation target")

partner_devices = (1, 2)
quality_policy_ok = True
for variant in ("local", "default"):
    log_text = (root / f"quality/{variant}.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if ("score_official: runtime_path=production" not in log_text or
            "CUDA EP forced pipeline split 22/21" not in log_text):
        quality_policy_ok = False
    audit_rows = read_audit(root / f"quality/{variant}.q8-audit.csv")
    classes, _, unexpected = collect(audit_rows, partner_devices)
    if not class_evidence_valid(variant, classes) or unexpected != 0:
        quality_policy_ok = False
    binding_rows = read_rows(root / f"quality/{variant}.bindings.csv")
    partner = [row for row in binding_rows if row["partner_offload"] == "1"]
    if variant == "local":
        if partner or "CUDA q8 fp16 partner summary:" in log_text:
            quality_policy_ok = False
    else:
        if ("partner-classes=t256" not in log_text or
                "CUDA q8 fp16 partner summary:" not in log_text or
                not partner or any(
                    row["in_dim"] != "8192" or row["out_dim"] != "4096"
                    for row in partner
                )):
            quality_policy_ok = False

planner_text = (root / "planner-unit.log").read_text(
    encoding="utf-8", errors="replace"
)
gpu_test_text = (root / "gpu-exactness.log").read_text(
    encoding="utf-8", errors="replace"
)
exact_tests_ok = (
    "checks passed (0 failed)" in planner_text
    and "q8 partner projection exactness OK (3 classes)" in gpu_test_text
)

local_raw = read_rows(root / "quality/local.tsv", "\t")
default_raw = read_rows(root / "quality/default.tsv", "\t")
local_rows = {row["id"]: row for row in local_raw}
default_rows = {row["id"]: row for row in default_raw}
if (len(local_raw) != 100 or len(default_raw) != 100 or
        len(local_rows) != len(local_raw) or
        len(default_rows) != len(default_raw)):
    fail("quality inputs must each contain exactly 100 unique case IDs")
if set(local_rows) != set(default_rows):
    fail("local/default quality case IDs differ")
ids = sorted(local_rows)
if len(ids) != 100:
    fail(f"production quality suite requires 100 cases, found {len(ids)}")

for case_id in ids:
    local = local_rows[case_id]
    candidate = default_rows[case_id]
    if local["target_tokens"] != candidate["target_tokens"]:
        fail(f"target-token count differs for {case_id}")
    for key in ("nll", "avg_nll"):
        if not math.isfinite(float(local[key])) or not math.isfinite(float(candidate[key])):
            fail(f"non-finite {key} for {case_id}")

local_quality = quality_summary(local_rows)
default_quality = quality_summary(default_rows)
local_pairs = [(float(local_rows[item]["nll"]), int(local_rows[item]["target_tokens"])) for item in ids]
default_pairs = [(float(default_rows[item]["nll"]), int(default_rows[item]["target_tokens"])) for item in ids]
quality_upper95 = bootstrap_upper_delta(local_pairs, default_pairs)
case_avg_deltas = [
    float(default_rows[item]["avg_nll"]) - float(local_rows[item]["avg_nll"])
    for item in ids
]
first_match_loss = int(local_quality["first_matches"] - default_quality["first_matches"])
lcp_loss = local_quality["avg_lcp"] - default_quality["avg_lcp"]
quality_ok = (
    quality_upper95 <= 0.002
    and max(case_avg_deltas) <= 0.05
    and first_match_loss <= 1
    and lcp_loss <= 0.1
)

runs = read_rows(root / "performance/runs.tsv", "\t")
samples: dict[tuple[int, str], dict[int, tuple[int, float]]] = {}
run_order: dict[int, dict[int, str]] = defaultdict(dict)
for run in runs:
    repeat = int(run["repeat"])
    variant = run["variant"]
    slot = int(run["slot"])
    key = (repeat, variant)
    if key in samples:
        fail(f"duplicate performance run: repeat={repeat} variant={variant}")
    if slot in run_order[repeat]:
        fail(f"duplicate performance slot: repeat={repeat} slot={slot}")
    run_order[repeat][slot] = variant
    rows = read_rows(Path(run["csv"]))
    contexts = [int(row["ctx_tokens"]) for row in rows]
    if len(rows) != 3 or len(set(contexts)) != 3:
        fail(f"repeat {repeat} variant {variant} must have three unique frontiers")
    for row in rows:
        tps = float(row["prefill_tps"])
        if not math.isfinite(tps) or tps <= 0.0:
            fail(f"repeat {repeat} variant {variant} has invalid prefill throughput")
    samples[key] = {
        int(row["ctx_tokens"]): (int(row["prefill_tokens"]), float(row["prefill_tps"]))
        for row in rows
    }
repeats = sorted({repeat for repeat, _ in samples})
if len(repeats) < 3:
    fail("production performance validation requires at least three repeats")
if repeats != list(range(1, len(repeats) + 1)):
    fail(f"performance repeats must be consecutive from 1, found {repeats}")
expected_run_keys = {
    (repeat, variant)
    for repeat in repeats
    for variant in ("local", "default")
}
if set(samples) != expected_run_keys:
    fail("performance runs must contain exactly one local/default pair per repeat")
expected_orders = {("local", "default"), ("default", "local")}
observed_orders: set[tuple[str, str]] = set()
slot_counts = {
    variant: {1: 0, 2: 0} for variant in ("local", "default")
}
for repeat in repeats:
    if set(run_order[repeat]) != {1, 2}:
        fail(f"repeat {repeat} lacks the exact two benchmark slots")
    order = (run_order[repeat][1], run_order[repeat][2])
    if order not in expected_orders:
        fail(f"repeat {repeat} has an invalid local/default order")
    observed_orders.add(order)
    for slot, variant in run_order[repeat].items():
        slot_counts[variant][slot] += 1
if observed_orders != expected_orders or any(
        abs(counts[1] - counts[2]) > 1 for counts in slot_counts.values()
):
    fail("local/default run order is not counterbalanced")

by_context: dict[int, list[dict[str, float]]] = defaultdict(list)
for repeat in repeats:
    local = samples.get((repeat, "local"))
    candidate = samples.get((repeat, "default"))
    if not local or not candidate or set(local) != set(candidate):
        fail(f"repeat {repeat} lacks matched local/default samples")
    for context in sorted(local):
        local_tokens, local_tps = local[context]
        candidate_tokens, candidate_tps = candidate[context]
        if local_tokens != candidate_tokens or local_tokens <= 0:
            fail(f"repeat {repeat} context {context} has mismatched prefill work")
        expected_tokens = 16384 if context in (16384, 32768) else 32768
        if local_tokens != expected_tokens:
            fail(f"repeat {repeat} context {context} has unexpected prefill work")
        local_seconds = local_tokens / local_tps
        candidate_seconds = candidate_tokens / candidate_tps
        by_context[context].append({
            "local_tps": local_tps,
            "candidate_tps": candidate_tps,
            "ratio": candidate_tps / local_tps,
            "normalized_saved_seconds":
                (local_seconds - candidate_seconds) / (local_tokens / 2048.0),
        })

expected_contexts = {16384, 32768, 65536}
if set(by_context) != expected_contexts:
    fail(f"expected 16K/32K/64K frontiers, found {sorted(by_context)}")

performance_rows: list[dict[str, float | int]] = []
performance_ok = True
near_threshold = False
for context in sorted(by_context):
    rows = by_context[context]
    median_ratio = statistics.median(row["ratio"] for row in rows)
    median_saved = statistics.median(row["normalized_saved_seconds"] for row in rows)
    minimum_saved = min(row["normalized_saved_seconds"] for row in rows)
    context_ok = median_ratio >= 1.05 and median_saved >= 0.70 and minimum_saved >= 0.60
    performance_ok = performance_ok and context_ok
    near_threshold = near_threshold or (
        median_ratio < 1.07 or median_saved < 0.80 or minimum_saved < 0.70
    )
    performance_rows.append({
        "context": context,
        "median_local_tps": statistics.median(row["local_tps"] for row in rows),
        "median_default_tps": statistics.median(row["candidate_tps"] for row in rows),
        "median_ratio": median_ratio,
        "minimum_ratio": min(row["ratio"] for row in rows),
        "median_saved_seconds_per_2k": median_saved,
        "minimum_saved_seconds_per_2k": minimum_saved,
        "samples": len(rows),
        "pass": int(context_ok),
    })

evidence = read_rows(root / "performance/class-evidence.csv")
expected_evidence_keys = {(repeat, "default") for repeat in repeats}
evidence_keys = [(int(row["repeat"]), row["variant"]) for row in evidence]
if (len(evidence_keys) != len(expected_evidence_keys) or
        set(evidence_keys) != expected_evidence_keys):
    fail("class evidence must contain exactly one default row per repeat")
evidence_ok = all(row["evidence_status"] == "ok" for row in evidence)

expected_frontiers = {
    f"frontier_{context:06d}.logits.json" for context in expected_contexts
}
logits = read_rows(root / "performance/logit-comparison.csv")
expected_logit_keys = {
    (repeat, "default", frontier)
    for repeat in repeats
    for frontier in expected_frontiers
}
logit_keys = [
    (int(row["repeat"]), row["variant"], row["frontier"])
    for row in logits
]
if (len(logit_keys) != len(expected_logit_keys) or
        set(logit_keys) != expected_logit_keys):
    fail("logit comparison lacks exact default x repeat x frontier coverage")
top1_ok = all(row["top1_equal"] == "1" for row in logits)

determinism = read_rows(root / "performance/logit-determinism.csv")
expected_determinism_keys = {
    (repeat, variant, frontier)
    for repeat in repeats[1:]
    for variant in ("local", "default")
    for frontier in expected_frontiers
}
determinism_keys = [
    (int(row["repeat"]), row["variant"], row["frontier"])
    for row in determinism
]
if (len(determinism_keys) != len(expected_determinism_keys) or
        set(determinism_keys) != expected_determinism_keys):
    fail("determinism evidence lacks exact variant x repeat x frontier coverage")
determinism_ok = all(row["exact"] == "1" for row in determinism)

binding_counts: list[int] = []
bindings_ok = True
binding_paths = sorted((root / "performance/runs").glob("*-r*.bindings.csv"))
expected_binding_names = {
    f"{variant}-r{repeat}.bindings.csv"
    for repeat in repeats
    for variant in ("local", "default")
}
if {path.name for path in binding_paths} != expected_binding_names:
    fail("binding evidence must contain exactly one local/default export per repeat")
for path in binding_paths:
    partner = [row for row in read_rows(path) if row["partner_offload"] == "1"]
    if path.name.startswith("local-"):
        if partner:
            bindings_ok = False
        continue
    if not partner or any(
        row["in_dim"] != "8192" or row["out_dim"] != "4096" for row in partner
    ):
        bindings_ok = False
    binding_counts.append(len(partner))
if not binding_counts or len(set(binding_counts)) != 1:
    bindings_ok = False

needs_more_repeats = near_threshold and len(repeats) < 5
accepted = all((quality_ok, quality_policy_ok, exact_tests_ok,
                performance_ok, evidence_ok, top1_ok,
                determinism_ok, bindings_ok, not needs_more_repeats))
payload = {
    "accepted": accepted,
    "quality": {
        "local_avg_nll": local_quality["avg_nll"],
        "default_avg_nll": default_quality["avg_nll"],
        "delta_nll_per_token": default_quality["avg_nll"] - local_quality["avg_nll"],
        "bootstrap_upper95_delta_nll_per_token": quality_upper95,
        "max_case_avg_nll_delta": max(case_avg_deltas),
        "first_match_loss": first_match_loss,
        "average_lcp_loss": lcp_loss,
        "production_policy_evidence": quality_policy_ok,
        "pass": quality_ok,
    },
    "performance": performance_rows,
    "evidence": {
        "class_pure": evidence_ok,
        "top1_equal": top1_ok,
        "repeat_deterministic": determinism_ok,
        "bindings_t256_only": bindings_ok,
        "partner_bindings_per_run": binding_counts,
        "planner_and_gpu_exactness_tests": exact_tests_ok,
        "run_order_counterbalanced": True,
    },
    "extend_to_five_repeats": needs_more_repeats,
}
(root / "acceptance.json").write_text(json.dumps(payload, indent=2) + "\n")

lines = [
    "# Production-default T256 validation",
    "",
    f"Overall: **{'PASS' if accepted else 'FAIL'}**",
    "",
    "## Production-path quality",
    "",
    f"- Cases: {len(ids)}",
    f"- Local/default average NLL: {local_quality['avg_nll']:.9f} / {default_quality['avg_nll']:.9f}",
    f"- Delta NLL/token: {default_quality['avg_nll'] - local_quality['avg_nll']:+.9f}",
    f"- Paired bootstrap 95% upper bound: {quality_upper95:+.9f} (limit +0.002)",
    f"- Maximum per-case average-NLL delta: {max(case_avg_deltas):+.9f} (limit +0.05)",
    f"- Lost first-token matches: {first_match_loss} (limit 1)",
    f"- Average greedy-LCP loss: {lcp_loss:+.3f} (limit 0.1)",
    f"- Production dispatch/audit/bindings: {'PASS' if quality_policy_ok else 'FAIL'}",
    "",
    "## Long-context prefill",
    "",
    "| Context | Local tok/s | Default tok/s | Ratio | Saved s/2K | Minimum saved s/2K | Result |",
    "|---:|---:|---:|---:|---:|---:|---|",
]
for row in performance_rows:
    lines.append(
        f"| {int(row['context']) // 1024}K | {float(row['median_local_tps']):.2f} | "
        f"{float(row['median_default_tps']):.2f} | {float(row['median_ratio']):.4f}× | "
        f"{float(row['median_saved_seconds_per_2k']):.3f} | "
        f"{float(row['minimum_saved_seconds_per_2k']):.3f} | "
        f"{'PASS' if int(row['pass']) else 'FAIL'} |"
    )
lines.extend((
    "",
    "## Evidence gates",
    "",
    f"- Class-pure T256 audit: {'PASS' if evidence_ok else 'FAIL'}",
    f"- T256-only exported partner bindings: {'PASS' if bindings_ok else 'FAIL'} ({binding_counts})",
    f"- Local/default top-1 equality: {'PASS' if top1_ok else 'FAIL'}",
    f"- Exact repeat determinism: {'PASS' if determinism_ok else 'FAIL'}",
    f"- Planner + three-class GPU exactness tests: {'PASS' if exact_tests_ok else 'FAIL'}",
    "- Local/default run order: PASS (counterbalanced)",
    f"- Extend to five repeats: {'YES' if needs_more_repeats else 'NO'}",
    "",
))
(root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
raise SystemExit(0 if accepted else 1)
