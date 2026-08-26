#!/usr/bin/env python3
"""Validate frozen-home additive Q8 partner quality experiments."""

from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

from q8_partner_audit import REQUIRED_COLUMNS, class_evidence_valid, collect


VARIANTS = ("t256", "t32")
SHAPES = {
    "t256": (8192, 4096),
    "t32": (1024, 32768),
}


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key and key not in values:
                values[key] = value
    return values


def quality_summary(rows: dict[str, dict[str, str]]) -> dict[str, float]:
    tokens = sum(int(row["target_tokens"]) for row in rows.values())
    if not rows or tokens <= 0:
        fail("quality table is empty")
    return {
        "avg_nll": sum(float(row["nll"]) for row in rows.values()) / tokens,
        "first_matches": float(sum(int(row["first_match"]) for row in rows.values())),
        "avg_lcp": statistics.fmean(float(row["greedy_lcp"]) for row in rows.values()),
    }


def bootstrap_upper_delta(
    baseline: list[tuple[float, int]], candidate: list[tuple[float, int]],
    seed: int, draws: int = 10000,
) -> float:
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(draws):
        delta = 0.0
        tokens = 0
        for _ in range(len(baseline)):
            index = rng.randrange(len(baseline))
            delta += candidate[index][0] - baseline[index][0]
            tokens += baseline[index][1]
        values.append(delta / tokens)
    values.sort()
    return values[math.ceil(0.95 * len(values)) - 1]


def binding_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[key] for key in (
        "consumer_device", "weight_offset", "weight_bytes",
        "in_dim", "out_dim", "fp16_bytes", "label",
    ))


def binding_map(rows: list[dict[str, str]], path: Path) -> dict[tuple[str, ...], str]:
    result: dict[tuple[str, ...], str] = {}
    for row in rows:
        key = binding_key(row)
        if key in result:
            fail(f"{path} contains a duplicate binding key")
        result[key] = row["resident_device"]
    return result


def validate_additive_bindings(
    root: Path, variant: str,
) -> dict[str, object]:
    local_path = root / "quality/local.bindings.csv"
    candidate_path = root / f"quality/{variant}.bindings.csv"
    local = read_rows(local_path)
    candidate = read_rows(candidate_path)
    if any(row["partner_offload"] == "1" for row in local):
        fail("local control unexpectedly contains partner bindings")
    candidate_home = [row for row in candidate if row["partner_offload"] == "0"]
    candidate_partner = [row for row in candidate if row["partner_offload"] == "1"]
    if binding_map(local, local_path) != binding_map(candidate_home, candidate_path):
        fail(f"{variant} changed the primary/home binding set")
    if not candidate_partner:
        fail(f"{variant} has no additive partner binding")
    local_keys = {binding_key(row) for row in local}
    if any(binding_key(row) in local_keys for row in candidate_partner):
        fail(f"{variant} moved a home binding instead of adding a miss")
    in_dim, out_dim = SHAPES[variant]
    if any(
        int(row["in_dim"]) != in_dim or int(row["out_dim"]) != out_dim
        for row in candidate_partner
    ):
        fail(f"{variant} contains a partner binding from another class")
    layers: list[int] = []
    for row in candidate_partner:
        match = re.search(r"blk\.(\d+)\.", row["label"])
        if not match:
            fail(f"{variant} partner binding lacks a layer label")
        layers.append(int(match.group(1)))
    if len(layers) != len(set(layers)):
        fail(f"{variant} contains duplicate partner layers")
    return {
        "home_bindings": len(local),
        "home_bindings_identical": True,
        "additive_partner_bindings": len(candidate_partner),
        "partner_layers": sorted(layers),
    }


if len(sys.argv) != 2:
    fail("usage: summarize-q8-partner-quality-isolation.py OUTPUT_DIR")
root = Path(sys.argv[1]).resolve()
manifest = read_manifest(root / "manifest.txt")
if (manifest.get("gpu_devices") != "0,3,1,2" or
        manifest.get("gpu_vram") != "auto" or
        manifest.get("stage_split") != "22/21" or
        manifest.get("quality_ctx") != "32769" or
        manifest.get("home_plan") != "frozen-for-candidates" or
        manifest.get("variants") != "local,t256,t32"):
    fail("manifest does not describe the fixed quality-isolation experiment")

partner_devices = (1, 2)
logs: dict[str, str] = {}
for variant in ("local", *VARIANTS):
    logs[variant] = (root / f"quality/{variant}.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if ("score_official: runtime_path=production" not in logs[variant] or
            "CUDA EP forced pipeline split 22/21" not in logs[variant]):
        fail(f"{variant} did not use the required production path")
    audit_path = root / f"quality/{variant}.q8-audit.csv"
    with audit_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            fail(f"{audit_path} lacks audit columns: {','.join(sorted(missing))}")
        classes, _, unexpected = collect(reader, partner_devices)
    if not class_evidence_valid(variant, classes) or unexpected:
        fail(f"{variant} partner audit is not class-pure")
    if variant == "local":
        if "CUDA q8 fp16 partner summary:" in logs[variant]:
            fail("local control executed a partner projection")
    else:
        if (f"partner-classes={variant}" not in logs[variant] or
                "home-order=frozen" not in logs[variant] or
                "CUDA q8 fp16 partner summary:" not in logs[variant]):
            fail(f"{variant} did not execute the frozen-home partner policy")

local_raw = read_rows(root / "quality/local.tsv", "\t")
local_rows = {row["id"]: row for row in local_raw}
if len(local_raw) != 100 or len(local_rows) != 100:
    fail("local quality input must contain exactly 100 unique cases")
ids = sorted(local_rows)
local_summary = quality_summary(local_rows)
local_pairs = [
    (float(local_rows[item]["nll"]), int(local_rows[item]["target_tokens"]))
    for item in ids
]

results: dict[str, dict[str, object]] = {}
for variant_index, variant in enumerate(VARIANTS):
    raw = read_rows(root / f"quality/{variant}.tsv", "\t")
    rows = {row["id"]: row for row in raw}
    if len(raw) != 100 or len(rows) != 100 or set(rows) != set(local_rows):
        fail(f"{variant} quality cases do not match the local control")
    for case_id in ids:
        if rows[case_id]["target_tokens"] != local_rows[case_id]["target_tokens"]:
            fail(f"{variant} target-token count differs for {case_id}")
        for key in ("nll", "avg_nll"):
            if not math.isfinite(float(rows[case_id][key])):
                fail(f"{variant} has non-finite {key} for {case_id}")
    summary = quality_summary(rows)
    pairs = [
        (float(rows[item]["nll"]), int(rows[item]["target_tokens"]))
        for item in ids
    ]
    case_deltas = [
        float(rows[item]["avg_nll"]) - float(local_rows[item]["avg_nll"])
        for item in ids
    ]
    first_loss = int(local_summary["first_matches"] - summary["first_matches"])
    lcp_loss = local_summary["avg_lcp"] - summary["avg_lcp"]
    upper95 = bootstrap_upper_delta(
        local_pairs, pairs, 0x75256 + variant_index
    )
    quality_pass = (
        upper95 <= 0.002 and max(case_deltas) <= 0.05
        and first_loss <= 1 and lcp_loss <= 0.1
    )
    results[variant] = {
        **validate_additive_bindings(root, variant),
        "local_avg_nll": local_summary["avg_nll"],
        "candidate_avg_nll": summary["avg_nll"],
        "delta_nll_per_token": summary["avg_nll"] - local_summary["avg_nll"],
        "bootstrap_upper95_delta_nll_per_token": upper95,
        "max_case_avg_nll_delta": max(case_deltas),
        "first_match_loss": first_loss,
        "average_lcp_loss": lcp_loss,
        "quality_pass": quality_pass,
    }

payload = {
    "experiment_integrity": True,
    "scope": "quality-only frozen-home class isolation",
    "variants": results,
}
(root / "quality-isolation.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)

lines = [
    "# Frozen-home Q8 partner quality isolation",
    "",
    "Experiment integrity: **PASS**",
    "",
    "The candidate home binding set is byte-for-byte identical to the local control; "
    "only previously unadmitted tensors were added on the partner.",
    "",
    "| Variant | Added partner layers | Delta NLL/token | Bootstrap upper 95% | "
    "Max case delta | First loss | LCP loss | Quality gate |",
    "|---|---|---:|---:|---:|---:|---:|---|",
]
for variant in VARIANTS:
    row = results[variant]
    layer_text = ",".join(str(item) for item in row["partner_layers"])
    lines.append(
        f"| {variant} | {layer_text} | {float(row['delta_nll_per_token']):+.9f} | "
        f"{float(row['bootstrap_upper95_delta_nll_per_token']):+.9f} | "
        f"{float(row['max_case_avg_nll_delta']):+.9f} | "
        f"{int(row['first_match_loss'])} | {float(row['average_lcp_loss']):+.3f} | "
        f"{'PASS' if row['quality_pass'] else 'FAIL'} |"
    )
lines.append("")
(root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
