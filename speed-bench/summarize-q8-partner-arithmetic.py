#!/usr/bin/env python3
"""Validate and summarize production Q8 partner arithmetic decomposition."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path


ARMS = (
    "local",
    "f16",
    "w16-x16-sgemm",
    "w16-x32-sgemm",
    "w32-x32-sgemm",
    "w32-xq8-sgemm",
)
CONTRASTS = (
    ("f16", "w16-x16-sgemm", "Tensor GEMM product/accumulation"),
    ("w16-x16-sgemm", "w16-x32-sgemm", "activation FP16 rounding"),
    ("w16-x32-sgemm", "w32-x32-sgemm", "weight FP16 rounding"),
    ("w32-x32-sgemm", "w32-xq8-sgemm", "activation block-Q8 quantization"),
    ("w32-xq8-sgemm", "local", "native INT32-dot/reduction structure"),
)
SCORE_COLUMNS = {
    "id", "target_tokens", "nll", "avg_nll", "first_match",
    "greedy_lcp", "first_target_id", "first_greedy_id",
    "first_target_margin", "first_greedy_margin",
}


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
        if reader.fieldnames is None:
            fail(f"{path} has no header")
        return rows


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values.setdefault(key, value)
    return values


def binding_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[key] for key in (
        "consumer_device", "weight_offset", "weight_bytes",
        "in_dim", "out_dim", "fp16_bytes", "label",
    ))


def binding_map(rows: list[dict[str, str]]) -> dict[tuple[str, ...], str]:
    result: dict[tuple[str, ...], str] = {}
    for row in rows:
        key = binding_key(row)
        if key in result:
            fail("binding table contains a duplicate key")
        result[key] = row["resident_device"]
    return result


def parse_layers(value: str) -> list[int]:
    result: set[int] = set()
    for item in value.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", item)
        if not match:
            fail("manifest has an invalid t256_layers selector")
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first > last or last >= 43:
            fail("manifest has an out-of-range t256_layers selector")
        result.update(range(first, last + 1))
    return sorted(result)


if len(sys.argv) != 2:
    fail("usage: summarize-q8-partner-arithmetic.py OUTPUT_DIR")
root = Path(sys.argv[1]).resolve()
manifest = read_manifest(root / "manifest.txt")
if tuple(manifest.get("variants", "").split(",")) != ARMS:
    fail("manifest does not declare the complete ordered arithmetic matrix")
if (manifest.get("gpu_devices") != "0,3,1,2" or
        manifest.get("gpu_vram") != "auto" or
        manifest.get("stage_split") != "22/21" or
        manifest.get("quality_ctx") != "32769" or
        manifest.get("home_plan") != "frozen-for-all-arms"):
    fail("manifest does not describe the fixed production experiment")
requested_layers = parse_layers(manifest.get("t256_layers", ""))

local_bindings = read_rows(root / "quality/local.bindings.csv")
if any(row["partner_offload"] == "1" for row in local_bindings):
    fail("local control contains a partner binding")
local_map = binding_map(local_bindings)
reference_partner_keys: set[tuple[str, ...]] | None = None
binding_layers: list[int] | None = None

for arm in ARMS[1:]:
    path = root / f"quality/{arm}.bindings.csv"
    rows = read_rows(path)
    required = {"partner_arithmetic", "resident_weight_bytes"}
    if rows and not required.issubset(rows[0]):
        fail(f"{path} lacks arithmetic binding evidence")
    home = [row for row in rows if row["partner_offload"] == "0"]
    partner = [row for row in rows if row["partner_offload"] == "1"]
    if binding_map(home) != local_map:
        fail(f"{arm} changed the frozen home binding set")
    if not partner:
        fail(f"{arm} has no additive partner bindings")
    if any(row["partner_arithmetic"] != arm for row in partner):
        fail(f"{arm} binding arithmetic does not match its arm")
    if any((row["in_dim"], row["out_dim"]) != ("8192", "4096")
           for row in partner):
        fail(f"{arm} contains a non-T256 partner binding")
    keys = {binding_key(row) for row in partner}
    if reference_partner_keys is None:
        reference_partner_keys = keys
    elif keys != reference_partner_keys:
        fail(f"{arm} does not use the same additive tensor set")
    layers: list[int] = []
    for row in partner:
        match = re.search(r"blk\.(\d+)\.", row["label"])
        if not match:
            fail(f"{arm} partner binding lacks a layer label")
        layers.append(int(match.group(1)))
    layers.sort()
    if binding_layers is None:
        binding_layers = layers
    elif layers != binding_layers:
        fail(f"{arm} partner layers differ from the other arms")

    audit = read_rows(root / f"quality/{arm}.q8-audit.csv")
    hits = []
    for row in audit:
        if arm == "f16":
            hit = (row.get("result") == "f16_partner_hit" and
                   row.get("reason") == "nvlink_offload")
        else:
            hit = (row.get("result") == "f32_partner_hit" and
                   row.get("reason") == arm)
        if hit:
            hits.append(row)
    if not hits:
        fail(f"{arm} audit has no matching partner execution")
    if any((row.get("in_dim"), row.get("out_dim")) != ("8192", "4096") or
           row.get("physical_device") not in {"1", "2"} for row in hits):
        fail(f"{arm} audit contains wrong-class or wrong-device execution")

if binding_layers != requested_layers:
    fail(
        f"admitted partner layers {binding_layers} do not match requested "
        f"layers {requested_layers}"
    )

scores: dict[str, dict[str, dict[str, str]]] = {}
ids: list[str] | None = None
for arm in ARMS:
    path = root / f"quality/{arm}.tsv"
    rows = read_rows(path, "\t")
    if rows and not SCORE_COLUMNS.issubset(rows[0]):
        fail(f"{path} lacks first-token margin evidence")
    table = {row["id"]: row for row in rows}
    if len(table) != len(rows) or not rows:
        fail(f"{arm} score table is empty or has duplicate cases")
    if ids is None:
        ids = sorted(table)
    elif sorted(table) != ids:
        fail(f"{arm} cases differ from local")
    for row in rows:
        for field in ("nll", "avg_nll", "first_target_margin",
                      "first_greedy_margin"):
            if not math.isfinite(float(row[field])):
                fail(f"{arm}/{row['id']} has non-finite {field}")
    scores[arm] = table

assert ids is not None
arm_summary: dict[str, dict[str, float | int]] = {}
for arm in ARMS:
    rows = scores[arm]
    tokens = sum(int(rows[item]["target_tokens"]) for item in ids)
    arm_summary[arm] = {
        "nll_per_token": sum(float(rows[item]["nll"]) for item in ids) / tokens,
        "first_matches": sum(int(rows[item]["first_match"]) for item in ids),
        "average_lcp": statistics.fmean(
            float(rows[item]["greedy_lcp"]) for item in ids
        ),
    }

contrasts: list[dict[str, object]] = []
for left, right, factor in CONTRASTS:
    per_case = [
        float(scores[right][item]["avg_nll"]) -
        float(scores[left][item]["avg_nll"])
        for item in ids
    ]
    changed = [
        item for item in ids
        if scores[left][item]["first_greedy_id"] !=
           scores[right][item]["first_greedy_id"]
    ]
    contrasts.append({
        "left": left,
        "right": right,
        "factor": factor,
        "mean_case_avg_nll_delta": statistics.fmean(per_case),
        "max_abs_case_avg_nll_delta": max(abs(value) for value in per_case),
        "first_greedy_changes": changed,
    })

cases: dict[str, dict[str, object]] = {}
for item in ids:
    cases[item] = {
        arm: {
            "first_target_id": int(scores[arm][item]["first_target_id"]),
            "first_greedy_id": int(scores[arm][item]["first_greedy_id"]),
            "first_match": bool(int(scores[arm][item]["first_match"])),
            "target_margin": float(scores[arm][item]["first_target_margin"]),
            "greedy_margin": float(scores[arm][item]["first_greedy_margin"]),
            "avg_nll": float(scores[arm][item]["avg_nll"]),
        }
        for arm in ARMS
    }

payload = {
    "experiment_integrity": True,
    "scope": "production T256 arithmetic decomposition",
    "partner_layers": binding_layers,
    "arm_summary": arm_summary,
    "contrasts": contrasts,
    "cases": cases,
}
(root / "arithmetic-isolation.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)

lines = [
    "# Q8 partner arithmetic isolation",
    "",
    "Experiment integrity: **PASS**",
    "",
    "Every candidate used the identical frozen home binding set and the same "
    "additive T256 layer set.",
    "",
    "| Arm | NLL/token | First matches | Average LCP |",
    "|---|---:|---:|---:|",
]
for arm in ARMS:
    row = arm_summary[arm]
    lines.append(
        f"| {arm} | {float(row['nll_per_token']):.9f} | "
        f"{int(row['first_matches'])} | {float(row['average_lcp']):.3f} |"
    )
lines.extend([
    "",
    "| Controlled contrast | Mean case NLL delta | Max absolute delta | "
    "First-token changes |",
    "|---|---:|---:|---|",
])
for row in contrasts:
    changed = ",".join(row["first_greedy_changes"]) or "none"
    lines.append(
        f"| {row['left']} -> {row['right']}: {row['factor']} | "
        f"{float(row['mean_case_avg_nll_delta']):+.9f} | "
        f"{float(row['max_abs_case_avg_nll_delta']):.9f} | {changed} |"
    )
lines.extend([
    "",
    "| Case | " + " | ".join(ARMS) + " |",
    "|---|" + "---:|" * len(ARMS),
])
for item in ids:
    values = []
    for arm in ARMS:
        row = cases[item][arm]
        values.append(
            f"{row['first_greedy_id']} ({float(row['target_margin']):+.6f})"
        )
    lines.append(f"| {item} | " + " | ".join(values) + " |")
lines.append("")
(root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
