#!/usr/bin/env python3
"""Validate the exact native-Q8 T256 partner experiment."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_LAYERS = list(range(15, 22))
PARTNER_DEVICES = {1, 2}
AUDIT_COLUMNS = {
    "label", "layer", "physical_device", "in_dim", "out_dim",
    "result", "reason",
}


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            fail(f"{path} has no header")
        return list(reader)


def manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result.setdefault(key, value)
    return result


def binding_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[key] for key in (
        "consumer_device", "weight_offset", "weight_bytes", "in_dim",
        "out_dim", "fp16_bytes", "label",
    ))


def binding_map(table: list[dict[str, str]], path: Path) -> dict[tuple[str, ...], str]:
    result: dict[tuple[str, ...], str] = {}
    for row in table:
        key = binding_key(row)
        if key in result:
            fail(f"{path} contains duplicate binding keys")
        result[key] = row["resident_device"]
    return result


def validate_bindings(local_path: Path, native_path: Path) -> int:
    local = rows(local_path)
    native = rows(native_path)
    if any(row["partner_offload"] == "1" for row in local):
        fail(f"{local_path} contains a partner binding")
    native_home = [row for row in native if row["partner_offload"] == "0"]
    native_partner = [row for row in native if row["partner_offload"] == "1"]
    if binding_map(local, local_path) != binding_map(native_home, native_path):
        fail(f"{native_path} changed the frozen home binding set")
    if len(native_partner) != len(EXPECTED_LAYERS):
        fail(f"{native_path} has {len(native_partner)} partner bindings, expected 7")
    layers: list[int] = []
    for row in native_partner:
        match = re.search(r"blk\.(\d+)\.", row["label"])
        if not match:
            fail(f"{native_path} has a partner binding without a layer label")
        layers.append(int(match.group(1)))
        if ((row["in_dim"], row["out_dim"]) != ("8192", "4096") or
                row["partner_arithmetic"] != "native-q8" or
                int(row["resident_device"]) not in PARTNER_DEVICES or
                row["resident_weight_bytes"] != row["weight_bytes"]):
            fail(f"{native_path} contains a non-native-Q8 T256 partner binding")
    if sorted(layers) != EXPECTED_LAYERS:
        fail(f"{native_path} partner layers are {sorted(layers)}, expected 15-21")
    return len(native_partner)


def validate_audit(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = AUDIT_COLUMNS - set(reader.fieldnames or ())
        if missing:
            fail(f"{path} lacks audit columns: {','.join(sorted(missing))}")
        table = list(reader)
    hits = [
        row for row in table
        if row["result"] == "native_q8_partner_hit" and
        row["reason"] == "exact_sm75_mma"
    ]
    if not hits:
        fail(f"{path} contains no exact native-Q8 partner execution")
    if any(
        (row["in_dim"], row["out_dim"]) != ("8192", "4096") or
        int(row["physical_device"]) not in PARTNER_DEVICES or
        int(row["layer"]) not in EXPECTED_LAYERS
        for row in hits
    ):
        fail(f"{path} contains wrong-shape, wrong-layer, or wrong-device native hits")
    seen = Counter(int(row["layer"]) for row in hits)
    if sorted(seen) != EXPECTED_LAYERS or len(set(seen.values())) != 1:
        fail(f"{path} does not exercise layers 15-21 equally: {dict(seen)}")
    if any(row["result"] in {"f16_partner_hit", "f32_partner_hit"} for row in table):
        fail(f"{path} mixed another partner arithmetic path into the native arm")
    return len(hits)


if len(sys.argv) != 2:
    fail("usage: summarize-q8-partner-native-exact.py OUTPUT_DIR")
root = Path(sys.argv[1]).resolve()
meta = manifest(root / "manifest.txt")
if (meta.get("gpu_devices") != "0,3,1,2" or
        meta.get("gpu_vram") != "auto" or
        meta.get("stage_split") != "22/21" or
        meta.get("quality_ctx") != "32769" or
        meta.get("contexts") != "16384,32768" or
        meta.get("t256_layers") != "15-21" or
        meta.get("partner_arithmetic") != "native-q8" or
        meta.get("home_plan") != "frozen"):
    fail("manifest does not describe the fixed exact native-Q8 experiment")

local_score = root / "quality/local.tsv"
native_score = root / "quality/native-q8.tsv"
local_quality = rows(local_score, "\t")
native_quality = rows(native_score, "\t")
if len(local_quality) != 100 or len(native_quality) != 100:
    fail("quality arms must each contain exactly 100 cases")
if local_score.read_bytes() != native_score.read_bytes():
    fail("native-Q8 production quality output is not byte-identical to local")
quality_bindings = validate_bindings(
    root / "quality/local.bindings.csv",
    root / "quality/native-q8.bindings.csv",
)
quality_hits = validate_audit(root / "quality/native-q8.q8-audit.csv")

run_table = rows(root / "runs.tsv", "\t")
samples: dict[tuple[int, str], dict[int, float]] = {}
orders: dict[int, dict[int, str]] = defaultdict(dict)
binding_counts: list[int] = []
audit_hits: list[int] = []
logits_exact = True
for row in run_table:
    repeat = int(row["repeat"])
    slot = int(row["slot"])
    variant = row["variant"]
    key = (repeat, variant)
    if key in samples or slot in orders[repeat]:
        fail("performance table contains a duplicate run or slot")
    orders[repeat][slot] = variant
    perf_rows = rows(Path(row["csv"]))
    if len(perf_rows) != 2:
        fail(f"repeat {repeat} {variant} does not contain two frontiers")
    values: dict[int, float] = {}
    for item in perf_rows:
        context = int(item["ctx_tokens"])
        tps = float(item["prefill_tps"])
        if context not in {16384, 32768} or not math.isfinite(tps) or tps <= 0:
            fail(f"repeat {repeat} {variant} contains invalid performance data")
        values[context] = tps
    if set(values) != {16384, 32768}:
        fail(f"repeat {repeat} {variant} lacks a fixed frontier")
    samples[key] = values

repeats = sorted({repeat for repeat, _ in samples})
if len(repeats) < 3 or repeats != list(range(1, len(repeats) + 1)):
    fail("performance evidence requires at least three consecutive repeats")
expected = {(repeat, variant) for repeat in repeats for variant in ("local", "native-q8")}
if set(samples) != expected:
    fail("performance evidence lacks one local/native pair per repeat")
for repeat in repeats:
    order = tuple(orders[repeat][slot] for slot in sorted(orders[repeat]))
    if set(orders[repeat]) != {1, 2} or order not in {
        ("local", "native-q8"), ("native-q8", "local"),
    }:
        fail(f"repeat {repeat} has an invalid A/B order")
    local_row = next(
        row for row in run_table
        if int(row["repeat"]) == repeat and row["variant"] == "local"
    )
    native_row = next(
        row for row in run_table
        if int(row["repeat"]) == repeat and row["variant"] == "native-q8"
    )
    binding_counts.append(validate_bindings(
        Path(local_row["bindings"]), Path(native_row["bindings"])
    ))
    audit_hits.append(validate_audit(Path(native_row["audit"])))
    local_logits = Path(local_row["logits"])
    native_logits = Path(native_row["logits"])
    local_files = {path.name: path for path in local_logits.glob("*.json")}
    native_files = {path.name: path for path in native_logits.glob("*.json")}
    if len(local_files) != 2 or set(local_files) != set(native_files):
        fail(f"repeat {repeat} lacks matched frontier logits")
    if any(local_files[name].read_bytes() != native_files[name].read_bytes()
           for name in local_files):
        logits_exact = False
if not logits_exact:
    fail("native-Q8 frontier logits are not byte-identical to local")

performance: list[dict[str, float | int]] = []
for context in (16384, 32768):
    ratios = [
        samples[(repeat, "native-q8")][context] /
        samples[(repeat, "local")][context]
        for repeat in repeats
    ]
    performance.append({
        "context": context,
        "local_median_tps": statistics.median(
            samples[(repeat, "local")][context] for repeat in repeats
        ),
        "native_median_tps": statistics.median(
            samples[(repeat, "native-q8")][context] for repeat in repeats
        ),
        "median_ratio": statistics.median(ratios),
        "minimum_ratio": min(ratios),
        "samples": len(ratios),
    })

payload = {
    "experiment_integrity": True,
    "quality_byte_exact": True,
    "frontier_logits_byte_exact": True,
    "quality_partner_bindings": quality_bindings,
    "quality_native_hits": quality_hits,
    "performance_partner_bindings": binding_counts,
    "performance_native_hits": audit_hits,
    "performance": performance,
}
(root / "native-exact.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
lines = [
    "# Native-Q8 partner exactness",
    "",
    "Experiment integrity: **PASS**",
    "",
    "- 100-case production quality output: byte-exact",
    "- 16K/32K frontier logits: byte-exact in every repeat",
    f"- Additive T256 bindings: {quality_bindings} (layers 15-21)",
    f"- Production quality native partner calls: {quality_hits}",
    "",
    "| Context | Local tok/s | Native-Q8 tok/s | Median ratio | Minimum ratio |",
    "|---:|---:|---:|---:|---:|",
]
for item in performance:
    lines.append(
        f"| {int(item['context']) // 1024}K | "
        f"{float(item['local_median_tps']):.2f} | "
        f"{float(item['native_median_tps']):.2f} | "
        f"{float(item['median_ratio']):.4f}× | "
        f"{float(item['minimum_ratio']):.4f}× |"
    )
lines.append("")
(root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
