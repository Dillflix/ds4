#!/usr/bin/env python3
"""Validate and summarize fixed-22/21 stage-aware dense-F16 placement."""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


LAYER = re.compile(r"(?:^|:)blk\.(\d+)\.")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def benefit(label: str) -> int:
    if ".attn_output_b.weight" in label or ".attn_q_b.weight" in label:
        return 19000
    if ".attn_output_a.weight" in label or ".ffn_down_shexp.weight" in label:
        return 800
    if ".ffn_gate_shexp.weight" in label or ".ffn_up_shexp.weight" in label:
        return 200
    return 100


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            f"usage: {sys.argv[0]} PLAN_CSV BINDINGS_CSV GPU_DEVICES OUT_CSV"
        )
    plan_path, bindings_path, devices_text, out_path = (
        Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    )
    devices = [int(value) for value in devices_text.split(",")]
    if len(devices) != 4 or len(set(devices)) != 4:
        raise SystemExit("GPU_DEVICES must contain four unique physical IDs")
    plan = rows(plan_path)
    bindings = rows(bindings_path)
    if len(plan) != 344:
        raise SystemExit(f"stage-aware plan has {len(plan)} rows, expected 344")
    if len(bindings) != 344:
        raise SystemExit(f"binding inventory has {len(bindings)} rows, expected 344")

    binding_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in bindings:
        key = (row["label"], row["weight_offset"], row["in_dim"], row["out_dim"])
        if key in binding_by_key:
            raise SystemExit(f"duplicate binding key: {key}")
        binding_by_key[key] = row

    aggregate: dict[tuple[int, str, int], dict[str, int]] = defaultdict(
        lambda: {"projections": 0, "fp16_bytes": 0, "benefit_units": 0}
    )
    observed: set[tuple[str, str, str, str]] = set()
    for row in plan:
        match = LAYER.search(row["label"])
        if not match:
            raise SystemExit(f"cannot extract layer from {row['label']}")
        layer = int(match.group(1))
        if not 0 <= layer < 43:
            raise SystemExit(f"invalid layer {layer}: {row['label']}")
        stage = 0 if layer < 22 else 1
        home, partner = devices[stage], devices[stage + 2]
        consumer = int(row["consumer_device"])
        fallback = int(row["fallback_device"])
        target = int(row["target_device"])
        resident = int(row["resident_device"])
        if consumer != home:
            raise SystemExit(
                f"layer {layer} consumer {consumer} is not stage-{stage} home {home}"
            )
        if fallback not in (-1, partner):
            raise SystemExit(
                f"layer {layer} fallback {fallback} crosses stage-{stage} pair"
            )
        if target not in (home, partner) or resident != target:
            raise SystemExit(
                f"layer {layer} target/resident {target}/{resident} is outside "
                f"stage-{stage} pair {home}<->{partner}"
            )
        if row["status"] not in ("home", "partner"):
            raise SystemExit(f"unadmitted stage-aware row: {row['label']}")

        key = (row["label"], row["weight_offset"], row["in_dim"], row["out_dim"])
        binding = binding_by_key.get(key)
        if binding is None:
            raise SystemExit(f"plan row has no binding: {row['label']}")
        observed.add(key)
        if int(binding["consumer_device"]) != consumer or \
           int(binding["resident_device"]) != resident:
            raise SystemExit(f"binding placement differs from plan: {row['label']}")
        if int(binding["live"]) != 1 or int(binding["used_calls"]) <= 0:
            raise SystemExit(f"dense binding was not exercised: {row['label']}")

        role = "home" if resident == home else "partner"
        cell = aggregate[(stage, role, resident)]
        cell["projections"] += 1
        cell["fp16_bytes"] += int(row["fp16_bytes"])
        cell["benefit_units"] += benefit(row["label"])

    if observed != set(binding_by_key):
        raise SystemExit("binding inventory contains rows outside the 344-row plan")
    for stage in (0, 1):
        if not any(key[0] == stage and key[1] == "partner" for key in aggregate):
            raise SystemExit(f"stage {stage} assigned no dense work to its partner")

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stage", "role", "physical_device", "projections",
                "fp16_bytes", "benefit_units",
            ),
        )
        writer.writeheader()
        for (stage, role, device), values in sorted(aggregate.items()):
            writer.writerow(
                {"stage": stage, "role": role, "physical_device": device, **values}
            )
    print(
        "validated stage-aware dense-F16 placement: 344/344 live bindings, "
        "fixed 22/21 pair confinement, partner work on both stages"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
