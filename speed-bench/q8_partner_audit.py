#!/usr/bin/env python3
"""Classify and validate bounded Q8 partner-offload audit evidence."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = {
    "module",
    "label",
    "physical_device",
    "in_dim",
    "out_dim",
    "result",
    "reason",
}


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


def class_evidence_valid(variant: str, counts: Counter[str]) -> bool:
    total = sum(counts.values())
    if variant == "local":
        return total == 0
    if variant == "t32":
        return counts["t32"] > 0 and total == counts["t32"]
    if variant in ("t256", "default"):
        return counts["t256"] > 0 and total == counts["t256"]
    if variant == "shared_down":
        return counts["shared_down"] > 0 and total == counts["shared_down"]
    if variant == "legacy":
        return (
            counts["t32"] > 0
            and counts["t256"] > 0
            and total == counts["t32"] + counts["t256"]
        )
    return False


def collect(
    rows: Iterable[dict[str, str]], partner_devices: tuple[int, int]
) -> tuple[Counter[str], Counter[tuple[str, int]], int]:
    classes: Counter[str] = Counter()
    by_partner: Counter[tuple[str, int]] = Counter()
    unexpected = 0
    for row in rows:
        if not (
            row.get("result") == "f16_partner_hit"
            and row.get("reason") == "nvlink_offload"
        ):
            continue
        name = classify(row)
        classes[name] += 1
        try:
            device = int(row.get("physical_device", ""))
        except (TypeError, ValueError):
            unexpected += 1
            continue
        if device == partner_devices[0]:
            by_partner[(name, 0)] += 1
        elif device == partner_devices[1]:
            by_partner[(name, 1)] += 1
        else:
            unexpected += 1
    return classes, by_partner, unexpected


def format_evidence(
    classes: Counter[str],
    by_partner: Counter[tuple[str, int]],
    unexpected: int,
) -> str:
    total = sum(classes.values())
    return (
        f"total={total} "
        f"t32={classes['t32']}({by_partner[('t32', 0)]}/"
        f"{by_partner[('t32', 1)]}) "
        f"t256={classes['t256']}({by_partner[('t256', 0)]}/"
        f"{by_partner[('t256', 1)]}) "
        f"shared_down={classes['shared_down']}("
        f"{by_partner[('shared_down', 0)]}/"
        f"{by_partner[('shared_down', 1)]}) "
        f"other={classes['other']} unexpected_device={unexpected}"
    )


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "usage: q8_partner_audit.py VARIANT PARTNER0 PARTNER1 AUDIT.csv",
            file=sys.stderr,
        )
        return 2
    variant, partner0_text, partner1_text, audit_text = sys.argv[1:]
    try:
        partner_devices = (int(partner0_text), int(partner1_text))
    except ValueError:
        print("error: partner device indices must be integers", file=sys.stderr)
        return 2
    audit = Path(audit_text)
    try:
        with audit.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                print(
                    f"error: {audit} lacks columns: {','.join(sorted(missing))}",
                    file=sys.stderr,
                )
                return 2
            classes, by_partner, unexpected = collect(reader, partner_devices)
    except OSError as exc:
        print(f"error: cannot read {audit}: {exc}", file=sys.stderr)
        return 2
    print(f"{variant}: {format_evidence(classes, by_partner, unexpected)}")
    return 0 if class_evidence_valid(variant, classes) and unexpected == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
