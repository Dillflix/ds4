#!/usr/bin/env python3
"""Summarize CUDA Q8->F16 cache decisions and select native-Q8 NCU targets."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("targets_tsv", type=Path)
    parser.add_argument(
        "--min-targets",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="minimum native-Q8 profiler targets required (default: 2)",
    )
    args = parser.parse_args()

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Q8 cache audit is empty")

    grouped = defaultdict(lambda: {"calls": 0, "weights": {}, "bytes": 0, "max_cache": 0})
    for row in rows:
        key = (row["module"], row["label"], row["physical_device"],
               row["result"], row["reason"])
        item = grouped[key]
        item["calls"] += 1
        weight_key = (row["physical_device"], row["weight_offset"])
        item["weights"][weight_key] = int(row["fp16_bytes"])
        item["bytes"] += int(row["fp16_bytes"])
        item["max_cache"] = max(item["max_cache"], int(row["cache_bytes_after"]))

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("module", "label", "physical_device", "result", "reason",
                         "calls", "unique_weights", "unique_fp16_bytes",
                         "call_requested_fp16_bytes",
                         "max_cache_bytes_after"))
        for key in sorted(grouped):
            item = grouped[key]
            writer.writerow((*key, item["calls"], len(item["weights"]),
                             sum(item["weights"].values()), item["bytes"],
                             item["max_cache"]))

    # NCU scopes currently surround the named dense projections below. Avoid
    # selecting fused q_b/output paths even if their cache attempt is native.
    attention_modules = {"attn_q_a", "attn_kv"}
    shared_modules = {"shared_gate", "shared_up", "shared_down"}
    native = []
    seen = set()
    for row in rows:
        if row["result"] != "native_q8" or not row["layer"]:
            continue
        module = row["module"]
        if module not in attention_modules | shared_modules:
            continue
        key = (module, row["layer"], row["token_offset"],
               row["physical_device"])
        if key in seen:
            continue
        seen.add(key)
        native.append(row)

    selected = []
    for kind, modules in (("attention", attention_modules), ("shared", shared_modules)):
        candidate = next((row for row in native if row["module"] in modules), None)
        if candidate:
            selected.append((kind, candidate))
    if len(selected) < 2:
        already = {(row["module"], row["layer"], row["token_offset"],
                    row["physical_device"])
                   for _, row in selected}
        for row in native:
            key = (row["module"], row["layer"], row["token_offset"],
                   row["physical_device"])
            if key in already:
                continue
            selected.append(("dense", row))
            already.add(key)
            if len(selected) == 2:
                break
    if len(selected) < args.min_targets:
        raise SystemExit(
            f"fewer than {args.min_targets} profiled native-Q8 projection "
            "targets were found"
        )

    with args.targets_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("kind", "module", "layer", "token_offset",
                         "physical_device", "label", "in_dim", "out_dim",
                         "reason"))
        for kind, row in selected[:2]:
            writer.writerow((kind, row["module"], row["layer"],
                             row["token_offset"], row["physical_device"], row["label"],
                             row["in_dim"], row["out_dim"], row["reason"]))

    runtime_rows = [row for row in rows if row["layer"]]
    result_counts = defaultdict(int)
    for row in runtime_rows:
        result_counts[row["result"]] += 1
    unique_weights = {}
    for row in runtime_rows:
        key = (row["physical_device"], row["weight_offset"])
        entry = unique_weights.setdefault(key, {
            "bytes": int(row["fp16_bytes"]), "cached": False, "native": False
        })
        entry["cached"] |= row["result"] in {"f16_fill", "f16_hit"}
        entry["native"] |= row["result"] == "native_q8"
    total_unique_bytes = sum(entry["bytes"] for entry in unique_weights.values())
    cached_unique_bytes = sum(entry["bytes"] for entry in unique_weights.values()
                              if entry["cached"] and not entry["native"])
    print(f"Q8 cache audit records: {len(rows)} total, {len(runtime_rows)} runtime")
    for result in sorted(result_counts):
        print(f"  {result}: {result_counts[result]}")
    print(f"Unique device-weight slices: {len(unique_weights)}")
    if total_unique_bytes:
        print("Byte-weighted F16 coverage: "
              f"{cached_unique_bytes}/{total_unique_bytes} "
              f"({100.0 * cached_unique_bytes / total_unique_bytes:.2f}%)")
    print("Native-Q8 NCU targets:")
    for kind, row in selected[:2]:
        print(f"  {kind}: {row['module']} layer {row['layer']} "
              f"pos {row['token_offset']} "
              f"device {row['physical_device']} ({row['in_dim']}x{row['out_dim']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
