#!/usr/bin/env python3
"""Validate and summarize deliberate SM75 T256 execution placement policies."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
import re


LEGACY_VARIANTS = ("native", "all-local", "balanced", "overflow", "all-partner")
STRICT_VARIANTS = ("native", "all-local", "balanced", "all-partner")
EXPECTED_BINDINGS = {
    "native": (0, 0),
    "all-local": (43, 0),
    "balanced": (22, 21),
    "all-partner": (0, 43),
}
EXPECTED_RESULTS = {
    "native": {"native_q8"},
    "all-local": {"f16_hit"},
    "balanced": {"f16_hit", "f16_partner_hit"},
    "all-partner": {"f16_partner_hit"},
}
OVERFLOW_ELIGIBLE_PARTNER_LAYERS = frozenset(range(15, 22))
MIB = 1024 * 1024
COMPLETE_DENSE_CANDIDATES = 344
NATIVE_T256_ONLY_LIVE_BINDINGS = 301
EXPECTED_DENSE_CLASSES = Counter({
    "attn_q_a": 43,
    "attn_q_b": 43,
    "attn_kv": 43,
    "attn_output_a": 43,
    "attn_output_b": 43,
    "shared_down": 43,
    "shared_gate_up": 86,
})
DENSE_FAMILY_NEEDLES = (
    ("attn_output_b", "attn_output_b"),
    ("attn_output_a", "attn_output_a"),
    ("attn_q_b", "attn_q_b"),
    ("attn_q_a", "attn_q_a"),
    ("attn_kv", "attn_kv"),
    ("ffn_down_shexp", "shared_down"),
    ("ffn_gate_shexp", "shared_gate"),
    ("ffn_up_shexp", "shared_up"),
)
EXPECTED_DENSE_FAMILIES = {
    "attn_q_a": ("attn_q_a", 4096, 1024),
    "attn_q_b": ("attn_q_b", 1024, 32768),
    "attn_kv": ("attn_kv", 4096, 512),
    "attn_output_a": ("attn_output_a", 4096, 8192),
    "attn_output_b": ("attn_output_b", 8192, 4096),
    "shared_down": ("ffn_down_shexp", 2048, 4096),
    "shared_gate": ("ffn_gate_shexp", 4096, 2048),
    "shared_up": ("ffn_up_shexp", 4096, 2048),
}
PLAN_COLUMNS = {
    "sequence", "label", "consumer_device", "fallback_device",
    "target_device", "placement_locked", "resident_device",
    "weight_offset", "weight_bytes", "in_dim", "out_dim", "fp16_bytes",
    "status",
}
BINDING_COLUMNS = {
    "consumer_device", "resident_device", "partner_offload",
    "weight_offset", "weight_bytes", "in_dim", "out_dim", "fp16_bytes",
    "resident_weight_bytes", "partner_arithmetic", "label",
    "allocation_id", "used_calls", "live",
}
ALLOCATION_COLUMNS = {
    "allocation_id", "storage_kind", "half_rounded", "physical_device",
    "weight_offset", "weight_bytes", "in_dim", "out_dim", "resident_bytes",
    "logical_aliases", "live_aliases", "used_calls", "dead_bytes",
    "usage_tracking",
}
MEMORY_COLUMNS = {
    "logical_tier", "physical_device", "free_bytes", "total_bytes",
    "q8_fp16_cached_bytes", "q8_fp16_reserve_bytes",
}


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def rows(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                fail(f"{path} has no header")
            return list(reader)
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")


def manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result.setdefault(key, value)
    return result


def require_columns(
    path: Path, table: list[dict[str, str]], required: set[str]
) -> None:
    fields = set(table[0]) if table else set()
    if not fields:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                fields = set(next(csv.reader(handle)))
        except (OSError, StopIteration) as exc:
            fail(f"cannot inspect header in {path}: {exc}")
    missing = required - fields
    if missing:
        fail(f"{path} lacks columns: {','.join(sorted(missing))}")


def integer(row: dict[str, str], key: str, context: str, minimum: int = 0) -> int:
    try:
        value = int(row[key])
    except (KeyError, ValueError) as exc:
        fail(f"{context} has invalid {key}: {row.get(key, '<missing>')}")
        raise AssertionError from exc
    if value < minimum:
        fail(f"{context} has {key}={value}, expected at least {minimum}")
    return value


def dense_class(row: dict[str, str]) -> str:
    label = row.get("label", "")
    for needle, family in DENSE_FAMILY_NEEDLES:
        if needle in label:
            return "shared_gate_up" if family in {"shared_gate", "shared_up"} else family
    fail(f"unknown production dense projection: {label or '<missing>'}")
    raise AssertionError


def dense_family(row: dict[str, str]) -> str:
    label = row.get("label", "")
    for needle, family in DENSE_FAMILY_NEEDLES:
        if needle in label:
            return family
    fail(f"unknown production dense projection: {label or '<missing>'}")
    raise AssertionError


def candidate_descriptor(row: dict[str, str]) -> tuple[str, ...]:
    return (
        dense_class(row), row["label"], row["consumer_device"],
        row["weight_offset"], row["weight_bytes"], row["in_dim"],
        row["out_dim"], row["fp16_bytes"],
    )


def validate_dense_identity(
    row: dict[str, str], family: str, layer: int, context: str
) -> None:
    suffix, expected_in, expected_out = EXPECTED_DENSE_FAMILIES[family]
    expected_label = f"tensor:blk.{layer}.{suffix}.weight"
    if row.get("label") != expected_label:
        fail(
            f"{context} has non-canonical label {row.get('label', '<missing>')}; "
            f"expected {expected_label}"
        )
    observed = (
        integer(row, "in_dim", context, 1),
        integer(row, "out_dim", context, 1),
        integer(row, "fp16_bytes", context, 1),
    )
    expected = (expected_in, expected_out, expected_in * expected_out * 2)
    if observed != expected:
        fail(f"{context} has shape/FP16 bytes {observed}, expected {expected}")


def encode_inventory(items: Counter[tuple[str, ...]]) -> dict[str, int]:
    return {
        ";".join((f"class={key[0]}", f"label={key[1]}",
                  f"consumer={key[2]}", f"offset={key[3]}",
                  f"weight_bytes={key[4]}", f"shape={key[5]}x{key[6]}",
                  f"fp16_bytes={key[7]}")): value
        for key, value in sorted(items.items())
    }


def is_t256(row: dict[str, str]) -> bool:
    return (
        row.get("in_dim") == "8192"
        and row.get("out_dim") == "4096"
        and "attn_output_b" in f"{row.get('module', '')} {row.get('label', '')}"
    )


def layer_of(row: dict[str, str]) -> int:
    value = row.get("layer", "")
    if value.isdigit():
        return int(value)
    match = re.search(r"blk\.(\d+)\.", row.get("label", ""))
    if not match:
        fail(f"cannot recover layer from {row.get('label', '<missing>')}")
    return int(match.group(1))


def devices_for_layer(layer: int) -> tuple[int, int]:
    return (0, 1) if layer <= 21 else (3, 2)


def partner_layer(variant: str, layer: int) -> bool:
    if variant == "balanced":
        return (layer & 1) != 0
    return variant == "all-partner"


def validate_complete_dense_cache(
    run: dict[str, str],
    binding_rows: list[dict[str, str]],
    allocation_rows: list[dict[str, str]],
    min_free_mib: int,
) -> dict[str, object]:
    variant = run["variant"]
    context = f"{variant} repeat {run['repeat']}"
    plan_value = run.get("plan", "")
    if not plan_value:
        fail(f"{context} lacks the complete-cache plan audit path")
    plan_path = Path(plan_value)
    plan_rows = rows(plan_path)
    binding_path = Path(run["bindings"])
    allocation_path = Path(run["allocations"])
    memory_value = run.get("memory", "")
    if not memory_value:
        fail(f"{context} lacks post-warmup CUDA memory evidence")
    memory_path = Path(memory_value)
    memory_rows = rows(memory_path)
    require_columns(plan_path, plan_rows, PLAN_COLUMNS)
    require_columns(binding_path, binding_rows, BINDING_COLUMNS)
    require_columns(allocation_path, allocation_rows, ALLOCATION_COLUMNS)
    require_columns(memory_path, memory_rows, MEMORY_COLUMNS)

    sequences: set[int] = set()
    plan_by_descriptor: dict[tuple[str, ...], dict[str, str]] = {}
    admitted_plan_by_descriptor: dict[tuple[str, ...], dict[str, str]] = {}
    class_counts: Counter[str] = Counter()
    family_layers: Counter[tuple[str, int]] = Counter()
    for index, row in enumerate(plan_rows):
        item_context = f"{context} plan row {index}"
        sequence = integer(row, "sequence", item_context)
        if sequence in sequences:
            fail(f"{context} plan contains duplicate sequence {sequence}")
        sequences.add(sequence)
        consumer = integer(row, "consumer_device", item_context)
        resident = integer(row, "resident_device", item_context, -1)
        fallback = integer(row, "fallback_device", item_context, -1)
        target = integer(row, "target_device", item_context)
        locked = integer(row, "placement_locked", item_context)
        if locked not in {0, 1}:
            fail(f"{item_context} has invalid placement_locked={locked}")
        in_dim = integer(row, "in_dim", item_context, 1)
        out_dim = integer(row, "out_dim", item_context, 1)
        fp16_bytes = integer(row, "fp16_bytes", item_context, 1)
        integer(row, "weight_offset", item_context)
        integer(row, "weight_bytes", item_context, 1)
        if fp16_bytes != in_dim * out_dim * 2:
            fail(f"{item_context} has inconsistent FP16 byte count")
        cls = dense_class(row)
        family = dense_family(row)
        layer = layer_of(row)
        if layer not in range(43) or consumer != devices_for_layer(layer)[0]:
            fail(f"{item_context} is not on its production home stage")
        validate_dense_identity(row, family, layer, item_context)
        home, peer = devices_for_layer(layer)
        status = row["status"]
        native_unadmitted = variant == "native" and family == "attn_output_b"
        if native_unadmitted:
            if (
                status != "unadmitted" or resident != -1 or target != home
                or fallback != -1 or locked != 0
            ):
                fail(
                    f"{item_context} is not the deliberate native-T256 "
                    "unadmitted candidate"
                )
        else:
            if status not in {"home", "partner"}:
                fail(
                    f"{context} dense FP16 plan is incomplete: "
                    f"{row['label']} status={status}"
                )
            if (status == "home") != (resident == consumer):
                fail(f"{item_context} has inconsistent status/resident device")
            if family != "attn_output_b":
                if (status, resident, target, fallback, locked) != (
                    "home", home, home, -1, 0
                ):
                    fail(f"{item_context} displaced a non-T256 projection")
            elif variant == "all-local":
                if (status, resident, target, fallback, locked) != (
                    "home", home, home, -1, 0
                ):
                    fail(f"{item_context} violates all-local T256 policy")
            elif variant == "all-partner":
                if (status, resident, target, fallback, locked) != (
                    "partner", peer, peer, peer, 1
                ):
                    fail(f"{item_context} violates all-partner T256 policy")
            elif variant == "balanced":
                expected = (
                    ("partner", peer, peer, peer, 1)
                    if layer & 1 else ("home", home, home, -1, 0)
                )
                if (status, resident, target, fallback, locked) != expected:
                    fail(f"{item_context} violates balanced T256 policy")
            elif variant == "overflow":
                expected_fallback = peer if layer in OVERFLOW_ELIGIBLE_PARTNER_LAYERS else -1
                if fallback != expected_fallback or target != resident or locked != 0:
                    fail(f"{item_context} violates overflow T256 policy")
                if status == "partner" and layer not in OVERFLOW_ELIGIBLE_PARTNER_LAYERS:
                    fail(f"{item_context} uses an ineligible overflow partner")
        descriptor = candidate_descriptor(row)
        if descriptor in plan_by_descriptor:
            fail(f"{context} duplicates candidate {row['label']}")
        plan_by_descriptor[descriptor] = row
        if not native_unadmitted:
            admitted_plan_by_descriptor[descriptor] = row
        class_counts[cls] += 1
        family_layers[(family, layer)] += 1
    if sequences != set(range(len(plan_rows))):
        fail(f"{context} plan sequence is not contiguous from zero")
    if (
        len(plan_rows) != COMPLETE_DENSE_CANDIDATES
        or class_counts != EXPECTED_DENSE_CLASSES
    ):
        fail(
            f"{context} has incomplete production candidate inventory: "
            f"total={len(plan_rows)}/{COMPLETE_DENSE_CANDIDATES} "
            f"classes={dict(sorted(class_counts.items()))}"
        )
    expected_family_layers = Counter({
        (family, layer): 1
        for _needle, family in DENSE_FAMILY_NEEDLES
        for layer in range(43)
    })
    if family_layers != expected_family_layers:
        fail(f"{context} does not contain each production projection/layer once")

    binding_by_descriptor: dict[tuple[str, ...], dict[str, str]] = {}
    allocation_ids: set[int] = set()
    binding_uses: dict[int, int] = {}
    live_inventory: Counter[tuple[str, ...]] = Counter()
    for index, binding in enumerate(binding_rows):
        item_context = f"{context} binding {index}"
        descriptor = candidate_descriptor(binding)
        if descriptor in binding_by_descriptor:
            fail(f"{context} duplicates live binding {binding['label']}")
        plan = admitted_plan_by_descriptor.get(descriptor)
        if plan is None:
            fail(f"{context} binding was not an active candidate: {binding['label']}")
        binding_by_descriptor[descriptor] = binding
        consumer = integer(binding, "consumer_device", item_context)
        resident = integer(binding, "resident_device", item_context)
        partner = integer(binding, "partner_offload", item_context)
        if partner not in {0, 1} or partner != int(consumer != resident):
            fail(f"{item_context} has inconsistent consumer/resident placement")
        if resident != int(plan["resident_device"]):
            fail(f"{item_context} does not use its admitted plan residence")
        if dense_class(binding) != "attn_output_b" and partner != 0:
            fail(f"{context} partner-offloaded non-T256 projection {binding['label']}")
        if binding["partner_arithmetic"] != "f16":
            fail(f"{item_context} does not use production F16 arithmetic")
        if binding["live"] != "1":
            fail(f"{item_context} is not live")
        used_calls = integer(binding, "used_calls", item_context, 1)
        fp16_bytes = integer(binding, "fp16_bytes", item_context, 1)
        if integer(binding, "resident_weight_bytes", item_context, 1) != fp16_bytes:
            fail(f"{item_context} does not reference one complete F16 weight")
        allocation_id = integer(binding, "allocation_id", item_context, 1)
        if allocation_id in allocation_ids:
            fail(
                f"{context} allocation {allocation_id} aliases multiple active "
                "production candidates"
            )
        allocation_ids.add(allocation_id)
        binding_uses[allocation_id] = used_calls

    plan_inventory = Counter(plan_by_descriptor.keys())
    admitted_plan_inventory = Counter(admitted_plan_by_descriptor.keys())
    binding_inventory = Counter(binding_by_descriptor.keys())
    if binding_inventory != admitted_plan_inventory:
        missing = admitted_plan_inventory - binding_inventory
        fail(
            f"{context} did not bind every active production candidate; "
            f"missing={len(missing)}"
        )

    allocation_by_id: dict[int, dict[str, str]] = {}
    for index, allocation in enumerate(allocation_rows):
        item_context = f"{context} allocation row {index}"
        allocation_id = integer(allocation, "allocation_id", item_context, 1)
        if allocation_id in allocation_by_id:
            fail(f"{context} duplicates allocation_id={allocation_id}")
        allocation_by_id[allocation_id] = allocation
        if allocation["storage_kind"] != "f16" or allocation["half_rounded"] != "0":
            fail(f"{item_context} is not an exact production F16 expansion")
        if allocation["usage_tracking"] != "1":
            fail(f"{item_context} lacks usage tracking")
        if (
            integer(allocation, "logical_aliases", item_context) != 1
            or integer(allocation, "live_aliases", item_context) != 1
            or integer(allocation, "used_calls", item_context, 1) <= 0
            or integer(allocation, "dead_bytes", item_context) != 0
        ):
            fail(f"{item_context} is dead, aliased, or contains dead bytes")
    if set(allocation_by_id) != allocation_ids:
        unreferenced = set(allocation_by_id) - allocation_ids
        missing = allocation_ids - set(allocation_by_id)
        fail(
            f"{context} allocation inventory is not one-to-one with bindings: "
            f"unreferenced={len(unreferenced)} missing={len(missing)}"
        )

    for descriptor, binding in binding_by_descriptor.items():
        allocation_id = int(binding["allocation_id"])
        allocation = allocation_by_id[allocation_id]
        expected = {
            "physical_device": binding["resident_device"],
            "weight_offset": binding["weight_offset"],
            "weight_bytes": binding["weight_bytes"],
            "in_dim": binding["in_dim"],
            "out_dim": binding["out_dim"],
            "resident_bytes": binding["resident_weight_bytes"],
            "used_calls": str(binding_uses[allocation_id]),
        }
        wrong = {
            key: (allocation.get(key), value)
            for key, value in expected.items()
            if allocation.get(key) != value
        }
        if wrong:
            fail(
                f"{context} allocation {allocation_id} does not match "
                f"binding {binding['label']}: {wrong}"
            )
        if dense_class(binding) != "attn_output_b":
            live_inventory[(
                *descriptor,
                binding["resident_device"], binding["partner_offload"],
                binding["partner_arithmetic"], allocation["storage_kind"],
                allocation["half_rounded"], allocation["resident_bytes"],
            )] += 1

    expected_tier_devices = {0: 0, 1: 3, 2: 1, 3: 2}
    allocation_bytes_by_device: Counter[int] = Counter()
    for allocation in allocation_rows:
        allocation_bytes_by_device[int(allocation["physical_device"])] += int(
            allocation["resident_bytes"]
        )
    memory_by_device: dict[int, dict[str, int]] = {}
    for index, memory in enumerate(memory_rows):
        item_context = f"{context} memory row {index}"
        tier = integer(memory, "logical_tier", item_context)
        device = integer(memory, "physical_device", item_context)
        if expected_tier_devices.get(tier) != device or device in memory_by_device:
            fail(f"{item_context} has wrong or duplicate logical/physical mapping")
        free_bytes = integer(memory, "free_bytes", item_context)
        total_bytes = integer(memory, "total_bytes", item_context, 1)
        cached_bytes = integer(memory, "q8_fp16_cached_bytes", item_context)
        reserve_bytes = integer(memory, "q8_fp16_reserve_bytes", item_context)
        if free_bytes > total_bytes:
            fail(f"{item_context} reports free bytes greater than total bytes")
        if cached_bytes != allocation_bytes_by_device[device]:
            fail(
                f"{item_context} cached bytes {cached_bytes} do not match "
                f"live allocations {allocation_bytes_by_device[device]}"
            )
        if free_bytes < min_free_mib * MIB:
            fail(
                f"{item_context} has only {free_bytes / MIB:.1f} MiB free; "
                f"requires at least {min_free_mib} MiB"
            )
        memory_by_device[device] = {
            "logical_tier": tier,
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "q8_fp16_cached_bytes": cached_bytes,
            "q8_fp16_reserve_bytes": reserve_bytes,
        }
    if set(memory_by_device) != set(expected_tier_devices.values()):
        fail(f"{context} memory evidence does not cover all four devices")

    return {
        "complete_dense_cache": True,
        "dense_candidates": len(plan_rows),
        "live_dense_bindings": len(binding_rows),
        "dense_class_inventory": dict(sorted(class_counts.items())),
        "candidate_inventory": encode_inventory(plan_inventory),
        "non_t256_live_inventory": {
            "|".join(key): value for key, value in sorted(live_inventory.items())
        },
        "dead_or_unreferenced_allocations": 0,
        "post_warmup_memory_by_device": memory_by_device,
        "minimum_free_mib": min_free_mib,
    }


def validate_run(
    run: dict[str, str], complete_dense_cache: bool = False,
    min_free_mib: int = 0,
) -> tuple[dict[int, float], dict[str, object]]:
    variant = run["variant"]
    binding_rows = rows(Path(run["bindings"]))
    allocation_rows = rows(Path(run["allocations"]))
    complete_evidence = (
        validate_complete_dense_cache(
            run, binding_rows, allocation_rows, min_free_mib
        )
        if complete_dense_cache else {}
    )
    t256_bindings = [row for row in binding_rows if is_t256(row)]
    local = [row for row in t256_bindings if row["partner_offload"] == "0"]
    partner = [row for row in t256_bindings if row["partner_offload"] == "1"]
    binding_partner_layers = sorted(layer_of(row) for row in partner)
    binding_partner_layer_set = set(binding_partner_layers)
    if variant == "overflow":
        if len(t256_bindings) != 43 or len(local) + len(partner) != 43:
            fail(
                f"overflow repeat {run['repeat']} has T256 bindings "
                f"{len(local)}+{len(partner)}, expected 43 total live bindings"
            )
        if len(partner) > len(OVERFLOW_ELIGIBLE_PARTNER_LAYERS):
            fail(
                f"overflow repeat {run['repeat']} has {len(partner)} partner "
                "bindings, expected at most 7"
            )
        ineligible = sorted(
            binding_partner_layer_set - OVERFLOW_ELIGIBLE_PARTNER_LAYERS
        )
        if ineligible:
            fail(
                f"overflow repeat {run['repeat']} has ineligible partner "
                f"layers {ineligible}; eligible layers are 15-21"
            )
        if len(binding_partner_layers) != len(binding_partner_layer_set):
            fail(f"overflow repeat {run['repeat']} duplicates a partner layer")
    else:
        expected_local, expected_partner = EXPECTED_BINDINGS[variant]
        if (len(local), len(partner)) != (expected_local, expected_partner):
            fail(
                f"{variant} repeat {run['repeat']} has T256 bindings "
                f"{len(local)}+{len(partner)}, expected "
                f"{expected_local}+{expected_partner}"
            )
    if any(row["partner_arithmetic"] != "f16" for row in t256_bindings):
        fail(f"{variant} repeat {run['repeat']} contains non-F16 T256 bindings")
    observed_bindings = Counter(
        (
            layer_of(row), int(row["consumer_device"]),
            int(row["resident_device"]), int(row["partner_offload"]),
        )
        for row in t256_bindings
    )
    expected_bindings: Counter[tuple[int, int, int, int]] = Counter()
    if variant != "native":
        for layer in range(43):
            home, peer = devices_for_layer(layer)
            uses_partner = (
                layer in binding_partner_layer_set
                if variant == "overflow"
                else partner_layer(variant, layer)
            )
            if uses_partner:
                expected_bindings[(layer, home, peer, 1)] += 1
            else:
                expected_bindings[(layer, home, home, 0)] += 1
    if observed_bindings != expected_bindings:
        fail(
            f"{variant} repeat {run['repeat']} has the expected aggregate "
            "binding count but the wrong per-layer consumer/resident mapping"
        )

    allocation_by_id: dict[int, dict[str, str]] = {}
    for row in allocation_rows:
        allocation_id = int(row["allocation_id"])
        if allocation_id <= 0 or allocation_id in allocation_by_id:
            fail(
                f"{variant} repeat {run['repeat']} has an invalid or duplicate "
                f"physical allocation id {allocation_id}"
            )
        allocation_by_id[allocation_id] = row
        if row.get("usage_tracking") != "1":
            fail(f"{variant} repeat {run['repeat']} lacks warm-up liveness tracking")
        resident_bytes = int(row["resident_bytes"])
        used_calls = int(row["used_calls"])
        expected_dead = resident_bytes if used_calls == 0 else 0
        if int(row["dead_bytes"]) != expected_dead:
            fail(
                f"{variant} repeat {run['repeat']} allocation {allocation_id} "
                "has inconsistent dead-byte accounting"
            )

    t256_allocation_ids: set[int] = set()
    for row in t256_bindings:
        allocation_id = int(row.get("allocation_id", "0"))
        allocation = allocation_by_id.get(allocation_id)
        if not allocation:
            fail(
                f"{variant} repeat {run['repeat']} has a T256 binding without "
                "a physical allocation"
            )
        t256_allocation_ids.add(allocation_id)
        if row.get("live") != "1" or int(row.get("used_calls", "0")) <= 0:
            fail(
                f"{variant} repeat {run['repeat']} has a dead T256 logical binding"
            )
        if (
            allocation["storage_kind"] != "f16"
            or allocation["in_dim"] != "8192"
            or allocation["out_dim"] != "4096"
            or int(allocation["logical_aliases"]) != 1
            or int(allocation["live_aliases"]) != 1
            or int(allocation["used_calls"]) <= 0
            or int(allocation["dead_bytes"]) != 0
        ):
            fail(
                f"{variant} repeat {run['repeat']} T256 allocation "
                f"{allocation_id} is not one live physical weight"
            )
    expected_physical_t256 = 0 if variant == "native" else 43
    physical_t256_rows = [
        row for row in allocation_rows
        if row["storage_kind"] == "f16"
        and row["in_dim"] == "8192"
        and row["out_dim"] == "4096"
    ]
    if (
        len(t256_allocation_ids) != expected_physical_t256
        or len(physical_t256_rows) != expected_physical_t256
    ):
        fail(
            f"{variant} repeat {run['repeat']} has "
            f"{len(physical_t256_rows)} physical T256 allocations and "
            f"{len(t256_allocation_ids)} referenced, expected "
            f"{expected_physical_t256}"
        )

    audit_rows = [row for row in rows(Path(run["audit"])) if is_t256(row)]
    if not audit_rows:
        fail(f"{variant} repeat {run['repeat']} has no T256 runtime evidence")
    result_counts = Counter(row["result"] for row in audit_rows)
    expected_results = (
        {"f16_hit"} | ({"f16_partner_hit"} if partner else set())
        if variant == "overflow"
        else EXPECTED_RESULTS[variant]
    )
    if set(result_counts) != expected_results:
        fail(
            f"{variant} repeat {run['repeat']} has wrong T256 paths: "
            f"{dict(result_counts)}"
        )
    layer_counts = Counter(layer_of(row) for row in audit_rows)
    if set(layer_counts) != set(range(43)) or len(set(layer_counts.values())) != 1:
        fail(f"{variant} repeat {run['repeat']} has unequal layer coverage")
    partner_layers = sorted({
        layer_of(row) for row in audit_rows
        if row["result"] == "f16_partner_hit"
    })
    expected_partner_layers = (
        binding_partner_layers
        if variant == "overflow"
        else {
            "native": [],
            "all-local": [],
            "balanced": list(range(1, 43, 2)),
            "all-partner": list(range(43)),
        }[variant]
    )
    if partner_layers != expected_partner_layers:
        fail(
            f"{variant} repeat {run['repeat']} partner layers are "
            f"{partner_layers}, expected {expected_partner_layers}"
        )
    for row in audit_rows:
        layer = layer_of(row)
        home, peer = devices_for_layer(layer)
        uses_partner = (
            layer in binding_partner_layer_set
            if variant == "overflow"
            else partner_layer(variant, layer)
        )
        if variant == "native":
            expected_result, expected_reason, expected_device = (
                "native_q8", "disabled_t256_by_env", home
            )
        elif uses_partner:
            expected_result, expected_reason, expected_device = (
                "f16_partner_hit", "nvlink_offload", peer
            )
        else:
            expected_result, expected_reason, expected_device = (
                "f16_hit", "resident", home
            )
        observed = (
            row["result"], row.get("reason", ""),
            int(row.get("physical_device", "-1")),
        )
        if observed != (expected_result, expected_reason, expected_device):
            fail(
                f"{variant} repeat {run['repeat']} layer {layer} executed "
                f"{observed}, expected "
                f"{(expected_result, expected_reason, expected_device)}"
            )

    log = Path(run["log"]).read_text(encoding="utf-8", errors="replace")
    for marker in (
        "CUDA EP forced pipeline split 22/21",
        f"t256-placement={'overflow' if variant == 'native' else variant}",
    ):
        if marker not in log:
            fail(f"{variant} repeat {run['repeat']} lacks marker: {marker}")
    if variant != "native" and "T256-output_b=43/43" not in log:
        fail(f"{variant} repeat {run['repeat']} did not admit 43/43 T256 bindings")

    perf = rows(Path(run["csv"]))
    values: dict[int, float] = {}
    for row in perf:
        context = int(row["ctx_tokens"])
        tps = float(row["prefill_tps"])
        if context in values or not math.isfinite(tps) or tps <= 0:
            fail(f"{variant} repeat {run['repeat']} contains invalid throughput")
        values[context] = tps
    if not values:
        fail(f"{variant} repeat {run['repeat']} has no performance rows")

    # Record all binding shapes so policy-induced displacement is explicit.
    binding_shapes = Counter(
        (row.get("in_dim", ""), row.get("out_dim", ""), row["partner_offload"])
        for row in binding_rows
    )
    evidence = {
        "repeat": int(run["repeat"]),
        "variant": variant,
        "t256_local_bindings": len(local),
        "t256_partner_bindings": len(partner),
        "t256_runtime_results": dict(sorted(result_counts.items())),
        "t256_calls_per_layer": next(iter(layer_counts.values())),
        "partner_layers": partner_layers,
        "total_bindings": len(binding_rows),
        "physical_allocations": len(allocation_rows),
        "physical_resident_bytes": sum(
            int(row["resident_bytes"]) for row in allocation_rows
        ),
        "physical_dead_bytes": sum(
            int(row["dead_bytes"]) for row in allocation_rows
        ),
        "t256_physical_allocations": len(t256_allocation_ids),
        "binding_shapes": {
            f"{key[0]}x{key[1]}:{'partner' if key[2] == '1' else 'local'}": value
            for key, value in sorted(binding_shapes.items())
        },
    }
    evidence.update(complete_evidence)
    return values, evidence


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: summarize-q8-t256-placement.py OUTPUT_DIR")
    root = Path(sys.argv[1]).resolve()
    meta = manifest(root / "manifest.txt")
    complete_value = meta.get("require_complete_dense_cache", "0")
    if complete_value not in {"0", "1"}:
        fail("manifest has invalid require_complete_dense_cache")
    complete_dense_cache = complete_value == "1"
    variants = STRICT_VARIANTS if complete_dense_cache else LEGACY_VARIANTS
    if (
        meta.get("gpu_devices") != "0,3,1,2"
        or meta.get("gpu_vram") != "auto"
        or meta.get("stage_split") != "22/21"
        or meta.get("variants") != ",".join(variants)
        or meta.get("overflow_partner_eligibility") != "15-21"
    ):
        fail("manifest does not describe the fixed T256 placement experiment")
    if complete_dense_cache and (
        meta.get("ctx_alloc") != "262273"
        or meta.get("native_control") != "native-t256-only"
        or meta.get("prefill_attention_head_split") != "off"
        or meta.get("dense_cache_scope") != "complete-production-active"
        or meta.get("dense_cache_plan_candidates") != "344"
        or meta.get("dense_cache_native_live_bindings") != "301"
        or not meta.get("dense_cache_min_free_mib", "").isdigit()
        or int(meta["dense_cache_min_free_mib"]) < 512
        or meta.get("dense_cache_inventory")
        != "q_a:43,q_b:43,kv:43,output_a:43,output_b:43,shared_down:43,gate_up:86"
    ):
        fail("manifest does not describe the complete 256K dense-cache experiment")
    min_free_mib = (
        int(meta["dense_cache_min_free_mib"])
        if complete_dense_cache else 0
    )

    run_rows = rows(root / "runs.tsv", "\t")
    samples: dict[tuple[int, str], dict[int, float]] = {}
    slots: dict[int, dict[int, str]] = defaultdict(dict)
    evidence: list[dict[str, object]] = []
    for run in run_rows:
        repeat = int(run["repeat"])
        slot = int(run["slot"])
        variant = run["variant"]
        if variant not in variants:
            fail(f"unknown variant in run table: {variant}")
        key = (repeat, variant)
        if key in samples or slot in slots[repeat]:
            fail("run table contains a duplicate variant or slot")
        slots[repeat][slot] = variant
        samples[key], run_evidence = validate_run(
            run, complete_dense_cache, min_free_mib
        )
        evidence.append(run_evidence)

    repeats = sorted({repeat for repeat, _variant in samples})
    if not repeats or repeats != list(range(1, len(repeats) + 1)):
        fail("repeats must be consecutive starting at one")
    expected = {(repeat, variant) for repeat in repeats for variant in variants}
    if set(samples) != expected:
        fail("run table lacks one result for every variant/repeat")
    contexts = sorted(set.intersection(*(set(value) for value in samples.values())))
    if any(set(value) != set(contexts) for value in samples.values()):
        fail("performance contexts differ between runs")
    for repeat in repeats:
        order = [slots[repeat][slot] for slot in sorted(slots[repeat])]
        expected_order = [
            variants[(slot + repeat - 1) % len(variants)]
            for slot in range(len(variants))
        ]
        if order != expected_order:
            fail(f"repeat {repeat} order is {order}, expected {expected_order}")

    complete_inventory: dict[str, int] | None = None
    complete_non_t256_inventory: dict[str, int] | None = None
    if complete_dense_cache:
        complete_inventory = evidence[0]["candidate_inventory"]  # type: ignore[assignment]
        if any(
            item["candidate_inventory"] != complete_inventory
            for item in evidence
        ):
            fail("placement arms do not have identical complete dense plan inventories")
        complete_non_t256_inventory = evidence[0]["non_t256_live_inventory"]  # type: ignore[assignment]
        if any(
            item["non_t256_live_inventory"] != complete_non_t256_inventory
            for item in evidence
        ):
            fail(
                "T256 placement arms changed the live non-T256 binding/allocation "
                "inventory"
            )

    summary: dict[str, dict[int, dict[str, float]]] = {}
    for variant in variants:
        summary[variant] = {}
        for context in contexts:
            values = [samples[(repeat, variant)][context] for repeat in repeats]
            median = statistics.median(values)
            item = {
                "median_tps": median,
                "min_tps": min(values),
                "max_tps": max(values),
                "vs_native": median / statistics.median(
                    samples[(repeat, "native")][context] for repeat in repeats
                ),
            }
            if "overflow" in variants:
                item["vs_overflow"] = median / statistics.median(
                    samples[(repeat, "overflow")][context] for repeat in repeats
                )
            summary[variant][context] = item

    overflow_observed = [
        {
            "repeat": item["repeat"],
            "local_bindings": item["t256_local_bindings"],
            "partner_bindings": item["t256_partner_bindings"],
            "partner_layers": item["partner_layers"],
        }
        for item in evidence
        if item["variant"] == "overflow"
    ]
    overflow_signatures = {
        (
            item["local_bindings"], item["partner_bindings"],
            tuple(item["partner_layers"]),
        )
        for item in overflow_observed
    }
    overflow_placement_stable = (
        len(overflow_signatures) == 1 if "overflow" in variants else None
    )
    high_confidence = len(repeats) >= len(variants) and {
        tuple(slots[repeat][slot] for slot in sorted(slots[repeat]))
        for repeat in repeats[:len(variants)]
    } == {
        tuple(variants[offset:] + variants[:offset])
        for offset in range(len(variants))
    }
    payload = {
        "experiment_integrity": True,
        "native_control": "native-t256-only",
        "require_complete_dense_cache": complete_dense_cache,
        "high_confidence_counterbalanced": high_confidence,
        "repeats": len(repeats),
        "contexts": contexts,
        "performance": summary,
        "run_evidence": evidence,
    }
    if "overflow" in variants:
        payload["overflow_placement_stable"] = overflow_placement_stable
        payload["overflow_observed"] = overflow_observed
    if complete_dense_cache:
        payload["complete_dense_cache"] = {
            "ctx_alloc": 262273,
            "plan_candidates_per_arm": COMPLETE_DENSE_CANDIDATES,
            "native_t256_only_live_bindings": NATIVE_T256_ONLY_LIVE_BINDINGS,
            "native_intentionally_unadmitted_t256": 43,
            "minimum_post_warmup_free_mib": min_free_mib,
            "class_inventory": dict(sorted(EXPECTED_DENSE_CLASSES.items())),
            "candidate_descriptor_inventory": complete_inventory,
            "non_t256_live_descriptor_inventory": complete_non_t256_inventory,
            "all_required_candidates_admitted": True,
            "all_bindings_and_allocations_live": True,
            "dead_or_unreferenced_allocations": 0,
            "plan_inventory_exact_across_arms": True,
            "native_non_t256_inventory_exact": True,
        }
    (root / "t256-placement.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# T256 execution-placement comparison",
        "",
        "Experiment integrity: **PASS**",
        "",
        f"Counterbalanced confidence: **{'PASS' if high_confidence else 'SCREEN ONLY'}** "
        f"({len(repeats)} repeats).",
        "",
        "Binding inventory and active runtime paths were validated independently for every run.",
    ]
    if "overflow" in variants:
        lines.extend((
            "",
            "Natural-overflow placement stability: "
            f"**{'PASS' if overflow_placement_stable else 'FAIL'}**.",
            "",
            "Observed natural-overflow placement (local/partner bindings): "
            + "; ".join(
                f"r{item['repeat']}={item['local_bindings']}/{item['partner_bindings']} "
                f"layers={item['partner_layers']}"
                for item in overflow_observed
            )
            + ".",
            "",
            "| Context | Native T256 only | All local | 22 local + 21 partner | Overflow | All partner | Best |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ))
    else:
        lines.extend((
            "",
            "| Context | Native T256 only | All local | 22 local + 21 partner | All partner | Best |",
            "|---:|---:|---:|---:|---:|---|",
        ))
    for context in contexts:
        medians = {
            variant: summary[variant][context]["median_tps"]
            for variant in variants
        }
        best = max(medians, key=medians.get)
        if "overflow" in variants:
            lines.append(
                f"| {context} | {medians['native']:.2f} | {medians['all-local']:.2f} | "
                f"{medians['balanced']:.2f} | {medians['overflow']:.2f} | "
                f"{medians['all-partner']:.2f} | {best} |"
            )
        else:
            lines.append(
                f"| {context} | {medians['native']:.2f} | {medians['all-local']:.2f} | "
                f"{medians['balanced']:.2f} | {medians['all-partner']:.2f} | {best} |"
            )
    if complete_dense_cache:
        lines.extend((
            "",
            "Complete dense-cache integrity: **PASS**. Every arm exported the same "
            "344-row production plan. All 344 candidates were admitted, used, and "
            "live in each FP16 placement arm; the native-T256-only control deliberately "
            "left its 43 output-B rows unadmitted while preserving all 301 non-T256 "
            "entries. No arm displaced a required entry or retained dead expanded-weight "
            "bytes. Per-device memory was not equalized, so these results combine "
            "execution placement with its memory-pressure effect.",
            "",
        ))
    else:
        lines.extend((
            "",
            "`all-local` is allowed to displace lower-ranked cache entries in order to make all "
            "43 active T256 weights local. The exported binding and allocation inventories "
            "record that resource tradeoff; this is an end-to-end cache-policy comparison, "
            "not an isolated kernel test.",
            "",
        ))
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
