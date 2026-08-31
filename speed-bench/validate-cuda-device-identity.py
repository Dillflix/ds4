#!/usr/bin/env python3
"""Fail-closed CUDA, NVML/nvidia-smi, NVLink, and runtime identity validation."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable


BUS_RE = re.compile(
    r"^(?P<domain>[0-9a-f]{4}|[0-9a-f]{8}):"
    r"(?P<bus>[0-9a-f]{2}):(?P<device>[0-9a-f]{2})\."
    r"(?P<function>[0-7])$",
    re.IGNORECASE,
)
UUID_RE = re.compile(
    r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
GPU_TOKEN_RE = re.compile(r"^GPU(?P<index>[0-9]+)$")
NVLINK_RE = re.compile(r"^NV[1-9][0-9]*$")
ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TOPOLOGY_LABELS = frozenset(
    {"X", "SYS", "NODE", "PHB", "PXB", "PIX", "C2C", "SOC", "N/A"}
)

INVENTORY_FIELDS = {
    "cuda_ordinal": ("cuda_ordinal", "pci_bus_id", "uuid"),
    "nvidia_index": ("nvidia_index", "pci_bus_id", "uuid"),
}
EXPECTED_FIELDS = (
    "logical_tier",
    "logical_pair",
    "logical_role",
    "cuda_ordinal",
    "pci_bus_id",
    "uuid",
)
SELECTED_FIELDS = (
    "logical_tier",
    "logical_pair",
    "logical_role",
    "cuda_ordinal",
    "nvidia_index",
    "pci_bus_id",
    "uuid",
    "nvlink_partner_cuda_ordinal",
    "nvlink_partner_nvidia_index",
    "nvlink_link",
)
NORMALIZED_TOPOLOGY_FIELDS = ("nvidia_index", "peer_nvidia_index", "link")
RUNTIME_FIELDS = (
    "record_type",
    "line_number",
    "logical_tier",
    "cuda_ordinal",
    "pci_bus_id",
    "uuid",
    "property_query_status",
    "pci_query_status",
)

RUNTIME_INVENTORY_PREFIX = "ds4: CUDA ordinal inventory"
RUNTIME_SELECTED_PREFIX = "ds4: CUDA selected device identity"
RUNTIME_INVENTORY_RE = re.compile(
    r"^ds4: CUDA ordinal inventory ordinal=(?P<cuda_ordinal>[0-9]+) "
    r"pci_bus_id=(?P<pci_bus_id>\S+) uuid=(?P<uuid>\S+) "
    r"query_status=(?P<property_query_status>\S+)/(?P<pci_query_status>\S+)$"
)
RUNTIME_SELECTED_RE = re.compile(
    r"^ds4: CUDA selected device identity logical_tier=(?P<logical_tier>[0-9]+) "
    r"cuda_ordinal=(?P<cuda_ordinal>[0-9]+) pci_bus_id=(?P<pci_bus_id>\S+) "
    r"uuid=(?P<uuid>\S+) "
    r"query_status=(?P<property_query_status>\S+)/(?P<pci_query_status>\S+)$"
)


class ValidationError(ValueError):
    pass


class Topology(dict[tuple[int, int], str]):
    """A complete nvidia-smi topology matrix in the nvidia/NVML namespace."""

    def __init__(self, indices: Iterable[int], matrix: dict[tuple[int, int], str]):
        super().__init__(matrix)
        self.indices = tuple(indices)


def normalize_bus_id(value: str) -> str:
    value = value.strip()
    match = BUS_RE.fullmatch(value)
    if not match:
        raise ValidationError(f"invalid PCI bus ID: {value!r}")
    domain = int(match.group("domain"), 16)
    bus = int(match.group("bus"), 16)
    device = int(match.group("device"), 16)
    function = int(match.group("function"), 16)
    return f"{domain:08x}:{bus:02x}:{device:02x}.{function:x}"


def normalize_uuid(value: str) -> str:
    value = value.strip()
    if not UUID_RE.fullmatch(value):
        raise ValidationError(f"invalid GPU UUID: {value!r}")
    return "GPU-" + value[4:].lower()


def parse_index(value: str, label: str) -> int:
    value = value.strip()
    if not value.isdecimal():
        raise ValidationError(f"invalid {label}: {value!r}")
    return int(value, 10)


def _open_dict_reader(path: Path, expected_fields: tuple[str, ...]):
    """Return an open handle and strict DictReader; caller closes the handle."""
    try:
        handle = path.open(newline="", encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc
    try:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != list(expected_fields):
            raise ValidationError(
                f"{path}: expected CSV header {','.join(expected_fields)!r}, "
                f"got {reader.fieldnames!r}"
            )
    except ValidationError:
        handle.close()
        raise
    except (csv.Error, UnicodeError) as exc:
        handle.close()
        raise ValidationError(f"cannot parse {path}: {exc}") from exc
    return handle, reader


def _validate_csv_row(path: Path, line_number: int, row: dict) -> None:
    if None in row or any(value is None or isinstance(value, list) for value in row.values()):
        raise ValidationError(f"{path}:{line_number}: malformed CSV row")


def read_inventory(
    path: Path,
    index_column: str,
    expected_columns: tuple[str, str, str] | None = None,
) -> dict[int, tuple[str, str]]:
    expected_columns = expected_columns or INVENTORY_FIELDS[index_column]
    handle, reader = _open_dict_reader(path, expected_columns)
    result: dict[int, tuple[str, str]] = {}
    seen_bus: set[str] = set()
    seen_uuid: set[str] = set()
    try:
        for line_number, row in enumerate(reader, start=2):
            _validate_csv_row(path, line_number, row)
            index = parse_index(str(row[index_column]), index_column)
            bus_id = normalize_bus_id(str(row["pci_bus_id"]))
            uuid = normalize_uuid(str(row["uuid"]))
            if index in result:
                raise ValidationError(
                    f"{path}:{line_number}: duplicate {index_column} {index}"
                )
            if bus_id in seen_bus:
                raise ValidationError(
                    f"{path}:{line_number}: duplicate PCI bus ID {bus_id}"
                )
            if uuid in seen_uuid:
                raise ValidationError(f"{path}:{line_number}: duplicate GPU UUID {uuid}")
            result[index] = (bus_id, uuid)
            seen_bus.add(bus_id)
            seen_uuid.add(uuid)
    except (csv.Error, UnicodeError) as exc:
        raise ValidationError(f"cannot parse {path}: {exc}") from exc
    finally:
        handle.close()
    if not result:
        raise ValidationError(f"{path}: inventory is empty")
    return result


def parse_selected(value: str) -> list[int]:
    fields = value.split(",")
    if not fields or any(not field.strip().isdecimal() for field in fields):
        raise ValidationError(f"invalid selected CUDA ordinal list: {value!r}")
    selected = [int(field.strip(), 10) for field in fields]
    if len(selected) == 0 or len(selected) % 2 != 0:
        raise ValidationError("selected CUDA ordinal count must be nonzero and even")
    if len(set(selected)) != len(selected):
        raise ValidationError("selected CUDA ordinals must be unique")
    return selected


def read_expected_selected(path: Path) -> list[dict[str, str | int]]:
    handle, reader = _open_dict_reader(path, EXPECTED_FIELDS)
    rows: list[dict[str, str | int]] = []
    seen_tiers: set[int] = set()
    seen_ordinals: set[int] = set()
    seen_bus: set[str] = set()
    seen_uuid: set[str] = set()
    try:
        for line_number, row in enumerate(reader, start=2):
            _validate_csv_row(path, line_number, row)
            tier = parse_index(str(row["logical_tier"]), "logical_tier")
            pair = parse_index(str(row["logical_pair"]), "logical_pair")
            role = str(row["logical_role"]).strip()
            ordinal = parse_index(str(row["cuda_ordinal"]), "cuda_ordinal")
            bus_id = normalize_bus_id(str(row["pci_bus_id"]))
            uuid = normalize_uuid(str(row["uuid"]))
            if role not in {"home", "partner"}:
                raise ValidationError(
                    f"{path}:{line_number}: invalid logical_role {role!r}"
                )
            if tier in seen_tiers:
                raise ValidationError(f"{path}:{line_number}: duplicate logical_tier {tier}")
            if ordinal in seen_ordinals:
                raise ValidationError(f"{path}:{line_number}: duplicate cuda_ordinal {ordinal}")
            if bus_id in seen_bus:
                raise ValidationError(f"{path}:{line_number}: duplicate selected PCI bus ID")
            if uuid in seen_uuid:
                raise ValidationError(f"{path}:{line_number}: duplicate selected GPU UUID")
            rows.append(
                {
                    "logical_tier": tier,
                    "logical_pair": pair,
                    "logical_role": role,
                    "cuda_ordinal": ordinal,
                    "pci_bus_id": bus_id,
                    "uuid": uuid,
                }
            )
            seen_tiers.add(tier)
            seen_ordinals.add(ordinal)
            seen_bus.add(bus_id)
            seen_uuid.add(uuid)
    except (csv.Error, UnicodeError) as exc:
        raise ValidationError(f"cannot parse {path}: {exc}") from exc
    finally:
        handle.close()
    if not rows or len(rows) % 2 != 0:
        raise ValidationError(
            f"{path}: expected selected identity must have a nonzero even row count"
        )
    if [int(row["logical_tier"]) for row in rows] != list(range(len(rows))):
        raise ValidationError(f"{path}: logical tiers must be ordered and contiguous from zero")
    half = len(rows) // 2
    for tier, row in enumerate(rows):
        expected_pair = tier if tier < half else tier - half
        expected_role = "home" if tier < half else "partner"
        if row["logical_pair"] != expected_pair or row["logical_role"] != expected_role:
            raise ValidationError(
                f"{path}: logical tier {tier} must be pair {expected_pair} role {expected_role}"
            )
    return rows


def _valid_topology_label(value: str) -> bool:
    return value in TOPOLOGY_LABELS or NVLINK_RE.fullmatch(value) is not None


def read_topology(path: Path) -> Topology:
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc

    header: list[int] | None = None
    header_line = 0
    for line_number, raw_line in enumerate(lines, start=1):
        line = ANSI_CSI_RE.sub("", raw_line)
        tokens = line.split()
        if not tokens or GPU_TOKEN_RE.fullmatch(tokens[0]) is None:
            continue
        indices: list[int] = []
        for token in tokens:
            match = GPU_TOKEN_RE.fullmatch(token)
            if match is None:
                break
            indices.append(int(match.group("index"), 10))
        if len(indices) >= 2:
            header = indices
            header_line = line_number
            break
    if not header:
        raise ValidationError(f"{path}: nvidia-smi topology GPU header not found")
    if len(set(header)) != len(header):
        raise ValidationError(f"{path}: duplicate GPU columns in topology header")

    matrix: dict[tuple[int, int], str] = {}
    seen_rows: set[int] = set()
    for line_number, raw_line in enumerate(
        lines[header_line:], start=header_line + 1
    ):
        line = ANSI_CSI_RE.sub("", raw_line)
        tokens = line.split()
        if not tokens:
            continue
        match = GPU_TOKEN_RE.fullmatch(tokens[0])
        if match is None:
            continue
        row = int(match.group("index"), 10)
        if row not in header:
            raise ValidationError(f"{path}:{line_number}: unexpected topology row GPU{row}")
        if row in seen_rows:
            raise ValidationError(f"{path}:{line_number}: duplicate topology row GPU{row}")
        if len(tokens) < len(header) + 1:
            raise ValidationError(f"{path}:{line_number}: truncated topology row for GPU{row}")
        links = tokens[1 : len(header) + 1]
        for column, link in zip(header, links):
            if not _valid_topology_label(link):
                raise ValidationError(
                    f"{path}:{line_number}: invalid topology label {link!r} "
                    f"for GPU{row}->GPU{column}"
                )
            matrix[(row, column)] = link
        seen_rows.add(row)

    if seen_rows != set(header):
        missing = ",".join(f"GPU{index}" for index in sorted(set(header) - seen_rows))
        raise ValidationError(f"{path}: topology is missing rows: {missing}")
    for index in header:
        if matrix[(index, index)] != "X":
            raise ValidationError(
                f"{path}: topology diagonal GPU{index}->GPU{index} must be X"
            )
    for left, right in combinations(header, 2):
        forward = matrix[(left, right)]
        reverse = matrix[(right, left)]
        if forward == "X" or reverse == "X":
            raise ValidationError(
                f"{path}: off-diagonal topology GPU{left}<->GPU{right} cannot be X"
            )
        if forward != reverse:
            raise ValidationError(
                f"{path}: asymmetric topology GPU{left}<->GPU{right}: "
                f"{forward}/{reverse}"
            )
    return Topology(header, matrix)


def validate(
    cuda_inventory: dict[int, tuple[str, str]],
    nvidia_inventory: dict[int, tuple[str, str]],
    topology: Topology,
    selected: list[int],
    expected_selected: list[dict[str, str | int]],
) -> list[dict[str, str | int]]:
    expected_cuda_ordinals = set(range(len(cuda_inventory)))
    if set(cuda_inventory) != expected_cuda_ordinals:
        raise ValidationError(
            "CUDA ordinals must be contiguous from zero; got "
            + ",".join(str(index) for index in sorted(cuda_inventory))
        )
    cuda_devices = set(cuda_inventory.values())
    nvidia_devices = set(nvidia_inventory.values())
    if cuda_devices != nvidia_devices:
        raise ValidationError(
            "CUDA and nvidia-smi inventories do not describe the same PCI/UUID set"
        )
    if set(topology.indices) != set(nvidia_inventory):
        raise ValidationError(
            "nvidia-smi topology GPU indices do not match the nvidia-smi inventory"
        )

    expected_ordinals = [int(row["cuda_ordinal"]) for row in expected_selected]
    if selected != expected_ordinals:
        raise ValidationError(
            "selected CUDA ordinal order differs from the trusted baseline: "
            f"got {','.join(map(str, selected))}, expected "
            f"{','.join(map(str, expected_ordinals))}"
        )
    if set(selected) != set(cuda_inventory):
        raise ValidationError(
            "trusted selected CUDA ordinals must cover the complete CUDA inventory"
        )
    expected_devices = {
        (str(row["pci_bus_id"]), str(row["uuid"])) for row in expected_selected
    }
    if expected_devices != cuda_devices:
        raise ValidationError(
            "complete CUDA identity set differs from the trusted selected-identity baseline"
        )
    for row in expected_selected:
        ordinal = int(row["cuda_ordinal"])
        expected_identity = (str(row["pci_bus_id"]), str(row["uuid"]))
        if cuda_inventory.get(ordinal) != expected_identity:
            actual = cuda_inventory.get(ordinal)
            raise ValidationError(
                f"CUDA ordinal {ordinal} identity differs from trusted baseline: "
                f"got {actual!r}, expected {expected_identity!r}"
            )

    nvidia_by_identity = {
        identity: index for index, identity in nvidia_inventory.items()
    }
    cuda_to_nvidia = {
        ordinal: nvidia_by_identity[identity]
        for ordinal, identity in cuda_inventory.items()
    }

    half = len(selected) // 2
    intended_pairs = {
        frozenset(
            (
                cuda_to_nvidia[selected[index]],
                cuda_to_nvidia[selected[index + half]],
            )
        )
        for index in range(half)
    }
    for left, right in combinations(topology.indices, 2):
        link = topology[(left, right)]
        is_intended = frozenset((left, right)) in intended_pairs
        is_nvlink = NVLINK_RE.fullmatch(link) is not None
        if is_intended and not is_nvlink:
            raise ValidationError(
                f"intended logical pair in NVML namespace GPU{left}<->GPU{right} "
                f"is not NVLink: {link}"
            )
        if not is_intended and is_nvlink:
            raise ValidationError(
                "unexpected cross-pair NVLink in NVML namespace "
                f"GPU{left}<->GPU{right}: {link}"
            )

    rows: list[dict[str, str | int]] = []
    for logical_tier, ordinal in enumerate(selected):
        if logical_tier < half:
            logical_pair = logical_tier
            logical_role = "home"
            partner_ordinal = selected[logical_tier + half]
        else:
            logical_pair = logical_tier - half
            logical_role = "partner"
            partner_ordinal = selected[logical_tier - half]
        bus_id, uuid = cuda_inventory[ordinal]
        nvidia_index = cuda_to_nvidia[ordinal]
        partner_nvidia_index = cuda_to_nvidia[partner_ordinal]
        expected_row = expected_selected[logical_tier]
        if (
            expected_row["logical_pair"] != logical_pair
            or expected_row["logical_role"] != logical_role
        ):
            raise ValidationError(
                f"logical tier {logical_tier} pair/role differs from trusted baseline"
            )
        rows.append(
            {
                "logical_tier": logical_tier,
                "logical_pair": logical_pair,
                "logical_role": logical_role,
                "cuda_ordinal": ordinal,
                "nvidia_index": nvidia_index,
                "pci_bus_id": bus_id,
                "uuid": uuid,
                "nvlink_partner_cuda_ordinal": partner_ordinal,
                "nvlink_partner_nvidia_index": partner_nvidia_index,
                "nvlink_link": topology[(nvidia_index, partner_nvidia_index)],
            }
        )
    return rows


def _write_dict_rows(
    path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, str | int]]
) -> None:
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, csv.Error) as exc:
        raise ValidationError(f"cannot write {path}: {exc}") from exc


def write_rows(path: Path, rows: list[dict[str, str | int]]) -> None:
    _write_dict_rows(path, SELECTED_FIELDS, rows)


def write_inventory(
    path: Path, inventory: dict[int, tuple[str, str]], index_column: str
) -> None:
    rows = [
        {index_column: index, "pci_bus_id": identity[0], "uuid": identity[1]}
        for index, identity in sorted(inventory.items())
    ]
    _write_dict_rows(path, INVENTORY_FIELDS[index_column], rows)


def write_topology(path: Path, topology: Topology) -> None:
    rows = [
        {
            "nvidia_index": row,
            "peer_nvidia_index": column,
            "link": topology[(row, column)],
        }
        for row in sorted(topology.indices)
        for column in sorted(topology.indices)
    ]
    _write_dict_rows(path, NORMALIZED_TOPOLOGY_FIELDS, rows)


def read_selected_identity(path: Path) -> list[dict[str, str | int]]:
    handle, reader = _open_dict_reader(path, SELECTED_FIELDS)
    rows: list[dict[str, str | int]] = []
    try:
        for line_number, row in enumerate(reader, start=2):
            _validate_csv_row(path, line_number, row)
            parsed: dict[str, str | int] = {
                "logical_tier": parse_index(str(row["logical_tier"]), "logical_tier"),
                "logical_pair": parse_index(str(row["logical_pair"]), "logical_pair"),
                "logical_role": str(row["logical_role"]).strip(),
                "cuda_ordinal": parse_index(str(row["cuda_ordinal"]), "cuda_ordinal"),
                "nvidia_index": parse_index(str(row["nvidia_index"]), "nvidia_index"),
                "pci_bus_id": normalize_bus_id(str(row["pci_bus_id"])),
                "uuid": normalize_uuid(str(row["uuid"])),
                "nvlink_partner_cuda_ordinal": parse_index(
                    str(row["nvlink_partner_cuda_ordinal"]),
                    "nvlink_partner_cuda_ordinal",
                ),
                "nvlink_partner_nvidia_index": parse_index(
                    str(row["nvlink_partner_nvidia_index"]),
                    "nvlink_partner_nvidia_index",
                ),
                "nvlink_link": str(row["nvlink_link"]).strip(),
            }
            if parsed["logical_role"] not in {"home", "partner"}:
                raise ValidationError(f"{path}:{line_number}: invalid logical_role")
            if NVLINK_RE.fullmatch(str(parsed["nvlink_link"])) is None:
                raise ValidationError(f"{path}:{line_number}: invalid NVLink label")
            rows.append(parsed)
    except (csv.Error, UnicodeError) as exc:
        raise ValidationError(f"cannot parse {path}: {exc}") from exc
    finally:
        handle.close()
    if not rows:
        raise ValidationError(f"{path}: selected identity is empty")
    if len(rows) % 2 != 0:
        raise ValidationError(f"{path}: selected identity row count must be even")
    tiers = [int(row["logical_tier"]) for row in rows]
    if tiers != list(range(len(rows))):
        raise ValidationError(f"{path}: logical tiers must be ordered and contiguous from zero")
    if len({int(row["cuda_ordinal"]) for row in rows}) != len(rows):
        raise ValidationError(f"{path}: duplicate selected cuda_ordinal")
    if len({int(row["nvidia_index"]) for row in rows}) != len(rows):
        raise ValidationError(f"{path}: duplicate selected nvidia_index")
    identities = {(str(row["pci_bus_id"]), str(row["uuid"])) for row in rows}
    if len(identities) != len(rows):
        raise ValidationError(f"{path}: duplicate selected PCI/UUID identity")
    half = len(rows) // 2
    for tier, row in enumerate(rows):
        pair = tier if tier < half else tier - half
        role = "home" if tier < half else "partner"
        partner_tier = tier + half if tier < half else tier - half
        partner = rows[partner_tier]
        if row["logical_pair"] != pair or row["logical_role"] != role:
            raise ValidationError(f"{path}: logical tier {tier} has invalid pair/role")
        if (
            row["nvlink_partner_cuda_ordinal"] != partner["cuda_ordinal"]
            or row["nvlink_partner_nvidia_index"] != partner["nvidia_index"]
            or row["nvlink_link"] != partner["nvlink_link"]
        ):
            raise ValidationError(
                f"{path}: logical tier {tier} has inconsistent NVLink partner evidence"
            )
    return rows


def read_runtime_log(path: Path) -> tuple[list[dict[str, str | int]], list[str]]:
    records: list[dict[str, str | int]] = []
    malformed: list[str] = []
    try:
        with path.open(encoding="utf-8", errors="strict") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\r\n")
                match = RUNTIME_INVENTORY_RE.fullmatch(line)
                record_type = "inventory"
                if match is None:
                    match = RUNTIME_SELECTED_RE.fullmatch(line)
                    record_type = "selected"
                if match is None:
                    if line.startswith(RUNTIME_INVENTORY_PREFIX) or line.startswith(
                        RUNTIME_SELECTED_PREFIX
                    ):
                        malformed.append(f"line {line_number}: {line}")
                    continue
                groups = match.groupdict()
                records.append(
                    {
                        "record_type": record_type,
                        "line_number": line_number,
                        "logical_tier": groups.get("logical_tier") or "",
                        "cuda_ordinal": groups["cuda_ordinal"],
                        "pci_bus_id": groups["pci_bus_id"],
                        "uuid": groups["uuid"],
                        "property_query_status": groups["property_query_status"],
                        "pci_query_status": groups["pci_query_status"],
                    }
                )
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc
    return records, malformed


def write_runtime_rows(path: Path, rows: list[dict[str, str | int]]) -> None:
    _write_dict_rows(path, RUNTIME_FIELDS, rows)


def validate_runtime(
    records: list[dict[str, str | int]],
    malformed: list[str],
    cuda_inventory: dict[int, tuple[str, str]],
    selected_identity: list[dict[str, str | int]],
) -> None:
    if malformed:
        raise ValidationError("malformed CUDA identity log line: " + malformed[0])
    inventory_records: dict[int, dict[str, str | int]] = {}
    selected_records: dict[int, dict[str, str | int]] = {}
    for record in records:
        ordinal = parse_index(str(record["cuda_ordinal"]), "runtime cuda_ordinal")
        tier_text = str(record["logical_tier"])
        if (
            record["property_query_status"] != "cudaSuccess"
            or record["pci_query_status"] != "cudaSuccess"
        ):
            raise ValidationError(
                f"runtime {record['record_type']} identity query failed at log line "
                f"{record['line_number']}: {record['property_query_status']}/"
                f"{record['pci_query_status']}"
            )
        identity = (
            normalize_bus_id(str(record["pci_bus_id"])),
            normalize_uuid(str(record["uuid"])),
        )
        record["cuda_ordinal"] = ordinal
        record["pci_bus_id"] = identity[0]
        record["uuid"] = identity[1]
        if record["record_type"] == "inventory":
            if tier_text:
                raise ValidationError(
                    "runtime inventory record unexpectedly has a logical tier"
                )
            if ordinal in inventory_records:
                raise ValidationError(
                    f"duplicate runtime inventory line for CUDA ordinal {ordinal}"
                )
            inventory_records[ordinal] = record
        else:
            tier = parse_index(tier_text, "runtime logical_tier")
            record["logical_tier"] = tier
            if tier in selected_records:
                raise ValidationError(
                    f"duplicate runtime selected line for logical tier {tier}"
                )
            selected_records[tier] = record

    if set(inventory_records) != set(cuda_inventory):
        raise ValidationError(
            "runtime CUDA ordinal inventory lines are missing or unexpected; got "
            + ",".join(map(str, sorted(inventory_records)))
        )
    for ordinal, expected_identity in cuda_inventory.items():
        record = inventory_records[ordinal]
        actual_identity = (str(record["pci_bus_id"]), str(record["uuid"]))
        if actual_identity != expected_identity:
            raise ValidationError(
                f"runtime CUDA ordinal {ordinal} identity differs from preflight: "
                f"got {actual_identity!r}, expected {expected_identity!r}"
            )

    expected_tiers = set(range(len(selected_identity)))
    if set(selected_records) != expected_tiers:
        raise ValidationError(
            "runtime selected identity lines are missing or unexpected; got tiers "
            + ",".join(map(str, sorted(selected_records)))
        )
    for tier, expected in enumerate(selected_identity):
        record = selected_records[tier]
        actual = (
            int(record["cuda_ordinal"]),
            str(record["pci_bus_id"]),
            str(record["uuid"]),
        )
        wanted = (
            int(expected["cuda_ordinal"]),
            str(expected["pci_bus_id"]),
            str(expected["uuid"]),
        )
        if actual != wanted:
            raise ValidationError(
                f"runtime logical tier {tier} identity differs from preflight: "
                f"got {actual!r}, expected {wanted!r}"
            )


def capture(args: argparse.Namespace) -> None:
    cuda_inventory = read_inventory(args.cuda_inventory, "cuda_ordinal")
    nvidia_inventory = read_inventory(args.nvidia_inventory, "nvidia_index")
    selected = parse_selected(args.selected)
    expected_selected = read_expected_selected(args.expected_selected)
    topology = read_topology(args.topology)
    rows = validate(
        cuda_inventory,
        nvidia_inventory,
        topology,
        selected,
        expected_selected,
    )
    write_inventory(args.normalized_cuda_output, cuda_inventory, "cuda_ordinal")
    write_inventory(args.normalized_nvidia_output, nvidia_inventory, "nvidia_index")
    write_topology(args.normalized_topology_output, topology)
    write_rows(args.output, rows)


def validate_runtime_command(args: argparse.Namespace) -> None:
    records, malformed = read_runtime_log(args.runtime_log)
    # Preserve every parseable identity line even when subsequent validation fails.
    write_runtime_rows(args.output, records)
    cuda_inventory = read_inventory(args.cuda_inventory, "cuda_ordinal")
    selected_identity = read_selected_identity(args.selected_identity)
    validate_runtime(records, malformed, cuda_inventory, selected_identity)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--cuda-inventory", required=True, type=Path)
    capture_parser.add_argument("--nvidia-inventory", required=True, type=Path)
    capture_parser.add_argument("--topology", required=True, type=Path)
    capture_parser.add_argument("--selected", required=True)
    capture_parser.add_argument("--expected-selected", required=True, type=Path)
    capture_parser.add_argument("--normalized-cuda-output", required=True, type=Path)
    capture_parser.add_argument("--normalized-nvidia-output", required=True, type=Path)
    capture_parser.add_argument("--normalized-topology-output", required=True, type=Path)
    capture_parser.add_argument("--output", required=True, type=Path)
    capture_parser.set_defaults(action=capture)

    runtime_parser = commands.add_parser("runtime")
    runtime_parser.add_argument("--runtime-log", required=True, type=Path)
    runtime_parser.add_argument("--cuda-inventory", required=True, type=Path)
    runtime_parser.add_argument("--selected-identity", required=True, type=Path)
    runtime_parser.add_argument("--output", required=True, type=Path)
    runtime_parser.set_defaults(action=validate_runtime_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.action(args)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
