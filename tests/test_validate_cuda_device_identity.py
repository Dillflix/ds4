import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "speed-bench" / "validate-cuda-device-identity.py"
SPEC = importlib.util.spec_from_file_location("validate_cuda_identity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


UUIDS = [
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-11111111-1111-1111-1111-111111111111",
    "GPU-22222222-2222-2222-2222-222222222222",
    "GPU-33333333-3333-3333-3333-333333333333",
]
BUSES = [
    "00000000:02:00.0",
    "00000000:03:00.0",
    "00000000:81:00.0",
    "00000000:82:00.0",
]
SELECTED = [0, 3, 1, 2]

TOPOLOGY = """\
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV2     SYS     SYS     0-19            0
GPU1    NV2      X      SYS     SYS     0-19            0
GPU2    SYS     SYS      X      NV2     20-39           1
GPU3    SYS     SYS     NV2      X      20-39           1

Legend:
  X = Self
  NV# = Connection traversing bonded NVLinks
"""


def inventory(indices, buses=BUSES, uuids=UUIDS):
    return {index: (bus, uuid) for index, bus, uuid in zip(indices, buses, uuids)}


def expected_rows(cuda_inventory):
    pairs = [0, 1, 0, 1]
    roles = ["home", "home", "partner", "partner"]
    return [
        {
            "logical_tier": tier,
            "logical_pair": pairs[tier],
            "logical_role": roles[tier],
            "cuda_ordinal": ordinal,
            "pci_bus_id": cuda_inventory[ordinal][0],
            "uuid": cuda_inventory[ordinal][1],
        }
        for tier, ordinal in enumerate(SELECTED)
    ]


def write_inventory(path, index_column, values):
    path.write_text(
        f"{index_column},pci_bus_id,uuid\n"
        + "".join(
            f"{index},{identity[0]},{identity[1]}\n"
            for index, identity in sorted(values.items())
        ),
        encoding="utf-8",
    )


def write_expected(path, rows):
    path.write_text(
        "logical_tier,logical_pair,logical_role,cuda_ordinal,pci_bus_id,uuid\n"
        + "".join(
            f"{row['logical_tier']},{row['logical_pair']},{row['logical_role']},"
            f"{row['cuda_ordinal']},{row['pci_bus_id']},{row['uuid']}\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def runtime_log(cuda_inventory, selected_rows):
    lines = [
        "ds4: CUDA ordinal inventory "
        f"ordinal={ordinal} pci_bus_id={identity[0]} uuid={identity[1]} "
        "query_status=cudaSuccess/cudaSuccess"
        for ordinal, identity in sorted(cuda_inventory.items())
    ]
    lines.extend(
        "ds4: CUDA selected device identity "
        f"logical_tier={row['logical_tier']} cuda_ordinal={row['cuda_ordinal']} "
        f"pci_bus_id={row['pci_bus_id']} uuid={row['uuid']} "
        "query_status=cudaSuccess/cudaSuccess"
        for row in selected_rows
    )
    return "\n".join(lines) + "\n"


class ValidateCudaDeviceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.cuda = inventory(range(4))
        self.nvidia = dict(self.cuda)
        self.expected = expected_rows(self.cuda)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(TOPOLOGY)
            self.topology_path = Path(handle.name)
        self.topology = MODULE.read_topology(self.topology_path)

    def tearDown(self):
        self.topology_path.unlink(missing_ok=True)

    def test_fixed_production_order_maps_to_two_nvlink_pairs(self):
        rows = MODULE.validate(
            self.cuda, self.nvidia, self.topology, SELECTED, self.expected
        )
        self.assertEqual([row["logical_pair"] for row in rows], [0, 1, 0, 1])
        self.assertEqual(
            [row["nvlink_partner_cuda_ordinal"] for row in rows], [1, 2, 0, 3]
        )
        self.assertEqual({row["nvlink_link"] for row in rows}, {"NV2"})

    def test_swapped_nvidia_indices_are_joined_by_bus_and_uuid(self):
        self.nvidia = {
            0: self.cuda[1],
            1: self.cuda[0],
            2: self.cuda[3],
            3: self.cuda[2],
        }
        rows = MODULE.validate(
            self.cuda, self.nvidia, self.topology, SELECTED, self.expected
        )
        self.assertEqual([row["nvidia_index"] for row in rows], [1, 2, 0, 3])
        self.assertEqual(
            [row["nvlink_partner_nvidia_index"] for row in rows], [0, 3, 1, 2]
        )

    def test_topology_is_validated_in_derived_nvml_index_namespace(self):
        self.nvidia = {
            0: self.cuda[0],
            1: self.cuda[2],
            2: self.cuda[1],
            3: self.cuda[3],
        }
        matrix = {
            (left, right): (
                "X"
                if left == right
                else "NV2"
                if frozenset((left, right)) in {frozenset((0, 2)), frozenset((1, 3))}
                else "SYS"
            )
            for left in range(4)
            for right in range(4)
        }
        topology = MODULE.Topology(range(4), matrix)
        rows = MODULE.validate(
            self.cuda, self.nvidia, topology, SELECTED, self.expected
        )
        self.assertEqual([row["nvidia_index"] for row in rows], [0, 3, 2, 1])
        self.assertEqual({row["nvlink_link"] for row in rows}, {"NV2"})

    def test_noncontiguous_cuda_ordinals_are_rejected(self):
        del self.cuda[2]
        with self.assertRaisesRegex(MODULE.ValidationError, "contiguous"):
            MODULE.validate(
                self.cuda, self.nvidia, self.topology, SELECTED, self.expected
            )

    def test_trusted_baseline_identity_mismatch_is_rejected(self):
        self.expected[0]["uuid"] = UUIDS[1]
        with self.assertRaisesRegex(MODULE.ValidationError, "trusted"):
            MODULE.validate(
                self.cuda, self.nvidia, self.topology, SELECTED, self.expected
            )

    def test_checked_in_trusted_baseline_has_post_swap_mapping(self):
        rows = MODULE.read_expected_selected(
            REPO / "speed-bench" / "sm75-small-bar1-expected-device-identity.csv"
        )
        self.assertEqual(
            [
                (
                    row["logical_tier"],
                    row["logical_pair"],
                    row["logical_role"],
                    row["cuda_ordinal"],
                    row["pci_bus_id"],
                    row["uuid"],
                )
                for row in rows
            ],
            [
                (0, 0, "home", 0, "00000000:02:00.0", "GPU-cca9cdaa-29ba-d17f-e4e1-0c9d47503c62"),
                (1, 1, "home", 3, "00000000:82:00.0", "GPU-4bd650ce-a8ff-4411-3cc9-0dc3abaad694"),
                (2, 0, "partner", 1, "00000000:03:00.0", "GPU-ba2c2d0b-6320-580f-208c-59e548ac9227"),
                (3, 1, "partner", 2, "00000000:81:00.0", "GPU-56707308-5293-2d44-ef0d-1763abc7584f"),
            ],
        )

    def test_non_nvlink_intended_pair_is_rejected_in_nvml_namespace(self):
        self.topology[(0, 1)] = "PHB"
        self.topology[(1, 0)] = "PHB"
        with self.assertRaisesRegex(MODULE.ValidationError, "is not NVLink"):
            MODULE.validate(
                self.cuda, self.nvidia, self.topology, SELECTED, self.expected
            )

    def test_unexpected_cross_pair_nvlink_is_rejected(self):
        self.topology[(0, 2)] = "NV1"
        self.topology[(2, 0)] = "NV1"
        with self.assertRaisesRegex(MODULE.ValidationError, "unexpected cross-pair"):
            MODULE.validate(
                self.cuda, self.nvidia, self.topology, SELECTED, self.expected
            )

    def test_topology_index_set_must_match_nvidia_inventory(self):
        del self.nvidia[3]
        with self.assertRaisesRegex(MODULE.ValidationError, "same PCI/UUID set"):
            MODULE.validate(
                self.cuda, self.nvidia, self.topology, SELECTED, self.expected
            )

    def test_duplicate_selected_ordinal_is_rejected(self):
        with self.assertRaisesRegex(MODULE.ValidationError, "must be unique"):
            MODULE.parse_selected("0,3,1,1")

    def test_csv_normalizes_cuda_four_digit_domain(self):
        content = (
            "cuda_ordinal,pci_bus_id,uuid\n"
            "0,0000:02:00.0,GPU-AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA\n"
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(content)
            path = Path(handle.name)
        try:
            parsed = MODULE.read_inventory(path, "cuda_ordinal")
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(
            parsed[0],
            ("00000000:02:00.0", "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        )

    def test_csv_rejects_extra_fields(self):
        content = (
            "cuda_ordinal,pci_bus_id,uuid\n"
            f"0,{BUSES[0]},{UUIDS[0]},unexpected\n"
        )
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(content)
            path = Path(handle.name)
        try:
            with self.assertRaisesRegex(MODULE.ValidationError, "malformed CSV row"):
                MODULE.read_inventory(path, "cuda_ordinal")
        finally:
            path.unlink(missing_ok=True)

    def test_topology_rejects_duplicate_rows(self):
        path = self._write_topology(TOPOLOGY.replace("GPU1    NV2", "GPU0    NV2"))
        try:
            with self.assertRaisesRegex(MODULE.ValidationError, "duplicate topology row"):
                MODULE.read_topology(path)
        finally:
            path.unlink(missing_ok=True)

    def test_topology_accepts_ansi_underlined_header_from_nvidia_smi(self):
        topology = TOPOLOGY.replace(
            "        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity",
            "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity"
            "\tGPU NUMA ID\x1b[0m",
        )
        path = self._write_topology(topology)
        try:
            parsed = MODULE.read_topology(path)
            self.assertEqual(parsed.indices, (0, 1, 2, 3))
            self.assertEqual(parsed[(0, 1)], "NV2")
            self.assertEqual(parsed[(2, 3)], "NV2")
        finally:
            path.unlink(missing_ok=True)

    def test_topology_rejects_missing_rows(self):
        text = "\n".join(
            line for line in TOPOLOGY.splitlines() if not line.startswith("GPU3 ")
        )
        path = self._write_topology(text)
        try:
            with self.assertRaisesRegex(MODULE.ValidationError, "missing rows"):
                MODULE.read_topology(path)
        finally:
            path.unlink(missing_ok=True)

    def test_topology_rejects_bad_diagonal(self):
        path = self._write_topology(TOPOLOGY.replace("GPU0     X", "GPU0    SYS"))
        try:
            with self.assertRaisesRegex(MODULE.ValidationError, "diagonal"):
                MODULE.read_topology(path)
        finally:
            path.unlink(missing_ok=True)

    def test_topology_rejects_asymmetric_links(self):
        path = self._write_topology(TOPOLOGY.replace("GPU1    NV2", "GPU1    SYS"))
        try:
            with self.assertRaisesRegex(MODULE.ValidationError, "asymmetric"):
                MODULE.read_topology(path)
        finally:
            path.unlink(missing_ok=True)

    def _write_topology(self, text):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(text)
            return Path(handle.name)

    def test_capture_cli_writes_normalized_full_evidence_and_derived_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_cuda = root / "cuda.raw.csv"
            raw_nvidia = root / "nvidia.raw.csv"
            raw_topology = root / "topology.txt"
            expected = root / "expected.csv"
            normalized_cuda = root / "cuda.csv"
            normalized_nvidia = root / "nvidia.csv"
            normalized_topology = root / "topology.csv"
            selected_output = root / "selected.csv"
            swapped_nvidia = {
                0: self.cuda[1],
                1: self.cuda[0],
                2: self.cuda[3],
                3: self.cuda[2],
            }
            write_inventory(raw_cuda, "cuda_ordinal", self.cuda)
            write_inventory(raw_nvidia, "nvidia_index", swapped_nvidia)
            raw_topology.write_text(TOPOLOGY, encoding="utf-8")
            write_expected(expected, self.expected)
            status = MODULE.main(
                [
                    "capture",
                    "--cuda-inventory",
                    str(raw_cuda),
                    "--nvidia-inventory",
                    str(raw_nvidia),
                    "--topology",
                    str(raw_topology),
                    "--selected",
                    "0,3,1,2",
                    "--expected-selected",
                    str(expected),
                    "--normalized-cuda-output",
                    str(normalized_cuda),
                    "--normalized-nvidia-output",
                    str(normalized_nvidia),
                    "--normalized-topology-output",
                    str(normalized_topology),
                    "--output",
                    str(selected_output),
                ]
            )
            selected_text = selected_output.read_text(encoding="utf-8")
            topology_text = normalized_topology.read_text(encoding="utf-8")
        self.assertEqual(status, 0)
        self.assertTrue(selected_text.startswith(",".join(MODULE.SELECTED_FIELDS) + "\n"))
        self.assertIn(f"0,0,home,0,1,{BUSES[0]},{UUIDS[0]},1,0,NV2", selected_text)
        self.assertEqual(len(topology_text.splitlines()), 17)

    def _prepare_runtime_files(self, root):
        normalized_cuda = root / "cuda.csv"
        selected_output = root / "selected.csv"
        MODULE.write_inventory(normalized_cuda, self.cuda, "cuda_ordinal")
        selected_rows = MODULE.validate(
            self.cuda, self.nvidia, self.topology, SELECTED, self.expected
        )
        MODULE.write_rows(selected_output, selected_rows)
        return normalized_cuda, selected_output, selected_rows

    def _runtime_status(self, root, text):
        cuda_path, selected_path, selected_rows = self._prepare_runtime_files(root)
        log_path = root / "arm.log"
        evidence_path = root / "arm-device-identity.csv"
        log_path.write_text(text(selected_rows) if callable(text) else text, encoding="utf-8")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = MODULE.main(
                [
                    "runtime",
                    "--runtime-log",
                    str(log_path),
                    "--cuda-inventory",
                    str(cuda_path),
                    "--selected-identity",
                    str(selected_path),
                    "--output",
                    str(evidence_path),
                ]
            )
        return status, evidence_path.read_text(encoding="utf-8"), stderr.getvalue()

    def test_runtime_cli_accepts_exact_inventory_and_selected_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, evidence, error = self._runtime_status(
                root, lambda rows: runtime_log(self.cuda, rows)
            )
        self.assertEqual((status, error), (0, ""))
        self.assertEqual(len(evidence.splitlines()), 9)

    def test_runtime_cli_rejects_missing_selected_line_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, evidence, error = self._runtime_status(
                root,
                lambda rows: "\n".join(runtime_log(self.cuda, rows).splitlines()[:-1])
                + "\n",
            )
        self.assertEqual(status, 2)
        self.assertIn("missing or unexpected", error)
        self.assertEqual(len(evidence.splitlines()), 8)

    def test_runtime_cli_rejects_query_failure_and_preserves_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, evidence, error = self._runtime_status(
                root,
                lambda rows: runtime_log(self.cuda, rows).replace(
                    "query_status=cudaSuccess/cudaSuccess",
                    "query_status=cudaErrorInvalidDevice/cudaSuccess",
                    1,
                ),
            )
        self.assertEqual(status, 2)
        self.assertIn("identity query failed", error)
        self.assertIn("cudaErrorInvalidDevice,cudaSuccess", evidence)

    def test_runtime_cli_rejects_identity_mismatch_and_preserves_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, evidence, error = self._runtime_status(
                root,
                lambda rows: runtime_log(self.cuda, rows).replace(BUSES[0], "0000:04:00.0", 1),
            )
        self.assertEqual(status, 2)
        self.assertIn("differs from preflight", error)
        self.assertIn("0000:04:00.0", evidence)

    def test_runtime_cli_rejects_identity_line_without_query_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, evidence, error = self._runtime_status(
                root,
                lambda rows: runtime_log(self.cuda, rows).replace(
                    " query_status=cudaSuccess/cudaSuccess", "", 1
                ),
            )
        self.assertEqual(status, 2)
        self.assertIn("malformed CUDA identity log line", error)
        self.assertEqual(len(evidence.splitlines()), 8)


if __name__ == "__main__":
    unittest.main()
