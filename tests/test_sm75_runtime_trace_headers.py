#!/usr/bin/env python3
"""CPU-only private-ABI gate regression tests; no CUDA library is loaded."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "speed-bench/check-sm75-runtime-trace-headers.py"
SPEC = importlib.util.spec_from_file_location("trace_headers", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class HeaderGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "crt").mkdir()
        (self.root / "cuda_runtime_api.h").write_text("#define CUDART_VERSION 13020\n")
        headers = {}
        for name, (path, ret, params) in MOD.EXPECTED.items():
            headers.setdefault(path, []).append(
                f'extern "C" {ret} CUDARTAPI {name}({", ".join(params)});\n')
        for path, lines in headers.items():
            (self.root / path).write_text("".join(lines))

    def test_exact_known_declarations(self):
        out = MOD.validate(self.root)
        self.assertEqual(out["status"], "header-declarations-match")
        self.assertEqual(len(out["symbols"]), 6)

    def test_expected_default_arguments_and_whitespace(self):
        path = self.root / "crt/device_functions.h"
        path.write_text(path.read_text().replace("dim3 blockDim,", "dim3 blockDim = 1,", 1)
                        .replace("size_t sharedMem,", "size_t sharedMem = 0,", 1)
                        .replace("struct CUstream_st *stream", "struct CUstream_st * stream = 0", 1))
        self.assertEqual(MOD.validate(self.root)["gpu_calls"], 0)

    def test_wrong_version_rejected(self):
        (self.root / "cuda_runtime_api.h").write_text("#define CUDART_VERSION 13030\n")
        with self.assertRaisesRegex(ValueError, "13.2"):
            MOD.validate(self.root)

    def test_pop_void_double_pointer_rejected(self):
        path = self.root / "crt/host_runtime.h"
        path.write_text(path.read_text().replace("void *stream", "void **stream"))
        with self.assertRaisesRegex(ValueError, "private ABI"):
            MOD.validate(self.root)

    def test_push_wrong_return_type_rejected(self):
        path = self.root / "crt/device_functions.h"
        path.write_text(path.read_text().replace("unsigned CUDARTAPI", "cudaError_t CUDARTAPI"))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MOD.validate(self.root)

    def test_ambiguous_duplicate_rejected(self):
        path = self.root / "crt/host_runtime.h"
        path.write_text(path.read_text() * 2)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MOD.validate(self.root)


if __name__ == "__main__":
    unittest.main()
