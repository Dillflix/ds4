#!/usr/bin/env python3
"""Validate compiler-private declarations in installed official CUDA13.2 CRT.

This check reads header text only. It neither loads CUDA nor launches the target.
Public wrapper definitions are checked by the C++ compiler against vendor headers.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED = {
    "__cudaPushCallConfiguration": (
        "crt/device_functions.h", "unsigned", [
            "dim3 gridDim", "dim3 blockDim", "size_t sharedMem", "struct CUstream_st *stream",
        ]),
    "__cudaGetKernel": ("crt/device_functions.h", "cudaError_t", ["cudaKernel_t *", "const void *"]),
    "__cudaLaunchKernel": (
        "crt/device_functions.h", "cudaError_t", [
            "cudaKernel_t kernel", "dim3 gridDim", "dim3 blockDim", "void **args",
            "size_t sharedMem", "cudaStream_t stream",
        ]),
    "__cudaPopCallConfiguration": (
        "crt/host_runtime.h", "cudaError_t", [
            "dim3 *gridDim", "dim3 *blockDim", "size_t *sharedMem", "void *stream",
        ]),
    "__cudaRegisterFunction": (
        "crt/host_runtime.h", "void", [
            "void **fatCubinHandle", "const char *hostFun", "char *deviceFun", "const char *deviceName",
            "int thread_limit", "uint3 *tid", "uint3 *bid", "dim3 *bDim", "dim3 *gDim", "int *wSize",
        ]),
    "__cudaUnregisterFatBinary": ("crt/host_runtime.h", "void", ["void **fatCubinHandle"]),
}


def normalize(text):
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    text = re.sub(r"\s*=\s*[^,]+", "", text)
    return re.sub(r"\s+", "", text)


def validate(include, crt_include=None):
    include = Path(include)
    crt_include = Path(crt_include or include)
    runtime = (include / "cuda_runtime_api.h").read_text(encoding="utf-8")
    if not re.search(r"^\s*#\s*define\s+CUDART_VERSION\s+13020\b", runtime, re.M):
        raise ValueError("official CUDA 13.2 header version 13020 is required")
    checked = {}
    for symbol, (relative, return_type, params) in EXPECTED.items():
        path = crt_include / relative
        raw = path.read_bytes()
        header = raw.decode("utf-8")
        matches = list(re.finditer(
            r'extern\s+(?:"C"\s+)?(?:__host__\s+__device__\s+)?' + return_type +
            r"\s+CUDARTAPI\s+" + symbol + r"\s*\(([^;]+?)\)\s*;", header, re.S))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one official declaration for {symbol}")
        got = [normalize(part) for part in matches[0].group(1).split(",")]
        if got != [normalize(part) for part in params]:
            raise ValueError(f"unsupported compiler-private ABI for {symbol}")
        checked[symbol] = {"header": relative, "sha256": hashlib.sha256(raw).hexdigest()}
    return {"status": "header-declarations-match", "cuda_header_version": 13020,
            "symbols": checked, "gpu_calls": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include", type=Path, required=True)
    ap.add_argument("--crt-include", type=Path, help="optional separate official CRT include root")
    args = ap.parse_args()
    try:
        result = validate(args.include, args.crt_include)
    except (OSError, UnicodeError, ValueError) as exc:
        ap.exit(2, f"runtime trace header validation failed: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
