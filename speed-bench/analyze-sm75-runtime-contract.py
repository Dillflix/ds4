#!/usr/bin/env python3
"""Read-only, CPU-only analysis of an explicitly instrumented CUDA API trace.

Input is evidence, never instructions. This checks recorded API contracts and
allocation envelopes, not GPU instructions, library correctness or race freedom.
Missing observations are unknowns, not successful checks. No CUDA is imported.
"""
import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


SCHEMA = "ds4-runtime-contract-v1"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 1024 * 1024
MAX_RECORDS = 200000
MAX_RECORD_NODES = 4096
MAX_PROCESSES = 16
MAX_RELATION_CHECKS = 20_000_000
U64_MAX = (1 << 64) - 1
I64_MAX = (1 << 63) - 1
DTYPE_BYTES = {0: 4, 1: 8, 2: 2, 3: 1, 4: 8, 5: 16, 6: 4,
               8: 1, 10: 4, 12: 4, 14: 2}
DTYPE_NAMES = {"CUDA_R_32F": 0, "CUDA_R_64F": 1, "CUDA_R_16F": 2,
               "CUDA_R_8I": 3, "CUDA_C_32F": 4, "CUDA_C_64F": 5,
               "CUDA_C_16F": 6, "CUDA_R_8U": 8, "CUDA_R_32I": 10,
               "CUDA_R_32U": 12, "CUDA_R_16BF": 14}
SYNCHRONOUS_ALLOCATORS = {"cudaMalloc", "cudaMallocManaged"}


class EvidenceError(ValueError):
    """Malformed or excessive evidence; not an application GPU failure."""


class ContractError(ValueError):
    """An unsupported or invalid recorded matrix contract."""


class WorkBudget:
    def __init__(self):
        self.remaining = MAX_RELATION_CHECKS

    def consume(self, count):
        self.remaining -= count
        if self.remaining < 0:
            raise EvidenceError("analysis relation-work limit exceeded")


def integer(value, name, minimum=0, maximum=U64_MAX):
    if type(value) is not int or not minimum <= value <= maximum:
        raise EvidenceError("invalid integer " + name)
    return value


def pointer(value):
    if isinstance(value, str) and value.startswith("0x") and 1 <= len(value[2:]) <= 16:
        try:
            value = int(value[2:], 16)
        except ValueError:
            raise EvidenceError("invalid hexadecimal pointer") from None
    return integer(value, "pointer")


def checked_product(*values):
    result = 1
    for value in values:
        if type(value) is not int or value < 0:
            raise ContractError("negative-or-noninteger-matrix-field")
        result *= value
        if result > U64_MAX:
            raise ContractError("matrix-size-overflow")
    return result


def checked_sum(*values):
    result = sum(values)
    if result > U64_MAX:
        raise ContractError("matrix-size-overflow")
    return result


def matrix_extents(args):
    """Minimum touched byte envelopes, column-major; strides are in elements.

    Leading-dimension padding between columns is included. End padding is not
    assumed touched. Pointer-array batching and negative strides are unsupported.
    The envelopes do not prove that the implementation respects those bounds.
    """
    def dimension(key):
        value = args.get(key)
        if type(value) is not int or value < 0 or value > I64_MAX:
            raise ContractError("invalid-" + key)
        return value

    def operation(key):
        value = args.get(key)
        aliases = {0: "N", 1: "T", 2: "C", "N": "N", "T": "T", "C": "C",
                   "CUBLAS_OP_N": "N", "CUBLAS_OP_T": "T", "CUBLAS_OP_C": "C"}
        if isinstance(value, bool) or not isinstance(value, (int, str)) or value not in aliases:
            raise ContractError("unsupported-" + key)
        return aliases[value]

    m, n, k = (dimension(name) for name in ("m", "n", "k"))
    ta, tb = operation("transa"), operation("transb")
    count = dimension("batch_count") if "batch_count" in args else 1
    shapes = {"A": (m, k) if ta == "N" else (k, m),
              "B": (k, n) if tb == "N" else (n, k), "C": (m, n)}
    result = {}
    for operand, (rows, columns) in shapes.items():
        key = operand.lower()
        ld = dimension("ld" + key)
        if ld < max(1, rows):
            raise ContractError("invalid-leading-dimension-" + operand)
        dtype = args.get(key + "type")
        if isinstance(dtype, str):
            dtype = DTYPE_NAMES.get(dtype)
        if type(dtype) is not int or dtype not in DTYPE_BYTES:
            raise ContractError("unsupported-dtype-" + operand)
        stride_key = "stride_" + key
        if count > 1 and stride_key not in args:
            raise ContractError("unsupported-pointer-array-or-missing-stride-" + operand)
        if type(args.get(stride_key)) is int and args[stride_key] < 0:
            raise ContractError("unsupported-negative-stride-" + operand)
        stride = dimension(stride_key) if stride_key in args else 0
        elements = checked_sum(checked_product(columns - 1, ld), rows) if rows and columns else 0
        batch_elements = checked_sum(checked_product(count - 1, stride), elements) if count and elements else 0
        # m==0 or n==0 is a GEMM quick-return case, irrespective of A/B storage.
        if not m or not n:
            batch_elements = 0
        result[operand] = {"rows": rows, "columns": columns, "leading_dimension": ld,
                           "element_bytes": DTYPE_BYTES[dtype], "batch_count": count,
                           "stride_elements": stride, "bytes": checked_product(batch_elements, DTYPE_BYTES[dtype]),
                           "envelope_is_contiguous": (columns <= 1 or rows == ld) and
                                                     (count <= 1 or stride <= elements)}
    return result


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise EvidenceError("duplicate JSON object key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise EvidenceError("non-finite JSON constant: " + value)


def _validate_tree(value):
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAX_RECORD_NODES or depth > 32:
            raise EvidenceError("JSON record complexity limit exceeded")
        if type(item) is int and not -(1 << 63) <= item <= U64_MAX:
            raise EvidenceError("JSON integer exceeds supported 64-bit evidence range")
        if isinstance(item, float) and not math.isfinite(item):
            raise EvidenceError("non-finite JSON number")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)


def read_trace(path):
    records, gaps = [], []
    total = 0
    digest = hashlib.sha256()
    seen = set()
    previous = 0
    with open(path, "rb") as source:
        while True:
            line = source.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            total += len(line)
            digest.update(line)
            if total > MAX_FILE_BYTES or len(line) > MAX_LINE_BYTES:
                raise EvidenceError("trace byte or line-size limit exceeded")
            if len(records) >= MAX_RECORDS:
                raise EvidenceError("trace record-count limit exceeded")
            if not line.endswith(b"\n"):
                # Only the unterminated final line may be partial. A malformed
                # complete line is never silently discarded as a fault tail.
                gaps.append({"code": "unterminated-final-record", "line": len(records) + 1})
            try:
                record = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs,
                                    parse_constant=_reject_constant)
            except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
                if not line.endswith(b"\n"):
                    gaps.append({"code": "partial-final-record", "bytes": len(line)})
                    break
                raise EvidenceError("invalid complete JSON record") from error
            if not isinstance(record, dict) or record.get("schema") != SCHEMA:
                raise EvidenceError("unexpected trace record schema")
            _validate_tree(record)
            for key in ("seq", "pid", "tid", "monotonic_ns", "realtime_ns"):
                integer(record.get(key), key)
            seq = record["seq"]
            if seq == 0 or seq in seen:
                raise EvidenceError("zero or duplicate sequence number")
            seen.add(seq)
            if seq != previous + 1:
                gaps.append({"code": "noncontiguous-sequence", "previous": previous, "current": seq})
            previous = seq
            event = record.get("event")
            if event not in ("enter", "exit", "observation", "meta"):
                raise EvidenceError("unsupported record event")
            if not isinstance(record.get("api"), str) or not record["api"] or len(record["api"]) > 256:
                raise EvidenceError("invalid API name")
            if event in ("enter", "exit"):
                integer(record.get("call_id"), "call_id", minimum=1)
            if event == "observation":
                integer(record.get("parent_call_id"), "parent_call_id", minimum=1)
            if event == "exit":
                integer(record.get("status"), "status", minimum=-(1 << 31), maximum=(1 << 31) - 1)
            for key in ("args", "result"):
                if key in record and not isinstance(record[key], dict):
                    raise EvidenceError(key + " must be an object")
            if event == "observation" and record["api"] == "gemm_contract":
                result = record.get("result", {})
                for key in ("handle", "operands", "scalars"):
                    if key in result and not isinstance(result[key], dict):
                        raise EvidenceError("invalid GEMM observation " + key)
                if any(not isinstance(v, dict) for v in result.get("operands", {}).values()):
                    raise EvidenceError("invalid operand observation")
                for obj in [result, result.get("handle", {}), *result.get("operands", {}).values()]:
                    for key, value in obj.items():
                        if key.endswith("status") and value is not None:
                            integer(value, "observation " + key, minimum=-(1 << 31), maximum=(1 << 31)-1)
                    for key in ("device", "type", "pointer_mode", "math_mode"):
                        if key in obj:
                            integer(obj[key], "observation " + key, minimum=-1, maximum=(1 << 31)-1)
            records.append(record)
    return records, gaps, digest.hexdigest(), total


def _issue(code, call=None, **values):
    result = {"code": code}
    if call is not None:
        result.update(pid=call["pid"], call_id=call["call_id"], seq=call["enter_seq"])
    result.update(values)
    return result


def _device(call):
    if call.get("args", {}).get("device_query_status") == 0:
        value = call["args"].get("device")
        if type(value) is int:
            return value
    for observation in call.get("observations", []):
        result = observation.get("result", {})
        if result.get("device_query_status") == 0 and type(result.get("device")) is int:
            return result["device"]
    return call.get("recorded_device")


def _context(call):
    sources = [call.get("args", {})] + [r.get("result", {}) for r in call.get("observations", [])]
    for source in sources:
        if source.get("context_query_status") == 0 and "context" in source:
            return pointer(source["context"])
    return None


def _is_gemm(api):
    return "gemm" in api.lower() and api != "gemm_contract"


def _shape(call, n, algorithm):
    args = call["args"]
    return (call.get("library_forward_seq") is not None and "gemmex" in call["api"].lower() and
            args.get("algorithm_explicit", True) is not False and
            args.get("m") == 4096 and args.get("n") == n and
            args.get("k") == 8192 and args.get("algorithm") == algorithm and
            args.get("transa") in (1, "T", "CUBLAS_OP_T") and
            args.get("transb") in (0, "N", "CUBLAS_OP_N") and
            args.get("lda") == 8192 and args.get("ldb") == 8192 and args.get("ldc") == 4096 and
            args.get("atype") in (2, "CUDA_R_16F") and args.get("btype") in (2, "CUDA_R_16F") and
            args.get("ctype") in (0, "CUDA_R_32F") and args.get("compute_type") in (68, 69))


def _production_observations(call):
    handle = call.get("handle_contract", {})
    scalars = call.get("scalar_contract", {})
    host_mode = handle.get("pointer_mode_query_status") == 0 and handle.get("pointer_mode") == 0
    alpha_one = scalars.get("alpha", {}).get("bits_hex") == "0000803f"
    beta_zero = scalars.get("beta", {}).get("bits_hex") == "00000000"
    return {"host_pointer_mode_observed": host_mode, "alpha_positive_one_f32_observed": alpha_one,
            "beta_positive_zero_f32_observed": beta_zero,
            "expected_scalar_handle_contract_observed": host_mode and alpha_one and beta_zero}


def analyze_conversions(calls, history, violations, gaps, budget):
    """Check only the exact recognized conversion's three captured arguments.

    This is a CPU reconstruction of its nominal source/destination envelopes.
    It neither decodes arbitrary kernel arguments nor infers GPU completion.
    """
    conversions = []
    launch_apis = {"cudaLaunchKernel", "cudaLaunchKernelExC", "__cudaLaunchKernel"}
    unsupported_launches = 0
    for call in calls:
        if call["api"] not in launch_apis:
            continue
        observations = [r for r in call["observations"] if r["api"] == "conversion_contract"]
        if len(observations) != 1:
            unsupported_launches += 1
            continue
        observation = observations[0]
        result = observation.get("result", {})
        item = {"pid": call["pid"], "tid": call["tid"], "call_id": call["call_id"],
                "enter_seq": call["enter_seq"], "observation_seq": observation["seq"],
                "kernel_name": result.get("kernel_name"), "stream": call["args"].get("stream"),
                "context": hex(_context(call)) if _context(call) is not None else None,
                "device": _device(call), "api_return_status": call.get("status"),
                "exit_seq": call.get("exit_seq"), "ranges": {}}
        conversions.append(item)
        if result.get("recognized") is not True or result.get("kernel_name") != "_Z17f32_to_f16_kernelP6__halfPKfm":
            gaps.append(_issue("conversion-kernel-identity-not-recognized", call))
            continue
        if result.get("argument_read_status") != "complete":
            gaps.append(_issue("conversion-arguments-unreadable", call))
            continue
        count = integer(result.get("count"), "conversion count")
        item["count"] = count
        for operand, width in (("src", 4), ("dst", 2)):
            address = pointer(result.get(operand))
            try:
                length = checked_product(count, width)
            except ContractError:
                violations.append(_issue("conversion-span-overflow", call, operand=operand))
                continue
            detail = {"pointer": hex(address), "bytes": length, "range_status": "unknown"}
            item["ranges"][operand] = detail
            if address + length > U64_MAX:
                violations.append(_issue("conversion-address-overflow", call, operand=operand))
                continue
            if length == 0:
                detail["range_status"] = "zero-byte-contract"
                continue
            if address == 0:
                violations.append(_issue("null-nonempty-conversion-operand", call, operand=operand))
            budget.consume(len(history))
            candidates = [a for a in history if a["pid"] == call["pid"] and
                          a["allocate_exit_seq"] < observation["seq"] and
                          (a["free_exit_seq"] is None or a["free_exit_seq"] > observation["seq"]) and
                          pointer(a["base"]) <= address < pointer(a["base"]) + a["bytes"]]
            if len(candidates) == 1:
                allocation = candidates[0]
                detail.update(allocation_generation=allocation["generation"],
                              allocate_call_id=allocation["allocate_call_id"], allocation_base=allocation["base"],
                              allocation_bytes=allocation["bytes"], offset_bytes=address-pointer(allocation["base"]),
                              range_status="within-recorded-allocation")
                if address + length > pointer(allocation["base"]) + allocation["bytes"]:
                    detail["range_status"] = "out-of-bounds"
                    violations.append(_issue("conversion-exceeds-recorded-allocation", call, operand=operand))
                if allocation["allocate_tid"] != call["tid"] or allocation.get("failed_free_calls"):
                    gaps.append(_issue("conversion-allocation-lifetime-order-unresolved", call, operand=operand))
                if allocation["free_exit_seq"] is not None:
                    gaps.append(_issue("conversion-free-vs-GPU-completion-unresolved", call, operand=operand))
            else:
                gaps.append(_issue("conversion-allocation-unobserved-at-argument-snapshot", call, operand=operand))
        src, dst = item["ranges"].get("src"), item["ranges"].get("dst")
        if src and dst and src["bytes"] and dst["bytes"]:
            if pointer(src["pointer"]) < pointer(dst["pointer"]) + dst["bytes"] and pointer(dst["pointer"]) < pointer(src["pointer"]) + src["bytes"]:
                gaps.append(_issue("conversion-source-destination-overlap-no-race-proof", call))
    if unsupported_launches:
        gaps.append(_issue("other-kernel-arguments-not-decoded", count=unsupported_launches))
    for call in calls:
        if not _is_gemm(call["api"]):
            continue
        call["conversion_producer_candidates"] = []
        for operand in ("A", "B"):
            consumer = call.get("operand_contracts", {}).get(operand)
            if not consumer or consumer.get("element_bytes") != 2:
                continue
            address = pointer(consumer["pointer"])
            budget.consume(len(conversions))
            prior = [item for item in conversions if item["pid"] == call["pid"] and item["api_return_status"] == 0 and
                     item["exit_seq"] is not None and item["exit_seq"] < call.get("library_forward_seq", call["enter_seq"]) and
                     "dst" in item["ranges"] and pointer(item["ranges"]["dst"]["pointer"]) <= address and
                     address + consumer["bytes"] <= pointer(item["ranges"]["dst"]["pointer"]) + item["ranges"]["dst"]["bytes"]]
            if not prior:
                continue
            producer = max(prior, key=lambda item: item["exit_seq"])
            output_range = producer["ranges"]["dst"]
            same_generation = (consumer.get("allocate_call_id") is not None and
                               output_range.get("allocate_call_id") == consumer.get("allocate_call_id"))
            same_context = producer["context"] is not None and _context(call) is not None and pointer(producer["context"]) == _context(call)
            handle = call.get("handle_contract", {})
            same_stream = (producer["stream"] is not None and handle.get("stream_query_status") == 0 and
                           "stream" in handle and pointer(producer["stream"]) == pointer(handle["stream"]))
            call["conversion_producer_candidates"].append({"operand": operand, "conversion_call_id": producer["call_id"],
                                                           "same_recorded_allocation_generation": same_generation,
                                                           "same_recorded_context": same_context,
                                                           "same_recorded_stream_identifier": same_stream,
                                                           "same_host_thread": producer["tid"] == call["tid"],
                                                           "interpretation": "address-envelope-and-host-order-candidate-not-exclusive-producer-or-race-proof"})
            if not same_generation or not same_context or not same_stream or producer["tid"] != call["tid"]:
                gaps.append(_issue("conversion-to-gemm-dependency-not-established", call, operand=operand))
    return conversions


def analyze_file(path):
    records, gaps, digest, total = read_trace(Path(path))
    budget = WorkBudget()
    calls, open_calls, errors, violations = [], [], [], []
    by_id, exits, observations = {}, {}, {}
    starts = [r for r in records if r["event"] == "meta" and r["api"] == "trace_start"]
    ends = [r for r in records if r["event"] == "meta" and r["api"] == "trace_end"]
    pids = sorted({r["pid"] for r in records})
    if len(pids) > MAX_PROCESSES:
        raise EvidenceError("process-count limit exceeded")
    for pid in pids:
        if sum(r["pid"] == pid for r in starts) != 1:
            gaps.append(_issue("missing-or-duplicate-trace-start", pid=pid))
        if sum(r["pid"] == pid for r in ends) != 1:
            gaps.append(_issue("missing-or-duplicate-trace-end", pid=pid))
        process_records = [r for r in records if r["pid"] == pid]
        if process_records and process_records[0]["api"] != "trace_start":
            gaps.append(_issue("metadata-does-not-bracket-records", pid=pid))
    if not records:
        gaps.append(_issue("empty-trace"))
    if len(pids) > 1:
        gaps.append(_issue("multiple-process-trace-no-cross-process-order-proof"))
    for record in sorted(records, key=lambda r: r["seq"]):
        event = record["event"]
        key = (record["pid"], record.get("call_id"))
        if event == "enter":
            if key in by_id:
                raise EvidenceError("duplicate call ID")
            call = {"pid": record["pid"], "tid": record["tid"], "call_id": record["call_id"],
                    "api": record["api"], "enter_seq": record["seq"],
                    "enter_monotonic_ns": record["monotonic_ns"],
                    "enter_realtime_ns": record["realtime_ns"], "args": record.get("args", {}),
                    "observations": []}
            by_id[key] = call
            calls.append(call)
        elif event == "exit":
            if key in exits:
                raise EvidenceError("duplicate API exit")
            exits[key] = record
        elif event == "observation":
            key = (record["pid"], record["parent_call_id"])
            observations.setdefault(key, []).append(record)
    for key, call in by_id.items():
        exit_record = exits.pop(key, None)
        call["observations"] = observations.pop(key, [])
        if exit_record is None:
            open_calls.append({k: call[k] for k in ("pid", "tid", "call_id", "api", "enter_seq")})
            gaps.append(_issue("missing-api-exit", call))
        else:
            if (exit_record["api"] != call["api"] or exit_record["tid"] != call["tid"] or
                    exit_record["seq"] <= call["enter_seq"]):
                raise EvidenceError("API enter/exit identity or order mismatch")
            call.update(exit_seq=exit_record["seq"], status=exit_record["status"],
                        exit_monotonic_ns=exit_record["monotonic_ns"],
                        exit_realtime_ns=exit_record["realtime_ns"], result=exit_record.get("result", {}))
            if exit_record["monotonic_ns"] < call["enter_monotonic_ns"]:
                gaps.append(_issue("nonmonotonic-call-clock", call))
            if call["status"] != 0:
                errors.append({k: call[k] for k in ("pid", "tid", "call_id", "api", "status", "enter_seq", "exit_seq")})
        for observation in call["observations"]:
            if (observation["tid"] != call["tid"] or observation["seq"] <= call["enter_seq"] or
                    (exit_record and observation["seq"] >= exit_record["seq"])):
                gaps.append(_issue("observation-outside-api-interval", call))
        if _is_gemm(call["api"]):
            forward = [r for r in call["observations"] if r["api"] == "forward_to_real"]
            if len(forward) == 1:
                call["library_forward_seq"] = forward[0]["seq"]
                call["forwarding_interpretation"] = "marker-before-real-host-library-call-not-proof-of-GPU-enqueue"
            else:
                gaps.append(_issue("gemm-real-library-forwarding-unresolved", call))
                call["forwarding_interpretation"] = "interposer-enter-only-no-explicit-real-library-forward-marker"
    for key in exits:
        gaps.append(_issue("orphan-api-exit", pid=key[0], call_id=key[1]))
    for key in observations:
        gaps.append(_issue("orphan-observation", pid=key[0], call_id=key[1]))
    accepted_late_finalizers = []
    for end in ends:
        trailing = [r for r in records if r["pid"] == end["pid"] and r["seq"] > end["seq"]]
        # The interposer's DSO finalizer is not process exit. A linked CUDA
        # module may unregister later. Only balanced successful unregister
        # pairs are allowed; any other late work remains a structural gap.
        allowed = bool(trailing) and all(r["api"] == "__cudaUnregisterFatBinary" and
                                        r["event"] in ("enter", "exit") for r in trailing)
        if allowed:
            tail_calls = [c for c in calls if c["pid"] == end["pid"] and c["enter_seq"] > end["seq"]]
            allowed = bool(tail_calls) and all(c.get("status") == 0 and "exit_seq" in c for c in tail_calls)
        if trailing and not allowed:
            gaps.append(_issue("unqualified-records-after-trace-finalizer", pid=end["pid"]))
        elif allowed:
            accepted_late_finalizers.extend({"pid": c["pid"], "call_id": c["call_id"], "exit_seq": c["exit_seq"]}
                                            for c in tail_calls)

    free_intervals = {}
    for call in calls:
        if call["api"] == "cudaFree":
            key = (call["pid"], pointer(call["args"].get("ptr")))
            free_intervals.setdefault(key, []).append(call)

    # Replay host records. A successful allocation exit introduces a generation;
    # a successful free exit ends it. This is NOT a GPU lifetime/race proof.
    live, history, generations, devices, handles = {}, [], {}, {}, {}
    events = []
    for call in calls:
        # The optional marker is the last recorded point before forwarding.
        # Queries within intercepted entry may interleave with other threads.
        events.append((call.get("library_forward_seq", call["enter_seq"]), "enter", call))
        if "exit_seq" in call:
            events.append((call["exit_seq"], "exit", call))
    gemms = []
    for seq, event, call in sorted(events, key=lambda item: item[0]):
        api, args = call["api"], call["args"]
        process_thread = (call["pid"], call["tid"])
        if event == "exit":
            if call.get("status") != 0:
                if api == "cudaFree":
                    allocation = live.get((call["pid"], pointer(args.get("ptr"))))
                    if allocation is not None:
                        allocation.setdefault("failed_free_calls", []).append(call["call_id"])
                    gaps.append(_issue("failed-free-allocation-lifetime-unknown", call))
                continue
            if api == "cudaSetDevice":
                devices[process_thread] = args.get("device")
            elif api == "cudaGetDevice":
                devices[process_thread] = call.get("result", {}).get("device")
            elif api in SYNCHRONOUS_ALLOCATORS:
                base = pointer(call.get("result", {}).get("ptr"))
                size = integer(args.get("size"), "allocation size")
                if base + size > U64_MAX:
                    violations.append(_issue("allocation-address-overflow", call))
                    continue
                allocation_key = (call["pid"], base)
                generations[allocation_key] = generations.get(allocation_key, 0) + 1
                allocation = {"pid": call["pid"], "base": hex(base), "bytes": size,
                              "generation": generations[allocation_key], "allocate_call_id": call["call_id"],
                              "allocate_exit_seq": seq, "allocate_tid": call["tid"],
                              "device": _device(call) if _device(call) is not None else devices.get(process_thread),
                              "context": hex(_context(call)) if _context(call) is not None else None,
                              "allocator": api, "free_exit_seq": None}
                budget.consume(len(live))
                for other in live.values():
                    other_base = pointer(other["base"])
                    if other["pid"] == call["pid"] and base < other_base + other["bytes"] and other_base < base + size:
                        violations.append(_issue("overlapping-live-allocation-records", call,
                                                 other_allocate_call_id=other["allocate_call_id"]))
                live[allocation_key] = allocation
                history.append(allocation)
            elif api == "cudaFree":
                base = pointer(args.get("ptr"))
                if base:
                    allocation = live.pop((call["pid"], base), None)
                    if allocation is None:
                        gaps.append(_issue("free-of-untracked-allocation", call, pointer=hex(base)))
                    else:
                        allocation.update(free_exit_seq=seq, free_call_id=call["call_id"], free_tid=call["tid"])
            elif api in ("cudaMallocAsync", "cudaFreeAsync", "cudaMallocPitch", "cuMemAlloc_v2", "cuMemFree_v2"):
                gaps.append(_issue("allocator-lifetime-model-unsupported", call))
            elif api in ("cublasCreate_v2", "cublasCreate"):
                value = call.get("result", {}).get("handle")
                if value is not None:
                    handles[(call["pid"], pointer(value))] = {"create_call_id": call["call_id"], "destroyed": False}
            elif api in ("cublasDestroy_v2", "cublasDestroy"):
                value = args.get("handle")
                if value is not None and (call["pid"], pointer(value)) in handles:
                    handles[(call["pid"], pointer(value))]["destroyed"] = True
            continue
        call["recorded_device"] = devices.get(process_thread)
        if not _is_gemm(api):
            continue
        gemms.append(call)
        call["contract_snapshot_seq"] = seq
        call["operand_contracts"] = {}
        if "Batched" in api and "StridedBatched" not in api:
            gaps.append(_issue("pointer-array-batched-gemm-unsupported", call))
            continue
        try:
            extents = matrix_extents(args)
        except ContractError as error:
            code = str(error)
            target = gaps if code.startswith("unsupported-") else violations
            target.append(_issue(code, call))
            continue
        contract_observations = [r.get("result", {}) for r in call["observations"] if r["api"] == "gemm_contract"]
        if len(contract_observations) != 1:
            gaps.append(_issue("missing-or-duplicate-gemm-contract-observation", call))
        observation = contract_observations[0] if len(contract_observations) == 1 else {}
        if observation.get("device_query_status") != 0:
            gaps.append(_issue("gemm-current-device-unresolved", call))
        if _context(call) is None:
            gaps.append(_issue("gemm-current-context-unresolved", call))
        handle_observation = observation.get("handle", {})
        call["handle_contract"] = dict(handle_observation)
        call["scalar_contract"] = {}
        for name in ("pointer_mode", "math_mode", "stream"):
            if handle_observation.get(name + "_query_status") != 0 or name not in handle_observation:
                gaps.append(_issue("handle-" + name.replace("_", "-") + "-unresolved", call))
        if handle_observation.get("stream_query_status") == 0 and "stream" in handle_observation:
            pointer(handle_observation["stream"])
        if handle_observation.get("pointer_mode_query_status") == 0:
            mode = handle_observation.get("pointer_mode")
            if mode not in (0, 1):
                gaps.append(_issue("unsupported-handle-pointer-mode", call))
            elif mode == 1:
                gaps.append(_issue("device-pointer-mode-scalar-allocation-contract-unresolved", call))
        if handle_observation.get("pointer_mode_query_status") == 0 and handle_observation.get("pointer_mode") == 0:
            if args.get("compute_type") in (68, 69):
                for scalar in ("alpha", "beta"):
                    values = observation.get("scalars", {}).get(scalar, {})
                    if not isinstance(values, dict):
                        raise EvidenceError("invalid scalar observation")
                    if values.get("host_read_bytes") == 4 and isinstance(values.get("bits_hex"), str):
                        try:
                            raw = bytes.fromhex(values["bits_hex"])
                        except ValueError:
                            raise EvidenceError("invalid scalar bits") from None
                        if len(raw) != 4:
                            raise EvidenceError("invalid scalar bit width")
                        bits = int.from_bytes(raw, "little")
                        value = struct.unpack("<f", raw)[0]
                        finite = bits & 0x7f800000 != 0x7f800000
                        call["scalar_contract"][scalar] = {"bits_hex": raw.hex(), "finite": finite,
                                                           "f32_value": value if finite else None,
                                                           "is_zero": bits & 0x7fffffff == 0}
                    else:
                        gaps.append(_issue("host-scalar-value-unresolved", call, scalar=scalar))
            else:
                gaps.append(_issue("scalar-compute-type-not-modeled", call))
        handle_value = args.get("handle")
        if handle_value is not None:
            handle_state = handles.get((call["pid"], pointer(handle_value)))
            if handle_state and handle_state["destroyed"]:
                violations.append(_issue("gemm-submitted-with-destroyed-handle", call))
            elif not handle_state:
                gaps.append(_issue("handle-creation-unobserved", call))
        for operand, extent in extents.items():
            address = pointer(args.get(operand))
            count = extent["bytes"]
            detail = dict(extent, pointer=hex(address), range_status="unknown", allocation_generation=None)
            call["operand_contracts"][operand] = detail
            if address + count > U64_MAX:
                violations.append(_issue("operand-address-overflow", call, operand=operand))
                detail["range_status"] = "out-of-bounds"
                continue
            if count == 0:
                detail["range_status"] = "zero-byte-contract"
                continue
            if address == 0:
                violations.append(_issue("null-nonempty-operand", call, operand=operand))
            budget.consume(len(live))
            allocations = [a for a in live.values() if a["pid"] == call["pid"] and
                           pointer(a["base"]) <= address < pointer(a["base"]) + a["bytes"]]
            if len(allocations) == 1:
                allocation = allocations[0]
                allocation_base = pointer(allocation["base"])
                detail.update(allocation_base=allocation["base"], allocation_bytes=allocation["bytes"],
                              allocation_generation=allocation["generation"],
                              allocate_call_id=allocation["allocate_call_id"], offset_bytes=address-allocation_base,
                              range_status="within-recorded-allocation")
                if address + count > allocation_base + allocation["bytes"]:
                    detail["range_status"] = "out-of-bounds"
                    violations.append(_issue("operand-exceeds-recorded-allocation", call, operand=operand))
                if allocation["allocate_tid"] != call["tid"]:
                    gaps.append(_issue("cross-thread-allocation-use-order-unproven", call, operand=operand))
                if allocation.get("failed_free_calls"):
                    detail["range_status"] = "recorded-envelope-lifetime-unknown-after-free-error"
                    gaps.append(_issue("operand-lifetime-after-failed-free-unresolved", call, operand=operand))
                budget.consume(len(free_intervals.get((call["pid"], allocation_base), [])))
                overlapping_free = [f for f in free_intervals.get((call["pid"], allocation_base), []) if
                                    f["enter_seq"] < seq < f.get("exit_seq", U64_MAX)]
                if overlapping_free:
                    gaps.append(_issue("operand-use-overlaps-host-free-interval", call, operand=operand))
                device = _device(call)
                if allocation["device"] is not None and device is not None and allocation["device"] != device:
                    gaps.append(_issue("operand-allocation-device-differs-from-gemm-device", call, operand=operand,
                                       allocation_device=allocation["device"], gemm_device=device))
                if allocation["context"] is not None and _context(call) is not None and pointer(allocation["context"]) != _context(call):
                    gaps.append(_issue("operand-allocation-context-differs-from-gemm-context", call, operand=operand,
                                       allocation_context=allocation["context"], gemm_context=hex(_context(call))))
            else:
                budget.consume(len(history))
                freed = [a for a in history if a["pid"] == call["pid"] and a["free_exit_seq"] is not None and
                         pointer(a["base"]) <= address < pointer(a["base"]) + a["bytes"]]
                if freed:
                    detail["range_status"] = "previously-freed-untracked-at-submit"
                    violations.append(_issue("operand-in-freed-recorded-allocation", call, operand=operand,
                                             generation=freed[-1]["generation"]))
                else:
                    gaps.append(_issue("operand-allocation-unobserved", call, operand=operand))
            queried = observation.get("operands", {}).get(operand, {})
            detail["pointer_query"] = queried
            if queried.get("attributes_status") != 0:
                gaps.append(_issue("operand-pointer-attributes-unresolved", call, operand=operand))
            if queried.get("range_status") == 0 and "base" in queried and "bytes" in queried:
                base = pointer(queried["base"])
                length = integer(queried["bytes"], "queried allocation bytes")
                if base + length > U64_MAX or not (base <= address and address + count <= base + length):
                    violations.append(_issue("operand-exceeds-queried-address-range", call, operand=operand))
                    detail["query_range_status"] = "out-of-bounds"
                else:
                    detail["query_range_status"] = "within-queried-range-at-observation"
                if len(allocations) == 1 and (base != pointer(allocations[0]["base"]) or length < allocations[0]["bytes"]):
                    gaps.append(_issue("allocator-and-driver-range-observations-differ", call, operand=operand))
            else:
                gaps.append(_issue("operand-driver-address-range-unresolved", call, operand=operand))
        # Detect C/input overlap only as an unsupported alias contract, not a
        # proof that every cuBLAS path must fail. A and B may legitimately alias.
        for operand in ("A", "B"):
            a, c = call["operand_contracts"].get(operand), call["operand_contracts"].get("C")
            if a and c and a["bytes"] and c["bytes"]:
                if pointer(a["pointer"]) < pointer(c["pointer"]) + c["bytes"] and pointer(c["pointer"]) < pointer(a["pointer"]) + a["bytes"]:
                    if a["envelope_is_contiguous"] and c["envelope_is_contiguous"]:
                        violations.append(_issue("input-output-address-envelopes-overlap", call, operand=operand))
                    else:
                        gaps.append(_issue("padded-input-output-envelopes-overlap-exact-access-intersection-unresolved", call, operand=operand))

    # Host successful device synchronization can support completion only for
    # the same observed device and submitting thread. Events/streams require a
    # fuller dependency model; recording them alone does not establish an edge.
    successful_syncs = [c for c in calls if c["api"] == "cudaDeviceSynchronize" and c.get("status") == 0]
    allocation_by_call = {(a["pid"], a["allocate_call_id"]): a for a in history}
    sync_index = {}
    for sync in successful_syncs:
        key = (sync["pid"], sync["tid"], _device(sync), _context(sync))
        sync_index.setdefault(key, []).append(sync)
    sync_sequences = {key: [s["enter_seq"] for s in group] for key, group in sync_index.items()}
    for call in gemms:
        device = _device(call)
        context = _context(call)
        key = (call["pid"], call["tid"], device, context)
        matching = sync_index.get(key, []) if device is not None and context is not None and call.get("status") == 0 and call.get("library_forward_seq") is not None else []
        index = bisect.bisect_right(sync_sequences.get(key, []) if matching else [], call.get("exit_seq", U64_MAX))
        call["completion_evidence"] = {"status": "unproven"}
        if index < len(matching):
            sync = matching[index]
            call["completion_evidence"] = {"status": "successful-recorded-device-synchronization",
                                            "call_id": sync["call_id"], "exit_seq": sync["exit_seq"], "device": device}
        else:
            gaps.append(_issue("gemm-completion-not-established", call))
        if call.get("status") == 0:
            call["return_interpretation"] = "host-library-return-success-not-gpu-completion"
        for operand, detail in call.get("operand_contracts", {}).items():
            allocation = allocation_by_call.get((call["pid"], detail.get("allocate_call_id")))
            if allocation and allocation["free_exit_seq"] is not None:
                completion_seq = call["completion_evidence"].get("exit_seq")
                if completion_seq is None or allocation["free_exit_seq"] < completion_seq:
                    gaps.append(_issue("free-before-explicit-completion-evidence", call, operand=operand,
                                       free_call_id=allocation["free_call_id"]))
    tids = {c["tid"] for c in calls}
    if len(tids) > 1:
        gaps.append(_issue("cross-thread-or-cross-stream-dependencies-not-fully-modeled"))
    if any(c["api"] in ("cudaStreamSynchronize", "cudaEventSynchronize", "cudaStreamWaitEvent",
                          "cudaEventRecord", "cublasSetStream_v2", "cublasSetStream") for c in calls):
        gaps.append(_issue("event-and-stream-edge-reconstruction-not-implemented"))

    conversions = analyze_conversions(calls, history, violations, gaps, budget)

    halves = [c for c in gemms if _shape(c, 256, 103)]
    defaults = [c for c in gemms if _shape(c, 512, -1)]
    completed_half_sequences = {}
    for half in halves:
        completion_seq = half.get("completion_evidence", {}).get("exit_seq")
        if half.get("status") == 0 and completion_seq is not None:
            completed_half_sequences.setdefault((half["pid"], half["tid"], _device(half), _context(half)), []).append(completion_seq)
    for sequences in completed_half_sequences.values():
        sequences.sort()
    transitions = []
    last_default, process_gemm_ordinal, prior_half_counts = {}, {}, {}
    for half in gemms:
        thread_key = (half["pid"], half["tid"], _device(half), _context(half))
        ordinal = process_gemm_ordinal.get(half["pid"], 0)
        process_gemm_ordinal[half["pid"]] = ordinal + 1
        if _shape(half, 512, -1):
            completed_count = bisect.bisect_left(completed_half_sequences.get(thread_key, []), half["library_forward_seq"])
            last_default[thread_key] = (half, ordinal, prior_half_counts.get(thread_key, 0), completed_count)
        if _shape(half, 256, 103):
            prior_half_counts[thread_key] = prior_half_counts.get(thread_key, 0) + 1
        if not _shape(half, 256, 103) or thread_key not in last_default:
            continue
        full, prior_ordinal, before_default_count, completed_count = last_default[thread_key]
        completion = full.get("completion_evidence", {})
        transitions.append({"pid": half["pid"], "default512_call_id": full["call_id"],
                            "tid": half["tid"], "device": _device(half),
                            "context": hex(_context(half)) if _context(half) is not None else None,
                            "algorithm103_half256_call_id": half["call_id"],
                            "intervening_gemm_calls": ordinal-prior_ordinal-1,
                            "observed_production103_half_calls_before_default512": before_default_count,
                            "observed_completed_production103_half_calls_before_default512": completed_count,
                            "default512_observations": _production_observations(full),
                            "half256_observations": _production_observations(half),
                            "default512_successful_return": full.get("status") == 0,
                            "default512_device_sync_completed_before_half_submit":
                            completion.get("exit_seq", U64_MAX) < half["library_forward_seq"],
                            "submit_field_meaning": "recorded-host-forward-marker-not-GPU-execution-start"})
    if not halves:
        gaps.append(_issue("expected-production103-half256-not-observed"))
    if not defaults:
        gaps.append(_issue("expected-default512-not-observed"))
    if not transitions:
        gaps.append(_issue("expected-default512-to-production103-half256-transition-not-observed"))
    structural_codes = {"unterminated-final-record", "partial-final-record", "noncontiguous-sequence",
                        "missing-or-duplicate-trace-start", "missing-or-duplicate-trace-end", "empty-trace",
                        "missing-api-exit", "orphan-api-exit", "orphan-observation", "observation-outside-api-interval",
                        "metadata-does-not-bracket-records", "unqualified-records-after-trace-finalizer"}
    structurally_complete = not any(g["code"] in structural_codes for g in gaps)
    api_counts = {}
    for call in calls:
        api_counts[call["api"]] = api_counts.get(call["api"], 0) + 1
    kernel_calls = sum(count for api, count in api_counts.items() if api in ("cudaLaunchKernel", "cudaLaunchKernelExC", "__cudaLaunchKernel"))
    allocation_calls = sum(api_counts.get(api, 0) for api in SYNCHRONOUS_ALLOCATORS)
    required_capture_present = bool(starts and ends and len(pids) == 1 and gemms and kernel_calls and allocation_calls)
    qualified_transitions = [t for t in transitions if t["default512_successful_return"] and
                             t["default512_device_sync_completed_before_half_submit"] and
                             t["default512_observations"]["expected_scalar_handle_contract_observed"] and
                             t["half256_observations"]["expected_scalar_handle_contract_observed"]]
    suffix_candidates = [t for t in qualified_transitions if t["intervening_gemm_calls"] == 0 and
                         t["observed_completed_production103_half_calls_before_default512"] >= 1024]
    if not qualified_transitions:
        gaps.append(_issue("expected-forwarded-scalar-and-synchronization-transition-unresolved"))
    if not kernel_calls:
        gaps.append(_issue("application-kernel-launch-coverage-missing"))
    if not allocation_calls:
        gaps.append(_issue("application-allocation-coverage-missing"))
    gaps.append(_issue("interposition-does-not-cover-library-internal-kernels-or-all-driver-apis"))
    return {"schema": "ds4-runtime-contract-analysis-v1", "source_sha256": digest, "source_bytes": total,
            "records": len(records), "process_ids": pids, "metadata": starts + ends,
            "api_counts": api_counts, "required_capture_present": required_capture_present,
            "accepted_late_cuda_module_finalizers": accepted_late_finalizers,
            "status": "violations-observed" if violations else ("incomplete" if not structurally_complete else "observed-contract-with-gaps"),
            "trace_complete": structurally_complete,
            "trace_complete_meaning": "record-envelope-completeness-only-not-total-API-or-GPU-coverage",
            "violations": violations, "gaps": gaps, "api_errors": errors, "open_calls": open_calls,
            "allocation_history": history, "calls": calls, "conversion_contracts": conversions,
            "expected_transition": {"production103_half256_calls": len(halves), "default512_calls": len(defaults),
                                    "matching_meaning": "explicit-operation-dimensions-leading-dimensions-dtypes-compute-and-algorithm",
                                    "expected_transition_required_present": bool(qualified_transitions),
                                    "postburnin_transition_required_present": bool(suffix_candidates),
                                    "post_1024_observed_half_calls_adjacent_transition_candidates": suffix_candidates,
                                    "post_burnin_candidate_meaning": "recorded-count-and-API-order-pattern-not-application-phase-or-numerical-proof",
                                    "observed_transitions": transitions},
            "limitations": ["No conclusion of race freedom, driver correctness, hardware health or numerical exactness.",
                            "Host timestamps and API returns are not GPU execution/completion timestamps.",
                            "Allocation envelopes are recorded caller contracts, not bounds on proprietary kernel accesses.",
                            "Interposition and observation queries change timing; this is not the untouched baseline."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args(argv)
    try:
        report = analyze_file(args.trace)
    except (EvidenceError, OSError) as error:
        print(json.dumps({"schema": "ds4-runtime-contract-analysis-v1", "status": "invalid-evidence",
                          "trace_complete": False, "error": str(error)}))
        return 2
    json.dump(report, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
