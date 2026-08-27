#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARIZER = ROOT / "speed-bench" / "summarize-sm75-native-q4-t256-profile.py"


FIELDS = [
    "trace", "kind", "start_ns", "end_ns", "duration_ns", "device",
    "stream", "bytes", "name", "stage_range", "layer_range",
    "handoff_range", "partner_range", "attention_rows_range", "wave_range",
    "embedding_range", "output_range",
]


def operation(**values: object) -> dict[str, object]:
    row = {field: "" for field in FIELDS}
    row.update(
        trace="combined", kind="kernel", start_ns=0, end_ns=100,
        duration_ns=100, device=0, stream=0, bytes=0,
        name="synthetic",
    )
    row.update(values)
    return row


def main() -> int:
    schema = subprocess.run(
        [sys.executable, str(SUMMARIZER), "--schema"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert schema.stdout.strip() == "post-row-split-v2"
    with tempfile.TemporaryDirectory(prefix="ds4-combined-profile-test-") as temp:
        output = Path(temp)
        (output / "nsys").mkdir()
        rows = [
            operation(name="moe_gate_up_mid_sm75_native_q4_tile8_kernel<512>"),
            operation(name="moe_gate_up_mid_iq2_tile16_mma_sm75_kernel<512>"),
            operation(name="moe_gate_up_mid_expert_tile4_row32_kernel"),
            operation(name="moe_down_sm75_native_q4_tile_kernel<512,16>"),
            operation(name="matmul_q8_0_mma_sm75_exact_kernel<256>"),
            operation(name="attention_exact_kernel"),
            operation(name="turing_s1688gemm_fp16_256x128_ldg8_tn"),
            operation(name="indexer_scores_wmma128_kernel"),
            operation(name="q8_K_quantize_kernel"),
            operation(name="hc_expand_kernel"),
            operation(name="rope_tail_kernel"),
            operation(name="compressor_set_rows_kernel"),
            operation(name="add_kernel"),
            operation(name="unclassified_synthetic_kernel"),
            operation(
                kind="memcpy",
                name="memcpy",
                attention_rows_range=(
                    "ds4/prefill/attention-rows/kind=indexed/layer=2/"
                    "pos=0/tokens=512/home_tier=0/partner_tier=2/"
                    "home_rows=256/partner_rows=256"
                ),
                bytes=32 * 1024 * 1024,
            ),
        ]
        partner = (
            "ds4/q8/partner/label=attn_output_b/"
            "home_tier=0/home_device=0/partner_tier=2/partner_device=1/"
            "tokens=512/in=8192/out=4096/result=f32/arithmetic=f16"
        )
        for _layer in range(1, 43, 2):
            for _microbatch in range(4):
                rows.extend(
                    [
                        operation(
                            name="f32_to_f16_kernel", partner_range=partner,
                        ),
                        operation(
                            name="volta_h884gemm", partner_range=partner,
                            device=1,
                        ),
                        operation(
                            kind="memcpy", name="memcpy", partner_range=partner,
                            bytes=8 * 1024 * 1024,
                        ),
                        operation(
                            kind="memcpy", name="memcpy", partner_range=partner,
                            bytes=8 * 1024 * 1024,
                        ),
                    ]
                )
        with (output / "operation-attribution.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with (output / "nsys" / "combined-benchmark.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["ctx_tokens", "prefill_tokens", "prefill_tps"]
            )
            writer.writeheader()
            writer.writerow(
                {"ctx_tokens": 2048, "prefill_tokens": 2048, "prefill_tps": 541.0}
            )
        binding_fields = [
            "consumer_device", "resident_device", "partner_offload",
            "in_dim", "out_dim", "partner_arithmetic", "label",
        ]
        with (output / "nsys" / "bindings.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=binding_fields)
            writer.writeheader()
            for layer in range(43):
                partner_offload = int(layer % 2 == 1)
                writer.writerow(
                    {
                        "consumer_device": 0,
                        "resident_device": partner_offload,
                        "partner_offload": partner_offload,
                        "in_dim": 8192,
                        "out_dim": 4096,
                        "partner_arithmetic": "f16",
                        "label": f"tensor:blk.{layer}.attn_output_b.weight",
                    }
                )

        subprocess.run([sys.executable, str(SUMMARIZER), str(output)], check=True)
        evidence = json.loads((output / "combined-profile.json").read_text())
        assert evidence["accepted"] is True
        assert evidence["partner_t256_binding_count"] == 21
        assert evidence["partner_t256_projection_count"] == 84
        assert evidence["groups"]["partner_t256_cublas"]["operations"] == 84
        assert evidence["groups"]["partner_t256_memcpy"]["operations"] == 168
        assert evidence["groups"]["iq2_gate_up"]["operations"] == 2
        assert evidence["groups"]["local_fp16_gemm"]["operations"] == 1
        assert evidence["groups"]["indexer"]["operations"] == 1
        assert evidence["groups"]["format_quant_pack"]["operations"] == 1
        assert evidence["groups"]["norm_hyperconnection"]["operations"] == 1
        assert evidence["groups"]["attention_row_split_memcpy"]["operations"] == 1
        assert evidence["groups"]["other_unknown"]["operations"] == 1
        assert (output / "combined-kernel-groups.csv").stat().st_size > 0
        assert (output / "kernel-name-groups.csv").stat().st_size > 0
        assert (output / "kernel-groups-device-stage.csv").stat().st_size > 0
        assert (output / "unknown-kernels.csv").stat().st_size > 0
        assert (output / "partner-t256-ranges.csv").stat().st_size > 0
    print("combined native-Q4/T256 profile summarizer test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
