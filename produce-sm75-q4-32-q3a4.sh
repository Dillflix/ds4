#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Produce and benchmark an SM75-native routed-expert model whose only expert
placements are:
  down:    Q4-32
  gate/up: Q4-32 or Q3A4 (always matched within a layer)

Usage:
  Q3A4_LAYERS=CSV bash produce-sm75-q4-32-q3a4.sh \
    HF_DIR [OUTPUT.gguf] [RESULTS.csv]

Q3A4_LAYERS is an explicit layer allocation. It accepts:
  none            all routed gate/up tensors use Q4-32 (default)
  all             all routed gate/up tensors use Q3A4
  3,7,12-18,42    only those layers use Q3A4; all others use Q4-32

The wrapper deliberately has no guessed mixed allocation. Use the quality and
capacity policy to choose Q3A4_LAYERS, then record that exact choice with the
model. All ordinary production controls accepted by
produce-benchmark-iq2-iq2-q4.sh remain available, including QUANTIZE_ONLY,
OVERWRITE, REUSE_MODEL, GPU_DEVICES, CTX_MAX, and DS4_TEMPLATE_GGUF.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

[[ ${1:-} != -h && ${1:-} != --help ]] || {
    usage
    exit 0
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
q3a4_spec=${Q3A4_LAYERS:-none}
gate_up_default=sm75_q4_32
overrides=()
normalized=none

if [[ $q3a4_spec == all ]]; then
    gate_up_default=sm75_q3a4
    normalized=all
elif [[ $q3a4_spec != none ]]; then
    declare -A selected=()
    IFS=',' read -r -a entries <<< "$q3a4_spec"
    for entry in "${entries[@]}"; do
        [[ -n $entry ]] || die "Q3A4_LAYERS contains an empty entry"
        if [[ $entry =~ ^([0-9]+)-([0-9]+)$ ]]; then
            first=$((10#${BASH_REMATCH[1]}))
            last=$((10#${BASH_REMATCH[2]}))
            (( first <= last )) || die "descending Q3A4_LAYERS range: $entry"
            for ((layer = first; layer <= last; layer++)); do
                (( layer <= 42 )) || die "Q3A4 layer is outside 0..42: $layer"
                selected[$layer]=1
            done
        elif [[ $entry =~ ^[0-9]+$ ]]; then
            layer=$((10#$entry))
            (( layer <= 42 )) || die "Q3A4 layer is outside 0..42: $layer"
            selected[$layer]=1
        else
            die "bad Q3A4_LAYERS entry: $entry"
        fi
    done
    mapfile -t layers < <(printf '%s\n' "${!selected[@]}" | sort -n)
    normalized=$(IFS=,; printf '%s' "${layers[*]}")
    for layer in "${layers[@]}"; do
        overrides+=("blk.$layer.ffn_gate_exps.weight=sm75_q3a4")
        overrides+=("blk.$layer.ffn_up_exps.weight=sm75_q3a4")
    done
fi

override_csv=
if ((${#overrides[@]})); then
    override_csv=$(IFS=,; printf '%s' "${overrides[*]}")
fi
if [[ -n ${TENSOR_TYPE_OVERRIDES:-} ]]; then
    die "TENSOR_TYPE_OVERRIDES is owned by this wrapper; use Q3A4_LAYERS"
fi
if [[ -n ${ROUTED_W1:-} || -n ${ROUTED_W2:-} || -n ${ROUTED_W3:-} ]]; then
    die "ROUTED_W1/W2/W3 are fixed by this production recipe"
fi
if [[ ${SM75_NATIVE_Q4:-0} != 0 ]]; then
    die "SM75_NATIVE_Q4 is a different tagged Q4_K layout and is not part of this recipe"
fi

out_default=$PWD/gguf/DeepSeek-V4-Flash-0731-SM75-Q4-32-Q3A4.gguf
out_arg=${2:-${OUT:-$out_default}}
recipe="gate/up:$gate_up_default,q3a4_layers:$normalized,down:sm75_q4_32"

ROUTED_W1=$gate_up_default \
ROUTED_W2=sm75_q4_32 \
ROUTED_W3=$gate_up_default \
TENSOR_TYPE_OVERRIDES=$override_csv \
SM75_NATIVE_Q4=0 \
QUANT_RECIPE=$recipe \
PLOT_TITLE="RTX 8000 2x2 TP - SM75 Q4-32/Q3A4 experts" \
bash "$script_dir/produce-benchmark-iq2-iq2-q4.sh" \
    "${1:?HF_DIR is required}" "$out_arg" "${3:-}"
