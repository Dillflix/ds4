#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Audit bounded software-prefetch depths for the production SM75 Q3A4
tile32-DP4A K4 gate/up kernel. The shipping K4 path is the depth-0 control;
depths 1 and 2 are candidates. Q4-32 and every multi-GPU boundary are unchanged.

Optional environment:
  PROFILE_GPU=0
  CUDA_ARCH=sm_75
  TIMING_ROUNDS=9
  TIMING_REPEATS=25
  RUN_SANITIZER=1
  RUN_NCU=1
  NCU_USE_SUDO=0
  SKIP_BUILD=0
  RESUME=0
  CREATE_ARCHIVE=1
  Q3A4_PREFETCH_DIR=/absolute/output/directory
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
(( $# == 0 )) || die "this script takes no positional arguments"

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

PROFILE_GPU=${PROFILE_GPU:-0}
CUDA_ARCH=${CUDA_ARCH:-sm_75}
TIMING_ROUNDS=${TIMING_ROUNDS:-9}
TIMING_REPEATS=${TIMING_REPEATS:-25}
RUN_SANITIZER=${RUN_SANITIZER:-1}
RUN_NCU=${RUN_NCU:-1}
NCU_USE_SUDO=${NCU_USE_SUDO:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
RESUME=${RESUME:-0}
CREATE_ARCHIVE=${CREATE_ARCHIVE:-1}
OUTPUT_DIR=${Q3A4_PREFETCH_DIR:-$repo_dir/sm75-decode-q3a4-prefetch-$(date -u +%Y%m%dT%H%M%SZ)}

[[ $PROFILE_GPU =~ ^[0-9]+$ ]] || die "PROFILE_GPU must be a physical GPU index"
[[ $TIMING_ROUNDS =~ ^[1-9][0-9]*$ ]] && (( TIMING_ROUNDS % 2 == 1 )) ||
    die "TIMING_ROUNDS must be a positive odd integer"
[[ $TIMING_REPEATS =~ ^[1-9][0-9]*$ ]] || die "TIMING_REPEATS must be positive"
for value in "$RUN_SANITIZER" "$RUN_NCU" "$NCU_USE_SUDO" \
             "$SKIP_BUILD" "$RESUME" "$CREATE_ARCHIVE"; do
    [[ $value == 0 || $value == 1 ]] || die "binary options must be 0 or 1"
done

tools=(awk basename cat c++filt cmp cuobjdump cut date dirname env git grep id
       make mkdir mktemp mv nproc nvidia-smi python3 rm sha256sum sort tail tar
       tr)
(( RUN_NCU == 0 )) || tools+=(ncu)
for tool in "${tools[@]}"; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
if [[ $RUN_SANITIZER == 1 ]]; then
    command -v compute-sanitizer >/dev/null 2>&1 ||
        die "RUN_SANITIZER=1 but compute-sanitizer was not found"
fi

compute_cap=$(nvidia-smi -i "$PROFILE_GPU" --query-gpu=compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')
[[ $compute_cap == 7.5 ]] ||
    die "physical GPU $PROFILE_GPU has compute capability ${compute_cap:-unknown}; SM75 is required"

if [[ $RESUME == 1 ]]; then
    [[ -n ${Q3A4_PREFETCH_DIR:-} ]] ||
        die "RESUME=1 requires Q3A4_PREFETCH_DIR"
    [[ $SKIP_BUILD == 1 ]] || die "RESUME=1 requires SKIP_BUILD=1"
    [[ $RUN_NCU == 1 ]] || die "RESUME=1 requires RUN_NCU=1"
    [[ -d $OUTPUT_DIR ]] || die "resume directory does not exist: $OUTPUT_DIR"
    [[ -d $OUTPUT_DIR/ncu ]] || die "resume directory is missing its ncu subdirectory"
else
    [[ ! -e $OUTPUT_DIR ]] || die "output path already exists: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/smoke" "$OUTPUT_DIR/timing" "$OUTPUT_DIR/ncu"
fi
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

current_phase=initialization
finalize() {
    local status=$?
    trap - EXIT INT TERM HUP
    if [[ -d $OUTPUT_DIR ]]; then
        printf 'state=%s\nexit_status=%s\nlast_phase=%s\ndate_utc=%s\n' \
            "$([[ $status == 0 ]] && printf finished || printf failed)" \
            "$status" "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >"$OUTPUT_DIR/run-status.txt"
        if [[ $CREATE_ARCHIVE == 1 ]]; then
            local archive="$OUTPUT_DIR.tar.gz"
            local partial_archive="$archive.partial.$$"
            if tar -C "$(dirname "$OUTPUT_DIR")" -czf "$partial_archive" \
                    "$(basename "$OUTPUT_DIR")" &&
               mv -f -- "$partial_archive" "$archive"; then
                printf 'Archive to return: %s\n' "$archive"
            else
                status=1
                rm -f -- "$partial_archive"
                printf 'state=failed\nexit_status=1\nlast_phase=%s\ndate_utc=%s\n' \
                    "$current_phase" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                    >"$OUTPUT_DIR/run-status.txt"
            fi
        fi
    fi
    exit "$status"
}
trap finalize EXIT
trap 'current_phase=interrupted; exit 130' INT TERM HUP

write_manifest() {
    [[ $RESUME == 0 ]] || printf '\n[resume]\n'
    printf 'date_utc=%s\nrepo=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$repo_dir"
    printf 'git_commit=%s\ngit_branch=%s\n' \
        "$(git rev-parse HEAD)" "$(git branch --show-current)"
    printf 'scope=q3a4-k4-prefetch-depth-0-1-2\n'
    printf 'baseline=q3a4-tile32-dp4a-k4\nq4_32=unchanged\n'
    printf 'profile_gpu=%s\ncuda_arch=%s\n' "$PROFILE_GPU" "$CUDA_ARCH"
    printf 'timing_rounds=%s\ntiming_repeats=%s\n' \
        "$TIMING_ROUNDS" "$TIMING_REPEATS"
    printf 'run_sanitizer=%s\nrun_ncu=%s\nresume=%s\n' \
        "$RUN_SANITIZER" "$RUN_NCU" "$RESUME"
    printf '\n[gpu]\n'
    nvidia-smi -i "$PROFILE_GPU" \
        --query-gpu=index,name,pci.bus_id,memory.total,memory.free,driver_version \
        --format=csv
    printf '\n[git status]\n'; git status --short
}
if [[ $RESUME == 1 ]]; then
    write_manifest >>"$OUTPUT_DIR/manifest.txt"
else
    write_manifest >"$OUTPUT_DIR/manifest.txt"
fi

smoke=tests/cuda_long_context_smoke
harness=tests/cuda_sm75_decode_weight_profile
exact_marker='SM75 Q3A4 tile32-dp4a K4 prefetch-depth 1/2 nonzero exact'
declare -A scenario=(
    [baseline]=q3a4-tile32-dp4a-k4
    [prefetch1]=q3a4-tile32-dp4a-k4-prefetch1
    [prefetch2]=q3a4-tile32-dp4a-k4-prefetch2
)
declare -A depth=([baseline]=0 [prefetch1]=1 [prefetch2]=2)
variants=(baseline prefetch1 prefetch2)
candidates=(prefetch1 prefetch2)

validate_prefetch_audit() {
    local variant=$1 log=$2 mode=$3 expected_depth=${depth[$1]}
    awk -v expected_depth="$expected_depth" -v mode="$mode" '
        /SM75 Q3A4 decode mapping audit/ {
            seen++
            delete value
            for (i=1; i<=NF; i++) {
                split($i, item, "=")
                value[item[1]]=item[2]+0
            }
            total=value["tile32-dp4a"]
            good=(value["control"]==0 && value["hwarp16"]==0 &&
                  value["tile32"]==0 && value["k1"]==0 &&
                  value["k2"]==0 && value["k4"]==total && total>0 &&
                  value["pf0"]+value["pf1"]+value["pf2"]==total)
            if (mode=="single") {
                good=good && total==1
                for (depth=0; depth<=2; depth++)
                    good=good && value["pf" depth]==(depth==expected_depth)
            } else {
                good=good && expected_depth>0 && value["pf0"]>0 &&
                     value["pf" expected_depth]>0
                for (depth=1; depth<=2; depth++)
                    if (depth!=expected_depth) good=good && value["pf" depth]==0
            }
        }
        END {exit !(seen==1 && good)}
    ' "$log"
}

validate_smoke_log() {
    local variant=$1 log=$2 expected_depth=${depth[$1]}
    grep -Fxq "scenario=${scenario[$variant]}" "$log" || return 1
    grep -Fxq 'output_validation=exact-zero' "$log" || return 1
    grep -Fxq 'q3a4_decode_mapping=3' "$log" || return 1
    grep -Fxq 'q3a4_decode_ksplit=4' "$log" || return 1
    grep -Fxq "q3a4_decode_prefetch_depth=$expected_depth" "$log" || return 1
    local marker="tile32-dp4a-k4"
    [[ $expected_depth == 0 ]] || marker="$marker-prefetch$expected_depth"
    [[ $expected_depth != 0 ]] || marker="$marker (production default)"
    grep -Fxq "ds4: SM75 Q3A4 decode gate/up mapping=$marker" "$log" ||
        return 1
    validate_prefetch_audit "$variant" "$log" single || return 1
    grep -Fxq 'harness_status=ok' "$log"
}

validate_timing_log() {
    local variant=$1 log=$2 expected="q3a4-tile32-dp4a-k4-${variant}"
    grep -Fxq 'timing_scope=production-owned-call-inclusive' "$log" || return 1
    grep -Fxq "candidate_kind=$expected" "$log" || return 1
    grep -Fxq "q3a4_decode_prefetch_depth=${depth[$variant]}" "$log" || return 1
    validate_prefetch_audit "$variant" "$log" paired
}

if [[ $RESUME == 1 ]]; then
    current_phase=resume-validation
    [[ -x $smoke && -x $harness ]] ||
        die "resume requires the previously built CUDA binaries"
    [[ -s $OUTPUT_DIR/build.log && -s $OUTPUT_DIR/build.demangled.log &&
       -s $OUTPUT_DIR/sass.demangled.txt &&
       -s $OUTPUT_DIR/resource-summary.csv &&
       -s $OUTPUT_DIR/timing-summary.csv &&
       -s $OUTPUT_DIR/harness-sha256.txt ]] ||
        die "resume directory is missing build/resource/timing evidence"
    exact_log="$OUTPUT_DIR/smoke/cuda-long-context.log"
    grep -Fq "$exact_marker" "$exact_log" ||
        die "resume nonzero exact-regression evidence is invalid"
    grep -Fq 'SM75 Q3A4 tile32-dp4a-k4 production default' "$exact_log" ||
        die "resume K4 production-default evidence is invalid"
    grep -Fxq 'cuda long-context regression: OK' "$exact_log" ||
        die "resume exact-regression completion marker is missing"
    grep -Fq 'ERROR SUMMARY: 0 errors' "$OUTPUT_DIR/memcheck.log" ||
        die "resume depth-2 memcheck evidence is not clean"
    for variant in "${variants[@]}"; do
        validate_smoke_log "$variant" "$OUTPUT_DIR/smoke/$variant.log" ||
            die "resume $variant smoke evidence is invalid"
    done
    for variant in "${candidates[@]}"; do
        validate_timing_log "$variant" "$OUTPUT_DIR/timing/$variant.log" ||
            die "resume $variant timing evidence is invalid"
    done

    evidence_commit=$(python3 - "$OUTPUT_DIR/manifest.txt" \
              "$OUTPUT_DIR/resource-summary.csv" \
              "$OUTPUT_DIR/timing-summary.csv" \
              "$PROFILE_GPU" "$CUDA_ARCH" <<'PY'
import csv, math, sys
manifest_path, resource_path, timing_path, profile_gpu, cuda_arch = sys.argv[1:]
initial = {}
with open(manifest_path, encoding="utf-8", errors="replace") as handle:
    for raw in handle:
        line = raw.strip()
        if line == "[resume]":
            break
        if "=" in line:
            key, value = line.split("=", 1)
            initial.setdefault(key, value)
required = {
    "scope": "q3a4-k4-prefetch-depth-0-1-2",
    "baseline": "q3a4-tile32-dp4a-k4",
    "q4_32": "unchanged",
    "profile_gpu": profile_gpu,
    "cuda_arch": cuda_arch,
    "run_sanitizer": "1",
}
for key, expected in required.items():
    if initial.get(key) != expected:
        raise SystemExit(
            f"resume provenance mismatch for {key}: "
            f"{initial.get(key)!r} != {expected!r}")
commit = initial.get("git_commit", "")
if not commit:
    raise SystemExit("resume manifest has no original git commit")
with open(resource_path, newline="", encoding="utf-8") as handle:
    resources = {row["variant"]: row for row in csv.DictReader(handle)}
if set(resources) != {"baseline", "prefetch1", "prefetch2"}:
    raise SystemExit("resume resource inventory is incomplete")
for variant, depth in (("baseline", 0), ("prefetch1", 1), ("prefetch2", 2)):
    row = resources[variant]
    if row.get("resource_gate") != "pass" or int(row["prefetch_depth"]) != depth:
        raise SystemExit(f"resume resource gate failed for {variant}")
    if int(row["block_size"]) != 512 or int(row["allocated_registers"]) > 64:
        raise SystemExit(f"resume launch/register gate failed for {variant}")
    if int(row["shared_memory_bytes"]) != 4096:
        raise SystemExit(f"resume shared-memory gate failed for {variant}")
    if any(int(row[field]) for field in (
            "stack_frame_bytes", "spill_store_bytes", "spill_load_bytes",
            "sass_ldl", "sass_stl")):
        raise SystemExit(f"resume spill/local evidence failed for {variant}")
with open(timing_path, newline="", encoding="utf-8") as handle:
    timings = {row["variant"]: row for row in csv.DictReader(handle)}
if set(timings) != {"prefetch1", "prefetch2"}:
    raise SystemExit("resume timing inventory is incomplete")
for variant, row in timings.items():
    values = [float(row[key]) for key in (
        "baseline_median_ms", "candidate_median_ms", "candidate_speedup")]
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise SystemExit(f"resume timing evidence is invalid for {variant}")
print(commit)
PY
    )
    git cat-file -e "$evidence_commit^{commit}" 2>/dev/null ||
        die "resume evidence commit is not present in this repository"
    git merge-base --is-ancestor "$evidence_commit" HEAD ||
        die "resume evidence commit is not an ancestor of the current checkout"
    evidence_sources=(
        Makefile
        ds4_cuda.cu
        ds4_cuda_sm75_q32_native.inc.cu
        tests/cuda_long_context_smoke.c
        tests/cuda_sm75_decode_weight_profile.c
    )
    git diff --quiet "$evidence_commit" -- "${evidence_sources[@]}" ||
        die "Q3A4 implementation or evidence sources changed since the reused run"
    sha256sum --check --status "$OUTPUT_DIR/harness-sha256.txt" ||
        die "current profile harness differs from the reused evidence binary"
    printf 'Reusing validated exactness, sanitizer, resource, and timing evidence.\n'
else
    current_phase=build
    if [[ $SKIP_BUILD == 0 ]]; then
        evidence_nvccflags=${EVIDENCE_NVCCFLAGS:-"-O3 -g -lineinfo --use_fast_math -arch=$CUDA_ARCH -Xcompiler -march=native -Xcompiler -pthread -Xptxas -v"}
        set +e
        make -B -j"$(nproc)" "$smoke" "$harness" CUDA_ARCH="$CUDA_ARCH" \
            NVCCFLAGS="$evidence_nvccflags" >"$OUTPUT_DIR/build.log" 2>&1
        build_rc=$?
        set -e
        if (( build_rc != 0 )); then
            tail -n 180 "$OUTPUT_DIR/build.log" >&2 || true
            die "SM75 Q3A4 prefetch evidence build failed"
        fi
    else
        die "SKIP_BUILD=1 cannot provide fresh PTXAS evidence; use SKIP_BUILD=0"
    fi
    [[ -x $smoke && -x $harness ]] || die "required CUDA binaries are missing"
    sha256sum "$harness" >"$OUTPUT_DIR/harness-sha256.txt"

    current_phase=byte-exact-regression
    printf 'Running nonzero byte-exact Q3A4 K4 prefetch regression...\n'
    env -u DS4_CUDA_MOE_Q32_DECODE_FUSED_LOWREG \
        -u DS4_CUDA_NO_MOE_Q32_DECODE_FUSED_LOWREG \
        -u DS4_CUDA_MOE_Q3A4_DECODE_MAPPING \
        -u DS4_CUDA_NO_MOE_Q3A4_DECODE_MAPPING \
        -u DS4_CUDA_MOE_Q3A4_DECODE_KSPLIT \
        -u DS4_CUDA_MOE_Q3A4_DECODE_PREFETCH \
        -u DS4_CUDA_MOE_Q3A4_DECODE_PREFETCH_DEPTH \
        CUDA_VISIBLE_DEVICES="$PROFILE_GPU" ./$smoke \
        >"$OUTPUT_DIR/smoke/cuda-long-context.log" 2>&1 || {
            tail -n 180 "$OUTPUT_DIR/smoke/cuda-long-context.log" >&2 || true
            die "byte-exact CUDA regression failed"
        }
    grep -Fq "$exact_marker" "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
        die "Q3A4 K4 prefetch nonzero exact marker missing"
    grep -Fq 'SM75 Q3A4 tile32-dp4a-k4 production default' \
        "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
        die "K4 production-default assertion missing"
    grep -Fxq 'cuda long-context regression: OK' \
        "$OUTPUT_DIR/smoke/cuda-long-context.log" ||
        die "byte-exact regression completion marker missing"

    for variant in "${variants[@]}"; do
        printf 'Synthetic owned-call smoke: %s...\n' "$variant"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            DS4_CUDA_MOE_Q3A4_DECODE_MAPPING_AUDIT=1 \
            ./$harness "${scenario[$variant]}" \
            >"$OUTPUT_DIR/smoke/$variant.log" 2>&1 || {
                tail -n 120 "$OUTPUT_DIR/smoke/$variant.log" >&2 || true
                die "$variant smoke failed"
            }
        validate_smoke_log "$variant" "$OUTPUT_DIR/smoke/$variant.log" ||
            die "$variant smoke validation failed"
    done

    if [[ $RUN_SANITIZER == 1 ]]; then
        current_phase=memcheck
        printf 'Compute Sanitizer: prefetch-depth 2 candidate...\n'
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            compute-sanitizer --tool memcheck --error-exitcode 3 \
            ./$harness "${scenario[prefetch2]}" \
            >"$OUTPUT_DIR/memcheck.log" 2>&1 || {
                tail -n 160 "$OUTPUT_DIR/memcheck.log" >&2 || true
                die "prefetch-depth 2 compute-sanitizer memcheck failed"
            }
        grep -Fq 'ERROR SUMMARY: 0 errors' "$OUTPUT_DIR/memcheck.log" ||
            die "compute-sanitizer did not report a clean error summary"
    else
        printf 'skipped explicitly: RUN_SANITIZER=0\n' >"$OUTPUT_DIR/memcheck.log"
    fi

    current_phase=resource-audit
    cuobjdump --list-elf "$harness" >"$OUTPUT_DIR/elf-list.txt" 2>&1
    grep -Eq '\.sm_75\.cubin([[:space:]]|$)' "$OUTPUT_DIR/elf-list.txt" ||
        die "decode harness does not contain an sm_75 cubin"
    cuobjdump --dump-sass "$harness" | c++filt \
        >"$OUTPUT_DIR/sass.demangled.txt" 2>&1
    c++filt <"$OUTPUT_DIR/build.log" >"$OUTPUT_DIR/build.demangled.log"

    python3 - "$OUTPUT_DIR/sass.demangled.txt" \
            "$OUTPUT_DIR/build.demangled.log" \
            "$OUTPUT_DIR/resource-summary.csv" <<'PY'
import csv, re, sys
sass_path, build_path, output_path = sys.argv[1:]

# Accept the two natural source spellings while keeping each specialization
# unique. Once demangled, integral template arguments may retain their `u`.
targets = {
    "baseline": (
        r"moe_gate_up_mid_decode_sm75_q3a4_tile32_ksplit_owned_kernel<[^>]*4[^>]*>",
        0),
    "prefetch1": (
        r"moe_gate_up_mid_decode_sm75_q3a4_tile32_k4_prefetch_owned_kernel<[^>]*1[^>]*>",
        1),
    "prefetch2": (
        r"moe_gate_up_mid_decode_sm75_q3a4_tile32_k4_prefetch_owned_kernel<[^>]*2[^>]*>",
        2),
}

def sections(path, marker, sass):
    result, current = {}, None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.search(marker, line)
            if match:
                current = match.group(1)
                result.setdefault(current, [] if sass else {})
                continue
            if current is None:
                continue
            if sass:
                result[current].append(line)
            else:
                match = re.search(
                    r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*"
                    r"(\d+) bytes spill loads", line)
                if match:
                    result[current].update(zip(
                        ("stack", "spill_stores", "spill_loads"),
                        map(int, match.groups())))
                match = re.search(r"Used\s+(\d+) registers", line)
                if match:
                    result[current]["registers"] = int(match.group(1))
                match = re.search(
                    r"(\d+) bytes (?:smem(?:\[\d+\])?|shared memory)", line)
                if match:
                    result[current]["shared_memory"] = int(match.group(1))
    return result

sass = sections(sass_path, r"Function\s*:\s*(.*\S)", True)
ptxas = sections(build_path, r"Function properties for\s+(.*\S)", False)
rows = []
for label, (expression, prefetch_depth) in targets.items():
    pattern = re.compile(expression)
    sm = [(name, body) for name, body in sass.items() if pattern.search(name)]
    pm = [(name, values) for name, values in ptxas.items() if pattern.search(name)]
    if len(sm) != 1 or len(pm) != 1:
        raise SystemExit(
            f"kernel identity failure for {label}: SASS={len(sm)} PTXAS={len(pm)}")
    sass_name, body = sm[0]
    ptxas_name, values = pm[0]
    missing = {
        "stack", "spill_stores", "spill_loads", "registers",
        "shared_memory",
    } - values.keys()
    if missing:
        raise SystemExit(f"incomplete PTXAS evidence for {label}: {sorted(missing)}")
    text = "".join(body)
    ldl = len(re.findall(r"\bLDL(?:\.|\b)", text))
    stl = len(re.findall(r"\bSTL(?:\.|\b)", text))
    idp4a = len(re.findall(r"\bIDP\.4A(?:\.|\b)", text))
    atom = len(re.findall(r"\b(?:ATOM|RED)(?:\.|\b)", text))
    registers = values["registers"]
    allocated = ((registers + 7) // 8) * 8
    if any((values["stack"], values["spill_stores"],
            values["spill_loads"], ldl, stl)):
        raise SystemExit(
            f"{label}: stack/spill/local traffic: stack={values['stack']} "
            f"spill_stores={values['spill_stores']} "
            f"spill_loads={values['spill_loads']} LDL={ldl} STL={stl}")
    if allocated > 64:
        raise SystemExit(
            f"{label}: allocated registers {allocated} exceed 64")
    if values["shared_memory"] != 4096:
        raise SystemExit(
            f"{label}: expected exactly 4096 bytes shared memory, "
            f"got {values['shared_memory']}")
    if idp4a == 0:
        raise SystemExit(f"{label}: SASS contains no IDP.4A instruction")
    if atom:
        raise SystemExit(f"{label}: unexpected atomic/reduction SASS count={atom}")
    rows.append({
        "variant": label, "prefetch_depth": prefetch_depth,
        "block_size": 512, "registers": registers,
        "allocated_registers": allocated, "register_limit": 64,
        "shared_memory_bytes": values["shared_memory"],
        "stack_frame_bytes": values["stack"],
        "spill_store_bytes": values["spill_stores"],
        "spill_load_bytes": values["spill_loads"],
        "sass_ldl": ldl, "sass_stl": stl, "sass_idp4a": idp4a,
        "sass_atom_red": atom, "resource_gate": "pass",
        "sass_symbol": sass_name, "ptxas_symbol": ptxas_name,
    })
with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print("validated K4 prefetch depth 0/1/2 resource identities and hard gates")
PY
    cat "$OUTPUT_DIR/resource-summary.csv"

    current_phase=inclusive-timing
    printf 'variant,baseline_median_ms,candidate_median_ms,candidate_speedup\n' \
        >"$OUTPUT_DIR/timing-summary.csv"
    for variant in "${candidates[@]}"; do
        log="$OUTPUT_DIR/timing/$variant.log"
        printf 'Inclusive production-owned-call timing: K4 vs %s...\n' "$variant"
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            DS4_CUDA_MOE_Q3A4_DECODE_MAPPING_AUDIT=1 \
            TIMING_ROUNDS="$TIMING_ROUNDS" TIMING_REPEATS="$TIMING_REPEATS" \
            ./$harness "${scenario[$variant]}-ab" >"$log" 2>&1 || {
                tail -n 120 "$log" >&2 || true
                die "$variant inclusive timing failed"
            }
        validate_timing_log "$variant" "$log" ||
            die "$variant timing validation failed"
        control=$(grep '^control_median_ms=' "$log" | cut -d= -f2)
        candidate=$(grep '^candidate_median_ms=' "$log" | cut -d= -f2)
        speedup=$(grep '^candidate_speedup=' "$log" | cut -d= -f2)
        [[ -n $control && -n $candidate && -n $speedup ]] ||
            die "$variant timing summary is incomplete"
        printf '%s,%s,%s,%s\n' "$variant" "$control" "$candidate" "$speedup" \
            >>"$OUTPUT_DIR/timing-summary.csv"
    done
    cat "$OUTPUT_DIR/timing-summary.csv"
fi

if [[ $RUN_NCU == 1 ]]; then
    current_phase=nsight-compute
    ncu_bin=$(command -v ncu)
    ncu_command=("$ncu_bin")
    if [[ $NCU_USE_SUDO == 1 ]]; then
        command -v sudo >/dev/null 2>&1 || die "sudo not found"
        sudo -v
        ncu_command=(sudo -E "$ncu_bin")
    fi
    metrics=(
        gpu__time_duration.sum
        dram__bytes.sum
        dram__bytes.sum.per_second
        smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
        smsp__warps_eligible.avg.per_cycle_active
        launch__block_size
        launch__registers_per_thread
        launch__shared_mem_per_block
    )
    printf '%s\n' "${metrics[@]}" >"$OUTPUT_DIR/ncu/metrics-required.txt"
    metric_csv=$(IFS=,; printf '%s' "${metrics[*]}")

    for variant in "${variants[@]}"; do
        case "$variant" in
            baseline)
                regex='moe_gate_up_mid_decode_sm75_q3a4_tile32_ksplit_owned_kernel.*'
                ;;
            prefetch1|prefetch2)
                regex='moe_gate_up_mid_decode_sm75_q3a4_tile32_k4_prefetch_owned_kernel.*'
                ;;
        esac
        base="$OUTPUT_DIR/ncu/$variant"
        printf 'Nsight Compute: %s...\n' "$variant"
        rc=0
        env CUDA_VISIBLE_DEVICES="$PROFILE_GPU" \
            "${ncu_command[@]}" --config-file off \
                --target-processes application-only --devices 0 \
                --kernel-name-base function --kernel-name "regex:$regex" \
                --launch-count 1 --replay-mode kernel --cache-control all \
                --clock-control none --metrics "$metric_csv" \
                --disable-extra-suffixes --force-overwrite --export "$base" \
                ./$harness "${scenario[$variant]}" >"$base.log" 2>&1 || rc=$?
        if (( rc != 0 )) ||
           grep -Eq '==ERROR==|No kernels were profiled|Failed to (profile|create report)' \
               "$base.log"; then
            tail -n 120 "$base.log" >&2 || true
            die "Nsight Compute failed for $variant"
        fi
        [[ -s $base.ncu-rep ]] || die "Nsight report missing for $variant"
        if [[ $NCU_USE_SUDO == 1 ]]; then
            sudo chown -- "$(id -u):$(id -g)" "$base.ncu-rep"
        fi
        "$ncu_bin" --config-file off --import "$base.ncu-rep" --csv --page raw \
            >"$base.csv" 2>"$base-import.log" ||
            die "Nsight import failed for $variant"
        python3 speed-bench/validate-ncu-capture.py "$base.csv" "$regex" 0 \
            --process cuda_sm75_decode_weight_profile --block-size 512 \
            >"$base-validation.txt" 2>&1 || {
                cat "$base-validation.txt" >&2 || true
                die "Nsight validation failed for $variant"
            }
        cat "$base-validation.txt"
    done

    python3 - "$OUTPUT_DIR/ncu" "$OUTPUT_DIR/ncu-summary.csv" <<'PY'
import csv, math, pathlib, sys
ncu_dir, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
required = [line.strip() for line in
            (ncu_dir / "metrics-required.txt").read_text().splitlines()
            if line.strip()]
out = []
for variant in ("baseline", "prefetch1", "prefetch2"):
    rows = list(csv.DictReader(open(
        ncu_dir / f"{variant}.csv", newline="", encoding="utf-8-sig")))
    data = [row for row in rows if (row.get("ID") or "").strip()]
    units = [row for row in rows if not (row.get("ID") or "").strip()]
    if len(data) != 1:
        raise SystemExit(
            f"{variant}: expected one NCU kernel row, got {len(data)}")
    unit = units[-1] if units else {}
    for metric in required:
        value = (data[0].get(metric) or "").strip()
        if not value or value.lower() in {"n/a", "not available"}:
            raise SystemExit(f"{variant}: required metric {metric} has no value")
        try:
            number = float(value.replace(",", ""))
        except ValueError:
            raise SystemExit(
                f"{variant}: required metric {metric} is non-numeric: {value!r}")
        if not math.isfinite(number):
            raise SystemExit(
                f"{variant}: required metric {metric} is non-finite: {value!r}")
        if metric == "gpu__time_duration.sum" and number <= 0.0:
            raise SystemExit(f"{variant}: duration is not positive")
        out.append({
            "variant": variant, "metric": metric,
            "unit": (unit.get(metric) or "").strip(), "value": value,
        })
with open(output, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=out[0].keys())
    writer.writeheader(); writer.writerows(out)
PY
    cat "$OUTPUT_DIR/ncu-summary.csv"
fi

current_phase=complete
printf 'SM75 Q3A4 K4 prefetch-depth audit complete: %s\n' "$OUTPUT_DIR"
