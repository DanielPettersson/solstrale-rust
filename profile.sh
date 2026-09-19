#!/bin/bash
# CPU profile of one criterion benchmark.
#
# This is the instrument for the CPU-bound side of the renderer -- BVH build,
# scene flatten, OBJ load, image readback. It says nothing about what the GPU
# is doing: `SOLSTRALE_GPU_TIMING=1` is that instrument, and
# `gpu_pass_timings` is the table it prints.
#
# Usage: ./profile.sh [criterion filter] [seconds]
#   ./profile.sh bvh_build 20
#   ./profile.sh 'obj_load/total'
#
# `--profile-time` tells criterion to skip warm-up and statistical analysis and
# simply run the benchmark for N seconds, which is the shape a sampling
# profiler wants. DWARF call graphs work because `[profile.release] debug = 1`
# is set in Cargo.toml, and criterion's bench profile inherits it.
set -euo pipefail

FILTER="${1:-bvh_build}"
SECONDS_TO_RUN="${2:-10}"
OUT="${PERF_DATA:-perf.data}"

# Build first and pick the harness out of cargo's JSON, so `perf` profiles the
# benchmark rather than rustc and cargo.
BENCH_BIN=$(cargo bench --bench solstrale_benchmark --no-run --message-format=json 2>/dev/null |
    grep -o '"executable":"[^"]*solstrale_benchmark[^"]*"' | tail -1 | cut -d'"' -f4)

if [ -z "$BENCH_BIN" ]; then
    echo "could not find the built benchmark binary" >&2
    exit 1
fi

perf record -F 999 --call-graph dwarf -o "$OUT" -- \
    "$BENCH_BIN" --bench --profile-time "$SECONDS_TO_RUN" "$FILTER"

echo
echo "wrote $OUT -- read it with:  perf report -i $OUT"
