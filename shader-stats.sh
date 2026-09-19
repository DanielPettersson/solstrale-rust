#!/bin/bash
# Dump the tracer's AMD ISA and per-shader occupancy stats out of RADV.
#
# Usage: ./shader-stats.sh [test name] [output file]
#   ./shader-stats.sh                       # test_scene -> /tmp/radv-tracer.txt
#   ./shader-stats.sh test_render_obj_with_textures
#
# Pick a test with no post-processing chain, so only the tracing pipeline
# compiles and there is one obvious shader in the dump.
#
# Four things about the flags, each of which has cost an hour at least once:
#
#  - `RADV_DEBUG=asm` is the *radeonsi* spelling. For RADV it is `shaders`.
#  - `nocache` and `MESA_SHADER_CACHE_DISABLE` both matter. A cached pipeline
#    compiles nothing and therefore prints nothing, which is the usual reason
#    this looks like it does not work.
#  - `RADV_DEBUG=help` prints nothing on Mesa 26.2.2. The flag names drift
#    between releases, so read them out of the driver instead:
#      strings /usr/lib64/libvulkan_radeon.so | grep -A 60 '^nofastclears$'
#    (`shaders` itself will not be in that list -- it is a suffix of
#    `metashaders`, so the linker merged the two strings.)
#  - `RADV_DEBUG=spirv` dumps what naga actually handed the driver, which is
#    the thing to read when the ISA does not look like the WGSL.
#
# Everything here is diagnostic. `SOLSTRALE_GPU_TIMING=1` is what measures how
# long a pass takes; this is what says why.
set -euo pipefail

TEST="${1:-test_scene}"
OUT="${2:-/tmp/radv-tracer.txt}"

RADV_DEBUG=shaders,shaderstats,nocache MESA_SHADER_CACHE_DISABLE=true \
    cargo test --test integration_tests "$TEST" -- --exact --nocapture \
    2> "$OUT" >/dev/null

echo "full ISA and stats in $OUT"
echo

# One block per pipeline compiled, largest last -- which on a chainless test is
# the tracer.
awk '
    /\*\*\* SHADER STATS \*\*\*/ { block = ""; capture = 1; next }
    /^\*\*\*\*\*\*\*/            { if (capture) print block "---"; capture = 0; next }
    capture                      { block = block $0 "\n" }
' "$OUT" |
    grep -E '^(SGPRs|VGPRs|Spilled SGPRs|Spilled VGPRs|Code size|LDS size|Scratch size|Subgroups per SIMD|Instructions|VALU|SALU|VMEM|SMEM):|^---$'
