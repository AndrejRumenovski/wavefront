#!/usr/bin/env bash
# Measures solver throughput (steps/s) across thread counts, on two domain
# shapes with the *same total voxel count* but very different Z-slab
# halo-exchange plane sizes -- see PERFORMANCE.md for why the shape matters.
#
# Writes benchmarks/thread_scaling.csv; plot it with
# benchmarks/plot_thread_scaling.py.
#
# Usage:
#   RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" cargo +nightly build --release
#   benchmarks/thread_scaling.sh [scratch_dir] [threads...]
#
# scratch_dir defaults to a temp directory; threads default to a sweep up to
# nproc. Point scratch_dir at real disk (not tmpfs) with a few GB free --
# each trial writes one snapshot (~384 MiB at the cube shape's default size).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
BIN="$REPO_ROOT/target/release/wavefront"
SCRATCH="${1:-$(mktemp -d)}"
shift || true
THREADS=("$@")
if [ ${#THREADS[@]} -eq 0 ]; then
    NPROC=$(nproc)
    THREADS=(1 2 4 6 8 10 "$NPROC")
fi

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found -- build it first:" >&2
    echo '  RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" cargo +nightly build --release' >&2
    exit 1
fi

mkdir -p "$SCRATCH"
OUT_CSV="$HERE/thread_scaling.csv"
echo "shape,threads,slabs,elapsed_s,steps_per_s" > "$OUT_CSV"

run_shape() {
    local shape_name="$1" nx="$2" ny="$3" nz="$4" steps="$5"
    for t in "${THREADS[@]}"; do
        rm -f "$SCRATCH"/*.grid "$SCRATCH"/*.bin
        local out
        out=$(RAYON_NUM_THREADS="$t" "$BIN" \
            --nx "$nx" --ny "$ny" --nz "$nz" \
            --steps "$steps" --snapshot-every 1000 \
            --materials "$SCRATCH/m.grid" --output "$SCRATCH/t.bin" 2>&1 \
            | grep "steps over")
        local slabs elapsed sps
        slabs=$(echo "$out" | sed -E 's/.*over ([0-9]+) slab.*/\1/')
        elapsed=$(echo "$out" | sed -E 's/.*in ([0-9.]+)s.*/\1/')
        sps=$(echo "$out" | sed -E 's/.*\(([0-9.]+) steps\/s\).*/\1/')
        echo "$shape_name,$t,$slabs,$elapsed,$sps" | tee -a "$OUT_CSV"
    done
    rm -f "$SCRATCH"/*.grid "$SCRATCH"/*.bin
}

# Same total voxel count (256^3 = 64*64*4096 = 16,777,216) at two very
# different aspect ratios: "cube" has a large XY halo-exchange plane
# (256x256 voxels), "tall" has a 16x smaller one (64x64 voxels).
echo "=== cube: 256x256x256 ==="
run_shape cube 256 256 256 150
echo "=== tall: 64x64x4096 (same total voxels, 16x smaller halo plane) ==="
run_shape tall 64 64 4096 150

echo "wrote $OUT_CSV"
