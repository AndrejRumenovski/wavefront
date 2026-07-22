#!/usr/bin/env bash
# Measures the O_DIRECT snapshot writer's sustained throughput by comparing
# a compute-only run (one snapshot, at step 0 -- unavoidable, see
# PERFORMANCE.md) against a run that writes every step, at the same domain
# and step count. The difference in elapsed time, divided by the difference
# in bytes written, isolates the writer's throughput from solver compute
# time -- overlapped by the double buffering, so this is genuinely marginal
# I/O cost, not a naive total-bytes/total-time average.
#
# Usage:
#   RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" cargo +nightly build --release
#   benchmarks/snapshot_throughput.sh [scratch_dir] [nx] [steps]
#
# scratch_dir must be on real disk (not tmpfs) with a few GB free -- the
# every-step run writes steps * snapshot_bytes total.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
BIN="$REPO_ROOT/target/release/wavefront"
SCRATCH="${1:-$(mktemp -d)}"
NX="${2:-256}"
STEPS="${3:-20}"

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found -- build it first:" >&2
    echo '  RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" cargo +nightly build --release' >&2
    exit 1
fi

mkdir -p "$SCRATCH"
rm -f "$SCRATCH"/*.grid "$SCRATCH"/*.bin

echo "=== compute-only baseline (snapshot_every=1000, effectively 1 write) ==="
BASELINE_OUT=$("$BIN" --nx "$NX" --ny "$NX" --nz "$NX" --steps "$STEPS" --snapshot-every 1000 \
    --materials "$SCRATCH/m.grid" --output "$SCRATCH/baseline.bin" 2>&1 | grep "steps over")
echo "$BASELINE_OUT"
BASELINE_ELAPSED=$(echo "$BASELINE_OUT" | sed -E 's/.*in ([0-9.]+)s.*/\1/')
BASELINE_SIZE=$(stat -c%s "$SCRATCH/baseline.bin")
rm -f "$SCRATCH"/*.grid "$SCRATCH"/baseline.bin

echo
echo "=== every-step snapshots (snapshot_every=1, $STEPS writes) ==="
HEAVY_OUT=$("$BIN" --nx "$NX" --ny "$NX" --nz "$NX" --steps "$STEPS" --snapshot-every 1 \
    --materials "$SCRATCH/m.grid" --output "$SCRATCH/heavy.bin" 2>&1 | grep "steps over")
echo "$HEAVY_OUT"
HEAVY_ELAPSED=$(echo "$HEAVY_OUT" | sed -E 's/.*in ([0-9.]+)s.*/\1/')
HEAVY_SIZE=$(stat -c%s "$SCRATCH/heavy.bin")
rm -f "$SCRATCH"/*.grid "$SCRATCH"/heavy.bin

python3 -c "
extra_bytes = $HEAVY_SIZE - $BASELINE_SIZE
extra_time = $HEAVY_ELAPSED - $BASELINE_ELAPSED
mbps = extra_bytes / extra_time / 1e6
print()
print(f'extra bytes written: {extra_bytes:,}')
print(f'extra elapsed time: {extra_time:.3f} s')
print(f'sustained O_DIRECT write throughput: {mbps:.1f} MB/s')
"
