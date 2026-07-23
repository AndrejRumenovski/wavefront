# Performance: is the SIMD real, and does the parallelism actually help?

This document covers two questions that "it runs and produces correct
output" (see `VALIDATION.md`) doesn't answer: is the AVX2 vectorization
`src/fdtd.rs` is written for actually being emitted, rather than silently
falling back to scalar code — and does `src/engine.rs`'s rayon-based domain
decomposition actually make the solver faster as more threads are thrown at
it. The second question turned up a real, counterintuitive result: more
threads *used to* make the solver slower here, not faster, for a specific,
identifiable reason — which made it fixable, not just a vague "parallelism
has overhead" hand-wave. The fix (direct cross-slab reads instead of a
per-boundary channel clone, replacing the halo-exchange design entirely) is
in this repo now; the "before" section below is kept because the
methodology and the reasoning that found the cause are as much the point as
the fixed numbers.

## AVX2 codegen: confirmed, not assumed

`Cargo.toml`'s build instructions insist on `-C target-cpu=native -C
target-feature=+avx2`, and `src/fdtd.rs`'s docs claim every row of a Yee
update is "loaded, updated, and stored as a single SIMD instruction" — but a
claim like that is easy to get wrong silently: a missed `RUSTFLAGS`, a
codegen decision that falls back to scalar, or `std::simd` lowering
differently than expected would all still compile and run correctly (just
slower), with nothing in `cargo build`'s output to flag it.

Checked directly, by disassembling the release binary and finding the
actual hot-loop closures:

```sh
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" cargo +nightly build --release
objdump -d --no-show-raw-insn target/release/wavefront > /tmp/wf_disasm.txt

# Which functions have the most 256-bit (ymm-register) AVX instructions?
awk '
/^[0-9a-f]+ </ { if (name != "") print count, name; name=$0; count=0; next }
/ymm/ { count++ }
END { if (name != "") print count, name }
' /tmp/wf_disasm.txt | sort -rn | head -5
```

The top two results (237 and 223 `ymm`-register instructions each) are
`rayon`'s `bridge_producer_consumer::helper` closures for `engine::run`'s two
`par_chunks_mut(...).for_each(...)` calls — the H-update and E-update phases
— with `update_slab_h`/`update_slab_e` (and, through them,
`fdtd::update_h_field`/`update_e_field`) fully inlined into them by LTO.
Nothing else in this crate has any reason to touch a 256-bit vector
register. Pulling just the H-update closure's instruction mix out confirms
it's genuine float32 SIMD arithmetic matching the Yee curl update's actual
operations (`Da*Hx - Db*curl`), not, say, a vectorized `memcpy`:

| Instruction | Count | Meaning |
|---|---:|---|
| `vmovups`      | 78 | unaligned load/store of a `f32x8` row |
| `vmulps`       | 42 | multiply (coefficient × field/curl term) |
| `vsubps`       | 36 | subtract (the curl differences) |
| `vaddps`       | 12 | add |
| `vbroadcastss` | 12 | broadcast a scalar into a lane (the `MaterialCoeffs` gather) |

This is direct evidence, not an assumption: the exact functions that run
every timestep, on every worker thread, are genuinely vectorized.

## Thread scaling: a real, mechanism-level finding (and a fix)

### Method

`benchmarks/thread_scaling.sh` runs the same total voxel count (256³ =
16,777,216 voxels) at two very different aspect ratios, across
`RAYON_NUM_THREADS` from 1 up to `nproc`, timing only the solver loop
(`engine::run`'s own `steps/s` printout, which excludes material
voxelization and coefficient-grid setup):

- **`cube`**: 256×256×256. Z-slab decomposition (`src/engine.rs`) splits
  this into up to 11-12 slabs at high thread counts, each with a large
  256×256-voxel (32×32-block) XY cross-section.
- **`tall`**: 64×64×4096. Same 16,777,216 total voxels, same number of
  slabs at a given thread count, but each slab's cross-section is
  64×64 voxels (8×8 blocks) — **16× smaller**.

The only structural difference between the two runs is the size of the
plane of field data exchanged at each internal slab boundary, once per
H-update and once per E-update, per boundary, per step (see `src/engine.rs`'s
module docs). At the time this comparison was first run, that exchange was
`crossbeam_channel`-based; it no longer is — see "The fix" below.

### Result (before the fix)

| Threads | `cube` steps/s | `tall` steps/s |
|---:|---:|---:|
| 1  | 8.3 | 7.4 |
| 2  | 8.0 | 7.3 |
| 4  | 7.7 | 7.3 |
| 6  | 7.3 | 7.2 |
| 8  | 7.0 | 7.2 |
| 10 | 7.0 | 7.1 |
| 12 | 6.7 | 7.0 |

- **`cube` got *slower* with more threads**: 8.3 → 6.7 steps/s from 1 to 12
  threads, a **20% regression**, monotonic at every step along the way. In
  cell-updates/second terms (a standard FDTD throughput metric): 140M
  cells/s at 1 thread vs. 112M cells/s at 12 threads — using all 12 threads
  did *less* total work per second than using one.
- **`tall` was nearly flat**: 7.4 → 7.0 steps/s, a 4% regression — the same
  qualitative direction, but five times smaller.
- Both measurements repeated tightly (two full re-runs of the 1-thread and
  12-thread `cube` cases landed within 1-2% of each other), so this wasn't
  noise.

### Why: halo-exchange bandwidth, not core count or fundamental memory bandwidth

`src/engine.rs`'s halo exchange used to send a *cloned, newly
heap-allocated* copy of a full boundary XY-plane of `FieldBlock`s
(`slab[..plane].to_vec()` / `slab[slab.len()-plane..].to_vec()`) through a
`crossbeam_channel`, once per phase, per internal slab boundary, every
single timestep. For the `cube` shape at 12 threads (11 slabs, 10 internal
boundaries), that was:

```
plane size    = 32 x 32 blocks x 12,288 bytes/block  = 12.58 MB
per step      = 10 boundaries x 2 phases x 12.58 MB   = 251.7 MB
```

251.7 MB of clone-and-channel-transfer traffic, *every step*, is a lot to
pay relative to the actual compute: 16,777,216 voxels split across 11 slabs
is only ~1.5M voxels of real Yee-update work per slab. For the `tall` shape,
the same math gave **17.3 MB/step** — a ~14.5× reduction, closely tracking
the 16× smaller plane (the difference is just 10 vs. 11 boundaries at
slightly different slab counts) — and the measured scaling penalty shrank by
almost exactly the same factor (20% → 4%). That wasn't a coincidence; it was
the mechanism, and (see below) removing it entirely confirms that reading.

## The fix: direct cross-slab reads, no clone, no channel, no allocation

The channel-based design assumed reaching into a neighboring slab's memory
was inherently a data race. That premise was too conservative: within a
single phase, every slab's closure writes only *one* field type of its own
blocks (H during the H-phase, E during the E-phase) and reads only the
*other* field type, off itself and its neighbors. Two concurrently running
closures therefore never touch the same memory location with at least one
write — one thread's `Hx/Hy/Hz` writes and another thread's `Ex/Ey/Ez` reads
land in disjoint array fields of the same `FieldBlock`. That's exactly the
reasoning `update_slab_h`/`update_slab_e` already relied on for *intra*-slab
neighbor access (their own long-standing `SAFETY` comments); `CrossSlabPtr`
(`src/engine.rs`) extends the identical argument across the
`par_chunks_mut` boundary. Each slab now reads its neighbor's already-settled
boundary plane directly through a raw pointer into the single shared backing
allocation — no clone, no heap allocation, no channel, at any thread count.

This is a genuine behavior change to the trickiest concurrency code in the
crate, so it shipped with a new correctness test
(`engine::tests::multi_slab_decomposition_matches_single_slab_bit_for_bit`)
that runs the same domain/source/steps once forced onto a single-thread
pool (no cross-slab reads at all) and once forced onto a multi-thread pool
(`num_slabs > 1`, exercising `CrossSlabPtr` on every boundary), and asserts
the final field state is **bit-for-bit identical** — domain decomposition is
purely a scheduling detail, so anything less than exact equality would mean
a real bug, not acceptable floating-point noise.

### Result (after the fix)

![256³ cube: steps/s vs. thread count, before vs. after the halo-exchange fix](benchmarks/halo_fix_before_after.png)

| Threads | `cube` before | `cube` after | `tall` before | `tall` after |
|---:|---:|---:|---:|---:|
| 1  | 8.3 | 8.7  | 7.4 | 7.5 |
| 2  | 8.0 | 10.8 | 7.3 | 9.2 |
| 4  | 7.7 | 11.7 | 7.3 | 10.1 |
| 6  | 7.3 | 11.7 | 7.2 | 9.8 |
| 8  | 7.0 | 11.2 | 7.2 | 9.6 |
| 10 | 7.0 | 11.3 | 7.1 | 9.5 |
| 12 | 6.7 | 10.9 | 7.0 | 9.3 |

- **`cube`**: 8.7 → 10.9 steps/s, 1 to 12 threads — **+25%**, and peaks
  around 4-6 threads at **+34%** (11.7 steps/s), instead of the old -20%
  regression. This is now the *faster* shape of the two, not the slower one
  — with the clone cost gone, its larger per-slab compute-to-boundary ratio
  works in its favor rather than against it.
- **`tall`**: 7.5 → 9.3 steps/s — **+24%**, peaking near 4 threads at +35%.
  Both shapes now show the *same* qualitative curve: real speedup up to
  roughly the machine's 6 physical cores, then a mild decline as the
  remaining threads are SMT siblings sharing execution units rather than
  independent cores.

![Steps/s vs. thread count, cube vs. tall-thin domain shape, after the fix](benchmarks/thread_scaling.png)

### A hypothesis this also revises

The original write-up (kept above for the reasoning, not because it's still
believed) proposed that `tall`'s *residual* flatness — even after removing
most of the halo cost by reshaping — was likely because a single thread
already saturates this machine's memory bandwidth, leaving little headroom
for more threads regardless of the halo fix. The post-fix data doesn't
support that as the primary explanation: both shapes now scale
*substantially* (+34%/+35% at their peaks), which is hard to reconcile with
"already bandwidth-saturated at one thread." The more consistent reading is
that the *reshaped* `tall` domain in the original measurement still paid a
real, if much smaller, per-step cost for the clone-and-channel round trip
(a 16× smaller plane is not a *zero* plane) — enough to mask most of the
real available parallelism. Memory bandwidth may still be part of why
scaling plateaus rather than continuing to climb past ~6 threads (a
reasonable inference from the SMT topology and the kernel's low arithmetic
intensity) — see "Profiling the plateau" below, where this is checked
against actual hardware counters instead of left as inference.

### Profiling the plateau: what `perf stat` actually shows

This machine's `kernel.perf_event_paranoid` defaults to `4` (unprivileged
`perf_event_open` fully disabled), which is why the rest of this document
had left memory bandwidth as an inference rather than a measurement.
Temporarily lowering it (`sudo sysctl kernel.perf_event_paranoid=1`, runtime
only — reverts on reboot, or explicitly with the same command and `=4`)
unblocks core-PMU counters for an unprivileged process. It does *not*
unblock the AMD data-fabric/UMC uncore PMU that would give a direct DRAM
GB/s figure (`perf list`'s `nps1_die_to_dram` metric errors with "Bad event
or PMU" even at `paranoid=1` on this kernel/hardware combination) — so this
is still a proxy-based reading, via cache-miss and stall-cycle counters, not
a direct bandwidth number. `AMD stalled-cycles-backend` is also
`<not supported>` on this chip (Zen 3, `Ryzen 5 5600G`: 6 cores / 12
threads, 3 MiB L2 total (6×512 KiB private), 16 MiB shared L3, one CCX).

**Method**: `perf stat -e cycles,instructions,cache-references,cache-misses,
l2_cache_misses_from_dc_misses,stalled-cycles-frontend,branch-misses`
wrapped around the same `cube` (256³) case `thread_scaling.sh` uses, at
`RAYON_NUM_THREADS` 1/4/6/8/12 — chosen to straddle the physical-core count
(6) and the plateau/decline `thread_scaling.csv` already showed starting
somewhere past it. `perf stat` requested more events than this chip has
hardware counters for, so the kernel time-slices them (visible as
`(71.43%)` next to every line) and scales the reported counts back up —
standard practice, but it means these are statistically-scaled estimates,
not exact hardware counts. As a sanity check, a clean re-run of the same
five thread counts *without* `perf` attached landed within 1% of the
perf-instrumented run's own `steps/s` at every point (e.g. 1 thread:
10.3 vs. 10.2 steps/s; 12 threads: 13.8 vs. 13.7 steps/s) — `perf`'s own
overhead isn't distorting the comparison. These runs are faster in absolute
terms than the committed `thread_scaling.csv` table above (10.2 steps/s at
1 thread here vs. 8.7 there) — expected, since this document already treats
throughput as machine/load-dependent and not CI-gated; this sweep is only
used for the *relative* trend across thread counts within itself, not as a
replacement for `thread_scaling.csv`/`.png`.

| Threads | steps/s | cache-miss rate | cycles/thread (normalized) | frontend-stall % of cycles | L2-miss traffic/thread (normalized) |
|---:|---:|---:|---:|---:|---:|
| 1  | 10.2 | 20.4% | 68.1B | 2.1% | 575.0M |
| 4  | 14.9 | 19.2% | 41.6B | 1.2% | 353.3M |
| 6  | 14.5 | 20.0% | 38.1B | 1.7% | 242.0M |
| 8  | 14.2 | 22.6% | 38.8B | 1.8% | 209.2M |
| 12 | 13.7 | 25.8% | 37.1B | 1.9% | 163.0M |

("Normalized" columns divide `perf stat`'s process-wide total by thread
count, since `perf stat` sums counters across every rayon worker thread —
comparing *that* raw sum across thread counts would just measure "how many
cores were burning cycles," not efficiency.)

Two things point toward shared-cache/memory contention, not SMT
execution-port contention, as the more consistent explanation for the
plateau and mild decline past 4-6 threads:

- **Cache-miss rate climbs monotonically past 4 threads** (19.2% → 25.8%,
  4 to 12 threads) even though total `cache-references` stays essentially
  flat (~7.9-8.0B) across that same range — the same volume of cache
  traffic is increasingly missing, consistent with more concurrent slabs'
  working sets competing for the single 16 MiB L3 shared across all 6
  cores/12 threads of this one-CCX chip.
- **Per-thread cycles stop shrinking once threads exceed the physical core
  count**, even though each thread's share of the domain (and so its real
  compute) keeps shrinking: 1→4 threads nearly halves cycles/thread
  (68.1B→41.6B) as expected from real parallel speedup, but 4→12 threads
  is flat (41.6B→37.1B) despite 3× fewer voxels per thread. Work is
  shrinking per-thread; cycles aren't — the difference is being spent
  somewhere other than the Yee-update arithmetic itself.

Meanwhile, **frontend-stall cycles stay a small, only mildly-rising fraction
of total cycles throughout (1.2-2.1%)** — if SMT sibling threads fighting
over the frontend/execution ports were the dominant effect, this is the
counter that would be expected to climb sharply past 6 threads (the
physical-core boundary, where SMT siblings start actually sharing a core);
it doesn't. That doesn't rule SMT contention out entirely (this chip has no
working `stalled-cycles-backend` counter to check execution-port pressure
directly), but the cache-miss-rate and per-thread-cycles evidence is more
direct and points the same direction: **the post-~4-6-thread plateau is
better explained by shared L3 capacity/bandwidth contention across this
chip's single CCX than by core-count or SMT execution-unit limits** — a more
specific conclusion than the inference this section previously stood on,
though still short of a directly-measured DRAM GB/s ceiling, since the
uncore PMU needed for that remains inaccessible on this hardware/kernel
combination.

**Practical takeaway**: the channel-based halo exchange was actively
counterproductive for domains with a large XY cross-section relative to
their Z extent, and *partially* counterproductive even for the reshaped
domain that was supposed to control for that. Removing it (not deferring it
as a follow-up, which is what the first version of this document planned)
uncovered real parallelism the earlier measurement had misattributed to
fundamental hardware limits.

## Snapshot writer throughput

### Method

`benchmarks/snapshot_throughput.sh` runs the same domain and step count
twice: once with `--snapshot-every` large enough that only the unavoidable
step-0 snapshot is written (see `src/engine.rs` — `step % snapshot_every ==
0` always fires at `step = 0` regardless of the configured interval), and
once with `--snapshot-every 1` (a write every step). The difference in
elapsed time, divided by the difference in bytes written, isolates the
writer's actual sustained throughput from solver compute time — the
double-buffered design overlaps writes with the *next* snapshot's compute,
so this is genuine marginal I/O cost, not a naive total-bytes-over-total-time
average that would also count time the writer spent idle waiting on compute.

### Result

At 256³ voxels (384 MiB/snapshot), 20 steps:

| Run | Elapsed | Snapshots written |
|---|---:|---:|
| Compute-only (1 snapshot) | 6.19s | 384 MiB |
| Every-step (20 snapshots) | 69.9s | 7.5 GiB |

```
extra bytes:     7,650,410,496
extra time:      63.7 s
throughput:      120 MB/s
```

120 MB/s is squarely in spinning-HDD sequential-write territory (not NVMe)
— consistent with `VALIDATION.md`'s out-of-core section, which already
established this machine's only Linux-accessible storage with free capacity
is an ext4-formatted HDD, not the NVMe device (fully consumed by a Windows
dual-boot install). This number characterizes *this machine's disk*, not
the writer's ceiling; on real NVMe the `O_DIRECT`/`io_uring` path itself
should sustain far more, but that's unverified here for the same reason the
out-of-core validation flagged it as an open item.

## Reproducing this

```sh
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" \
    cargo +nightly build --release
cargo +nightly test --release engine::  # includes the bit-for-bit multi-slab check

benchmarks/thread_scaling.sh /path/on/a/real/disk
python3 benchmarks/plot_thread_scaling.py           # regenerates benchmarks/thread_scaling.png
python3 benchmarks/plot_halo_fix_before_after.py    # regenerates benchmarks/halo_fix_before_after.png,
                                                     # comparing against benchmarks/thread_scaling_before_fix.csv
                                                     # (a saved snapshot from commit 87480c4, pre-fix)

benchmarks/snapshot_throughput.sh /path/on/a/real/disk
```

For the `perf stat` sweep in "Profiling the plateau" above, unprivileged
`perf_event_open` needs unblocking first (root, so run this yourself, not
from an automated/sandboxed shell):

```sh
sudo sysctl kernel.perf_event_paranoid=1   # runtime only; revert with =4, or just reboot

SCRATCH=$(mktemp -d)
for t in 1 4 6 8 12; do
    rm -f "$SCRATCH"/*.grid "$SCRATCH"/*.bin
    RAYON_NUM_THREADS=$t perf stat \
        -e cycles,instructions,cache-references,cache-misses,l2_cache_misses_from_dc_misses,stalled-cycles-frontend,branch-misses \
        target/release/wavefront --nx 256 --ny 256 --nz 256 --steps 150 --snapshot-every 1000 \
        --materials "$SCRATCH/m.grid" --output "$SCRATCH/t.bin"
done
```

Both `.sh` scripts default their thread sweep / domain size to sensible
values but accept overrides — see each script's header comment. Point them
at a real block device, not `tmpfs`, for the snapshot writer measurement to
mean anything.

These are not CI-gated (unlike `convergence_study.rs`/
`pml_reflection_study.rs`): throughput numbers are machine- and
load-dependent in a way phase velocity and reflection coefficients aren't,
so asserting a specific steps/s or MB/s in CI would just be flaky, not
rigorous.
