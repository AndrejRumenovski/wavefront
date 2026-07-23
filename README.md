# wavefront

Asynchronous, out-of-core 3D Finite-Difference Time-Domain (FDTD)
electromagnetic simulator. Solves Maxwell's curl equations on a dense
voxelized material grid (up to ~200 GB, mmap-backed, larger than physical
RAM), on a single Linux workstation.

**[VALIDATION.md](VALIDATION.md)** measures the solver's numerical
dispersion against the Yee scheme's own exact closed-form prediction
(confirming second-order convergence — measured order 2.16 vs. theoretical
2.0) and the CPML absorbing boundary's actual reflection coefficient against
its configured target — the evidence that this is a numerically correct
FDTD implementation, not just code that produces plausible-looking output.

- Material grid: a flat, disk-resident byte array, one byte per voxel,
  memory-mapped via `memmap2` (`src/layout.rs`).
- Field grid: `Ex/Ey/Ez/Hx/Hy/Hz`, tiled into `#[repr(align(64))]`
  Array-of-Structures-of-Arrays blocks sized to the AVX2 `f32x8` lane width
  (`src/layout.rs`).
- Solver: SIMD (`std::simd`) Yee-lattice curl update kernels
  (`src/fdtd.rs`).
- Scheduling: the grid is decomposed into Z-slabs and fanned out across
  `rayon`'s work-stealing thread pool. Cross-slab boundary data is read
  directly out of the shared backing allocation via a raw pointer
  (`CrossSlabPtr`) rather than copied through a channel -- provably
  race-free because each phase writes only one field type and reads only
  the other (`src/engine.rs`; see `PERFORMANCE.md` for the measured
  scaling impact of this over an earlier `crossbeam-channel`-based design).
- Boundaries: a graded Convolutional PML (CPML) absorbs outgoing waves at
  each of the 6 domain faces, so the domain behaves like open space instead
  of a sealed reflective box. Its auxiliary convolution memory is only
  allocated for the thin shell of boundary blocks, not the whole volume
  (`src/layout.rs`'s `PmlContext`/`PmlAuxGrid`, dispatched per-block in
  `src/engine.rs`).
- I/O: field snapshots are streamed out via double-buffered, `O_DIRECT`
  `io_uring` writes (through the `rio` crate), so storage latency never
  stalls the timestep loop (`src/engine.rs`).
- Excitation: one or more point soft sources drive a field component every
  timestep with a configurable time-domain waveform (Gaussian pulse,
  sinusoid, or Ricker wavelet) -- not just a one-shot initial condition
  (`src/source.rs`).
- Frequency-domain probes: a point probe accumulates a running discrete
  Fourier transform (DFT) at one or more frequencies while the simulation
  runs, so a driven-sinusoid run can report steady-state amplitude/phase
  response (resonance, transmission) directly, without streaming and
  post-processing the full time-domain snapshot history (`src/probe.rs`).
- Geometry: structures can be described in a small plain-text scene format
  (spheres and boxes tagged with material constants) and voxelized into the
  material grid, instead of only the hardcoded demo sphere (`src/scene.rs`).
- Visualization: `wavefront-view`, a second binary in this crate, renders a
  2D slice or a whole-domain maximum-intensity projection of one snapshot as
  a PPM image, or a range of snapshots as an animated GIF, so a run's output
  can actually be looked at (`src/bin/wavefront-view.rs`).

## Requirements

- **Rust nightly.** The Yee kernels use `std::simd` (`portable_simd`),
  which isn't stabilized yet:

  ```sh
  rustup toolchain install nightly
  rustup override set nightly
  ```

- **Linux**, kernel >= 5.6 (io_uring), ideally >= 5.11.
- An output path on a filesystem that supports `O_DIRECT` (ext4, xfs, btrfs
  all work; tmpfs and some network filesystems do not).
- An x86_64 or aarch64 host. The material grid file needs enough free disk
  space for `nx * ny * nz` bytes; the snapshot stream needs `snapshot_bytes
  * (steps / snapshot_every)`.

## Build

Always build in release mode with native-CPU vectorization enabled — this
is what turns the Maxwell solver's inner loops into real AVX2 instructions
instead of falling back to scalar code:

```sh
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" \
    cargo +nightly build --release
```

This produces two binaries, both tuned to the exact machine they were built
on (`target-cpu=native`) — don't copy them to a different CPU
microarchitecture; rebuild there instead:

- `target/release/wavefront` — the simulator.
- `target/release/wavefront-view` — the snapshot-to-image/animation
  post-processing tool (see [Visualize](#visualize)).

Both share the core solver code via a library crate (`src/lib.rs`); the
simulator binary itself is just `src/main.rs`'s CLI/orchestration layer.

> **Licensing note:** the `rio` crate is GPL-3.0 by default (an MIT/Apache-2.0
> dual license is available by sponsoring the author). Confirm this is
> acceptable before distributing a binary built against it.

## Run

```sh
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" \
    cargo +nightly run --release -- [OPTIONS]
```

or invoke the built binary directly:

```sh
./target/release/wavefront [OPTIONS]
```

Absent `--scene`, the demo scenario voxelizes a dielectric sphere
(`eps_r = 4.0`) at the center of an otherwise-vacuum domain. A single point
source (default: a Ricker wavelet on `Ez` at the domain center) is
re-injected every timestep.

### Options

| Flag                    | Meaning                                              | Default              |
|-------------------------|-------------------------------------------------------|----------------------|
| `--nx`, `--ny`, `--nz`  | Grid size per axis, in voxels (must be a multiple of 8) | `64`                |
| `--dx`                  | Uniform cell size, in meters                          | `1.0e-3`             |
| `--dt`                  | Timestep, in seconds (must satisfy the Courant limit for `dx`) | `1.5e-12`   |
| `--steps`               | Number of timesteps to run                            | `200`                |
| `--snapshot-every`      | Timesteps between snapshot writes                     | `20`                 |
| `--pml-thickness <N>`  | Absorbing boundary depth, in voxels, at each domain face. `0` disables it (fully reflective boundary) | `8` |
| `--scene <PATH>`        | Plain-text scene file (see below); omit for the demo sphere | (demo sphere) |
| `--source-x/-y/-z <N>`  | First source's voxel position                        | domain center        |
| `--source-component <C>`| First source's field component: `ex`, `ey`, `ez`     | `ez`                 |
| `--source-waveform <W>` | First source's waveform: `gaussian`, `sinusoid`, or `ricker` | `ricker`      |
| `--source-freq <HZ>`    | First source's drive frequency                        | `1 / (20 * dt)`      |
| `--source-amplitude <A>`| First source's peak amplitude                         | `1.0`                |
| `--source <SPEC>`       | Add another source: `key=value,...` (`x`, `y`, `z`, `component`, `waveform`, `freq`, `amplitude`) | (none) |
| `--probe-x/-y/-z <N>`   | First probe's voxel position -- all three required together with `--probe-freq` to enable it | (disabled) |
| `--probe-component <C>` | First probe's field component: `ex`, `ey`, `ez`, `hx`, `hy`, `hz` | `ez`  |
| `--probe-freq <HZ,...>` | Comma-separated frequencies (Hz) the first probe's running DFT tracks | (disabled) |
| `--probe-start <SECONDS>`| First probe's ignore-samples-before time (skips startup transient) | `0.0` |
| `--probe <SPEC>`        | Add another probe: `key=value,...` (`x`, `y`, `z`, `component`, `freq` [`;`-separated for multiple], `start`) | (none) |
| `--materials <PATH>`    | Backing file for the mmap'd material grid             | `materials.grid`     |
| `--output <PATH>`       | Direct I/O snapshot stream path                       | `wave_trajectory.bin`|
| `-h`, `--help`          | Print usage                                           |                      |

`engine::run` has always accepted a slice of sources/probes; the
`--source-*`/`--probe-*` flags above configure only the *first* one
(auto-created on first use). Repeat `--source`/`--probe` to add more —
mixing both styles (shorthand for the first, `--source`/`--probe` for
additional ones) is well-defined, not an error. `--probe`'s `freq` uses `;`
to separate multiple frequencies (commas are already the `key=value` pair
separator), so quote the whole value in your shell:

```sh
./target/release/wavefront \
    --source-x 5 --source-y 32 --source-z 32 --source-waveform sinusoid --source-freq 3e10 \
    --source "x=59,y=32,z=32,waveform=ricker,amplitude=0.5" \
    --probe "x=32,y=32,z=32,freq=3e10;6e10,start=1e-10" \
    --probe "x=45,y=32,z=32,component=hz,freq=3e10"
```

### Example

```sh
./target/release/wavefront --nx 128 --ny 128 --nz 128 \
    --steps 500 --snapshot-every 25 \
    --scene scenes/two_spheres.scene \
    --source-waveform sinusoid --source-freq 3e10 \
    --materials /mnt/nvme/materials.grid \
    --output /mnt/nvme/wave_trajectory.bin
```

`materials.grid` and `wave_trajectory.bin` are working files generated at
run time (see `.gitignore`) — point `--materials`/`--output` at your NVMe
mount for large grids rather than leaving the defaults in the repo checkout.

Add `--probe-x/-y/-z`, `--probe-freq`, and (optionally) `--probe-start` to
get a frequency-domain readout printed after the run, instead of only the
raw time-domain snapshot stream:

```sh
./target/release/wavefront --source-waveform sinusoid --source-freq 3e10 \
    --probe-x 42 --probe-y 32 --probe-z 32 \
    --probe-freq 3e10 --probe-start 2e-10
```

```
wavefront: probe (42, 32, 32) Ez frequency response:
  3.0000e10 Hz: amplitude 3.513664e-2, phase -0.1586 rad
```

### Scene format

`--scene` loads a plain-text file describing geometric primitives, applied
in order (later ones overwrite earlier ones where they overlap). Geometric
parameters are in voxel-index units, not meters:

```text
# comment
sphere <eps_r> <mu_r> <sigma> <cx> <cy> <cz> <radius>
box    <eps_r> <mu_r> <sigma> <x0> <y0> <z0> <x1> <y1> <z1>
```

See `scenes/two_spheres.scene` for a working example. Each distinct
`(eps_r, mu_r, sigma)` triple gets its own material slot automatically (up
to 255 non-vacuum materials).

### Output format

`wave_trajectory.bin` is a raw concatenation of snapshots; each snapshot is
every `FieldBlock` in the grid, in block-major (Z, then Y, then X) order,
each block serialized as six back-to-back `f32` arrays (`Ex, Ey, Ez, Hx, Hy,
Hz`), 512 voxels per array (8x8x8 block, row-major with X fastest-varying).
There's no header, so any reader (like `wavefront-view`) needs to already
know `nx`/`ny`/`nz`.

## Visualize

`wavefront-view` renders one snapshot as a binary PPM image (`.ppm` --
viewable directly in GIMP, or converted with `magick slice.ppm slice.png`),
or a range of snapshots as one animated GIF. It needs the same
`--nx`/`--ny`/`--nz` the simulation was run with, since the trajectory file
has no header:

```sh
# One 2D slice of one snapshot, as a PPM:
./target/release/wavefront-view \
    --input wave_trajectory.bin --nx 128 --ny 128 --nz 128 \
    --snapshot 10 --axis z --component energy --output slice.ppm

# A maximum-intensity projection through the whole domain, same snapshot:
./target/release/wavefront-view \
    --input wave_trajectory.bin --nx 128 --ny 128 --nz 128 \
    --snapshot 10 --mode volume --axis z --output volume.ppm

# Every snapshot from 0 to 40, as one looping animated GIF:
./target/release/wavefront-view \
    --input wave_trajectory.bin --nx 128 --ny 128 --nz 128 \
    --snapshots 0:40 --fps 12 --output wave.gif
```

| Flag                | Meaning                                                    | Default       |
|---------------------|-------------------------------------------------------------|---------------|
| `--input <PATH>`    | Trajectory file to read (required)                          |               |
| `--nx/-ny/-nz <N>`  | Grid dimensions the run used (required)                      |               |
| `--snapshot <N>`    | Which snapshot to render, 0-indexed                          | `0`           |
| `--snapshots <A>:<B>` | Render an inclusive snapshot range as one animated GIF instead (requires `--output` to end in `.gif`; mutually exclusive with `--snapshot`) | |
| `--fps <N>`         | Animation playback rate, for `--snapshots`                  | `10`          |
| `--mode <slice\|volume>` | `slice`: one 2D cross-section. `volume`: a maximum-intensity projection through the *entire* domain along `--axis` (`--slice` is ignored) | `slice` |
| `--axis <x\|y\|z>`  | Which axis to hold fixed (`slice`) or project along (`volume`) | `z`         |
| `--slice <N>`       | Index along `--axis` to slice at (`slice` mode only)         | middle        |
| `--component <C>`   | `ex`, `ey`, `ez`, `hx`, `hy`, `hz`, or `energy` (sum of squares of all six) | `energy` |
| `--output <PATH>`   | Output path (`.ppm` for a single snapshot, `.gif` for a range) | `slice.ppm` |

Values are normalized by the maximum magnitude across every frame being
rendered (a single snapshot's own max, in the single-frame case, so this
isn't a behavior change there) and mapped through a white-to-red/
white-to-blue diverging colormap (white-to-red only for `energy`, which is
never negative). The GIF encoder is hand-written, like the PPM writer --
real (adaptive-dictionary) LZW compression and a `NETSCAPE2.0` looping
extension, no image/GIF dependency added.

## Tests

```sh
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" \
    cargo +nightly test --release
```

Covers: fixed-point round-tripping, material/PML coefficient formulas
against their closed forms, Yee kernel invariants (a uniform field has zero
curl and is left unchanged), scene parsing, source waveform shapes, DFT
probe amplitude/phase recovery against a known synthetic sinusoid, an
end-to-end numerical check that a point source in vacuum radiates outward
at approximately the speed of light, and that multi-slab domain
decomposition produces bit-for-bit identical output to a single slab
(`src/engine.rs`'s `CrossSlabPtr` cross-slab halo reads, forced onto local
1-thread and multi-thread `rayon` pools). Neither of the last two goes
through `engine::run`'s `O_DIRECT` snapshot writer, so they have no
filesystem dependency and can't be flaky in a sandboxed CI environment.

CI (`.github/workflows/ci.yml`) builds and tests on nightly with the same
`RUSTFLAGS` as local development, on every push to `main` and every pull
request.

## Validation

Two examples are separate, deeper correctness checks, using the same
field-update kernels the production engine calls. See
**[VALIDATION.md](VALIDATION.md)** for the methodology and results of both.

`examples/convergence_study.rs` measures the solver's numerical phase
velocity against the Yee scheme's own exact closed-form dispersion relation
at four grid resolutions:

```sh
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" \
    cargo +nightly run --release --example convergence_study
python3 validation/plot_convergence.py   # regenerates validation/convergence.png
```

`examples/pml_reflection_study.rs` measures the CPML absorbing boundary's
actual reflection coefficient at four layer thicknesses, via two-run
subtraction against a reflection-free reference domain:

```sh
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" \
    cargo +nightly run --release --example pml_reflection_study
python3 validation/plot_pml_reflection.py   # regenerates validation/pml_reflection.png
```

## Performance

**[PERFORMANCE.md](PERFORMANCE.md)** confirms the AVX2 vectorization is
genuinely being emitted (not silently falling back to scalar, verified by
disassembling the release binary) and measures thread scaling and
`O_DIRECT` snapshot-writer throughput. It also tells the story of a real
finding-and-fix: the original `crossbeam-channel`-based halo exchange made
more `rayon` threads make the solver *slower* on cube-shaped domains
(-20% at 12 threads); replacing it with direct cross-slab reads
(`CrossSlabPtr`, `src/engine.rs`) — provably race-free by the same
disjoint-field-array argument the per-slab kernels already relied on —
turned that into genuine speedup instead (+25 to +34%), verified
bit-for-bit identical to a single-slab reference.

```sh
cargo +nightly test --release engine::             # includes the bit-for-bit multi-slab check
benchmarks/thread_scaling.sh /path/on/a/real/disk
python3 benchmarks/plot_thread_scaling.py          # regenerates benchmarks/thread_scaling.png
python3 benchmarks/plot_halo_fix_before_after.py   # regenerates benchmarks/halo_fix_before_after.png
benchmarks/snapshot_throughput.sh /path/on/a/real/disk
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
