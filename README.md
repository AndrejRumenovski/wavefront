# wavefront

Asynchronous, out-of-core 3D Finite-Difference Time-Domain (FDTD)
electromagnetic simulator. Solves Maxwell's curl equations on a dense
voxelized material grid (up to ~200 GB, mmap-backed, larger than physical
RAM), on a single Linux workstation.

- Material grid: a flat, disk-resident byte array, one byte per voxel,
  memory-mapped via `memmap2` (`src/layout.rs`).
- Field grid: `Ex/Ey/Ez/Hx/Hy/Hz`, tiled into `#[repr(align(64))]`
  Array-of-Structures-of-Arrays blocks sized to the AVX2 `f32x8` lane width
  (`src/layout.rs`).
- Solver: SIMD (`std::simd`) Yee-lattice curl update kernels
  (`src/fdtd.rs`).
- Scheduling: the grid is decomposed into Z-slabs and fanned out across
  `rayon`'s work-stealing thread pool, with `crossbeam-channel` used for
  lock-free halo exchange across slab boundaries (`src/engine.rs`).
- I/O: field snapshots are streamed out via double-buffered, `O_DIRECT`
  `io_uring` writes (through the `rio` crate), so storage latency never
  stalls the timestep loop (`src/engine.rs`).

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

The resulting binary (`target/release/wavefront`) is tuned to the exact
machine it was built on (`target-cpu=native`) — don't copy it to a
different CPU microarchitecture; rebuild there instead.

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

The demo scenario voxelizes a dielectric sphere (`eps_r = 4.0`) at the
center of an otherwise-vacuum domain, injects a Gaussian pulse into `Ez` at
t=0, and lets the solver propagate it.

### Options

| Flag                    | Meaning                                              | Default              |
|-------------------------|-------------------------------------------------------|----------------------|
| `--nx`, `--ny`, `--nz`  | Grid size per axis, in voxels (must be a multiple of 8) | `64`                |
| `--dx`                  | Uniform cell size, in meters                          | `1.0e-3`             |
| `--dt`                  | Timestep, in seconds (must satisfy the Courant limit for `dx`) | `1.5e-12`   |
| `--steps`               | Number of timesteps to run                            | `200`                |
| `--snapshot-every`      | Timesteps between snapshot writes                     | `20`                 |
| `--materials <PATH>`    | Backing file for the mmap'd material grid             | `materials.grid`     |
| `--output <PATH>`       | Direct I/O snapshot stream path                       | `wave_trajectory.bin`|
| `-h`, `--help`          | Print usage                                           |                      |

### Example

```sh
./target/release/wavefront --nx 128 --ny 128 --nz 128 \
    --steps 500 --snapshot-every 25 \
    --materials /mnt/nvme/materials.grid \
    --output /mnt/nvme/wave_trajectory.bin
```

`materials.grid` and `wave_trajectory.bin` are working files generated at
run time (see `.gitignore`) — point `--materials`/`--output` at your NVMe
mount for large grids rather than leaving the defaults in the repo checkout.

### Output format

`wave_trajectory.bin` is a raw concatenation of snapshots; each snapshot is
every `FieldBlock` in the grid, in block-major (Z, then Y, then X) order,
each block serialized as six back-to-back `f32` arrays (`Ex, Ey, Ez, Hx, Hy,
Hz`), 512 voxels per array (8x8x8 block, row-major with X fastest-varying).
