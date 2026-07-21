//! `wavefront-view` -- a minimal post-processing tool that turns one 2D
//! slice of one snapshot out of a `wave_trajectory.bin` file into a viewable
//! image.
//!
//! `wave_trajectory.bin` has no header (see `src/engine.rs::serialize_snapshot`
//! and `src/layout.rs::FieldBlock`'s doc comment for the exact on-disk
//! layout), so this tool needs the same `--nx`/`--ny`/`--nz` you ran the
//! simulation with to make sense of the raw bytes.
//!
//! Images are written as binary PPM (`.ppm`, the "P6" format): a tiny,
//! trivially-specified, uncompressed format that needs zero extra
//! dependencies to write by hand -- deliberately, since this crate keeps
//! its dependency set small and pinned (see `Cargo.toml`). Most image
//! viewers, GIMP, and ImageMagick's `convert`/`magick` all read it directly;
//! convert to PNG with `magick slice.ppm slice.png` if you need one.
//!
//! The file is memory-mapped (via `memmap2`, already a dependency) rather
//! than read into a `Vec` -- consistent with the rest of the crate's
//! zero-copy philosophy, and it means only the handful of pages this tool
//! actually touches ever get faulted in, regardless of how large the
//! trajectory file is.

use memmap2::Mmap;
use std::fs::File;
use std::path::PathBuf;
use std::process::ExitCode;
use wavefront::layout::{FieldBlock, GridDims, BLOCK_DIM, VOXELS_PER_BLOCK};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy)]
enum Component {
    Ex,
    Ey,
    Ez,
    Hx,
    Hy,
    Hz,
    /// Sum of squares of all 6 components -- always non-negative and
    /// nonzero wherever the wave has reached, so it's a good default that
    /// needs no sign-aware colormap.
    Energy,
}

impl Component {
    /// Index of this component's `[f32; VOXELS_PER_BLOCK]` array within
    /// `FieldBlock`, matching its `repr(C)` declaration order. `None` for
    /// `Energy`, which reads all six.
    fn field_index(self) -> Option<usize> {
        match self {
            Component::Ex => Some(0),
            Component::Ey => Some(1),
            Component::Ez => Some(2),
            Component::Hx => Some(3),
            Component::Hy => Some(4),
            Component::Hz => Some(5),
            Component::Energy => None,
        }
    }

    fn is_signed(self) -> bool {
        !matches!(self, Component::Energy)
    }
}

struct Config {
    input: PathBuf,
    nx: usize,
    ny: usize,
    nz: usize,
    snapshot: usize,
    axis: Axis,
    slice: Option<usize>,
    component: Component,
    output: PathBuf,
}

fn print_usage() {
    eprintln!(
        "wavefront-view -- render one 2D slice of a wave_trajectory.bin snapshot as a PPM image\n\n\
         USAGE:\n    wavefront-view --input <PATH> --nx <N> --ny <N> --nz <N> [OPTIONS]\n\n\
         REQUIRED:\n\
         \x20   --input <PATH>       wave_trajectory.bin (or equivalent) to read\n\
         \x20   --nx/--ny/--nz <N>   grid dimensions the simulation was run with\n\n\
         OPTIONS:\n\
         \x20   --snapshot <N>       which snapshot to render, 0-indexed [default: 0]\n\
         \x20   --axis <x|y|z>       which axis to hold fixed (slice normal) [default: z]\n\
         \x20   --slice <N>          index along --axis to slice at [default: middle]\n\
         \x20   --component <C>      ex, ey, ez, hx, hy, hz, or energy [default: energy]\n\
         \x20   --output <PATH>      output PPM path [default: slice.ppm]\n\
         \x20   -h, --help           print this message"
    );
}

fn parse_args() -> Result<Config, String> {
    let mut input: Option<PathBuf> = None;
    let mut nx: Option<usize> = None;
    let mut ny: Option<usize> = None;
    let mut nz: Option<usize> = None;
    let mut snapshot = 0usize;
    let mut axis = Axis::Z;
    let mut slice: Option<usize> = None;
    let mut component = Component::Energy;
    let mut output = PathBuf::from("slice.ppm");

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut next_value = |name: &str| -> Result<String, String> {
            args.next()
                .ok_or_else(|| format!("missing value for {name}"))
        };
        let parse_num = |name: &str, s: String| -> Result<usize, String> {
            s.parse::<usize>()
                .map_err(|_| format!("invalid integer for {name}: {s}"))
        };

        match arg.as_str() {
            "--input" => input = Some(PathBuf::from(next_value("--input")?)),
            "--nx" => nx = Some(parse_num("--nx", next_value("--nx")?)?),
            "--ny" => ny = Some(parse_num("--ny", next_value("--ny")?)?),
            "--nz" => nz = Some(parse_num("--nz", next_value("--nz")?)?),
            "--snapshot" => snapshot = parse_num("--snapshot", next_value("--snapshot")?)?,
            "--axis" => {
                axis = match next_value("--axis")?.as_str() {
                    "x" => Axis::X,
                    "y" => Axis::Y,
                    "z" => Axis::Z,
                    other => return Err(format!("invalid --axis: {other} (expected x, y, or z)")),
                }
            }
            "--slice" => slice = Some(parse_num("--slice", next_value("--slice")?)?),
            "--component" => {
                component = match next_value("--component")?.as_str() {
                    "ex" => Component::Ex,
                    "ey" => Component::Ey,
                    "ez" => Component::Ez,
                    "hx" => Component::Hx,
                    "hy" => Component::Hy,
                    "hz" => Component::Hz,
                    "energy" => Component::Energy,
                    other => {
                        return Err(format!(
                            "invalid --component: {other} (expected ex, ey, ez, hx, hy, hz, or energy)"
                        ))
                    }
                }
            }
            "--output" => output = PathBuf::from(next_value("--output")?),
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }

    Ok(Config {
        input: input.ok_or("--input is required")?,
        nx: nx.ok_or("--nx is required")?,
        ny: ny.ok_or("--ny is required")?,
        nz: nz.ok_or("--nz is required")?,
        snapshot,
        axis,
        slice,
        component,
        output,
    })
}

/// Reads one field component at voxel `(x, y, z)` out of `snapshot_bytes`
/// (the byte slice for exactly one snapshot's worth of blocks), given the
/// grid's block dimensions.
fn read_component(
    snapshot_bytes: &[u8],
    bx_n: usize,
    by_n: usize,
    x: usize,
    y: usize,
    z: usize,
    field_index: usize,
) -> f32 {
    let (bx, by, bz) = (x / BLOCK_DIM, y / BLOCK_DIM, z / BLOCK_DIM);
    let (lx, ly, lz) = (x % BLOCK_DIM, y % BLOCK_DIM, z % BLOCK_DIM);
    let block_index = (bz * by_n + by) * bx_n + bx;
    let local = FieldBlock::local_index(lx, ly, lz);

    let block_bytes = std::mem::size_of::<FieldBlock>();
    let component_bytes = VOXELS_PER_BLOCK * std::mem::size_of::<f32>();
    let offset =
        block_index * block_bytes + field_index * component_bytes + local * std::mem::size_of::<f32>();

    let bytes: [u8; 4] = snapshot_bytes[offset..offset + 4]
        .try_into()
        .expect("computed offset is always in bounds for a validated snapshot");
    f32::from_le_bytes(bytes)
}

fn sample(snapshot_bytes: &[u8], bx_n: usize, by_n: usize, x: usize, y: usize, z: usize, component: Component) -> f32 {
    match component.field_index() {
        Some(idx) => read_component(snapshot_bytes, bx_n, by_n, x, y, z, idx),
        None => (0..6)
            .map(|idx| {
                let v = read_component(snapshot_bytes, bx_n, by_n, x, y, z, idx);
                v * v
            })
            .sum(),
    }
}

/// Maps a normalized value `t` (in `[-1, 1]` for signed components, `[0, 1]`
/// for energy) to an RGB pixel: white at zero, blending to blue for
/// negative and red for positive (or just white-to-red for energy, since
/// it's never negative).
fn colormap(t: f32, signed: bool) -> [u8; 3] {
    let lerp = |a: u8, b: u8, f: f32| (a as f32 + (b as f32 - a as f32) * f).round() as u8;
    const WHITE: [u8; 3] = [255, 255, 255];
    const RED: [u8; 3] = [214, 39, 40];
    const BLUE: [u8; 3] = [31, 119, 180];

    let (from, to, f) = if signed && t < 0.0 {
        (WHITE, BLUE, (-t).clamp(0.0, 1.0))
    } else {
        (WHITE, RED, t.clamp(0.0, 1.0))
    };

    [
        lerp(from[0], to[0], f),
        lerp(from[1], to[1], f),
        lerp(from[2], to[2], f),
    ]
}

fn write_ppm(path: &std::path::Path, width: usize, height: usize, pixels: &[[u8; 3]]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::io::BufWriter::new(File::create(path)?);
    write!(file, "P6\n{width} {height}\n255\n")?;
    for pixel in pixels {
        file.write_all(pixel)?;
    }
    Ok(())
}

fn run(config: Config) -> Result<(), String> {
    let dims = GridDims::new(config.nx, config.ny, config.nz);
    let (bx_n, by_n, bz_n) = dims.block_dims();

    let file = File::open(&config.input).map_err(|e| format!("failed to open {:?}: {e}", config.input))?;
    // SAFETY: `Mmap::map` is unsafe because the file could in principle be
    // truncated or modified by another process while mapped, which would
    // turn later reads into a SIGBUS/torn read. This is a short-lived,
    // read-only, single-shot CLI tool reading a completed simulation's
    // output file, not a long-running process racing a concurrent writer,
    // so that hazard is not a realistic concern here.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("failed to mmap {:?}: {e}", config.input))?;

    let block_bytes = std::mem::size_of::<FieldBlock>();
    let snapshot_bytes = bx_n * by_n * bz_n * block_bytes;
    if snapshot_bytes == 0 || mmap.len() % snapshot_bytes != 0 {
        return Err(format!(
            "{:?} ({} bytes) is not a whole number of {}x{}x{} snapshots ({} bytes each) -- \
             check --nx/--ny/--nz match the run that produced this file",
            config.input,
            mmap.len(),
            dims.nx,
            dims.ny,
            dims.nz,
            snapshot_bytes
        ));
    }
    let num_snapshots = mmap.len() / snapshot_bytes;
    if config.snapshot >= num_snapshots {
        return Err(format!(
            "--snapshot {} out of range: file has {num_snapshots} snapshot(s)",
            config.snapshot
        ));
    }
    let snapshot_bytes_slice =
        &mmap[config.snapshot * snapshot_bytes..(config.snapshot + 1) * snapshot_bytes];

    let (width, height, slice_index, extent_along_axis) = match config.axis {
        Axis::X => (dims.ny, dims.nz, config.slice.unwrap_or(dims.nx / 2), dims.nx),
        Axis::Y => (dims.nx, dims.nz, config.slice.unwrap_or(dims.ny / 2), dims.ny),
        Axis::Z => (dims.nx, dims.ny, config.slice.unwrap_or(dims.nz / 2), dims.nz),
    };
    if slice_index >= extent_along_axis {
        return Err(format!(
            "--slice {slice_index} out of range for axis {:?} (extent {extent_along_axis})",
            config.axis
        ));
    }

    let mut values = vec![0.0f32; width * height];
    for v in 0..height {
        for u in 0..width {
            let (x, y, z) = match config.axis {
                Axis::X => (slice_index, u, v),
                Axis::Y => (u, slice_index, v),
                Axis::Z => (u, v, slice_index),
            };
            values[v * width + u] = sample(snapshot_bytes_slice, bx_n, by_n, x, y, z, config.component);
        }
    }

    let max_abs = values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    let signed = config.component.is_signed();
    let pixels: Vec<[u8; 3]> = values
        .iter()
        .map(|&v| {
            let t = if max_abs > 0.0 { v / max_abs } else { 0.0 };
            colormap(t, signed)
        })
        .collect();

    write_ppm(&config.output, width, height, &pixels)
        .map_err(|e| format!("failed to write {:?}: {e}", config.output))?;

    eprintln!(
        "wavefront-view: wrote {:?} ({width}x{height}, snapshot {}/{num_snapshots}, axis {:?} @ {slice_index}, \
         component {:?}, max |value| = {max_abs:e})",
        config.output, config.snapshot, config.axis, config.component
    );
    Ok(())
}

fn main() -> ExitCode {
    let config = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("wavefront-view: {e}\n");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    match run(config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("wavefront-view: {e}");
            ExitCode::FAILURE
        }
    }
}
