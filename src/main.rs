//! `wavefront` -- CLI entry point, configuration parsing, and temporal loop
//! orchestration for the out-of-core 3D FDTD electromagnetic simulator.
//!
//! This binary wires together the three library modules:
//!   - `layout`: mmap'd material grid, cache-aligned AoSoA field grid.
//!   - `fdtd`: the SIMD Yee-lattice curl update kernels.
//!   - `engine`: spatial decomposition, rayon scheduling, crossbeam halo
//!     exchange, and the `io_uring` snapshot writer.
//!
//! It sets up a single demonstration scenario -- a dielectric sphere
//! embedded in vacuum, excited by a Gaussian point source at the domain
//! center -- and runs the timestep loop to completion.

#![feature(portable_simd)]

mod engine;
mod fdtd;
mod layout;

use layout::{CoeffGrid, FieldGrid, GridDims, MaterialGrid, MaterialId, MaterialTable, BLOCK_DIM};
use std::path::PathBuf;
use std::process::ExitCode;

struct Config {
    /// Grid dimensions in voxels along each axis (each must be a multiple
    /// of `BLOCK_DIM`).
    nx: usize,
    ny: usize,
    nz: usize,
    /// Uniform cell size, in meters.
    dx: f32,
    /// Timestep, in seconds. Must satisfy the Courant stability limit for
    /// `dx`; the default is chosen conservatively below that limit.
    dt: f32,
    num_steps: usize,
    snapshot_every: usize,
    materials_path: PathBuf,
    output_path: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            nx: 64,
            ny: 64,
            nz: 64,
            dx: 1.0e-3,
            dt: 1.5e-12,
            num_steps: 200,
            snapshot_every: 20,
            materials_path: PathBuf::from("materials.grid"),
            output_path: PathBuf::from("wave_trajectory.bin"),
        }
    }
}

fn print_usage() {
    eprintln!(
        "wavefront -- out-of-core 3D FDTD electromagnetic simulator\n\n\
         USAGE:\n    wavefront [OPTIONS]\n\n\
         OPTIONS:\n\
         \x20   --nx <N>               grid size along X in voxels (multiple of {BLOCK_DIM}) [default: 64]\n\
         \x20   --ny <N>               grid size along Y in voxels (multiple of {BLOCK_DIM}) [default: 64]\n\
         \x20   --nz <N>               grid size along Z in voxels (multiple of {BLOCK_DIM}) [default: 64]\n\
         \x20   --dx <METERS>          uniform cell size [default: 1.0e-3]\n\
         \x20   --dt <SECONDS>         timestep [default: 1.5e-12]\n\
         \x20   --steps <N>            number of timesteps to run [default: 200]\n\
         \x20   --snapshot-every <N>   timesteps between snapshot writes [default: 20]\n\
         \x20   --materials <PATH>     backing file for the mmap'd material grid [default: materials.grid]\n\
         \x20   --output <PATH>        Direct I/O snapshot stream path [default: wave_trajectory.bin]\n\
         \x20   -h, --help             print this message"
    );
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config::default();
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
        let parse_float = |name: &str, s: String| -> Result<f32, String> {
            s.parse::<f32>()
                .map_err(|_| format!("invalid number for {name}: {s}"))
        };

        match arg.as_str() {
            "--nx" => config.nx = parse_num("--nx", next_value("--nx")?)?,
            "--ny" => config.ny = parse_num("--ny", next_value("--ny")?)?,
            "--nz" => config.nz = parse_num("--nz", next_value("--nz")?)?,
            "--dx" => config.dx = parse_float("--dx", next_value("--dx")?)?,
            "--dt" => config.dt = parse_float("--dt", next_value("--dt")?)?,
            "--steps" => config.num_steps = parse_num("--steps", next_value("--steps")?)?,
            "--snapshot-every" => {
                config.snapshot_every = parse_num("--snapshot-every", next_value("--snapshot-every")?)?
            }
            "--materials" => config.materials_path = PathBuf::from(next_value("--materials")?),
            "--output" => config.output_path = PathBuf::from(next_value("--output")?),
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }

    Ok(config)
}

/// Material ID assigned to the demonstration dielectric sphere.
const MATERIAL_DIELECTRIC: MaterialId = MaterialId(1);

/// Voxelizes a sphere of the dielectric material at the domain center into
/// `grid`, radius one quarter of the smallest axis extent.
fn voxelize_demo_sphere(grid: &mut MaterialGrid) {
    let dims = grid.dims();
    let (cx, cy, cz) = (
        dims.nx as f32 / 2.0,
        dims.ny as f32 / 2.0,
        dims.nz as f32 / 2.0,
    );
    let radius = dims.nx.min(dims.ny).min(dims.nz) as f32 / 4.0;
    let radius_sq = radius * radius;

    for z in 0..dims.nz {
        for y in 0..dims.ny {
            for x in 0..dims.nx {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                let dz = z as f32 + 0.5 - cz;
                if dx * dx + dy * dy + dz * dz <= radius_sq {
                    grid.set_material_at(x, y, z, MATERIAL_DIELECTRIC);
                }
            }
        }
    }
}

/// Injects a Gaussian-in-space impulse into Ez at the domain center as the
/// initial condition, giving the solver something physically meaningful to
/// propagate.
fn inject_initial_pulse(field_grid: &mut FieldGrid, dims: GridDims) {
    let (cx, cy, cz) = (dims.nx / 2, dims.ny / 2, dims.nz / 2);
    let sigma: f32 = 2.0;
    let sigma_sq = sigma * sigma;

    let spread = (3.0 * sigma).ceil() as isize;
    for dz in -spread..=spread {
        for dy in -spread..=spread {
            for dx in -spread..=spread {
                let x = cx as isize + dx;
                let y = cy as isize + dy;
                let z = cz as isize + dz;
                if x < 0 || y < 0 || z < 0 {
                    continue;
                }
                let (x, y, z) = (x as usize, y as usize, z as usize);
                if x >= dims.nx || y >= dims.ny || z >= dims.nz {
                    continue;
                }
                let r_sq = (dx * dx + dy * dy + dz * dz) as f32;
                let amplitude = (-r_sq / (2.0 * sigma_sq)).exp();

                let bx = x / BLOCK_DIM;
                let by = y / BLOCK_DIM;
                let bz = z / BLOCK_DIM;
                let (lx, ly, lz) = (x % BLOCK_DIM, y % BLOCK_DIM, z % BLOCK_DIM);
                let local = layout::FieldBlock::local_index(lx, ly, lz);
                field_grid.block_mut(bx, by, bz).ez[local] = amplitude;
            }
        }
    }
}

fn run(config: Config) -> Result<(), String> {
    let dims = GridDims::new(config.nx, config.ny, config.nz);

    eprintln!(
        "wavefront: {}x{}x{} voxels ({} MiB field state, dx={:.3e} m, dt={:.3e} s, {} steps)",
        dims.nx,
        dims.ny,
        dims.nz,
        (dims.block_count() * std::mem::size_of::<layout::FieldBlock>()) / (1024 * 1024),
        config.dx,
        config.dt,
        config.num_steps
    );

    let mut material_grid = MaterialGrid::create(&config.materials_path, dims)
        .map_err(|e| format!("failed to create material grid at {:?}: {e}", config.materials_path))?;
    voxelize_demo_sphere(&mut material_grid);
    material_grid
        .flush()
        .map_err(|e| format!("failed to flush material grid: {e}"))?;

    let mut material_table = MaterialTable::vacuum_filled(config.dt, config.dx);
    material_table.set_material(MATERIAL_DIELECTRIC, 4.0, 1.0, 0.0, config.dt, config.dx);

    let coeff_grid = CoeffGrid::build(&material_grid, &material_table);

    let mut field_grid = FieldGrid::zeroed(dims);
    inject_initial_pulse(&mut field_grid, dims);

    let engine_config = engine::EngineConfig {
        num_steps: config.num_steps,
        snapshot_every: config.snapshot_every.max(1),
        output_path: config.output_path,
    };

    engine::run(&mut field_grid, &coeff_grid, &engine_config)
        .map_err(|e| format!("simulation run failed: {e}"))
}

fn main() -> ExitCode {
    let config = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("wavefront: {e}\n");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    match run(config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("wavefront: {e}");
            ExitCode::FAILURE
        }
    }
}
