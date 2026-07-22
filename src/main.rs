//! `wavefront` -- CLI entry point, configuration parsing, and temporal loop
//! orchestration for the out-of-core 3D FDTD electromagnetic simulator.
//!
//! This binary is a thin driver over the `wavefront` library crate
//! (`src/lib.rs`), which owns the actual `layout`/`fdtd`/`engine`/`scene`/
//! `source` modules -- shared with the `wavefront-view` post-processing
//! tool in `src/bin/`.
//!
//! Absent a `--scene` file, it falls back to a single demonstration
//! scenario -- a dielectric sphere embedded in vacuum -- excited by a
//! configurable point source (default: a Ricker wavelet on Ez at the
//! domain center) re-injected every timestep.

use wavefront::layout::{
    CoeffGrid, FieldGrid, GridDims, MaterialGrid, MaterialId, MaterialTable, PmlConfig,
    PmlContext, BLOCK_DIM,
};
use wavefront::probe::{self, Probe};
use wavefront::source::{FieldComponent, Source, Waveform};
use wavefront::{engine, scene};
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
    /// Absorbing boundary layer thickness, in voxels, at each of the 6
    /// domain faces. `0` disables the PML and reverts to a zero-field
    /// (fully reflective) boundary.
    pml_thickness: usize,
    /// Optional plain-text scene file (see `src/scene.rs`); falls back to
    /// the hardcoded demo sphere if not given.
    scene_path: Option<PathBuf>,
    /// Source voxel position; defaults to the domain center if unset.
    source_pos: (Option<usize>, Option<usize>, Option<usize>),
    source_component: FieldComponent,
    source_waveform: WaveformKind,
    /// Source drive frequency, in Hz; defaults to `1 / (20 * dt)` (20
    /// samples per period) if unset.
    source_freq: Option<f32>,
    source_amplitude: f32,
    /// Probe voxel position; unset (all three unset) disables the probe
    /// entirely. All three or none must be given together.
    probe_pos: (Option<usize>, Option<usize>, Option<usize>),
    probe_component: probe::FieldComponent,
    /// Frequencies (Hz) the probe's running DFT tracks; empty disables the
    /// probe. Required together with `probe_pos` to enable one.
    probe_freqs: Vec<f32>,
    /// Simulation time (seconds) before which the probe's accumulators
    /// ignore samples, so a source's startup transient doesn't contaminate
    /// a frequency response that's only meaningful once the field has
    /// settled.
    probe_start: f32,
    materials_path: PathBuf,
    output_path: PathBuf,
}

/// CLI-selectable source waveform shape. Translated into a concrete
/// `source::Waveform` (with its `t0`/`spread` parameters derived from the
/// drive frequency) once `dt` is known -- see [`build_waveform`].
#[derive(Debug, Clone, Copy)]
enum WaveformKind {
    Gaussian,
    Sinusoid,
    Ricker,
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
            pml_thickness: PmlConfig::default().thickness,
            scene_path: None,
            source_pos: (None, None, None),
            source_component: FieldComponent::Ez,
            source_waveform: WaveformKind::Ricker,
            source_freq: None,
            source_amplitude: 1.0,
            probe_pos: (None, None, None),
            probe_component: probe::FieldComponent::Ez,
            probe_freqs: Vec::new(),
            probe_start: 0.0,
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
         \x20   --pml-thickness <N>    absorbing boundary depth in voxels, 0 disables it [default: 8]\n\
         \x20   --scene <PATH>         plain-text scene file (see src/scene.rs); omit for the demo sphere\n\
         \x20   --source-x/-y/-z <N>   source voxel position [default: domain center]\n\
         \x20   --source-component <C> field component the source drives: ex, ey, ez [default: ez]\n\
         \x20   --source-waveform <W>  gaussian, sinusoid, or ricker [default: ricker]\n\
         \x20   --source-freq <HZ>     source drive frequency [default: 1/(20*dt)]\n\
         \x20   --source-amplitude <A> source peak amplitude [default: 1.0]\n\
         \x20   --probe-x/-y/-z <N>    probe voxel position; enables the probe together with --probe-freq\n\
         \x20   --probe-component <C>  field component the probe tracks: ex, ey, ez, hx, hy, hz [default: ez]\n\
         \x20   --probe-freq <HZ,...>  comma-separated frequencies (Hz) the probe's running DFT tracks\n\
         \x20   --probe-start <SECONDS> simulation time before which the probe ignores samples [default: 0.0]\n\
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
            "--pml-thickness" => {
                config.pml_thickness = parse_num("--pml-thickness", next_value("--pml-thickness")?)?
            }
            "--scene" => config.scene_path = Some(PathBuf::from(next_value("--scene")?)),
            "--source-x" => config.source_pos.0 = Some(parse_num("--source-x", next_value("--source-x")?)?),
            "--source-y" => config.source_pos.1 = Some(parse_num("--source-y", next_value("--source-y")?)?),
            "--source-z" => config.source_pos.2 = Some(parse_num("--source-z", next_value("--source-z")?)?),
            "--source-component" => {
                let v = next_value("--source-component")?;
                config.source_component = match v.as_str() {
                    "ex" => FieldComponent::Ex,
                    "ey" => FieldComponent::Ey,
                    "ez" => FieldComponent::Ez,
                    other => return Err(format!("invalid --source-component: {other} (expected ex, ey, or ez)")),
                };
            }
            "--source-waveform" => {
                let v = next_value("--source-waveform")?;
                config.source_waveform = match v.as_str() {
                    "gaussian" => WaveformKind::Gaussian,
                    "sinusoid" => WaveformKind::Sinusoid,
                    "ricker" => WaveformKind::Ricker,
                    other => {
                        return Err(format!(
                            "invalid --source-waveform: {other} (expected gaussian, sinusoid, or ricker)"
                        ))
                    }
                };
            }
            "--source-freq" => config.source_freq = Some(parse_float("--source-freq", next_value("--source-freq")?)?),
            "--source-amplitude" => {
                config.source_amplitude = parse_float("--source-amplitude", next_value("--source-amplitude")?)?
            }
            "--probe-x" => config.probe_pos.0 = Some(parse_num("--probe-x", next_value("--probe-x")?)?),
            "--probe-y" => config.probe_pos.1 = Some(parse_num("--probe-y", next_value("--probe-y")?)?),
            "--probe-z" => config.probe_pos.2 = Some(parse_num("--probe-z", next_value("--probe-z")?)?),
            "--probe-component" => {
                let v = next_value("--probe-component")?;
                config.probe_component = match v.as_str() {
                    "ex" => probe::FieldComponent::Ex,
                    "ey" => probe::FieldComponent::Ey,
                    "ez" => probe::FieldComponent::Ez,
                    "hx" => probe::FieldComponent::Hx,
                    "hy" => probe::FieldComponent::Hy,
                    "hz" => probe::FieldComponent::Hz,
                    other => {
                        return Err(format!(
                            "invalid --probe-component: {other} (expected ex, ey, ez, hx, hy, or hz)"
                        ))
                    }
                };
            }
            "--probe-freq" => {
                let v = next_value("--probe-freq")?;
                config.probe_freqs = v
                    .split(',')
                    .map(|s| parse_float("--probe-freq", s.to_string()))
                    .collect::<Result<Vec<f32>, String>>()?;
            }
            "--probe-start" => {
                config.probe_start = parse_float("--probe-start", next_value("--probe-start")?)?
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

/// Builds the single configured [`Source`] from CLI options, resolving its
/// position to the domain center and its frequency to `1/(20*dt)` wherever
/// the user didn't override them, and translating the CLI's waveform
/// *shape* selection into concrete `t0`/`spread` parameters derived from
/// that frequency.
fn build_source(config: &Config, dims: GridDims) -> Source {
    let x = config.source_pos.0.unwrap_or(dims.nx / 2);
    let y = config.source_pos.1.unwrap_or(dims.ny / 2);
    let z = config.source_pos.2.unwrap_or(dims.nz / 2);
    let freq_hz = config.source_freq.unwrap_or(1.0 / (20.0 * config.dt));

    let waveform = match config.source_waveform {
        WaveformKind::Sinusoid => Waveform::Sinusoid { freq_hz },
        WaveformKind::Ricker => Waveform::RickerWavelet {
            peak_freq_hz: freq_hz,
            t0: 1.0 / freq_hz,
        },
        WaveformKind::Gaussian => {
            let spread = 0.5 / freq_hz;
            Waveform::GaussianPulse {
                t0: 4.0 * spread,
                spread,
            }
        }
    };

    Source {
        x,
        y,
        z,
        component: config.source_component,
        amplitude: config.source_amplitude,
        waveform,
    }
}

/// Builds the configured [`Probe`] from CLI options, if `--probe-x/-y/-z`
/// and `--probe-freq` were both given (all-or-nothing: a partial
/// specification -- e.g. a position with no frequency -- is a usage error,
/// not silently ignored).
fn build_probe(config: &Config, dims: GridDims) -> Result<Option<Probe>, String> {
    let (px, py, pz) = config.probe_pos;
    let has_pos = px.is_some() || py.is_some() || pz.is_some();
    let has_freq = !config.probe_freqs.is_empty();

    if !has_pos && !has_freq {
        return Ok(None);
    }
    let (Some(x), Some(y), Some(z)) = (px, py, pz) else {
        return Err(
            "--probe-x, --probe-y, and --probe-z must all be given together to enable a probe"
                .to_string(),
        );
    };
    if !has_freq {
        return Err("--probe-freq is required (comma-separated Hz values) to enable a probe".to_string());
    }
    if x >= dims.nx || y >= dims.ny || z >= dims.nz {
        return Err(format!(
            "probe position ({x}, {y}, {z}) is outside the {}x{}x{} domain",
            dims.nx, dims.ny, dims.nz
        ));
    }

    Ok(Some(Probe::new(
        x,
        y,
        z,
        config.probe_component,
        config.probe_freqs.clone(),
        config.probe_start,
    )))
}

fn run(config: Config) -> Result<(), String> {
    let dims = GridDims::new(config.nx, config.ny, config.nz);

    eprintln!(
        "wavefront: {}x{}x{} voxels ({} MiB field state, dx={:.3e} m, dt={:.3e} s, {} steps, \
         pml={} voxels)",
        dims.nx,
        dims.ny,
        dims.nz,
        (dims.block_count() * std::mem::size_of::<wavefront::layout::FieldBlock>()) / (1024 * 1024),
        config.dx,
        config.dt,
        config.num_steps,
        config.pml_thickness
    );

    let mut material_grid = MaterialGrid::create(&config.materials_path, dims)
        .map_err(|e| format!("failed to create material grid at {:?}: {e}", config.materials_path))?;
    let mut material_table = MaterialTable::vacuum_filled(config.dt, config.dx);

    match &config.scene_path {
        Some(path) => {
            let scene = scene::Scene::load(path)?;
            let n_materials = scene.voxelize(&mut material_grid, &mut material_table, config.dt, config.dx)?;
            eprintln!("wavefront: loaded scene {path:?} ({n_materials} distinct materials)");
        }
        None => {
            voxelize_demo_sphere(&mut material_grid);
            material_table.set_material(MATERIAL_DIELECTRIC, 4.0, 1.0, 0.0, config.dt, config.dx);
        }
    }
    material_grid
        .flush()
        .map_err(|e| format!("failed to flush material grid: {e}"))?;

    let coeff_grid = CoeffGrid::build(&material_grid, &material_table);

    let mut field_grid = FieldGrid::zeroed(dims);
    let source = build_source(&config, dims);
    eprintln!(
        "wavefront: source at ({}, {}, {}) driving {:?}, {:?}",
        source.x, source.y, source.z, source.component, source.waveform
    );

    let mut probes: Vec<Probe> = build_probe(&config, dims)?.into_iter().collect();
    if let Some(probe) = probes.first() {
        let (x, y, z) = probe.position();
        eprintln!(
            "wavefront: probe at ({x}, {y}, {z}) tracking {:?} at {} frequency(s), recording from t={}s",
            probe.component(),
            config.probe_freqs.len(),
            config.probe_start
        );
    }

    let pml_config = PmlConfig {
        thickness: config.pml_thickness,
        ..PmlConfig::default()
    };
    let (pml_context, mut pml_aux_grid) = PmlContext::build(dims, &pml_config, config.dt, config.dx);
    let pml_context_ref = (config.pml_thickness > 0).then_some(&pml_context);

    let engine_config = engine::EngineConfig {
        num_steps: config.num_steps,
        snapshot_every: config.snapshot_every.max(1),
        dt: config.dt,
        output_path: config.output_path,
    };

    engine::run(
        &mut field_grid,
        &coeff_grid,
        pml_context_ref,
        &mut pml_aux_grid,
        &[source],
        &mut probes,
        &engine_config,
    )
    .map_err(|e| format!("simulation run failed: {e}"))?;

    for probe in &probes {
        let (x, y, z) = probe.position();
        eprintln!("wavefront: probe ({x}, {y}, {z}) {:?} frequency response:", probe.component());
        for response in probe.spectrum() {
            eprintln!(
                "  {:.4e} Hz: amplitude {:.6e}, phase {:.4} rad",
                response.freq_hz, response.amplitude, response.phase_rad
            );
        }
    }

    Ok(())
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
