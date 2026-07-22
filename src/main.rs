//! `wavefront` -- CLI entry point, configuration parsing, and temporal loop
//! orchestration for the out-of-core 3D FDTD electromagnetic simulator.
//!
//! This binary is a thin driver over the `wavefront` library crate
//! (`src/lib.rs`), which owns the actual `layout`/`fdtd`/`engine`/`scene`/
//! `source`/`probe` modules -- shared with the `wavefront-view`
//! post-processing tool in `src/bin/`.
//!
//! Absent a `--scene` file, it falls back to a single demonstration
//! scenario -- a dielectric sphere embedded in vacuum -- excited by a
//! configurable point source (default: a Ricker wavelet on Ez at the
//! domain center) re-injected every timestep.
//!
//! ## Multiple sources and probes
//!
//! `engine::run` has always accepted a `&[Source]`/`&mut [Probe]` slice, but
//! until now the CLI only ever built exactly one of each. The individual
//! `--source-x/-y/-z/-component/-waveform/-freq/-amplitude` and
//! `--probe-x/-y/-z/-component/-freq/-start` flags still work exactly as
//! before and configure the *first* source/probe (auto-created on first
//! use); repeat the new `--source <key=value,...>` / `--probe
//! <key=value,...>` flags to add more. Both forms build into the same
//! underlying `Vec<SourceSpec>`/`Vec<ProbeSpec>`, so mixing them (e.g. the
//! simple flags for a primary source, `--source ...` for a couple more) is
//! well-defined, not an error.

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
    /// Empty means "no sources configured yet" -- [`build_sources`] falls
    /// back to a single default source in that case, matching this CLI's
    /// original single-source behavior exactly. Once any source has been
    /// configured (by either flag style), that fallback no longer applies.
    sources: Vec<SourceSpec>,
    /// Empty means no probes at all (unlike `sources`, there is no
    /// probe-by-default fallback -- probes are opt-in).
    probes: Vec<ProbeSpec>,
    materials_path: PathBuf,
    output_path: PathBuf,
}

/// CLI-selectable source waveform shape. Translated into a concrete
/// `source::Waveform` (with its `t0`/`spread` parameters derived from the
/// drive frequency) once `dt` is known -- see [`build_one_source`].
#[derive(Debug, Clone, Copy)]
enum WaveformKind {
    Gaussian,
    Sinusoid,
    Ricker,
}

/// One source's configuration, before its position/frequency defaults
/// (domain center, `1/(20*dt)`) are resolved -- either produced by the
/// `--source-*` shorthand flags (mutating `sources[0]`, auto-created) or
/// parsed whole from a `--source <key=value,...>` flag.
#[derive(Debug, Clone)]
struct SourceSpec {
    x: Option<usize>,
    y: Option<usize>,
    z: Option<usize>,
    component: FieldComponent,
    waveform: WaveformKind,
    freq: Option<f32>,
    amplitude: f32,
}

impl Default for SourceSpec {
    fn default() -> Self {
        Self {
            x: None,
            y: None,
            z: None,
            component: FieldComponent::Ez,
            waveform: WaveformKind::Ricker,
            freq: None,
            amplitude: 1.0,
        }
    }
}

/// One probe's configuration. Unlike [`SourceSpec`], `x`/`y`/`z`/`freqs`
/// have no usable default -- [`build_one_probe`] requires all of them.
#[derive(Debug, Clone, Default)]
struct ProbeSpec {
    x: Option<usize>,
    y: Option<usize>,
    z: Option<usize>,
    component: ProbeComponentSpec,
    freqs: Vec<f32>,
    start: f32,
}

/// Thin wrapper so [`ProbeSpec`] can `#[derive(Default)]` (`probe::FieldComponent`
/// itself has no `Default` impl, and shouldn't need one just for this).
#[derive(Debug, Clone, Copy)]
struct ProbeComponentSpec(probe::FieldComponent);

impl Default for ProbeComponentSpec {
    fn default() -> Self {
        Self(probe::FieldComponent::Ez)
    }
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
            sources: Vec::new(),
            probes: Vec::new(),
            materials_path: PathBuf::from("materials.grid"),
            output_path: PathBuf::from("wave_trajectory.bin"),
        }
    }
}

fn parse_source_component(v: &str) -> Result<FieldComponent, String> {
    match v {
        "ex" => Ok(FieldComponent::Ex),
        "ey" => Ok(FieldComponent::Ey),
        "ez" => Ok(FieldComponent::Ez),
        other => Err(format!("invalid source component: {other} (expected ex, ey, or ez)")),
    }
}

fn parse_probe_component(v: &str) -> Result<probe::FieldComponent, String> {
    match v {
        "ex" => Ok(probe::FieldComponent::Ex),
        "ey" => Ok(probe::FieldComponent::Ey),
        "ez" => Ok(probe::FieldComponent::Ez),
        "hx" => Ok(probe::FieldComponent::Hx),
        "hy" => Ok(probe::FieldComponent::Hy),
        "hz" => Ok(probe::FieldComponent::Hz),
        other => Err(format!(
            "invalid probe component: {other} (expected ex, ey, ez, hx, hy, or hz)"
        )),
    }
}

fn parse_waveform_kind(v: &str) -> Result<WaveformKind, String> {
    match v {
        "gaussian" => Ok(WaveformKind::Gaussian),
        "sinusoid" => Ok(WaveformKind::Sinusoid),
        "ricker" => Ok(WaveformKind::Ricker),
        other => Err(format!("invalid waveform: {other} (expected gaussian, sinusoid, or ricker)")),
    }
}

/// Parses one `--source <key=value,...>` flag's value into a full
/// [`SourceSpec`]. Recognized keys: `x`, `y`, `z`, `component`, `waveform`,
/// `freq`, `amplitude` -- any key not given keeps [`SourceSpec::default`]'s
/// value (so e.g. `--source x=10,y=10,z=10` is valid, using default
/// component/waveform/freq/amplitude).
fn parse_source_spec(s: &str) -> Result<SourceSpec, String> {
    let mut spec = SourceSpec::default();
    for pair in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (key, value) = pair
            .split_once('=')
            .ok_or_else(|| format!("--source: expected key=value, got '{pair}'"))?;
        let value = value.trim();
        match key.trim() {
            "x" => spec.x = Some(value.parse().map_err(|_| format!("--source: invalid x '{value}'"))?),
            "y" => spec.y = Some(value.parse().map_err(|_| format!("--source: invalid y '{value}'"))?),
            "z" => spec.z = Some(value.parse().map_err(|_| format!("--source: invalid z '{value}'"))?),
            "component" => spec.component = parse_source_component(value)?,
            "waveform" => spec.waveform = parse_waveform_kind(value)?,
            "freq" => spec.freq = Some(value.parse().map_err(|_| format!("--source: invalid freq '{value}'"))?),
            "amplitude" => {
                spec.amplitude = value.parse().map_err(|_| format!("--source: invalid amplitude '{value}'"))?
            }
            other => return Err(format!("--source: unknown key '{other}'")),
        }
    }
    Ok(spec)
}

/// Parses one `--probe <key=value,...>` flag's value into a full
/// [`ProbeSpec`]. Recognized keys: `x`, `y`, `z`, `component`, `freq`
/// (semicolon-separated, e.g. `freq=3e10;6e10` -- commas are already taken
/// by the outer key=value separator), `start`.
fn parse_probe_spec(s: &str) -> Result<ProbeSpec, String> {
    let mut spec = ProbeSpec::default();
    for pair in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (key, value) = pair
            .split_once('=')
            .ok_or_else(|| format!("--probe: expected key=value, got '{pair}'"))?;
        let value = value.trim();
        match key.trim() {
            "x" => spec.x = Some(value.parse().map_err(|_| format!("--probe: invalid x '{value}'"))?),
            "y" => spec.y = Some(value.parse().map_err(|_| format!("--probe: invalid y '{value}'"))?),
            "z" => spec.z = Some(value.parse().map_err(|_| format!("--probe: invalid z '{value}'"))?),
            "component" => spec.component = ProbeComponentSpec(parse_probe_component(value)?),
            "freq" => {
                spec.freqs = value
                    .split(';')
                    .map(|f| f.trim().parse().map_err(|_| format!("--probe: invalid freq '{f}'")))
                    .collect::<Result<Vec<f32>, String>>()?
            }
            "start" => spec.start = value.parse().map_err(|_| format!("--probe: invalid start '{value}'"))?,
            other => return Err(format!("--probe: unknown key '{other}'")),
        }
    }
    Ok(spec)
}

fn first_source_mut(config: &mut Config) -> &mut SourceSpec {
    if config.sources.is_empty() {
        config.sources.push(SourceSpec::default());
    }
    &mut config.sources[0]
}

fn first_probe_mut(config: &mut Config) -> &mut ProbeSpec {
    if config.probes.is_empty() {
        config.probes.push(ProbeSpec::default());
    }
    &mut config.probes[0]
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
         \x20   --source-x/-y/-z <N>   first source's voxel position [default: domain center]\n\
         \x20   --source-component <C> first source's field component: ex, ey, ez [default: ez]\n\
         \x20   --source-waveform <W>  first source's waveform: gaussian, sinusoid, or ricker [default: ricker]\n\
         \x20   --source-freq <HZ>     first source's drive frequency [default: 1/(20*dt)]\n\
         \x20   --source-amplitude <A> first source's peak amplitude [default: 1.0]\n\
         \x20   --source <SPEC>        add another source: key=value,... (x,y,z,component,waveform,freq,amplitude)\n\
         \x20   --probe-x/-y/-z <N>    first probe's voxel position; enables it together with --probe-freq\n\
         \x20   --probe-component <C>  first probe's field component: ex, ey, ez, hx, hy, hz [default: ez]\n\
         \x20   --probe-freq <HZ,...>  comma-separated frequencies (Hz) the first probe's running DFT tracks\n\
         \x20   --probe-start <SECONDS> first probe's ignore-samples-before time [default: 0.0]\n\
         \x20   --probe <SPEC>         add another probe: key=value,... (x,y,z,component,freq[;freq...],start)\n\
         \x20   --materials <PATH>     backing file for the mmap'd material grid [default: materials.grid]\n\
         \x20   --output <PATH>        Direct I/O snapshot stream path [default: wave_trajectory.bin]\n\
         \x20   -h, --help             print this message\n\n\
         Repeat --source/--probe to configure more than one; the individual\n\
         --source-*/--probe-* flags configure only the first. --probe's freq\n\
         list uses ';' (quote the whole value in your shell, e.g.\n\
         --probe \"x=10,y=10,z=10,freq=3e10;6e10\")."
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
            "--source-x" => {
                let v = parse_num("--source-x", next_value("--source-x")?)?;
                first_source_mut(&mut config).x = Some(v);
            }
            "--source-y" => {
                let v = parse_num("--source-y", next_value("--source-y")?)?;
                first_source_mut(&mut config).y = Some(v);
            }
            "--source-z" => {
                let v = parse_num("--source-z", next_value("--source-z")?)?;
                first_source_mut(&mut config).z = Some(v);
            }
            "--source-component" => {
                let v = parse_source_component(&next_value("--source-component")?)?;
                first_source_mut(&mut config).component = v;
            }
            "--source-waveform" => {
                let v = parse_waveform_kind(&next_value("--source-waveform")?)?;
                first_source_mut(&mut config).waveform = v;
            }
            "--source-freq" => {
                let v = parse_float("--source-freq", next_value("--source-freq")?)?;
                first_source_mut(&mut config).freq = Some(v);
            }
            "--source-amplitude" => {
                let v = parse_float("--source-amplitude", next_value("--source-amplitude")?)?;
                first_source_mut(&mut config).amplitude = v;
            }
            "--source" => {
                let spec = parse_source_spec(&next_value("--source")?)?;
                config.sources.push(spec);
            }
            "--probe-x" => {
                let v = parse_num("--probe-x", next_value("--probe-x")?)?;
                first_probe_mut(&mut config).x = Some(v);
            }
            "--probe-y" => {
                let v = parse_num("--probe-y", next_value("--probe-y")?)?;
                first_probe_mut(&mut config).y = Some(v);
            }
            "--probe-z" => {
                let v = parse_num("--probe-z", next_value("--probe-z")?)?;
                first_probe_mut(&mut config).z = Some(v);
            }
            "--probe-component" => {
                let v = parse_probe_component(&next_value("--probe-component")?)?;
                first_probe_mut(&mut config).component = ProbeComponentSpec(v);
            }
            "--probe-freq" => {
                let v = next_value("--probe-freq")?;
                let freqs = v
                    .split(',')
                    .map(|s| parse_float("--probe-freq", s.to_string()))
                    .collect::<Result<Vec<f32>, String>>()?;
                first_probe_mut(&mut config).freqs = freqs;
            }
            "--probe-start" => {
                let v = parse_float("--probe-start", next_value("--probe-start")?)?;
                first_probe_mut(&mut config).start = v;
            }
            "--probe" => {
                let spec = parse_probe_spec(&next_value("--probe")?)?;
                config.probes.push(spec);
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

/// Resolves one [`SourceSpec`] into a concrete [`Source`]: position
/// defaults to the domain center, frequency to `1/(20*dt)`, wherever the
/// spec left them unset, and the waveform *shape* selection is translated
/// into concrete `t0`/`spread` parameters derived from that frequency.
fn build_one_source(spec: &SourceSpec, dt: f32, dims: GridDims) -> Source {
    let x = spec.x.unwrap_or(dims.nx / 2);
    let y = spec.y.unwrap_or(dims.ny / 2);
    let z = spec.z.unwrap_or(dims.nz / 2);
    let freq_hz = spec.freq.unwrap_or(1.0 / (20.0 * dt));

    let waveform = match spec.waveform {
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
        component: spec.component,
        amplitude: spec.amplitude,
        waveform,
    }
}

/// Builds every configured source. An empty `config.sources` (no
/// `--source-*`/`--source` flags given at all) falls back to one default
/// source, matching this CLI's original single-source-by-default behavior.
fn build_sources(config: &Config, dims: GridDims) -> Vec<Source> {
    if config.sources.is_empty() {
        vec![build_one_source(&SourceSpec::default(), config.dt, dims)]
    } else {
        config
            .sources
            .iter()
            .map(|spec| build_one_source(spec, config.dt, dims))
            .collect()
    }
}

/// Resolves one [`ProbeSpec`] into a concrete [`Probe`]. Unlike sources,
/// position and frequency have no usable default -- both are required.
fn build_one_probe(spec: &ProbeSpec, dims: GridDims) -> Result<Probe, String> {
    let (Some(x), Some(y), Some(z)) = (spec.x, spec.y, spec.z) else {
        return Err(
            "each probe requires x, y, and z (via --probe-x/-y/-z, or x=/y=/z= in --probe)"
                .to_string(),
        );
    };
    if spec.freqs.is_empty() {
        return Err(
            "each probe requires at least one frequency (via --probe-freq, or freq= in --probe)"
                .to_string(),
        );
    }
    if x >= dims.nx || y >= dims.ny || z >= dims.nz {
        return Err(format!(
            "probe position ({x}, {y}, {z}) is outside the {}x{}x{} domain",
            dims.nx, dims.ny, dims.nz
        ));
    }

    Ok(Probe::new(x, y, z, spec.component.0, spec.freqs.clone(), spec.start))
}

/// Builds every configured probe (empty if none were configured -- probes
/// are opt-in, unlike sources).
fn build_probes(config: &Config, dims: GridDims) -> Result<Vec<Probe>, String> {
    config.probes.iter().map(|spec| build_one_probe(spec, dims)).collect()
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
    let sources = build_sources(&config, dims);
    for source in &sources {
        eprintln!(
            "wavefront: source at ({}, {}, {}) driving {:?}, {:?}",
            source.x, source.y, source.z, source.component, source.waveform
        );
    }

    let mut probes = build_probes(&config, dims)?;
    for probe in &probes {
        let (x, y, z) = probe.position();
        eprintln!(
            "wavefront: probe at ({x}, {y}, {z}) tracking {:?}",
            probe.component()
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
        &sources,
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
