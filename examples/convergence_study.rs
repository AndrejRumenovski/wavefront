//! Validates the Yee-lattice FDTD scheme's numerical phase velocity against
//! its own *exact, closed-form* dispersion relation, at several grid
//! resolutions.
//!
//! ## Why phase velocity, and why compare to a closed form (not just "it
//! gets better with resolution")
//!
//! An earlier version of this study measured the causal arrival time of a
//! broadband pulse at a probe point. That doesn't work: FDTD's numerical
//! dispersion means different frequency components of a broadband pulse
//! travel at different (resolution-dependent) speeds, so a wideband pulse
//! doesn't have a single well-defined "arrival time" to begin with -- it
//! disperses. Phase velocity, by contrast, is a well-defined, single-valued
//! quantity *per frequency*, and the Yee scheme has an exact, closed-form
//! prediction for it (Taflove & Hagness, *Computational Electrodynamics*,
//! ch. 4): for a plane wave of angular frequency `omega` propagating along
//! a grid axis with cell size `dx` and timestep `dt`,
//!
//! ```text
//! [sin(omega*dt/2) / (c*dt)]^2 = [sin(k*dx/2) / dx]^2
//! ```
//!
//! solving for the numerical wavenumber `k` gives the numerical phase
//! velocity `v_p = omega / k`, which differs from `c` by an amount that
//! shrinks as `O((dx/lambda)^2)` for fixed Courant number.
//!
//! This example *measures* the phase velocity empirically from a real
//! simulation, and separately *computes* the exact prediction from the
//! closed form above -- then checks that they agree. Matching a specific,
//! textbook quantitative prediction (not just "converges to something
//! reasonable") is a much stronger correctness check than the informal
//! qualitative kind, since it rules out compensating errors that might
//! otherwise still show *a* trend toward zero error.
//!
//! ## Setup: an exact 1D plane wave in a 3D solver
//!
//! Rather than a point source (whose near-field is analytically messy) or
//! a wide 3D domain (needed to keep transverse boundary reflections from
//! contaminating the measurement), this drives a full-transverse-plane
//! ("sheet") hard source: every voxel at a fixed `x` is forced to
//! `Ez = sin(omega*t)` every step. Since the domain is only one
//! `BLOCK_DIM`-wide in Y and Z, the field has *no* Y/Z variation anywhere,
//! and Y/Z neighbor lookups wrap periodically (implemented directly in this
//! example's serial loop) rather than hitting a boundary at all -- so the
//! only boundary that matters is the two ends of the long X axis, and the
//! measurement window is sized to finish well before a reflection from
//! either one can return to the probe.
//!
//! Run with:
//! ```sh
//! RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" \
//!     cargo +nightly run --release --example convergence_study
//! ```
//! Writes `validation/convergence_data.csv`; plot it with
//! `validation/plot_convergence.py`.

use wavefront::fdtd::{self, EUpdateNeighbors, HUpdateNeighbors};
use wavefront::layout::{
    CoeffGrid, FieldBlock, FieldGrid, GridDims, MaterialGrid, MaterialTable, BLOCK_DIM,
};

const SPEED_OF_LIGHT_M_PER_S: f32 = 299_792_458.0;

/// The exact Yee-scheme numerical phase velocity for a plane wave of
/// angular frequency `2*pi*freq_hz` propagating along a grid axis, per the
/// closed-form 1D dispersion relation (see module docs).
fn yee_1d_theoretical_phase_velocity(dx: f32, dt: f32, freq_hz: f32) -> f32 {
    let omega = 2.0 * std::f32::consts::PI * freq_hz;
    let rhs = (dx / (SPEED_OF_LIGHT_M_PER_S * dt)) * (omega * dt / 2.0).sin();
    let k = (2.0 / dx) * rhs.asin();
    omega / k
}

/// Runs the serial (single-threaded, PML-disabled, vacuum-only) plane-wave
/// timestep loop and returns the probe voxel's `Ez` value at every step.
///
/// Like the production engine, this reads neighbor blocks with plain,
/// safe `.clone()` calls rather than the engine's raw-pointer aliasing
/// trick -- clarity over performance, appropriate for a validation script.
#[allow(clippy::too_many_arguments)]
fn run_plane_wave(
    dims: GridDims,
    dx: f32,
    dt: f32,
    freq_hz: f32,
    x_source: usize,
    x_probe: usize,
    steps: usize,
) -> Vec<f32> {
    let (bx_n, by_n, bz_n) = dims.block_dims();
    debug_assert_eq!(by_n, 1, "this study assumes a single Y block (periodic wrap)");
    debug_assert_eq!(bz_n, 1, "this study assumes a single Z block (periodic wrap)");

    let path = std::env::temp_dir().join(format!(
        "wavefront_convergence_{}_{dx}.grid",
        std::process::id()
    ));
    let material_grid = MaterialGrid::create(&path, dims).expect("create scratch material grid");
    let table = MaterialTable::vacuum_filled(dt, dx);
    let coeff_grid = CoeffGrid::build(&material_grid, &table);
    let _ = std::fs::remove_file(&path);

    let mut field_grid = FieldGrid::zeroed(dims);
    let (bx_src, lx_src) = (x_source / BLOCK_DIM, x_source % BLOCK_DIM);
    let (bx_probe, lx_probe) = (x_probe / BLOCK_DIM, x_probe % BLOCK_DIM);
    let omega = 2.0 * std::f32::consts::PI * freq_hz;

    let mut trace = Vec::with_capacity(steps);

    for step in 0..steps {
        // ---- H update (Y/Z neighbors wrap periodically; X does not) ----
        for bz in 0..bz_n {
            for by in 0..by_n {
                for bx in 0..bx_n {
                    let idx = (bz * by_n + by) * bx_n + bx;
                    let plus_x = if bx + 1 < bx_n {
                        field_grid.block(bx + 1, by, bz).clone()
                    } else {
                        FieldBlock::ZERO
                    };
                    let plus_y = field_grid.block(bx, (by + 1) % by_n, bz).clone();
                    let plus_z = field_grid.block(bx, by, (bz + 1) % bz_n).clone();
                    let coeffs = &coeff_grid.blocks()[idx];
                    let center = field_grid.block_mut(bx, by, bz);
                    fdtd::update_h_field(
                        center,
                        HUpdateNeighbors {
                            plus_x: &plus_x,
                            plus_y: &plus_y,
                            plus_z: &plus_z,
                        },
                        coeffs,
                    );
                }
            }
        }

        // ---- E update ---------------------------------------------------
        for bz in 0..bz_n {
            for by in 0..by_n {
                for bx in 0..bx_n {
                    let idx = (bz * by_n + by) * bx_n + bx;
                    let minus_x = if bx > 0 {
                        field_grid.block(bx - 1, by, bz).clone()
                    } else {
                        FieldBlock::ZERO
                    };
                    let minus_y = field_grid.block(bx, (by + by_n - 1) % by_n, bz).clone();
                    let minus_z = field_grid.block(bx, by, (bz + bz_n - 1) % bz_n).clone();
                    let coeffs = &coeff_grid.blocks()[idx];
                    let center = field_grid.block_mut(bx, by, bz);
                    fdtd::update_e_field(
                        center,
                        EUpdateNeighbors {
                            minus_x: &minus_x,
                            minus_y: &minus_y,
                            minus_z: &minus_z,
                        },
                        coeffs,
                    );
                }
            }
        }

        // ---- hard sheet source: force every (y, z) at x_source ---------
        let t = (step as f32 + 1.0) * dt;
        let value = (omega * t).sin();
        let source_block = field_grid.block_mut(bx_src, 0, 0);
        for ly in 0..BLOCK_DIM {
            for lz in 0..BLOCK_DIM {
                source_block.ez[FieldBlock::local_index(lx_src, ly, lz)] = value;
            }
        }

        trace.push(field_grid.block(bx_probe, 0, 0).ez[FieldBlock::local_index(lx_probe, 0, 0)]);
    }

    trace
}

fn log_log_slope(xs: &[f32], ys: &[f32]) -> f32 {
    let (lx, ly): (Vec<f32>, Vec<f32>) = xs
        .iter()
        .zip(ys)
        .map(|(&x, &y)| (x.log10(), y.log10()))
        .unzip();
    let n = lx.len() as f32;
    let mean_x = lx.iter().sum::<f32>() / n;
    let mean_y = ly.iter().sum::<f32>() / n;
    let cov: f32 = lx.iter().zip(&ly).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
    let var: f32 = lx.iter().map(|x| (x - mean_x).powi(2)).sum();
    cov / var
}

struct Row {
    dx_m: f32,
    points_per_wavelength: f32,
    measured_phase_velocity: f32,
    theoretical_phase_velocity: f32,
    measured_relative_error: f32,
    theoretical_relative_error: f32,
}

fn main() {
    let freq_hz = 3.0e10_f32; // 30 GHz
    let wavelength_m = SPEED_OF_LIGHT_M_PER_S / freq_hz; // 10 mm
    let courant_number = 0.4_f32;

    // Fix the source-to-probe separation in WAVELENGTHS (not voxels): any
    // fixed-size startup transient from the hard source's sudden turn-on
    // takes a roughly resolution-independent number of *periods* to decay
    // at the probe. Earlier revisions of this study held the voxel
    // separation fixed instead, which meant higher-PPW (finer) runs
    // covered fewer wavelengths in that fixed voxel span -- shrinking the
    // ratio of "clean settled propagation" to "startup transient" exactly
    // as resolution improved, and swamping the genuine (and much smaller)
    // dispersion signal with startup-transient contamination. Holding
    // wavelengths-traveled fixed removes that confound: every resolution
    // gets the same relative settling margin, so the number of *voxels*
    // between source and probe (and the domain length, and the run length)
    // all scale up proportionally with points-per-wavelength instead.
    let periods_traveled = 8.0_f32;
    let points_per_wavelength_values = [10.0_f32, 15.0, 20.0, 30.0];
    let mut rows = Vec::with_capacity(points_per_wavelength_values.len());

    for &ppw in &points_per_wavelength_values {
        let dx = wavelength_m / ppw;
        let dt = courant_number * dx / SPEED_OF_LIGHT_M_PER_S;
        let period_s = 1.0 / freq_hz;
        let period_steps = ppw / courant_number;

        let n_voxels = (periods_traveled * ppw).round() as usize;
        let x_source = 3 * n_voxels;
        let x_probe = x_source + n_voxels;
        let nx8 = |v: usize| v.div_ceil(BLOCK_DIM) * BLOCK_DIM;
        let nx = nx8(7 * n_voxels);
        let steps = ((periods_traveled + 12.0) * period_steps).ceil() as usize;

        let dims = GridDims::new(nx, BLOCK_DIM, BLOCK_DIM);
        let expected_delay_s = (n_voxels as f32 * dx) / SPEED_OF_LIGHT_M_PER_S;

        let omega = 2.0 * std::f64::consts::PI * freq_hz as f64;
        let trace = run_plane_wave(dims, dx, dt, freq_hz, x_source, x_probe, steps);

        // Extract this frequency's phase from the settled window via
        // quadrature (in-phase/quadrature) demodulation: project the trace
        // onto `cos(omega t)` and `sin(omega t)` and average. This uses
        // *every* sample in the window rather than the handful that happen
        // to cross zero, which makes it far more robust to per-sample
        // interpolation noise than timing individual zero crossings (an
        // earlier revision of this study did the latter, and the residual
        // noise it left in was large enough to obscure the dispersion
        // trend being measured). Accumulated in f64 since this sums
        // hundreds of single-precision samples.
        let skip_before_s = expected_delay_s + 5.0 * period_s;
        let mut sum_cos = 0.0f64;
        let mut sum_sin = 0.0f64;
        let mut count = 0u32;
        for (i, &v) in trace.iter().enumerate() {
            let t = (i as f32 + 1.0) * dt;
            if t > skip_before_s {
                let phase = omega * t as f64;
                sum_cos += v as f64 * phase.cos();
                sum_sin += v as f64 * phase.sin();
                count += 1;
            }
        }
        assert!(
            count > 0,
            "no settled samples for PPW={ppw} -- widen the measurement window or check reflection timing"
        );
        let a = 2.0 * sum_cos / count as f64; // coefficient of cos(omega t)
        let b = 2.0 * sum_sin / count as f64; // coefficient of sin(omega t)

        // A retarded plane wave sin(omega*(t - delay)) expands as
        // sin(omega t)*cos(omega delay) - cos(omega t)*sin(omega delay),
        // i.e. a = -sin(omega delay), b = cos(omega delay) -- solve for
        // delay (mod one period), then pick the integer number of extra
        // periods `k` using the approximate (c-based) expected delay.
        let raw_delay = (-a).atan2(b) / omega;
        let k = ((expected_delay_s as f64 - raw_delay) / period_s as f64).round();
        let measured_delay_s = (raw_delay + k * period_s as f64) as f32;

        let measured_v = (n_voxels as f32 * dx) / measured_delay_s;
        let theory_v = yee_1d_theoretical_phase_velocity(dx, dt, freq_hz);

        rows.push(Row {
            dx_m: dx,
            points_per_wavelength: ppw,
            measured_phase_velocity: measured_v,
            theoretical_phase_velocity: theory_v,
            measured_relative_error: (measured_v - SPEED_OF_LIGHT_M_PER_S).abs() / SPEED_OF_LIGHT_M_PER_S,
            theoretical_relative_error: (theory_v - SPEED_OF_LIGHT_M_PER_S).abs() / SPEED_OF_LIGHT_M_PER_S,
        });
    }

    println!(
        "dx_m,points_per_wavelength,measured_phase_velocity,theoretical_phase_velocity,\
         measured_relative_error,theoretical_relative_error"
    );
    for r in &rows {
        println!(
            "{},{},{},{},{},{}",
            r.dx_m,
            r.points_per_wavelength,
            r.measured_phase_velocity,
            r.theoretical_phase_velocity,
            r.measured_relative_error,
            r.theoretical_relative_error
        );
    }

    let dxs: Vec<f32> = rows.iter().map(|r| r.dx_m).collect();
    let measured_errors: Vec<f32> = rows.iter().map(|r| r.measured_relative_error).collect();
    let theory_errors: Vec<f32> = rows.iter().map(|r| r.theoretical_relative_error).collect();
    let measured_slope = log_log_slope(&dxs, &measured_errors);
    let theory_slope = log_log_slope(&dxs, &theory_errors);

    eprintln!(
        "\nmeasured convergence order (log-log slope of measured error vs dx):    {measured_slope:.3}"
    );
    eprintln!(
        "theoretical convergence order (log-log slope of closed-form error vs dx): {theory_slope:.3}"
    );
    eprintln!("(Yee scheme theory predicts both should be ~2.0)");

    let max_discrepancy = rows
        .iter()
        .map(|r| (r.measured_relative_error - r.theoretical_relative_error).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "max |measured - theoretical| relative error across all resolutions: {max_discrepancy:.2e}"
    );

    // Hard pass/fail gate, not just a printed number: generous enough to
    // absorb the empirical measurement's real point-to-point noise (see
    // VALIDATION.md), tight enough to fail loudly if a future change to the
    // update equations breaks second-order convergence outright (e.g. a
    // sign error, a dropped term, or an accidentally first-order scheme).
    assert!(
        (1.5..2.6).contains(&measured_slope),
        "measured convergence order {measured_slope:.3} is far from the Yee scheme's theoretical \
         ~2.0 -- something in the update equations likely broke second-order accuracy"
    );
    assert!(
        max_discrepancy < 0.01,
        "measured phase velocity error diverges too far ({max_discrepancy:.2e}) from the \
         closed-form prediction -- the solver may no longer implement the documented dispersion relation"
    );

    std::fs::create_dir_all("validation").expect("create validation/ directory");
    let mut csv = String::from(
        "dx_m,points_per_wavelength,measured_phase_velocity,theoretical_phase_velocity,\
         measured_relative_error,theoretical_relative_error\n",
    );
    for r in &rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{}\n",
            r.dx_m,
            r.points_per_wavelength,
            r.measured_phase_velocity,
            r.theoretical_phase_velocity,
            r.measured_relative_error,
            r.theoretical_relative_error
        ));
    }
    std::fs::write("validation/convergence_data.csv", csv).expect("write validation/convergence_data.csv");
    eprintln!("wrote validation/convergence_data.csv");
}
