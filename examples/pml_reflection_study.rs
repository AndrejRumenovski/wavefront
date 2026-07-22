//! Measures the CPML absorbing boundary's *actual* reflection coefficient,
//! as a function of PML thickness, and compares it against the closed-form
//! target used to derive its grading (`PmlConfig::target_reflection`).
//!
//! ## Why this is a separate study from `convergence_study.rs`
//!
//! The PML was previously verified only qualitatively (an energy-decay
//! comparison: with PML off, total field energy in a point-source run
//! bounces/grows; with PML on, it decays monotonically once the wave
//! reaches the boundary). That's evidence the PML does *something*
//! absorbing, but it doesn't quantify *how much* -- it can't distinguish "a
//! good PML" from "a mediocre PML that's merely better than a bare
//! zero-field wall". This study measures the actual reflection coefficient
//! and checks it both shrinks with PML thickness (the theoretically
//! expected trend) and stays small in absolute terms.
//!
//! ## Method: two-run subtraction
//!
//! A single simulation's probe reading near a PML-terminated boundary
//! contains *both* the incident wave and whatever the PML reflects back --
//! there's no way to separate them from one trace alone. So this runs the
//! same driven sheet-source plane wave (see `convergence_study.rs` for why
//! a full-transverse-plane hard source gives a clean 1D plane wave in a
//! domain only `BLOCK_DIM` voxels wide in Y/Z) twice, at identical source-
//! to-probe geometry:
//!
//!   - **Test run**: a short domain with CPML at both X faces (the
//!     thickness under test). The probe sees incident + reflected.
//!   - **Reference run**: PML disabled entirely, with both X boundaries
//!     pushed far enough away that no reflection from *either* one can
//!     possibly return to the probe within the run. The probe sees pure
//!     incident field.
//!
//! Because the medium is homogeneous vacuum and both runs use the same
//! source waveform and the same source-to-probe distance, causality
//! guarantees the two probe traces are identical up until a reflection
//! first reaches the probe -- so subtracting the reference trace from the
//! test trace, sample by sample, isolates the reflected wave alone.
//! Quadrature demodulation (see `convergence_study.rs`) then extracts each
//! wave's steady-state amplitude from its respective trace, and the
//! reflection coefficient is just the ratio of the two.
//!
//! ## Why Y/Z can stay "periodic-wrap" even with a real 3D PmlProfile1D
//!
//! `src/layout.rs`'s `PmlProfile1D`/`PmlContext` machinery is written for a
//! genuine finite domain on every axis, not this study's periodic-Y/Z
//! trick -- so this file builds its own X-only profile directly (skipping
//! `PmlContext::build`, which would also grade Y and Z) and passes
//! `PmlCoeffs::IDENTITY` windows for Y and Z to the PML-aware kernels. That
//! is exact, not an approximation: the CPML correction only ever modifies a
//! *raw derivative* along that axis (see `src/fdtd.rs`), and this study's
//! field is by construction translationally uniform across Y and Z (a full-
//! transverse-plane source, periodic wrap), so every raw Y/Z derivative is
//! identically zero everywhere -- multiplying zero by any `kappa`/`a`/`b`
//! still gives zero. Only the X-axis profile ever does real work here.
//!
//! Run with:
//! ```sh
//! RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2" \
//!     cargo +nightly run --release --example pml_reflection_study
//! ```
//! Writes `validation/pml_reflection_data.csv`; plot it with
//! `validation/plot_pml_reflection.py`.

use wavefront::fdtd::{
    self, update_e_field_pml, update_h_field_pml, EUpdateNeighbors, HUpdateNeighbors,
};
use wavefront::layout::{
    CoeffGrid, FieldBlock, FieldGrid, GridDims, MaterialGrid, MaterialTable, PmlAux, PmlCoeffs,
    PmlConfig, PmlContext, PmlProfile1D, BLOCK_DIM,
};

const SPEED_OF_LIGHT_M_PER_S: f32 = 299_792_458.0;

/// Runs the serial (single-threaded, vacuum-only) sheet-source plane-wave
/// timestep loop and returns the probe voxel's `Ez` value at every step.
///
/// `pml_thickness_voxels == 0` disables the PML entirely (plain kernels,
/// zero-field boundary at both X ends) -- this is what the reference run
/// uses. A nonzero value grades a CPML layer of that thickness at *both*
/// X faces (symmetric, like `PmlProfile1D::build` always is), absorbing
/// both the source's leftward-launched wave and whatever reaches the right
/// face.
#[allow(clippy::too_many_arguments)]
fn run_plane_wave(
    dims: GridDims,
    dx: f32,
    dt: f32,
    freq_hz: f32,
    x_source: usize,
    x_probe: usize,
    steps: usize,
    pml_thickness_voxels: usize,
) -> Vec<f32> {
    let (bx_n, by_n, bz_n) = dims.block_dims();
    debug_assert_eq!(by_n, 1, "this study assumes a single Y block (periodic wrap)");
    debug_assert_eq!(bz_n, 1, "this study assumes a single Z block (periodic wrap)");

    let path = std::env::temp_dir().join(format!(
        "wavefront_pml_reflection_{}_{dx}_{pml_thickness_voxels}.grid",
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

    // X-only PML profile (see module docs for why Y/Z stay identity), a
    // whole number of BLOCK_DIM-sized block layers deep at each end, capped
    // so the two ends can never overlap.
    let thickness_blocks = pml_thickness_voxels.div_ceil(BLOCK_DIM).min(bx_n / 2);
    let thickness_voxels_eff = thickness_blocks * BLOCK_DIM;
    let pml_config = PmlConfig {
        thickness: pml_thickness_voxels,
        ..PmlConfig::default()
    };
    let profile_x = PmlProfile1D::build(dims.nx, &pml_config, dt, dx, thickness_voxels_eff);
    let identity_window = [PmlCoeffs::IDENTITY; BLOCK_DIM];
    let mut aux: Vec<PmlAux> = vec![PmlAux::ZERO; bx_n];
    let in_shell =
        |bx: usize| thickness_blocks > 0 && (bx < thickness_blocks || bx >= bx_n - thickness_blocks);

    let mut trace = Vec::with_capacity(steps);

    // `bx` also drives `field_grid.block(bx, ..)`/`coeff_grid.blocks()[bx]`
    // below, so clippy's `needless_range_loop` (which only sees the `aux`
    // use) doesn't apply cleanly here -- same situation as
    // `update_h_field_pml`/`update_e_field_pml` in `src/fdtd.rs`.
    #[allow(clippy::needless_range_loop)]
    for step in 0..steps {
        // ---- H update (Y/Z wrap onto the same sole block; X does not) --
        for bx in 0..bx_n {
            let plus_x = if bx + 1 < bx_n {
                field_grid.block(bx + 1, 0, 0).clone()
            } else {
                FieldBlock::ZERO
            };
            let plus_y = field_grid.block(bx, 0, 0).clone();
            let plus_z = field_grid.block(bx, 0, 0).clone();
            let coeffs = &coeff_grid.blocks()[bx];
            let center = field_grid.block_mut(bx, 0, 0);
            let nbrs = HUpdateNeighbors {
                plus_x: &plus_x,
                plus_y: &plus_y,
                plus_z: &plus_z,
            };
            if in_shell(bx) {
                let px = PmlContext::axis_window(&profile_x, bx);
                update_h_field_pml(center, nbrs, coeffs, &px, &identity_window, &identity_window, &mut aux[bx]);
            } else {
                fdtd::update_h_field(center, nbrs, coeffs);
            }
        }

        // ---- E update ---------------------------------------------------
        for bx in 0..bx_n {
            let minus_x = if bx > 0 {
                field_grid.block(bx - 1, 0, 0).clone()
            } else {
                FieldBlock::ZERO
            };
            let minus_y = field_grid.block(bx, 0, 0).clone();
            let minus_z = field_grid.block(bx, 0, 0).clone();
            let coeffs = &coeff_grid.blocks()[bx];
            let center = field_grid.block_mut(bx, 0, 0);
            let nbrs = EUpdateNeighbors {
                minus_x: &minus_x,
                minus_y: &minus_y,
                minus_z: &minus_z,
            };
            if in_shell(bx) {
                let px = PmlContext::axis_window(&profile_x, bx);
                update_e_field_pml(center, nbrs, coeffs, &px, &identity_window, &identity_window, &mut aux[bx]);
            } else {
                fdtd::update_e_field(center, nbrs, coeffs);
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

/// Extracts a settled sinusoid's steady-state amplitude from `trace` via
/// quadrature (in-phase/quadrature) demodulation over every sample past
/// `skip_before_s` -- see `convergence_study.rs` for why this is far more
/// robust to per-sample noise than timing zero crossings.
fn quadrature_amplitude(trace: &[f32], dt: f32, omega: f64, skip_before_s: f32) -> f64 {
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
        "no settled samples -- widen the measurement window or check the skip_before_s timing"
    );
    let a = 2.0 * sum_cos / count as f64;
    let b = 2.0 * sum_sin / count as f64;
    (a * a + b * b).sqrt()
}

/// Rounds `v` up to the next multiple of `BLOCK_DIM`.
fn nx8(v: usize) -> usize {
    v.div_ceil(BLOCK_DIM) * BLOCK_DIM
}

struct Row {
    thickness_blocks: usize,
    thickness_voxels: usize,
    reflection_coeff: f64,
}

fn main() {
    let freq_hz = 3.0e10_f32; // 30 GHz, same as convergence_study.rs
    let wavelength_m = SPEED_OF_LIGHT_M_PER_S / freq_hz;
    let courant_number = 0.4_f32;
    let ppw = 20.0_f32; // fixed resolution -- this study sweeps PML thickness, not dx
    let dx = wavelength_m / ppw;
    let dt = courant_number * dx / SPEED_OF_LIGHT_M_PER_S;
    let period_s = 1.0 / freq_hz;

    let probe_offset_voxels = (3.0 * ppw).round() as usize; // source -> probe
    let right_gap_voxels = (2.0 * ppw).round() as usize; // probe -> right PML face, test run

    let thicknesses_blocks = [1usize, 2, 4, 8];
    let mut rows = Vec::with_capacity(thicknesses_blocks.len());

    for &tb in &thicknesses_blocks {
        let thickness_voxels = tb * BLOCK_DIM;

        // ---- test run: short domain, CPML at both X faces --------------
        let x_source_test = thickness_voxels + (2.0 * ppw).round() as usize;
        let x_probe_test = x_source_test + probe_offset_voxels;
        let x_pml_face_test = x_probe_test + right_gap_voxels;
        let nx_test = nx8(x_pml_face_test + thickness_voxels);

        // Time budget: incident settle, plus the round trip from the probe
        // out to the PML face and back, plus a few periods for the
        // reflected component itself to settle into steady state, plus a
        // multi-period measurement window at the end.
        let incident_delay_s = probe_offset_voxels as f32 * dx / SPEED_OF_LIGHT_M_PER_S;
        let round_trip_voxels = 2 * (x_pml_face_test - x_probe_test);
        let round_trip_s = round_trip_voxels as f32 * dx / SPEED_OF_LIGHT_M_PER_S;
        let settle_periods = 6.0_f32;
        let skip_before_s = incident_delay_s + round_trip_s + settle_periods * period_s;
        let measurement_periods = 10.0_f32;
        let steps = ((skip_before_s + measurement_periods * period_s) / dt).ceil() as usize + 4;

        let dims_test = GridDims::new(nx_test, BLOCK_DIM, BLOCK_DIM);
        let test_trace = run_plane_wave(
            dims_test, dx, dt, freq_hz, x_source_test, x_probe_test, steps, thickness_voxels,
        );

        // ---- reference run: no PML, both boundaries pushed out of reach -
        //
        // `required_distance_voxels` is how far a boundary must sit from
        // the probe for its (100%-reflective, zero-field) round trip to
        // exceed the full `steps` duration plus a couple of periods of
        // margin. Both runs share the same dx/dt/freq, so this bound is
        // independent of the test run's own geometry -- only `steps`
        // matters. Because the medium is homogeneous vacuum, causality only
        // cares about the *source-to-probe distance* (kept identical to the
        // test run via `probe_offset_voxels`), not the absolute coordinate
        // origin -- so the reference run is free to place its source
        // wherever gives it roomy margins on both sides.
        let required_distance_voxels = ((steps as f32 * dt + 2.0 * period_s)
            * SPEED_OF_LIGHT_M_PER_S
            / dx)
            .ceil() as usize
            + BLOCK_DIM;
        let x_probe_ref = required_distance_voxels + probe_offset_voxels;
        let x_source_ref = x_probe_ref - probe_offset_voxels;
        let nx_ref = nx8(x_probe_ref + required_distance_voxels);

        let dims_ref = GridDims::new(nx_ref, BLOCK_DIM, BLOCK_DIM);
        let reference_trace = run_plane_wave(
            dims_ref, dx, dt, freq_hz, x_source_ref, x_probe_ref, steps, 0,
        );

        // ---- isolate the reflected wave and extract both amplitudes ----
        let omega = 2.0 * std::f64::consts::PI * freq_hz as f64;
        let incident_amp = quadrature_amplitude(&reference_trace, dt, omega, skip_before_s);
        let diff: Vec<f32> = test_trace
            .iter()
            .zip(&reference_trace)
            .map(|(t, r)| t - r)
            .collect();
        let reflected_amp = quadrature_amplitude(&diff, dt, omega, skip_before_s);
        let reflection_coeff = reflected_amp / incident_amp;

        rows.push(Row {
            thickness_blocks: tb,
            thickness_voxels,
            reflection_coeff,
        });
    }

    println!("thickness_blocks,thickness_voxels,reflection_coeff");
    for r in &rows {
        println!(
            "{},{},{}",
            r.thickness_blocks, r.thickness_voxels, r.reflection_coeff
        );
    }

    eprintln!(
        "\ntarget reflection coefficient (PmlConfig::default): {:.2e}",
        PmlConfig::default().target_reflection
    );
    for r in &rows {
        eprintln!(
            "thickness {:>3} voxels ({} blocks): measured |R| = {:.3e}",
            r.thickness_voxels, r.thickness_blocks, r.reflection_coeff
        );
    }

    // Hard pass/fail gates, mirroring convergence_study.rs's philosophy:
    // generous enough to absorb real measurement noise, tight enough to
    // fail loudly if the PML implementation regresses.
    //
    // 1. Monotonic trend: each successive (thicker) layer should reflect
    //    less than the previous one. This is the qualitative signature a
    //    working graded absorber must show -- a real bug (e.g. a sign
    //    error in the recursive convolution) would far more likely produce
    //    a flat or non-monotonic curve than an accidentally-monotonic wrong
    //    one.
    for pair in rows.windows(2) {
        assert!(
            pair[1].reflection_coeff < pair[0].reflection_coeff,
            "reflection coefficient did not decrease from {}-voxel PML ({:.3e}) to {}-voxel PML \
             ({:.3e}) -- thicker PML should reflect less",
            pair[0].thickness_voxels,
            pair[0].reflection_coeff,
            pair[1].thickness_voxels,
            pair[1].reflection_coeff
        );
    }

    // 2. Absolute bound on the thickest layer tested: comfortably above the
    //    closed-form target (1e-6 in PmlConfig::default -- unrealistic to
    //    hit exactly on a discretized, staircased grid) but far below "not
    //    really absorbing" (a broken PML backed by a PEC wall reflects
    //    close to 100%).
    let thickest = rows.last().expect("at least one thickness tested");
    assert!(
        thickest.reflection_coeff < 1.0e-2,
        "thickest PML tested ({} voxels) still reflects {:.3e} -- expected well under 1e-2",
        thickest.thickness_voxels,
        thickest.reflection_coeff
    );

    std::fs::create_dir_all("validation").expect("create validation/ directory");
    let mut csv = String::from("thickness_blocks,thickness_voxels,reflection_coeff\n");
    for r in &rows {
        csv.push_str(&format!(
            "{},{},{}\n",
            r.thickness_blocks, r.thickness_voxels, r.reflection_coeff
        ));
    }
    std::fs::write("validation/pml_reflection_data.csv", csv)
        .expect("write validation/pml_reflection_data.csv");
    eprintln!("\nwrote validation/pml_reflection_data.csv");
}
