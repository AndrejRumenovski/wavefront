//! SIMD-vectorized Yee-lattice Maxwell curl-equation update kernels.
//!
//! Wavefront advances the electromagnetic field with the classic leapfrog
//! Yee scheme (Yee, 1966): H is updated a half-timestep ahead of E, using
//! central-difference curls on the staggered lattice. Every kernel in this
//! file operates on exactly one cache-aligned [`FieldBlock`] (see
//! `src/layout.rs`) plus the handful of face-neighbor blocks its curl
//! stencil reads across the block boundary.
//!
//! `BLOCK_DIM` (8, see `src/layout.rs`) is chosen to equal the width of an
//! AVX2 `f32x8` vector register, so every X-row inside a block is loaded,
//! updated, and stored as a single SIMD instruction with no scalar
//! remainder loop.
//!
//! ## Why only face neighbors, and why raw row loads
//!
//! The Yee update for a field component only ever needs its two transverse
//! neighbors offset by one voxel in each of the other two axes (a 6-point
//! stencil), never a diagonal neighbor. Differences along Y and Z are
//! "free" -- they're just a different row within the same block, or a row
//! in a face-adjacent block, loaded with an ordinary contiguous
//! [`f32x8::from_slice`]. Differences along X, the SIMD lane axis, need the
//! row shifted by one lane; since that shift can cross a block edge, we
//! build the shifted operand on the stack (no heap allocation) by borrowing
//! exactly one lane from the appropriate neighbor block -- see
//! [`shifted_row_plus`] and [`shifted_row_minus`].

use crate::layout::{FieldBlock, MaterialCoeffs, PmlAux, PmlCoeffs, BLOCK_DIM, VOXELS_PER_BLOCK};
use std::simd::f32x8;

/// Builds `v` such that `v[i] == row_at_x(i + 1)` for `i in 0..BLOCK_DIM`,
/// i.e. `row` shifted one lane towards +X, with the value that would fall
/// off the end of the block (`i == BLOCK_DIM - 1`) borrowed from
/// `next_first`, the first lane of the corresponding row in the `+X`
/// neighbor block.
///
/// Built entirely on the stack (a fixed-size `[f32; BLOCK_DIM + 1]`), so
/// this never allocates.
#[inline(always)]
fn shifted_row_plus(row: &[f32; BLOCK_DIM], next_first: f32) -> f32x8 {
    let mut padded = [0.0f32; BLOCK_DIM + 1];
    padded[..BLOCK_DIM].copy_from_slice(row);
    padded[BLOCK_DIM] = next_first;
    f32x8::from_slice(&padded[1..BLOCK_DIM + 1])
}

/// Builds `v` such that `v[i] == row_at_x(i - 1)` for `i in 0..BLOCK_DIM`,
/// i.e. `row` shifted one lane towards -X, with the value that would fall
/// off the start of the block (`i == 0`) borrowed from `prev_last`, the last
/// lane of the corresponding row in the `-X` neighbor block.
#[inline(always)]
fn shifted_row_minus(row: &[f32; BLOCK_DIM], prev_last: f32) -> f32x8 {
    let mut padded = [0.0f32; BLOCK_DIM + 1];
    padded[0] = prev_last;
    padded[1..].copy_from_slice(row);
    f32x8::from_slice(&padded[0..BLOCK_DIM])
}

/// Loads one contiguous X-row of a field component out of a block.
#[inline(always)]
fn load_row(component: &[f32; VOXELS_PER_BLOCK], row_base: usize) -> f32x8 {
    f32x8::from_slice(&component[row_base..row_base + BLOCK_DIM])
}

/// Gathers the per-voxel `Ca`/`Cb` (or `Da`/`Db`) coefficient for one X-row
/// out of the block's per-voxel [`MaterialCoeffs`] array into a SIMD vector.
///
/// This is a small scalar gather (materials are rarely uniform within a
/// block, so it cannot be a single vector load), but it stays entirely on
/// the stack and touches only the `BLOCK_DIM` entries this row needs.
#[inline(always)]
fn gather_row(
    coeffs: &[MaterialCoeffs; VOXELS_PER_BLOCK],
    row_base: usize,
    select: fn(&MaterialCoeffs) -> f32,
) -> f32x8 {
    let mut lanes = [0.0f32; BLOCK_DIM];
    for (i, lane) in lanes.iter_mut().enumerate() {
        *lane = select(&coeffs[row_base + i]);
    }
    f32x8::from_array(lanes)
}

/// Stores a SIMD row back into one contiguous X-row of a field component.
#[inline(always)]
fn store_row(component: &mut [f32; VOXELS_PER_BLOCK], row_base: usize, v: f32x8) {
    v.copy_to_slice(&mut component[row_base..row_base + BLOCK_DIM]);
}

/// Borrows one contiguous X-row of a psi-memory component as a fixed-size
/// array reference, for the CPML correction helpers below.
#[inline(always)]
fn psi_row_mut(component: &mut [f32; VOXELS_PER_BLOCK], row_base: usize) -> &mut [f32; BLOCK_DIM] {
    (&mut component[row_base..row_base + BLOCK_DIM])
        .try_into()
        .unwrap()
}

// =============================================================================
// CPML STRETCHED-COORDINATE DERIVATIVE CORRECTION
// =============================================================================
//
// See `src/layout.rs`'s CPML section for the derivation. Both helpers below
// implement the same one-tap recursive convolution:
//
//   psi_new = b * psi_old + a * raw_derivative
//   corrected_derivative = raw_derivative / kappa + psi_new
//
// They differ only in whether `b`/`a`/`inv_kappa` are per-lane (X, the SIMD
// axis) or a single scalar broadcast across all 8 lanes (Y/Z, since every
// lane in a Y- or Z-derivative row shares the same Y or Z position).

/// Builds the per-lane `b`, `a`, and `1/kappa` vectors for a block's X-axis
/// window, once per kernel call (not per row) -- the X profile depends only
/// on the block's `bx`, so it's identical for every row inside the block.
#[inline(always)]
fn pml_x_vectors(window: &[PmlCoeffs; BLOCK_DIM]) -> (f32x8, f32x8, f32x8) {
    let mut b = [0.0f32; BLOCK_DIM];
    let mut a = [0.0f32; BLOCK_DIM];
    let mut inv_kappa = [0.0f32; BLOCK_DIM];
    for i in 0..BLOCK_DIM {
        b[i] = window[i].b;
        a[i] = window[i].a;
        inv_kappa[i] = window[i].inv_kappa;
    }
    (
        f32x8::from_array(b),
        f32x8::from_array(a),
        f32x8::from_array(inv_kappa),
    )
}

/// Applies the CPML correction to a raw X-direction derivative, where each
/// of the 8 SIMD lanes has its own coefficient (one per X position).
#[inline(always)]
fn pml_correct_x(
    raw: f32x8,
    b: f32x8,
    a: f32x8,
    inv_kappa: f32x8,
    psi: &mut [f32; BLOCK_DIM],
) -> f32x8 {
    let psi_old = f32x8::from_array(*psi);
    let psi_new = b * psi_old + a * raw;
    psi_new.copy_to_slice(psi);
    raw * inv_kappa + psi_new
}

/// Applies the CPML correction to a raw Y- or Z-direction derivative, where
/// `b`/`a`/`inv_kappa` are a single scalar shared by the whole row.
#[inline(always)]
fn pml_correct_scalar(
    raw: f32x8,
    b: f32,
    a: f32,
    inv_kappa: f32,
    psi: &mut [f32; BLOCK_DIM],
) -> f32x8 {
    let psi_old = f32x8::from_array(*psi);
    let psi_new = f32x8::splat(b) * psi_old + f32x8::splat(a) * raw;
    psi_new.copy_to_slice(psi);
    raw * f32x8::splat(inv_kappa) + psi_new
}

// =============================================================================
// H-FIELD UPDATE
// =============================================================================

/// Face-neighbor blocks whose boundary-facing E-field rows this block's
/// H-field update needs to read across the block seam. The Yee H-update
/// uses *forward* differences of E, so only the `+X`/`+Y`/`+Z` neighbors are
/// ever needed.
pub struct HUpdateNeighbors<'a> {
    pub plus_x: &'a FieldBlock,
    pub plus_y: &'a FieldBlock,
    pub plus_z: &'a FieldBlock,
}

/// Advances `center`'s Hx/Hy/Hz one half-timestep, in place, using the
/// current (already fully updated) E field of `center` and its `+X`/`+Y`/`+Z`
/// neighbors:
///
/// ```text
/// Hx += -Db * (dEz/dy - dEy/dz)
/// Hy += -Db * (dEx/dz - dEz/dx)
/// Hz += -Db * (dEy/dx - dEx/dy)
/// ```
///
/// (`Da` is folded in as a multiplicative decay on the old H value; for a
/// lossless medium `Da == 1` and this reduces to the textbook update.)
///
/// Iterates one X-row (`BLOCK_DIM` voxels) at a time so every derivative,
/// multiply, and store is a single `f32x8` vector instruction.
pub fn update_h_field(
    center: &mut FieldBlock,
    nbrs: HUpdateNeighbors,
    coeffs: &[MaterialCoeffs; VOXELS_PER_BLOCK],
) {
    for lz in 0..BLOCK_DIM {
        for ly in 0..BLOCK_DIM {
            let row_base = FieldBlock::local_index(0, ly, lz);

            let ex = load_row(&center.ex, row_base);
            let ey = load_row(&center.ey, row_base);
            let ez = load_row(&center.ez, row_base);

            // ---- +Y row of Ez (dHx term) and +Y row of Ex (dHz term) ----
            let (ez_py, ex_py) = if ly + 1 < BLOCK_DIM {
                let b = FieldBlock::local_index(0, ly + 1, lz);
                (load_row(&center.ez, b), load_row(&center.ex, b))
            } else {
                let b = FieldBlock::local_index(0, 0, lz);
                (load_row(&nbrs.plus_y.ez, b), load_row(&nbrs.plus_y.ex, b))
            };

            // ---- +Z row of Ey (dHx term) and +Z row of Ex (dHy term) ----
            let (ey_pz, ex_pz) = if lz + 1 < BLOCK_DIM {
                let b = FieldBlock::local_index(0, ly, lz + 1);
                (load_row(&center.ey, b), load_row(&center.ex, b))
            } else {
                let b = FieldBlock::local_index(0, ly, 0);
                (load_row(&nbrs.plus_z.ey, b), load_row(&nbrs.plus_z.ex, b))
            };

            // ---- +X-shifted Ez (dHy term) and Ey (dHz term) -------------
            let ez_row: [f32; BLOCK_DIM] = center.ez[row_base..row_base + BLOCK_DIM]
                .try_into()
                .unwrap();
            let ey_row: [f32; BLOCK_DIM] = center.ey[row_base..row_base + BLOCK_DIM]
                .try_into()
                .unwrap();
            let plus_x_row_base = FieldBlock::local_index(0, ly, lz);
            let ez_px = shifted_row_plus(&ez_row, nbrs.plus_x.ez[plus_x_row_base]);
            let ey_px = shifted_row_plus(&ey_row, nbrs.plus_x.ey[plus_x_row_base]);

            // ---- curls ---------------------------------------------------
            let curl_hx = (ez_py - ez) - (ey_pz - ey); // dEz/dy - dEy/dz
            let curl_hy = (ex_pz - ex) - (ez_px - ez); // dEx/dz - dEz/dx
            let curl_hz = (ey_px - ey) - (ex_py - ex); // dEy/dx - dEx/dy

            let da = gather_row(coeffs, row_base, |c| c.da.to_f32());
            let db = gather_row(coeffs, row_base, |c| c.db.to_f32());

            let hx = load_row(&center.hx, row_base);
            let hy = load_row(&center.hy, row_base);
            let hz = load_row(&center.hz, row_base);

            store_row(&mut center.hx, row_base, da * hx - db * curl_hx);
            store_row(&mut center.hy, row_base, da * hy - db * curl_hy);
            store_row(&mut center.hz, row_base, da * hz - db * curl_hz);
        }
    }
}

/// CPML-aware variant of [`update_h_field`] for blocks inside the PML
/// shell: identical Yee update, except each raw derivative that needs
/// stretched-coordinate correction is passed through [`pml_correct_x`] or
/// [`pml_correct_scalar`] before it enters the curl, using `aux`'s psi
/// memory for that term and the per-axis coefficient windows in
/// `pml_x`/`pml_y`/`pml_z`.
///
/// `pml_x`/`pml_y`/`pml_z` are this block's `BLOCK_DIM`-wide coefficient
/// windows (one [`PmlCoeffs`] per local voxel position along that axis),
/// built once per call by `src/engine.rs` via
/// `PmlContext::axis_window`. Passing [`PmlCoeffs::IDENTITY`] for an axis
/// that isn't part of this block's PML shell makes that axis's correction
/// an exact no-op, which is how corner/edge blocks touching more than one
/// face are handled without any special-casing here.
#[allow(clippy::too_many_arguments)]
pub fn update_h_field_pml(
    center: &mut FieldBlock,
    nbrs: HUpdateNeighbors,
    coeffs: &[MaterialCoeffs; VOXELS_PER_BLOCK],
    pml_x: &[PmlCoeffs; BLOCK_DIM],
    pml_y: &[PmlCoeffs; BLOCK_DIM],
    pml_z: &[PmlCoeffs; BLOCK_DIM],
    aux: &mut PmlAux,
) {
    let (px_b, px_a, px_inv_kappa) = pml_x_vectors(pml_x);

    for lz in 0..BLOCK_DIM {
        let pz = pml_z[lz];
        for ly in 0..BLOCK_DIM {
            let py = pml_y[ly];
            let row_base = FieldBlock::local_index(0, ly, lz);

            let ex = load_row(&center.ex, row_base);
            let ey = load_row(&center.ey, row_base);
            let ez = load_row(&center.ez, row_base);

            let (ez_py, ex_py) = if ly + 1 < BLOCK_DIM {
                let b = FieldBlock::local_index(0, ly + 1, lz);
                (load_row(&center.ez, b), load_row(&center.ex, b))
            } else {
                let b = FieldBlock::local_index(0, 0, lz);
                (load_row(&nbrs.plus_y.ez, b), load_row(&nbrs.plus_y.ex, b))
            };

            let (ey_pz, ex_pz) = if lz + 1 < BLOCK_DIM {
                let b = FieldBlock::local_index(0, ly, lz + 1);
                (load_row(&center.ey, b), load_row(&center.ex, b))
            } else {
                let b = FieldBlock::local_index(0, ly, 0);
                (load_row(&nbrs.plus_z.ey, b), load_row(&nbrs.plus_z.ex, b))
            };

            let ez_row: [f32; BLOCK_DIM] = center.ez[row_base..row_base + BLOCK_DIM]
                .try_into()
                .unwrap();
            let ey_row: [f32; BLOCK_DIM] = center.ey[row_base..row_base + BLOCK_DIM]
                .try_into()
                .unwrap();
            let plus_x_row_base = FieldBlock::local_index(0, ly, lz);
            let ez_px = shifted_row_plus(&ez_row, nbrs.plus_x.ez[plus_x_row_base]);
            let ey_px = shifted_row_plus(&ey_row, nbrs.plus_x.ey[plus_x_row_base]);

            // ---- raw derivatives, each CPML-corrected in place ----------
            let d_ez_dy = pml_correct_scalar(ez_py - ez, py.b, py.a, py.inv_kappa, psi_row_mut(&mut aux.psi_ez_dy, row_base));
            let d_ex_dy = pml_correct_scalar(ex_py - ex, py.b, py.a, py.inv_kappa, psi_row_mut(&mut aux.psi_ex_dy, row_base));
            let d_ey_dz = pml_correct_scalar(ey_pz - ey, pz.b, pz.a, pz.inv_kappa, psi_row_mut(&mut aux.psi_ey_dz, row_base));
            let d_ex_dz = pml_correct_scalar(ex_pz - ex, pz.b, pz.a, pz.inv_kappa, psi_row_mut(&mut aux.psi_ex_dz, row_base));
            let d_ez_dx = pml_correct_x(ez_px - ez, px_b, px_a, px_inv_kappa, psi_row_mut(&mut aux.psi_ez_dx, row_base));
            let d_ey_dx = pml_correct_x(ey_px - ey, px_b, px_a, px_inv_kappa, psi_row_mut(&mut aux.psi_ey_dx, row_base));

            // ---- curls ---------------------------------------------------
            let curl_hx = d_ez_dy - d_ey_dz;
            let curl_hy = d_ex_dz - d_ez_dx;
            let curl_hz = d_ey_dx - d_ex_dy;

            let da = gather_row(coeffs, row_base, |c| c.da.to_f32());
            let db = gather_row(coeffs, row_base, |c| c.db.to_f32());

            let hx = load_row(&center.hx, row_base);
            let hy = load_row(&center.hy, row_base);
            let hz = load_row(&center.hz, row_base);

            store_row(&mut center.hx, row_base, da * hx - db * curl_hx);
            store_row(&mut center.hy, row_base, da * hy - db * curl_hy);
            store_row(&mut center.hz, row_base, da * hz - db * curl_hz);
        }
    }
}

// =============================================================================
// E-FIELD UPDATE
// =============================================================================

/// Face-neighbor blocks whose boundary-facing H-field rows this block's
/// E-field update needs to read across the block seam. The Yee E-update
/// uses *backward* differences of H, so only the `-X`/`-Y`/`-Z` neighbors
/// are ever needed.
pub struct EUpdateNeighbors<'a> {
    pub minus_x: &'a FieldBlock,
    pub minus_y: &'a FieldBlock,
    pub minus_z: &'a FieldBlock,
}

/// Advances `center`'s Ex/Ey/Ez one full timestep, in place, using the
/// current H field of `center` and its `-X`/`-Y`/`-Z` neighbors:
///
/// ```text
/// Ex = Ca * Ex + Cb * (dHz/dy - dHy/dz)
/// Ey = Ca * Ey + Cb * (dHx/dz - dHz/dx)
/// Ez = Ca * Ez + Cb * (dHy/dx - dHx/dy)
/// ```
pub fn update_e_field(
    center: &mut FieldBlock,
    nbrs: EUpdateNeighbors,
    coeffs: &[MaterialCoeffs; VOXELS_PER_BLOCK],
) {
    for lz in 0..BLOCK_DIM {
        for ly in 0..BLOCK_DIM {
            let row_base = FieldBlock::local_index(0, ly, lz);

            let hx = load_row(&center.hx, row_base);
            let hy = load_row(&center.hy, row_base);
            let hz = load_row(&center.hz, row_base);

            // ---- -Y row of Hz (dEx term) and -Y row of Hx (dEz term) ----
            let (hz_my, hx_my) = if ly > 0 {
                let b = FieldBlock::local_index(0, ly - 1, lz);
                (load_row(&center.hz, b), load_row(&center.hx, b))
            } else {
                let b = FieldBlock::local_index(0, BLOCK_DIM - 1, lz);
                (load_row(&nbrs.minus_y.hz, b), load_row(&nbrs.minus_y.hx, b))
            };

            // ---- -Z row of Hy (dEx term) and -Z row of Hx (dEy term) ----
            let (hy_mz, hx_mz) = if lz > 0 {
                let b = FieldBlock::local_index(0, ly, lz - 1);
                (load_row(&center.hy, b), load_row(&center.hx, b))
            } else {
                let b = FieldBlock::local_index(0, ly, BLOCK_DIM - 1);
                (load_row(&nbrs.minus_z.hy, b), load_row(&nbrs.minus_z.hx, b))
            };

            // ---- -X-shifted Hz (dEy term) and Hy (dEz term) -------------
            let hz_row: [f32; BLOCK_DIM] = center.hz[row_base..row_base + BLOCK_DIM]
                .try_into()
                .unwrap();
            let hy_row: [f32; BLOCK_DIM] = center.hy[row_base..row_base + BLOCK_DIM]
                .try_into()
                .unwrap();
            let minus_x_row_base = FieldBlock::local_index(BLOCK_DIM - 1, ly, lz);
            let hz_mx = shifted_row_minus(&hz_row, nbrs.minus_x.hz[minus_x_row_base]);
            let hy_mx = shifted_row_minus(&hy_row, nbrs.minus_x.hy[minus_x_row_base]);

            // ---- curls ---------------------------------------------------
            let curl_ex = (hz - hz_my) - (hy - hy_mz); // dHz/dy - dHy/dz
            let curl_ey = (hx - hx_mz) - (hz - hz_mx); // dHx/dz - dHz/dx
            let curl_ez = (hy - hy_mx) - (hx - hx_my); // dHy/dx - dHx/dy

            let ca = gather_row(coeffs, row_base, |c| c.ca.to_f32());
            let cb = gather_row(coeffs, row_base, |c| c.cb.to_f32());

            let ex = load_row(&center.ex, row_base);
            let ey = load_row(&center.ey, row_base);
            let ez = load_row(&center.ez, row_base);

            store_row(&mut center.ex, row_base, ca * ex + cb * curl_ex);
            store_row(&mut center.ey, row_base, ca * ey + cb * curl_ey);
            store_row(&mut center.ez, row_base, ca * ez + cb * curl_ez);
        }
    }
}

/// CPML-aware variant of [`update_e_field`], mirroring
/// [`update_h_field_pml`]: each raw H-derivative that needs
/// stretched-coordinate correction runs through [`pml_correct_x`] /
/// [`pml_correct_scalar`] using `aux`'s psi memory before entering the
/// curl.
#[allow(clippy::too_many_arguments)]
pub fn update_e_field_pml(
    center: &mut FieldBlock,
    nbrs: EUpdateNeighbors,
    coeffs: &[MaterialCoeffs; VOXELS_PER_BLOCK],
    pml_x: &[PmlCoeffs; BLOCK_DIM],
    pml_y: &[PmlCoeffs; BLOCK_DIM],
    pml_z: &[PmlCoeffs; BLOCK_DIM],
    aux: &mut PmlAux,
) {
    let (px_b, px_a, px_inv_kappa) = pml_x_vectors(pml_x);

    for lz in 0..BLOCK_DIM {
        let pz = pml_z[lz];
        for ly in 0..BLOCK_DIM {
            let py = pml_y[ly];
            let row_base = FieldBlock::local_index(0, ly, lz);

            let hx = load_row(&center.hx, row_base);
            let hy = load_row(&center.hy, row_base);
            let hz = load_row(&center.hz, row_base);

            let (hz_my, hx_my) = if ly > 0 {
                let b = FieldBlock::local_index(0, ly - 1, lz);
                (load_row(&center.hz, b), load_row(&center.hx, b))
            } else {
                let b = FieldBlock::local_index(0, BLOCK_DIM - 1, lz);
                (load_row(&nbrs.minus_y.hz, b), load_row(&nbrs.minus_y.hx, b))
            };

            let (hy_mz, hx_mz) = if lz > 0 {
                let b = FieldBlock::local_index(0, ly, lz - 1);
                (load_row(&center.hy, b), load_row(&center.hx, b))
            } else {
                let b = FieldBlock::local_index(0, ly, BLOCK_DIM - 1);
                (load_row(&nbrs.minus_z.hy, b), load_row(&nbrs.minus_z.hx, b))
            };

            let hz_row: [f32; BLOCK_DIM] = center.hz[row_base..row_base + BLOCK_DIM]
                .try_into()
                .unwrap();
            let hy_row: [f32; BLOCK_DIM] = center.hy[row_base..row_base + BLOCK_DIM]
                .try_into()
                .unwrap();
            let minus_x_row_base = FieldBlock::local_index(BLOCK_DIM - 1, ly, lz);
            let hz_mx = shifted_row_minus(&hz_row, nbrs.minus_x.hz[minus_x_row_base]);
            let hy_mx = shifted_row_minus(&hy_row, nbrs.minus_x.hy[minus_x_row_base]);

            // ---- raw derivatives, each CPML-corrected in place ----------
            let d_hz_dy = pml_correct_scalar(hz - hz_my, py.b, py.a, py.inv_kappa, psi_row_mut(&mut aux.psi_hz_dy, row_base));
            let d_hx_dy = pml_correct_scalar(hx - hx_my, py.b, py.a, py.inv_kappa, psi_row_mut(&mut aux.psi_hx_dy, row_base));
            let d_hy_dz = pml_correct_scalar(hy - hy_mz, pz.b, pz.a, pz.inv_kappa, psi_row_mut(&mut aux.psi_hy_dz, row_base));
            let d_hx_dz = pml_correct_scalar(hx - hx_mz, pz.b, pz.a, pz.inv_kappa, psi_row_mut(&mut aux.psi_hx_dz, row_base));
            let d_hz_dx = pml_correct_x(hz - hz_mx, px_b, px_a, px_inv_kappa, psi_row_mut(&mut aux.psi_hz_dx, row_base));
            let d_hy_dx = pml_correct_x(hy - hy_mx, px_b, px_a, px_inv_kappa, psi_row_mut(&mut aux.psi_hy_dx, row_base));

            // ---- curls ---------------------------------------------------
            let curl_ex = d_hz_dy - d_hy_dz;
            let curl_ey = d_hx_dz - d_hz_dx;
            let curl_ez = d_hy_dx - d_hx_dy;

            let ca = gather_row(coeffs, row_base, |c| c.ca.to_f32());
            let cb = gather_row(coeffs, row_base, |c| c.cb.to_f32());

            let ex = load_row(&center.ex, row_base);
            let ey = load_row(&center.ey, row_base);
            let ez = load_row(&center.ez, row_base);

            store_row(&mut center.ex, row_base, ca * ex + cb * curl_ex);
            store_row(&mut center.ey, row_base, ca * ey + cb * curl_ey);
            store_row(&mut center.ez, row_base, ca * ez + cb * curl_ez);
        }
    }
}
