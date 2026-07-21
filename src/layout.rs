//! Zero-copy, cache-aligned memory layout primitives for the Wavefront FDTD engine.
//!
//! This module owns two structurally different memory regions:
//!
//!   1. The **material grid** -- a static, disk-resident, memory-mapped byte
//!      array of per-voxel material IDs ([`MaterialGrid`]). It never changes
//!      during a run, so `memmap2::MmapMut` gives us effectively free,
//!      zero-copy random access to a structure that can legitimately be
//!      hundreds of gigabytes -- far larger than physical RAM -- without a
//!      single explicit `read()`/`write()` syscall; the kernel's page cache
//!      does the paging for us.
//!
//!   2. The **field grid** -- the live, mutable Ex/Ey/Ez/Hx/Hy/Hz state that
//!      the solver advances every timestep ([`FieldGrid`]). It is tiled into
//!      [`FieldBlock`] cells: fixed-size, `#[repr(align(64))]`,
//!      Array-of-Structures-of-Arrays (AoSoA) units sized so that a single
//!      contiguous row of any field component is exactly one AVX2 `f32x8`
//!      vector register load (see `src/fdtd.rs`).
//!
//! Both grids are allocated exactly once, at setup. Nothing in `src/fdtd.rs`
//! or `src/engine.rs`'s per-timestep hot path allocates on the heap.

use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::io;
use std::path::Path;

/// Vacuum permittivity, F/m.
const EPS0: f32 = 8.854_187_8e-12;
/// Vacuum permeability, H/m.
const MU0: f32 = 1.256_637_1e-6;

// =============================================================================
// GRID DIMENSIONS
// =============================================================================

/// Edge length, in voxels, of one cubic [`FieldBlock`] tile along every axis.
///
/// This is deliberately equal to the AVX2 `f32x8` lane count: a contiguous
/// row of `BLOCK_DIM` voxels along X loads into a single vector register in
/// one instruction, with no partial-lane masking, in `src/fdtd.rs`.
pub const BLOCK_DIM: usize = 8;

/// Number of voxels held by a single [`FieldBlock`] (`BLOCK_DIM^3`).
pub const VOXELS_PER_BLOCK: usize = BLOCK_DIM * BLOCK_DIM * BLOCK_DIM;

/// Global grid dimensions, expressed in voxels.
///
/// Each axis must be a multiple of [`BLOCK_DIM`] so the field grid tiles it
/// exactly, with no partial edge blocks and therefore no bounds-checked slow
/// path inside the hot loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridDims {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl GridDims {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        assert!(
            nx % BLOCK_DIM == 0 && ny % BLOCK_DIM == 0 && nz % BLOCK_DIM == 0,
            "grid dimensions must be multiples of BLOCK_DIM ({BLOCK_DIM}) so the field grid \
             tiles exactly with no partial edge blocks"
        );
        Self { nx, ny, nz }
    }

    #[inline(always)]
    pub fn voxel_count(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Grid dimensions expressed in whole [`FieldBlock`] tiles.
    #[inline(always)]
    pub fn block_dims(&self) -> (usize, usize, usize) {
        (
            self.nx / BLOCK_DIM,
            self.ny / BLOCK_DIM,
            self.nz / BLOCK_DIM,
        )
    }

    #[inline(always)]
    pub fn block_count(&self) -> usize {
        let (bx, by, bz) = self.block_dims();
        bx * by * bz
    }
}

// =============================================================================
// FIXED-POINT COEFFICIENT QUANTIZATION
// =============================================================================

/// Q16.16 fixed-point number used for precomputed Yee update coefficients.
///
/// FDTD update coefficients (`Ca`/`Cb`/`Da`/`Db`, see [`MaterialCoeffs`])
/// depend only on a voxel's material and the fixed simulation timestep --
/// they are identical on every iteration. Wavefront computes them once, at
/// setup, from floating point material constants, and freezes them as
/// quantized fixed-point integers here. The timestep loop itself therefore
/// never touches a material-property division, `sqrt`, or any other slow
/// floating-point-unit boundary; it only ever multiplies-and-adds already
/// quantized coefficients that were prepared ahead of time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct FixedQ16_16(pub i32);

const FIXED_FRAC_BITS: u32 = 16;
const FIXED_ONE: i32 = 1 << FIXED_FRAC_BITS;

impl FixedQ16_16 {
    #[inline(always)]
    pub fn from_f32(v: f32) -> Self {
        Self((v * FIXED_ONE as f32).round() as i32)
    }

    /// Converts back to `f32` for use as a SIMD multiplier. This compiles to
    /// a single multiply by a compile-time-constant reciprocal -- not a
    /// division -- so it stays on the fast path even though the coefficient
    /// itself started life as a quantized integer.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 * (1.0 / FIXED_ONE as f32)
    }
}

// =============================================================================
// MATERIALS
// =============================================================================

/// A single byte identifying the material occupying one voxel.
///
/// The material grid is bit-packed to one byte per voxel -- not because a
/// byte holds the physical constants directly, but because it is an index
/// into a small, in-memory [`MaterialTable`] of at most 256 distinct
/// materials, which is more than sufficient for structural/optical voxel
/// models and keeps the on-disk grid at exactly `nx * ny * nz` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct MaterialId(pub u8);

impl MaterialId {
    pub const VACUUM: MaterialId = MaterialId(0);
}

/// Precomputed, quantized Yee update coefficients for one material.
///
/// For the electric field update: `E_new = Ca * E_old + Cb * curl(H)`.
/// For the magnetic field update: `H_new = Da * H_old - Db * curl(E)`.
///
/// See Taflove & Hagness, *Computational Electrodynamics*, ch. 3, for the
/// standard lossy-medium derivation of `Ca/Cb/Da/Db` from relative
/// permittivity, relative permeability, electric conductivity, timestep,
/// and cell size.
#[derive(Debug, Clone, Copy)]
pub struct MaterialCoeffs {
    pub ca: FixedQ16_16,
    pub cb: FixedQ16_16,
    pub da: FixedQ16_16,
    pub db: FixedQ16_16,
}

/// Lookup table mapping the 256 possible [`MaterialId`] values to their
/// precomputed update coefficients. Indexed directly by `MaterialId.0`, so a
/// lookup is a single array access with no branching.
pub struct MaterialTable {
    coeffs: [MaterialCoeffs; 256],
}

impl MaterialTable {
    /// Builds a table where every material slot is vacuum, for the given
    /// timestep `dt` and uniform cell size `d`. Callers then override
    /// individual slots with [`MaterialTable::set_material`].
    pub fn vacuum_filled(dt: f32, d: f32) -> Self {
        let vacuum = Self::coeffs_from_physical(1.0, 1.0, 0.0, dt, d);
        Self {
            coeffs: [vacuum; 256],
        }
    }

    pub fn set_material(
        &mut self,
        id: MaterialId,
        eps_r: f32,
        mu_r: f32,
        sigma: f32,
        dt: f32,
        d: f32,
    ) {
        self.coeffs[id.0 as usize] = Self::coeffs_from_physical(eps_r, mu_r, sigma, dt, d);
    }

    fn coeffs_from_physical(eps_r: f32, mu_r: f32, sigma: f32, dt: f32, d: f32) -> MaterialCoeffs {
        let eps = eps_r * EPS0;
        let mu = mu_r * MU0;
        let loss = sigma * dt / (2.0 * eps);
        let ca = (1.0 - loss) / (1.0 + loss);
        let cb = (dt / (eps * d)) / (1.0 + loss);
        let da = 1.0;
        let db = dt / (mu * d);
        MaterialCoeffs {
            ca: FixedQ16_16::from_f32(ca),
            cb: FixedQ16_16::from_f32(cb),
            da: FixedQ16_16::from_f32(da),
            db: FixedQ16_16::from_f32(db),
        }
    }

    #[inline(always)]
    pub fn get(&self, id: MaterialId) -> MaterialCoeffs {
        // SAFETY: `MaterialId` wraps a `u8`, so `id.0 as usize` is always in
        // `0..256`, which is exactly `self.coeffs.len()`.
        unsafe { *self.coeffs.get_unchecked(id.0 as usize) }
    }
}

/// The [`MaterialCoeffs`] for every voxel in the domain, precomputed once at
/// setup and tiled identically to the [`FieldGrid`] it accompanies.
///
/// Freezing per-voxel coefficients here (rather than re-deriving them from
/// [`MaterialGrid`] + [`MaterialTable`] on every timestep) keeps the
/// per-step solver loop to pure multiply-adds over already-resolved values.
pub struct CoeffGrid {
    blocks: Box<[[MaterialCoeffs; VOXELS_PER_BLOCK]]>,
    dims: GridDims,
}

impl CoeffGrid {
    pub fn build(material_grid: &MaterialGrid, table: &MaterialTable) -> Self {
        let dims = material_grid.dims();
        let (bx_n, by_n, bz_n) = dims.block_dims();
        let mut blocks = vec![
            [table.get(MaterialId::VACUUM); VOXELS_PER_BLOCK];
            bx_n * by_n * bz_n
        ]
        .into_boxed_slice();

        for bz in 0..bz_n {
            for by in 0..by_n {
                for bx in 0..bx_n {
                    let block_idx = (bz * by_n + by) * bx_n + bx;
                    let block_coeffs = &mut blocks[block_idx];
                    for lz in 0..BLOCK_DIM {
                        for ly in 0..BLOCK_DIM {
                            for lx in 0..BLOCK_DIM {
                                let x = bx * BLOCK_DIM + lx;
                                let y = by * BLOCK_DIM + ly;
                                let z = bz * BLOCK_DIM + lz;
                                let id = material_grid.material_at(x, y, z);
                                block_coeffs[FieldBlock::local_index(lx, ly, lz)] = table.get(id);
                            }
                        }
                    }
                }
            }
        }

        Self { blocks, dims }
    }

    pub fn dims(&self) -> GridDims {
        self.dims
    }

    /// The full, flat, block-major coefficient array -- sliced by
    /// `src/engine.rs` into the same contiguous per-slab chunks it uses for
    /// [`FieldGrid::blocks_mut`], so every slab's `bz`-range lines up
    /// between the two grids.
    pub fn blocks(&self) -> &[[MaterialCoeffs; VOXELS_PER_BLOCK]] {
        &self.blocks
    }
}

// =============================================================================
// MEMORY-MAPPED MATERIAL GRID
// =============================================================================

/// A flat, disk-resident, memory-mapped material grid: one byte per voxel,
/// indexing into a [`MaterialTable`]. Backed by `memmap2::MmapMut` so the OS
/// pages it in and out on demand -- this is how Wavefront supports
/// structural models far larger than physical RAM.
pub struct MaterialGrid {
    mmap: MmapMut,
    dims: GridDims,
}

impl MaterialGrid {
    /// Creates (or truncates) a backing file of exactly `dims.voxel_count()`
    /// bytes and memory-maps it read/write.
    pub fn create(path: impl AsRef<Path>, dims: GridDims) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(dims.voxel_count() as u64)?;
        // SAFETY: `file` is a regular on-disk file that we just created and
        // sized to exactly `dims.voxel_count()` bytes, and no other handle
        // to it is handed out before this call returns. `MmapMut::map_mut`'s
        // only real hazard -- the backing file being truncated or resized by
        // another process while mapped, which can turn access into a SIGBUS
        // -- cannot happen here because `self` is the sole owner of the path
        // for the lifetime of the mapping.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { mmap, dims })
    }

    /// Opens a previously-created material grid file, mapping it read/write.
    pub fn open_existing(path: impl AsRef<Path>, dims: GridDims) -> io::Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let len = file.metadata()?.len();
        assert_eq!(
            len,
            dims.voxel_count() as u64,
            "material grid file size does not match the requested dimensions"
        );
        // SAFETY: identical invariant to `create` -- the file is verified
        // above to be exactly `dims.voxel_count()` bytes, and `self` is the
        // sole owner of this mapping for its lifetime.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self { mmap, dims })
    }

    #[inline(always)]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dims.ny + y) * self.dims.nx + x
    }

    /// Reads the material at voxel `(x, y, z)`.
    ///
    /// Bounds are checked with `debug_assert!` only. Release builds trust
    /// the caller -- always a spatial-decomposition loop already bounded by
    /// `self.dims` -- and use an unchecked read to stay on the branch-free
    /// fast path.
    #[inline(always)]
    pub fn material_at(&self, x: usize, y: usize, z: usize) -> MaterialId {
        debug_assert!(x < self.dims.nx && y < self.dims.ny && z < self.dims.nz);
        let idx = self.index(x, y, z);
        // SAFETY: `idx = (z * ny + y) * nx + x` with `x < nx`, `y < ny`,
        // `z < nz` (checked above in debug builds; guaranteed by the
        // caller's decomposition-loop bounds in release builds) is strictly
        // less than `nx * ny * nz`, which is exactly `self.mmap.len()`.
        MaterialId(unsafe { *self.mmap.get_unchecked(idx) })
    }

    #[inline(always)]
    pub fn set_material_at(&mut self, x: usize, y: usize, z: usize, id: MaterialId) {
        debug_assert!(x < self.dims.nx && y < self.dims.ny && z < self.dims.nz);
        let idx = self.index(x, y, z);
        // SAFETY: see `material_at` -- identical index-bound invariant.
        unsafe { *self.mmap.get_unchecked_mut(idx) = id.0 };
    }

    pub fn dims(&self) -> GridDims {
        self.dims
    }

    /// Flushes dirty pages back to disk (e.g. after voxelizing a structure).
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

// =============================================================================
// CPML ABSORBING BOUNDARY
// =============================================================================
//
// A domain edge that just terminates the grid (zero field just outside it,
// as `src/engine.rs`'s `OUTER_ZERO_BLOCK` does) reflects nearly all outgoing
// energy straight back into the simulation, which is wrong for any open-
// space problem. Wavefront implements a Convolutional PML (CPML, Roden &
// Gedney 2000): a graded, lossy region near each domain edge that absorbs
// outgoing waves with a controllable, very small reflection coefficient,
// without splitting the field components (unlike the original Berenger
// PML).
//
// CPML works by replacing the ordinary spatial derivative in the curl
// update with a "stretched-coordinate" version:
//
//   d/du  ->  (1/kappa_u) * d/du + psi_u
//
// where `psi_u` is an auxiliary memory variable updated by a one-tap
// recursive convolution each timestep:
//
//   psi_u_new = b_u * psi_u_old + a_u * (raw derivative)
//
// `kappa_u`, `b_u`, and `a_u` are derived once, at setup, from a graded
// conductivity profile `sigma_u(rho)` (`rho` = normalized depth into the
// layer) via the standard formulas in Taflove & Hagness ch. 7. Everywhere
// outside the PML layer, `kappa=1, b=1, a=0`, which makes the stretched
// derivative and the psi update both exact no-ops -- so a PML-aware kernel
// call over a fully-interior block is mathematically identical to the
// plain kernel, just with a bit of wasted arithmetic. `src/engine.rs`
// exploits this by only ever paying for PML-aware dispatch on the thin
// shell of boundary blocks, keeping the O(surface-area) `PmlAux` memory
// cost far below the O(volume) cost of the field grid itself.

/// Tunable parameters for the CPML layer. See Taflove & Hagness,
/// *Computational Electrodynamics*, ch. 7, for the standard ranges these
/// come from.
#[derive(Debug, Clone, Copy)]
pub struct PmlConfig {
    /// Thickness of the absorbing layer, in voxels, at each of the 6 domain
    /// faces. Rounded up to a whole number of [`BLOCK_DIM`]-sized block
    /// layers (PML bookkeeping is allocated per-block, not per-voxel).
    /// `0` disables the PML entirely, reverting to a zero-field boundary.
    pub thickness: usize,
    /// Polynomial grading order `m` for `sigma` and `kappa` (typical 3-4).
    pub grading_order: f32,
    /// Maximum coordinate-stretching factor `kappa_max` (typical 5-15;
    /// `1.0` disables stretching and leaves only the conductive loss).
    pub kappa_max: f32,
    /// Maximum CFS-PML relaxation parameter `alpha_max`, which improves
    /// absorption of near-grazing and low-frequency waves (typical
    /// 0.05-0.24).
    pub alpha_max: f32,
    /// Target normal-incidence reflection coefficient `R0` used to derive
    /// `sigma_max` (typical 1e-6 to 1e-8; smaller is a "better" but not
    /// necessarily more stable PML).
    pub target_reflection: f32,
}

impl Default for PmlConfig {
    fn default() -> Self {
        Self {
            thickness: BLOCK_DIM,
            grading_order: 3.0,
            kappa_max: 7.0,
            alpha_max: 0.05,
            target_reflection: 1.0e-6,
        }
    }
}

/// The precomputed CPML recursive-convolution coefficients for one voxel
/// position along one axis.
#[derive(Debug, Clone, Copy)]
pub struct PmlCoeffs {
    pub b: f32,
    pub a: f32,
    pub inv_kappa: f32,
}

impl PmlCoeffs {
    /// The no-op coefficient set: `1/kappa = 1`, `b = 1`, `a = 0`. Applying
    /// a PML correction with these coefficients leaves the derivative and
    /// the psi memory unchanged -- this is what every non-PML voxel uses.
    pub const IDENTITY: PmlCoeffs = PmlCoeffs {
        b: 1.0,
        a: 0.0,
        inv_kappa: 1.0,
    };
}

/// A full-axis-length array of [`PmlCoeffs`], graded near both ends of the
/// axis and [`PmlCoeffs::IDENTITY`] everywhere in the interior.
///
/// This is cheap regardless of grid size (a handful of floats per voxel
/// *position*, not per voxel volume) and is built once at setup.
pub struct PmlProfile1D {
    coeffs: Box<[PmlCoeffs]>,
}

impl PmlProfile1D {
    /// Builds the graded profile for an axis of length `n`, with a
    /// `thickness_voxels`-deep absorbing layer at each end (already
    /// clamped/rounded by the caller -- see
    /// [`effective_pml_thickness_blocks`]).
    pub fn build(n: usize, config: &PmlConfig, dt: f32, dx: f32, thickness_voxels: usize) -> Self {
        let mut coeffs = vec![PmlCoeffs::IDENTITY; n].into_boxed_slice();
        if thickness_voxels == 0 {
            return Self { coeffs };
        }

        let thickness = thickness_voxels;
        let pml_depth_m = thickness as f32 * dx;
        let eta0 = (MU0 / EPS0).sqrt();
        let sigma_max = -(config.grading_order + 1.0) * config.target_reflection.ln()
            / (2.0 * eta0 * pml_depth_m);

        for g in 0..thickness {
            // rho = 1 at the true domain edge (g == 0), grading down to
            // ~1/thickness at the innermost PML cell, adjacent to the
            // ordinary interior.
            let rho = (thickness - g) as f32 / thickness as f32;
            let sigma = sigma_max * rho.powf(config.grading_order);
            let kappa = 1.0 + (config.kappa_max - 1.0) * rho.powf(config.grading_order);
            let alpha = config.alpha_max * (1.0 - rho);
            let c = Self::coeffs_from_physical(sigma, kappa, alpha, dt);

            coeffs[g] = c; // low-side face, g == 0 is the true edge
            coeffs[n - 1 - g] = c; // high-side face, mirrored
        }

        Self { coeffs }
    }

    fn coeffs_from_physical(sigma: f32, kappa: f32, alpha: f32, dt: f32) -> PmlCoeffs {
        let b = (-(sigma / kappa + alpha) * dt / EPS0).exp();
        let denom = kappa * (sigma + kappa * alpha);
        let a = if denom.abs() > 1.0e-30 {
            sigma * (b - 1.0) / denom
        } else {
            0.0
        };
        PmlCoeffs {
            b,
            a,
            inv_kappa: 1.0 / kappa,
        }
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> PmlCoeffs {
        self.coeffs[i]
    }
}

/// Clamps a requested PML thickness (in voxels) to a whole number of
/// [`BLOCK_DIM`]-sized block layers that fits within half of the shortest
/// axis (so the low- and high-side layers of any axis can never overlap).
/// `0` in, `0` out -- PML stays disabled if the caller asked for that.
pub fn effective_pml_thickness_blocks(dims: GridDims, requested_thickness_voxels: usize) -> usize {
    if requested_thickness_voxels == 0 {
        return 0;
    }
    let (bx_n, by_n, bz_n) = dims.block_dims();
    let requested_blocks = requested_thickness_voxels.div_ceil(BLOCK_DIM).max(1);
    let max_blocks = (bx_n.min(by_n).min(bz_n) / 2).max(1);
    requested_blocks.min(max_blocks)
}

/// Per-block CPML auxiliary ("psi") convolution memory: one persistent
/// value per voxel, per raw derivative term that needs stretched-coordinate
/// correction. Only ever allocated for blocks inside the PML shell (see
/// [`PmlAuxGrid`]) -- everywhere else, the correction is a no-op and no
/// memory is spent on it.
///
/// Field naming is `psi_<source>_d<axis>`: e.g. `psi_ez_dx` is the psi
/// memory for the `dEz/dx` term (which feeds the Hy update).
#[repr(align(64))]
#[derive(Clone)]
pub struct PmlAux {
    pub psi_ez_dx: [f32; VOXELS_PER_BLOCK],
    pub psi_ey_dx: [f32; VOXELS_PER_BLOCK],
    pub psi_ez_dy: [f32; VOXELS_PER_BLOCK],
    pub psi_ex_dy: [f32; VOXELS_PER_BLOCK],
    pub psi_ey_dz: [f32; VOXELS_PER_BLOCK],
    pub psi_ex_dz: [f32; VOXELS_PER_BLOCK],
    pub psi_hz_dx: [f32; VOXELS_PER_BLOCK],
    pub psi_hy_dx: [f32; VOXELS_PER_BLOCK],
    pub psi_hx_dy: [f32; VOXELS_PER_BLOCK],
    pub psi_hz_dy: [f32; VOXELS_PER_BLOCK],
    pub psi_hy_dz: [f32; VOXELS_PER_BLOCK],
    pub psi_hx_dz: [f32; VOXELS_PER_BLOCK],
}

impl PmlAux {
    pub const ZERO: PmlAux = PmlAux {
        psi_ez_dx: [0.0; VOXELS_PER_BLOCK],
        psi_ey_dx: [0.0; VOXELS_PER_BLOCK],
        psi_ez_dy: [0.0; VOXELS_PER_BLOCK],
        psi_ex_dy: [0.0; VOXELS_PER_BLOCK],
        psi_ey_dz: [0.0; VOXELS_PER_BLOCK],
        psi_ex_dz: [0.0; VOXELS_PER_BLOCK],
        psi_hz_dx: [0.0; VOXELS_PER_BLOCK],
        psi_hy_dx: [0.0; VOXELS_PER_BLOCK],
        psi_hx_dy: [0.0; VOXELS_PER_BLOCK],
        psi_hz_dy: [0.0; VOXELS_PER_BLOCK],
        psi_hy_dz: [0.0; VOXELS_PER_BLOCK],
        psi_hx_dz: [0.0; VOXELS_PER_BLOCK],
    };
}

/// The sparse, block-major companion to [`FieldGrid`] holding [`PmlAux`]
/// memory only for blocks inside the PML shell (`Some`); every interior
/// block is `None` and costs only one pointer-sized slot.
///
/// Because it mirrors `FieldGrid`'s block-major layout exactly, it can be
/// sliced into the same per-thread Z-slabs `src/engine.rs` already uses for
/// `FieldGrid` and `CoeffGrid`, with no extra decomposition logic.
pub struct PmlAuxGrid {
    blocks: Box<[Option<Box<PmlAux>>]>,
    dims: GridDims,
}

impl PmlAuxGrid {
    /// `thickness_voxels` should be the same (already-clamped) value passed
    /// to each axis's [`PmlProfile1D::build`], so the shell of allocated
    /// blocks lines up exactly with the region where the profiles carry
    /// non-identity coefficients.
    pub fn build(dims: GridDims, thickness_voxels: usize) -> Self {
        let (bx_n, by_n, bz_n) = dims.block_dims();
        let thickness_blocks = thickness_voxels / BLOCK_DIM;

        let mut blocks = Vec::with_capacity(bx_n * by_n * bz_n);
        for bz in 0..bz_n {
            for by in 0..by_n {
                for bx in 0..bx_n {
                    let in_shell = thickness_blocks > 0
                        && (bx < thickness_blocks
                            || bx >= bx_n - thickness_blocks
                            || by < thickness_blocks
                            || by >= by_n - thickness_blocks
                            || bz < thickness_blocks
                            || bz >= bz_n - thickness_blocks);
                    blocks.push(in_shell.then(|| Box::new(PmlAux::ZERO)));
                }
            }
        }

        Self {
            blocks: blocks.into_boxed_slice(),
            dims,
        }
    }

    pub fn dims(&self) -> GridDims {
        self.dims
    }

    pub fn blocks_mut(&mut self) -> &mut [Option<Box<PmlAux>>] {
        &mut self.blocks
    }
}

/// The three per-axis graded profiles that drive the CPML correction,
/// bundled together with the constructor that builds them (and the
/// matching [`PmlAuxGrid`]) consistently from one [`PmlConfig`].
pub struct PmlContext {
    pub profile_x: PmlProfile1D,
    pub profile_y: PmlProfile1D,
    pub profile_z: PmlProfile1D,
}

impl PmlContext {
    pub fn build(dims: GridDims, config: &PmlConfig, dt: f32, dx: f32) -> (Self, PmlAuxGrid) {
        let thickness_blocks = effective_pml_thickness_blocks(dims, config.thickness);
        let thickness_voxels = thickness_blocks * BLOCK_DIM;

        let profile_x = PmlProfile1D::build(dims.nx, config, dt, dx, thickness_voxels);
        let profile_y = PmlProfile1D::build(dims.ny, config, dt, dx, thickness_voxels);
        let profile_z = PmlProfile1D::build(dims.nz, config, dt, dx, thickness_voxels);
        let aux_grid = PmlAuxGrid::build(dims, thickness_voxels);

        (
            Self {
                profile_x,
                profile_y,
                profile_z,
            },
            aux_grid,
        )
    }

    /// Gathers one block's worth (`BLOCK_DIM` consecutive positions,
    /// starting at `block_index * BLOCK_DIM`) of coefficients out of a 1D
    /// profile -- cheap, and only ever called once per PML-shell block per
    /// update phase (not per voxel).
    #[inline]
    pub fn axis_window(profile: &PmlProfile1D, block_index: usize) -> [PmlCoeffs; BLOCK_DIM] {
        let base = block_index * BLOCK_DIM;
        let mut out = [PmlCoeffs::IDENTITY; BLOCK_DIM];
        for (l, o) in out.iter_mut().enumerate() {
            *o = profile.get(base + l);
        }
        out
    }
}

// =============================================================================
// CACHE-ALIGNED FIELD GRID (AoSoA)
// =============================================================================

/// One cache-line-aligned AoSoA tile of the live field grid.
///
/// All six Yee field components for a `BLOCK_DIM^3` voxel neighborhood live
/// together in one allocation, aligned to the 64-byte cache line boundary
/// via `#[repr(align(64))]`. That guarantees a worker thread touching this
/// block starts each fetch on a cache line boundary rather than straddling
/// one -- and, because blocks are the unit of ownership handed to rayon
/// worker threads in `src/engine.rs`, guarantees no cache line is ever
/// shared between two blocks owned by different threads (no false
/// sharing).
#[repr(align(64))]
#[derive(Clone)]
pub struct FieldBlock {
    pub ex: [f32; VOXELS_PER_BLOCK],
    pub ey: [f32; VOXELS_PER_BLOCK],
    pub ez: [f32; VOXELS_PER_BLOCK],
    pub hx: [f32; VOXELS_PER_BLOCK],
    pub hy: [f32; VOXELS_PER_BLOCK],
    pub hz: [f32; VOXELS_PER_BLOCK],
}

impl FieldBlock {
    pub const ZERO: FieldBlock = FieldBlock {
        ex: [0.0; VOXELS_PER_BLOCK],
        ey: [0.0; VOXELS_PER_BLOCK],
        ez: [0.0; VOXELS_PER_BLOCK],
        hx: [0.0; VOXELS_PER_BLOCK],
        hy: [0.0; VOXELS_PER_BLOCK],
        hz: [0.0; VOXELS_PER_BLOCK],
    };

    /// Row-major local index of voxel `(lx, ly, lz)` within one block, with
    /// `lx` the fastest-varying axis -- so that `[base .. base + BLOCK_DIM]`
    /// for `base = local_index(0, ly, lz)` is exactly one contiguous SIMD
    /// row along X.
    #[inline(always)]
    pub fn local_index(lx: usize, ly: usize, lz: usize) -> usize {
        (lz * BLOCK_DIM + ly) * BLOCK_DIM + lx
    }
}

/// The live, mutable field state for the whole simulation domain: a flat,
/// row-major array of [`FieldBlock`] tiles, allocated exactly once.
///
/// Wavefront never grows, shrinks, or reallocates this buffer after setup --
/// every timestep only mutates values already in place, satisfying the
/// zero-heap-allocation mandate for the hot loop.
pub struct FieldGrid {
    blocks: Box<[FieldBlock]>,
    dims: GridDims,
}

impl FieldGrid {
    pub fn zeroed(dims: GridDims) -> Self {
        let (bx, by, bz) = dims.block_dims();
        let blocks = vec![FieldBlock::ZERO; bx * by * bz].into_boxed_slice();
        Self { blocks, dims }
    }

    pub fn dims(&self) -> GridDims {
        self.dims
    }

    #[inline(always)]
    fn block_index(&self, bx: usize, by: usize, bz: usize) -> usize {
        let (nbx, nby, _) = self.dims.block_dims();
        debug_assert!(bx < nbx && by < nby);
        (bz * nby + by) * nbx + bx
    }

    #[inline(always)]
    pub fn block(&self, bx: usize, by: usize, bz: usize) -> &FieldBlock {
        let idx = self.block_index(bx, by, bz);
        // SAFETY: `block_index` combines `bx < nbx`, `by < nby` (asserted
        // above in debug builds) with a `bz` bounded by the caller's
        // domain-decomposition slab range, which `src/engine.rs` always
        // keeps inside `0..nbz`; the resulting index is therefore always
        // `< self.blocks.len()`.
        unsafe { self.blocks.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn block_mut(&mut self, bx: usize, by: usize, bz: usize) -> &mut FieldBlock {
        let idx = self.block_index(bx, by, bz);
        // SAFETY: identical index-bound invariant to `block`.
        unsafe { self.blocks.get_unchecked_mut(idx) }
    }

    /// Exposes the whole block array as a flat mutable slice so that
    /// `src/engine.rs` can partition it into disjoint, per-thread slabs
    /// (contiguous ranges of whole Z-planes of blocks) with
    /// `rayon::slice::ParallelSliceMut::par_chunks_mut` for the
    /// work-stealing scheduler.
    pub fn blocks_mut(&mut self) -> &mut [FieldBlock] {
        &mut self.blocks
    }

    pub fn blocks(&self) -> &[FieldBlock] {
        &self.blocks
    }
}
