//! Macro-scale execution engine: spatial domain decomposition across
//! rayon-scheduled worker threads, lock-free `crossbeam-channel` boundary
//! exchange between them, and an `io_uring`-backed, double-buffered Direct
//! I/O snapshot writer.
//!
//! ## Domain decomposition
//!
//! The grid is sliced along Z -- its major (outermost, slowest-varying)
//! index in [`FieldGrid`]'s block-major layout -- into contiguous slabs of
//! whole XY block-planes, one slab per rayon worker. Because each slab is a
//! disjoint, non-overlapping mutable sub-slice of the single backing
//! allocation, `rayon`'s work-stealing scheduler can hand them to threads
//! with no locking and no copying.
//!
//! ## Cross-slab boundaries
//!
//! A slab's own Yee stencil is self-sufficient in X and Y (each slab spans
//! the full X/Y extent of the grid), but at its Z boundary it needs one
//! plane of field data that belongs to the *neighboring* slab -- which a
//! different thread may be concurrently mutating. Reaching across into
//! another thread's slice would be a data race, so instead each slab
//! publishes its own boundary plane and receives its neighbor's over a pair
//! of bounded, lock-free `crossbeam_channel` ring channels, once per
//! timestep, per boundary. This is the same halo-exchange pattern used in
//! MPI-decomposed FDTD codes, adapted to threads.
//!
//! ## Out-of-core snapshot streaming
//!
//! Field snapshots are serialized into one of two page-aligned buffers and
//! handed to `io_uring` (via `rio`) as an `O_DIRECT` write. `rio::Completion`
//! is a borrow-checked future tied to the buffer and file it reads from, so
//! simply *not* waiting on it immediately -- and instead continuing the
//! timestep loop while writing the *next* snapshot into the other buffer --
//! is how the double buffering here achieves overlap without unsafe
//! trickery. The loop only ever blocks on a previous write when it cycles
//! back around to a buffer that's still in flight.

use crate::fdtd::{self, EUpdateNeighbors, HUpdateNeighbors};
use crate::layout::{
    CoeffGrid, FieldBlock, FieldGrid, MaterialCoeffs, PmlAux, PmlAuxGrid, PmlContext,
    VOXELS_PER_BLOCK,
};
use crate::probe::Probe;
use crate::source::Source;
use crossbeam_channel::bounded;
use rayon::prelude::*;
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::fs::OpenOptions;
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::time::Instant;

/// A single all-zero block used as the boundary condition one voxel outside
/// the true edges of the domain (X/Y always, and the Z edges of the first
/// and last slab).
///
/// This is a perfect-electric-conductor (zero-field) backing termination.
/// With PML disabled (`PmlConfig::thickness == 0`), it's the *only*
/// boundary condition, and it reflects outgoing energy. With PML enabled
/// (`src/layout.rs`'s `PmlContext`/`PmlAuxGrid`), the outermost
/// `thickness`-deep shell of blocks applies a graded absorbing correction
/// before a wave ever reaches this backing plane, so by the time a wave
/// *does* reach it, it has already been attenuated to whatever residual
/// the target reflection coefficient allows -- the same "PEC-backed PML"
/// setup used in most production FDTD codes.
static OUTER_ZERO_BLOCK: FieldBlock = FieldBlock::ZERO;

// =============================================================================
// PER-SLAB YEE UPDATE
// =============================================================================

/// Advances the H field for every block in one slab.
///
/// `slab` is a contiguous, disjoint sub-slice of the global block array
/// covering full XY planes for local Z range `0..bz_n`; `coeffs` is the
/// matching slice of precomputed per-voxel coefficients; `plus_z_halo` is
/// the bottom E-plane of the *next* slab up (received over a
/// `crossbeam_channel`), used as the `+Z` neighbor for this slab's own top
/// plane.
///
/// `pml` is `None` when the PML is disabled entirely; otherwise it carries
/// the three per-axis coefficient profiles, and `pml_aux_slab` is the
/// matching slice of `PmlAuxGrid` blocks (`Some` only inside the PML
/// shell). `bz_offset` is this slab's starting block index along Z in the
/// *global* grid, needed to look up the right window of the Z profile.
#[allow(clippy::too_many_arguments)]
fn update_slab_h(
    slab: &mut [FieldBlock],
    coeffs: &[[MaterialCoeffs; VOXELS_PER_BLOCK]],
    pml_aux_slab: &mut [Option<Box<PmlAux>>],
    pml: Option<&PmlContext>,
    bx_n: usize,
    by_n: usize,
    bz_n: usize,
    bz_offset: usize,
    plus_z_halo: &[FieldBlock],
) {
    let plane = bx_n * by_n;
    let base: *mut FieldBlock = slab.as_mut_ptr();

    for bz in 0..bz_n {
        for by in 0..by_n {
            for bx in 0..bx_n {
                let idx = (bz * by_n + by) * bx_n + bx;

                // SAFETY: `idx`, and each neighbor index used below
                // (`idx + 1`, `idx + bx_n`, `idx + plane`) when its guard
                // condition holds, are all distinct, in-bounds indices into
                // `slab` (`bx_n * by_n * bz_n` elements total, matching
                // `slab.len()`) -- distinct because `bx`, `by`, `bz` range
                // over disjoint coordinates and the neighbor offsets never
                // wrap back onto `idx` itself. `update_h_field`/
                // `update_h_field_pml` only ever write `center`'s
                // `hx`/`hy`/`hz` arrays and only ever read `ex`/`ey`/`ez` off
                // of `center` and its neighbors, so the exclusive borrow of
                // `center` and the shared borrows of its neighbors never
                // alias the same memory, even though the borrow checker
                // cannot see that fact through the raw pointer arithmetic.
                // Each iteration's borrows are also scoped to that iteration
                // alone (dropped before the next `base.add(..)`
                // dereference), so there is no overlap across iterations
                // either.
                let center: &mut FieldBlock = unsafe { &mut *base.add(idx) };

                let plus_x: &FieldBlock = if bx + 1 < bx_n {
                    unsafe { &*base.add(idx + 1) }
                } else {
                    &OUTER_ZERO_BLOCK
                };
                let plus_y: &FieldBlock = if by + 1 < by_n {
                    unsafe { &*base.add(idx + bx_n) }
                } else {
                    &OUTER_ZERO_BLOCK
                };
                let plus_z: &FieldBlock = if bz + 1 < bz_n {
                    unsafe { &*base.add(idx + plane) }
                } else {
                    &plus_z_halo[by * bx_n + bx]
                };

                let nbrs = HUpdateNeighbors {
                    plus_x,
                    plus_y,
                    plus_z,
                };

                match (pml, pml_aux_slab[idx].as_deref_mut()) {
                    (Some(pml), Some(aux)) => {
                        let px = PmlContext::axis_window(&pml.profile_x, bx);
                        let py = PmlContext::axis_window(&pml.profile_y, by);
                        let pz = PmlContext::axis_window(&pml.profile_z, bz_offset + bz);
                        fdtd::update_h_field_pml(center, nbrs, &coeffs[idx], &px, &py, &pz, aux);
                    }
                    _ => {
                        fdtd::update_h_field(center, nbrs, &coeffs[idx]);
                    }
                }
            }
        }
    }
}

/// Advances the E field for every block in one slab. Mirror image of
/// [`update_slab_h`]: reads `-X`/`-Y`/`-Z` neighbors, with `minus_z_halo`
/// the top H-plane of the *previous* slab down. See [`update_slab_h`] for
/// the `pml`/`pml_aux_slab`/`bz_offset` parameters.
#[allow(clippy::too_many_arguments)]
fn update_slab_e(
    slab: &mut [FieldBlock],
    coeffs: &[[MaterialCoeffs; VOXELS_PER_BLOCK]],
    pml_aux_slab: &mut [Option<Box<PmlAux>>],
    pml: Option<&PmlContext>,
    bx_n: usize,
    by_n: usize,
    bz_n: usize,
    bz_offset: usize,
    minus_z_halo: &[FieldBlock],
) {
    let plane = bx_n * by_n;
    let base: *mut FieldBlock = slab.as_mut_ptr();

    for bz in 0..bz_n {
        for by in 0..by_n {
            for bx in 0..bx_n {
                let idx = (bz * by_n + by) * bx_n + bx;

                // SAFETY: identical reasoning to `update_slab_h` -- distinct,
                // in-bounds indices, and `update_e_field`/`update_e_field_pml`
                // write only `center`'s `ex`/`ey`/`ez` while reading only
                // `hx`/`hy`/`hz` off of `center` and its neighbors, so the
                // exclusive and shared borrows taken here never alias the
                // same field arrays.
                let center: &mut FieldBlock = unsafe { &mut *base.add(idx) };

                let minus_x: &FieldBlock = if bx > 0 {
                    unsafe { &*base.add(idx - 1) }
                } else {
                    &OUTER_ZERO_BLOCK
                };
                let minus_y: &FieldBlock = if by > 0 {
                    unsafe { &*base.add(idx - bx_n) }
                } else {
                    &OUTER_ZERO_BLOCK
                };
                let minus_z: &FieldBlock = if bz > 0 {
                    unsafe { &*base.add(idx - plane) }
                } else {
                    &minus_z_halo[by * bx_n + bx]
                };

                let nbrs = EUpdateNeighbors {
                    minus_x,
                    minus_y,
                    minus_z,
                };

                match (pml, pml_aux_slab[idx].as_deref_mut()) {
                    (Some(pml), Some(aux)) => {
                        let px = PmlContext::axis_window(&pml.profile_x, bx);
                        let py = PmlContext::axis_window(&pml.profile_y, by);
                        let pz = PmlContext::axis_window(&pml.profile_z, bz_offset + bz);
                        fdtd::update_e_field_pml(center, nbrs, &coeffs[idx], &px, &py, &pz, aux);
                    }
                    _ => {
                        fdtd::update_e_field(center, nbrs, &coeffs[idx]);
                    }
                }
            }
        }
    }
}

// =============================================================================
// PAGE-ALIGNED BUFFERS FOR O_DIRECT
// =============================================================================

/// Linux's `O_DIRECT` requires the userspace buffer address, length, and
/// file offset to all be aligned to the underlying block device's logical
/// block size. 4096 bytes covers every NVMe device in practice (logical
/// block sizes are 512 or 4096; 4096 is a multiple of both).
const DIRECT_IO_ALIGN: usize = 4096;

/// A heap buffer whose address and length are both guaranteed multiples of
/// [`DIRECT_IO_ALIGN`], suitable as the source buffer for an `O_DIRECT`
/// write submitted through `rio`.
///
/// Allocated once per snapshot-writer buffer slot at setup and reused for
/// the lifetime of the run -- never reallocated inside the timestep loop.
struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

// SAFETY: `AlignedBuffer` is a unique owner of a plain heap allocation with
// no thread-affinity (no TLS, no non-Send interior state); moving it to
// another thread and dropping it there is sound.
unsafe impl Send for AlignedBuffer {}

impl AlignedBuffer {
    fn new(min_len: usize) -> Self {
        let len = min_len.div_ceil(DIRECT_IO_ALIGN) * DIRECT_IO_ALIGN;
        let layout = Layout::from_size_align(len, DIRECT_IO_ALIGN)
            .expect("len is a positive multiple of a nonzero power-of-two alignment");
        // SAFETY: `layout` has non-zero size (`len >= DIRECT_IO_ALIGN > 0`).
        // `alloc` returns either a valid, non-null pointer to `len` freshly
        // allocated bytes aligned to `DIRECT_IO_ALIGN`, or null on failure,
        // which is checked immediately below via `NonNull::new` before the
        // pointer is used for anything.
        let raw = unsafe { alloc(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));
        Self { ptr, len, layout }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` was allocated with exactly `layout.size() == len`
        // bytes in `new`, and `&mut self` guarantees this is the only live
        // borrow of that memory, so a mutable slice of exactly `len` bytes
        // starting at `ptr` is valid for the lifetime of this borrow.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl AsRef<[u8]> for AlignedBuffer {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: same allocation invariant as `as_mut_slice`, but shared;
        // `rio::write_at` only ever reads through this borrow.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated by `alloc` with exactly
        // `self.layout` in `AlignedBuffer::new`, is uniquely owned by
        // `self`, and has not been freed before now (this is the one and
        // only `Drop` for this allocation) -- deallocating it here with the
        // same layout is the exact inverse of the allocation.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// A raw, unsafely-constructed view of one [`AlignedBuffer`]'s bytes,
/// captured once at setup so it can be handed to `rio::write_at` without
/// tying the resulting `rio::Completion`'s lifetime to a live borrow of the
/// owning `[AlignedBuffer; 2]` array.
///
/// Without this indirection, storing a `Completion<'a, usize>` borrowed
/// directly from `buffers[slot]` inside a `pending` array that outlives the
/// current loop iteration would force the borrow checker to treat the
/// *whole* `buffers` array as continuously borrowed for as long as any
/// snapshot write might still be in flight -- permanently blocking the
/// `&mut buffers[slot]` access `run` needs to serialize the *next* snapshot
/// into that same slot once its previous write has completed. `RawIoVec`
/// carries no lifetime of its own, so it sidesteps that unification
/// entirely; the safety obligation it would otherwise offload to the borrow
/// checker is instead upheld manually by `run`'s wait-before-reuse protocol.
#[derive(Clone, Copy)]
struct RawIoVec {
    ptr: *const u8,
    len: usize,
}

impl RawIoVec {
    /// # Safety
    /// The caller must ensure `buf`'s allocation is not moved, deallocated,
    /// or mutably accessed for as long as this `RawIoVec` -- or any
    /// `rio::Completion` built from a reference to it -- may still be
    /// reachable. `run` upholds this by only ever taking `&mut` access to a
    /// buffer slot immediately after `.wait()`-ing every `Completion` for
    /// that same slot to completion.
    unsafe fn from_buffer(buf: &AlignedBuffer) -> Self {
        let bytes = buf.as_ref();
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    /// Splits `buf`'s bytes into consecutive [`RawIoVec`] sub-views of at
    /// most `max_chunk_bytes` each (the last one may be shorter) -- see
    /// [`MAX_DIRECT_IO_WRITE_BYTES`] on why a snapshot write must never be
    /// submitted to `io_uring` as one single-shot request above a certain
    /// size. Every resulting view's length stays a multiple of
    /// `DIRECT_IO_ALIGN`, since both `buf`'s total length (via
    /// `AlignedBuffer::new`'s rounding) and `max_chunk_bytes` are.
    ///
    /// # Safety
    /// Identical obligation to [`RawIoVec::from_buffer`], for the lifetime of
    /// every element of the returned `Vec`.
    unsafe fn chunks_of(buf: &AlignedBuffer, max_chunk_bytes: usize) -> Vec<RawIoVec> {
        let whole = Self::from_buffer(buf);
        let mut out = Vec::with_capacity(whole.len.div_ceil(max_chunk_bytes).max(1));
        let mut start = 0;
        while start < whole.len {
            let len = max_chunk_bytes.min(whole.len - start);
            // `whole.ptr..whole.ptr + whole.len` is the full valid range
            // established by `from_buffer` above (same call, same caller
            // obligation), and `start + len <= whole.len` by construction
            // (`len` is clamped to the remaining distance to `whole.len`),
            // so `whole.ptr.add(start)` stays in-bounds (or
            // one-past-the-end only when `len == 0`, which the loop
            // condition `start < whole.len` never allows here). No separate
            // `unsafe {}` needed -- this whole function is already `unsafe`.
            out.push(RawIoVec {
                ptr: whole.ptr.add(start),
                len,
            });
            start += len;
        }
        out
    }
}

impl AsRef<[u8]> for RawIoVec {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: per `RawIoVec::from_buffer`'s invariant, the allocation
        // this points to is guaranteed live and not concurrently mutated for
        // as long as this `RawIoVec` is reachable, which includes the
        // duration of this borrow.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// `O_DIRECT`'s numeric value on the Linux x86_64/aarch64 syscall ABI
/// (`0o40000`). A handful of exotic architectures (alpha, sparc, mips,
/// parisc) define it differently; this crate targets x86_64/aarch64 NVMe
/// workstations, per the AVX2 build requirement in `Cargo.toml`. Not sourced
/// from the `libc` crate to avoid an otherwise-unneeded dependency for one
/// constant.
const O_DIRECT: i32 = 0o40000;

/// The largest number of bytes a single Linux `write()`/`pwrite()`-family
/// syscall will actually transfer, no matter how large a buffer is handed to
/// it: the kernel silently caps (does not error on) any single request at
/// `MAX_RW_COUNT` (`0x7ffff000` -- `INT_MAX` rounded down to a page
/// boundary; see `fs/read_write.c`), and `io_uring`'s `IORING_OP_WRITEV`
/// resolves through the same VFS write path, so it is bound by the same
/// cap. A snapshot whose serialized size exceeds this -- which happens at
/// domain sizes well within what this project otherwise claims to support
/// (any domain past roughly 450 voxels per axis) -- would silently lose its
/// tail past this many bytes on a single-shot `write_at` call, corrupting
/// `wave_trajectory.bin` with no I/O error raised anywhere: `rio`'s own
/// `write_at` docs warn about exactly this ("Be sure to check the returned
/// `io_uring_cqe`'s `res` field to see if a short write happened"), and nothing
/// in this file used to. `run` now splits every snapshot into
/// `MAX_DIRECT_IO_WRITE_BYTES`-sized chunks (see
/// `RawIoVec::chunks_of`) and checks each chunk's completion against the
/// byte count it actually requested. Picked well under the true kernel cap,
/// and (like the cap itself) already a whole `DIRECT_IO_ALIGN` multiple.
const MAX_DIRECT_IO_WRITE_BYTES: usize = 1 << 30; // 1 GiB

// =============================================================================
// ENGINE CONFIG & ORCHESTRATION
// =============================================================================

pub struct EngineConfig {
    pub num_steps: usize,
    pub snapshot_every: usize,
    /// Timestep, in seconds -- needed here (not just at setup) so source
    /// waveforms can be evaluated at the correct simulation time each step.
    pub dt: f32,
    pub output_path: PathBuf,
}

/// Serializes every field component of every block in `grid`, in block-major
/// order, into `out` -- a flat `Ex,Ey,Ez,Hx,Hy,Hz` byte stream per block.
/// `out` must be at least `grid.blocks().len() * size_of::<FieldBlock>()`
/// bytes (the aligned snapshot buffers built in [`run`] are sized to
/// guarantee this, rounded up to the `O_DIRECT` alignment).
fn serialize_snapshot(grid: &FieldGrid, out: &mut [u8]) {
    let block_bytes = std::mem::size_of::<FieldBlock>();
    for (block, chunk) in grid.blocks().iter().zip(out.chunks_mut(block_bytes)) {
        let src: &[u8] = unsafe {
            // SAFETY: `FieldBlock` is a `#[repr(align(64))]` struct made
            // entirely of `[f32; VOXELS_PER_BLOCK]` arrays -- a plain-old-data
            // layout with no padding-sensitive niches, no pointers, and no
            // `Drop` impl -- so reinterpreting it as its constituent bytes
            // for exactly `size_of::<FieldBlock>()` bytes is well-defined.
            // The resulting slice's lifetime is tied to `block`'s borrow, so
            // it cannot outlive the data it points to.
            std::slice::from_raw_parts(
                (block as *const FieldBlock).cast::<u8>(),
                block_bytes,
            )
        };
        chunk[..src.len()].copy_from_slice(src);
    }
}

/// Runs the full explicit timestep loop: alternating H/E Yee updates across
/// rayon-scheduled Z-slabs with crossbeam-channel halo exchange at slab
/// boundaries, periodically streaming a field snapshot out via `rio`
/// double-buffered `O_DIRECT` writes.
///
/// `pml` is `None` to disable the absorbing boundary entirely (falling back
/// to the zero-field `OUTER_ZERO_BLOCK` termination at every domain face);
/// otherwise `pml_aux` must have been built by the same
/// `PmlContext::build` call (so its shell of allocated blocks matches the
/// profiles' graded region).
pub fn run(
    field_grid: &mut FieldGrid,
    coeff_grid: &CoeffGrid,
    pml: Option<&PmlContext>,
    pml_aux: &mut PmlAuxGrid,
    sources: &[Source],
    probes: &mut [Probe],
    config: &EngineConfig,
) -> io::Result<()> {
    let dims = field_grid.dims();
    let (bx_n, by_n, bz_n_total) = dims.block_dims();
    let plane = bx_n * by_n;

    let num_threads = rayon::current_num_threads().max(1);
    let rows_per_slab = bz_n_total.div_ceil(num_threads).max(1);
    let blocks_per_slab = rows_per_slab * plane;
    let num_slabs = bz_n_total.div_ceil(rows_per_slab);

    // One pair of bounded(1) rendezvous channels per internal boundary --
    // lock-free ring channels connecting exactly two fixed thread domains,
    // alive for the whole run. `e_*` carries E-plane data flowing downward
    // (slab i+1 -> slab i, consumed before the H update); `h_*` carries
    // H-plane data flowing upward (slab i -> slab i+1, consumed before the
    // E update).
    let mut e_tx = Vec::with_capacity(num_slabs.saturating_sub(1));
    let mut e_rx = Vec::with_capacity(num_slabs.saturating_sub(1));
    let mut h_tx = Vec::with_capacity(num_slabs.saturating_sub(1));
    let mut h_rx = Vec::with_capacity(num_slabs.saturating_sub(1));
    for _ in 0..num_slabs.saturating_sub(1) {
        let (tx, rx) = bounded::<Box<[FieldBlock]>>(1);
        e_tx.push(tx);
        e_rx.push(rx);
        let (tx, rx) = bounded::<Box<[FieldBlock]>>(1);
        h_tx.push(tx);
        h_rx.push(rx);
    }

    // ---- io_uring double-buffered O_DIRECT snapshot writer ----------------
    //
    // `ring`, `out_file`, and `buffers` all live for the rest of this
    // function; `pending[slot]` borrows whichever of them backed that
    // slot's last write. Because none of the three are ever moved or
    // exclusively re-borrowed while a `Completion` referencing them is
    // alive, the borrow checker accepts this without any `unsafe` -- this
    // is exactly the pattern `rio`'s own API is designed around (see
    // `rio::Completion`'s docs on tying a write's lifetime to its buffer).
    let ring = rio::new()?;
    let out_file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(O_DIRECT)
        .open(&config.output_path)?;
    let snapshot_bytes = std::mem::size_of_val(field_grid.blocks());
    let mut buffers = [
        AlignedBuffer::new(snapshot_bytes),
        AlignedBuffer::new(snapshot_bytes),
    ];
    // SAFETY: `buffers` lives for the remainder of `run` and its elements
    // are never relocated (arrays don't move their elements out from under
    // a live reference), so each captured pointer/length stays valid for as
    // long as `run` runs. The loop below upholds `RawIoVec::from_buffer`'s
    // "wait before mutate" obligation for the pointed-to memory.
    //
    // Each buffer is split into `MAX_DIRECT_IO_WRITE_BYTES`-sized chunks
    // (see that constant's docs) rather than handed to `write_at` as one
    // single-shot request -- a single request above the kernel's per-syscall
    // cap would silently write only its first `MAX_RW_COUNT` bytes with no
    // error, corrupting every snapshot past that size.
    let chunk_iovs: [Vec<RawIoVec>; 2] = unsafe {
        [
            RawIoVec::chunks_of(&buffers[0], MAX_DIRECT_IO_WRITE_BYTES),
            RawIoVec::chunks_of(&buffers[1], MAX_DIRECT_IO_WRITE_BYTES),
        ]
    };
    let mut pending: [Vec<rio::Completion<'_, usize>>; 2] = [Vec::new(), Vec::new()];
    let mut active_buffer = 0usize;
    let mut write_offset: u64 = 0;

    /// Waits on every completion for one buffer slot's in-flight writes,
    /// verifying each chunk actually transferred every byte it was asked
    /// to -- a short write here means silent, undetected snapshot
    /// corruption (see `MAX_DIRECT_IO_WRITE_BYTES`), so this turns that
    /// into a loud `io::Error` instead.
    fn drain_pending<'a>(
        pending: &mut Vec<rio::Completion<'a, usize>>,
        chunks: &[RawIoVec],
    ) -> io::Result<()> {
        for (completion, chunk) in pending.drain(..).zip(chunks) {
            let written = completion.wait()?;
            if written != chunk.len {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    format!(
                        "short O_DIRECT write: wrote {written} of {} requested bytes -- \
                         wave_trajectory.bin would be corrupted",
                        chunk.len
                    ),
                ));
            }
        }
        Ok(())
    }

    let started = Instant::now();

    for step in 0..config.num_steps {
        let all_blocks = field_grid.blocks_mut();
        let all_coeffs = coeff_grid.blocks();
        let all_pml_aux = pml_aux.blocks_mut();

        // ---- H update phase, fanned out across slabs by rayon ----------
        all_blocks
            .par_chunks_mut(blocks_per_slab)
            .zip(all_coeffs.par_chunks(blocks_per_slab))
            .zip(all_pml_aux.par_chunks_mut(blocks_per_slab))
            .enumerate()
            .for_each(|(i, ((slab, slab_coeffs), slab_pml_aux))| {
                let bz_n = slab.len() / plane;
                let bz_offset = i * rows_per_slab;

                let plus_z_halo: Box<[FieldBlock]> = if i + 1 < num_slabs {
                    e_rx[i].recv().expect("adjacent slab's channel half was dropped")
                } else {
                    vec![FieldBlock::ZERO; plane].into_boxed_slice()
                };

                update_slab_h(
                    slab,
                    slab_coeffs,
                    slab_pml_aux,
                    pml,
                    bx_n,
                    by_n,
                    bz_n,
                    bz_offset,
                    &plus_z_halo,
                );

                if i > 0 {
                    let bottom_plane: Box<[FieldBlock]> = slab[..plane].to_vec().into_boxed_slice();
                    e_tx[i - 1]
                        .send(bottom_plane)
                        .expect("adjacent slab's channel half was dropped");
                }
            });

        // ---- E update phase, fanned out across slabs by rayon ----------
        all_blocks
            .par_chunks_mut(blocks_per_slab)
            .zip(all_coeffs.par_chunks(blocks_per_slab))
            .zip(all_pml_aux.par_chunks_mut(blocks_per_slab))
            .enumerate()
            .for_each(|(i, ((slab, slab_coeffs), slab_pml_aux))| {
                let bz_n = slab.len() / plane;
                let bz_offset = i * rows_per_slab;

                let minus_z_halo: Box<[FieldBlock]> = if i > 0 {
                    h_rx[i - 1].recv().expect("adjacent slab's channel half was dropped")
                } else {
                    vec![FieldBlock::ZERO; plane].into_boxed_slice()
                };

                update_slab_e(
                    slab,
                    slab_coeffs,
                    slab_pml_aux,
                    pml,
                    bx_n,
                    by_n,
                    bz_n,
                    bz_offset,
                    &minus_z_halo,
                );

                if i + 1 < num_slabs {
                    let top_plane: Box<[FieldBlock]> =
                        slab[slab.len() - plane..].to_vec().into_boxed_slice();
                    h_tx[i]
                        .send(top_plane)
                        .expect("adjacent slab's channel half was dropped");
                }
            });

        // ---- source injection --------------------------------------------
        //
        // Soft sources are re-applied every timestep, after the E-field
        // update completes -- each one only ever touches the single voxel
        // it lives at, so this runs single-threaded with no meaningful
        // cost regardless of grid size. `t` is the simulation time the E
        // field now represents, one full timestep past its initial state.
        let t = (step as f32 + 1.0) * config.dt;
        for source in sources {
            source.inject(field_grid, t);
        }

        // ---- probe accumulation -------------------------------------------
        //
        // Same cost profile as source injection above: each probe only ever
        // reads the single voxel it lives at, so this stays cheap regardless
        // of grid size. Reads the same post-E-update field state the sources
        // just finished writing into, at the same `t`.
        for probe in probes.iter_mut() {
            probe.accumulate(field_grid, t);
        }

        if step % config.snapshot_every == 0 {
            // Reclaim this buffer slot: wait only if the writes that
            // previously used it haven't finished yet. With two buffers,
            // this is the ONLY point the timestep loop can stall on
            // storage, and only because we've cycled back to a slot still
            // in flight -- rare if the NVMe write completes faster than
            // `snapshot_every` timesteps take to compute.
            drain_pending(&mut pending[active_buffer], &chunk_iovs[active_buffer])?;

            serialize_snapshot(field_grid, buffers[active_buffer].as_mut_slice());
            let mut chunk_offset = write_offset;
            for chunk in &chunk_iovs[active_buffer] {
                pending[active_buffer].push(ring.write_at(&out_file, chunk, chunk_offset));
                chunk_offset += chunk.len as u64;
            }

            write_offset += snapshot_bytes as u64;
            active_buffer ^= 1; // flip to the alternate pre-allocated page
        }
    }

    // Drain any writes still in flight before returning.
    drain_pending(&mut pending[0], &chunk_iovs[0])?;
    drain_pending(&mut pending[1], &chunk_iovs[1])?;

    let elapsed = started.elapsed();
    eprintln!(
        "wavefront: {} steps over {} slab(s) in {:.3}s ({:.1} steps/s)",
        config.num_steps,
        num_slabs,
        elapsed.as_secs_f64(),
        config.num_steps as f64 / elapsed.as_secs_f64().max(1e-9)
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{GridDims, MaterialGrid, MaterialTable, BLOCK_DIM};
    use crate::source::{FieldComponent, Source, Waveform};

    const SPEED_OF_LIGHT_M_PER_S: f32 = 299_792_458.0;

    /// Numerical correctness check: a point source in vacuum should radiate
    /// outward at (approximately) the speed of light. This exercises the
    /// solver directly -- `update_slab_h`/`update_slab_e` treating the whole
    /// grid as one slab, with PML disabled -- deliberately bypassing
    /// `run`'s rayon/crossbeam/rio machinery so the test has no filesystem
    /// or `O_DIRECT` dependency and can't be flaky in a CI sandbox that
    /// doesn't support Direct I/O.
    ///
    /// FDTD has numerical dispersion (grid-driven group velocity error), so
    /// this allows a generous tolerance -- it's meant to catch a badly wrong
    /// update equation (wrong sign, wrong constant, factor-of-two error),
    /// not to be a tight dispersion-accuracy benchmark.
    #[test]
    fn wave_propagates_at_approximately_speed_of_light() {
        let dims = GridDims::new(64, 64, 64);
        let dx = 1.0e-3_f32;
        let dt = 1.5e-12_f32; // Courant number ~0.45, comfortably stable.
        let table = MaterialTable::vacuum_filled(dt, dx);

        let path = std::env::temp_dir().join(format!(
            "wavefront_test_materials_{}_{:?}.grid",
            std::process::id(),
            std::thread::current().id()
        ));
        let material_grid = MaterialGrid::create(&path, dims).expect("create test material grid");
        let coeff_grid = CoeffGrid::build(&material_grid, &table);
        let _ = std::fs::remove_file(&path);

        let mut field_grid = FieldGrid::zeroed(dims);
        let (bx_n, by_n, bz_n) = dims.block_dims();
        let plane = bx_n * by_n;
        let zero_plane = vec![FieldBlock::ZERO; plane];
        let mut pml_aux = PmlAuxGrid::build(dims, 0);

        let center = (dims.nx / 2, dims.ny / 2, dims.nz / 2);
        let freq_hz = 3.0e10_f32;
        let source = Source {
            x: center.0,
            y: center.1,
            z: center.2,
            component: FieldComponent::Ez,
            amplitude: 1.0,
            waveform: Waveform::RickerWavelet {
                peak_freq_hz: freq_hz,
                t0: 1.0 / freq_hz,
            },
        };

        // Probe 20 voxels away along +X -- well before any wall reflection
        // could contaminate the reading (the domain edge is 32 voxels out).
        let probe_x = center.0 + 20;
        let (probe_bx, probe_by, probe_bz) = (probe_x / BLOCK_DIM, center.1 / BLOCK_DIM, center.2 / BLOCK_DIM);
        let probe_local =
            FieldBlock::local_index(probe_x % BLOCK_DIM, center.1 % BLOCK_DIM, center.2 % BLOCK_DIM);

        let expected_time_s = (20.0 * dx) / SPEED_OF_LIGHT_M_PER_S;
        let threshold = 5.0e-4;
        let max_steps = 200;
        let mut arrival_step: Option<usize> = None;

        for step in 0..max_steps {
            update_slab_h(
                field_grid.blocks_mut(),
                coeff_grid.blocks(),
                pml_aux.blocks_mut(),
                None,
                bx_n,
                by_n,
                bz_n,
                0,
                &zero_plane,
            );
            update_slab_e(
                field_grid.blocks_mut(),
                coeff_grid.blocks(),
                pml_aux.blocks_mut(),
                None,
                bx_n,
                by_n,
                bz_n,
                0,
                &zero_plane,
            );

            let t = (step as f32 + 1.0) * dt;
            source.inject(&mut field_grid, t);

            let probe_value = field_grid.block(probe_bx, probe_by, probe_bz).ez[probe_local];
            if arrival_step.is_none() && probe_value.abs() > threshold {
                arrival_step = Some(step);
                break;
            }
        }

        let arrival_step = arrival_step.expect("wave never reached the probe point within max_steps");
        let measured_time_s = (arrival_step as f32 + 1.0) * dt;

        let relative_error = (measured_time_s - expected_time_s).abs() / expected_time_s;
        assert!(
            relative_error < 0.20,
            "measured arrival time {measured_time_s:e}s vs expected {expected_time_s:e}s \
             (speed of light over 20 voxels) -- relative error {relative_error:.3} exceeds 20%"
        );
    }

    /// Regression test for a real bug found running a large-scale (512^3
    /// voxel) out-of-core validation: a snapshot whose serialized size
    /// exceeded Linux's per-syscall `write()` cap (`MAX_RW_COUNT`,
    /// `0x7ffff000` bytes) was silently truncated by the kernel on a
    /// single-shot `write_at` -- corrupting `wave_trajectory.bin` with no
    /// error raised anywhere, since the code never checked the returned
    /// byte count against what it requested. `RawIoVec::chunks_of` is the
    /// fix: split every write into sub-`MAX_RW_COUNT` chunks. This test
    /// exercises its pure chunk-boundary math directly (small buffer, small
    /// chunk size) rather than an actual multi-gigabyte `O_DIRECT` write --
    /// consistent with this suite's existing avoidance of Direct-I/O-
    /// dependent tests, which can be flaky in a CI sandbox that doesn't
    /// support it (see `wave_propagates_at_approximately_speed_of_light`
    /// above).
    #[test]
    fn raw_io_vec_chunks_of_covers_the_whole_buffer_with_no_gaps_or_overlap() {
        // `AlignedBuffer::new` always rounds up to a `DIRECT_IO_ALIGN`
        // (4096-byte) multiple, so a buffer "logically" 10000 bytes long is
        // actually 12288 bytes (3 alignment units) -- exercise a chunk size
        // that doesn't evenly divide that, to prove the last chunk is
        // correctly shorter rather than out-of-bounds or panicking.
        let mut buf = AlignedBuffer::new(10_000);
        assert_eq!(buf.len, 12_288, "sanity: AlignedBuffer rounds up to a 4096 multiple");

        // Fill with a distinct byte per position so any misplaced chunk
        // boundary would show up as wrong data, not just a wrong length.
        for (i, b) in buf.as_mut_slice().iter_mut().enumerate() {
            *b = (i % 251) as u8; // 251 is prime, avoids the 256-wraparound aliasing every byte
        }

        let chunk_size = 5_000usize; // deliberately not a multiple of 4096 or of buf.len
        // SAFETY: `buf` is a local that outlives every chunk produced below
        // and is never mutated for the duration of this test.
        let chunks = unsafe { RawIoVec::chunks_of(&buf, chunk_size) };

        assert_eq!(chunks.len(), 3, "12288 bytes in 5000-byte chunks should be [5000, 5000, 2288]");
        assert_eq!(chunks[0].len, 5_000);
        assert_eq!(chunks[1].len, 5_000);
        assert_eq!(chunks[2].len, 2_288);

        let total: usize = chunks.iter().map(|c| c.len).sum();
        assert_eq!(total, buf.len, "chunks must cover every byte of the buffer exactly once");

        let reassembled: Vec<u8> = chunks.iter().flat_map(|c| c.as_ref().to_vec()).collect();
        assert_eq!(
            reassembled,
            buf.as_ref().to_vec(),
            "concatenated chunks must reproduce the original buffer byte-for-byte, in order"
        );
    }
}
