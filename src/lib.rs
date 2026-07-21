//! Wavefront: an asynchronous, out-of-core 3D FDTD electromagnetic
//! simulator, exposed as a library so the `wavefront` binary and auxiliary
//! tools (e.g. `wavefront-view`, in `src/bin/`) can share its core modules
//! without duplicating them.
//!
//! - `layout`: mmap'd material grid, cache-aligned AoSoA field grid, CPML
//!   coefficient/auxiliary storage.
//! - `fdtd`: the SIMD Yee-lattice curl update kernels.
//! - `engine`: spatial decomposition, rayon scheduling, crossbeam halo
//!   exchange, and the `io_uring` snapshot writer.
//! - `scene`: plain-text material structure description format.
//! - `source`: time-domain source excitation.

#![feature(portable_simd)]

pub mod engine;
pub mod fdtd;
pub mod layout;
pub mod scene;
pub mod source;
