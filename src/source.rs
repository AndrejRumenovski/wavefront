//! Time-domain source excitation.
//!
//! The original demo injected a single spatial Gaussian blob directly into
//! the field grid as an initial condition at t=0 -- fine for watching one
//! pulse spread and reflect, but useless for anything needing a sustained
//! or repeatable excitation (steady-state response, frequency-domain
//! extraction via a running DFT, S-parameter sweeps, etc.). This module
//! replaces that with the standard FDTD approach: one or more point *soft
//! sources*, each driven by a time-domain waveform and re-injected every
//! timestep by `src/engine.rs`.
//!
//! "Soft" means the source *adds* to whatever the Yee update already
//! computed at that voxel, rather than overwriting it -- so a wave
//! generated elsewhere in the domain can still pass through the source
//! location instead of being blocked by it.

use crate::layout::{FieldBlock, FieldGrid, BLOCK_DIM};

/// Which field component a [`Source`] drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldComponent {
    Ex,
    Ey,
    Ez,
}

/// A time-domain excitation waveform, evaluated at simulation time `t`
/// (seconds).
#[derive(Debug, Clone, Copy)]
pub enum Waveform {
    /// A single Gaussian pulse in time, centered at `t0` with standard
    /// deviation `spread` (both in seconds). Has nonzero DC content (its
    /// time integral isn't zero), which is fine for a one-shot transient
    /// but can leave a small residual field drift under long runs.
    GaussianPulse { t0: f32, spread: f32 },
    /// A continuous sinusoid at `freq_hz`, for steady-state / CW studies.
    Sinusoid { freq_hz: f32 },
    /// A Ricker wavelet (the negative normalized second derivative of a
    /// Gaussian) peaking at frequency `peak_freq_hz`, centered at `t0`
    /// seconds. DC-free, which makes it the standard default excitation
    /// for broadband FDTD transient sweeps.
    RickerWavelet { peak_freq_hz: f32, t0: f32 },
}

impl Waveform {
    pub fn evaluate(&self, t: f32) -> f32 {
        match *self {
            Waveform::GaussianPulse { t0, spread } => {
                let arg = (t - t0) / spread;
                (-0.5 * arg * arg).exp()
            }
            Waveform::Sinusoid { freq_hz } => (2.0 * std::f32::consts::PI * freq_hz * t).sin(),
            Waveform::RickerWavelet { peak_freq_hz, t0 } => {
                let arg = std::f32::consts::PI * peak_freq_hz * (t - t0);
                let arg_sq = arg * arg;
                (1.0 - 2.0 * arg_sq) * (-arg_sq).exp()
            }
        }
    }
}

/// A single point soft source: a voxel location, the component it drives,
/// a peak amplitude, and a waveform.
#[derive(Debug, Clone, Copy)]
pub struct Source {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub component: FieldComponent,
    pub amplitude: f32,
    pub waveform: Waveform,
}

impl Source {
    /// Additively injects this source's value at time `t` (seconds) into
    /// `field_grid`. Called once per timestep by `src/engine.rs`, after the
    /// E-field update -- cheap regardless of grid size, since it only ever
    /// touches the single voxel this source lives at.
    pub fn inject(&self, field_grid: &mut FieldGrid, t: f32) {
        let dims = field_grid.dims();
        debug_assert!(self.x < dims.nx && self.y < dims.ny && self.z < dims.nz);

        let value = self.amplitude * self.waveform.evaluate(t);
        let (bx, by, bz) = (self.x / BLOCK_DIM, self.y / BLOCK_DIM, self.z / BLOCK_DIM);
        let (lx, ly, lz) = (self.x % BLOCK_DIM, self.y % BLOCK_DIM, self.z % BLOCK_DIM);
        let local = FieldBlock::local_index(lx, ly, lz);

        let block = field_grid.block_mut(bx, by, bz);
        let component = match self.component {
            FieldComponent::Ex => &mut block.ex,
            FieldComponent::Ey => &mut block.ey,
            FieldComponent::Ez => &mut block.ez,
        };
        component[local] += value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::GridDims;

    #[test]
    fn ricker_wavelet_is_dc_free_over_a_full_period() {
        // Sampling a Ricker wavelet densely around its center should
        // integrate to approximately zero -- its defining property, and
        // the reason it's preferred over a plain Gaussian pulse for long
        // driven runs.
        let w = Waveform::RickerWavelet {
            peak_freq_hz: 1.0,
            t0: 2.0,
        };
        let n = 4000;
        let span = 4.0;
        let dt = span / n as f32;
        let sum: f32 = (0..n).map(|i| w.evaluate(i as f32 * dt)).sum();
        assert!(
            (sum * dt).abs() < 0.05,
            "Ricker wavelet integral should be ~0, got {}",
            sum * dt
        );
    }

    #[test]
    fn sinusoid_matches_a_sine_wave() {
        let w = Waveform::Sinusoid { freq_hz: 1.0 };
        assert!((w.evaluate(0.0) - 0.0).abs() < 1e-6);
        assert!((w.evaluate(0.25) - 1.0).abs() < 1e-5); // quarter period -> peak
    }

    #[test]
    fn inject_adds_to_existing_field_value() {
        let dims = GridDims::new(16, 16, 16);
        let mut grid = FieldGrid::zeroed(dims);
        let source = Source {
            x: 8,
            y: 8,
            z: 8,
            component: FieldComponent::Ez,
            amplitude: 2.0,
            waveform: Waveform::Sinusoid { freq_hz: 1.0 },
        };

        // At t = 0.25s (quarter period), sin() peaks at 1.0, so the
        // injected value should be amplitude * 1.0 = 2.0.
        source.inject(&mut grid, 0.25);
        let local = FieldBlock::local_index(0, 0, 0);
        let value = grid.block(1, 1, 1).ez[local];
        assert!((value - 2.0).abs() < 1e-4);

        // A second injection should *add*, not overwrite -- confirming this
        // is a soft source.
        source.inject(&mut grid, 0.25);
        let value = grid.block(1, 1, 1).ez[local];
        assert!((value - 4.0).abs() < 1e-4);
    }
}
