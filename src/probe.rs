//! Frequency-domain response via a running discrete Fourier transform (DFT)
//! at fixed voxel locations.
//!
//! Everything else in this crate reports the field in the time domain: a raw
//! `Ex/Ey/Ez/Hx/Hy/Hz` snapshot every `snapshot_every` steps
//! (`src/engine.rs`). That's the right format for the snapshot stream, but
//! it's the wrong tool for asking "what's the steady-state response at this
//! frequency at this point" (resonance, transmission) -- you'd have to
//! stream the whole run to disk and post-process it.
//!
//! [`Probe`] instead accumulates, incrementally, one timestep at a time, the
//! exact same quadrature (in-phase/quadrature) demodulation the validation
//! studies use to recover phase and amplitude from a saved trace
//! (`examples/convergence_study.rs`, `examples/pml_reflection_study.rs`) --
//! except live, inside the timestep loop, at however many frequencies are of
//! interest, with no need to keep the full time-domain history around at
//! all. This is the standard FDTD "runtime Fourier transform" technique
//! (Taflove & Hagness, *Computational Electrodynamics*, ch. 5): for a
//! frequency `f`, running sums of `value(t) * cos(2*pi*f*t)` and
//! `value(t) * sin(2*pi*f*t)` converge to that frequency's steady-state
//! in-phase/quadrature components as the accumulation window grows, without
//! ever computing a full FFT over stored samples.

use crate::layout::{FieldBlock, FieldGrid, BLOCK_DIM};

/// Which of the six Yee field components a [`Probe`] samples. Unlike
/// `source::FieldComponent` (E-only, since only E is soft-sourced), a probe
/// can read any of the six -- resonance/transmission studies routinely care
/// about H as much as E.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldComponent {
    Ex,
    Ey,
    Ez,
    Hx,
    Hy,
    Hz,
}

/// One frequency's recovered steady-state response at a [`Probe`]'s
/// location: amplitude and phase of the component being tracked, extracted
/// via quadrature demodulation over the accumulation window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyResponse {
    pub freq_hz: f32,
    /// Peak amplitude of the sinusoidal steady-state response at this
    /// frequency (not RMS).
    pub amplitude: f64,
    /// Phase, in radians, of `amplitude * sin(2*pi*freq_hz*t + phase)`
    /// relative to a `t = 0` reference.
    pub phase_rad: f64,
}

/// A point probe that accumulates a running DFT of one field component, at
/// one or more frequencies, over the course of a simulation run.
///
/// Accumulation only starts once the simulation clock passes
/// `start_recording_at_s` -- skipping a startup transient this way keeps
/// early, non-steady-state field behavior (e.g. a source's initial turn-on)
/// from contaminating a frequency response that's only meaningful once the
/// field has settled, exactly as the validation studies' `skip_before_s`
/// does for their own quadrature extraction.
pub struct Probe {
    x: usize,
    y: usize,
    z: usize,
    component: FieldComponent,
    frequencies_hz: Vec<f32>,
    start_recording_at_s: f32,
    /// Per-frequency running accumulators, in f64 (this sums potentially
    /// millions of single-precision samples over a long run).
    sum_cos: Vec<f64>,
    sum_sin: Vec<f64>,
    samples_recorded: u64,
}

impl Probe {
    pub fn new(
        x: usize,
        y: usize,
        z: usize,
        component: FieldComponent,
        frequencies_hz: Vec<f32>,
        start_recording_at_s: f32,
    ) -> Self {
        let n = frequencies_hz.len();
        Self {
            x,
            y,
            z,
            component,
            frequencies_hz,
            start_recording_at_s,
            sum_cos: vec![0.0; n],
            sum_sin: vec![0.0; n],
            samples_recorded: 0,
        }
    }

    /// Reads this probe's tracked component at its voxel location.
    fn sample(&self, field_grid: &FieldGrid) -> f32 {
        let dims = field_grid.dims();
        debug_assert!(self.x < dims.nx && self.y < dims.ny && self.z < dims.nz);

        let (bx, by, bz) = (self.x / BLOCK_DIM, self.y / BLOCK_DIM, self.z / BLOCK_DIM);
        let (lx, ly, lz) = (self.x % BLOCK_DIM, self.y % BLOCK_DIM, self.z % BLOCK_DIM);
        let local = FieldBlock::local_index(lx, ly, lz);

        let block = field_grid.block(bx, by, bz);
        let component = match self.component {
            FieldComponent::Ex => &block.ex,
            FieldComponent::Ey => &block.ey,
            FieldComponent::Ez => &block.ez,
            FieldComponent::Hx => &block.hx,
            FieldComponent::Hy => &block.hy,
            FieldComponent::Hz => &block.hz,
        };
        component[local]
    }

    /// Folds this timestep's sample into every tracked frequency's running
    /// accumulator, if `t` is past `start_recording_at_s`. Called once per
    /// timestep by `src/engine.rs`, after the E-field update -- cheap
    /// regardless of grid size, like [`crate::source::Source::inject`], since
    /// it only ever touches the single voxel this probe lives at.
    pub fn accumulate(&mut self, field_grid: &FieldGrid, t: f32) {
        if t < self.start_recording_at_s {
            return;
        }
        let value = self.sample(field_grid) as f64;
        for (i, &freq_hz) in self.frequencies_hz.iter().enumerate() {
            let omega = 2.0 * std::f64::consts::PI * freq_hz as f64;
            let phase = omega * t as f64;
            self.sum_cos[i] += value * phase.cos();
            self.sum_sin[i] += value * phase.sin();
        }
        self.samples_recorded += 1;
    }

    /// Finalizes the running accumulators into one [`FrequencyResponse`] per
    /// tracked frequency. Can be called at any point (including mid-run);
    /// typically called once, after the timestep loop finishes.
    pub fn spectrum(&self) -> Vec<FrequencyResponse> {
        let n = self.samples_recorded.max(1) as f64; // avoid div-by-zero if never accumulated
        self.frequencies_hz
            .iter()
            .zip(&self.sum_cos)
            .zip(&self.sum_sin)
            .map(|((&freq_hz, &sum_cos), &sum_sin)| {
                let a = 2.0 * sum_cos / n;
                let b = 2.0 * sum_sin / n;
                // For v(t) = amplitude*sin(omega*t + phase), a = amplitude*sin(phase)
                // and b = amplitude*cos(phase) fall straight out of the
                // sin/cos product-to-sum identities averaged over a whole
                // number of periods, so phase = atan2(a, b) recovers it
                // directly (no sign flip -- that only applies to the
                // different delay-recovery convention the validation
                // examples use, where phase enters as `sin(omega*(t -
                // delay))` instead of an additive offset).
                FrequencyResponse {
                    freq_hz,
                    amplitude: (a * a + b * b).sqrt(),
                    phase_rad: a.atan2(b),
                }
            })
            .collect()
    }

    pub fn position(&self) -> (usize, usize, usize) {
        (self.x, self.y, self.z)
    }

    pub fn component(&self) -> FieldComponent {
        self.component
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::GridDims;

    /// Drives a probe's tracked voxel directly with a known sinusoid (no
    /// solver involved) and confirms the running DFT recovers the correct
    /// amplitude and phase at the driven frequency, and near-zero amplitude
    /// at a frequency the signal doesn't contain -- the two properties any
    /// DFT extraction must have to be useful.
    #[test]
    fn recovers_amplitude_and_phase_of_a_known_sinusoid() {
        let dims = GridDims::new(16, 16, 16);
        let mut grid = FieldGrid::zeroed(dims);

        let true_freq_hz = 1.0e9_f32;
        let true_amplitude = 3.5_f32;
        let true_phase_rad = 0.7_f64;
        let dt = 1.0e-12_f32;
        let steps = 4000;

        let mut probe = Probe::new(
            8, 8, 8,
            FieldComponent::Ez,
            vec![true_freq_hz, 3.0 * true_freq_hz], // a second, undriven frequency
            0.0,
        );

        let (bx, by, bz) = (1, 1, 1); // voxel (8,8,8) is block (1,1,1), local (0,0,0)
        let local = FieldBlock::local_index(0, 0, 0);
        let omega = 2.0 * std::f32::consts::PI * true_freq_hz;

        for step in 0..steps {
            let t = (step as f32 + 1.0) * dt;
            let value = true_amplitude * (omega * t + true_phase_rad as f32).sin();
            grid.block_mut(bx, by, bz).ez[local] = value;
            probe.accumulate(&grid, t);
        }

        let spectrum = probe.spectrum();
        assert_eq!(spectrum.len(), 2);

        let driven = spectrum[0];
        assert_eq!(driven.freq_hz, true_freq_hz);
        assert!(
            (driven.amplitude - true_amplitude as f64).abs() < 0.05 * true_amplitude as f64,
            "recovered amplitude {} vs true {}",
            driven.amplitude,
            true_amplitude
        );
        let phase_error = (driven.phase_rad - true_phase_rad + std::f64::consts::PI)
            .rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;
        assert!(
            phase_error.abs() < 0.05,
            "recovered phase {} vs true {}",
            driven.phase_rad,
            true_phase_rad
        );

        let undriven = spectrum[1];
        assert!(
            undriven.amplitude < 0.05 * true_amplitude as f64,
            "expected near-zero amplitude at an undriven frequency, got {}",
            undriven.amplitude
        );
    }

    #[test]
    fn samples_before_start_recording_at_are_ignored() {
        let dims = GridDims::new(16, 16, 16);
        let mut grid = FieldGrid::zeroed(dims);
        let mut probe = Probe::new(0, 0, 0, FieldComponent::Ex, vec![1.0e9], 5.0e-9);

        // A huge, wrong value before the recording window opens should be
        // completely excluded from the accumulators.
        grid.block_mut(0, 0, 0).ex[FieldBlock::local_index(0, 0, 0)] = 1.0e6;
        probe.accumulate(&grid, 1.0e-9);
        probe.accumulate(&grid, 4.9e-9);

        let spectrum = probe.spectrum();
        assert_eq!(spectrum[0].amplitude, 0.0, "no samples should have been recorded yet");
    }
}
