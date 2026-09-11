use crate::BLOCK;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;

const FFT: usize = BLOCK * 2;

/// Uniform partitioned overlap-add: every FFT is 512 points regardless of
/// room filter length. All spectra and FFT scratch are allocated at startup.
pub struct Convolver {
    filters: Vec<Vec<Complex32>>,
    history: Vec<Vec<Complex32>>,
    head: usize,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    work: Vec<Complex32>,
    scratch: Vec<Complex32>,
    overlap: [f32; BLOCK],
}

impl Convolver {
    pub fn new(impulse: &[f32]) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT);
        let ifft = planner.plan_fft_inverse(FFT);
        let count = impulse.len().max(1).div_ceil(BLOCK);
        let mut filters = vec![vec![Complex32::default(); FFT]; count];
        let mut scratch = vec![
            Complex32::default();
            fft.get_inplace_scratch_len()
                .max(ifft.get_inplace_scratch_len())
        ];
        for (p, spectrum) in filters.iter_mut().enumerate() {
            for (i, sample) in impulse.iter().skip(p * BLOCK).take(BLOCK).enumerate() {
                spectrum[i].re = *sample;
            }
            fft.process_with_scratch(spectrum, &mut scratch);
        }
        Self {
            filters,
            history: vec![vec![Complex32::default(); FFT]; count],
            head: 0,
            fft,
            ifft,
            work: vec![Complex32::default(); FFT],
            scratch,
            overlap: [0.0; BLOCK],
        }
    }

    pub fn reset(&mut self) {
        for h in &mut self.history {
            h.fill(Complex32::default());
        }
        self.overlap.fill(0.0);
        self.head = 0;
    }

    pub fn process(&mut self, input: &[f32; BLOCK], output: &mut [f32; BLOCK]) {
        let current = &mut self.history[self.head];
        current.fill(Complex32::default());
        for (c, &x) in current.iter_mut().zip(input) {
            c.re = x;
        }
        self.fft.process_with_scratch(current, &mut self.scratch);
        self.work.fill(Complex32::default());
        let n = self.filters.len();
        for p in 0..n {
            let h = &self.filters[p];
            let x = &self.history[(self.head + n - p) % n];
            for i in 0..FFT {
                self.work[i] += h[i] * x[i];
            }
        }
        self.ifft
            .process_with_scratch(&mut self.work, &mut self.scratch);
        for i in 0..BLOCK {
            output[i] = self.work[i].re / FFT as f32 + self.overlap[i];
            self.overlap[i] = self.work[i + BLOCK].re / FFT as f32;
        }
        self.head = (self.head + 1) % n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_direct_sum_across_partitions_and_tail() {
        for length in [1, 255, 256, 257, 801] {
            let h: Vec<_> = (0..length)
                .map(|i| (i as f32 * 0.13).cos() * 0.01)
                .collect();
            let x: Vec<_> = (0..BLOCK * 4).map(|i| (i as f32 * 0.031).sin()).collect();
            let mut conv = Convolver::new(&h);
            for b in 0..9 {
                let input = std::array::from_fn(|i| x.get(b * BLOCK + i).copied().unwrap_or(0.0));
                let mut out = [0.0; BLOCK];
                conv.process(&input, &mut out);
                for (i, y) in out.iter().enumerate() {
                    let t = b * BLOCK + i;
                    let expected: f32 = h
                        .iter()
                        .enumerate()
                        .filter_map(|(k, h)| t.checked_sub(k).and_then(|j| x.get(j)).map(|x| x * h))
                        .sum();
                    assert!((y - expected).abs() < 2e-5, "length={length} t={t}");
                }
            }
            conv.reset();
            let mut out = [1.0; BLOCK];
            conv.process(&[0.0; BLOCK], &mut out);
            assert!(out.iter().all(|x| x.abs() < 1e-7));
        }
    }
}
