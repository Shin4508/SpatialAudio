use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

#[derive(Clone)]
pub struct Biquad {
    pub b: [f32; 3],
    pub a: [f32; 3],
    z1: f32,
    z2: f32,
}

impl Biquad {
    pub fn new(b: [f32; 3], a: [f32; 3]) -> Self {
        Self { b, a, z1: 0.0, z2: 0.0 }
    }

    pub fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.b[0] * x + self.z1;
        self.z1 = self.b[1] * x - self.a[1] * y + self.z2;
        self.z2 = self.b[2] * x - self.a[2] * y;
        y
    }

    pub fn process_block(&mut self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| self.process_sample(x)).collect()
    }
}

pub fn peaking(sr: f32, f0: f32, gain_db: f32, q: f32) -> ([f32; 3], [f32; 3]) {
    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * f0 / sr;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();

    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0 - alpha * a;
    let a0 = 1.0 + alpha / a;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha / a;

    ([b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0])
}

pub fn low_shelf(sr: f32, f0: f32, gain_db: f32, q: f32) -> ([f32; 3], [f32; 3]) {
    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * f0 / sr;
    let alpha = w0.sin() / 2.0 * ((a + 1.0/a) * (1.0/q - 1.0) + 2.0).sqrt();
    let c = w0.cos();
    let t = 2.0 * a.sqrt() * alpha;

    let b0 = a * ((a + 1.0) - (a - 1.0)*c + t);
    let b1 = 2.0*a*((a - 1.0) - (a + 1.0)*c);
    let b2 = a * ((a + 1.0) - (a - 1.0)*c - t);
    let a0 = (a + 1.0) + (a - 1.0)*c + t;
    let a1 = -2.0*((a - 1.0) + (a + 1.0)*c);
    let a2 = (a + 1.0) + (a - 1.0)*c - t;

    ([b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0])
}

pub fn high_shelf(sr: f32, f0: f32, gain_db: f32, q: f32) -> ([f32; 3], [f32; 3]) {
    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * f0 / sr;
    let alpha = w0.sin() / 2.0 * ((a + 1.0/a) * (1.0/q - 1.0) + 2.0).sqrt();
    let c = w0.cos();
    let t = 2.0 * a.sqrt() * alpha;

    let b0 = a * ((a + 1.0) + (a - 1.0)*c + t);
    let b1 = -2.0*a*((a - 1.0) + (a + 1.0)*c);
    let b2 = a * ((a + 1.0) + (a - 1.0)*c - t);
    let a0 = (a + 1.0) - (a - 1.0)*c + t;
    let a1 = 2.0*((a - 1.0) - (a + 1.0)*c);
    let a2 = (a + 1.0) - (a - 1.0)*c - t;

    ([b0/a0, b1/a0, b2/a0], [1.0, a1/a0, a2/a0])
}

pub struct FftConvolver {
    block_size: usize,
    fft_size: usize,
    h_fft: Vec<Complex<f32>>,
    input_buffer: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
}

impl FftConvolver {
    pub fn new(ir: &[f32], block_size: usize) -> Self {
        let filter_len = ir.len().max(1);
        let fft_size = (block_size + filter_len - 1).next_power_of_two();

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        let mut h_fft = vec![Complex::new(0.0, 0.0); fft_size];
        for (i, &x) in ir.iter().enumerate() {
            h_fft[i] = Complex::new(x, 0.0);
        }
        fft.process(&mut h_fft);

        Self {
            block_size,
            fft_size,
            h_fft,
            input_buffer: vec![0.0; fft_size],
            fft,
            ifft,
        }
    }

    pub fn process(&mut self, block: &[f32]) -> Vec<f32> {
        assert_eq!(block.len(), self.block_size);

        self.input_buffer.copy_within(self.block_size.., 0);
        let start = self.fft_size - self.block_size;
        self.input_buffer[start..].copy_from_slice(block);

        let mut work: Vec<Complex<f32>> = self.input_buffer
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        self.fft.process(&mut work);
        for (x, h) in work.iter_mut().zip(self.h_fft.iter()) {
            *x *= *h;
        }
        self.ifft.process(&mut work);

        let scale = 1.0 / self.fft_size as f32;
        work[start..].iter().map(|c| c.re * scale).collect()
    }
}
