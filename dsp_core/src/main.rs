use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

fn get_low_shelf_coeffs(sr: f32, f0: f32, gain_db: f32, q: f32) -> ([f32; 3], [f32; 3]) {
    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * f0 / sr;
    let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0).sqrt();
    let cos_w0 = w0.cos();
    let sqrt_a_alpha_2 = 2.0 * a.sqrt() * alpha;

    let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + sqrt_a_alpha_2);
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - sqrt_a_alpha_2);
    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + sqrt_a_alpha_2;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - sqrt_a_alpha_2;

    ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

fn get_high_shelf_coeffs(sr: f32, f0: f32, gain_db: f32, q: f32) -> ([f32; 3], [f32; 3]) {
    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * PI * f0 / sr;
    let alpha = w0.sin() / 2.0 * ((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0).sqrt();
    let cos_w0 = w0.cos();
    let sqrt_a_alpha_2 = 2.0 * a.sqrt() * alpha;

    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + sqrt_a_alpha_2);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - sqrt_a_alpha_2);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + sqrt_a_alpha_2;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - sqrt_a_alpha_2;

    ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

fn butter_lowpass_2nd(sr: f32, cutoff: f32) -> ([f32; 3], [f32; 3]) {
    let w0 = 2.0 * PI * cutoff / sr;
    let alpha = w0.sin() / (2.0 * 0.7071); // Q = 1/sqrt(2)
    let cos_w0 = w0.cos();

    let b0 = (1.0 - cos_w0) / 2.0;
    let b1 = 1.0 - cos_w0;
    let b2 = (1.0 - cos_w0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

fn butter_highpass_2nd(sr: f32, cutoff: f32) -> ([f32; 3], [f32; 3]) {
    let w0 = 2.0 * PI * cutoff / sr;
    let alpha = w0.sin() / (2.0 * 0.70710678); // Butterworth Q = 1/sqrt(2)
    let cos_w0 = w0.cos();

    let b0 = (1.0 + cos_w0) / 2.0;
    let b1 = -(1.0 + cos_w0);
    let b2 = (1.0 + cos_w0) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

fn get_peaking_eq_coeffs(sr: f32, f0: f32, gain_db: f32, q: f32) -> ([f32; 3], [f32; 3]) {
    let a = 10.0_f32.powf(gain_db / 40.0);
    let w0 = 2.0 * std::f32::consts::PI * f0 / sr;
    let alpha = w0.sin() / (2.0 * q);

    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * w0.cos();
    let b2 = 1.0 - alpha * a;
    let a0 = 1.0 + alpha / a;
    let a1 = -2.0 * w0.cos();
    let a2 = 1.0 - alpha / a;

    ([b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0])
}

#[derive(Clone)]
pub struct RealTimeBiquadFilter {
    b: [f32; 3],
    a: [f32; 3],
    z1: f32,
    z2: f32,
}

impl RealTimeBiquadFilter {
    pub fn new(b: [f32; 3], a: [f32; 3]) -> Self {
        Self {
            b,
            a,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Write into caller-owned storage. No allocation occurs here.
    pub fn process_into(&mut self, input: &[f32], output: &mut [f32]) {
        assert!(output.len() >= input.len());
        for (&x, y_out) in input.iter().zip(output.iter_mut()) {
            let y = self.b[0] * x + self.z1;
            self.z1 = self.b[1] * x - self.a[1] * y + self.z2;
            self.z2 = self.b[2] * x - self.a[2] * y;
            *y_out = y;
        }
    }

    /// Biquads are naturally safe to run in-place because each input sample is
    /// consumed before that slot is overwritten with the output sample.
    pub fn process_in_place(&mut self, buffer: &mut [f32]) {
        for sample in buffer {
            let x = *sample;
            let y = self.b[0] * x + self.z1;
            self.z1 = self.b[1] * x - self.a[1] * y + self.z2;
            self.z2 = self.b[2] * x - self.a[2] * y;
            *sample = y;
        }
    }
}

pub struct RealTimeCrossover {
    // Linkwitz-Riley 4th order = two cascaded 2nd-order Butterworth sections.
    // Low + High are phase aligned at crossover and sum close to flat.
    lp1: RealTimeBiquadFilter,
    lp2: RealTimeBiquadFilter,
    hp1: RealTimeBiquadFilter,
    hp2: RealTimeBiquadFilter,
}

impl RealTimeCrossover {
    pub fn new(sr: f32, cutoff_freq: f32) -> Self {
        let (b_lp, a_lp) = butter_lowpass_2nd(sr, cutoff_freq);
        let (b_hp, a_hp) = butter_highpass_2nd(sr, cutoff_freq);

        Self {
            lp1: RealTimeBiquadFilter::new(b_lp, a_lp),
            lp2: RealTimeBiquadFilter::new(b_lp, a_lp),
            hp1: RealTimeBiquadFilter::new(b_hp, a_hp),
            hp2: RealTimeBiquadFilter::new(b_hp, a_hp),
        }
    }

    pub fn process_into(&mut self, block: &[f32], low: &mut [f32], high: &mut [f32]) {
        let n = block.len();
        assert!(low.len() >= n && high.len() >= n);

        low[..n].copy_from_slice(block);
        self.lp1.process_in_place(&mut low[..n]);
        self.lp2.process_in_place(&mut low[..n]);

        high[..n].copy_from_slice(block);
        self.hp1.process_in_place(&mut high[..n]);
        self.hp2.process_in_place(&mut high[..n]);
    }
}

pub struct RealTimeActiveEQ {
    sr: f32,
    target_freq: f32,
    filter: RealTimeBiquadFilter,
    current_gain_db: f32,
    smoothing_factor: f32,
}

impl RealTimeActiveEQ {
    pub fn new(sr: f32, target_freq: f32) -> Self {
        let (b, a) = get_low_shelf_coeffs(sr, target_freq, 0.0, 0.707);
        Self {
            sr,
            target_freq,
            filter: RealTimeBiquadFilter::new(b, a),
            current_gain_db: 0.0,
            smoothing_factor: 0.05,
        }
    }

    pub fn process_in_place(&mut self, block: &mut [f32]) {
        if block.is_empty() {
            return;
        }

        let rms = (block.iter().map(|&x| x * x).sum::<f32>() / block.len() as f32).sqrt();
        let threshold = 0.05;

        let target_gain_db = if rms < threshold {
            6.0 * (1.0 - (rms / threshold)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        self.current_gain_db +=
            (target_gain_db - self.current_gain_db) * self.smoothing_factor;

        let (b, a) = get_low_shelf_coeffs(
            self.sr,
            self.target_freq,
            self.current_gain_db,
            0.707,
        );
        self.filter.b = b;
        self.filter.a = a;
        self.filter.process_in_place(block);
    }
}

pub struct RealTimeDelayLine {
    gain: f32,
    buffer: Vec<f32>,
    write_idx: usize,
    delay_samples: usize,
    lpf: RealTimeBiquadFilter,
}

impl RealTimeDelayLine {
    pub fn new(sr: f32, delay_ms: f32, gain: f32, cutoff_hz: f32) -> Self {
        let delay_samples = (sr * (delay_ms / 1000.0)) as usize;
        let (b, a) = butter_lowpass_2nd(sr, cutoff_hz);
        Self {
            gain,
            buffer: vec![0.0; delay_samples + 1024], // リングバッファ
            write_idx: 0,
            delay_samples,
            lpf: RealTimeBiquadFilter::new(b, a),
        }
    }

    pub fn process_into(&mut self, block: &[f32], output: &mut [f32]) {
        let n = block.len();
        assert!(output.len() >= n);
        let cap = self.buffer.len();

        // Use the destination itself as the temporary delayed block.
        for (&sample, delayed) in block.iter().zip(output.iter_mut()).take(n) {
            let read_idx = (self.write_idx + cap - self.delay_samples) % cap;
            *delayed = self.buffer[read_idx];

            self.buffer[self.write_idx] = sample;
            self.write_idx = (self.write_idx + 1) % cap;
        }

        self.lpf.process_in_place(&mut output[..n]);
        for x in &mut output[..n] {
            *x *= self.gain;
        }
    }
}

pub struct MidSideProcessor {
    current_width: f32,
    target_width: f32,
    smoothing_factor: f32,
}

impl MidSideProcessor {
    pub fn new(width: f32) -> Self {
        Self {
            current_width: width,
            target_width: width,
            smoothing_factor: 0.03,
        }
    }

    pub fn set_width(&mut self, width: f32) {
        self.target_width = width.clamp(0.0, 1.5);
    }

    /// M=(L+R)/2, S=(L-R)/2, then only S is width-scaled.
    pub fn process_into(
        &mut self,
        input_l: &[f32],
        input_r: &[f32],
        mid: &mut [f32],
        side: &mut [f32],
    ) {
        let n = input_l.len().min(input_r.len());
        assert!(mid.len() >= n && side.len() >= n);

        self.current_width +=
            (self.target_width - self.current_width) * self.smoothing_factor;
        let width = self.current_width;

        for i in 0..n {
            let l = input_l[i];
            let r = input_r[i];
            mid[i] = (l + r) * 0.5;
            side[i] = (l - r) * 0.5 * width;
        }
    }

    pub fn decode_into(mid: &[f32], side: &[f32], left: &mut [f32], right: &mut [f32]) {
        let n = mid.len().min(side.len());
        assert!(left.len() >= n && right.len() >= n);
        for i in 0..n {
            left[i] = mid[i] + side[i];
            right[i] = mid[i] - side[i];
        }
    }
}

struct FdnDelay {
    buffer: Vec<f32>,
    index: usize,
    damp_state: f32,
}

impl FdnDelay {
    fn new(delay_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            index: 0,
            damp_state: 0.0,
        }
    }

    fn read_damped(&mut self, damping_alpha: f32) -> f32 {
        let x = self.buffer[self.index];
        self.damp_state += damping_alpha * (x - self.damp_state);
        self.damp_state
    }

    fn write_and_advance(&mut self, x: f32) {
        self.buffer[self.index] = x;
        self.index += 1;
        if self.index >= self.buffer.len() {
            self.index = 0;
        }
    }
}

pub struct StereoFdnReverb {
    delays: [FdnDelay; 4],
    feedback: f32,
    damping_alpha: f32,
    mid_input_gain: f32,
    side_input_gain: f32,
    output_gain: f32,
}

impl StereoFdnReverb {
    pub fn new(sr: f32) -> Self {
        // Mutually different delays reduce obvious periodic coloration.
        let delay_ms = [29.7_f32, 37.1, 41.1, 43.7];
        let delays = delay_ms.map(|ms| FdnDelay::new((sr * ms / 1000.0).round() as usize));

        // One-pole damping inside the feedback loop. Late reverberation is
        // intentionally darker than the direct HRTF / early-reflection field.
        let damping_cutoff_hz = 5500.0_f32;
        let damping_alpha = 1.0 - (-2.0 * PI * damping_cutoff_hz / sr).exp();

        Self {
            delays,
            feedback: 0.72,
            damping_alpha,
            mid_input_gain: 0.22,
            side_input_gain: 0.14,
            output_gain: 0.35,
        }
    }

    /// Feed the diffuse field from both M and S into caller-owned outputs.
    pub fn process_ms_into(
        &mut self,
        mid: &[f32],
        side: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let n = mid.len().min(side.len());
        assert!(out_l.len() >= n && out_r.len() >= n);

        let mid_pattern = [1.0_f32, 1.0, 1.0, 1.0];
        let side_pattern = [1.0_f32, -1.0, -1.0, 1.0];

        for i_sample in 0..n {
            let m = mid[i_sample];
            let s = side[i_sample];
            let mut d = [0.0_f32; 4];
            for (i, delay) in self.delays.iter_mut().enumerate() {
                d[i] = delay.read_damped(self.damping_alpha);
            }

            let mixed = [
                (d[0] + d[1] + d[2] + d[3]) * 0.5,
                (d[0] - d[1] + d[2] - d[3]) * 0.5,
                (d[0] + d[1] - d[2] - d[3]) * 0.5,
                (d[0] - d[1] - d[2] + d[3]) * 0.5,
            ];

            let mid_in = m * self.mid_input_gain;
            let side_in = s * self.side_input_gain;
            for i in 0..4 {
                let inject = mid_in * mid_pattern[i] + side_in * side_pattern[i];
                self.delays[i].write_and_advance(inject + self.feedback * mixed[i]);
            }

            out_l[i_sample] = (d[0] + d[2]) * 0.70710678 * self.output_gain;
            out_r[i_sample] = (d[1] + d[3]) * 0.70710678 * self.output_gain;
        }
    }
}

pub struct RealTimeOverlapSave {
    l: usize,
    fft_size: usize,
    h_fft: Vec<Complex<f32>>,
    input_buffer: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
}

impl RealTimeOverlapSave {
    pub fn new(hrtf_impulse: &[f32], block_size: usize) -> Self {
        let n = hrtf_impulse.len();
        let m = block_size + n - 1;
        let fft_size = m.next_power_of_two(); // N = 2^ceil(log2(M))

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        // インパルス応答をパディングしてFFT
        let mut h_padded: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); fft_size];
        for (i, &val) in hrtf_impulse.iter().enumerate() {
            h_padded[i] = Complex::new(val, 0.0);
        }
        fft.process(&mut h_padded);

        Self {
            l: block_size,
            fft_size,
            h_fft: h_padded,
            input_buffer: vec![0.0; fft_size],
            fft,
            ifft,
        }
    }

    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    pub fn scratch_len(&self) -> usize {
        self.fft
            .get_inplace_scratch_len()
            .max(self.ifft.get_inplace_scratch_len())
    }

    /// Overlap-save using shared FFT work/scratch buffers owned by the engine.
    /// This avoids both the Complex Vec allocation and RustFFT scratch allocation
    /// on every audio block.
    pub fn process_into(
        &mut self,
        block: &[f32],
        output: &mut [f32],
        work_buf: &mut [Complex<f32>],
        fft_scratch: &mut [Complex<f32>],
    ) {
        assert_eq!(block.len(), self.l, "overlap-save requires the configured block size");
        assert!(output.len() >= self.l);
        assert!(work_buf.len() >= self.fft_size);
        assert!(fft_scratch.len() >= self.scratch_len());

        self.input_buffer.copy_within(self.l.., 0);
        let start_idx = self.fft_size - self.l;
        self.input_buffer[start_idx..].copy_from_slice(block);

        let work = &mut work_buf[..self.fft_size];
        for (dst, &x) in work.iter_mut().zip(self.input_buffer.iter()) {
            *dst = Complex::new(x, 0.0);
        }

        self.fft.process_with_scratch(work, fft_scratch);
        for (x, h) in work.iter_mut().zip(self.h_fft.iter()) {
            *x *= *h;
        }
        self.ifft.process_with_scratch(work, fft_scratch);

        let scale = 1.0 / self.fft_size as f32;
        for i in 0..self.l {
            output[i] = work[start_idx + i].re * scale;
        }
    }
}

pub struct RoomAcousticAnalyzer {
    // 平滑化（Smoothing）用パラメータ
    current_er_pct: f32,
    current_reverb_pct: f32,
    smoothing_factor: f32, // 急激なゲイン変化によるノイズ（Zipper Noise）防止
}
impl RoomAcousticAnalyzer {
    pub fn new() -> Self {
        Self {
            current_er_pct: 0.5,
            current_reverb_pct: 0.5,
            smoothing_factor: 0.05,
        }
    }

    /// Estimate wetness from total M/S energy rather than Mid alone. This
    /// matters for strongly stereo / anti-correlated material where Mid can
    /// be small even though the program material is loud.
    pub fn analyze_ms(&mut self, mid: &[f32], side: &[f32]) -> (f32, f32) {
        let n = mid.len().min(side.len());
        if n < 4 {
            return (self.current_er_pct, self.current_reverb_pct);
        }

        let energy = mid
            .iter()
            .zip(side.iter())
            .take(n)
            .map(|(&m, &s)| m * m + s * s)
            .sum::<f32>();
        let rms = (energy / n as f32).sqrt();

        if rms < 0.001 {
            return (self.current_er_pct, self.current_reverb_pct);
        }

        let sub_block_size = n / 4;
        let mut sub_rms = [0.0_f32; 4];
        for i in 0..4 {
            let start = i * sub_block_size;
            let end = if i == 3 { n } else { start + sub_block_size };
            let len = (end - start).max(1);

            let e = mid[start..end]
                .iter()
                .zip(side[start..end].iter())
                .map(|(&m, &s)| m * m + s * s)
                .sum::<f32>();
            sub_rms[i] = (e / len as f32).sqrt();
        }

        let transient_ratio = sub_rms[0] / (rms + 1e-6);
        let target_er_pct = if transient_ratio > 1.5 { 0.85 } else { 0.30 };

        let tail_ratio = sub_rms[3] / (sub_rms[0] + 1e-6);
        let target_reverb_pct = if tail_ratio < 0.1 { 0.70 } else { 0.20 };

        self.current_er_pct += (target_er_pct - self.current_er_pct) * self.smoothing_factor;
        self.current_reverb_pct +=
            (target_reverb_pct - self.current_reverb_pct) * self.smoothing_factor;

        (self.current_er_pct, self.current_reverb_pct)
    }
}

pub struct DistanceFilter {
    sr: f32,
    low_shelf: RealTimeBiquadFilter,
    high_shelf: RealTimeBiquadFilter,
    current_distance: f32,
}

impl DistanceFilter {
    pub fn new(sr: f32) -> Self {
        // 初期状態は 1.0m (標準距離)
        let (b_ls, a_ls) = get_low_shelf_coeffs(sr, 200.0, 0.0, 0.707);
        let (b_hs, a_hs) = get_high_shelf_coeffs(sr, 6000.0, 0.0, 0.707);

        Self {
            sr,
            low_shelf: RealTimeBiquadFilter::new(b_ls, a_ls),
            high_shelf: RealTimeBiquadFilter::new(b_hs, a_hs),
            current_distance: 1.0,
        }
    }

    /// Update distance-dependent coefficients and filter the block in-place.
    pub fn process_in_place(&mut self, block: &mut [f32], target_distance_m: f32) {
        let d = target_distance_m.max(0.1);
        self.current_distance += (d - self.current_distance) * 0.05;
        let dist = self.current_distance;

        let proximity_gain_db = if dist > 0.1 {
            -6.0 * (dist / 0.1).log10().clamp(0.0, 1.5)
        } else {
            0.0
        };
        let (b_ls, a_ls) = get_low_shelf_coeffs(self.sr, 180.0, proximity_gain_db, 0.707);
        self.low_shelf.b = b_ls;
        self.low_shelf.a = a_ls;

        let air_absorption_gain_db = -1.5 * (dist - 1.0).max(0.0);
        let (b_hs, a_hs) = get_high_shelf_coeffs(
            self.sr,
            6000.0,
            air_absorption_gain_db.max(-12.0),
            0.707,
        );
        self.high_shelf.b = b_hs;
        self.high_shelf.a = a_hs;

        let ref_distance = 0.5;
        let distance_gain = (ref_distance / dist).min(1.0);

        self.low_shelf.process_in_place(block);
        self.high_shelf.process_in_place(block);
        for x in block {
            *x *= distance_gain;
        }
    }
}

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::HeapRb;
use ringbuf::traits::{Consumer, Producer, Split};
use std::process::Command;

/// Reusable time-domain buffers. They are allocated once in `new()` and
/// recycled for every audio callback. Some names are deliberately generic:
/// tmp/late storage is reused by later stages after earlier values are consumed.
pub struct DspBuffers {
    mid: Vec<f32>,
    side: Vec<f32>,
    mid_low: Vec<f32>,
    mid_high: Vec<f32>,
    rendered_l: Vec<f32>,
    rendered_r: Vec<f32>,
    tmp_l: Vec<f32>,
    tmp_r: Vec<f32>,
    tmp2_l: Vec<f32>,
    tmp2_r: Vec<f32>,
    late_l: Vec<f32>,
    late_r: Vec<f32>,
}

impl DspBuffers {
    fn new(block_size: usize) -> Self {
        let zero = || vec![0.0; block_size];
        Self {
            mid: zero(),
            side: zero(),
            mid_low: zero(),
            mid_high: zero(),
            rendered_l: zero(),
            rendered_r: zero(),
            tmp_l: zero(),
            tmp_r: zero(),
            tmp2_l: zero(),
            tmp2_r: zero(),
            late_l: zero(),
            late_r: zero(),
        }
    }

    fn len(&self) -> usize {
        self.mid.len()
    }
}

pub struct SpatialAudioEngine {
    analyzer: RoomAcousticAnalyzer,
    ms: MidSideProcessor,
    dist_filter_mid: DistanceFilter,
    dist_filter_side: DistanceFilter,
    crossover: RealTimeCrossover,
    eq_mid: RealTimeActiveEQ,
    eq_side: RealTimeActiveEQ,

    // DIRECT FIELD, preserving the original intended geometry:
    //   Mid  -> front 0
    //   +Side -> left-front 63
    //   -Side -> right-front 9
    mid_front_l: RealTimeOverlapSave,
    mid_front_r: RealTimeOverlapSave,
    side_front_l: RealTimeOverlapSave,
    side_front_r: RealTimeOverlapSave,

    // EARLY REFLECTIONS, also preserving the original M/S idea:
    //   Mid  -> left-rear 40 / right-rear 32
    //   Side -> left-rear 40 / right-rear 32, with M/S polarity retained.
    // Mid and Side need independent convolution state because they are
    // independent input streams even when they use the same IRs.
    mid_er_left_rear: RealTimeOverlapSave,
    mid_er_right_rear: RealTimeOverlapSave,
    side_er_left_rear: RealTimeOverlapSave,
    side_er_right_rear: RealTimeOverlapSave,

    mid_er_delay_left: RealTimeDelayLine,
    mid_er_delay_right: RealTimeDelayLine,
    side_er_delay_left: RealTimeDelayLine,
    side_er_delay_right: RealTimeDelayLine,

    // Direct HRTF cues stay full-band. Only the reflected field is darkened.
    er_lp_l: RealTimeBiquadFilter,
    er_lp_r: RealTimeBiquadFilter,
    side_shelf_l: RealTimeBiquadFilter,
    side_shelf_r: RealTimeBiquadFilter,

    late_reverb: StereoFdnReverb,

    // Small residual headphone crossfeed. HRTF rendering already provides
    // contralateral energy, so this is intentionally subtle.
    cross_delay_l: RealTimeDelayLine,
    cross_delay_r: RealTimeDelayLine,

    // Shared workspaces: one set for the whole serial convolution pipeline.
    // Keeping these in each convolver would waste ~FFT_SIZE * 8 bytes per path.
    fft_work: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
    buffers: DspBuffers,
}

impl SpatialAudioEngine {
    pub fn new(sr: f32, hrtf_data: &HrtfData, block_size: usize) -> Self {
        let (b_er_lp, a_er_lp) = butter_lowpass_2nd(sr, 6500.0);
        let (b_sh, a_sh) = get_high_shelf_coeffs(sr, 2500.0, -2.5, 0.707);

        // Build the convolvers first so the largest FFT and RustFFT scratch
        // requirements can be measured once and shared across every path.
        let mid_front_l = RealTimeOverlapSave::new(&hrtf_data.mid_front_l, block_size);
        let mid_front_r = RealTimeOverlapSave::new(&hrtf_data.mid_front_r, block_size);
        let side_front_l = RealTimeOverlapSave::new(&hrtf_data.side_front_l, block_size);
        let side_front_r = RealTimeOverlapSave::new(&hrtf_data.side_front_r, block_size);
        let mid_er_left_rear = RealTimeOverlapSave::new(&hrtf_data.rear_left, block_size);
        let mid_er_right_rear = RealTimeOverlapSave::new(&hrtf_data.rear_right, block_size);
        let side_er_left_rear = RealTimeOverlapSave::new(&hrtf_data.rear_left, block_size);
        let side_er_right_rear = RealTimeOverlapSave::new(&hrtf_data.rear_right, block_size);

        let fft_work_len = [
            mid_front_l.fft_size(),
            mid_front_r.fft_size(),
            side_front_l.fft_size(),
            side_front_r.fft_size(),
            mid_er_left_rear.fft_size(),
            mid_er_right_rear.fft_size(),
            side_er_left_rear.fft_size(),
            side_er_right_rear.fft_size(),
        ]
        .into_iter()
        .max()
        .unwrap_or(1);

        let fft_scratch_len = [
            mid_front_l.scratch_len(),
            mid_front_r.scratch_len(),
            side_front_l.scratch_len(),
            side_front_r.scratch_len(),
            mid_er_left_rear.scratch_len(),
            mid_er_right_rear.scratch_len(),
            side_er_left_rear.scratch_len(),
            side_er_right_rear.scratch_len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        Self {
            analyzer: RoomAcousticAnalyzer::new(),
            ms: MidSideProcessor::new(0.80),
            crossover: RealTimeCrossover::new(sr, 100.0),
            dist_filter_mid: DistanceFilter::new(sr),
            dist_filter_side: DistanceFilter::new(sr),
            eq_mid: RealTimeActiveEQ::new(sr, 80.0),
            eq_side: RealTimeActiveEQ::new(sr, 80.0),

            mid_front_l,
            mid_front_r,
            side_front_l,
            side_front_r,
            mid_er_left_rear,
            mid_er_right_rear,
            side_er_left_rear,
            side_er_right_rear,

            mid_er_delay_left: RealTimeDelayLine::new(sr, 23.0, 0.34, 5500.0),
            mid_er_delay_right: RealTimeDelayLine::new(sr, 18.0, 0.34, 5500.0),
            side_er_delay_left: RealTimeDelayLine::new(sr, 23.0, 0.25, 5200.0),
            side_er_delay_right: RealTimeDelayLine::new(sr, 18.0, 0.25, 5200.0),

            er_lp_l: RealTimeBiquadFilter::new(b_er_lp, a_er_lp),
            er_lp_r: RealTimeBiquadFilter::new(b_er_lp, a_er_lp),
            side_shelf_l: RealTimeBiquadFilter::new(b_sh, a_sh),
            side_shelf_r: RealTimeBiquadFilter::new(b_sh, a_sh),
            late_reverb: StereoFdnReverb::new(sr),
            cross_delay_l: RealTimeDelayLine::new(sr, 0.5, 1.0, 1200.0),
            cross_delay_r: RealTimeDelayLine::new(sr, 0.5, 1.0, 1200.0),

            fft_work: vec![Complex::new(0.0, 0.0); fft_work_len],
            fft_scratch: vec![Complex::new(0.0, 0.0); fft_scratch_len],
            buffers: DspBuffers::new(block_size),
        }
    }

    /// Real-time entry point. All working memory already exists before this is
    /// called, so the processing path performs no deliberate Vec allocation.
    pub fn process_block_into(
        &mut self,
        input_l: &[f32],
        input_r: &[f32],
        target_distance_m: f32,
        out_left: &mut [f32],
        out_right: &mut [f32],
    ) {
        let n = input_l.len().min(input_r.len());
        assert_eq!(n, self.buffers.len(), "input must match configured DSP block size");
        assert!(out_left.len() >= n && out_right.len() >= n);
        let input_l = &input_l[..n];
        let input_r = &input_r[..n];

        // 1) M/S encode directly into persistent buffers.
        self.ms.process_into(
            input_l,
            input_r,
            &mut self.buffers.mid,
            &mut self.buffers.side,
        );

        // Distance and active EQ are in-place; no intermediate Mid/Side Vecs.
        self.dist_filter_mid
            .process_in_place(&mut self.buffers.mid, target_distance_m);
        self.dist_filter_side
            .process_in_place(&mut self.buffers.side, target_distance_m);
        self.eq_mid.process_in_place(&mut self.buffers.mid);
        self.eq_side.process_in_place(&mut self.buffers.side);

        let (mut er_pct, mut reverb_pct) = self
            .analyzer
            .analyze_ms(&self.buffers.mid, &self.buffers.side);
        let distance_factor = 1.0 + (target_distance_m - 0.5) * 0.4;
        er_pct = (er_pct * distance_factor).clamp(0.0, 1.0);
        reverb_pct = (reverb_pct * distance_factor).clamp(0.0, 1.0);

        // 2) MID DIRECT. Crossover outputs are persistent buffers.
        self.crossover.process_into(
            &self.buffers.mid,
            &mut self.buffers.mid_low,
            &mut self.buffers.mid_high,
        );

        // Render Mid HRTF directly into the final accumulation buffers.
        self.mid_front_l.process_into(
            &self.buffers.mid_high,
            &mut self.buffers.rendered_l,
            &mut self.fft_work,
            &mut self.fft_scratch,
        );
        self.mid_front_r.process_into(
            &self.buffers.mid_high,
            &mut self.buffers.rendered_r,
            &mut self.fft_work,
            &mut self.fft_scratch,
        );
        for i in 0..n {
            self.buffers.rendered_l[i] =
                self.buffers.mid_low[i] + self.buffers.rendered_l[i] * 0.90;
            self.buffers.rendered_r[i] =
                self.buffers.mid_low[i] + self.buffers.rendered_r[i] * 0.90;
        }

        // 3) SIDE DIRECT. tmp L/R are reused again later for delay/crossfeed.
        self.side_front_l.process_into(
            &self.buffers.side,
            &mut self.buffers.tmp_l,
            &mut self.fft_work,
            &mut self.fft_scratch,
        );
        self.side_front_r.process_into(
            &self.buffers.side,
            &mut self.buffers.tmp_r,
            &mut self.fft_work,
            &mut self.fft_scratch,
        );
        self.side_shelf_l.process_in_place(&mut self.buffers.tmp_l);
        self.side_shelf_r.process_in_place(&mut self.buffers.tmp_r);
        for i in 0..n {
            self.buffers.rendered_l[i] += self.buffers.tmp_l[i] * 0.55;
            self.buffers.rendered_r[i] -= self.buffers.tmp_r[i] * 0.55;
        }

        // 4) MID ER. tmp = delayed source, tmp2 = convolved rear field.
        self.mid_er_delay_left
            .process_into(&self.buffers.mid, &mut self.buffers.tmp_l);
        self.mid_er_delay_right
            .process_into(&self.buffers.mid, &mut self.buffers.tmp_r);
        self.mid_er_left_rear.process_into(
            &self.buffers.tmp_l,
            &mut self.buffers.tmp2_l,
            &mut self.fft_work,
            &mut self.fft_scratch,
        );
        self.mid_er_right_rear.process_into(
            &self.buffers.tmp_r,
            &mut self.buffers.tmp2_r,
            &mut self.fft_work,
            &mut self.fft_scratch,
        );

        // 5) SIDE ER. Reuse tmp for sources and late L/R as temporary output;
        // the FDN will overwrite late L/R after the ER values are consumed.
        self.side_er_delay_left
            .process_into(&self.buffers.side, &mut self.buffers.tmp_l);
        self.side_er_delay_right
            .process_into(&self.buffers.side, &mut self.buffers.tmp_r);
        self.side_er_left_rear.process_into(
            &self.buffers.tmp_l,
            &mut self.buffers.late_l,
            &mut self.fft_work,
            &mut self.fft_scratch,
        );
        self.side_er_right_rear.process_into(
            &self.buffers.tmp_r,
            &mut self.buffers.late_r,
            &mut self.fft_work,
            &mut self.fft_scratch,
        );

        // Reconstruct reflected M/S field into tmp2 and darken only ER.
        for i in 0..n {
            self.buffers.tmp2_l[i] += self.buffers.late_l[i];
            self.buffers.tmp2_r[i] -= self.buffers.late_r[i];
        }
        self.er_lp_l.process_in_place(&mut self.buffers.tmp2_l);
        self.er_lp_r.process_in_place(&mut self.buffers.tmp2_r);
        for i in 0..n {
            self.buffers.rendered_l[i] += self.buffers.tmp2_l[i] * er_pct;
            self.buffers.rendered_r[i] += self.buffers.tmp2_r[i] * er_pct;
        }

        // 6) LATE REVERB overwrites the temporary Side-ER buffers.
        self.late_reverb.process_ms_into(
            &self.buffers.mid,
            &self.buffers.side,
            &mut self.buffers.late_l,
            &mut self.buffers.late_r,
        );
        for i in 0..n {
            self.buffers.rendered_l[i] += self.buffers.late_l[i] * reverb_pct;
            self.buffers.rendered_r[i] += self.buffers.late_r[i] * reverb_pct;
        }

        // 7) Residual crossfeed. tmp L/R are free again, so reuse them.
        let cross_feed_level = 0.08;
        self.cross_delay_l
            .process_into(&self.buffers.rendered_l, &mut self.buffers.tmp_l);
        self.cross_delay_r
            .process_into(&self.buffers.rendered_r, &mut self.buffers.tmp_r);

        for i in 0..n {
            let mixed_l = self.buffers.rendered_l[i] + self.buffers.tmp_r[i] * cross_feed_level;
            let mixed_r = self.buffers.rendered_r[i] + self.buffers.tmp_l[i] * cross_feed_level;
            out_left[i] = (mixed_l * 0.72).tanh();
            out_right[i] = (mixed_r * 0.72).tanh();
        }
    }
}

pub struct HrtfData {
    // Original MAT-derived geometry:
    // 0 = front, 63/9 = mirrored front pair, 40/32 = mirrored rear pair.
    pub mid_front_l: Vec<f32>,  // left ear, index 0
    pub mid_front_r: Vec<f32>,  // right ear, index 0
    pub side_front_l: Vec<f32>, // left-front, index 63
    pub side_front_r: Vec<f32>, // right-front, index 9
    pub rear_left: Vec<f32>,    // left-rear, index 40
    pub rear_right: Vec<f32>,   // right-rear, index 32
}

fn load_impulse_response(filename: &str) -> Vec<f32> {
    let mut reader = match hound::WavReader::open(filename) {
        Ok(r) => r,
        Err(_) => {
            eprintln!("❌ エラー: HRTFファイル '{}' が見つかりません！", filename);
            return vec![0.0; 256];
        }
    };

    let samples: Vec<f32> = reader
        .samples::<f32>()
        .map(|s| s.expect("サンプルの読み込みに失敗しました"))
        .collect();

    println!(
        "✅ {} を読み込みました！(長さ: {} サンプル)",
        filename,
        samples.len()
    );
    samples
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 空間オーディオ DSPエンジン起動...");

    let host = cpal::host_from_id(cpal::HostId::Jack).expect("JACKが動いていません");
    let input_device = host.default_input_device().expect("入力なし");
    let output_device = host.default_output_device().expect("出力なし");

    let config: cpal::StreamConfig = output_device.default_output_config()?.into();
    let sr = config.sample_rate.0 as f32;

    let block_size = 1024;

    let ring = HeapRb::<f32>::new((sr * 0.1) as usize * 2);
    let (mut producer, mut consumer) = ring.split();

    let hrtf_data = HrtfData {
        mid_front_l: load_impulse_response("hrtf_left_0.wav"),
        mid_front_r: load_impulse_response("hrtf_right_0.wav"),
        side_front_l: load_impulse_response("hrtf_left_63.wav"),
        side_front_r: load_impulse_response("hrtf_right_9.wav"),
        rear_left: load_impulse_response("hrtf_left_40.wav"),
        rear_right: load_impulse_response("hrtf_right_32.wav"),
    };
    let mut engine = SpatialAudioEngine::new(sr, &hrtf_data, block_size);

    let input_stream = input_device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            for &s in data {
                let _ = producer.try_push(s);
            }
        },
        |err| eprintln!("入力エラー: {}", err),
        None,
    )?;

    // Allocate callback buffers once; they are overwritten, never rebuilt.
    let mut buf_l = vec![0.0_f32; block_size];
    let mut buf_r = vec![0.0_f32; block_size];
    let mut out_l = vec![0.0_f32; block_size];
    let mut out_r = vec![0.0_f32; block_size];

    let output_stream = output_device.build_output_stream(
        &config,
        move |data: &mut [f32], _: &_| {
            let frames = data.len() / 2;
            if frames != block_size {
                // Overlap-save is configured for one fixed block size. Do not
                // allocate/resize in the real-time callback to accommodate a
                // surprise backend buffer size.
                data.fill(0.0);
                return;
            }

            for i in 0..block_size {
                buf_l[i] = consumer.try_pop().unwrap_or(0.0);
                buf_r[i] = consumer.try_pop().unwrap_or(0.0);
            }

            let target_distance_m: f32 = 1.5;
            engine.process_block_into(
                &buf_l,
                &buf_r,
                target_distance_m,
                &mut out_l,
                &mut out_r,
            );

            for (i, frame) in data.chunks_mut(2).enumerate() {
                frame[0] = out_l[i];
                frame[1] = out_r[i];
            }
        },
        |err| eprintln!("出力エラー: {}", err),
        None,
    )?;

    input_stream.play()?;
    output_stream.play()?;

    std::thread::sleep(std::time::Duration::from_millis(500));
    let auto_link_script = r#"
        L_PORT=$(pw-link -io | grep -o 'bluez_input[^:]*:output_FL' | head -n 1)
        R_PORT=$(pw-link -io | grep -o 'bluez_input[^:]*:output_FR' | head -n 1)

        # 2. ✂️ OSが勝手に繋いだ「直結バイパス」を強制切断（エラーは無視）
        pw-link -d "$L_PORT" "alsa_output.usb-Creative_Technology_Ltd_Sound_Blaster_Play__3_YDSB1730613003087M-00.analog-stereo:playback_FL" 2>/dev/null || true
        pw-link -d "$R_PORT" "alsa_output.usb-Creative_Technology_Ltd_Sound_Blaster_Play__3_YDSB1730613003087M-00.analog-stereo:playback_FR" 2>/dev/null || true
        if [ -n "$L_PORT" ] && [ -n "$R_PORT" ]; then
            pw-link "$L_PORT" "cpal_client_in:in_0"
            pw-link "$R_PORT" "cpal_client_in:in_1"
        fi
    "#;
    Command::new("bash")
        .arg("-c")
        .arg(auto_link_script)
        .status()
        .unwrap();

    println!("🎧 DSP稼働中... (Ctrl+Cで終了)");
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
