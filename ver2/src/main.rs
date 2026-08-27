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

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(input.len());
        for &x in input {
            // Direct Form II Transposed
            let y = self.b[0] * x + self.z1;
            self.z1 = self.b[1] * x - self.a[1] * y + self.z2;
            self.z2 = self.b[2] * x - self.a[2] * y;
            output.push(y);
        }
        output
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

    pub fn process(&mut self, block: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let low_stage1 = self.lp1.process(block);
        let low = self.lp2.process(&low_stage1);

        let high_stage1 = self.hp1.process(block);
        let high = self.hp2.process(&high_stage1);

        (low, high)
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

    pub fn process(&mut self, block: &[f32]) -> Vec<f32> {
        if block.is_empty() {
            return Vec::new();
        }

        let rms = (block.iter().map(|&x| x * x).sum::<f32>() / block.len() as f32).sqrt();
        let threshold = 0.05;

        // Original code tested rms > threshold and then computed (1 - rms/threshold),
        // which is always <= 0 in that branch, so the gain was permanently 0 dB.
        // Intended behavior here: gently restore up to +6 dB of low shelf on quiet material.
        let target_gain_db = if rms < threshold {
            6.0 * (1.0 - (rms / threshold)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Smooth coefficient changes to avoid zipper/pumping artifacts.
        self.current_gain_db += (target_gain_db - self.current_gain_db) * self.smoothing_factor;

        let (b, a) = get_low_shelf_coeffs(self.sr, self.target_freq, self.current_gain_db, 0.707);
        self.filter.b = b;
        self.filter.a = a;

        self.filter.process(block)
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

    pub fn process(&mut self, block: &[f32]) -> Vec<f32> {
        let mut delayed = Vec::with_capacity(block.len());
        let cap = self.buffer.len();

        for &sample in block {
            let read_idx = (self.write_idx + cap - self.delay_samples) % cap;
            delayed.push(self.buffer[read_idx]);

            self.buffer[self.write_idx] = sample;
            self.write_idx = (self.write_idx + 1) % cap;
        }

        self.lpf
            .process(&delayed)
            .iter()
            .map(|&x| x * self.gain)
            .collect()
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

    /// M/S is used as a basis transform / width control, not as a second room model.
    /// M=(L+R)/2, S=(L-R)/2, then only S is width-scaled.
    pub fn process(&mut self, input_l: &[f32], input_r: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = input_l.len().min(input_r.len());
        let mut mid = Vec::with_capacity(n);
        let mut side = Vec::with_capacity(n);

        self.current_width += (self.target_width - self.current_width) * self.smoothing_factor;
        let width = self.current_width;

        for (&l, &r) in input_l.iter().zip(input_r.iter()).take(n) {
            mid.push((l + r) * 0.5);
            side.push((l - r) * 0.5 * width);
        }

        (mid, side)
    }

    pub fn decode(mid: &[f32], side: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = mid.len().min(side.len());
        let mut left = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);

        for (&m, &s) in mid.iter().zip(side.iter()).take(n) {
            left.push(m + s);
            right.push(m - s);
        }

        (left, right)
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
    pre_delay_mid: RealTimeDelayLine,
    pre_delay_side: RealTimeDelayLine,
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
        let damping_cutoff_hz = 5000.0_f32;
        let damping_alpha = 1.0 - (-2.0 * PI * damping_cutoff_hz / sr).exp();

        Self {
            // Keep the late field perceptually separate from the discrete ER cluster.
            // The FDN begins after the last strong early reflection.
            pre_delay_mid: RealTimeDelayLine::new(sr, 55.0, 1.0, 12000.0),
            pre_delay_side: RealTimeDelayLine::new(sr, 55.0, 1.0, 12000.0),
            delays,
            feedback: 0.72,
            damping_alpha,
            mid_input_gain: 0.18,
            side_input_gain: 0.10,
            output_gain: 0.28,
        }
    }

    /// Feed the diffuse field from both M and S. Mid and Side use different
    /// orthogonal sign patterns so Side energy does not disappear from the
    /// late field, while the reverb still becomes decorrelated/diffuse.
    pub fn process_ms(&mut self, mid: &[f32], side: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = mid.len().min(side.len());
        let mut out_l = Vec::with_capacity(n);
        let mut out_r = Vec::with_capacity(n);

        // Two orthogonal Hadamard-basis injection patterns.
        let mid_pattern = [1.0_f32, 1.0, 1.0, 1.0];
        let side_pattern = [1.0_f32, -1.0, -1.0, 1.0];

        // Late-field pre-delay: discrete ER occupies the first ~50 ms.
        let pred_mid = self.pre_delay_mid.process(&mid[..n]);
        let pred_side = self.pre_delay_side.process(&side[..n]);

        for (&m, &s) in pred_mid.iter().zip(pred_side.iter()) {
            let mut d = [0.0_f32; 4];
            for (i, delay) in self.delays.iter_mut().enumerate() {
                d[i] = delay.read_damped(self.damping_alpha);
            }

            // Energy-preserving 4x4 Hadamard feedback matrix (scaled by 1/2).
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

            // Different mode combinations are sent to each ear to avoid a
            // mono late tail while keeping the overall energy balanced.
            let l = (d[0] + d[2]) * 0.70710678 * self.output_gain;
            let r = (d[1] + d[3]) * 0.70710678 * self.output_gain;
            out_l.push(l);
            out_r.push(r);
        }

        (out_l, out_r)
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

    pub fn process(&mut self, block: &[f32]) -> Vec<f32> {
        self.input_buffer.copy_within(self.l.., 0);
        let end_idx = self.fft_size;
        let start_idx = end_idx - self.l;
        self.input_buffer[start_idx..].copy_from_slice(block);

        let mut work_buf: Vec<Complex<f32>> = self
            .input_buffer
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        self.fft.process(&mut work_buf);

        for (x, h) in work_buf.iter_mut().zip(self.h_fft.iter()) {
            *x *= *h;
        }

        self.ifft.process(&mut work_buf);

        let scale = 1.0 / self.fft_size as f32;
        work_buf[start_idx..].iter().map(|c| c.re * scale).collect()
    }
}

/// One discrete early-reflection event.
///
/// Each event has its own delay, wall-loss low-pass, gain and cross-ear HRIR pair.
/// Side is injected with a sign chosen from the event's horizontal hemisphere so
/// that the original M/S stage width contributes without turning ER into a second
/// full-strength stereo image.
pub struct EarlyReflection {
    delay: RealTimeDelayLine,
    to_left: RealTimeOverlapSave,
    to_right: RealTimeOverlapSave,
    side_sign: f32,
    side_mix: f32,
}

impl EarlyReflection {
    pub fn new(
        sr: f32,
        block_size: usize,
        delay_ms: f32,
        gain: f32,
        cutoff_hz: f32,
        side_sign: f32,
        hrir_left: &[f32],
        hrir_right: &[f32],
    ) -> Self {
        Self {
            delay: RealTimeDelayLine::new(sr, delay_ms, gain, cutoff_hz),
            to_left: RealTimeOverlapSave::new(hrir_left, block_size),
            to_right: RealTimeOverlapSave::new(hrir_right, block_size),
            side_sign,
            side_mix: 0.45,
        }
    }

    pub fn process(&mut self, mid: &[f32], side: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = mid.len().min(side.len());
        let mut source = Vec::with_capacity(n);
        for i in 0..n {
            source.push(mid[i] + side[i] * self.side_sign * self.side_mix);
        }

        let delayed = self.delay.process(&source);
        let left = self.to_left.process(&delayed);
        let right = self.to_right.process(&delayed);
        (left, right)
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
            current_er_pct: 0.22,
            current_reverb_pct: 0.18,
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
        // Keep ER audible even on steady program material.
        let target_er_pct = if transient_ratio > 1.5 { 0.35 } else { 0.22 };

        let tail_ratio = sub_rms[3] / (sub_rms[0] + 1e-6);
        // The old 0.20 baseline buried the FDN under the direct field.
        let target_reverb_pct = if tail_ratio < 0.1 { 0.30 } else { 0.18 };

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

    /// 距離 d (m) に基づいてフィルタパラメータを自動計算・更新し、ブロック処理を行う
    pub fn process(&mut self, block: &[f32], target_distance_m: f32) -> Vec<f32> {
        // 距離の急変を防ぐスムージング (0.1m 未満にならないようクランプ)
        let d = target_distance_m.max(0.1);
        self.current_distance += (d - self.current_distance) * 0.05;
        let dist = self.current_distance;

        // ----------------------------------------------------
        // 1. 近接効果キャンセル (200Hz 以下の Low-shelf 減衰)
        // ----------------------------------------------------
        // オンマイク(0.05m〜0.2m)ほど低域が高く録られているため、
        // 離れるほど(d -> 離脱)近接効果分をカットしてフラットに戻す。
        // A mastered stereo mix already contains microphone/room coloration.
        // Keep this cue subtle instead of removing 7-9 dB of bass.
        // 0.5m = 0dB, increasing distance approaches at most -2.5dB.
        let proximity_gain_db = if dist > 0.5 {
            (-2.0 * (dist / 0.5).log10()).max(-2.5)
        } else {
            0.0
        };

        let (b_ls, a_ls) = get_low_shelf_coeffs(self.sr, 180.0, proximity_gain_db, 0.707);
        self.low_shelf.b = b_ls;
        self.low_shelf.a = a_ls;

        // ----------------------------------------------------
        // 2. 空気吸収 (6kHz 以上の High-shelf 減衰)
        // ----------------------------------------------------
        // Gentle high-frequency air-absorption cue for a finished stereo mix.
        let air_absorption_gain_db = -0.75 * (dist - 1.0).max(0.0);
        let (b_hs, a_hs) = get_high_shelf_coeffs(
            self.sr,
            6000.0,
            air_absorption_gain_db.max(-6.0), // keep the tonal change moderate
            0.707,
        );
        self.high_shelf.b = b_hs;
        self.high_shelf.a = a_hs;

        // ----------------------------------------------------
        // 3. 距離による音量減衰 (1/d 逆二乗則の近似)
        // ----------------------------------------------------
        let ref_distance = 0.5; // 基準距離 0.5m
        // Perceptual distance cue: gentler than 1/d for already-mastered music.
        // 1.5m => about -4.8dB instead of -9.5dB.
        let distance_gain = (ref_distance / dist).sqrt().min(1.0);

        // フィルタ処理実行
        let low_filtered = self.low_shelf.process(block);
        let fully_filtered = self.high_shelf.process(&low_filtered);

        // 距離音量を乗算して出力
        fully_filtered.iter().map(|&x| x * distance_gain).collect()
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
    left_0: RealTimeOverlapSave,
    right_0: RealTimeOverlapSave,

    // Cross-ear-complete Side virtual sources.
    front_left_to_l: RealTimeOverlapSave,   // 63 -> left ear
    front_left_to_r: RealTimeOverlapSave,   // 63 -> right ear
    front_right_to_l: RealTimeOverlapSave,  // 9 -> left ear
    front_right_to_r: RealTimeOverlapSave,  // 9 -> right ear,

    // Six discrete early reflections. Their delays, gains, wall absorption and
    // direction-specific HRIR pairs are independent. ER ends around 46 ms; the
    // FDN is pre-delayed to start around 55 ms.
    early_reflections: [EarlyReflection; 6],

    // Direct HRTF cues stay full-band. Only the reflected field is darkened.
    side_shelf_l: RealTimeBiquadFilter,
    side_shelf_r: RealTimeBiquadFilter,

    late_reverb: StereoFdnReverb,

    // Small residual headphone crossfeed. HRTF rendering already provides
    // contralateral energy, so this is intentionally subtle.
    cross_delay_l: RealTimeDelayLine,
    cross_delay_r: RealTimeDelayLine,
}

impl SpatialAudioEngine {
    pub fn new(sr: f32, hrtf_data: &HrtfData, block_size: usize) -> Self {
        let (b_sh, a_sh) = get_high_shelf_coeffs(sr, 2500.0, -2.5, 0.707);

        Self {
            analyzer: RoomAcousticAnalyzer::new(),
            // Original code used S * 0.8. Keep that intent explicitly.
            ms: MidSideProcessor::new(0.80),
            crossover: RealTimeCrossover::new(sr, 100.0),
            dist_filter_mid: DistanceFilter::new(sr),
            dist_filter_side: DistanceFilter::new(sr),
            eq_mid: RealTimeActiveEQ::new(sr, 80.0),
            eq_side: RealTimeActiveEQ::new(sr, 80.0),

            // Mid is physically anchored at the front (index 0).
            left_0: RealTimeOverlapSave::new(&hrtf_data.left_0, block_size),
            right_0: RealTimeOverlapSave::new(&hrtf_data.right_0, block_size),

            // Side virtual sources are each rendered to BOTH ears.
            front_left_to_l: RealTimeOverlapSave::new(&hrtf_data.front_left_to_l, block_size),
            front_left_to_r: RealTimeOverlapSave::new(&hrtf_data.front_left_to_r, block_size),
            front_right_to_l: RealTimeOverlapSave::new(&hrtf_data.front_right_to_l, block_size),
            front_right_to_r: RealTimeOverlapSave::new(&hrtf_data.front_right_to_r, block_size),

            // Six first-order-ish horizontal reflection events.
            // Indices follow the existing 64-direction horizontal HRTF set.
            // Earlier paths are brighter/stronger; later paths are darker/weaker.
            early_reflections: [
                EarlyReflection::new(sr, block_size, 11.0, 0.16, 7500.0,  1.0, &hrtf_data.er_left[0], &hrtf_data.er_right[0]), // idx 56
                EarlyReflection::new(sr, block_size, 14.0, 0.15, 7200.0, -1.0, &hrtf_data.er_left[1], &hrtf_data.er_right[1]), // idx 8
                EarlyReflection::new(sr, block_size, 22.0, 0.13, 6200.0,  1.0, &hrtf_data.er_left[2], &hrtf_data.er_right[2]), // idx 48
                EarlyReflection::new(sr, block_size, 27.0, 0.12, 5900.0, -1.0, &hrtf_data.er_left[3], &hrtf_data.er_right[3]), // idx 16
                EarlyReflection::new(sr, block_size, 38.0, 0.10, 4800.0,  1.0, &hrtf_data.er_left[4], &hrtf_data.er_right[4]), // idx 40
                EarlyReflection::new(sr, block_size, 46.0, 0.09, 4500.0, -1.0, &hrtf_data.er_left[5], &hrtf_data.er_right[5]), // idx 32
            ],

            side_shelf_l: RealTimeBiquadFilter::new(b_sh, a_sh),
            side_shelf_r: RealTimeBiquadFilter::new(b_sh, a_sh),
            late_reverb: StereoFdnReverb::new(sr),
            cross_delay_l: RealTimeDelayLine::new(sr, 0.5, 1.0, 1200.0),
            cross_delay_r: RealTimeDelayLine::new(sr, 0.5, 1.0, 1200.0),
        }
    }

    pub fn process_block(
        &mut self,
        input_l: &[f32],
        input_r: &[f32],
        target_distance_m: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = input_l.len().min(input_r.len());
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let input_l = &input_l[..n];
        let input_r = &input_r[..n];

        // 1) Encode once into the actual basis used by the spatial model.
        // M = (L+R)/2, S = (L-R)/2 * width.
        let (raw_mid, raw_side) = self.ms.process(input_l, input_r);

        // Mid and Side describe the same virtual stage, so both receive the
        // same distance law / air absorption, with independent filter state.
        let mid = self.dist_filter_mid.process(&raw_mid, target_distance_m);
        let side = self.dist_filter_side.process(&raw_side, target_distance_m);
        let mid_eq = self.eq_mid.process(&mid);
        let side_eq = self.eq_side.process(&side);

        // Analyze total M/S energy. A pure Side signal must not look like
        // silence simply because Mid happens to be near zero.
        let (mut er_pct, mut reverb_pct) = self.analyzer.analyze_ms(&mid_eq, &side_eq);
        let distance_factor = 1.0 + (target_distance_m - 0.5) * 0.25;
        er_pct = (er_pct * distance_factor).clamp(0.0, 1.0);
        reverb_pct = (reverb_pct * distance_factor).clamp(0.0, 1.0);

        // 2) MID DIRECT: keep very low frequencies centered, spatialize the
        // directional band from the front (index 0).
        let (mid_low, mid_high) = self.crossover.process(&mid_eq);

        let left_0: Vec<f32> = self
            .left_0
            .process(&mid_high)
            .into_iter()
            .map(|x| x * 0.90)
            .collect();
        let right_0: Vec<f32> = self
            .right_0
            .process(&mid_high)
            .into_iter()
            .map(|x| x * 0.90)
            .collect();

        // 3) SIDE DIRECT, cross-ear complete.
        // +S is a source at left-front 63 and -S is a source at right-front 9.
        // Each virtual source must reach BOTH ears through its HRIR pair:
        //   ear L = +S*H63_L - S*H9_L
        //   ear R = +S*H63_R - S*H9_R
        let fl_l = self.front_left_to_l.process(&side_eq);
        let fl_r = self.front_left_to_r.process(&side_eq);
        let fr_l = self.front_right_to_l.process(&side_eq);
        let fr_r = self.front_right_to_r.process(&side_eq);

        let side_l_raw: Vec<f32> = fl_l
            .iter()
            .zip(fr_l.iter())
            .map(|(&a, &b)| (a - b) * 0.55)
            .collect();
        let side_r_raw: Vec<f32> = fl_r
            .iter()
            .zip(fr_r.iter())
            .map(|(&a, &b)| (a - b) * 0.55)
            .collect();

        let side_direct_l = self.side_shelf_l.process(&side_l_raw);
        let side_direct_r = self.side_shelf_r.process(&side_r_raw);

        // 4/5) SIX DISCRETE EARLY REFLECTIONS
        // Each event is delayed, wall-filtered and rendered to BOTH ears using
        // the HRIR pair for that event direction. Summing several modest events
        // avoids the "one huge delayed loudspeaker" character of the old 2-path ER.
        let mut er_l = vec![0.0_f32; n];
        let mut er_r = vec![0.0_f32; n];
        for reflection in self.early_reflections.iter_mut() {
            let (rl, rr) = reflection.process(&mid_eq, &side_eq);
            for i in 0..n {
                er_l[i] += rl[i];
                er_r[i] += rr[i];
            }
        }

        // 6) LATE REVERB: both Mid and Side energize the FDN using orthogonal
        // injection patterns, so stereo-only material still creates a tail.
        let (late_l, late_r) = self.late_reverb.process_ms(&mid_eq, &side_eq);

        // 7) Reconstruct the binaural field from the M/S components.
        let mut rendered_l = Vec::with_capacity(n);
        let mut rendered_r = Vec::with_capacity(n);
        for i in 0..n {
            let direct_l = mid_low[i] + left_0[i] + side_direct_l[i];
            let direct_r = mid_low[i] + right_0[i] + side_direct_r[i];

            rendered_l.push(direct_l + er_l[i] * er_pct + late_l[i] * reverb_pct);
            rendered_r.push(direct_r + er_r[i] * er_pct + late_r[i] * reverb_pct);
        }

        // 8) Very small residual crossfeed only. HRTF processing already
        // creates ear-to-ear cues, so large crossfeed would blur localization.
        let cross_feed_level = 0.08;
        let g_delayed_l = self.cross_delay_l.process(&rendered_l);
        let g_delayed_r = self.cross_delay_r.process(&rendered_r);

        let mut out_left = vec![0.0; n];
        let mut out_right = vec![0.0; n];
        for i in 0..n {
            let mixed_l = rendered_l[i] + g_delayed_r[i] * cross_feed_level;
            let mixed_r = rendered_r[i] + g_delayed_l[i] * cross_feed_level;

            // Keep the DSP linear here. Global peak normalization is done
            // after the full offline render, avoiding tanh coloration.
            out_left[i] = mixed_l * 0.82;
            out_right[i] = mixed_r * 0.82;
        }

        (out_left, out_right)
    }
}

pub struct HrtfData {
    // Original MAT-derived geometry:
    // 0 = front, 63/9 = mirrored front pair, 40/32 = mirrored rear pair.
    pub left_0: Vec<f32>,  // left ear, index 0
    pub right_0: Vec<f32>,  // right ear, index 0
    pub front_left_to_l: Vec<f32>,   // left-front 63 -> left ear
    pub front_left_to_r: Vec<f32>,   // left-front 63 -> right ear
    pub front_right_to_l: Vec<f32>,  // right-front 9 -> left ear
    pub front_right_to_r: Vec<f32>,  // right-front 9 -> right ear
    // Direction-specific HRIR pairs for six early-reflection events.
    // Event direction indices: [56, 8, 48, 16, 40, 32].
    pub er_left: [Vec<f32>; 6],
    pub er_right: [Vec<f32>; 6],
}


pub fn run_audio_pipeline(
    input_l: &[f32],
    input_r: &[f32],
    sr: f32,
    hrtf_data: &HrtfData,
    use_hrtf: bool,
    target_distance_m: f32,
) -> (Vec<f32>, Vec<f32>) {
    const BLOCK_SIZE: usize = 1024;
    let total_samples = input_l.len().min(input_r.len());

    let mut out_left = vec![0.0_f32; total_samples];
    let mut out_right = vec![0.0_f32; total_samples];

    if !use_hrtf {
        out_left.copy_from_slice(&input_l[..total_samples]);
        out_right.copy_from_slice(&input_r[..total_samples]);
        return (out_left, out_right);
    }

    let mut engine = SpatialAudioEngine::new(sr, hrtf_data, BLOCK_SIZE);

    for i in (0..total_samples).step_by(BLOCK_SIZE) {
        let end = usize::min(i + BLOCK_SIZE, total_samples);
        let len = end - i;

        if len == BLOCK_SIZE {
            let (l, r) = engine.process_block(
                &input_l[i..end],
                &input_r[i..end],
                target_distance_m,
            );
            out_left[i..end].copy_from_slice(&l);
            out_right[i..end].copy_from_slice(&r);
        } else {
            // Pad the final partial block so main(3)'s fixed-size overlap-save
            // implementation can process it without dropping the tail.
            let mut pad_l = vec![0.0_f32; BLOCK_SIZE];
            let mut pad_r = vec![0.0_f32; BLOCK_SIZE];
            pad_l[..len].copy_from_slice(&input_l[i..end]);
            pad_r[..len].copy_from_slice(&input_r[i..end]);

            let (l, r) = engine.process_block(&pad_l, &pad_r, target_distance_m);
            out_left[i..end].copy_from_slice(&l[..len]);
            out_right[i..end].copy_from_slice(&r[..len]);
        }
    }

    // Offline-only transparent peak protection. Do not waveshape each sample.
    let peak_l = out_left.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
    let peak_r = out_right.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
    let peak = peak_l.max(peak_r);
    if peak > 0.98 {
        let scale = 0.98 / peak;
        for x in &mut out_left {
            *x *= scale;
        }
        for x in &mut out_right {
            *x *= scale;
        }
    }

    (out_left, out_right)
}

use hound;

fn main() {
    println!("🚀 テスト実行を開始します...");

    // 1. サンプル音源の読み込み (WAV形式に変換済みのものを使用)
    let mut reader = hound::WavReader::open("output.wav")
        .expect("サンプルのWAVファイルが見つかりません。");
    let spec = reader.spec();
    let sr = spec.sample_rate as f32;

    // ステレオ(Interleaved: L, R, L, R...)のi16データを、LとRのf32配列に分離する(-1.0 ~ 1.0に正規化)
    let mut input_l = Vec::new();
    let mut input_r = Vec::new();

    let samples: Vec<i16> = reader.samples().map(|s| s.unwrap()).collect();
    for chunk in samples.chunks(2) {
        if chunk.len() == 2 {
            input_l.push(chunk[0] as f32 / i16::MAX as f32);
            input_r.push(chunk[1] as f32 / i16::MAX as f32);
        }
    }
    println!("✅ 音源ロード完了: {} Hz, {} サンプル", sr, input_l.len());

    // 2. HRTFデータの準備
    // ※ 実際はここで "hrtf_left_9.wav" などのWAVファイルからインパルス応答を読み込みます。
    // 今回はテスト実行を回すため、ダミーの配列（すべて0.0）で構造体を初期化します。
    let hrtf_data = HrtfData {
        // front 0 -> BOTH ears
        left_0: load_impulse_response("hrtf_left_0.wav")
            .unwrap_or(vec![0.0; 512]),
        right_0: load_impulse_response("hrtf_right_0.wav")
            .unwrap_or(vec![0.0; 512]),

        // left-front 63 -> BOTH ears
        front_left_to_l: load_impulse_response("hrtf_left_63.wav")
            .unwrap_or(vec![0.0; 512]),
        front_left_to_r: load_impulse_response("hrtf_right_63.wav")
            .unwrap_or(vec![0.0; 512]),

        // right-front 9 -> BOTH ears
        front_right_to_l: load_impulse_response("hrtf_left_9.wav")
            .unwrap_or(vec![0.0; 512]),
        front_right_to_r: load_impulse_response("hrtf_right_9.wav")
            .unwrap_or(vec![0.0; 512]),
        // Six ER directions -> BOTH ears.
        // [56, 8, 48, 16, 40, 32]
        er_left: [
            load_impulse_response("hrtf_left_56.wav").unwrap_or(vec![0.0; 512]),
            load_impulse_response("hrtf_left_8.wav").unwrap_or(vec![0.0; 512]),
            load_impulse_response("hrtf_left_48.wav").unwrap_or(vec![0.0; 512]),
            load_impulse_response("hrtf_left_16.wav").unwrap_or(vec![0.0; 512]),
            load_impulse_response("hrtf_left_40.wav").unwrap_or(vec![0.0; 512]),
            load_impulse_response("hrtf_left_32.wav").unwrap_or(vec![0.0; 512]),
        ],
        er_right: [
            load_impulse_response("hrtf_right_56.wav").unwrap_or(vec![0.0; 512]),
            load_impulse_response("hrtf_right_8.wav").unwrap_or(vec![0.0; 512]),
            load_impulse_response("hrtf_right_48.wav").unwrap_or(vec![0.0; 512]),
            load_impulse_response("hrtf_right_16.wav").unwrap_or(vec![0.0; 512]),
            load_impulse_response("hrtf_right_40.wav").unwrap_or(vec![0.0; 512]),
            load_impulse_response("hrtf_right_32.wav").unwrap_or(vec![0.0; 512]),
        ],
    };
    println!("✅ HRTFデータ準備完了");

    // 3. 空間オーディオパイプラインの実行
    println!("🎧 空間オーディオ処理を実行中...");
    let target_distance_m: f32 = 1.5;
    let (out_l, out_r) = run_audio_pipeline(
        &input_l,
        &input_r,
        sr,
        &hrtf_data,
        true,
        target_distance_m,
    );

    // 4. 結果をWAVファイルとして書き出し
    let mut writer =
        hound::WavWriter::create("final_rt_sim.wav", spec).expect("ファイルの作成に失敗しました。");

    for i in 0..out_l.len() {
        // f32 (-1.0 ~ 1.0) を i16 に戻す。クリッピング防止でclampする
        let l_sample = (out_l[i].clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        let r_sample = (out_r[i].clamp(-1.0, 1.0) * i16::MAX as f32) as i16;

        writer.write_sample(l_sample).unwrap();
        writer.write_sample(r_sample).unwrap();
    }
    writer
        .finalize()
        .expect("ファイルの書き込みに失敗しました。");

    println!("🎉 処理完了！ 'final_rt_sim.wav' として保存しました。");
}

// （おまけ）WAVファイルからインパルス応答(モノラル)を読み込むヘルパー関数
fn load_impulse_response(filename: &str) -> Option<Vec<f32>> {
    let mut reader = match hound::WavReader::open(filename) {
        Ok(r) => r,
        Err(_) => {
            // 読み込めなかったらターミナルにハッキリとエラーを出す！
            println!("❌ エラー: HRTFファイル '{}' が見つかりません！フォルダの場所あってる？", filename);
            return None;
        }
    };

    let samples: Vec<f32> = reader
        .samples::<f32>()
        .map(|s| s.expect("サンプルの読み込みに失敗しました"))
        .collect();
        
    println!("✅ {} を読み込みました！(長さ: {} サンプル)", filename, samples.len());
    Some(samples)
}
