use crate::room::LateParams;
use std::f32::consts::PI;

struct Delay {
    buffer: Vec<f32>,
    idx: usize,
    damp: f32,
}

impl Delay {
    fn new(n: usize) -> Self {
        Self {
            buffer: vec![0.0; n.max(1)],
            idx: 0,
            damp: 0.0,
        }
    }

    fn read_damped(&mut self, alpha: f32) -> f32 {
        let x = self.buffer[self.idx];
        self.damp += alpha * (x - self.damp);
        self.damp
    }

    fn write_advance(&mut self, x: f32) {
        self.buffer[self.idx] = x;
        self.idx += 1;
        if self.idx >= self.buffer.len() {
            self.idx = 0;
        }
    }
}

pub struct StereoFdn {
    delays: [Delay; 4],
    feedback: [f32; 4],
    damping_alpha: f32,
    predelay: Vec<f32>,
    predelay_idx: usize,
    output_gain: f32,
    enabled: bool,
}

impl StereoFdn {
    pub fn new(sr: f32, p: LateParams) -> Self {
        let d_ms = [29.7_f32, 37.1, 41.1, 43.7];
        let delays = d_ms.map(|ms| Delay::new((sr * ms / 1000.0).round() as usize));

        let feedback = d_ms.map(|ms| {
            if !p.enabled || p.rt60_s <= 0.01 {
                0.0
            } else {
                let ds = ms / 1000.0;
                10.0_f32.powf(-3.0 * ds / p.rt60_s).clamp(0.0, 0.985)
            }
        });

        let damping_alpha = 1.0 - (-2.0 * PI * p.damping_cutoff_hz.max(500.0) / sr).exp();

        let predelay_samples = (sr * p.predelay_ms.max(0.0) / 1000.0).round() as usize;

        Self {
            delays,
            feedback,
            damping_alpha,
            predelay: vec![0.0; predelay_samples.max(1)],
            predelay_idx: 0,
            output_gain: p.output_gain,
            enabled: p.enabled,
        }
    }

    pub fn process_block(
        &mut self,
        left: &[f32],
        right: &[f32],
        send: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = left.len().min(right.len());
        if !self.enabled || send <= 0.0 {
            return (vec![0.0; n], vec![0.0; n]);
        }

        let mut out_l = vec![0.0_f32; n];
        let mut out_r = vec![0.0_f32; n];
        let mid_pattern = [1.0_f32, 1.0, 1.0, 1.0];
        let side_pattern = [1.0_f32, -1.0, -1.0, 1.0];

        for i in 0..n {
            let m = 0.5 * (left[i] + right[i]);
            let s = 0.5 * (left[i] - right[i]);

            let in_now = (0.18 * m + 0.10 * s) * send;
            let delayed_in = self.predelay[self.predelay_idx];
            self.predelay[self.predelay_idx] = in_now;
            self.predelay_idx += 1;
            if self.predelay_idx >= self.predelay.len() {
                self.predelay_idx = 0;
            }

            let mut d = [0.0_f32; 4];
            for k in 0..4 {
                d[k] = self.delays[k].read_damped(self.damping_alpha);
            }

            let mixed = [
                (d[0] + d[1] + d[2] + d[3]) * 0.5,
                (d[0] - d[1] + d[2] - d[3]) * 0.5,
                (d[0] + d[1] - d[2] - d[3]) * 0.5,
                (d[0] - d[1] - d[2] + d[3]) * 0.5,
            ];

            for k in 0..4 {
                let inject = delayed_in * (0.75 * mid_pattern[k] + 0.25 * side_pattern[k]);
                self.delays[k].write_advance(inject + self.feedback[k] * mixed[k]);
            }

            out_l[i] = (d[0] + d[2]) * 0.70710678 * self.output_gain;
            out_r[i] = (d[1] + d[3]) * 0.70710678 * self.output_gain;
        }

        (out_l, out_r)
    }
}
