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

    fn read_components(&mut self, alpha: f32) -> (f32, f32) {
        let x = self.buffer[self.idx];
        self.damp += alpha * (x - self.damp);
        (x, self.damp)
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
    feedback_mid: [f32; 4],
    feedback_high: [f32; 4],
    damping_alpha: f32,
    predelay: Vec<[f32; 2]>,
    predelay_idx: usize,
    output_gain: f32,
    enabled: bool,
}

impl StereoFdn {
    pub fn new(sr: f32, p: LateParams) -> Self {
        let d_ms = [29.7_f32, 37.1, 41.1, 43.7];
        let delays = d_ms.map(|ms| Delay::new((sr * ms / 1000.0).round() as usize));

        let feedback_mid = d_ms.map(|ms| {
            if !p.enabled || p.rt60_s <= 0.01 {
                0.0
            } else {
                let ds = ms / 1000.0;
                10.0_f32.powf(-3.0 * ds / p.rt60_s).clamp(0.0, 0.985)
            }
        });

        let feedback_high = d_ms.map(|ms| {
            if !p.enabled || p.rt60_high_s <= 0.01 {
                0.0
            } else {
                let ds = ms / 1000.0;
                10.0_f32.powf(-3.0 * ds / p.rt60_high_s).clamp(0.0, 0.985)
            }
        });

        let damping_alpha = 1.0 - (-2.0 * PI * p.damping_cutoff_hz.max(500.0) / sr).exp();

        let predelay_samples = (sr * p.predelay_ms.max(0.0) / 1000.0).round() as usize;

        Self {
            delays,
            feedback_mid,
            feedback_high,
            damping_alpha,
            predelay: vec![[0.0; 2]; predelay_samples],
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
        let mut out_l = vec![0.0_f32; n];
        let mut out_r = vec![0.0_f32; n];
        self.process_into(&left[..n], &right[..n], send, &mut out_l, &mut out_r);
        (out_l, out_r)
    }

    /// Clear delay state without allocating, for restarting an inactive room.
    pub fn reset(&mut self) {
        for d in &mut self.delays {
            d.buffer.fill(0.0);
            d.idx = 0;
            d.damp = 0.0;
        }
        self.predelay.fill([0.0; 2]);
        self.predelay_idx = 0;
    }

    /// Identical DSP to `process_block`, using caller-owned output buffers.
    pub fn process_into(
        &mut self,
        left: &[f32],
        right: &[f32],
        send: f32,
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) {
        let n = left.len();
        assert_eq!(right.len(), n);
        assert_eq!(out_l.len(), n);
        assert_eq!(out_r.len(), n);
        if !self.enabled {
            out_l.fill(0.0);
            out_r.fill(0.0);
            return;
        }
        let mid_pattern = [1.0_f32, 1.0, 1.0, 1.0];
        let side_pattern = [1.0_f32, -1.0, -1.0, 1.0];

        for i in 0..n {
            let m = 0.5 * (left[i] + right[i]);
            let s = 0.5 * (left[i] - right[i]);

            let in_now = [0.18 * m * send.max(0.0), 0.10 * s * send.max(0.0)];
            let delayed_in = if self.predelay.is_empty() {
                in_now
            } else {
                let delayed = self.predelay[self.predelay_idx];
                self.predelay[self.predelay_idx] = in_now;
                self.predelay_idx = (self.predelay_idx + 1) % self.predelay.len();
                delayed
            };

            let mut d = [0.0_f32; 4];
            let mut d_low = [0.0_f32; 4];
            for k in 0..4 {
                (d[k], d_low[k]) = self.delays[k].read_components(self.damping_alpha);
            }

            let mix = |v: [f32; 4]| {
                [
                    (v[0] + v[1] + v[2] + v[3]) * 0.5,
                    (v[0] - v[1] + v[2] - v[3]) * 0.5,
                    (v[0] + v[1] - v[2] - v[3]) * 0.5,
                    (v[0] - v[1] - v[2] + v[3]) * 0.5,
                ]
            };
            let mixed = mix(d);
            let mixed_low = mix(d_low);
            for k in 0..4 {
                let inject =
                    0.75 * delayed_in[0] * mid_pattern[k] + 0.25 * delayed_in[1] * side_pattern[k];
                // Low/mid and high bands now decay with their own RT60 values.
                let feedback = self.feedback_high[k] * mixed[k]
                    + (self.feedback_mid[k] - self.feedback_high[k]) * mixed_low[k];
                self.delays[k].write_advance(inject + feedback);
            }

            out_l[i] = (d[0] + d[2]) * 0.70710678 * self.output_gain;
            out_r[i] = (d[1] + d[3]) * 0.70710678 * self.output_gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn params() -> LateParams {
        LateParams {
            enabled: true,
            rt60_s: 0.3,
            rt60_high_s: 0.2,
            predelay_ms: 0.0,
            damping_cutoff_hz: 5000.0,
            output_gain: 1.0,
        }
    }
    #[test]
    fn zero_send_keeps_existing_tail_running() {
        let mut fdn = StereoFdn::new(48000.0, params());
        fdn.process_block(&[1.0], &[1.0], 1.0);
        let zeros = vec![0.0; 48000];
        let (l, r) = fdn.process_block(&zeros, &zeros, 0.0);
        assert!(l.iter().chain(&r).any(|x| x.abs() > 0.001));
        assert!(l.iter().chain(&r).all(|x| x.is_finite()));
        let first: f32 = l[..12000].iter().chain(&r[..12000]).map(|x| x * x).sum();
        let last: f32 = l[36000..].iter().chain(&r[36000..]).map(|x| x * x).sum();
        assert!(last < first * 1e-4);
    }
    #[test]
    fn stereo_input_has_no_old_rank_one_null() {
        // Old 0.18*M + 0.10*S cancels when L=2 and R=-7.
        let mut fdn = StereoFdn::new(48000.0, params());
        let mut l = vec![0.0; 5000];
        let mut r = l.clone();
        l[0] = 2.0;
        r[0] = -7.0;
        let (a, b) = fdn.process_block(&l, &r, 1.0);
        assert!(a.iter().chain(&b).map(|x| x * x).sum::<f32>() > 0.001);
        assert_eq!(a.iter().position(|x| x.abs() > 1e-6), Some(1426));
    }
}
