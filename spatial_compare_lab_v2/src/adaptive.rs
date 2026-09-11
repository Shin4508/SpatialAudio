pub struct AdaptiveParams {
    pub early_gain: f32,
    pub late_send: f32,
    pub dry_anchor: f32,
}

// Keep adaptation close to the preferred fixed-room (06) rendering.
const MIN_EARLY: f32 = 0.94;
const MIN_LATE: f32 = 0.85;
const SMOOTH_SECONDS: f32 = 0.25;
const SILENCE_MEAN_SQUARE: f32 = 1e-8; // -80 dBFS RMS across both channels

pub struct AdaptiveController {
    early: f32,
    late: f32,
    sample_rate: f32,
}

impl AdaptiveController {
    pub fn new(sample_rate: f32) -> Self {
        assert!(sample_rate.is_finite() && sample_rate > 0.0);
        Self {
            early: 1.0,
            late: 1.0,
            sample_rate,
        }
    }

    pub fn analyze(&mut self, left: &[f32], right: &[f32]) -> AdaptiveParams {
        let n = left.len().min(right.len());
        let mut e_l = 0.0_f32;
        let mut e_r = 0.0_f32;
        let mut cross = 0.0_f32;
        let mut e_side = 0.0_f32;
        for (&l, &r) in left.iter().zip(right) {
            e_l += l * l;
            e_r += r * r;
            cross += l * r;
            e_side += (0.5 * (l - r)).powi(2);
        }

        let total = e_l + e_r;
        // Hold the room balance through silence and the zero-padded reverb tail.
        if n > 0 && total / (2.0 * n as f32) > SILENCE_MEAN_SQUARE {
            let geometric_energy = (e_l * e_r).sqrt();
            let corr = if geometric_energy > 0.0 {
                (cross / geometric_energy).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let balance = (2.0 * geometric_energy / total).clamp(0.0, 1.0);
            let side_ratio = (2.0 * e_side / total).clamp(0.0, 1.0);
            // Positive, balanced, mid-dominant content gets a small reduction.
            // Anti-phase or hard-panned content is not classified as a center.
            let centered = corr.max(0.0) * balance * (1.0 - side_ratio);
            let target_early = 1.0 - (1.0 - MIN_EARLY) * centered;
            let target_late = 1.0 - (1.0 - MIN_LATE) * centered;
            let smooth = 1.0 - (-(n as f32) / (self.sample_rate * SMOOTH_SECONDS)).exp();
            self.early += (target_early - self.early) * smooth;
            self.late += (target_late - self.late) * smooth;
        }

        AdaptiveParams {
            early_gain: self.early.clamp(MIN_EARLY, 1.0),
            late_send: self.late.clamp(MIN_LATE, 1.0),
            // Preserve the binaural direct path instead of blending headphone stereo.
            dry_anchor: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_content_only_gently_reduces_the_room() {
        let mut c = AdaptiveController::new(48000.0);
        let signal = vec![0.25; 256];
        let mut p = c.analyze(&signal, &signal);
        for _ in 0..1000 {
            p = c.analyze(&signal, &signal);
        }
        assert!((p.early_gain - MIN_EARLY).abs() < 1e-5);
        assert!((p.late_send - MIN_LATE).abs() < 1e-5);
        assert_eq!(p.dry_anchor, 0.0);
    }

    #[test]
    fn anti_phase_hard_pan_and_uncorrelated_content_keep_fixed_room() {
        for (l, r) in [
            (vec![0.25; 256], vec![-0.25; 256]),
            (vec![0.25; 256], vec![0.0; 256]),
            (vec![0.0; 256], vec![0.25; 256]),
            (
                vec![0.25; 256],
                (0..256)
                    .map(|i| if i % 2 == 0 { 0.25 } else { -0.25 })
                    .collect(),
            ),
        ] {
            let mut c = AdaptiveController::new(48000.0);
            for _ in 0..100 {
                let p = c.analyze(&l, &r);
                assert_eq!(p.early_gain, 1.0);
                assert_eq!(p.late_send, 1.0);
                assert_eq!(p.dry_anchor, 0.0);
            }
        }
    }

    #[test]
    fn silence_and_empty_input_hold_tail_balance() {
        let mut c = AdaptiveController::new(48000.0);
        let p = c.analyze(&[0.25; 256], &[0.25; 256]);
        for (l, r) in [(vec![0.0; 256], vec![0.0; 256]), (vec![], vec![])] {
            let held = c.analyze(&l, &r);
            assert_eq!(p.early_gain, held.early_gain);
            assert_eq!(p.late_send, held.late_send);
        }
    }

    #[test]
    fn smoothing_tracks_elapsed_time_across_rates_and_block_sizes() {
        let mut values = Vec::new();
        for (sr, block, count) in [
            (48000.0, 240, 200),
            (48000.0, 480, 100),
            (96000.0, 480, 200),
        ] {
            let mut c = AdaptiveController::new(sr);
            let x = vec![0.25; block];
            for _ in 0..count {
                c.analyze(&x, &x);
            }
            values.push(c.late);
        }
        for value in values {
            let expected = MIN_LATE + (1.0 - MIN_LATE) * (-4.0_f32).exp();
            assert!((value - expected).abs() < 1e-5);
        }
    }
}
