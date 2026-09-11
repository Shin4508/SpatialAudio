use spatial_compare_lab_v2::hrtf::HrtfProfile;
use std::error::Error;

/// Resample *impulse responses*, not program audio. Windowed-sinc low-pass
/// interpolation preserves time in seconds and scales coefficients by the
/// inverse rate ratio to preserve filter DC gain. Runs before JACK activation.
pub fn at_rate(mut profile: HrtfProfile, rate: u32) -> Result<HrtfProfile, Box<dyn Error>> {
    if ![44100, 48000].contains(&rate) || ![44100, 48000].contains(&profile.sample_rate) {
        return Err("v3 supports 44100 and 48000 Hz profiles/server rates".into());
    }
    for h in &profile.dirs {
        if h.left.is_empty()
            || h.right.is_empty()
            || h.left.iter().chain(&h.right).any(|x| !x.is_finite())
        {
            return Err("HRTF contains empty or non-finite samples".into());
        }
    }
    if rate != profile.sample_rate {
        for h in &mut profile.dirs {
            h.left = resample(&h.left, profile.sample_rate, rate);
            h.right = resample(&h.right, profile.sample_rate, rate);
        }
        profile.sample_rate = rate;
    }
    Ok(profile)
}

fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    let ratio = to as f64 / from as f64;
    let cutoff = ratio.min(1.0);
    let radius = 48.0;
    // Extend the right edge for the interpolation kernel's tail. Negative-time
    // lobes are truncated to keep the FIR causal, without adding a delay.
    let len = ((input.len() as f64 + radius) * ratio).ceil() as usize;
    (0..len)
        .map(|i| {
            let t = i as f64 / ratio;
            let first = ((t - radius).ceil() as isize).max(0) as usize;
            let last = ((t + radius).floor() as usize + 1).min(input.len());
            let mut sum = 0.0;
            for (k, &x) in input.iter().enumerate().take(last).skip(first) {
                let delta = t - k as f64;
                let phase = std::f64::consts::PI * cutoff * delta;
                let sinc = if phase.abs() < 1e-10 {
                    1.0
                } else {
                    phase.sin() / phase
                };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * delta / radius).cos();
                sum += x as f64 * cutoff * sinc * window;
            }
            (sum / ratio) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_interior_impulse_gain_and_arrival_time() {
        let mut h = vec![0.0; 200];
        h[80] = 1.0;
        let r = resample(&h, 48000, 44100);
        let peak = r
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        assert!((peak as f64 / 44100.0 - 80.0 / 48000.0).abs() < 1.0 / 44100.0);
        assert!((r.iter().sum::<f32>() - 1.0).abs() < 0.002);
        assert!(r.iter().all(|x| x.is_finite()));
    }
}
