use crate::audio::load_mono_wav;
use std::error::Error;
use std::path::Path;

#[derive(Clone)]
pub struct Hrir {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

#[derive(Clone)]
pub struct HrtfProfile {
    pub sample_rate: u32,
    pub dirs: Vec<Hrir>,
}

impl HrtfProfile {
    pub fn load(folder: &str) -> Result<Self, Box<dyn Error>> {
        let mut dirs = Vec::with_capacity(72);
        let mut profile_sr: Option<u32> = None;

        for idx in 0..72 {
            let l = Path::new(folder).join(format!("hrtf_left_{}.wav", idx));
            let r = Path::new(folder).join(format!("hrtf_right_{}.wav", idx));

            let (left, sr_l) = load_mono_wav(l.to_str().ok_or("invalid path")?)?;
            let (right, sr_r) = load_mono_wav(r.to_str().ok_or("invalid path")?)?;

            if sr_l != sr_r {
                return Err(format!("HRTF sample-rate mismatch at index {}", idx).into());
            }
            if let Some(sr) = profile_sr {
                if sr != sr_l {
                    return Err(format!("HRTF profile contains mixed sample rates").into());
                }
            } else {
                profile_sr = Some(sr_l);
            }

            dirs.push(Hrir { left, right });
        }

        Ok(Self {
            sample_rate: profile_sr.unwrap_or(48_000),
            dirs,
        })
    }

    /// Return the measured HRIR without spectral blending or reshaping.
    pub fn hrir(&self, azimuth_deg: f32) -> Hrir {
        let idx = azimuth_to_index(azimuth_deg);
        self.dirs[idx].clone()
    }

    /// Blend only the explicit HRTF-intensity comparison toward an
    /// energy-matched impulse while preserving its principal ITD/ILD cue.
    pub fn hrir_with_intensity(&self, azimuth_deg: f32, intensity: f32) -> Hrir {
        let measured = self.hrir(azimuth_deg);
        Hrir {
            left: apply_intensity(&measured.left, intensity),
            right: apply_intensity(&measured.right, intensity),
        }
    }
}

// Dataset convention used by the user's 72-direction set:
// index 0 = front, +5 degrees per index toward the right.
// Therefore -30 deg -> 66 and +30 deg -> 6.
pub fn azimuth_to_index(azimuth_deg: f32) -> usize {
    let mut d = azimuth_deg % 360.0;
    if d < 0.0 {
        d += 360.0;
    }
    ((d / 5.0).round() as usize) % 72
}

fn apply_intensity(ir: &[f32], intensity: f32) -> Vec<f32> {
    if ir.is_empty() {
        return vec![0.0];
    }

    let amount = intensity.clamp(0.0, 1.0);
    if amount >= 0.9999 {
        return ir.to_vec();
    }

    let mut peak_index = 0;
    let mut peak_magnitude = 0.0_f32;
    let mut energy = 0.0_f32;
    for (index, &sample) in ir.iter().enumerate() {
        energy += sample * sample;
        if sample.abs() > peak_magnitude {
            peak_magnitude = sample.abs();
            peak_index = index;
        }
    }

    let sign = if ir[peak_index] < 0.0 { -1.0 } else { 1.0 };
    let impulse = energy.sqrt() * sign;
    ir.iter()
        .enumerate()
        .map(|(index, &sample)| {
            let baseline = if index == peak_index { impulse } else { 0.0 };
            (1.0 - amount) * baseline + amount * sample
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Hrir, HrtfProfile};

    #[test]
    fn hrir_is_returned_sample_for_sample() {
        let measured = Hrir {
            left: vec![0.25, -0.5, 0.125],
            right: vec![-0.75, 0.375, 0.0],
        };
        let profile = HrtfProfile {
            sample_rate: 48_000,
            dirs: vec![measured.clone(); 72],
        };

        let selected = profile.hrir(30.0);
        assert_eq!(selected.left, measured.left);
        assert_eq!(selected.right, measured.right);
    }

    #[test]
    fn explicit_intensity_comparison_changes_the_hrir() {
        let measured = Hrir {
            left: vec![0.25, -0.5, 0.125],
            right: vec![-0.75, 0.375, 0.0],
        };
        let profile = HrtfProfile {
            sample_rate: 48_000,
            dirs: vec![measured.clone(); 72],
        };

        let selected = profile.hrir_with_intensity(30.0, 0.90);
        assert_ne!(selected.left, measured.left);
        assert_ne!(selected.right, measured.right);
        assert_eq!(selected.left.len(), measured.left.len());
        assert_eq!(selected.right.len(), measured.right.len());
    }
}

#[cfg(test)]
mod mapping_tests {
    #[test]
    fn reference_angles_and_wraparound() {
        for (angle, index) in [
            (-30.0, 66),
            (30.0, 6),
            (0.0, 0),
            (360.0, 0),
            (-5.0, 71),
            (27.0, 5),
            (28.0, 6),
        ] {
            assert_eq!(super::azimuth_to_index(angle), index);
        }
    }
}
