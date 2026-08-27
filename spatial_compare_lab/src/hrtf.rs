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

    pub fn hrir(&self, azimuth_deg: f32, intensity: f32) -> Hrir {
        let idx = azimuth_to_index(azimuth_deg);
        let src = &self.dirs[idx];
        Hrir {
            left: apply_intensity(&src.left, intensity),
            right: apply_intensity(&src.right, intensity),
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

// HRTF Intensity:
// intensity=1 -> original HRIR.
// intensity=0 -> energy-matched delayed impulse at each ear's HRIR peak.
// This preserves the main ITD/ILD anchor while reducing spectral coloration.
fn apply_intensity(ir: &[f32], intensity: f32) -> Vec<f32> {
    if ir.is_empty() {
        return vec![0.0];
    }

    let a = intensity.clamp(0.0, 1.0);
    if a >= 0.9999 {
        return ir.to_vec();
    }

    let mut peak_idx = 0usize;
    let mut peak_abs = 0.0_f32;
    let mut energy = 0.0_f32;

    for (i, &x) in ir.iter().enumerate() {
        energy += x*x;
        if x.abs() > peak_abs {
            peak_abs = x.abs();
            peak_idx = i;
        }
    }

    let sign = if ir[peak_idx] < 0.0 { -1.0 } else { 1.0 };
    let impulse_amp = energy.sqrt() * sign;
    let mut out = vec![0.0_f32; ir.len()];

    for i in 0..ir.len() {
        let base = if i == peak_idx { impulse_amp } else { 0.0 };
        out[i] = (1.0 - a) * base + a * ir[i];
    }
    out
}
