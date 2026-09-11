pub mod adaptive;
pub mod audio;
pub mod fdn;
pub mod filters;
pub mod headphone_eq;
pub mod hrtf;
pub mod room;

use adaptive::AdaptiveController;
use audio::write_stereo_wav;
use fdn::StereoFdn;
use filters::FftConvolver;
use headphone_eq::HeadphoneEqConfig;
use hrtf::HrtfProfile;
use room::{BakedSpeaker, EnvironmentKind, RoomBaker, RoomConfig};
use std::error::Error;

const BLOCK: usize = 256;

#[derive(Clone, Copy)]
pub struct RenderOptions {
    pub direct_hrtf_intensity: f32,
    pub environment: Option<EnvironmentKind>,
    pub headphone_eq: bool,
    pub adaptive: bool,
    pub distance_m: Option<f32>,
    pub angle_deg: f32,
}

pub fn ensure_rate(sr: u32, profile: &HrtfProfile) -> Result<(), Box<dyn Error>> {
    if sr != profile.sample_rate {
        return Err(format!(
            "Sample-rate mismatch: input={} Hz, HRTF={} Hz.",
            sr, profile.sample_rate
        )
        .into());
    }
    Ok(())
}

pub fn render_and_write(
    path: &str,
    input_l: &[f32],
    input_r: &[f32],
    sr: u32,
    profile: &HrtfProfile,
    eq_cfg: &HeadphoneEqConfig,
    opt: RenderOptions,
) -> Result<String, Box<dyn Error>> {
    render_with_config(path, input_l, input_r, sr, profile, eq_cfg, opt, None)
}

pub fn render_with_config(
    path: &str,
    input_l: &[f32],
    input_r: &[f32],
    sr: u32,
    profile: &HrtfProfile,
    eq_cfg: &HeadphoneEqConfig,
    opt: RenderOptions,
    config: Option<RoomConfig>,
) -> Result<String, Box<dyn Error>> {
    println!("\nRendering {}", path);

    let mut cfg = config.unwrap_or_else(|| match opt.environment {
        Some(env) => RoomConfig::preset(env),
        None => RoomConfig::preset(EnvironmentKind::OpenAir),
    });
    if let Some(distance_m) = opt.distance_m {
        cfg = cfg.with_distance(distance_m);
    }

    cfg.left_speaker_az_deg = -opt.angle_deg;
    cfg.right_speaker_az_deg = opt.angle_deg;
    cfg.validate()?;
    let left_az = cfg.left_speaker_az_deg;
    let right_az = cfg.right_speaker_az_deg;
    let baker = RoomBaker::new(sr, cfg);
    let distance = baker.distance_cues();

    let left_spk = baker.bake_speaker(profile, left_az, opt.direct_hrtf_intensity);
    let right_spk = baker.bake_speaker(profile, right_az, opt.direct_hrtf_intensity);

    if let Some(env) = opt.environment {
        println!("Environment: {}", env.label());
        print_events("L speaker", &left_spk, sr);
        print_events("R speaker", &right_spk, sr);

        let late = baker.late_params();
        println!(
            "Distance: {:.2}m direct={:.3} early={:.3} late_send={:.3}\nLate: on={} RT60={:.2}s HF_RT60={:.2}s pre={:.1}ms damp={:.0}Hz gain={:.3}",
            distance.distance_m,
            distance.direct_gain,
            distance.early_gain,
            distance.late_send,
            late.enabled,
            late.rt60_s,
            late.rt60_high_s,
            late.predelay_ms,
            late.damping_cutoff_hz,
            late.output_gain,
        );
    }

    let mut d_ll = FftConvolver::new(&left_spk.direct_l, BLOCK);
    let mut d_lr = FftConvolver::new(&left_spk.direct_r, BLOCK);
    let mut d_rl = FftConvolver::new(&right_spk.direct_l, BLOCK);
    let mut d_rr = FftConvolver::new(&right_spk.direct_r, BLOCK);

    let mut e_ll = FftConvolver::new(&left_spk.early_l, BLOCK);
    let mut e_lr = FftConvolver::new(&left_spk.early_r, BLOCK);
    let mut e_rl = FftConvolver::new(&right_spk.early_l, BLOCK);
    let mut e_rr = FftConvolver::new(&right_spk.early_r, BLOCK);

    let late_params = baker.late_params();
    let mut fdn = StereoFdn::new(sr as f32, late_params);
    let mut adaptive = AdaptiveController::new(sr as f32);
    let mut eq = eq_cfg.build(sr as f32);

    let input_n = input_l.len().min(input_r.len());
    let fir_tail = [
        &left_spk.direct_l,
        &left_spk.direct_r,
        &right_spk.direct_l,
        &right_spk.direct_r,
        &left_spk.early_l,
        &left_spk.early_r,
        &right_spk.early_l,
        &right_spk.early_r,
    ]
    .iter()
    .map(|h| h.len())
    .max()
    .unwrap_or(1)
        - 1;
    let late_tail = if opt.environment.is_some() && late_params.enabled {
        ((late_params.predelay_ms / 1000.0 + 0.044 + 1.5 * late_params.rt60_s) * sr as f32).ceil()
            as usize
    } else {
        0
    };
    let n = input_n + fir_tail.max(late_tail);
    let mut out_l = vec![0.0_f32; n];
    let mut out_r = vec![0.0_f32; n];

    for base in (0..n).step_by(BLOCK) {
        let len = (n - base).min(BLOCK);
        let mut bl = vec![0.0_f32; BLOCK];
        let mut br = vec![0.0_f32; BLOCK];
        let available = input_n.saturating_sub(base).min(BLOCK);
        if available > 0 {
            bl[..available].copy_from_slice(&input_l[base..base + available]);
            br[..available].copy_from_slice(&input_r[base..base + available]);
        }

        let x_ll = d_ll.process(&bl);
        let x_lr = d_lr.process(&bl);
        let x_rl = d_rl.process(&br);
        let x_rr = d_rr.process(&br);

        let mut block_l = vec![0.0_f32; BLOCK];
        let mut block_r = vec![0.0_f32; BLOCK];

        for i in 0..BLOCK {
            block_l[i] = (x_ll[i] + x_rl[i]) * distance.direct_gain;
            block_r[i] = (x_lr[i] + x_rr[i]) * distance.direct_gain;
        }

        let has_room = opt.environment.is_some();

        let params = if opt.adaptive {
            adaptive.analyze(&bl, &br)
        } else {
            adaptive::AdaptiveParams {
                early_gain: if has_room { 1.0 } else { 0.0 },
                late_send: if has_room && late_params.enabled {
                    1.0
                } else {
                    0.0
                },
                dry_anchor: 0.0,
            }
        };

        if has_room {
            let y_ll = e_ll.process(&bl);
            let y_lr = e_lr.process(&bl);
            let y_rl = e_rl.process(&br);
            let y_rr = e_rr.process(&br);

            let fdn_send = (if opt.adaptive { params.late_send } else { 1.0 }) * distance.late_send;
            let (late_l, late_r) = fdn.process_block(&bl, &br, fdn_send);

            for i in 0..BLOCK {
                let er_gain = params.early_gain * distance.early_gain;
                block_l[i] += (y_ll[i] + y_rl[i]) * er_gain + late_l[i];
                block_r[i] += (y_lr[i] + y_rr[i]) * er_gain + late_r[i];
            }
        }

        if opt.adaptive && params.dry_anchor > 0.0 {
            for i in 0..BLOCK {
                let wet = 1.0 - params.dry_anchor;
                block_l[i] = block_l[i] * wet + bl[i] * params.dry_anchor;
                block_r[i] = block_r[i] * wet + br[i] * params.dry_anchor;
            }
        }

        if opt.headphone_eq {
            eq.process(&mut block_l, &mut block_r);
        }

        out_l[base..base + len].copy_from_slice(&block_l[..len]);
        out_r[base..base + len].copy_from_slice(&block_r[..len]);
    }

    write_stereo_wav(path, &out_l, &out_r, sr)?;
    Ok(path.to_string())
}

fn print_events(name: &str, speaker: &BakedSpeaker, sr: u32) {
    println!("{} ER events:", name);

    for (i, e) in speaker.events.iter().enumerate() {
        let ms = e.delay_samples as f32 * 1000.0 / sr as f32;
        println!(
            "  {:>2}: az={:>6.1} delay={:>5.1}ms gain={:.3} refl={:.2}/{:.2}/{:.2}",
            i + 1,
            e.azimuth_deg,
            ms,
            e.gain,
            e.reflection_low,
            e.reflection_mid,
            e.reflection_high
        );
    }
}
