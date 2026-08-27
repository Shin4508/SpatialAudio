mod adaptive;
mod audio;
mod fdn;
mod filters;
mod headphone_eq;
mod hrtf;
mod room;

use adaptive::AdaptiveController;
use audio::{load_stereo_wav, prevent_clipping, write_stereo_wav};
use fdn::StereoFdn;
use filters::FftConvolver;
use headphone_eq::HeadphoneEqConfig;
use hrtf::HrtfProfile;
use room::{BakedSpeaker, EnvironmentKind, RoomBaker, RoomConfig};
use std::error::Error;
use std::fs;
use std::path::Path;

const BLOCK: usize = 256;

#[derive(Clone, Copy)]
struct RenderOptions {
    hrtf_intensity: f32,
    environment: Option<EnvironmentKind>,
    headphone_eq: bool,
    adaptive: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input_path = args.get(1).map(String::as_str).unwrap_or("output.wav");
    let default_profile_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("hrtf_profiles/default");
    let personal_profile_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("hrtf_profiles/personal");
    let eq_path = args
        .get(4)
        .map(String::as_str)
        .unwrap_or("config/headphone_eq.txt");

    fs::create_dir_all("compare_out")?;

    let (input_l, input_r, sr) = load_stereo_wav(input_path)?;
    let default_profile = HrtfProfile::load(default_profile_path)?;
    ensure_rate(sr, &default_profile)?;
    let eq_cfg = HeadphoneEqConfig::load_or_flat(eq_path)?;

    write_stereo_wav("compare_out/00_dry.wav", &input_l, &input_r, sr)?;

    render_and_write(
        "compare_out/01_reference_2speaker.wav",
        &input_l,
        &input_r,
        sr,
        &default_profile,
        &eq_cfg,
        RenderOptions {
            hrtf_intensity: 1.0,
            environment: None,
            headphone_eq: false,
            adaptive: false,
        },
    )?;

    render_and_write(
        "compare_out/02_hrtf_intensity_072.wav",
        &input_l,
        &input_r,
        sr,
        &default_profile,
        &eq_cfg,
        RenderOptions {
            hrtf_intensity: 0.72,
            environment: None,
            headphone_eq: false,
            adaptive: false,
        },
    )?;

    render_and_write(
        "compare_out/05_headphone_comp.wav",
        &input_l,
        &input_r,
        sr,
        &default_profile,
        &eq_cfg,
        RenderOptions {
            hrtf_intensity: 0.72,
            environment: None,
            headphone_eq: true,
            adaptive: false,
        },
    )?;

    for (i, env) in EnvironmentKind::all().into_iter().enumerate() {
        let path = format!(
            "compare_out/06{}_{}.wav",
            (b'a' + i as u8) as char,
            env.slug()
        );

        render_and_write(
            &path,
            &input_l,
            &input_r,
            sr,
            &default_profile,
            &eq_cfg,
            RenderOptions {
                hrtf_intensity: 0.72,
                environment: Some(env),
                headphone_eq: true,
                adaptive: false,
            },
        )?;
    }

    let personal_marker = Path::new(personal_profile_path).join("hrtf_left_0.wav");
    let mut profile_for_adaptive = default_profile.clone();

    if personal_marker.exists() {
        let personal = HrtfProfile::load(personal_profile_path)?;
        ensure_rate(sr, &personal)?;

        render_and_write(
            "compare_out/07_personal_jazz_club.wav",
            &input_l,
            &input_r,
            sr,
            &personal,
            &eq_cfg,
            RenderOptions {
                hrtf_intensity: 0.72,
                environment: Some(EnvironmentKind::JazzClub),
                headphone_eq: true,
                adaptive: false,
            },
        )?;

        profile_for_adaptive = personal;
    }

    for (i, env) in EnvironmentKind::all().into_iter().enumerate() {
        let path = format!(
            "compare_out/08{}_{}_adaptive.wav",
            (b'a' + i as u8) as char,
            env.slug()
        );

        render_and_write(
            &path,
            &input_l,
            &input_r,
            sr,
            &profile_for_adaptive,
            &eq_cfg,
            RenderOptions {
                hrtf_intensity: 0.72,
                environment: Some(env),
                headphone_eq: true,
                adaptive: true,
            },
        )?;
    }

    println!("Done. Compare files in compare_out/");
    Ok(())
}

fn ensure_rate(sr: u32, profile: &HrtfProfile) -> Result<(), Box<dyn Error>> {
    if sr != profile.sample_rate {
        return Err(format!(
            "Sample-rate mismatch: input={} Hz, HRTF={} Hz.",
            sr, profile.sample_rate
        )
        .into());
    }
    Ok(())
}

fn render_and_write(
    path: &str,
    input_l: &[f32],
    input_r: &[f32],
    sr: u32,
    profile: &HrtfProfile,
    eq_cfg: &HeadphoneEqConfig,
    opt: RenderOptions,
) -> Result<(), Box<dyn Error>> {
    println!("\nRendering {}", path);

    let cfg = match opt.environment {
        Some(env) => RoomConfig::preset(env),
        None => RoomConfig::preset(EnvironmentKind::OpenAir),
    };

    let left_az = cfg.left_speaker_az_deg;
    let right_az = cfg.right_speaker_az_deg;
    let baker = RoomBaker::new(sr, cfg);

    let left_spk = baker.bake_speaker(profile, left_az, opt.hrtf_intensity, 0.35);
    let right_spk = baker.bake_speaker(profile, right_az, opt.hrtf_intensity, 0.35);

    if let Some(env) = opt.environment {
        println!("Environment: {}", env.label());
        print_events("L speaker", &left_spk, sr);
        print_events("R speaker", &right_spk, sr);

        let late = baker.late_params();
        println!(
            "Late: on={} RT60={:.2}s HF_RT60={:.2}s pre={:.1}ms damp={:.0}Hz gain={:.3}",
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
    let mut adaptive = AdaptiveController::new();
    let mut eq = eq_cfg.build(sr as f32);

    let n = input_l.len().min(input_r.len());
    let mut out_l = vec![0.0_f32; n];
    let mut out_r = vec![0.0_f32; n];

    for base in (0..n).step_by(BLOCK) {
        let len = (n - base).min(BLOCK);
        let mut bl = vec![0.0_f32; BLOCK];
        let mut br = vec![0.0_f32; BLOCK];
        bl[..len].copy_from_slice(&input_l[base..base + len]);
        br[..len].copy_from_slice(&input_r[base..base + len]);

        let x_ll = d_ll.process(&bl);
        let x_lr = d_lr.process(&bl);
        let x_rl = d_rl.process(&br);
        let x_rr = d_rr.process(&br);

        let mut block_l = vec![0.0_f32; BLOCK];
        let mut block_r = vec![0.0_f32; BLOCK];

        for i in 0..BLOCK {
            block_l[i] = x_ll[i] + x_rl[i];
            block_r[i] = x_lr[i] + x_rr[i];
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

            let fdn_send = if opt.adaptive { params.late_send } else { 1.0 };
            let (late_l, late_r) = fdn.process_block(&bl, &br, fdn_send);

            for i in 0..BLOCK {
                block_l[i] += (y_ll[i] + y_rl[i]) * params.early_gain + late_l[i];
                block_r[i] += (y_lr[i] + y_rr[i]) * params.early_gain + late_r[i];
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

    prevent_clipping(&mut out_l, &mut out_r);
    write_stereo_wav(path, &out_l, &out_r, sr)?;
    Ok(())
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
