use spatial_compare_lab_v2::{
    audio::{load_stereo_wav, write_stereo_wav},
    ensure_rate,
    headphone_eq::HeadphoneEqConfig,
    hrtf::{self, HrtfProfile},
    render_and_write,
    room::EnvironmentKind,
    RenderOptions,
};
use std::{error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input_path = args.get(1).map(String::as_str).unwrap_or("output.wav");
    let default_profile_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("../spatial_compare_lab/hrtf_profiles/default");
    let personal_profile_path = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("../spatial_compare_lab/hrtf_profiles/personal");
    let eq_path = args
        .get(4)
        .map(String::as_str)
        .unwrap_or("config/headphone_eq.txt");
    let distance_m = args.get(5).map(|s| s.parse::<f32>()).transpose()?;
    if distance_m.is_some_and(|d| !d.is_finite() || !(0.15..=10.0).contains(&d)) {
        return Err("Distance must be finite and between 0.15 and 10 metres".into());
    }
    let angle_deg = args
        .get(6)
        .map(|s| s.parse::<f32>())
        .transpose()?
        .unwrap_or(30.0);
    if !angle_deg.is_finite() || !(0.0..=90.0).contains(&angle_deg) {
        return Err("Speaker half-angle must be finite and between 0 and 90 degrees".into());
    }
    let mut outputs = Vec::new();

    fs::create_dir_all("compare_out")?;

    let (input_l, input_r, sr) = load_stereo_wav(input_path)?;
    let default_profile = HrtfProfile::load(default_profile_path)?;
    ensure_rate(sr, &default_profile)?;
    let eq_cfg = HeadphoneEqConfig::load_or_flat(eq_path)?;

    write_stereo_wav("compare_out/00_dry.wav", &input_l, &input_r, sr)?;
    outputs.push("compare_out/00_dry.wav".to_string());

    outputs.push(render_and_write(
        "compare_out/01_reference_2speaker.wav",
        &input_l,
        &input_r,
        sr,
        &default_profile,
        &eq_cfg,
        RenderOptions {
            direct_hrtf_intensity: 1.0,
            environment: None,
            headphone_eq: false,
            adaptive: false,
            distance_m,
            angle_deg,
        },
    )?);

    outputs.push(render_and_write(
        "compare_out/02_hrtf_intensity_090.wav",
        &input_l,
        &input_r,
        sr,
        &default_profile,
        &eq_cfg,
        RenderOptions {
            direct_hrtf_intensity: 0.90,
            environment: None,
            headphone_eq: false,
            adaptive: false,
            distance_m,
            angle_deg,
        },
    )?);

    outputs.push(render_and_write(
        "compare_out/05_headphone_comp.wav",
        &input_l,
        &input_r,
        sr,
        &default_profile,
        &eq_cfg,
        RenderOptions {
            direct_hrtf_intensity: 1.0,
            environment: None,
            headphone_eq: true,
            adaptive: false,
            distance_m,
            angle_deg,
        },
    )?);

    for (i, env) in EnvironmentKind::all().into_iter().enumerate() {
        let path = format!(
            "compare_out/06{}_{}.wav",
            (b'a' + i as u8) as char,
            env.slug()
        );

        outputs.push(render_and_write(
            &path,
            &input_l,
            &input_r,
            sr,
            &default_profile,
            &eq_cfg,
            RenderOptions {
                direct_hrtf_intensity: 1.0,
                environment: Some(env),
                headphone_eq: true,
                adaptive: false,
                distance_m,
                angle_deg,
            },
        )?);
    }

    let personal_marker = Path::new(personal_profile_path).join("hrtf_left_0.wav");
    let mut profile_for_adaptive = default_profile.clone();

    if personal_marker.exists() {
        let personal = HrtfProfile::load(personal_profile_path)?;
        ensure_rate(sr, &personal)?;

        outputs.push(render_and_write(
            "compare_out/07_personal_jazz_club.wav",
            &input_l,
            &input_r,
            sr,
            &personal,
            &eq_cfg,
            RenderOptions {
                direct_hrtf_intensity: 1.0,
                environment: Some(EnvironmentKind::JazzClub),
                headphone_eq: true,
                adaptive: false,
                distance_m,
                angle_deg,
            },
        )?);

        profile_for_adaptive = personal;
    }

    for (i, env) in EnvironmentKind::all().into_iter().enumerate() {
        let path = format!(
            "compare_out/08{}_{}_adaptive.wav",
            (b'a' + i as u8) as char,
            env.slug()
        );

        outputs.push(render_and_write(
            &path,
            &input_l,
            &input_r,
            sr,
            &profile_for_adaptive,
            &eq_cfg,
            RenderOptions {
                direct_hrtf_intensity: 1.0,
                environment: Some(env),
                headphone_eq: true,
                adaptive: true,
                distance_m,
                angle_deg,
            },
        )?);
    }

    // One gain for this run preserves relative level and distance differences.
    let mut peak = 0.0_f32;
    for path in &outputs {
        let (l, r, _) = load_stereo_wav(path)?;
        for x in l.iter().chain(&r) {
            if !x.is_finite() {
                return Err("Non-finite render sample".into());
            }
            peak = peak.max(x.abs());
        }
    }
    let gain = if peak > 0.98 { 0.98 / peak } else { 1.0 };
    for path in &outputs {
        let (mut l, mut r, rate) = load_stereo_wav(path)?;
        for x in l.iter_mut().chain(&mut r) {
            *x *= gain;
        }
        write_stereo_wav(path, &l, &r, rate)?;
    }
    fs::write("compare_out/run.txt", format!(
        "input={input_path}\nprofile={default_profile_path}\npersonal={personal_profile_path}\neq={eq_path}\nsample_rate={sr}\ndistance_override={distance_m:?}\nhalf_angle_deg={angle_deg}\nleft_index={}\nright_index={}\ncommon_gain={gain}\npre_gain_peak={peak}\nfiles={outputs:#?}\n",
        hrtf::azimuth_to_index(-angle_deg), hrtf::azimuth_to_index(angle_deg)))?;
    println!("Done. Common gain={gain:.6}. Parameters in compare_out/run.txt");
    Ok(())
}
