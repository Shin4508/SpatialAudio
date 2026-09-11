use hound::{SampleFormat, WavSpec, WavWriter};
use spatial_compare_lab_v2::{
    audio::{load_stereo_wav, prevent_clipping, write_stereo_wav},
    headphone_eq::HeadphoneEqConfig,
    hrtf::HrtfProfile,
    render_with_config,
    room::{EnvironmentKind, RoomConfig},
    RenderOptions,
};
use std::{env, error::Error, path::PathBuf};

/// The first electronic-module profile. Keep this fixed while validating the
/// audio I/O, timing and enclosure; room controls can be added after that.
const DEFAULT_MODE: &str = "jazz_club";

fn usage() -> ! {
    eprintln!(
        "Usage: spatial-electronic-module <input.wav> <output.wav> [mode] [distance_m] [half_angle_deg]\n\
         mode: open_air, street, recording_studio, jazz_club, piano_hall, theater, concertgebouw\n\
         HRTF_PROFILE may point to a profile directory (default: spatial_compare_lab/hrtf_profiles/default)"
    );
    std::process::exit(2);
}

fn mode(value: &str) -> Result<EnvironmentKind, Box<dyn Error>> {
    EnvironmentKind::all()
        .into_iter()
        .find(|kind| kind.slug() == value)
        .ok_or_else(|| format!("Unknown mode '{value}'").into())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 6 {
        usage();
    }
    let input = &args[1];
    let output = &args[2];
    let selected = args.get(3).map(String::as_str).unwrap_or(DEFAULT_MODE);
    let kind = mode(selected)?;
    let distance = args.get(4).map(|value| value.parse::<f32>()).transpose()?;
    let angle = args
        .get(5)
        .map(|value| value.parse::<f32>())
        .transpose()?
        .unwrap_or(30.0);
    if !angle.is_finite() || !(0.0..=90.0).contains(&angle) {
        return Err("half_angle_deg must be between 0 and 90".into());
    }

    let profile_path = env::var_os("HRTF_PROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../spatial_compare_lab/hrtf_profiles/default"));
    let profile = HrtfProfile::load(profile_path.to_str().ok_or("Invalid HRTF profile path")?)?;
    let (left, right, sample_rate) = load_stereo_wav(input)?;
    if sample_rate != profile.sample_rate {
        return Err(format!(
            "Input is {} Hz but the HRTF profile is {} Hz",
            sample_rate, profile.sample_rate
        )
        .into());
    }

    let mut config = RoomConfig::preset(kind);
    if let Some(distance) = distance {
        if !distance.is_finite() || !(0.15..=10.0).contains(&distance) {
            return Err("distance_m must be between 0.15 and 10".into());
        }
        config = config.with_distance(distance);
    }
    config.left_speaker_az_deg = -angle;
    config.right_speaker_az_deg = angle;
    config.validate()?;
    let eq = HeadphoneEqConfig::load_or_flat("config/headphone_eq.txt")?;
    render_with_config(
        output,
        &left,
        &right,
        sample_rate,
        &profile,
        &eq,
        RenderOptions {
            direct_hrtf_intensity: 1.0,
            environment: Some(kind),
            headphone_eq: true,
            adaptive: false,
            distance_m: distance,
            angle_deg: angle,
        },
        Some(config),
    )?;

    // Apply the same safe peak ceiling as the comparison CLI to the one output.
    let (mut out_l, mut out_r, rate) = load_stereo_wav(output)?;
    prevent_clipping(&mut out_l, &mut out_r);
    write_stereo_wav(output, &out_l, &out_r, rate)?;
    println!(
        "Rendered one mode: {} ({}) -> {}",
        kind.slug(),
        kind.label(),
        output
    );
    Ok(())
}

// Keep the binary's WAV dependency visible to downstream hardware bring-up
// tools that use its manifest to generate test fixtures.
#[allow(dead_code)]
fn wav_spec(sample_rate: u32) -> WavSpec {
    WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    }
}

#[allow(dead_code)]
fn create_writer(
    path: &str,
    sample_rate: u32,
) -> Result<WavWriter<std::io::BufWriter<std::fs::File>>, hound::Error> {
    WavWriter::create(path, wav_spec(sample_rate))
}
