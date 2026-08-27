use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::error::Error;

pub fn load_mono_wav(path: &str) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    if channels == 0 {
        return Err("WAV has zero channels".into());
    }

    let interleaved = read_all_samples(&mut reader, spec.sample_format, spec.bits_per_sample)?;
    let mut mono = Vec::with_capacity(interleaved.len() / channels + 1);
    for frame in interleaved.chunks(channels) {
        if let Some(&x) = frame.first() {
            mono.push(x);
        }
    }
    Ok((mono, spec.sample_rate))
}

pub fn load_stereo_wav(path: &str) -> Result<(Vec<f32>, Vec<f32>, u32), Box<dyn Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels < 2 {
        return Err(format!("Input must be stereo, got {} channel(s)", spec.channels).into());
    }

    let channels = spec.channels as usize;
    let interleaved = read_all_samples(&mut reader, spec.sample_format, spec.bits_per_sample)?;

    let mut left = Vec::with_capacity(interleaved.len() / channels + 1);
    let mut right = Vec::with_capacity(interleaved.len() / channels + 1);

    for frame in interleaved.chunks(channels) {
        if frame.len() >= 2 {
            left.push(frame[0]);
            right.push(frame[1]);
        }
    }
    Ok((left, right, spec.sample_rate))
}

fn read_all_samples(
    reader: &mut WavReader<std::io::BufReader<std::fs::File>>,
    format: SampleFormat,
    bits: u16,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let out = match format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(|e| e.into()))
            .collect::<Result<Vec<f32>, Box<dyn Error>>>()?,
        SampleFormat::Int if bits <= 16 => {
            let scale = (1_i32 << (bits.saturating_sub(1) as u32)) as f32;
            reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / scale).map_err(|e| e.into()))
                .collect::<Result<Vec<f32>, Box<dyn Error>>>()?
        }
        SampleFormat::Int => {
            let scale = (1_i64 << (bits.saturating_sub(1) as u32)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale).map_err(|e| e.into()))
                .collect::<Result<Vec<f32>, Box<dyn Error>>>()?
        }
    };
    Ok(out)
}

pub fn write_stereo_wav(
    path: &str,
    left: &[f32],
    right: &[f32],
    sample_rate: u32,
) -> Result<(), Box<dyn Error>> {
    let n = left.len().min(right.len());
    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for i in 0..n {
        writer.write_sample(left[i])?;
        writer.write_sample(right[i])?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn prevent_clipping(left: &mut [f32], right: &mut [f32]) -> f32 {
    let mut peak = 0.0_f32;
    for &x in left.iter().chain(right.iter()) {
        peak = peak.max(x.abs());
    }
    if peak <= 0.98 || peak == 0.0 {
        return 1.0;
    }

    let gain = 0.98 / peak;
    for x in left.iter_mut() {
        *x *= gain;
    }
    for x in right.iter_mut() {
        *x *= gain;
    }
    gain
}
