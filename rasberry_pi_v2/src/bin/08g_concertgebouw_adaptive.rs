//! Raspberry Pi 5 real-time target for v2's 08g Concertgebouw adaptive mode.
//!
//! The expensive room bake happens before the audio streams start. The audio
//! callback then runs one fixed 256-sample mode and never changes rooms.
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SampleRate, StreamConfig};
use spatial_compare_lab_v2::{
    adaptive::AdaptiveController,
    fdn::StereoFdn,
    filters::FftConvolver,
    hrtf::HrtfProfile,
    room::{EnvironmentKind, RoomBaker, RoomConfig},
};
use std::{
    collections::VecDeque,
    env,
    error::Error,
    path::PathBuf,
    sync::{Arc, Mutex},
};

const BLOCK: usize = 256;
struct Engine {
    dl_l: FftConvolver,
    dl_r: FftConvolver,
    dr_l: FftConvolver,
    dr_r: FftConvolver,
    el_l: FftConvolver,
    el_r: FftConvolver,
    er_l: FftConvolver,
    er_r: FftConvolver,
    fdn: StereoFdn,
    adaptive: AdaptiveController,
    direct: f32,
    early: f32,
    late: f32,
    in_l: Vec<f32>,
    in_r: Vec<f32>,
    out_l: VecDeque<f32>,
    out_r: VecDeque<f32>,
}
impl Engine {
    fn new(
        sr: u32,
        profile: &HrtfProfile,
        distance: f32,
        angle: f32,
    ) -> Result<Self, Box<dyn Error>> {
        let mut cfg = RoomConfig::preset(EnvironmentKind::Concertgebouw).with_distance(distance);
        cfg.left_speaker_az_deg = -angle;
        cfg.right_speaker_az_deg = angle;
        cfg.validate()?;
        let baker = RoomBaker::new(sr, cfg);
        let left = baker.bake_speaker(profile, -angle, 1.0);
        let right = baker.bake_speaker(profile, angle, 1.0);
        let cues = baker.distance_cues();
        let late = baker.late_params();
        Ok(Self {
            dl_l: FftConvolver::new(&left.direct_l, BLOCK),
            dl_r: FftConvolver::new(&left.direct_r, BLOCK),
            dr_l: FftConvolver::new(&right.direct_l, BLOCK),
            dr_r: FftConvolver::new(&right.direct_r, BLOCK),
            el_l: FftConvolver::new(&left.early_l, BLOCK),
            el_r: FftConvolver::new(&left.early_r, BLOCK),
            er_l: FftConvolver::new(&right.early_l, BLOCK),
            er_r: FftConvolver::new(&right.early_r, BLOCK),
            fdn: StereoFdn::new(sr as f32, late),
            adaptive: AdaptiveController::new(sr as f32),
            direct: cues.direct_gain,
            early: cues.early_gain,
            late: cues.late_send,
            in_l: Vec::with_capacity(BLOCK),
            in_r: Vec::with_capacity(BLOCK),
            out_l: VecDeque::with_capacity(BLOCK),
            out_r: VecDeque::with_capacity(BLOCK),
        })
    }
    fn fill(&mut self, left: f32, right: f32) {
        self.in_l.push(left);
        self.in_r.push(right);
        if self.in_l.len() < BLOCK {
            return;
        }
        let bl = std::mem::take(&mut self.in_l);
        let br = std::mem::take(&mut self.in_r);
        let dl = self.dl_l.process(&bl);
        let dr = self.dr_l.process(&br);
        let dlr = self.dl_r.process(&bl);
        let drr = self.dr_r.process(&br);
        let el = self.el_l.process(&bl);
        let er = self.er_l.process(&br);
        let elr = self.el_r.process(&bl);
        let err = self.er_r.process(&br);
        let p = self.adaptive.analyze(&bl, &br);
        let (late_l, late_r) = self.fdn.process_block(&bl, &br, p.late_send * self.late);
        for i in 0..BLOCK {
            self.out_l.push_back(
                (dl[i] + dr[i]) * self.direct
                    + (el[i] + er[i]) * p.early_gain * self.early
                    + late_l[i],
            );
            self.out_r.push_back(
                (dlr[i] + drr[i]) * self.direct
                    + (elr[i] + err[i]) * p.early_gain * self.early
                    + late_r[i],
            );
        }
    }
    fn next(&mut self, left: f32, right: f32) -> (f32, f32) {
        self.fill(left, right);
        (
            self.out_l.pop_front().unwrap_or(0.0),
            self.out_r.pop_front().unwrap_or(0.0),
        )
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let distance = env::args()
        .nth(1)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(3.0_f32);
    let angle = env::args()
        .nth(2)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(30.0_f32);
    let profile_path = env::var_os("HRTF_PROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../spatial_compare_lab/hrtf_profiles/default"));
    let host = cpal::default_host();
    let input = host
        .default_input_device()
        .ok_or("No default input device")?;
    let output = host
        .default_output_device()
        .ok_or("No default output device")?;
    // The bundled HRIRs are 48 kHz. Explicitly choose that rate even when
    // PipeWire/ALSA reports 44.1 kHz as the system default.
    let in_cfg = choose_config(&input, true)?;
    let out_cfg = choose_config(&output, false)?;
    if in_cfg.sample_rate != out_cfg.sample_rate {
        return Err("Input/output sample rates differ".into());
    }
    let sr = in_cfg.sample_rate.0;
    let profile = HrtfProfile::load(profile_path.to_str().ok_or("Invalid HRTF_PROFILE")?)?;
    if profile.sample_rate != sr {
        return Err(format!(
            "HRTF is {} Hz but the audio device is {} Hz",
            profile.sample_rate, sr
        )
        .into());
    }
    let engine = Arc::new(Mutex::new(Engine::new(sr, &profile, distance, angle)?));
    let queue = Arc::new(Mutex::new(VecDeque::<(f32, f32)>::with_capacity(
        sr as usize / 2,
    )));
    let input_queue = Arc::clone(&queue);
    let input_stream = input.build_input_stream(
        &in_cfg,
        move |data: &[f32], _| {
            if let Ok(mut q) = input_queue.lock() {
                for frame in data.chunks(2) {
                    if frame.len() >= 2 {
                        q.push_back((frame[0], frame[1]));
                    }
                }
            }
        },
        |err| eprintln!("input stream: {err}"),
        None,
    )?;
    let output_queue = Arc::clone(&queue);
    let output_engine = Arc::clone(&engine);
    let channels = 2usize;
    let output_stream = output.build_output_stream(
        &out_cfg,
        move |data: &mut [f32], _| {
            if let (Ok(mut q), Ok(mut e)) = (output_queue.lock(), output_engine.lock()) {
                for frame in data.chunks_mut(channels) {
                    let (l, r) = q
                        .pop_front()
                        .map(|(l, r)| e.next(l, r))
                        .unwrap_or_else(|| e.next(0.0, 0.0));
                    frame[0] = l.clamp(-0.98, 0.98);
                    frame[1] = r.clamp(-0.98, 0.98);
                    for x in &mut frame[2..] {
                        *x = 0.0;
                    }
                }
            } else {
                data.fill(0.0);
            }
        },
        |err| eprintln!("output stream: {err}"),
        None,
    )?;
    println!(
        "08g Concertgebouw adaptive running at {sr} Hz, distance={distance:.2} m, angle=±{angle:.1}°"
    );
    input_stream.play()?;
    output_stream.play()?;
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

fn choose_config(device: &cpal::Device, input: bool) -> Result<StreamConfig, Box<dyn Error>> {
    let ranges = if input {
        device.supported_input_configs()?.collect::<Vec<_>>()
    } else {
        device.supported_output_configs()?.collect::<Vec<_>>()
    };
    let range = ranges
        .into_iter()
        .find(|r| {
            r.channels() >= 2
                && r.sample_format() == SampleFormat::F32
                && r.min_sample_rate().0 <= 48_000
                && r.max_sample_rate().0 >= 48_000
        })
        .ok_or("No stereo float32 48 kHz stream is available")?;
    let mut config = range.with_sample_rate(SampleRate(48_000)).config();
    config.channels = 2;
    Ok(config)
}
