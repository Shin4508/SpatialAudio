mod convolution;
pub mod profile;

use convolution::Convolver;
use spatial_compare_lab_v2::{
    fdn::StereoFdn,
    headphone_eq::{HeadphoneEq, HeadphoneEqConfig},
    hrtf::HrtfProfile,
    room::{DistanceCues, EnvironmentKind, RoomBaker, RoomConfig},
};
use std::error::Error;

pub const BLOCK: usize = 256;
pub const MODES: [&str; 7] = [
    "06a_open_air",
    "06b_street",
    "06c_recording_studio",
    "06d_jazz_club",
    "06e_piano_hall",
    "06f_theater",
    "06g_concertgebouw",
];

pub fn mode_index(name: &str) -> Option<usize> {
    MODES
        .iter()
        .position(|m| *m == name || &m[..3] == name || &m[4..] == name)
}

pub struct Room {
    filters: [Convolver; 8],
    fdn: StereoFdn,
    eq: HeadphoneEq,
    cues: DistanceCues,
    paths: [[f32; BLOCK]; 8],
}

impl Room {
    pub fn new(
        profile: &HrtfProfile,
        mode: usize,
        eq: &HeadphoneEqConfig,
        distance: Option<f32>,
        angle: f32,
    ) -> Result<Self, Box<dyn Error>> {
        if mode >= MODES.len()
            || !angle.is_finite()
            || !(0.0..=90.0).contains(&angle)
            || distance.is_some_and(|d| !d.is_finite() || !(0.15..=10.0).contains(&d))
        {
            return Err("Invalid mode, distance (0.15–10 m), or angle (0–90 degrees)".into());
        }
        let mut cfg = RoomConfig::preset(EnvironmentKind::all()[mode]);
        if let Some(d) = distance {
            cfg = cfg.with_distance(d);
        }
        cfg.left_speaker_az_deg = -angle;
        cfg.right_speaker_az_deg = angle;
        cfg.validate()?;
        let baker = RoomBaker::new(profile.sample_rate, cfg);
        let l = baker.bake_speaker(profile, -angle, 1.0);
        let r = baker.bake_speaker(profile, angle, 1.0);
        let impulses = [
            l.direct_l, l.direct_r, r.direct_l, r.direct_r, l.early_l, l.early_r, r.early_l,
            r.early_r,
        ];
        if impulses.iter().flatten().any(|x| !x.is_finite()) {
            return Err("Non-finite baked filter".into());
        }
        Ok(Self {
            filters: impulses.each_ref().map(|h| Convolver::new(h)),
            fdn: StereoFdn::new(profile.sample_rate as f32, baker.late_params()),
            eq: eq.build(profile.sample_rate as f32),
            cues: baker.distance_cues(),
            paths: [[0.0; BLOCK]; 8],
        })
    }

    fn reset(&mut self) {
        for f in &mut self.filters {
            f.reset();
        }
        self.fdn.reset();
        self.eq.reset();
    }

    /// Fixed 06 path: full measured HRTF, fixed early/late sends, headphone EQ,
    /// and no adaptive controller or dry anchor. Output is before peak limiting.
    pub fn process(&mut self, l: &[f32; BLOCK], r: &[f32; BLOCK], out: &mut [[f32; BLOCK]; 2]) {
        for i in 0..8 {
            self.filters[i].process(if i % 4 < 2 { l } else { r }, &mut self.paths[i]);
        }
        let [ol, or] = out;
        self.fdn.process_into(l, r, self.cues.late_send, ol, or);
        for i in 0..BLOCK {
            ol[i] += (self.paths[0][i] + self.paths[2][i]) * self.cues.direct_gain
                + (self.paths[4][i] + self.paths[6][i]) * self.cues.early_gain;
            or[i] += (self.paths[1][i] + self.paths[3][i]) * self.cues.direct_gain
                + (self.paths[5][i] + self.paths[7][i]) * self.cues.early_gain;
        }
        self.eq.process(ol, or);
    }
}

/// Fixed-size adapter accepts any JACK quantum. Always adds exactly BLOCK
/// frames of algorithmic buffering. At most two rooms run during a crossfade.
pub struct Player {
    rooms: Vec<Room>,
    active: usize,
    transition: Option<usize>,
    fade_pos: usize,
    fade_len: usize,
    position: usize,
    input: [[f32; BLOCK]; 2],
    output: [[f32; BLOCK]; 2],
    next: [[f32; BLOCK]; 2],
    gain: f32,
    gain_smooth: f32,
    pub limited: u64,
    pub invalid: u64,
}

impl Player {
    pub fn new(rooms: Vec<Room>, active: usize, sample_rate: u32, gain: f32) -> Self {
        assert_eq!(rooms.len(), MODES.len());
        assert!(active < rooms.len());
        Self {
            rooms,
            active,
            transition: None,
            fade_pos: 0,
            fade_len: (sample_rate as f32 * 0.1).round() as usize,
            position: 0,
            input: [[0.0; BLOCK]; 2],
            output: [[0.0; BLOCK]; 2],
            next: [[0.0; BLOCK]; 2],
            gain,
            gain_smooth: 1.0 - (-1.0 / (0.02 * sample_rate as f32)).exp(),
            limited: 0,
            invalid: 0,
        }
    }

    pub fn active(&self) -> usize {
        self.active
    }

    pub fn frame(&mut self, left: f32, right: f32, requested: usize, target_gain: f32) -> [f32; 2] {
        let i = self.position;
        self.gain += (target_gain - self.gain) * self.gain_smooth;
        let mut result = [0.0; 2];
        for (ch, x) in result.iter_mut().enumerate() {
            let v = self.output[ch][i] * self.gain;
            if !v.is_finite() {
                self.invalid += 1;
            } else {
                if v.abs() > 0.98 {
                    self.limited += 1;
                }
                *x = v.clamp(-0.98, 0.98);
            }
        }
        for (ch, value) in [left, right].into_iter().enumerate() {
            self.input[ch][i] = if value.is_finite() {
                value.clamp(-16.0, 16.0)
            } else {
                self.invalid += 1;
                0.0
            };
        }
        self.position += 1;
        if self.position == BLOCK {
            self.position = 0;
            if self.transition.is_none() && requested < self.rooms.len() && requested != self.active
            {
                self.rooms[requested].reset();
                self.transition = Some(requested);
                self.fade_pos = 0;
            }
            self.rooms[self.active].process(&self.input[0], &self.input[1], &mut self.output);
            if let Some(next) = self.transition {
                self.rooms[next].process(&self.input[0], &self.input[1], &mut self.next);
                for i in 0..BLOCK {
                    let mix = ((self.fade_pos + i) as f32 / self.fade_len as f32).min(1.0);
                    for ch in 0..2 {
                        self.output[ch][i] =
                            self.output[ch][i] * (1.0 - mix) + self.next[ch][i] * mix;
                    }
                }
                self.fade_pos += BLOCK;
                if self.fade_pos >= self.fade_len {
                    self.active = next;
                    self.transition = None;
                }
            }
        }
        result
    }
}
