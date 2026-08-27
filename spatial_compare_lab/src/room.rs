use crate::hrtf::HrtfProfile;
use num_complex::Complex;
use rustfft::FftPlanner;
use std::collections::HashSet;

const C: f32 = 343.0;

#[derive(Clone, Copy, Debug)]
pub enum EnvironmentKind {
    OpenAir,
    Street,
    RecordingStudio,
    JazzClub,
    PianoHall,
    Theater,
}

impl EnvironmentKind {
    pub fn all() -> [Self; 6] {
        [
            Self::OpenAir,
            Self::Street,
            Self::RecordingStudio,
            Self::JazzClub,
            Self::PianoHall,
            Self::Theater,
        ]
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::OpenAir => "open_air",
            Self::Street => "street",
            Self::RecordingStudio => "recording_studio",
            Self::JazzClub => "jazz_club",
            Self::PianoHall => "piano_hall",
            Self::Theater => "theater",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::OpenAir => "Open Air",
            Self::Street => "Street",
            Self::RecordingStudio => "Recording Studio",
            Self::JazzClub => "Wood Jazz Club",
            Self::PianoHall => "Piano Hall",
            Self::Theater => "Theater",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub absorption_low: f32,
    pub absorption_mid: f32,
    pub absorption_high: f32,
    pub scattering: f32,
}

impl Material {
    pub const fn new(low: f32, mid: f32, high: f32, scattering: f32) -> Self {
        Self {
            absorption_low: low,
            absorption_mid: mid,
            absorption_high: high,
            scattering,
        }
    }

    fn reflection_amp(self) -> [f32; 3] {
        [
            (1.0 - self.absorption_low).clamp(0.0, 1.0).sqrt(),
            (1.0 - self.absorption_mid).clamp(0.0, 1.0).sqrt(),
            (1.0 - self.absorption_high).clamp(0.0, 1.0).sqrt(),
        ]
    }
}

pub const OPEN: Material = Material::new(1.0, 1.0, 1.0, 1.0);
pub const ASPHALT: Material = Material::new(0.08, 0.12, 0.25, 0.10);
pub const BRICK: Material = Material::new(0.03, 0.05, 0.08, 0.12);
pub const PLASTER: Material = Material::new(0.04, 0.06, 0.10, 0.10);
pub const WOOD_PANEL: Material = Material::new(0.12, 0.18, 0.28, 0.28);
pub const WOOD_FLOOR: Material = Material::new(0.10, 0.15, 0.25, 0.18);
pub const CARPET: Material = Material::new(0.08, 0.35, 0.70, 0.35);
pub const CURTAIN: Material = Material::new(0.18, 0.55, 0.78, 0.45);
pub const ACOUSTIC_PANEL: Material = Material::new(0.38, 0.72, 0.88, 0.55);
pub const DIFFUSIVE_WOOD: Material = Material::new(0.10, 0.20, 0.32, 0.65);
pub const AUDIENCE: Material = Material::new(0.25, 0.65, 0.82, 0.55);
pub const CEILING_CLOUD: Material = Material::new(0.30, 0.68, 0.85, 0.60);

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Surface {
    Left,
    Right,
    Back,
    Front,
    Floor,
    Ceiling,
}

#[derive(Clone, Copy)]
struct SurfaceSpec {
    enabled: bool,
    material: Material,
}

impl SurfaceSpec {
    const fn on(material: Material) -> Self {
        Self {
            enabled: true,
            material,
        }
    }
    const fn off() -> Self {
        Self {
            enabled: false,
            material: OPEN,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReflectionEvent {
    pub azimuth_deg: f32,
    pub delay_samples: usize,
    pub gain: f32,
    pub reflection_low: f32,
    pub reflection_mid: f32,
    pub reflection_high: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct LateParams {
    pub enabled: bool,
    pub rt60_s: f32,
    pub rt60_high_s: f32,
    pub predelay_ms: f32,
    pub damping_cutoff_hz: f32,
    pub output_gain: f32,
}

pub struct RoomConfig {
    pub environment: EnvironmentKind,
    pub width_m: f32,
    pub depth_m: f32,
    pub height_m: f32,
    pub listener_x: f32,
    pub listener_y: f32,
    pub listener_z: f32,
    pub speaker_z: f32,

    pub speaker_distance_m: f32,
    pub left_speaker_az_deg: f32,
    pub right_speaker_az_deg: f32,

    pub max_events: usize,
    pub max_er_ms: f32,
    pub air_hf_loss_db_per_m: f32,

    left: SurfaceSpec,
    right: SurfaceSpec,
    back: SurfaceSpec,
    front: SurfaceSpec,
    floor: SurfaceSpec,
    ceiling: SurfaceSpec,

    late_override: Option<LateParams>,
    late_mix: f32,
}

impl RoomConfig {
    pub fn preset(kind: EnvironmentKind) -> Self {
        match kind {
            EnvironmentKind::OpenAir => Self {
                environment: kind,
                width_m: 60.0,
                depth_m: 80.0,
                height_m: 30.0,
                listener_x: 30.0,
                listener_y: 20.0,
                listener_z: 1.20,
                speaker_z: 1.20,
                speaker_distance_m: 1.5,
                left_speaker_az_deg: -30.0,
                right_speaker_az_deg: 30.0,
                max_events: 1,
                max_er_ms: 30.0,
                air_hf_loss_db_per_m: 0.035,
                left: SurfaceSpec::off(),
                right: SurfaceSpec::off(),
                back: SurfaceSpec::off(),
                front: SurfaceSpec::off(),
                floor: SurfaceSpec::on(ASPHALT),
                ceiling: SurfaceSpec::off(),
                late_override: Some(LateParams {
                    enabled: false,
                    rt60_s: 0.0,
                    rt60_high_s: 0.0,
                    predelay_ms: 0.0,
                    damping_cutoff_hz: 8000.0,
                    output_gain: 0.0,
                }),
                late_mix: 0.0,
            },

            EnvironmentKind::Street => Self {
                environment: kind,
                width_m: 12.0,
                depth_m: 50.0,
                height_m: 20.0,
                listener_x: 6.0,
                listener_y: 15.0,
                listener_z: 1.20,
                speaker_z: 1.20,
                speaker_distance_m: 2.5,
                left_speaker_az_deg: -30.0,
                right_speaker_az_deg: 30.0,
                max_events: 6,
                max_er_ms: 80.0,
                air_hf_loss_db_per_m: 0.04,
                left: SurfaceSpec::on(BRICK),
                right: SurfaceSpec::on(BRICK),
                back: SurfaceSpec::off(),
                front: SurfaceSpec::off(),
                floor: SurfaceSpec::on(ASPHALT),
                ceiling: SurfaceSpec::off(),
                late_override: Some(LateParams {
                    enabled: true,
                    rt60_s: 0.45,
                    rt60_high_s: 0.30,
                    predelay_ms: 70.0,
                    damping_cutoff_hz: 6500.0,
                    output_gain: 0.07,
                }),
                late_mix: 0.07,
            },

            EnvironmentKind::RecordingStudio => Self {
                environment: kind,
                width_m: 6.0,
                depth_m: 8.0,
                height_m: 3.0,
                listener_x: 3.0,
                listener_y: 2.8,
                listener_z: 1.20,
                speaker_z: 1.20,
                speaker_distance_m: 1.5,
                left_speaker_az_deg: -30.0,
                right_speaker_az_deg: 30.0,
                max_events: 6,
                max_er_ms: 45.0,
                air_hf_loss_db_per_m: 0.025,
                left: SurfaceSpec::on(ACOUSTIC_PANEL),
                right: SurfaceSpec::on(ACOUSTIC_PANEL),
                back: SurfaceSpec::on(ACOUSTIC_PANEL),
                front: SurfaceSpec::on(DIFFUSIVE_WOOD),
                floor: SurfaceSpec::on(CARPET),
                ceiling: SurfaceSpec::on(CEILING_CLOUD),
                late_override: None,
                late_mix: 0.075,
            },

            EnvironmentKind::JazzClub => Self {
                environment: kind,
                width_m: 8.0,
                depth_m: 12.0,
                height_m: 3.5,
                listener_x: 4.0,
                listener_y: 4.0,
                listener_z: 1.20,
                speaker_z: 1.20,
                speaker_distance_m: 1.8,
                left_speaker_az_deg: -30.0,
                right_speaker_az_deg: 30.0,
                max_events: 8,
                max_er_ms: 55.0,
                air_hf_loss_db_per_m: 0.03,
                left: SurfaceSpec::on(WOOD_PANEL),
                right: SurfaceSpec::on(WOOD_PANEL),
                back: SurfaceSpec::on(CURTAIN),
                front: SurfaceSpec::on(DIFFUSIVE_WOOD),
                floor: SurfaceSpec::on(WOOD_FLOOR),
                ceiling: SurfaceSpec::on(ACOUSTIC_PANEL),
                late_override: None,
                late_mix: 0.12,
            },

            EnvironmentKind::PianoHall => Self {
                environment: kind,
                width_m: 15.0,
                depth_m: 22.0,
                height_m: 8.0,
                listener_x: 7.5,
                listener_y: 7.0,
                listener_z: 1.20,
                speaker_z: 1.20,
                speaker_distance_m: 2.5,
                left_speaker_az_deg: -30.0,
                right_speaker_az_deg: 30.0,
                max_events: 10,
                max_er_ms: 80.0,
                air_hf_loss_db_per_m: 0.035,
                left: SurfaceSpec::on(WOOD_PANEL),
                right: SurfaceSpec::on(WOOD_PANEL),
                back: SurfaceSpec::on(PLASTER),
                front: SurfaceSpec::on(DIFFUSIVE_WOOD),
                floor: SurfaceSpec::on(WOOD_FLOOR),
                ceiling: SurfaceSpec::on(PLASTER),
                late_override: None,
                late_mix: 0.17,
            },

            EnvironmentKind::Theater => Self {
                environment: kind,
                width_m: 20.0,
                depth_m: 30.0,
                height_m: 12.0,
                listener_x: 10.0,
                listener_y: 10.0,
                listener_z: 1.20,
                speaker_z: 1.50,
                speaker_distance_m: 3.0,
                left_speaker_az_deg: -30.0,
                right_speaker_az_deg: 30.0,
                max_events: 10,
                max_er_ms: 85.0,
                air_hf_loss_db_per_m: 0.035,
                left: SurfaceSpec::on(AUDIENCE),
                right: SurfaceSpec::on(AUDIENCE),
                back: SurfaceSpec::on(CURTAIN),
                front: SurfaceSpec::on(DIFFUSIVE_WOOD),
                floor: SurfaceSpec::on(CARPET),
                ceiling: SurfaceSpec::on(DIFFUSIVE_WOOD),
                late_override: None,
                late_mix: 0.13,
            },
        }
    }

    pub fn with_distance(mut self, distance_m: f32) -> Self {
        self.speaker_distance_m = distance_m.max(0.15);
        self
    }

    fn spec(&self, s: Surface) -> SurfaceSpec {
        match s {
            Surface::Left => self.left,
            Surface::Right => self.right,
            Surface::Back => self.back,
            Surface::Front => self.front,
            Surface::Floor => self.floor,
            Surface::Ceiling => self.ceiling,
        }
    }

    fn active_surfaces(&self) -> Vec<Surface> {
        [
            Surface::Left,
            Surface::Right,
            Surface::Back,
            Surface::Front,
            Surface::Floor,
            Surface::Ceiling,
        ]
        .into_iter()
        .filter(|&s| self.spec(s).enabled)
        .collect()
    }

    pub fn late_params(&self) -> LateParams {
        if let Some(p) = self.late_override {
            return p;
        }

        let v = self.width_m * self.depth_m * self.height_m;
        let s_total = 2.0
            * (self.width_m * self.depth_m
                + self.width_m * self.height_m
                + self.depth_m * self.height_m);

        let mut a_mid = 0.0;
        let mut a_high = 0.0;

        for s in [
            Surface::Left,
            Surface::Right,
            Surface::Back,
            Surface::Front,
            Surface::Floor,
            Surface::Ceiling,
        ] {
            let sp = self.spec(s);
            if !sp.enabled {
                continue;
            }

            let area = match s {
                Surface::Left | Surface::Right => self.depth_m * self.height_m,
                Surface::Back | Surface::Front => self.width_m * self.height_m,
                Surface::Floor | Surface::Ceiling => self.width_m * self.depth_m,
            };

            a_mid += area * sp.material.absorption_mid;
            a_high += area * sp.material.absorption_high;
        }

        let rt60_mid = if a_mid > 1e-3 {
            (0.161 * v / a_mid).clamp(0.20, 3.50)
        } else {
            3.50
        };

        let rt60_high = if a_high > 1e-3 {
            (0.161 * v / a_high).clamp(0.15, rt60_mid)
        } else {
            rt60_mid
        };

        let mean_free_path = 4.0 * v / s_total.max(1e-3);
        let predelay_ms = (1000.0 * 0.70 * mean_free_path / C).clamp(24.0, 85.0);

        let ratio = (rt60_high / rt60_mid.max(1e-3)).clamp(0.2, 1.0);
        let damping_cutoff_hz = 2500.0 + 5500.0 * ratio;

        LateParams {
            enabled: true,
            rt60_s: rt60_mid,
            rt60_high_s: rt60_high,
            predelay_ms,
            damping_cutoff_hz,
            output_gain: self.late_mix,
        }
    }
}

impl Default for RoomConfig {
    fn default() -> Self {
        Self::preset(EnvironmentKind::JazzClub)
    }
}

pub struct BakedSpeaker {
    pub direct_l: Vec<f32>,
    pub direct_r: Vec<f32>,
    pub early_l: Vec<f32>,
    pub early_r: Vec<f32>,
    pub events: Vec<ReflectionEvent>,
}

pub struct RoomBaker {
    pub cfg: RoomConfig,
    pub sr: u32,
}

impl RoomBaker {
    pub fn new(sr: u32, cfg: RoomConfig) -> Self {
        Self { cfg, sr }
    }

    pub fn late_params(&self) -> LateParams {
        self.cfg.late_params()
    }

    pub fn bake_speaker(
        &self,
        profile: &HrtfProfile,
        speaker_az_deg: f32,
        direct_intensity: f32,
        er_intensity: f32,
    ) -> BakedSpeaker {
        let (sx, sy, sz) = self.source_position(speaker_az_deg);
        let direct_distance = distance3(
            sx,
            sy,
            sz,
            self.cfg.listener_x,
            self.cfg.listener_y,
            self.cfg.listener_z,
        );

        let direct = profile.hrir(speaker_az_deg, direct_intensity);
        let direct_hf = db_to_amp(-self.cfg.air_hf_loss_db_per_m * direct_distance);

        let direct_l = shape_ir_3band(&direct.left, self.sr, 1.0, 1.0, direct_hf);
        let direct_r = shape_ir_3band(&direct.right, self.sr, 1.0, 1.0, direct_hf);

        let events = self.generate_events(speaker_az_deg);

        let max_delay = events.iter().map(|e| e.delay_samples).max().unwrap_or(0);
        let hlen = profile.dirs[0].left.len().max(profile.dirs[0].right.len());
        let mut early_l = vec![0.0_f32; max_delay + hlen + 1];
        let mut early_r = vec![0.0_f32; max_delay + hlen + 1];

        for ev in &events {
            let h = profile.hrir(ev.azimuth_deg, er_intensity);
            let hl = shape_ir_3band(
                &h.left,
                self.sr,
                ev.reflection_low,
                ev.reflection_mid,
                ev.reflection_high,
            );
            let hr = shape_ir_3band(
                &h.right,
                self.sr,
                ev.reflection_low,
                ev.reflection_mid,
                ev.reflection_high,
            );

            for (i, &x) in hl.iter().enumerate() {
                let p = ev.delay_samples + i;
                if p < early_l.len() {
                    early_l[p] += x * ev.gain;
                }
            }
            for (i, &x) in hr.iter().enumerate() {
                let p = ev.delay_samples + i;
                if p < early_r.len() {
                    early_r[p] += x * ev.gain;
                }
            }
        }

        BakedSpeaker {
            direct_l,
            direct_r,
            early_l,
            early_r,
            events,
        }
    }

    fn generate_events(&self, speaker_az_deg: f32) -> Vec<ReflectionEvent> {
        let (sx, sy, sz) = self.source_position(speaker_az_deg);
        let lx = self.cfg.listener_x;
        let ly = self.cfg.listener_y;
        let lz = self.cfg.listener_z;
        let direct_dist = distance3(sx, sy, sz, lx, ly, lz);
        let active = self.cfg.active_surfaces();

        #[derive(Clone)]
        struct Candidate {
            x: f32,
            y: f32,
            z: f32,
            refl: [f32; 3],
            scattering: f32,
        }

        let mut candidates = Vec::<Candidate>::new();

        for &s in &active {
            let sp = self.cfg.spec(s);
            let (x, y, z) = reflect3(
                sx,
                sy,
                sz,
                s,
                self.cfg.width_m,
                self.cfg.depth_m,
                self.cfg.height_m,
            );
            candidates.push(Candidate {
                x,
                y,
                z,
                refl: sp.material.reflection_amp(),
                scattering: sp.material.scattering,
            });
        }

        for &s1 in &active {
            let sp1 = self.cfg.spec(s1);
            let (x1, y1, z1) = reflect3(
                sx,
                sy,
                sz,
                s1,
                self.cfg.width_m,
                self.cfg.depth_m,
                self.cfg.height_m,
            );
            for &s2 in &active {
                if s1 == s2 {
                    continue;
                }
                let sp2 = self.cfg.spec(s2);
                let (x2, y2, z2) = reflect3(
                    x1,
                    y1,
                    z1,
                    s2,
                    self.cfg.width_m,
                    self.cfg.depth_m,
                    self.cfg.height_m,
                );
                let r1 = sp1.material.reflection_amp();
                let r2 = sp2.material.reflection_amp();
                candidates.push(Candidate {
                    x: x2,
                    y: y2,
                    z: z2,
                    refl: [r1[0] * r2[0], r1[1] * r2[1], r1[2] * r2[2]],
                    scattering: 1.0
                        - (1.0 - sp1.material.scattering) * (1.0 - sp2.material.scattering),
                });
            }
        }

        let mut seen = HashSet::new();
        let mut events = Vec::new();

        for c in candidates {
            let key = (
                (c.x * 1000.0).round() as i32,
                (c.y * 1000.0).round() as i32,
                (c.z * 1000.0).round() as i32,
            );
            if !seen.insert(key) {
                continue;
            }

            let path = distance3(c.x, c.y, c.z, lx, ly, lz);
            if path <= direct_dist {
                continue;
            }

            let extra_s = (path - direct_dist) / C;
            let delay_ms = extra_s * 1000.0;
            if delay_ms <= 0.35 || delay_ms > self.cfg.max_er_ms {
                continue;
            }

            let specular = (1.0 - 0.65 * c.scattering).clamp(0.15, 1.0);
            let gain = (direct_dist / path) * specular;
            if gain < 0.015 {
                continue;
            }

            let azimuth_deg = (c.x - lx).atan2(c.y - ly).to_degrees();
            let delay_samples = (extra_s * self.sr as f32).round() as usize;
            let air_hf = db_to_amp(-self.cfg.air_hf_loss_db_per_m * path);

            events.push(ReflectionEvent {
                azimuth_deg,
                delay_samples,
                gain,
                reflection_low: c.refl[0],
                reflection_mid: c.refl[1],
                reflection_high: c.refl[2] * air_hf,
            });
        }

        events.sort_by(|a, b| {
            let sa = a.gain * a.reflection_mid;
            let sb = b.gain * b.reflection_mid;
            sb.partial_cmp(&sa).unwrap()
        });
        events.truncate(self.cfg.max_events);
        events.sort_by_key(|e| e.delay_samples);
        events
    }

    fn source_position(&self, az_deg: f32) -> (f32, f32, f32) {
        let a = az_deg.to_radians();
        (
            self.cfg.listener_x + self.cfg.speaker_distance_m * a.sin(),
            self.cfg.listener_y + self.cfg.speaker_distance_m * a.cos(),
            self.cfg.speaker_z,
        )
    }
}

fn distance3(x1: f32, y1: f32, z1: f32, x2: f32, y2: f32, z2: f32) -> f32 {
    ((x1 - x2).powi(2) + (y1 - y2).powi(2) + (z1 - z2).powi(2)).sqrt()
}

fn reflect3(
    x: f32,
    y: f32,
    z: f32,
    s: Surface,
    width: f32,
    depth: f32,
    height: f32,
) -> (f32, f32, f32) {
    match s {
        Surface::Left => (-x, y, z),
        Surface::Right => (2.0 * width - x, y, z),
        Surface::Back => (x, -y, z),
        Surface::Front => (x, 2.0 * depth - y, z),
        Surface::Floor => (x, y, -z),
        Surface::Ceiling => (x, y, 2.0 * height - z),
    }
}

fn db_to_amp(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

// Offline only: material coloration is baked into each ER HRIR.
fn shape_ir_3band(ir: &[f32], sr: u32, low: f32, mid: f32, high: f32) -> Vec<f32> {
    if ir.is_empty() {
        return vec![0.0];
    }

    let nfft = (ir.len() * 2).next_power_of_two().max(256);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(nfft);
    let ifft = planner.plan_fft_inverse(nfft);

    let mut buf = vec![Complex::new(0.0_f32, 0.0_f32); nfft];
    for (i, &x) in ir.iter().enumerate() {
        buf[i].re = x;
    }

    fft.process(&mut buf);

    for (k, bin) in buf.iter_mut().enumerate() {
        let kk = if k <= nfft / 2 { k } else { nfft - k };
        let f = kk as f32 * sr as f32 / nfft as f32;

        let g = if f <= 250.0 {
            low
        } else if f < 2000.0 {
            let t = (f - 250.0) / (2000.0 - 250.0);
            low + (mid - low) * t
        } else if f < 8000.0 {
            let t = (f - 2000.0) / (8000.0 - 2000.0);
            mid + (high - mid) * t
        } else {
            high
        };

        *bin *= g;
    }

    ifft.process(&mut buf);
    let scale = 1.0 / nfft as f32;
    buf[..ir.len()].iter().map(|c| c.re * scale).collect()
}

