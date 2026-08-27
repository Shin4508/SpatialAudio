use crate::filters::{high_shelf, low_shelf, peaking, Biquad};
use std::error::Error;
use std::fs;

#[derive(Clone)]
enum Kind {
    Peaking,
    LowShelf,
    HighShelf,
}

#[derive(Clone)]
struct EqSpec {
    channel: char,
    kind: Kind,
    f0: f32,
    gain_db: f32,
    q: f32,
}

#[derive(Clone)]
pub struct HeadphoneEqConfig {
    specs: Vec<EqSpec>,
}

impl HeadphoneEqConfig {
    pub fn load_or_flat(path: &str) -> Result<Self, Box<dyn Error>> {
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => return Ok(Self { specs: Vec::new() }),
        };

        let mut specs = Vec::new();
        for (line_no, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let p: Vec<&str> = line.split_whitespace().collect();
            if p.len() != 5 {
                return Err(format!("EQ line {} must be: CHANNEL TYPE FREQ GAIN_DB Q", line_no+1).into());
            }

            let channel = p[0].chars().next().unwrap_or('B').to_ascii_uppercase();
            let kind = match p[1].to_ascii_lowercase().as_str() {
                "peaking" => Kind::Peaking,
                "low_shelf" => Kind::LowShelf,
                "high_shelf" => Kind::HighShelf,
                other => return Err(format!("Unknown EQ type '{}'", other).into()),
            };

            specs.push(EqSpec {
                channel,
                kind,
                f0: p[2].parse()?,
                gain_db: p[3].parse()?,
                q: p[4].parse()?,
            });
        }

        Ok(Self { specs })
    }

    pub fn is_flat(&self) -> bool {
        self.specs.is_empty()
    }

    pub fn build(&self, sr: f32) -> HeadphoneEq {
        let mut left = Vec::new();
        let mut right = Vec::new();

        for s in &self.specs {
            let (b, a) = match s.kind {
                Kind::Peaking => peaking(sr, s.f0, s.gain_db, s.q),
                Kind::LowShelf => low_shelf(sr, s.f0, s.gain_db, s.q),
                Kind::HighShelf => high_shelf(sr, s.f0, s.gain_db, s.q),
            };

            if s.channel == 'B' || s.channel == 'L' {
                left.push(Biquad::new(b, a));
            }
            if s.channel == 'B' || s.channel == 'R' {
                right.push(Biquad::new(b, a));
            }
        }

        HeadphoneEq { left, right }
    }
}

pub struct HeadphoneEq {
    left: Vec<Biquad>,
    right: Vec<Biquad>,
}

impl HeadphoneEq {
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        for x in left.iter_mut() {
            let mut y = *x;
            for f in self.left.iter_mut() {
                y = f.process_sample(y);
            }
            *x = y;
        }

        for x in right.iter_mut() {
            let mut y = *x;
            for f in self.right.iter_mut() {
                y = f.process_sample(y);
            }
            *x = y;
        }
    }
}
