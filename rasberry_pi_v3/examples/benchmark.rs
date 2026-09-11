//! Run on the Pi itself; host timings do not predict Raspberry Pi deadlines.
use rasberry_pi_v3::{profile::at_rate, Player, Room, BLOCK, MODES};
use spatial_compare_lab_v2::{headphone_eq::HeadphoneEqConfig, hrtf::HrtfProfile};
use std::{error::Error, path::PathBuf, time::Instant};
fn main() -> Result<(), Box<dyn Error>> {
    let rate = std::env::args()
        .nth(1)
        .unwrap_or("44100".into())
        .parse::<u32>()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let path = std::env::var_os("HRTF_PROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("spatial_compare_lab/hrtf_profiles/default"));
    let profile = at_rate(
        HrtfProfile::load(path.to_str().ok_or("Invalid profile path")?)?,
        rate,
    )?;
    let eq = HeadphoneEqConfig::load_or_flat(
        root.join("spatial_compare_lab_v2/config/headphone_eq.txt")
            .to_str()
            .unwrap(),
    )?;
    let rooms = (0..7)
        .map(|m| Room::new(&profile, m, &eq, None, 30.0))
        .collect::<Result<Vec<_>, _>>()?;
    let mut player = Player::new(rooms, 0, rate, 0.5);
    println!(
        "{rate} Hz, {BLOCK} frames: {:.3} ms block budget. Includes mode-switch cost.",
        1000.0 * BLOCK as f64 / rate as f64
    );
    for (m, name) in MODES.iter().enumerate() {
        let mut worst = 0.0_f64;
        let start = Instant::now();
        for b in 0..400 {
            let block_start = Instant::now();
            for i in 0..BLOCK {
                let x = ((b * BLOCK + i) as f32 * 0.03).sin() * 0.1;
                std::hint::black_box(player.frame(x, -x, m, 0.5));
            }
            worst = worst.max(block_start.elapsed().as_secs_f64());
        }
        println!(
            "{name}: mean={:.3} ms max={:.3} ms",
            start.elapsed().as_secs_f64() * 1000.0 / 400.0,
            worst * 1000.0
        );
    }
    Ok(())
}
