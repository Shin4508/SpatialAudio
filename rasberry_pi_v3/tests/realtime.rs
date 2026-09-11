use rasberry_pi_v3::{mode_index, profile::at_rate, Player, Room, BLOCK, MODES};
use spatial_compare_lab_v2::{
    audio::load_stereo_wav, headphone_eq::HeadphoneEqConfig, hrtf::HrtfProfile, render_and_write,
    room::EnvironmentKind, RenderOptions,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    path::PathBuf,
};

thread_local! { static COUNT: Cell<Option<usize>> = const { Cell::new(None) }; }
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = COUNT.try_with(|c| {
            if let Some(n) = c.get() {
                c.set(Some(n + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let _ = COUNT.try_with(|c| {
            if let Some(n) = c.get() {
                c.set(Some(n + 1));
            }
        });
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;

fn fixtures(rate: u32) -> (HrtfProfile, HeadphoneEqConfig) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let profile = HrtfProfile::load(
        root.join("spatial_compare_lab/hrtf_profiles/default")
            .to_str()
            .unwrap(),
    )
    .unwrap();
    let eq = HeadphoneEqConfig::load_or_flat(
        root.join("spatial_compare_lab_v2/config/headphone_eq.txt")
            .to_str()
            .unwrap(),
    )
    .unwrap();
    (at_rate(profile, rate).unwrap(), eq)
}

#[test]
fn every_fixed_mode_matches_offline_v2_at_both_rates_including_tail() {
    let dir = tempfile::tempdir().unwrap();
    for rate in [44100, 48000] {
        let (profile, eq) = fixtures(rate);
        let l: Vec<_> = (0..BLOCK * 3 + 17)
            .map(|i| (i as f32 * 0.071).sin() * 0.1)
            .collect();
        let r: Vec<_> = (0..l.len())
            .map(|i| (i as f32 * 0.137).cos() * 0.07)
            .collect();
        for (mode, environment) in EnvironmentKind::all().into_iter().enumerate() {
            let path = dir.path().join("reference.wav");
            render_and_write(
                path.to_str().unwrap(),
                &l,
                &r,
                rate,
                &profile,
                &eq,
                RenderOptions {
                    direct_hrtf_intensity: 1.0,
                    environment: Some(environment),
                    headphone_eq: true,
                    adaptive: false,
                    distance_m: None,
                    angle_deg: 30.0,
                },
            )
            .unwrap();
            let (expected_l, expected_r, _) = load_stereo_wav(path.to_str().unwrap()).unwrap();
            let mut room = Room::new(&profile, mode, &eq, None, 30.0).unwrap();
            let mut max_error = 0.0_f32;
            for base in (0..expected_l.len()).step_by(BLOCK) {
                let bl = std::array::from_fn(|i| l.get(base + i).copied().unwrap_or(0.0));
                let br = std::array::from_fn(|i| r.get(base + i).copied().unwrap_or(0.0));
                let mut out = [[0.0; BLOCK]; 2];
                room.process(&bl, &br, &mut out);
                for i in 0..BLOCK.min(expected_l.len() - base) {
                    max_error = max_error
                        .max((out[0][i] - expected_l[base + i]).abs())
                        .max((out[1][i] - expected_r[base + i]).abs());
                }
            }
            assert!(max_error < 3e-5, "{} at {rate}: {max_error}", MODES[mode]);
        }
    }
}

#[test]
fn switching_every_room_is_finite_and_does_not_allocate_or_free() {
    let (profile, eq) = fixtures(44100);
    let rooms = (0..7)
        .map(|m| Room::new(&profile, m, &eq, None, 30.0).unwrap())
        .collect();
    let mut player = Player::new(rooms, 0, 44100, 0.5);
    let mut finite = true;
    COUNT.with(|c| c.set(Some(0)));
    for mode in (0..7).chain((0..7).rev()) {
        for i in 0..8192 {
            let x = (i as f32 * 0.02).sin() * 0.05;
            let out = player.frame(x, -x, mode, 0.5);
            finite &= out.iter().all(|v| v.is_finite() && v.abs() <= 0.98);
        }
    }
    let count = COUNT.with(|c| c.replace(None).unwrap());
    assert_eq!(count, 0, "DSP/switching touched the allocator");
    assert!(finite);
    assert_eq!(player.active(), 0);
    assert_eq!(player.invalid, 0);
}

#[test]
fn adapter_has_exactly_one_block_of_latency_and_preserves_stereo() {
    let (profile, eq) = fixtures(48000);
    let rooms = (0..7)
        .map(|m| Room::new(&profile, m, &eq, None, 30.0).unwrap())
        .collect();
    let mut player = Player::new(rooms, 3, 48000, 1.0);
    let mut room = Room::new(&profile, 3, &eq, None, 30.0).unwrap();
    let l = std::array::from_fn(|i| if i == 255 { 0.01 } else { 0.0 });
    let r = std::array::from_fn(|i| if i == 0 { 0.02 } else { 0.0 });
    let mut out = [[0.0; BLOCK]; 2];
    room.process(&l, &r, &mut out);
    for i in 0..BLOCK {
        assert_eq!(player.frame(l[i], r[i], 3, 1.0), [0.0; 2]);
    }
    for i in 0..BLOCK {
        let v = player.frame(0.0, 0.0, 3, 1.0);
        assert!((v[0] - out[0][i]).abs() < 1e-7 && (v[1] - out[1][i]).abs() < 1e-7);
    }
    assert_eq!(mode_index("06g"), Some(6));
    assert_eq!(mode_index("concertgebouw"), Some(6));
    assert_eq!(mode_index("08g"), None);
}
