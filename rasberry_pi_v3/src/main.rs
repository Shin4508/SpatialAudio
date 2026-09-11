use jack::{
    AudioIn, AudioOut, Client, ClientOptions, Control, NotificationHandler, Port, ProcessHandler,
    ProcessScope,
};
use rasberry_pi_v3::{mode_index, profile::at_rate, Player, Room, MODES};
use spatial_compare_lab_v2::{headphone_eq::HeadphoneEqConfig, hrtf::HrtfProfile};
use std::{
    error::Error,
    io::{self, BufRead},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed},
        Arc,
    },
};

#[derive(Default)]
struct Shared {
    requested: AtomicUsize,
    active: AtomicUsize,
    gain: AtomicU32,
    xruns: AtomicU64,
    limited: AtomicU64,
    invalid: AtomicU64,
    stopped: AtomicBool,
}
struct Notifications {
    shared: Arc<Shared>,
    rate: u32,
}
impl NotificationHandler for Notifications {
    fn xrun(&mut self, _: &Client) -> Control {
        self.shared.xruns.fetch_add(1, Relaxed);
        Control::Continue
    }
    fn sample_rate(&mut self, _: &Client, rate: jack::Frames) -> Control {
        if rate != self.rate {
            self.shared.stopped.store(true, Relaxed);
            Control::Quit
        } else {
            Control::Continue
        }
    }
    unsafe fn shutdown(&mut self, _: jack::ClientStatus, _: &str) {
        self.shared.stopped.store(true, Relaxed);
    }
}
struct Audio {
    inputs: [Port<AudioIn>; 2],
    outputs: [Port<AudioOut>; 2],
    player: Player,
    shared: Arc<Shared>,
    rate: u32,
}
impl ProcessHandler for Audio {
    fn process(&mut self, client: &Client, scope: &ProcessScope) -> Control {
        let l = self.inputs[0].as_slice(scope);
        let r = self.inputs[1].as_slice(scope);
        let [out_l, out_r] = &mut self.outputs;
        let ol = out_l.as_mut_slice(scope);
        let or = out_r.as_mut_slice(scope);
        if self.shared.stopped.load(Relaxed) || client.sample_rate() != self.rate {
            ol.fill(0.0);
            or.fill(0.0);
            self.shared.stopped.store(true, Relaxed);
            return Control::Quit;
        }
        let mode = self.shared.requested.load(Relaxed);
        let gain = f32::from_bits(self.shared.gain.load(Relaxed));
        for i in 0..ol.len() {
            let [a, b] = self.player.frame(l[i], r[i], mode, gain);
            ol[i] = a;
            or[i] = b;
        }
        self.shared.active.store(self.player.active(), Relaxed);
        self.shared.limited.store(self.player.limited, Relaxed);
        self.shared.invalid.store(self.player.invalid, Relaxed);
        Control::Continue
    }
}

fn help() {
    println!("rasberry_pi_v3 [--profile DIR] [--eq FILE] [--mode 06d] [--distance METRES] [--angle DEGREES] [--gain-db -6]\n\
        Commands during playback: list | mode 06a | 06g | next | prev | gain -9 | status | help | quit\n\
        Uses JACK/PipeWire server rate (44100 or 48000 Hz). Default mode=06d; gain=-6 dB.");
}
fn list() {
    for m in MODES {
        println!("  {m}");
    }
}
fn gain_db(s: &str) -> Result<f32, Box<dyn Error>> {
    let db: f32 = s.parse()?;
    if !db.is_finite() || !(-60.0..=0.0).contains(&db) {
        return Err("Gain must be -60 to 0 dB".into());
    }
    Ok(10.0_f32.powf(db / 20.0))
}
fn main() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut profile_path = std::env::var_os("HRTF_PROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("spatial_compare_lab/hrtf_profiles/default"));
    let mut eq_path = root.join("spatial_compare_lab_v2/config/headphone_eq.txt");
    let mut mode = 3;
    let mut distance = None;
    let mut angle = 30.0;
    let mut gain = gain_db("-6")?;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            help();
            list();
            return Ok(());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("Missing value for {arg}"))?;
        match arg.as_str() {
            "--profile" => profile_path = value.into(),
            "--eq" => eq_path = value.into(),
            "--mode" => mode = mode_index(&value).ok_or("Unknown mode; use --help")?,
            "--distance" => distance = Some(value.parse::<f32>()?),
            "--angle" => angle = value.parse::<f32>()?,
            "--gain-db" => gain = gain_db(&value)?,
            _ => return Err(format!("Unknown option {arg}").into()),
        }
    }
    let (client, _) = Client::new("spatial_v3", ClientOptions::NO_START_SERVER)?;
    let rate = client.sample_rate();
    let original = HrtfProfile::load(profile_path.to_str().ok_or("Invalid profile path")?)?;
    println!(
        "JACK server: {rate} Hz, {} frames/period. Profile: {} Hz.",
        client.buffer_size(),
        original.sample_rate
    );
    if original.sample_rate != rate {
        println!("Resampling HRIR filters to {rate} Hz before playback.");
    }
    let profile = at_rate(original, rate)?;
    // Unlike the offline loader's optional flat fallback, explicit missing EQ
    // paths should fail here rather than silently disabling compensation.
    std::fs::read_to_string(&eq_path)?;
    let eq = HeadphoneEqConfig::load_or_flat(eq_path.to_str().ok_or("Invalid EQ path")?)?;
    let mut rooms = Vec::with_capacity(7);
    for (i, name) in MODES.iter().enumerate() {
        println!("Preparing {name}...");
        rooms.push(Room::new(&profile, i, &eq, distance, angle)?);
    }
    let shared = Arc::new(Shared::default());
    shared.requested.store(mode, Relaxed);
    shared.active.store(mode, Relaxed);
    shared.gain.store(gain.to_bits(), Relaxed);
    let audio = Audio {
        inputs: [
            client.register_port("in_l", AudioIn::default())?,
            client.register_port("in_r", AudioIn::default())?,
        ],
        outputs: [
            client.register_port("out_l", AudioOut::default())?,
            client.register_port("out_r", AudioOut::default())?,
        ],
        player: Player::new(rooms, mode, rate, gain),
        shared: shared.clone(),
        rate,
    };
    println!(
        "Client: {}. Connect capture/player → in_l/in_r; out_l/out_r → headphones.",
        client.name()
    );
    let active = client.activate_async(
        Notifications {
            shared: shared.clone(),
            rate,
        },
        audio,
    )?;
    help();
    list();
    println!("Playing {}. Type a command and press Enter.", MODES[mode]);
    for line in io::stdin().lock().lines() {
        if shared.stopped.load(Relaxed) {
            eprintln!("Audio server stopped or rate changed. Restart v3 to rebuild filters.");
            break;
        }
        let line = line?;
        let parts: Vec<_> = line.split_whitespace().collect();
        let requested = shared.requested.load(Relaxed);
        let next = match parts.as_slice() {
            [] => continue,
            ["quit" | "exit"] => break,
            ["help"] => {
                help();
                continue;
            }
            ["list"] => {
                list();
                continue;
            }
            ["status"] => {
                println!("active={} requested={} gain={:.1} dB xruns={} limited_samples={} invalid_samples={} cpu={:.1}%",
                    MODES[shared.active.load(Relaxed)],MODES[requested],20.0*f32::from_bits(shared.gain.load(Relaxed)).log10(),
                    shared.xruns.load(Relaxed),shared.limited.load(Relaxed),shared.invalid.load(Relaxed),active.as_client().cpu_load());
                continue;
            }
            ["gain", db] => {
                match gain_db(db) {
                    Ok(g) => {
                        shared.gain.store(g.to_bits(), Relaxed);
                        println!("Gain {db} dB");
                    }
                    Err(e) => eprintln!("{e}"),
                };
                continue;
            }
            ["next"] => Some((requested + 1) % 7),
            ["prev"] => Some((requested + 6) % 7),
            ["mode", name] | [name] => mode_index(name),
            _ => None,
        };
        if let Some(n) = next {
            shared.requested.store(n, Relaxed);
            println!("Requested {} (100 ms crossfade)", MODES[n]);
        } else {
            eprintln!("Unknown command or mode. Type help or list.");
        }
    }
    active.deactivate()?;
    Ok(())
}
