# Raspberry Pi 5: switch all fixed 06 rooms during live playback

`rasberry_pi_v3` runs the seven **fixed 06 modes** from `spatial_compare_lab_v2`. Type commands in its terminal to switch rooms without restarting the audio stream. The default is `06d_jazz_club` with a -6 dB output gain.

| Command | Fixed environment |
| --- | --- |
| `06a` | Open air |
| `06b` | Street |
| `06c` | Recording studio |
| `06d` | Wood jazz club |
| `06e` | Piano hall |
| `06f` | Theater |
| `06g` | Concertgebouw-inspired grand hall |

This is the fixed 06 processing path: full direct HRTFs, room reflections, stereo FDN reverb and headphone EQ. The adaptive 08 controller is disabled. Each preset keeps its own default distance unless you supply `--distance`; speaker half-angle defaults to 30 degrees.

## 1. Install on the Pi

Use Raspberry Pi OS 64-bit and a stereo USB/I2S interface, or route a software music player into the JACK graph. A physical input is not necessary for software playback. These commands run **on the Pi**; the binary built on a Mac cannot be copied to the Pi and executed.

```sh
sudo apt update
sudo apt install -y build-essential pkg-config libjack-jackd2-dev pipewire-jack pipewire-bin git curl
```

For `jack_lsp`, `jack_connect` and `jack_disconnect`, install the package for your OS version (check `cat /etc/os-release`):

```sh
# Raspberry Pi OS based on Debian Trixie:
sudo apt install -y jack-example-tools

# Raspberry Pi OS based on Debian Bookworm, use this instead:
# sudo apt install -y jackd2
```

These utilities are in [jack-example-tools on Trixie](https://packages.debian.org/trixie/arm64/jack-example-tools/filelist) and [jackd2 on Bookworm](https://packages.debian.org/bookworm/arm64/jackd2/filelist). Installing JACK tools does not require starting a separate JACK server when using PipeWire.

If Rust is not installed, use the [official Rust installer](https://rust-lang.org/install.html):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
rustup update stable
```

For a new checkout:

```sh
git clone https://github.com/Shin4508/SpatialAudio.git
cd SpatialAudio
```

For your existing checkout, open the repository root and run `git pull --ff-only`.

## 2. Build

All commands below start from the repository root. Cargo places this crate's binary in **`rasberry_pi_v3/target/release/`**.

```sh
cargo build --release --locked --manifest-path rasberry_pi_v3/Cargo.toml
./rasberry_pi_v3/target/release/rasberry_pi_v3 --help
```

Default files are resolved from the repository location at build time:

- HRTFs: `spatial_compare_lab/hrtf_profiles/default` (144 mono HRIR WAVs).
- EQ: `spatial_compare_lab_v2/config/headphone_eq.txt` (supplied flat).

Keep the source checkout in place, or pass `--profile` and `--eq` after moving the binary. `HRTF_PROFILE` also works; `--profile` takes precedence. A missing or malformed profile stops startup. A missing EQ file also stops startup.

## 3. Start at your Pi's 44.1 kHz rate

With an existing PipeWire desktop session, inspect the graph and start v3:

```sh
pw-metadata -n settings 0
pw-jack jack_lsp
PIPEWIRE_LATENCY=256/44100 pw-jack ./rasberry_pi_v3/target/release/rasberry_pi_v3
```

[`pw-jack`](https://docs.pipewire.org/page_man_pw-jack_1.html) loads PipeWire's JACK libraries for the application. [`PIPEWIRE_LATENCY`](https://docs.pipewire.org/page_man_pipewire-jack_conf_5.html) requests a buffer duration; the startup message reports the actual server sample rate and period. It does not force the server to 44.1 kHz. Do not launch `pw-jack jackd -d dummy` for playback: a dummy server has no physical audio interface.

**44,100 Hz is 44.1 kHz.** v3 follows the JACK server at either 44,100 or 48,000 Hz. If the bundled 48 kHz profile differs from the server rate, v3 resamples all HRIR filters once before baking rooms. Audio is then processed directly at the server rate. There is no need to change your desktop's default to 48 kHz or manually retag the WAV files. Other sample rates are rejected.

At 48 kHz, the measured HRIRs are used unchanged. At 44.1 kHz, windowed-sinc conversion preserves timing in seconds and scales FIR coefficients to preserve filter gain; it is an approximation of the original filter. Negative-time interpolation lobes are truncated to retain causality. This is not a new measured HRTF dataset.

You can select the first room and level at launch:

```sh
PIPEWIRE_LATENCY=256/44100 pw-jack ./rasberry_pi_v3/target/release/rasberry_pi_v3 \
  --mode 06g --gain-db -9
```

Wait until all seven rooms are prepared. The program prints its actual client name, normally `spatial_v3`. If another copy already exists, JACK may suffix that name; use the printed name below.

## 4. Connect the music and headphones

Leave v3 running in terminal 1. In terminal 2, list actual ports:

```sh
pw-jack jack_lsp -c
```

v3 registers four predictable port names:

```text
spatial_v3:in_l     stereo source left → input
spatial_v3:in_r     stereo source right → input
spatial_v3:out_l    output → headphone left
spatial_v3:out_r    output → headphone right
```

If your interface exposes `system:capture_1/2` and `system:playback_1/2`, use:

```sh
pw-jack jack_connect system:capture_1 spatial_v3:in_l
pw-jack jack_connect system:capture_2 spatial_v3:in_r
pw-jack jack_connect spatial_v3:out_l system:playback_1
pw-jack jack_connect spatial_v3:out_r system:playback_2
```

PipeWire often uses longer device names. Replace the four `system:*` names with the exact capture/player-output and playback port names from `jack_lsp`. For a software music player, connect its left/right output ports to `in_l/in_r`, and disconnect its direct links to the headphones with `pw-jack jack_disconnect SOURCE_PORT HEADPHONE_PORT` to avoid hearing dry and processed audio together. A graph tool such as QjackCtl can also make these connections. v3 does not guess or automatically change your routing.

```text
Stereo interface capture / music player
                 ↓
       spatial_v3:in_l / in_r
                 ↓
        selected fixed 06 room
                 ↓
      spatial_v3:out_l / out_r
                 ↓
             Headphones
```

## 5. Change rooms by typing

In terminal 1, type one command and press **Enter**:

```text
list
mode 06a
06d
mode 06g
next
prev
gain -9
status
quit
```

| Command | Behavior |
| --- | --- |
| `list` | Show all seven modes |
| `mode 06c`, `06c`, `recording_studio`, or `06c_recording_studio` | Select the same recording-studio mode |
| `next` / `prev` | Cycle through 06a–06g, wrapping at the ends |
| `gain -9` | Set output gain in dB, allowed range -60 to 0; smoothed over about 20 ms |
| `status` | Active/requested mode, gain, JACK XRUNs, limited/invalid sample counts, JACK graph CPU load |
| `help` | Show commands and startup options |
| `quit` / `exit` | Deactivate the client and exit |

Switches use a 100 ms linear crossfade. The incoming room starts with cleared delay/EQ history and builds its reverb from live input; the outgoing tail fades out. This changes the acoustic scene without a stream restart. During a fade, at most two rooms process audio. If you type more switches during that fade, the latest requested mode is applied after the current fade finishes. `status` distinguishes the request from the room that has completed its fade. `08g` is intentionally not a valid fixed-06 mode.

Standard-input EOF also stops the application. Keep the terminal/SSH session open for interactive control. A server shutdown or sample-rate change stops audio; press Enter if the terminal is waiting, then restart v3 so its filters use the new rate.

## Startup options

```sh
pw-jack ./rasberry_pi_v3/target/release/rasberry_pi_v3 \
  --profile "$PWD/spatial_compare_lab/hrtf_profiles/default" \
  --eq "$PWD/spatial_compare_lab_v2/config/headphone_eq.txt" \
  --mode 06d --distance 1.5 --angle 30 --gain-db -6
```

`--distance` is an optional shared speaker-distance override (0.15–10 m). `--angle` is the speaker half-angle (0–90°). Both must fit every room's geometry because all modes are baked at startup. Omit `--distance` to keep the offline preset distances. Changing these two geometry options requires restarting v3; mode and output gain are live commands.

The default -6 dB master gain provides headroom, followed by a final ±0.98 sample clamp. `limited_samples` reports clamp activity; lower gain if it keeps increasing. This is not loudness matching or the offline comparison batch's common peak normalization: future live samples are unknown. Start at a modest headphone volume when testing input routing.

## Direct JACK/ALSA alternative

Use this only if you run a standalone JACK server rather than a PipeWire session owning the interface. Install `jackd2` and `alsa-utils`; find the interface with `aplay -l` / `arecord -l`. Replace `USB` with its ALSA card identifier:

```sh
jackd -R -d alsa -d hw:USB -r 44100 -p 256 -n 3
```

In another terminal, launch `./rasberry_pi_v3/target/release/rasberry_pi_v3` without `pw-jack`, and use `jack_lsp` / `jack_connect` without the wrapper. Hardware must support the requested full-duplex rate. v3 receives one synchronized JACK callback, so there are no separate audio-device queues or unhandled device-clock drift inside the application.

## Check timing on the Pi

The DSP uses 256-frame blocks, adding exactly 256 frames of buffering (5.80 ms at 44.1 kHz; 5.33 ms at 48 kHz), plus the server/device latency and filter delays. Other JACK period sizes are accepted by the fixed-size adapter. Eight partitioned convolvers per room use 512-point FFTs, with FFT scratch and history preallocated. The callback does no DSP allocation/freeing, file I/O, logging or mutex locking. Mode selection and gain use atomics.

Run this benchmark **on your Pi 5** before judging its headroom:

```sh
cargo run --release --locked --no-default-features \
  --manifest-path rasberry_pi_v3/Cargo.toml --example benchmark -- 44100
```

It prints mean and maximum block processing times, including switches, for all modes. It uses synthetic audio and does not include JACK scheduling/device costs. The 256-frame budget at 44.1 kHz is about 5.80 ms; allow margin for scheduling and the two-room transition. Use `status` and `pw-top` during actual playback to watch XRUNs/CPU. If XRUNs accumulate, try requesting `PIPEWIRE_LATENCY=512/44100` on restart, reduce other load, and check Pi cooling. This is a request; verify the actual period reported at startup. Larger periods provide scheduling slack but do not reduce the average DSP work per second.

## Verification and limits

```sh
cargo test --release --locked --manifest-path rasberry_pi_v3/Cargo.toml
cargo test --locked --manifest-path spatial_compare_lab_v2/Cargo.toml
```

For DSP tests on a machine without JACK development files, add `--no-default-features` to the first command. Tests require the bundled HRTF profile in the repository.

Tests compare all seven modes to the offline v2 engine at 44.1 and 48 kHz, including finite reverb tails, before the live master gain/clamp. At 44.1 kHz both engines use the same converted profile. Additional tests check convolution against a direct sum, resampled impulse timing/gain, stereo buffer delay, switching all modes, and zero allocator activity during playback/switching. These are numerical software tests; Pi 5 hardware latency, audio routing and uninterrupted playback still require measurement on your Pi.

JACK integration uses the [Rust JACK API](https://docs.rs/jack/0.13.5/jack/). Project code is Apache-2.0; imported HRTF datasets and dependencies retain their own licenses.
