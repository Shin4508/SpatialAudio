# Raspberry Pi 5: `08g_concertgebouw_adaptive` real-time mode

This mode runs one v2 environment continuously: the Concertgebouw-inspired grand hall (`EnvironmentKind::Concertgebouw`) with the adaptive controller enabled. It uses the measured direct HRTFs, baked early reflections, and the stereo FDN late field. It does not render the other comparison files or switch rooms while audio is running.

The executable is `08g_concertgebouw_adaptive`. It is intended for Raspberry Pi 5 with a stereo USB/I2S audio interface and JACK or PipeWire-JACK. The callback uses 256-sample blocks. The room bake happens before streams start; the first implementation is a validation target, so measure callback time and underruns before treating it as production firmware.

## Install on Raspberry Pi 5

```sh
sudo apt update
sudo apt install -y build-essential pkg-config libjack-jackd2-dev libasound2-dev git
git clone https://github.com/Shin4508/SpatialAudio.git
cd SpatialAudio
rustup toolchain install stable
```

The included profile is large. Keep it on local storage and verify that all 144 files exist:

```sh
test "$(find spatial_compare_lab/hrtf_profiles/default -name 'hrtf_*.wav' | wc -l)" -eq 144
```

## Build

```sh
cargo build --release --locked --manifest-path rasberry_pi_v2/Cargo.toml --bin 08g_concertgebouw_adaptive
```

## Start audio

With PipeWire-JACK:

```sh
pw-jack jackd -d dummy -r 48000 -p 256 -n 2
```

On a normal desktop session, PipeWire may already provide JACK; in that case do not start a second JACK server. Confirm the graph and device names:

```sh
pw-link -io
jack_lsp
```

The program uses the default input and output devices, but explicitly requests a 48 kHz stereo float32 stream. This is intentional because your Pi may report 44.1 kHz as its system default while the included HRTFs are 48 kHz. The default can remain 44.1 kHz as long as the codec advertises 48 kHz as a supported rate. Select a different device through the JACK/PipeWire default or an ALSA device profile before launching.

## Run one mode

From the repository root:

```sh
HRTF_PROFILE="$PWD/spatial_compare_lab/hrtf_profiles/default" \
  ./target/release/08g_concertgebouw_adaptive 3.0 30
```

Arguments are optional:

```text
08g_concertgebouw_adaptive [distance_m] [half_angle_deg]
```

Defaults are `3.0` metres and `±30°`, matching the v2 Concertgebouw preset. For example:

```sh
./target/release/08g_concertgebouw_adaptive 2.5 25
```

Distance must be 0.15–10 m and half-angle must be 0–90°. Press `Ctrl-C` to stop. Connect the graph after launch if automatic links are not present:

```sh
jack_lsp
pw-link YOUR_INPUT:capture_FL 08g_concertgebouw_adaptive:input_1
pw-link YOUR_INPUT:capture_FR 08g_concertgebouw_adaptive:input_2
pw-link 08g_concertgebouw_adaptive:output_1 YOUR_HEADPHONES:playback_FL
pw-link 08g_concertgebouw_adaptive:output_2 YOUR_HEADPHONES:playback_FR
```

Port names vary by PipeWire/JACK device. Use `jack_lsp -c` to inspect actual names. If the CPAL client is not visible, run the process under `pw-jack`:

```sh
pw-jack env HRTF_PROFILE="$PWD/spatial_compare_lab/hrtf_profiles/default" \
  ./target/release/08g_concertgebouw_adaptive 3.0 30
```

## Verify before listening

```sh
cargo test --locked --manifest-path spatial_compare_lab_v2/Cargo.toml
cargo check --locked --manifest-path rasberry_pi_v2/Cargo.toml --bin 08g_concertgebouw_adaptive
```

Start at a low headphone volume. Watch `pw-top` or `jack_iodelay` for XRUNs. If the device exposes only 44.1 kHz, select a codec profile that supports 48 kHz or regenerate the complete HRTF profile at 44.1 kHz; do not mix 44.1 kHz audio with the included 48 kHz HRIRs. If the callback misses deadlines, use a larger period (`-p 512`), close other audio clients, and keep the fixed 48 kHz/256-sample configuration while profiling. The callback currently keeps state behind a mutex and the v2 FFT path still allocates temporary vectors; this is deliberate for a first Pi 5 reference and must be optimized before moving the exact design to an MCU.
