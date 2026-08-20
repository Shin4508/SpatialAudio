# Spatial Sound

A Rust experiment for converting stereo audio into a binaural spatial mix with HRTF convolution, Mid/Side processing, room reflections, reverb, distance filtering, and headphone crossfeed.

The project currently has two main implementations:

| Directory | Mode | Purpose |
| --- | --- | --- |
| `rust_impl` | Offline | Processes a stereo WAV file and writes a rendered WAV file |
| `rasberry_pi_v2` | Real time | Processes live stereo audio through JACK on a Raspberry Pi/Linux system |

Both implementations are prototypes. They use fixed settings and file names defined in their respective `main.rs` files.

## How it works

The renderer converts the stereo input into Mid and Side signals:

```text
Mid  = (Left + Right) / 2
Side = (Left - Right) / 2
```

The Mid signal represents the center image, while the Side signal represents stereo width. Different HRTF impulse responses are applied to these components to position them around the listener.

The general processing flow is:

```text
Stereo input
  -> Mid/Side encoding
  -> dynamic EQ and crossover
  -> HRTF convolution
  -> early reflections
  -> late reverb
  -> crossfeed
  -> soft clipping
  -> binaural stereo output
```

FFT overlap-save convolution is implemented with `rustfft`. WAV files are read with `hound`, and live audio is handled by `cpal`.

## Offline renderer: `rust_impl`

The offline implementation reads a stereo WAV file, processes it in blocks of 128 frames, and writes the result to `final_rt_sim.wav`.

Its processing chain includes:

- Mid/Side encoding
- adaptive low-frequency shelf EQ
- 120 Hz low/high crossover
- HRTF convolution for the direct sound
- 18 ms and 23 ms delayed rear reflections
- low-pass filtering of the reflected field
- 70% processed and 30% original signal blending
- 0.5 ms crossfeed
- `tanh` soft clipping

### Required files

The executable looks for the following files in its working directory:

```text
output.wav
hrtf_left_9.wav
hrtf_right_9.wav
hrtf_left_63.wav
hrtf_left_40.wav
hrtf_right_32.wav
```

`output.wav` must be stereo 16-bit PCM. The HRTF files are read as mono 32-bit floating-point WAV files.

The included audio files are stored in subdirectories, so copy them into `rust_impl` before running:

```bash
cd rust_impl
cp src/output.wav .
cp hrtf_wav/*.wav .
cargo run --release
```

Output:

```text
rust_impl/final_rt_sim.wav
```

If an HRTF file cannot be loaded, the offline renderer substitutes a silent impulse response. If `output.wav` is missing or has an unsupported sample format, the program exits.

The final incomplete block is not processed, so the last 127 frames or fewer may be silent.

## Real-time renderer: `rasberry_pi_v2`

The real-time implementation is designed for a Raspberry Pi or another Linux system running JACK, optionally through PipeWire's JACK compatibility layer.

It receives live stereo input, processes the audio inside the output callback, and sends the resulting binaural signal to the default JACK output device.

Compared with `rust_impl`, this version adds:

- a fourth-order Linkwitz-Riley crossover at 100 Hz
- smoothed Mid/Side width control
- distance-based volume and spectral attenuation
- adaptive early-reflection and late-reverb levels
- separate Mid and Side reflection paths
- a four-line feedback delay network reverb
- improved full-band direct HRTF rendering

### HRTF layout

Place these files directly inside `rasberry_pi_v2`:

| File | Use |
| --- | --- |
| `hrtf_left_0.wav` | Front source, left ear |
| `hrtf_right_0.wav` | Front source, right ear |
| `hrtf_left_63.wav` | Left-front Side source |
| `hrtf_right_9.wav` | Right-front Side source |
| `hrtf_left_40.wav` | Left-rear reflection |
| `hrtf_right_32.wav` | Right-rear reflection |

The HRTFs must be mono 32-bit floating-point WAV files. Their sample rate should match the JACK device sample rate because the engine does not perform resampling.

The repository does not currently contain `hrtf_left_0.wav` or `hrtf_right_0.wav`. Missing HRTFs are replaced with silence, which means missing front responses will remove most of the centered direct field.

### System requirements

- A Raspberry Pi or Linux computer
- Rust with Edition 2024 support
- JACK development headers
- A running JACK server or PipeWire JACK compatibility layer
- `pw-link` for the automatic port-routing script
- Stereo input and output devices

Example packages for Raspberry Pi OS or Debian:

```bash
sudo apt update
sudo apt install build-essential pkg-config libjack-jackd2-dev pipewire-jack pipewire-bin
```

### Build and run

Start the JACK-compatible audio server, place the HRTF files in the crate directory, and run:

```bash
cd rasberry_pi_v2
cargo run --release
```

The process runs until interrupted with `Ctrl+C`.

### Hardware configuration

The current code is configured for the original development setup rather than a generic Raspberry Pi installation. Before running it on different hardware, review `rasberry_pi_v2/src/main.rs`.

The following values are hard-coded:

- JACK is selected explicitly as the CPAL host.
- Audio is assumed to be interleaved stereo `f32`.
- The target virtual distance is fixed at 1.5 metres.
- The HRTF engine is initialized with a 256-frame block size.
- Bluetooth input ports are discovered using a `bluez_input` name pattern.
- Input is connected to `cpal_client_in:in_0` and `cpal_client_in:in_1`.
- A specific Creative Sound Blaster output connection is disconnected.

Update or remove the embedded `pw-link` script if those port names do not match your system. You can inspect available ports with:

```bash
pw-link -io
```

The output callback uses the audio server's callback length, while the convolution engines are initialized for 256 frames. Configure JACK to use the expected buffer size before running the current implementation.

## Real-time signal path

The Raspberry Pi engine processes each block in the following order:

1. Encode stereo input into Mid and Side.
2. Apply distance attenuation, proximity correction, and high-frequency air absorption.
3. Apply smoothed low-frequency dynamic EQ.
4. Keep Mid bass below 100 Hz centered.
5. Convolve Mid with the front HRTFs.
6. Convolve Side with the left-front and right-front HRTFs.
7. Generate delayed rear HRTF reflections from both Mid and Side.
8. Estimate early-reflection and reverb levels from the block's energy distribution.
9. Generate a decorrelated late field with a four-delay FDN reverb.
10. Add short crossfeed and apply a `tanh` output limiter.

## Important limitations

- There are no command-line options or runtime controls.
- HRTF and input sample rates are not validated or converted.
- WAV input formats are fixed by the code.
- Missing HRTFs produce silence instead of stopping execution.
- The live engine assumes stereo input and output without checking the channel count.
- The real-time callback allocates multiple vectors and performs FFT processing, so it is not yet strictly real-time safe.
- `rasberry_pi_v2` only builds when CPAL exposes its JACK backend. It will not build as written on macOS or on Linux without JACK support.

## Development

Check the offline implementation with:

```bash
cargo check --manifest-path rust_impl/Cargo.toml
```

On a JACK-enabled Linux machine, check the real-time implementation with:

```bash
cargo check --manifest-path rasberry_pi_v2/Cargo.toml
```

The directory name `rasberry_pi_v2` preserves the spelling used by the existing project.
