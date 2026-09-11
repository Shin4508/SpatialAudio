# Spatial Sound Comparison Lab

## Web interface

Upload music, edit environment parameters, and play or download spatial audio using the HTML/CSS/JavaScript frontend and Rust backend:

```sh
cargo run --release --locked --manifest-path web/Cargo.toml
```

Open <http://127.0.0.1:3000>. See [web setup and controls](web/README.md) for supported formats and HRTF configuration.

For a single-mode electronics bring-up, see the [electronic module candidate showcase](electronic_module/CANDIDATES.md) and run the [single-mode runner](electronic_module/README.md).

For Raspberry Pi 5 live processing with terminal switching between all seven fixed `06` rooms, see [rasberry_pi_v3 setup and commands](rasberry_pi_v3/README.md). It supports 44.1 and 48 kHz JACK/PipeWire servers.

An offline Rust laboratory for comparing binaural rendering strategies on the same stereo recording. The main project, [`spatial_compare_lab`](spatial_compare_lab), renders a dry reference, unmodified-HRTF virtual loudspeakers, headphone compensation, six acoustic environments, optional personal HRTFs, and content-adaptive variants.

The goal is to make spatial-audio decisions audible and measurable before moving the processing to a real-time or embedded target such as the ESP32-S3.

## What it compares

The renderer treats the input channels as virtual loudspeakers at approximately -30° and +30°. Each loudspeaker is rendered to both ears through a 72-direction horizontal HRTF profile.

The comparison stages are:

1. Dry stereo input.
2. Full-strength HRTF virtual loudspeakers.
3. Optional per-ear parametric headphone EQ.
4. Baked early reflections and late reverberation for six environments while keeping the direct HRTFs unchanged.
5. An optional personal HRTF profile.
6. Content-adaptive room rendering based on channel correlation and Mid/Side energy.

```text
Stereo WAV
  ├─> dry reference
  └─> two virtual loudspeakers
        ├─> direct HRTF paths (speaker × ear)
        ├─> image-source early reflections
        ├─> stereo FDN late field
        ├─> optional headphone EQ
        └─> clipping prevention
              └─> 32-bit float comparison WAVs
```

## Why the renderer is bake-oriented

Room geometry and HRTF lookup are performed before audio blocks are processed. For each environment, the room baker creates four direct filters and four early-reflection filters:

- left speaker to left ear
- left speaker to right ear
- right speaker to left ear
- right speaker to right ear

This separates expensive geometry work from the runtime signal path. The desktop implementation currently uses FFT convolution, while the baked filters and lightweight adaptive controller are intended to be portable to an embedded FIR/FFT implementation later.

## Requirements

- Rust and Cargo
- A stereo WAV input
- An HRTF profile whose sample rate matches the input

The included default profile is tagged as 48 kHz, so use a 48 kHz input unless you regenerate the profile with the correct sample-rate metadata.

## Quick start

Place a stereo input at `spatial_compare_lab/output.wav`, then run the program from that directory:

```bash
cd spatial_compare_lab
cargo run --release
```

You can also provide all paths explicitly:

```bash
cd spatial_compare_lab
cargo run --release -- \
  /path/to/input.wav \
  hrtf_profiles/default \
  hrtf_profiles/personal \
  config/headphone_eq.txt
```

Arguments are positional and ordered as follows:

| Position | Default | Purpose |
| --- | --- | --- |
| 1 | `output.wav` | Stereo source WAV |
| 2 | `hrtf_profiles/default` | Complete default HRTF profile |
| 3 | `hrtf_profiles/personal` | Optional personal HRTF profile |
| 4 | `config/headphone_eq.txt` | Per-ear headphone EQ configuration |

All output is written to `spatial_compare_lab/compare_out/` when the command is run from the project directory.

## Comparison outputs

Every run creates these baseline files:

```text
compare_out/00_dry.wav
compare_out/01_reference_2speaker.wav
compare_out/02_hrtf_intensity_090.wav
compare_out/05_headphone_comp.wav
```

It also creates fixed and adaptive renders for each environment:

| Prefix | Environment |
| --- | --- |
| `06a` / `08a` | Open air |
| `06b` / `08b` | Street |
| `06c` / `08c` | Recording studio |
| `06d` / `08d` | Wood jazz club |
| `06e` / `08e` | Piano hall |
| `06f` / `08f` | Theater |

The `06` files use fixed room settings. The `08` files add content adaptation. If `hrtf_profiles/personal/hrtf_left_0.wav` exists and the personal profile is complete, the renderer also writes:

```text
compare_out/07_personal_jazz_club.wav
```

For useful A/B listening, begin with `00`, then compare `01`, `02`, and `05` before moving through matching `06` and `08` environment pairs.

## Rendering stages

### Virtual loudspeaker reference

`01_reference_2speaker.wav` uses the measured HRTFs without spectral blending, room effects, or headphone EQ. It is the clean binaural baseline:

```text
Input L -> virtual speaker at -30° -> both ears
Input R -> virtual speaker at +30° -> both ears
```

### 90% HRTF intensity

`02_hrtf_intensity_090.wav` blends each direct HRIR by 10% toward an energy-matched impulse at its principal arrival time. This comparison softens HRTF spectral coloration slightly while retaining the main timing and level cues. Room renders continue to use the original, unmodified HRTFs.

### Headphone compensation

`05_headphone_comp.wav` enables the EQ engine. The supplied configuration is intentionally flat. Add filters only from a measured response or a known correction target.

Each non-comment line in `config/headphone_eq.txt` has this format:

```text
CHANNEL TYPE FREQUENCY_HZ GAIN_DB Q
```

`CHANNEL` is `L`, `R`, or `B` for both ears. Supported filter types are `peaking`, `low_shelf`, and `high_shelf`.

### Acoustic environments

The room baker uses source/listener geometry, frequency-dependent surface absorption, scattering, air loss, and image sources. It selects the strongest early events within each preset's time window, then derives or applies late-field parameters for the stereo feedback delay network.

Environment definitions live in `spatial_compare_lab/src/room.rs` and include open-air, street, studio, jazz-club, piano-hall, and theater presets.

### Content adaptation

Every 256 samples, the adaptive controller estimates:

- left/right correlation
- Mid energy
- Side energy

Coherent, center-heavy material stays more focused and dry. Wide or diffuse material receives more early-reflection gain, late-field send, and a small dry anchor. This analysis uses only energy and dot-product calculations—no source separation or neural network is required.

## HRTF profiles

The included profile contains 144 mono 32-bit float WAV files: left and right HRIRs for 72 horizontal directions.

```text
hrtf_profiles/default/
  hrtf_left_0.wav
  hrtf_right_0.wav
  ...
  hrtf_left_71.wav
  hrtf_right_71.wav
```

The direction layout is:

- 200 samples per HRIR
- 72 directions at 5° intervals
- index `0` is straight ahead
- index `6` is +30°
- index `66` is -30°

The input and HRTF sample rates must match; the renderer exits on a mismatch.

### Personal HRTFs

Place a complete profile with the same naming scheme in `spatial_compare_lab/hrtf_profiles/personal/`. When detected, it is used for the personal jazz-club render and all adaptive environment renders.

### Convert a MAT dataset

The conversion utility expects `left` and `right` arrays shaped as impulse samples × directions. It requires Python with NumPy, SciPy, and SoundFile.

```bash
cd spatial_compare_lab
python tools/mat_hrtf_to_wav.py /path/to/small_pinna_final.mat \
  --output hrtf_profiles/default \
  --sr 48000
```

The MAT file does not provide a sample rate, so set `--sr` to the dataset's real rate rather than assuming 48 kHz.

## Development and tuning

Check the main project with:

```bash
cargo check --locked --manifest-path spatial_compare_lab/Cargo.toml
```

Useful parameters to change one at a time include:

- speaker distance and azimuth in `spatial_compare_lab/src/room.rs`
- room dimensions and surface materials in `spatial_compare_lab/src/room.rs`
- early, late, and dry ranges in `spatial_compare_lab/src/adaptive.rs`
- FDN decay, damping, predelay, and output gain

A convincing early field may change externalization and room impression without producing an obvious echo, so evaluate localization, timbre, and distance as well as reverberance.

## Repository layout

| Path | Purpose |
| --- | --- |
| `spatial_compare_lab/` | Primary offline comparison renderer |
| `spatial_compare_lab_sofa/` | Separate comparison renderer using the AKO536 Meta SS2 SOFA HRTF profile |
| `spatial_compare_lab/src/` | HRTF, room, convolution, EQ, adaptive, and audio modules |
| `spatial_compare_lab/hrtf_profiles/` | Default and optional personal HRIR sets |
| `spatial_compare_lab/config/` | Headphone EQ configuration |
| `spatial_compare_lab/tools/` | MAT-to-WAV conversion utility |
| `rust_impl/` | Earlier offline rendering prototype |
| `rasberry_pi_v2/` | JACK-based real-time Raspberry Pi/Linux prototype |
| `ver2/` | Additional real-time experiment |

Generated render directories, Cargo build artifacts, virtual environments, and the large local HRTF database are excluded by `.gitignore`.

## License

Project code is licensed under the [Apache License 2.0](LICENSE), using the [official license text](https://www.apache.org/licenses/LICENSE-2.0.txt). Third-party dependencies, HRTF datasets, and music retain their own licenses.
