# Spatial Compare Lab

A separate comparison project for experimenting with:

1. **Reference renderer** — normal stereo L/R becomes two virtual loudspeakers at -30°/+30°, each rendered to both ears.
2. **HRTF Intensity** — keeps the main ITD/ILD anchor while reducing HRTF spectral coloration.
5. **Headphone compensation** — optional per-ear PEQ stage. Flat by default until you provide a measured headphone response.
6. **Room physics** — rectangular-room image sources generate early reflections; the strongest 6 events per speaker are baked into BRIRs. Late field is a small FDN after 55 ms.
7. **Personal HRTF** — drop another complete 72-direction profile into `hrtf_profiles/personal/`.
8. **Content-adaptive rendering** — cheap L/R correlation + Mid/Side energy analysis adjusts ER, FDN send, and a small dry anchor.

This is intentionally a **separate project** from the current 6ER renderer so A/B comparison is easy.

## Why this layout is ESP32-S3 friendly

The expensive geometry/HRTF work belongs in the **baker**, not in the eventual embedded runtime.

The room stage creates four fixed filters:

- left speaker -> left ear
- left speaker -> right ear
- right speaker -> left ear
- right speaker -> right ear

Direct and early-reflection filters are separated so a cheap runtime controller can change ER level without ray tracing.

The current desktop version uses FFT convolution. Later, the same baked filters can be exported to binary and implemented with ESP-DSP FIR/FFT routines. Content adaptation only needs dot-products/energy calculations and a few smoothed gains.

## HRTF dataset

`hrtf_profiles/default/` is generated from the uploaded `small_pinna_final.mat`:

- 200 samples per HRIR
- 72 horizontal directions
- index 0 = front
- 5° per index
- -30° = index 66
- +30° = index 6

The included WAVs are tagged as **48 kHz** because the MAT file itself does not contain a sample-rate field. If the actual dataset rate is not 48 kHz, regenerate the folder with the correct `--sr`.

## Build / run

Put your stereo input WAV in the project root as:

```bash
output.wav
```

Then:

```bash
cargo run --release
```

Or specify paths:

```bash
cargo run --release -- \
  output.wav \
  hrtf_profiles/default \
  hrtf_profiles/personal \
  config/headphone_eq.txt
```

## Comparison outputs

The program writes:

```text
compare_out/00_dry.wav
compare_out/01_reference_2speaker.wav
compare_out/02_hrtf_intensity_072.wav
compare_out/05_headphone_comp.wav
compare_out/06_room_baked.wav
compare_out/07_personal_profile.wav     # only if personal profile exists
compare_out/08_content_adaptive.wav
```

Listen in order rather than comparing only 00 vs 08.

### 01 Reference

No M/S redistribution. No room. No FDN.

```text
Input L -> virtual speaker -30° -> both ears
Input R -> virtual speaker +30° -> both ears
```

This is the clean baseline.

### 02 HRTF intensity

`0.72` is used by default.

The 0%-HRTF endpoint is not a dry signal. For each ear it is an energy-matched impulse placed at that HRIR's main arrival time. That keeps the large ITD/ILD anchor while removing most spectral peaks/notches.

### 05 Headphone EQ

The engine is implemented but the supplied config is flat.

Do not invent compensation curves. Add filters only when you have a measurement or a known correction target.

### 06 Room baked

The source/listener model is currently:

```text
room:      5.0 m x 6.0 m
listener:  (2.5, 2.0)
speakers:  1.5 m away at +/-30°
ER:        first + second-order horizontal image sources
selection: strongest 6 per speaker, <= 50 ms
late:      FDN with 55 ms predelay
```

Floor/ceiling reflections are intentionally omitted because this HRTF set is horizontal-only. Adding fake elevation cues would be less physical than leaving them diffuse.

### 07 Personal

Copy all 144 files into:

```text
hrtf_profiles/personal/
  hrtf_left_0.wav
  hrtf_right_0.wav
  ...
  hrtf_left_71.wav
  hrtf_right_71.wav
```

Then rerun. The program will create `07_personal_profile.wav`, and 08 will use that profile too.

### 08 Adaptive

This deliberately avoids stem separation or a neural network.

Every 256 samples it estimates:

- L/R correlation
- Mid energy
- Side energy

Highly coherent/centered content gets a focused, drier presentation. Wide/diffuse content gets more ER/FDN and a small amount of original stereo as a "dry anchor".

This is the part most likely to map cleanly to ESP32-S3.

## MAT -> WAV tool

```bash
python tools/mat_hrtf_to_wav.py small_pinna_final.mat \
  -o hrtf_profiles/default \
  --sr 48000
```

## First things to tune

Try these one at a time:

- `hrtf_intensity`: `0.55`, `0.65`, `0.72`, `0.80`
- room `speaker_distance_m`: `1.0` to `2.0`
- room reflection coefficients in `src/room.rs`
- adaptive early/late ranges in `src/adaptive.rs`
- FDN `feedback` and output gain in `src/fdn.rs`

Avoid increasing ER just because you cannot hear an echo. A good early field often works by changing externalization/room impression rather than sounding like a separate reflection.

## Note

This project was generated as a comparison prototype. The current environment does not contain `cargo`/`rustc`, so it could not be compiled here. The code is structured to use only `hound`, `rustfft`, and `num-complex`.
