# Spatial Compare Lab v2

Adaptive settings were revised after feedback favoring 06. See [ADAPTIVE_PARAMETER_REVIEW.md](ADAPTIVE_PARAMETER_REVIEW.md) for the current subtle 08 settings and the four-song comparison batch.

Room parameters were subsequently revised for the Don't Start Now render. See [ROOM_PARAMETER_REVIEW.md](ROOM_PARAMETER_REVIEW.md) for current distance gains and Eyring decay estimation; earlier descriptions below document the initial v2 settings.

A comparison branch of the current working `spatial_compare_lab`, reviewed 2026-09-05. The original code and the existing SOFA variant are untouched. This keeps the same eight Rust modules and dependencies. Read [DSP_REVIEW.md](DSP_REVIEW.md) before choosing parameters.

## Run

Run from this folder (paths are relative to the working directory):

```sh
cd spatial_compare_lab_v2
cargo run --release -- /absolute/path/input_48k.wav
```

Default HRTFs are read from `../spatial_compare_lab/hrtf_profiles/default`; no dataset or old render files are duplicated. The input sample rate must match the profile. Optional positional arguments:

```sh
cargo run --release -- /absolute/path/input_48k.wav \
  ../spatial_compare_lab/hrtf_profiles/default \
  ../spatial_compare_lab/hrtf_profiles/personal \
  config/headphone_eq.txt 1.5 30
```

The last two numbers are speaker distance in metres and symmetric speaker half-angle in degrees. Here speakers are at -30/+30 degrees, indices 66/6. Distance must be 0.15–10 m and speakers must lie inside every rendered preset's geometry. Angle must be 0–90 degrees. These are input limits, not validated perceptual operating ranges. Omit distance to use each environment's original default; angle defaults to 30 degrees.

To compare the already converted SOFA dataset, replace the profile argument with `../spatial_compare_lab_sofa/hrtf_profiles/default`. This still uses the 72-direction WAV loader, not native SOFA lookup.

## Changes

- FDN preserves separate mid and side inputs through predelay and injects them with different sign patterns. Previously both were summed into one scalar, creating a stereo cancellation null. This changes reverb balance; listening comparison is needed.
- A zero reverb send stops new excitation while existing state continues decaying. Zero predelay now really bypasses the predelay buffer.
- Renders flush convolution and reverb tails with zero input. Late-tail allowance is predelay + 44 ms + 1.5 times nominal mid-band RT60; it is a finite estimate, not a measured -90 dB guarantee. EQ settling is not independently measured.
- All generated files, including dry, receive one common peak-protection gain after rendering. This preserves relative levels within a run. It is not loudness matching; separate runs can have different gains.
- Distance parsing rejects malformed/non-finite inputs, geometry is checked, and speaker half-angle is adjustable without editing Rust.
- `compare_out/run.txt` records input/profile paths, sample rate, requested distance and angle, direct HRTF indices, common gain and the generated file list. Capture console output for each preset's actual distance, ER events and RT60.

Outputs retain the original names: 00 dry, 01 reference, 02 intensity, 05 EQ, seven 06 environments, optional 07 personal jazz club, and seven 08 adaptive environments. Existing output filenames are overwritten; move each completed `compare_out` folder before another parameter run. Old optional files may remain but are excluded from the current run's manifest and gain calculation. If rendering fails, outputs may be partial and have no final common gain; use only a completed run.

## Listening protocol

1. Begin with 1.5 m and 30 degrees using one profile, flat EQ, and the same input excerpt. Use the fixed 06 renders before judging the adaptive 08 outputs.
2. Hold distance fixed and try half-angles 20, 30, 40 degrees (indices L/R: 68/4, 66/6, 64/8). Compare center stability, width, front/back errors and timbre.
3. Hold the chosen angle fixed and try 1.0, 1.5, 2.0 m. These are experiment points, not recommended universal distances. Keep playback level fixed for a distance-cue test; for timbre preference, make separate loudness-matched listening copies.
4. Compare HRTF profiles with identical parameters and headphones. Profile differences can include gain and recording conventions as well as anatomy.
5. Record ratings before revealing filenames if possible. Keep run.txt and console logs with the audio. No code analysis can select the best personal HRTF or externalization distance without listening/measurement.

HRTF index is derived from angle, not an independent width control. Arbitrary angles use the nearest 5-degree measurement, with at most 2.5-degree grid error. Room geometry still uses the requested angle. A converted sparse SOFA profile may have additional source-grid error.

## Validation

```sh
cargo test --offline
cargo build --release --offline
python3 tools/smoke_test.py
```

Six tests cover measured-HRIR preservation, explicit intensity blending, angle wrapping, convolution against a direct sum across block boundaries, the former stereo reverb null, and decay after send is zero. An impulse at the final input frame is also used for an end-to-end render check. Offline commands require cached dependencies; omit `--offline` on a fresh setup.

## Concertgebouw-inspired grand hall

The seventh environment (`concertgebouw`) is a larger shoebox room, 27.8 m wide × 44 m deep × 17.5 m high, inspired by the Grote Zaal dimensions reported in the [Institute of Acoustics study](https://www.ioa.org.uk/system/files/proceedings/pa_clements_the_acoustic_design_of_the_concertgebouw_amsterdam_and_resolution_of_the_halls_early_acoustical_difficultie.pdf). It is included automatically in both 06 fixed and 08 adaptive renders.

The preset uses a 2.2 s mid-band decay target, 1.65 s high-band decay, 45 ms late predelay, and up to 16 early reflections within 160 ms. These are model tuning choices, not a measured reconstruction of the venue. The listener is centered across the hall, 15 m from the back wall; speakers default to 3 m distance and ±30°, with the same command-line overrides as other rooms. Plaster boundaries and a wood floor approximate reflective surfaces; balconies, seating, and detailed diffusion are not modeled.
