# Room parameter revision — 2026-09-05

Scope: only `src/room.rs` DSP changes. The previous three music renders retain the earlier settings. The new Don't Start Now render uses this revision; snapshots of room.rs before/after are stored alongside that render.

## Changes and reasoning

- **Early gain: `1/sqrt(distance)` with clamps -> 1.0.** Each image source already has `1/path` pressure spreading. Applying another distance factor suppresses reflections a second time. At preset distances this raises early-reflection amplitude by approximately 1.8 dB (1.5 m), 2.6 dB (1.8 m), 4.0 dB (2.5 m), and 4.8 dB (3 m), before common output scaling.
- **Late send: `0.55 + 0.45*sqrt(distance)` with clamps -> 1.0.** For a fixed-power source, a constant diffuse-field excitation is a more defensible starting assumption than increasing it with listener distance. Preset late_mix still sets artistic reverb level; this is not a calibrated diffuse-energy solution, particularly outdoors. Nominal send is reduced by roughly 0.8–2.5 dB at preset distances.
- **Direct gain uses 3-D distance.** `hypot(horizontal_distance, source_height-listener_height)` now matches the distance used when subtracting the direct path from reflection travel times. Only Theater differs at default settings: 3.000 -> 3.015 m, a very small level change. The CLI distance remains horizontal radius. Existing near/far gain clamps remain deliberate limits.
- **Enclosed-room nominal RT60 uses Eyring.** `0.161*V / [-S*ln(1-A/S)]`, where A is absorption area and S is total surface area, replaces `0.161*V/A`. This uses the same material coefficients and accounts for finite mean absorption instead of the small-absorption approximation. It shortens predicted decay; it is still an approximate diffuse-field estimate with nonuniform materials, not a measured improvement. The 0.20 s mid/0.15 s high minimums and 3.50 s maximum remain artistic limits. Open Air and Street keep their explicit overrides. The existing damping cutoff follows the revised high/mid decay ratio.

Eyring and Sabine are both supported reference estimates in [Pyroomacoustics](https://pyroomacoustics.readthedocs.io/en/stable/pyroomacoustics.acoustics.html#pyroomacoustics.acoustics.rt60_eyring). Formula cross-check: [python-acoustics room implementation](https://github.com/python-acoustics/python-acoustics/blob/master/acoustics/room.py). Choosing Eyring here is an engineering judgment for these treated-room presets, not a claim of measured accuracy.

## Parameters retained after review

| Parameters | Decision |
| --- | --- |
| Sound speed 343 m/s | Reasonable fixed room-temperature approximation. Temperature/humidity are not simulated. |
| Room dimensions: studio 6x8x3, club 8x12x3.5, hall 15x22x8, theater 20x30x12 m | Geometrically plausible illustrative spaces. All default speaker/listener positions pass bounds checks. |
| Default speaker radius 1.5–3 m; angles +/-30 degrees | Reasonable virtual loudspeaker starting points. Large hall dimensions do not require a distant source; these represent a nearby source within a large space. Angles map to measured indices 66/6. |
| Height 1.2 m, theater source 1.5 m | Plausible seated-ear/source heights; horizontal HRIR projection still cannot reproduce elevation. |
| Material absorption/scattering | Bounded values with broadly plausible frequency trends, but not measured material specifications. Wood mounting, carpet thickness and audience distribution can change them substantially. Replacing numbers without those details would add false precision. Audience assigned to theater side walls is an effective treatment, not literal audience geometry. |
| ER cap 1/6/6/8/10/10, windows 30/80/45/55/80/85 ms | Reasonable static comparison/compute budget, not complete room responses. Strongest-N selection can omit meaningful paths. |
| Predelay ~24 ms indoors; Street 70 ms | Artistic values. Actual late onset also includes the FDN's shortest delay (~29.7 ms). No arbitrary offset retuning without examining the ER-to-late transition by listening. |
| Air HF loss 0.025–0.04 dB/m | Gentle artistic high-frequency tilt, not a frequency/temperature/humidity-dependent atmospheric model. Direct HRIR remains unfiltered by this term. |
| Late output mixes 0–0.17 | Uncalibrated listening levels. Keep for continuity; new distance send changes effective level. |
| Scattering factor, minimum ER gain 0.015 and delay 0.35 ms | Heuristic sparsity/level controls, not physical conservation. Do not infer measured room energy from them. |

Floor/ceiling directions are still projected horizontally. Material filtering and event generation algorithms are otherwise unchanged; the scope is room parameters and their derivation, not a broader DSP redesign.

## Validation

Ten Rust tests pass, including four new room tests: inverse-distance ratio without extra ER attenuation, 3-D direct distance, analytical Eyring and its low-absorption limit, and preset geometry/event bounds. The end-to-end impulse smoke test passes for all 16 outputs. Don't Start Now is resampled to 48 kHz and rendered with the default HRIR profile and flat EQ. Each full-track output is decoded with ffmpeg after rendering.

Nominal decay values from the old/new render logs (mid / high seconds):

| Preset | Before | After |
| --- | --- | --- |
| Recording Studio | 0.23 / 0.17 | 0.20 / 0.15 |
| Jazz Club | 0.45 / 0.33 | 0.36 / 0.24 |
| Piano Hall | 2.59 / 1.61 | 2.42 / 1.43 |
| Theater | 1.19 / 0.79 | 0.92 / 0.51 |

These are FDN targets, not measured output RT60s. Completed output: `music_renders/20260905_215219/dont_start_now/compare_out/` (16 verified WAVs).
