# Adaptive revision: preserve the preferred 06 room balance

The listener feedback is that 06 is preferred to 08. This revision makes 08 a restrained variation of 06. It is a preference-driven hypothesis to evaluate, not a claim that listeners universally prefer these new numbers.

## Why the old settings were a large change

In main.rs, 06 uses early_gain=1, late_send=1, dry_anchor=0. The old controller instead used late sends of 0.06–0.24 for ordinary steady-state examples: about 24–12 dB less FDN excitation than 06. It also blended original headphone stereo into the binaural output. Therefore 08 was not a small refinement of 06; it substantially changed room/direct balance. That is a plausible explanation for the reported preference, not an established cause from a controlled listening test.

Steady-state examples (early gain / late send / dry anchor):

| Input | Previous 08 | Revised 08 | Fixed 06 |
| --- | --- | --- | --- |
| Identical L/R (center) | 0.72 / 0.06 / 0.02 | 0.94 / 0.85 / 0 | 1 / 1 / 0 |
| Balanced uncorrelated channels | 1.16 / 0.24 / 0.12 | 1 / 1 / 0 | 1 / 1 / 0 |
| Equal anti-phase channels | 0.84 / 0.14 / 0.06 | 1 / 1 / 0 | 1 / 1 / 0 |

Revised centered-content reductions are only about 0.54 dB for ER and 1.41 dB for late excitation. These are component gains, not whole-track loudness differences. Wide passages recover toward 06 over the smoothing interval rather than switching instantly.

## Implemented in adaptive.rs

- Start at exactly 06's unity room gains.
- Bound ER gain to 0.94–1.0 and late send to 0.85–1.0; never boost above 06.
- Set dry_anchor to zero to keep the measured binaural direct path.
- Estimate centeredness from positive correlation times L/R energy balance times mid-energy fraction. Negative correlation is not treated as centered coherence; one-sided input also produces zero centeredness. This is a simple signal heuristic, not vocal detection.
- Smooth with `1-exp(-block_samples/(sample_rate*0.25))`: a 250 ms time constant independent of sample rate/block size. It reaches about 63% of a target change after 250 ms. The prior fixed 0.08 coefficient was approximately 64 ms at 48 kHz/256 samples.
- Freeze gains below mean-square 1e-8 (-80 dBFS RMS across channels), including empty input and zero-padded tails. Internal FDN state continues running as before.

The only change outside adaptive.rs is passing sample rate to its constructor in main.rs. The revised room.rs from the preceding task is retained. Per-block gain updates remain; sample-level ramps are not introduced in this revision. The small range and slower control reduce gain-step size, but this does not prove inaudibility.

## Four-song comparison

Fresh renders use the same current room parameters for all four songs:

- Sing Sing Sing
- Summer Montage / Madeline
- Never Enough
- Don't Start Now

Output root: `music_renders/20260905_220344/`. Each song has 16 WAVs in compare_out, a render log, parameter record, and a compare_06_08.m3u playlist alternating fixed/adaptive for the same environment. This is fresh 06 versus revised 08, not an old-08/new-08 comparison. Earlier render folders remain intact.

Use matching pairs such as `06d_jazz_club.wav` and `08d_jazz_club_adaptive.wav`. Keep playback volume fixed within each pair. Their common output gain is identical within a song; cross-song gains may differ. Start with a centered vocal section, then a wide instrumental section. Record externalization, room impression, clarity, and overall preference separately. If 06 still wins, keeping adaptation disabled is a valid outcome; additional processing is not a requirement.

The render root contains snapshots of the current adaptive.rs, room.rs and main.rs, plus adaptive_before.rs. All renders use the original default 48 kHz HRTF profile, +/-30-degree speakers, preset-specific distances, and flat EQ. No personal profile is installed.

Validation: 14 unit tests pass, including center limits, anti-phase/hard-pan/uncorrelated behavior, silence hold, and elapsed-time smoothing across rates/block sizes. The 16-output impulse smoke test passes. Each full-track WAV is decoded with ffmpeg after generation.
