# Single-mode electronic module runner

This crate is the desktop reference for the first hardware profile: one `spatial_compare_lab_v2` environment per run. It defaults to `jazz_club`, then writes one peak-protected stereo WAV for comparison with a future I2S codec module.

```sh
cargo run --release --locked -- ../input_48k.wav ../jazz_club.wav
cargo run --release --locked -- ../input_48k.wav ../theater.wav theater 2.0 30
```

The positional arguments are input WAV, output WAV, optional environment slug, optional speaker distance in metres, and optional speaker half-angle in degrees. The input must be stereo and match the HRTF profile sample rate. Set `HRTF_PROFILE` to select another profile. See [`CANDIDATES.md`](CANDIDATES.md) for the electronics shortlist and ESP32-S3 bring-up plan.
