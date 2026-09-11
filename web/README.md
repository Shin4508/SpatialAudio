# Spatial Sound web app

A plain HTML/CSS/JavaScript interface served by a Rust Axum backend. It uses the same v2 DSP library as the offline CLI.

## Run

From the repository root:

```sh
cargo run --release --locked --manifest-path web/Cargo.toml
```

Open <http://127.0.0.1:3000>. Choose music, select an environment, edit the controls, and render. The result can be played in the page or downloaded as a 32-bit float stereo WAV. The server handles one render at a time; another request receives HTTP 429.

Rust/Cargo and a complete HRTF profile are required. The default is `spatial_compare_lab/hrtf_profiles/default`, resolved relative to the source tree, independent of the working directory. To use another profile:

```sh
HRTF_PROFILE=/absolute/path/to/profile cargo run --release --locked --manifest-path web/Cargo.toml
```

The browser decodes formats supported by its audio codecs (including typical WAV/MP3 files), duplicates mono to stereo, and resamples to the profile rate. Uploads are limited to 70 MiB and 180 seconds. The upload WAV is 16-bit PCM; the rendered download is 32-bit float. Files with more than two channels are rejected. Rendering includes a tail and peak protection; original and rendered playback are not loudness matched. Browser decoding can consume substantial memory, so use short excerpts for comparison.

Editable controls: seven environment presets, speaker distance and half-angle, room width/depth/height, late reverb RT60, predelay, output gain, and content adaptation. Preset materials remain fixed. Listener floor coordinates scale with room dimensions; listener/speaker height remains 1.2 m. Geometry is validated before processing. Late reverb controls explicitly override preset values, including enabling reverb in open air when gain is nonzero. High-frequency RT60 is 75% of the selected decay; damping follows the preset. HRTF angles use the existing 5° lookup grid. Each request produces one selected environment render; the offline CLI still creates its full comparison batch.

The server binds only to localhost. This is a local application, not a public hosting configuration. GitHub stores the source; GitHub Pages cannot execute the Rust backend. Uploaded and rendered files live in a temporary directory that is removed after each request; the browser retains the downloaded audio until the page closes or another result replaces it.

## API and validation

- `GET /api/presets`: profile sample rate and preset control defaults.
- `POST /api/render?environment=jazz_club&distance=1.5&angle=30&width=12&depth=18&height=4&rt60=1&predelay=20&reverb=0.15&adaptive=false`: raw stereo WAV request body; WAV response. All query fields are required. Errors use non-2xx status codes and text bodies.

```sh
cargo test --locked --manifest-path web/Cargo.toml
cargo test --locked --manifest-path spatial_compare_lab_v2/Cargo.toml
python3 web/tests/smoke.py  # with the server running
```

HTTP implementation uses [Axum](https://docs.rs/axum/0.8.9/axum/). Project code is licensed under [Apache 2.0](../LICENSE). Third-party dependencies, HRTF datasets, and music retain their respective licenses.
