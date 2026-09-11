# Electronic module candidates

This shortlist is for a physical stereo-in/stereo-out headphone module running one mode from `spatial_compare_lab_v2`. The first mode is `jazz_club`: it exercises the direct HRTF, early reflections and late FDN path while avoiding the memory and CPU cost of rendering all seven comparison modes.

## Candidates

| Candidate | Audio I/O | DSP headroom | Strength | Constraint | Fit |
| --- | --- | --- | --- | --- | --- |
| **ESP32-S3 + ES8388** | I2S stereo codec, 24-bit/48 kHz | Moderate; use partitioned FIR and a short late field | Low cost, Wi-Fi/BLE, broad maker support | Full measured HRTF convolution and seven-room batch do not fit unchanged | **Recommended prototype** |
| Raspberry Pi Zero 2 W + USB/I2S codec | USB or I2S stereo | High; can run the current Rust renderer with streaming changes | Fastest path to the existing v2 DSP | Linux boot time, larger power draw and enclosure | Best software-validation bridge |
| Daisy Seed (STM32H7) + codec board | I2S stereo codec | High real-time MCU DSP | Audio-focused board, deterministic timing, built-in DSP ecosystem | Smaller Rust ecosystem; custom carrier needed | Strong second hardware choice |
| Teensy 4.1 + Audio Shield | SGTL5000 stereo codec | High MCU DSP | Mature audio library and easy bench testing | Less integrated wireless/enclosure story | Good educational/prototype option |

## Recommended module

Use an **ESP32-S3 development board with PSRAM**, an **ES8388 stereo codec**, a 3.5 mm headphone amplifier, two rotary encoders, and one mode button. The button selects the one compiled `jazz_club` profile; the encoders adjust distance and output gain after the fixed profile is stable. Feed line/headphone input into the codec ADC and connect the codec DAC to the headphone amplifier.

```text
stereo line-in
    │
 ES8388 ADC ── I2S ── ESP32-S3
                         │
              jazz_club HRTF + early ER + short FDN
                         │
 ES8388 DAC ◄── I2S ────┘
    │
 headphone amplifier ── headphones
```

The repository runner is a desktop bring-up tool with the same v2 library:

```sh
cd electronic_module
cargo run --release -- ../input_48k.wav ../jazz_club.wav jazz_club 1.5 30
```

It renders one mode only. The input must be stereo and match the HRTF sample rate (the included profile is 48 kHz). Set `HRTF_PROFILE` when using another profile. The binary writes a 32-bit float stereo WAV and applies the v2 peak ceiling.

## Firmware boundary

The current v2 renderer is an offline reference and allocates FFT buffers, loads all HRIR WAV files, and reads a complete input file. Porting it directly to an ESP32-S3 is not the first milestone. Bake one `jazz_club` configuration on the host, export compact direct/early FIR partitions, and implement a streaming block engine on the module. Keep 48 kHz, 256-sample blocks, stereo I2S, and a fixed ±30° speaker layout for the first test. Measure callback time, heap high-water mark, PSRAM use, output underruns and peak level before adding adaptive mode or user-editable room geometry.

## Bring-up sequence

1. Validate the codec loopback and a dry stereo path.
2. Compare the module's fixed `jazz_club` output against `spatial_compare_lab_v2` using the same impulse and music excerpt.
3. Add the early-reflection FIR partitions, then the short late field.
4. Add encoders for distance/output gain with bounded, smoothed updates.
5. Only after timing is stable, consider another environment or adaptive processing.

The candidate comparison is an engineering recommendation, not a measured benchmark. Board prices, codec availability and library support can change; confirm current parts before ordering.
