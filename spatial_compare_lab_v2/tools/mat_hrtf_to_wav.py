#!/usr/bin/env python3
"""
Convert small_pinna_final.mat HRTFs to mono WAV files.

Expected MAT structure:
    left  : (impulse_samples, directions)
    right : (impulse_samples, directions)

For the uploaded file this is:
    left/right = (200, 72)

Examples:
    python mat_hrtf_to_wav.py small_pinna_final.mat
    python mat_hrtf_to_wav.py small_pinna_final.mat --sr 44100
    python mat_hrtf_to_wav.py small_pinna_final.mat --indices 0 8 9 16 32 40 48 56 63
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import loadmat


def convert(mat_path: Path, out_dir: Path, sr: int, indices=None):
    data = loadmat(mat_path)

    if "left" not in data or "right" not in data:
        raise KeyError("MAT file must contain 'left' and 'right' arrays.")

    left = np.asarray(data["left"], dtype=np.float64)
    right = np.asarray(data["right"], dtype=np.float64)

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"Expected 2-D arrays, got left={left.shape}, right={right.shape}")
    if left.shape != right.shape:
        raise ValueError(f"left/right shapes differ: {left.shape} vs {right.shape}")

    n_samples, n_directions = left.shape

    if indices is None:
        indices = list(range(n_directions))

    bad = [i for i in indices if i < 0 or i >= n_directions]
    if bad:
        raise IndexError(
            f"Invalid direction indices {bad}; valid range is 0..{n_directions - 1}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # FLOAT WAV is intentional: HRIR values are preserved without PCM quantization
    # or per-file normalization, which would destroy relative HRTF levels.
    for idx in indices:
        sf.write(
            out_dir / f"hrtf_left_{idx}.wav",
            left[:, idx].astype(np.float32),
            sr,
            subtype="FLOAT",
        )
        sf.write(
            out_dir / f"hrtf_right_{idx}.wav",
            right[:, idx].astype(np.float32),
            sr,
            subtype="FLOAT",
        )

    print(f"MAT: {mat_path}")
    print(f"HRTF shape: {left.shape} = {n_samples} samples x {n_directions} directions")
    print(f"Sample rate written to WAV: {sr} Hz")
    print(f"Output directory: {out_dir}")
    print(f"Created {len(indices) * 2} WAV files.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_file", type=Path)
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("hrtf_wav"),
        help="Output directory (default: hrtf_wav)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=48000,
        help="WAV sample rate metadata (default: 48000). "
             "Set this to the actual sampling rate of the HRTF dataset.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Only export these direction indices. Default: export all.",
    )
    args = parser.parse_args()

    convert(args.mat_file, args.output, args.sr, args.indices)


if __name__ == "__main__":
    main()
