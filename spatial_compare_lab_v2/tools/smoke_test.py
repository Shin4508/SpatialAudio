"""Run after cargo build --release; uses only Python's standard library."""
from pathlib import Path
import math
import struct
import subprocess
import tempfile
import wave

root = Path(__file__).resolve().parents[1]
exe = root / 'target/release/spatial_compare_lab_v2'
profile = root.parent / 'spatial_compare_lab/hrtf_profiles/default'

def read_float_wav(path):
    raw = path.read_bytes()
    pos = 12
    while pos + 8 <= len(raw):
        kind, size = struct.unpack_from('<4sI', raw, pos)
        pos += 8
        if kind == b'data':
            return struct.unpack('<' + 'f' * (size // 4), raw[pos:pos + size])
        pos += size + size % 2
    raise AssertionError(f'No data: {path}')

with tempfile.TemporaryDirectory(prefix='spatial_v2_test_') as folder:
    work = Path(folder)
    source = work / 'impulse.wav'
    with wave.open(str(source), 'wb') as w:
        w.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
        w.writeframes(b''.join(struct.pack('<hh', 16000 if i == 479 else 0,
            -8000 if i == 479 else 0) for i in range(480)))
    cmd = [str(exe), str(source), str(profile), str(work / 'no_personal'),
           str(root / 'config/headphone_eq.txt')]
    result = subprocess.run(cmd + ['1.5', '30'], cwd=work, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    files = sorted((work / 'compare_out').glob('*.wav'))
    assert len(files) == 18, len(files)
    hall_files = [path for path in files if 'concertgebouw' in path.name]
    assert len(hall_files) == 2, hall_files
    for path in files:
        samples = read_float_wav(path)
        assert all(math.isfinite(x) and abs(x) <= 0.980001 for x in samples), path
        if not path.name.startswith('00_'):
            assert len(samples) > 960, path
            assert any(abs(x) > 1e-7 for x in samples[960:]), path
    manifest = (work / 'compare_out/run.txt').read_text()
    assert 'left_index=66' in manifest and 'right_index=6' in manifest
    gain = float(next(line.split('=')[1] for line in manifest.splitlines()
                      if line.startswith('common_gain=')))
    dry = read_float_wav(work / 'compare_out/00_dry.wav')
    assert abs(dry[-2] - 16000 / 32768 * gain) < 1e-7
    for distance, angle in [('NaN', '30'), ('bad', '30'), ('1.5', 'inf'), ('10', '90')]:
        failed = subprocess.run(cmd + [distance, angle], cwd=work, capture_output=True)
        assert failed.returncode != 0, (distance, angle)
print('PASS: 18 renders, finite/headroom checks, final-frame tails, shared dry gain, indices, invalid inputs')
