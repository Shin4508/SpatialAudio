"""End-to-end checks against the running Rust server; standard library only."""
import io
import json
import math
import struct
import urllib.error
import urllib.parse
import urllib.request
import wave

BASE = "http://127.0.0.1:3000"
def request(path, data=None):
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "audio/wav"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()

code, body = request("/api/presets")
assert code == 200
config = json.loads(body)
assert len(config["presets"]) == 7
sr = config["sample_rate"]
for path in ["/", "/style.css", "/app.js"]:
    assert request(path)[0] == 200
assert request("/../Cargo.toml")[0] == 404
buffer = io.BytesIO()
with wave.open(buffer, "wb") as wav:
    wav.setparams((2, 2, sr, 0, "NONE", "not compressed"))
    wav.writeframes(b"".join(struct.pack("<hh", int(5000*math.sin(2*math.pi*440*i/sr)), int(3500*math.sin(2*math.pi*660*i/sr))) for i in range(sr//10)))
audio = buffer.getvalue()
params = dict(environment="jazz_club", distance=1.5, angle=30, width=12, depth=18, height=4, rt60=0.3, predelay=20, reverb=0.15, adaptive="false")
def render(**changes):
    return request("/api/render?" + urllib.parse.urlencode(params | changes), audio)
def samples(body):
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"
    pos = 12
    while pos+8 <= len(body):
        size = struct.unpack_from("<I", body, pos+4)[0]
        if body[pos:pos+4] == b"data":
            result = struct.unpack("<"+"f"*(size//4), body[pos+8:pos+8+size])
            assert all(math.isfinite(v) and abs(v) <= 0.981 for v in result)
            assert any(abs(v) > 0.001 for v in result)
            return result
        pos += 8 + size + size%2
    raise AssertionError("Missing audio data")
code, first = render()
assert code == 200, first
fixed = samples(first)
assert len(fixed) > sr//10*2
code, changed = render(distance=2.0, adaptive="true")
assert code == 200, changed
assert samples(changed) != fixed
for changes in [dict(distance="NaN"), dict(distance=10,width=3,angle=90), dict(rt60=100), dict(environment="missing")]:
    assert render(**changes)[0] == 400, changes
assert request("/api/render?"+urllib.parse.urlencode(params), b"not audio")[0] == 400
print("PASS: static assets, presets, WAV render/tail/peak, changed controls, malformed audio, geometry and parameter rejection")
