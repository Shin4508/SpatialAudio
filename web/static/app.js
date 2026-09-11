"use strict";
const form = document.querySelector("#render-form");
const status = document.querySelector("#status");
const submit = document.querySelector("#submit");
const music = document.querySelector("#music");
let sampleRate, presets = [], originalUrl, resultUrl;
function message(text, error = false) { status.textContent = text; status.classList.toggle("error", error); }
function setPreset() {
  const p = presets.find(p => p.id === form.elements.environment.value);
  if (!p) return;
  for (const key of ["width", "depth", "height", "distance", "rt60", "predelay", "reverb"]) form.elements[key].value = p[key];
}
music.addEventListener("change", () => {
  if (originalUrl) URL.revokeObjectURL(originalUrl);
  const audio = document.querySelector("#original");
  audio.hidden = !music.files.length;
  if (music.files.length) audio.src = originalUrl = URL.createObjectURL(music.files[0]);
});
form.elements.environment.addEventListener("change", setPreset);
// Decode common music formats and resample to the HRTF rate before upload.
async function toWav(file) {
  if (!file || file.size > 70 * 1024 * 1024) throw new Error("Choose an audio file no larger than 70 MiB.");
  const context = new AudioContext();
  let decoded;
  try { decoded = await context.decodeAudioData(await file.arrayBuffer()); }
  catch { throw new Error("Your browser could not decode this file. Try WAV or MP3."); }
  finally { await context.close(); }
  if (decoded.duration <= 0 || decoded.duration > 180) throw new Error("Choose a track or excerpt up to 3 minutes long.");
  if (decoded.numberOfChannels > 2) throw new Error("Choose mono or stereo audio; surround files are not supported.");
  const offline = new OfflineAudioContext(2, Math.ceil(decoded.duration * sampleRate), sampleRate);
  const source = offline.createBufferSource(); source.buffer = decoded; source.connect(offline.destination); source.start();
  const audio = await offline.startRendering();
  const buffer = new ArrayBuffer(44 + audio.length * 4), view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0,"RIFF"); view.setUint32(4,buffer.byteLength-8,true); text(8,"WAVE"); text(12,"fmt ");
  view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,2,true);
  view.setUint32(24,sampleRate,true); view.setUint32(28,sampleRate*4,true); view.setUint16(32,4,true); view.setUint16(34,16,true);
  text(36,"data"); view.setUint32(40,audio.length*4,true);
  const left=audio.getChannelData(0), right=audio.getChannelData(1);
  for(let i=0;i<audio.length;i++) for(let ch=0;ch<2;ch++) {
    const value=Math.max(-1,Math.min(1,ch ? right[i] : left[i]));
    view.setInt16(44+i*4+ch*2,Math.round(value*(value<0?32768:32767)),true);
  }
  return buffer;
}
form.addEventListener("submit", async event => {
  event.preventDefault();
  const file = music.files[0];
  const query = new URLSearchParams(new FormData(form));
  query.set("adaptive", String(form.elements.adaptive.checked));
  const label = form.elements.environment.selectedOptions[0].textContent;
  const controls = [...form.querySelectorAll("input, select, button")];
  controls.forEach(c => c.disabled = true);
  document.querySelector("#results").hidden = true;
  document.querySelector("#result").pause();
  try {
    message("Preparing your audio…");
    const wav = await toWav(file);
    message("Rendering your space… Longer tracks may take a few minutes.");
    const response = await fetch(`/api/render?${query}`, {method:"POST", headers:{"Content-Type":"audio/wav"}, body:wav});
    if (!response.ok) throw new Error((await response.text()).slice(0,500) || `Render failed (${response.status}).`);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    resultUrl = URL.createObjectURL(await response.blob());
    document.querySelector("#result").src = resultUrl;
    document.querySelector("#download").href = resultUrl;
    document.querySelector("#result-info").textContent = `${file.name} · ${label} · ${query.get("distance")} m · ±${query.get("angle")}° · RT60 ${query.get("rt60")} s`;
    document.querySelector("#results").hidden = false;
    message("Ready. Compare the original and rendered audio below.");
  } catch (error) { message(error.message, true); }
  finally { controls.forEach(c => c.disabled = false); }
});
(async () => {
  try {
    const response = await fetch("/api/presets");
    if (!response.ok) throw new Error("Could not load environments. Reload to retry.");
    const data = await response.json(); sampleRate = data.sample_rate; presets = data.presets;
    for (const p of presets) form.elements.environment.add(new Option(p.name, p.id));
    form.elements.environment.value = "jazz_club"; setPreset(); submit.disabled = false;
    message(`Ready · audio will be converted to ${sampleRate / 1000} kHz stereo.`);
  } catch (error) { message(error.message, true); }
})();
