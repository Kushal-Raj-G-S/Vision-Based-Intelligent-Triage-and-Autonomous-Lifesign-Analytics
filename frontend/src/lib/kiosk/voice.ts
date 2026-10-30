// Browser-side voice for the kiosk: plays the pre-rendered Sonia clips, records the patient's
// answer with silence detection, and encodes it as 16 kHz mono WAV for the local Whisper server.
// Nothing here synthesises speech at run time, so the voice never changes mid-sentence.

import { VOICE_BACKUP, VOICE_PRIMARY } from "./config";

// ── captions (what is being said, for subtitles / accessibility) ────────────────
let captions: Record<string, { text: string; when: string }> = {};
export async function loadCaptions(): Promise<void> {
  try {
    const r = await fetch("/voice/manifest.json");
    if (r.ok) captions = await r.json();
  } catch { /* captions are a nicety; speech still works without them */ }
}
export const captionFor = (id: string): string => captions[id]?.text ?? "";

// ── speaking ─────────────────────────────────────────────────────────────────────
type SpeakListener = (speaking: boolean, id: string | null) => void;
const speakListeners = new Set<SpeakListener>();
export function onSpeaking(cb: SpeakListener): () => void {
  speakListeners.add(cb);
  return () => { speakListeners.delete(cb); };
}
const emit = (speaking: boolean, id: string | null) => speakListeners.forEach((cb) => cb(speaking, id));

let current: HTMLAudioElement | null = null;
let generation = 0;

export function stopSpeaking(): void {
  generation++;
  if (current) { current.pause(); current = null; }
  emit(false, null);
}

function playUrl(url: string, gen: number): Promise<boolean> {
  return new Promise((resolve) => {
    const a = new Audio(url);
    current = a;
    a.onended = () => resolve(true);
    a.onerror = () => resolve(false);
    a.play().catch(() => resolve(false));
    // if stopSpeaking() was called meanwhile, don't hang the caller
    const iv = setInterval(() => { if (gen !== generation) { clearInterval(iv); a.pause(); resolve(true); } }, 100);
    a.addEventListener("ended", () => clearInterval(iv));
    a.addEventListener("error", () => clearInterval(iv));
  });
}

/** Speak one pre-rendered line. Resolves when it has finished (or immediately if it cannot be
 *  played — the caption is still shown for a reading-time delay, so the flow never stalls). */
export async function say(id: string): Promise<void> {
  const gen = ++generation;
  emit(true, id);
  let ok = await playUrl(`${VOICE_PRIMARY}/${id}.mp3`, gen);
  if (!ok && gen === generation) ok = await playUrl(`${VOICE_BACKUP}/${id}.mp3`, gen);
  if (!ok && gen === generation) {
    const words = captionFor(id).split(/\s+/).filter(Boolean).length;
    await new Promise((r) => setTimeout(r, Math.min(6000, 600 + words * 330)));
  }
  if (gen === generation) { current = null; emit(false, null); }
}

export async function sayAll(ids: string[], gapMs = 120): Promise<void> {
  for (const id of ids) {
    await say(id);
    await new Promise((r) => setTimeout(r, gapMs));
  }
}

/** Browsers refuse to play audio until the page has had a user gesture. Call this inside the
 *  first tap/click handler. */
export async function unlockAudio(): Promise<void> {
  try {
    const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new AC();
    await ctx.resume();
    await ctx.close();
  } catch { /* harmless */ }
}

// ── microphone ───────────────────────────────────────────────────────────────────
let micStream: MediaStream | null = null;

/** Ask for the microphone (needs a user gesture the first time). null = denied/unavailable —
 *  the kiosk then runs fully on touch/typing. */
export async function ensureMic(): Promise<MediaStream | null> {
  if (micStream && micStream.active) return micStream;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    return micStream;
  } catch {
    micStream = null;
    return null;
  }
}
export function releaseMic(): void {
  micStream?.getTracks().forEach((t) => t.stop());
  micStream = null;
}

function makeAnalyser(stream: MediaStream) {
  const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AC();
  const src = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  src.connect(analyser);
  const buf = new Float32Array(analyser.fftSize);
  const rms = () => {
    analyser.getFloatTimeDomainData(buf);
    let s = 0;
    for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
    return Math.sqrt(s / buf.length);
  };
  return { rms, close: () => { void ctx.close(); } };
}

export interface RecordOptions {
  /** give up if nobody starts speaking within this long */
  startTimeoutMs?: number;
  /** stop this long after speech ends */
  silenceMs?: number;
  /** hard cap on one answer */
  maxMs?: number;
  onLevel?: (level01: number) => void;
  signal?: AbortSignal;
}

/** Record ONE spoken answer. Resolves with a 16 kHz mono WAV, or null if nothing was said
 *  (or it was aborted). Speech start/stop is detected with an adaptive energy threshold: the
 *  room's noise floor is measured over the first ~400 ms and speech must clearly exceed it. */
export async function recordUtterance(stream: MediaStream, opts: RecordOptions = {}): Promise<Blob | null> {
  const { startTimeoutMs = 7000, silenceMs = 1300, maxMs = 12000, onLevel, signal } = opts;
  const an = makeAnalyser(stream);
  const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  const chunks: Blob[] = [];
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  rec.start(100);

  const t0 = performance.now();
  let floor = 0, floorN = 0, speaking = false, lastVoice = 0, spokeAt = 0;
  const result = await new Promise<boolean>((resolve) => {
    const tick = setInterval(() => {
      const now = performance.now();
      const level = an.rms();
      onLevel?.(Math.min(1, level * 12));
      if (now - t0 < 400) { floor += level; floorN++; return; }          // calibrate the noise floor
      const thr = Math.max(0.018, (floor / Math.max(1, floorN)) * 3.2);
      if (level > thr) {
        if (!speaking) { speaking = true; spokeAt = now; }
        lastVoice = now;
      }
      const aborted = signal?.aborted;
      if (aborted) { clearInterval(tick); resolve(false); return; }
      if (!speaking && now - t0 > startTimeoutMs) { clearInterval(tick); resolve(false); return; }
      if (speaking && now - lastVoice > silenceMs && now - spokeAt > 350) { clearInterval(tick); resolve(true); return; }
      if (now - t0 > maxMs) { clearInterval(tick); resolve(speaking); }
    }, 50);
  });

  await new Promise<void>((r) => { rec.onstop = () => r(); rec.stop(); });
  an.close();
  onLevel?.(0);
  if (!result || !chunks.length) return null;
  try { return await toWav16k(new Blob(chunks, { type: rec.mimeType || "audio/webm" })); } catch { return null; }
}

// ── WAV encoding ────────────────────────────────────────────────────────────────
async function toWav16k(blob: Blob): Promise<Blob> {
  const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AC();
  const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  void ctx.close();
  const frames = Math.max(1, Math.ceil(decoded.duration * 16000));
  const off = new OfflineAudioContext(1, frames, 16000);          // resamples + mixes down to mono
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const mono = (await off.startRendering()).getChannelData(0);
  const pcm = new DataView(new ArrayBuffer(44 + mono.length * 2));
  const w = (o: number, s: string) => { for (let i = 0; i < s.length; i++) pcm.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); pcm.setUint32(4, 36 + mono.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  pcm.setUint32(16, 16, true); pcm.setUint16(20, 1, true); pcm.setUint16(22, 1, true);
  pcm.setUint32(24, 16000, true); pcm.setUint32(28, 32000, true); pcm.setUint16(32, 2, true); pcm.setUint16(34, 16, true);
  w(36, "data"); pcm.setUint32(40, mono.length * 2, true);
  for (let i = 0; i < mono.length; i++) {
    const s = Math.max(-1, Math.min(1, mono[i]));
    pcm.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([pcm.buffer], { type: "audio/wav" });
}

/** During the scan: report when the patient has been talking for a sustained stretch (speech is this
 *  project's worst HR condition — see CLAUDE.md — so we ask them to stay quiet). */
export function startTalkMonitor(stream: MediaStream, onTalking: () => void, sustainMs = 1500): () => void {
  const an = makeAnalyser(stream);
  let floor = 0, n = 0, since = 0;
  const t0 = performance.now();
  const iv = setInterval(() => {
    const now = performance.now();
    const level = an.rms();
    if (now - t0 < 800) { floor += level; n++; return; }
    const thr = Math.max(0.03, (floor / Math.max(1, n)) * 4);
    if (level > thr) {
      if (!since) since = now;
      if (now - since >= sustainMs) { since = 0; onTalking(); }
    } else since = 0;
  }, 80);
  return () => { clearInterval(iv); an.close(); };
}
