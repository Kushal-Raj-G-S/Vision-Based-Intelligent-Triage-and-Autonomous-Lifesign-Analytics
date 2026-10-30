"""Local GPU speech-to-text for the patient kiosk (Whisper on CUDA).

Run it with the GPU environment (NOT the project venv):
    tools/gpu_lab/venv/Scripts/python.exe tools/gpu_lab/stt_server.py
The Flask backend proxies POST /api/stt to this server (port 5003). If this server
is not running the kiosk falls back to typing, so a patient is never stuck.

Why local: patient audio never leaves the machine, it works without internet, and a
large Whisper model is far better than browser speech recognition on names.

API: POST /transcribe  (multipart: audio=<WAV file>, hint=name|age|sex|yesno|complaint)
     -> {success, text, field, confidence, ...}
"""
import io
import os
import re
import sys
import time

import numpy as np
import soundfile as sf
import torch
import whisper
from flask import Flask, jsonify, request

MODEL_NAME = os.getenv("WHISPER_MODEL", "turbo")     # large-v3-turbo: ~1.6 GB, fast, accurate
PORT = int(os.getenv("STT_PORT", "5003"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = {
    # A short in-domain prompt biases Whisper toward the vocabulary it should expect.
    "name": "Patient registration at a hospital in India. The patient says their name, for example Manoj Kumar, "
            "Priya Sharma, Aarav Mehta, Lakshmi Narayanan, Karthik, Ananya, Mohammed Irfan, Fatima, Gurpreet Singh.",
    "age": "The patient says their age in years, for example thirty four, sixty one, eight, or two months.",
    "sex": "The patient says male, female, other, or prefer not to say.",
    "yesno": "The patient answers yes or no.",
    "complaint": "The patient describes in their own words what is wrong, for example chest pain, fever, headache, "
                 "stomach ache, breathing difficulty, an injury or dizziness.",
}

print(f"[STT] loading Whisper '{MODEL_NAME}' on {DEVICE} (first run downloads the model)...", flush=True)
t0 = time.time()
model = whisper.load_model(MODEL_NAME, device=DEVICE)
print(f"[STT] model ready in {time.time() - t0:.1f}s", flush=True)

# Warm-up: the first real transcription compiles CUDA kernels (~18 s measured). Do that now,
# on noise, so the first patient gets the normal ~0.2 s response.
try:
    _w = time.time()
    _noise = (np.random.RandomState(0).randn(16000 * 3) * 0.05).astype(np.float32)
    model.transcribe(_noise, language="en", fp16=(DEVICE == "cuda"), temperature=0.0, condition_on_previous_text=False)
    print(f"[STT] warmed up in {time.time() - _w:.1f}s", flush=True)
except Exception as _exc:
    print(f"[STT] warm-up skipped: {_exc}", flush=True)

app = Flask(__name__)

from stt_parse import PARSERS  # noqa: E402  (pure functions, unit-tested)


def _to_16k_mono(data, sr):
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != 16000:
        import torchaudio
        data = torchaudio.functional.resample(torch.from_numpy(data), sr, 16000).numpy()
    return data


@app.route("/health")
def health():
    return jsonify({"ok": True, "model": MODEL_NAME, "device": DEVICE})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    f = request.files.get("audio")
    if f is None:
        return jsonify({"success": False, "error": "no audio"}), 400
    hint = request.form.get("hint", "")
    try:
        data, sr = sf.read(io.BytesIO(f.read()), dtype="float32", always_2d=False)
    except Exception as exc:
        return jsonify({"success": False, "error": f"unreadable audio: {exc}"}), 400
    audio = _to_16k_mono(data, sr)
    seconds = len(audio) / 16000.0
    if seconds < 0.3 or float(np.abs(audio).max()) < 0.003:
        return jsonify({"success": True, "text": "", "field": None, "confidence": 0.0, "seconds": seconds,
                        "note": "silence"})
    t0 = time.time()
    res = model.transcribe(audio, language="en", fp16=(DEVICE == "cuda"), temperature=0.0,
                           condition_on_previous_text=False, initial_prompt=PROMPTS.get(hint),
                           no_speech_threshold=0.6, logprob_threshold=-1.0)
    text = (res.get("text") or "").strip()
    segs = res.get("segments") or []
    if segs:
        no_speech = float(np.mean([s.get("no_speech_prob", 0.0) for s in segs]))
        avg_lp = float(np.mean([s.get("avg_logprob", -1.0) for s in segs]))
    else:
        no_speech, avg_lp = 1.0, -3.0
    confidence = float(np.clip((1.0 - no_speech) * np.exp(avg_lp), 0.0, 1.0))   # ~ per-token probability
    field = None
    if text and hint in PARSERS:
        parsed = PARSERS[hint](text)
        field = {"age": parsed[0], "age_months": parsed[1]} if hint == "age" else parsed
    return jsonify({"success": True, "text": text, "field": field, "confidence": round(confidence, 3),
                    "no_speech_prob": round(no_speech, 3), "seconds": round(seconds, 2),
                    "latency_s": round(time.time() - t0, 2)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, threaded=False)
