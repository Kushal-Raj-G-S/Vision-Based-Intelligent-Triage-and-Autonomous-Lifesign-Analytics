import asyncio
from pathlib import Path
import numpy as np, soundfile as sf
HERE = Path(__file__).parent
OUT = HERE / "samples" / "short"; OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*"): f.unlink()
LINE = ("Hello, and welcome. Please sit upright, look at the screen, and breathe normally. "
        "Thank you, Manoj. You're doing really well.")
voices = [
    ("edge", "en-IN-NeerjaNeural"), ("edge", "en-IN-NeerjaExpressiveNeural"), ("edge", "en-IN-PrabhatNeural"),
    ("edge", "en-US-AvaNeural"), ("edge", "en-US-JennyNeural"), ("edge", "en-GB-SoniaNeural"),
    ("kokoro", "af_heart"), ("kokoro", "af_sarah"), ("kokoro", "af_sky"),
    ("kokoro", "bf_emma"), ("kokoro", "bf_isabella"), ("kokoro", "am_michael"), ("kokoro", "af_nicole"),
]
words = "one two three four five six seven eight nine ten eleven twelve thirteen".split()
from kokoro_onnx import Kokoro
import edge_tts
k = Kokoro(str(HERE/"models"/"kokoro-v1.0.onnx"), str(HERE/"models"/"voices-v1.0.bin"))
async def edge(text, voice, path):
    await edge_tts.Communicate(text, voice, rate="-8%", pitch="-2Hz").save(str(path))
for i, (eng, v) in enumerate(voices):
    text = f"Voice {words[i]}. {LINE}"
    wav = OUT / f"{i+1:02d}_{eng}_{v}.wav"
    if eng == "kokoro":
        s, sr = k.create(text, voice=v, speed=0.92, lang="en-gb" if v.startswith("b") else "en-us")
    else:
        mp3 = OUT / "tmp.mp3"; asyncio.run(edge(text, v, mp3))
        s, sr = sf.read(str(mp3)); mp3.unlink()
    if getattr(s, "ndim", 1) > 1: s = s.mean(axis=1)
    sf.write(wav, s, sr)
    print(f"{i+1:2d}  {eng:6s} {v:32s} {len(s)/sr:5.1f}s")
