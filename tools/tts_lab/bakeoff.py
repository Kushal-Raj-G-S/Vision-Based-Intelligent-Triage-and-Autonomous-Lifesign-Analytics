import asyncio, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
HERE = Path(__file__).parent
OUT = HERE / "samples"; OUT.mkdir(exist_ok=True)

SCRIPT = (
    "Hello, and welcome. I'm going to take a quick, contactless health check. "
    "It only takes about a minute and a half, and you don't need to touch anything. "
    "Could you please tell me your name? "
    "Thank you, Manoj. Let's get you comfortable. Please sit upright and look at the screen. "
    "A little closer, please. Perfect. Hold just like that. "
    "Now, breathe normally, and stay as still as you can. I'll count you in. Three, two, one. "
    "You're doing really well. Just keep breathing naturally. "
    "All done. Thank you for your patience. A member of our team will be with you shortly."
)

# ---- Kokoro (local, on-device) ----
from kokoro_onnx import Kokoro
k = Kokoro(str(HERE/"models"/"kokoro-v1.0.onnx"), str(HERE/"models"/"voices-v1.0.bin"))
avail = sorted(k.get_voices())
print("Kokoro voices:", avail)
want = [v for v in ("af_heart","af_nicole","af_sarah","af_sky","bf_emma","bf_isabella","am_michael","hf_alpha") if v in avail]
for v in want:
    t0 = time.time()
    lang = "hi" if v.startswith("h") else ("en-gb" if v.startswith("b") else "en-us")
    text = SCRIPT if lang != "hi" else SCRIPT
    samples, sr = k.create(text, voice=v, speed=0.92, lang=lang)
    sf.write(OUT/f"kokoro_{v}.wav", samples, sr)
    print(f"  kokoro {v}: {len(samples)/sr:.1f}s audio in {time.time()-t0:.1f}s")

# ---- Microsoft neural voices via edge-tts (online) ----
import edge_tts
async def edge():
    vs = await edge_tts.list_voices()
    names = {x["ShortName"] for x in vs}
    cand = ["en-IN-NeerjaNeural","en-IN-NeerjaExpressiveNeural","en-IN-PrabhatNeural","en-US-AvaNeural","en-US-JennyNeural","en-US-EmmaNeural","en-GB-SoniaNeural","en-GB-LibbyNeural"]
    print("edge voices available of candidates:", [c for c in cand if c in names])
    for c in cand:
        if c not in names: continue
        try:
            com = edge_tts.Communicate(SCRIPT, c, rate="-8%", pitch="-2Hz")
            await com.save(str(OUT/f"edge_{c}.mp3"))
            print("  edge", c, "ok")
        except Exception as e:
            print("  edge", c, "FAILED", str(e)[:80])
try:
    asyncio.run(edge())
except Exception as e:
    print("edge-tts unavailable:", str(e)[:120])
