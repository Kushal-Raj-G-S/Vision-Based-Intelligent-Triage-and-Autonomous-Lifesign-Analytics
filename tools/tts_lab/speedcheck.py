import asyncio
from pathlib import Path
import soundfile as sf, edge_tts
OUT = Path(__file__).parent / "samples" / "final"
LINE = ("Hello, and welcome. Please sit upright, look at the screen, and breathe normally. "
        "Thank you, Manoj. You're doing really well. All done. A member of our team will be with you shortly.")
async def go():
    for tag, rate in (("C", "+10%"),):
        mp3 = OUT / f"tmp_{tag}.mp3"
        await edge_tts.Communicate(f"Option {tag}. {LINE}", "en-GB-SoniaNeural", rate=rate, pitch="-2Hz").save(str(mp3))
        s, sr = sf.read(str(mp3)); mp3.unlink()
        sf.write(OUT / f"sonia_option_{tag}_rate_plus10pct.wav", s, sr)
        print(tag, rate, f"{len(s)/sr:.1f}s")
asyncio.run(go())
