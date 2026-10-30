"""Pre-render every FIXED kiosk line once, in the primary and backup voice.
Dynamic lines (the person's name, results) are NOT here — they need run-time synthesis."""
import asyncio, json
from pathlib import Path
import edge_tts

HERE = Path(__file__).parent
CFG = json.load(open(HERE / "voice_config.json"))

# id -> (text, when it plays)
P = {
 # --- welcome & name ---
 "welcome":        ("Hello, and welcome. I'm going to take a quick, contactless health check. It takes about a minute and a half, and you won't need to touch anything.", "session start"),
 "ask_name":       ("To begin, could you please tell me your name?", "after welcome"),
 "name_retry":     ("Sorry, I didn't quite catch that. Could you say your name once more?", "STT empty/low confidence, 1st time"),
 "name_type":      ("No problem. You can type your name on the screen instead.", "STT failed twice / mic unavailable"),
 "name_confirm_q": ("Did I get that right?", "after showing the heard name on screen"),
 "name_thanks":    ("Thank you. Lovely to meet you.", "name confirmed"),
 # --- positioning ---
 "settle":         ("Let's get you comfortable. Please sit upright and look at the screen.", "start of positioning"),
 "no_face":        ("I can't see your face just yet. Could you look towards the camera, please?", "face not detected"),
 "closer":         ("A little closer, please.", "face too small"),
 "farther":        ("Could you move back just a little?", "face too large"),
 "move_left":      ("Please move a little to your left.", "face off-centre (CHECK mirror direction at integration)"),
 "move_right":     ("Please move a little to your right.", "face off-centre (CHECK mirror direction at integration)"),
 "keep_in_circle": ("Please keep your face inside the circle.", "drifting out of guide oval"),
 "perfect":        ("Perfect. Hold just like that.", "position good"),
 "more_light":     ("It's a little dark here. Could you face a light, or move somewhere brighter?", "lux < 50"),
 "too_bright":     ("It's quite bright. Try turning a little away from the light.", "lux > 210"),
 "hold_still":     ("Please try to stay as still as you can.", "motion above threshold"),
 "stay_quiet":     ("Please try not to talk during the scan. Just breathe naturally.", "speech/mouth motion detected"),
 "remove_cover":   ("If you're wearing a mask or covering your face, please take it off for the scan.", "face found but low signal quality"),
 # --- scan ---
 "count_in":       ("I'll start in just a moment.", "before countdown"),
 "three":          ("Three.", "countdown"),
 "two":            ("Two.", "countdown"),
 "one":            ("One.", "countdown"),
 "scan_start":     ("Starting now. Breathe normally, and stay as still as you can.", "scan begins"),
 "mid_1":          ("You're doing really well. Just keep breathing naturally.", "~25 s"),
 "mid_2":          ("Nice and steady. You're doing great.", "~50 s"),
 "mid_3":          ("Not much longer now.", "~70 s"),
 "final_secs":     ("Just a few more seconds.", "~last 10 s"),
 # --- finish ---
 "done":           ("All done. Thank you so much for your patience.", "scan complete"),
 "handoff_normal": ("A member of our team will be with you shortly.", "routine result"),
 "handoff_urgent": ("Please stay where you are. A member of our team is on their way to you.", "high-priority / shock flag"),
 "goodbye":        ("Take care.", "end"),
 # --- intake (age / sex / complaint) ---
 "name_check":     ("Please check your name on the screen. Is that right?", "name heard, waiting for yes/no"),
 "ask_age":        ("Thank you. And how old are you?", "after name confirmed"),
 "age_retry":      ("Sorry, I didn't catch your age. Could you say it once more?", "STT could not parse an age"),
 "age_type":       ("No problem. You can enter your age on the screen.", "age fallback to typing"),
 "ask_sex":        ("Are you male, female, or would you rather not say?", "after age"),
 "sex_type":       ("No problem. You can choose on the screen.", "sex fallback to tapping"),
 "ask_complaint":  ("Now, in your own words, what brings you in today?", "after sex"),
 "complaint_thanks": ("Thank you. I've noted that down.", "complaint captured"),
 "complaint_skip": ("That's fine. You can tell the nurse when you're called.", "complaint skipped/failed"),
 "go_ahead":       ("Go ahead, I'm listening.", "mic opens"),
 "tap_or_say":     ("You can tap the screen, or just say it.", "gentle hint on confirm steps"),
 # --- after the scan: token + wait ---
 "please_take_seat": ("Please take a seat. We'll call your number when it's your turn.", "after token"),
 "stay_seated":    ("Please stay seated and keep an eye on the screen. It will show when it's your turn.", "after token"),
 "wait_soon":      ("You should be seen very shortly.", "est wait <= 5 min"),
 "wait_long":      ("It may be longer than two hours. Please ask at the desk if you'd like an update.", "est wait > 120 min"),
 "no_token":       ("I couldn't print your number, but you're registered. Please tell the desk your name.", "DB/token failure"),
 # --- problems ---
 "weak_signal":    ("I'm sorry, I couldn't get a clear reading. Shall we try once more?", "scan produced no usable signal"),
 "retry":          ("Let's try that again. Please look at the screen and stay still.", "retry"),
 "camera_error":   ("I can't reach the camera right now. Please let a member of staff know.", "camera failure"),
 "mic_error":      ("I can't hear you at the moment. You can type your name on the screen instead.", "mic denied/unavailable"),
}

# Token numbers, wait estimates and queue counts are spoken from pre-rendered clips too, so
# the person's number is read in the SAME voice with no runtime synthesis.
for n in range(1, 151):
    d = f"{n:03d}"
    P[f"tok_A{d}"] = (f"Your number is A, {', '.join({'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine'}[c] for c in d)}.", f"token A-{d}")
for m in range(5, 125, 5):
    P[f"wait_{m}"] = (f"Your estimated wait is about {m} minutes.", f"est wait {m} min")
for k in range(0, 16):
    P[f"ahead_{k}"] = ("There is nobody ahead of you." if k == 0 else
                       ("There is one person ahead of you." if k == 1 else f"There are {k} people ahead of you."),
                       f"{k} ahead")


async def render(tag, v):
    out = HERE / "prompts" / tag; out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(8)

    async def one(pid, text):
        f = out / f"{pid}.mp3"
        if f.exists() and f.stat().st_size > 1000:
            return
        async with sem:
            for attempt in range(4):
                try:
                    await edge_tts.Communicate(text, v["voice"], rate=v["rate"], pitch=v["pitch"]).save(str(f)); return
                except Exception as e:
                    if attempt == 3: print("FAILED", tag, pid, str(e)[:80])
                    await asyncio.sleep(1.5 * (attempt + 1))

    await asyncio.gather(*(one(pid, text) for pid, (text, _) in P.items()))
    print(tag, "rendered", len(list(out.glob('*.mp3'))), "files", flush=True)


async def main():
    await render("sonia_primary", CFG["primary"])
    await render("neerja_backup", CFG["backup"])
    json.dump({k: {"text": t, "when": w} for k, (t, w) in P.items()}, open(HERE/"prompts"/"manifest.json", "w"), indent=1, ensure_ascii=False)
asyncio.run(main())
