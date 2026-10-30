export const BACKEND_URL = "http://127.0.0.1:5002";

/** Measurement window after calibration completes — same 90 s the staff dashboard uses
 *  (RR needs ~60 s of signal before its first reading, plus margin; see page.tsx). */
export const SCAN_SECONDS = 90;

/** Pre-rendered voice clips: public/voice/<voice>/<id>.mp3 (see tools/tts_lab/render_prompts.py). */
export const VOICE_PRIMARY = "/voice/sonia";
export const VOICE_BACKUP = "/voice/neerja";

/** Largest token we have a pre-rendered spoken clip for (tok_A001 .. tok_A150). */
export const MAX_SPOKEN_TOKEN = 150;

/** Idle time on the token screen before the kiosk resets for the next person. */
export const TOKEN_SCREEN_SECONDS = 30;
