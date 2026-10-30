// The kiosk conversation as a state machine, kept out of the React component so the component
// is only UI. A patient is guided through: greeting -> name -> age -> sex -> complaint -> review
// -> positioning -> countdown -> 90 s scan -> registration -> spoken token + wait.
//
// Principles:
//  * Voice and touch/typing ALWAYS race — whichever answers first wins. A noisy room, a denied
//    microphone or a stopped speech server must never trap a patient; typing is the floor.
//  * Everything spoken is a pre-rendered clip (see voice.ts), so the voice never changes.
//  * The deterministic triage decides priority; this file only collects data and guides.

import {
  generateReport, getStatus, releaseCamera, scanBegin, startWebcam, transcribe, triageRun,
  type KioskStatus, type SttResult, type TriageResult,
} from "./api";
import { MAX_SPOKEN_TOKEN, SCAN_SECONDS } from "./config";
import { assessPosition, ISSUE_PROMPT, PromptThrottle, StableTimer, type Issue } from "./guidance";
import {
  captionFor, ensureMic, recordUtterance, releaseMic, say as speak, startTalkMonitor, stopSpeaking, unlockAudio,
} from "./voice";

export type Step =
  | "idle" | "name" | "age" | "sex" | "complaint" | "review"
  | "position" | "countdown" | "scan" | "registering" | "token" | "problem";

export interface Details { name: string; age: number | null; sex: string; complaint: string }

/** What the microphone produced, shown to the patient so they can see they were understood. */
export type Heard =
  | { kind: "processing" }                     // recorded, being transcribed
  | { kind: "text"; text: string }             // the words the speech model returned
  | { kind: "none" }                           // nothing usable (silence / noise)
  | { kind: "offline"; why: "mic" | "speech" } // voice unavailable: typing only
  | null;
export const EMPTY_DETAILS: Details = { name: "", age: null, sex: "", complaint: "" };

export interface UI {
  setStep(s: Step): void;
  setCaption(t: string): void;
  setSpeaking(b: boolean): void;
  setListening(b: boolean): void;
  setLevel(n: number): void;
  setIssue(i: Issue): void;
  setCandidate(name: string | null): void;
  setDetails(d: Details): void;
  setCountdown(n: number | null): void;
  setScan(p: { elapsed: number; calibrated: boolean; faceLost: boolean }): void;
  setResult(r: TriageResult | null, urgent: boolean): void;
  setFeedKey(): void;
  setNotice(t: string | null): void;
  setVoiceMode(m: "voice" | "typing"): void;
  setHeard(h: Heard): void;
}

class Cancelled extends Error {}
class Walkaway extends Error {}
const sleepMs = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

interface Gate { promise: Promise<string>; resolve: (v: string) => void }

export interface FlowOptions {
  demo?: boolean;
  scanSeconds?: number;
  /** demo rehearsals only: pre-fill the details and jump straight to positioning */
  skipIntake?: boolean;
}

export class KioskFlow {
  private ui: UI;
  private demo: boolean;
  private skipIntake: boolean;
  private scanSeconds: number;
  private cancelled = false;
  private gate: Gate | null = null;
  private mic: MediaStream | null = null;
  private sttDown = false;
  private speaking = false;
  private details: Details = { ...EMPTY_DETAILS };
  private attempt = 0;
  private lastStatus: KioskStatus | null = null;
  private demoT0 = 0;

  constructor(ui: UI, opts: FlowOptions = {}) {
    this.ui = ui;
    this.demo = !!opts.demo;
    this.skipIntake = !!opts.demo && !!opts.skipIntake;
    this.scanSeconds = opts.scanSeconds ?? SCAN_SECONDS;
  }

  // ── controls used by the UI ────────────────────────────────────────────────
  /** A typed or tapped answer from the screen. */
  submit(value: string): void { this.gate?.resolve(value); }

  editDetails(patch: Partial<Details>): void {
    this.details = { ...this.details, ...patch };
    this.ui.setDetails(this.details);
  }

  cancel(): void {
    this.cancelled = true;
    this.gate?.resolve("__cancel__");
    stopSpeaking();
    releaseMic();
    if (!this.demo) void releaseCamera();
  }

  // ── plumbing ──────────────────────────────────────────────────────────────
  private check(): void { if (this.cancelled) throw new Cancelled(); }

  private newGate(): Gate {
    let resolve!: (v: string) => void;
    const promise = new Promise<string>((r) => { resolve = r; });
    this.gate = { promise, resolve };
    return this.gate;
  }

  private async say(id: string): Promise<void> {
    this.check();
    this.ui.setCaption(captionFor(id));
    this.speaking = true;
    this.ui.setSpeaking(true);
    try { await speak(id); } finally { this.speaking = false; this.ui.setSpeaking(false); }
    this.check();
  }

  private async sayAll(ids: string[]): Promise<void> {
    for (const id of ids) { await this.say(id); await sleepMs(110); }
  }

  private async status(): Promise<KioskStatus | null> {
    if (this.demo) return this.demoStatus();
    const s = await getStatus();
    if (s) this.lastStatus = s;
    return s;
  }

  /** Scripted camera for testing the whole experience without a camera or a patient (?demo=1). */
  private demoStatus(): KioskStatus {
    const t = (performance.now() - this.demoT0) / 1000;
    const base: KioskStatus = {
      bpm: 0, rr: 0, status: "OK", is_live: true, calibration_done: false, calibration_progress: 0,
      face_detected: false, face_box: null, estimated_lux: 120, motion_delta: 1, warnings: [],
    };
    if (t > 2.5) { base.face_detected = true; base.face_box = { x: 0.43, y: 0.36, w: 0.11, h: 0.15 }; }   // too small
    if (t > 6)   { base.face_box = { x: 0.04, y: 0.25, w: 0.30, h: 0.40 }; }                                // off to one side
    if (t > 9.5) { base.face_box = { x: 0.35, y: 0.25, w: 0.30, h: 0.40 }; }                                // good
    return base;
  }

  private async listen(hint: string, signal: AbortSignal): Promise<SttResult | null> {
    if (!this.mic) return null;
    this.ui.setHeard(null);
    this.ui.setListening(true);
    try {
      const wav = await recordUtterance(this.mic, { onLevel: (l) => this.ui.setLevel(l), signal });
      if (signal.aborted || this.cancelled) return null;          // the patient typed instead
      if (!wav) { this.ui.setHeard({ kind: "none" }); return null; }
      this.ui.setListening(false);
      this.ui.setHeard({ kind: "processing" });
      const r = await transcribe(wav, hint);
      if (signal.aborted || this.cancelled) return null;
      if (!r) {
        this.sttDown = true;
        this.ui.setVoiceMode("typing");
        this.ui.setHeard({ kind: "offline", why: "speech" });
        return null;
      }
      this.ui.setHeard(r.text ? { kind: "text", text: r.text } : { kind: "none" });
      return r.text ? r : null;
    } finally {
      this.ui.setListening(false);
      this.ui.setLevel(0);
    }
  }

  /** Ask one question. Voice and touch race; invalid answers re-ask; after two voice failures we
   *  switch to typing and say so. Returns the accepted value. */
  private async ask<T>(cfg: {
    step: Step; prompt: string; retry: string; typeIt: string; hint: string;
    preface?: string;                              // spoken before the prompt, with the answer already open
    fromSpeech: (r: SttResult) => T | null;
    fromTyped: (s: string) => T | null;
    onHeard?: (v: T) => Promise<boolean>;          // confirmation; false = heard wrong, ask again
    optional?: boolean;                            // after voice fails twice, move on instead of waiting for typing
  }): Promise<T | null> {
    this.ui.setStep(cfg.step);
    let gate = this.newGate();                     // open first: a quick typist is not dropped while we speak
    if (cfg.preface) await this.say(cfg.preface);
    await this.say(cfg.prompt);
    let voiceTries = 0;
    for (;;) {
      this.check();
      const voiceOk = !!this.mic && !this.sttDown && !this.demo && voiceTries < 2;
      this.ui.setVoiceMode(voiceOk ? "voice" : "typing");
      let value: T | null = null;
      let fromVoice = false;

      if (voiceOk) {
        const ac = new AbortController();
        const voice = this.listen(cfg.hint, ac.signal).then((r) => ({ kind: "voice" as const, r }));
        const typed = gate.promise.then((v) => ({ kind: "typed" as const, v }));
        const first = await Promise.race([voice, typed]);
        if (first.kind === "typed") {
          ac.abort();
          await voice.catch(() => null);
          if (first.v === "__cancel__") throw new Cancelled();
          value = cfg.fromTyped(first.v);
          if (value === null) gate = this.newGate();
        } else {
          fromVoice = true;
          value = first.r ? cfg.fromSpeech(first.r) : null;
          if (value === null) voiceTries++;
        }
      } else {
        const v = await gate.promise;
        if (v === "__cancel__") throw new Cancelled();
        value = cfg.fromTyped(v);
        if (value === null) gate = this.newGate();
      }

      if (value !== null) {
        if (cfg.onHeard && fromVoice) {
          if (await cfg.onHeard(value)) { this.ui.setCandidate(null); return value; }
          voiceTries++;
          this.ui.setCandidate(null);
          gate = this.newGate();       // yesNo() replaced the shared gate; re-open one for the next answer
        } else {
          return value;
        }
      }
      if (voiceTries === 1 && fromVoice) await this.say(cfg.retry);
      else if (voiceTries >= 2 && voiceOk) {
        await this.say(cfg.typeIt);
        if (cfg.optional) return null;
        gate = this.gate ?? this.newGate();
      }
    }
  }

  private async yesNo(): Promise<boolean> {
    const gate = this.newGate();
    await this.say("name_check");
    const useVoice = !!this.mic && !this.sttDown && !this.demo;
    const typed = gate.promise.then((v) => (v === "yes" ? true : v === "no" ? false : null));
    if (!useVoice) return (await typed) ?? false;
    const ac = new AbortController();
    const voice = this.listen("yesno", ac.signal).then((r) => {
      const f = r?.field;
      return f === "yes" ? true : f === "no" ? false : null;
    });
    const first = await Promise.race([typed, voice]);
    ac.abort();
    if (first !== null) return first;
    return (await typed) ?? false;     // couldn't tell by voice: the on-screen buttons decide
  }

  // ── the conversation ──────────────────────────────────────────────────────
  async run(): Promise<"done" | "cancelled" | "walkaway" | "problem"> {
    try {
      this.demoT0 = performance.now();
      await unlockAudio();
      this.ui.setNotice(null);
      if (!this.demo) {
        this.mic = await ensureMic();
        if (!this.mic) this.ui.setHeard({ kind: "offline", why: "mic" });
        void startWebcam();                    // warm the camera up while the patient answers questions
      }
      this.ui.setVoiceMode(this.mic && !this.demo ? "voice" : "typing");

      if (this.skipIntake) {
        this.editDetails({ name: "Demo Patient", age: 34, sex: "female", complaint: "Mild headache for two days" });
      } else {
        const name = await this.ask<string>({
          step: "name", preface: "welcome", prompt: "ask_name", retry: "name_retry", typeIt: "name_type", hint: "name",
          fromSpeech: (r) => (typeof r.field === "string" && r.field ? r.field : null),
          fromTyped: (s) => { const t = s.trim(); return t.length >= 2 ? t.slice(0, 60) : null; },
          onHeard: async (n) => { this.ui.setCandidate(n); this.editDetails({ name: n }); return this.yesNo(); },
        });
        this.editDetails({ name: name ?? "" });

        const age = await this.ask<number>({
          step: "age", prompt: "ask_age", retry: "age_retry", typeIt: "age_type", hint: "age",
          fromSpeech: (r) => { const f = r.field as { age?: number | null } | null; return f && typeof f.age === "number" ? f.age : null; },
          fromTyped: (s) => { const n = parseInt(s, 10); return Number.isFinite(n) && n >= 0 && n <= 120 ? n : null; },
        });
        this.editDetails({ age: age ?? null });

        const sex = await this.ask<string>({
          step: "sex", prompt: "ask_sex", retry: "ask_sex", typeIt: "sex_type", hint: "sex",
          fromSpeech: (r) => (typeof r.field === "string" ? r.field : null),
          fromTyped: (s) => (["male", "female", "other", "unspecified"].includes(s) ? s : null),
        });
        this.editDetails({ sex: sex ?? "unspecified" });

        const complaint = await this.ask<string>({
          step: "complaint", prompt: "ask_complaint", retry: "go_ahead", typeIt: "complaint_skip", hint: "complaint", optional: true,
          fromSpeech: (r) => (r.text.trim().length >= 3 ? r.text.trim().slice(0, 500) : null),
          fromTyped: (s) => (s === "__skip__" ? "" : s.trim().length >= 2 ? s.trim().slice(0, 500) : null),
        });
        this.editDetails({ complaint: complaint ?? "" });
        await this.say(complaint ? "complaint_thanks" : "complaint_skip");

        // Review: a mis-heard age or name is a patient-safety issue, so the patient (or staff) sees it all
        this.ui.setStep("review");
        const g = this.newGate();
        await Promise.race([g.promise, sleepMs(14000)]);
        this.check();
      }

      for (;;) {
        const outcome = await this.measure();
        if (outcome === "retry") continue;
        break;
      }
      return "done";
    } catch (e) {
      if (e instanceof Cancelled) return "cancelled";
      if (e instanceof Walkaway) return "walkaway";
      console.error("[kiosk] flow error", e);
      this.ui.setStep("problem");
      this.ui.setNotice("Something went wrong. Please let a member of staff know.");
      try { await this.say("camera_error"); } catch { /* ignore */ }
      return "problem";
    } finally {
      releaseMic();
    }
  }

  /** positioning -> countdown -> scan -> register. Returns "retry" if the signal was too weak. */
  private async measure(): Promise<"retry" | "done"> {
    // ---- positioning ----
    this.ui.setStep("position");
    this.ui.setFeedKey();
    await this.say("settle");
    const throttle = new PromptThrottle(8000, 3500);
    const stable = new StableTimer(2000);
    const t0 = performance.now();
    this.demoT0 = t0;
    let sawFace = false;
    for (;;) {
      this.check();
      const st = await this.status();
      const now = performance.now();
      if (st?.face_detected) sawFace = true;
      const issue = st ? assessPosition(st) : "no_face";
      this.ui.setIssue(issue);
      if (stable.update(issue === "ok", now)) break;
      if (issue !== "ok") {
        const id = ISSUE_PROMPT[issue as Exclude<Issue, "ok">];
        if (throttle.allow(id, now)) { throttle.mark(id, now); await this.say(id); }
      }
      if (now - t0 > 75000) {
        if (!sawFace) { this.ui.setNotice("I couldn't see anyone, so I've paused. Touch the screen to begin again."); throw new Walkaway(); }
        break;                                    // a face but never perfect: measure anyway, quality is checked after
      }
      await sleepMs(300);
    }
    await this.say("perfect");

    // ---- countdown ----
    this.ui.setStep("countdown");
    await this.say("count_in");
    for (const [n, id] of [[3, "three"], [2, "two"], [1, "one"]] as const) {
      this.ui.setCountdown(n);
      await this.say(id);
    }
    this.ui.setCountdown(null);
    if (!this.demo) await scanBegin();            // clean measurement window: nothing from positioning leaks in

    // ---- scan ----
    this.ui.setStep("scan");
    await this.say("scan_start");
    const talk = { flag: false };
    const stopTalk = this.mic && !this.demo
      ? startTalkMonitor(this.mic, () => { if (!this.speaking) talk.flag = true; })
      : () => {};
    const scanThrottle = new PromptThrottle(14000, 7000);
    const marks = [[25, "mid_1"], [50, "mid_2"], [70, "mid_3"], [this.scanSeconds - 10, "final_secs"]] as const;
    const done = new Set<string>();
    let calibratedAt: number | null = null;
    let faceLostSince: number | null = null;
    const scanT0 = performance.now();
    const demoCalibrate = scanT0 + 2500;
    try {
      for (;;) {
        this.check();
        const st = await this.status();
        const now = performance.now();
        const calibrated = this.demo ? now >= demoCalibrate : !!st?.calibration_done;
        if (calibrated && calibratedAt === null) calibratedAt = now;
        const elapsed = calibratedAt === null ? 0 : (now - calibratedAt) / 1000;
        const lost = !st?.face_detected;
        if (lost) { faceLostSince = faceLostSince ?? now; } else faceLostSince = null;
        const faceLost = faceLostSince !== null && now - faceLostSince > 2000;
        this.ui.setScan({ elapsed, calibrated: calibratedAt !== null, faceLost });
        if (this.demo && st) this.lastStatus = { ...st, bpm: 78, rr: 14 };

        if (calibratedAt !== null && elapsed >= this.scanSeconds) break;
        if (calibratedAt === null && now - scanT0 > 45000) break;   // never calibrated: stop waiting, quality is checked below

        // gentle guidance while measuring
        let id: string | null = null;
        if (faceLost) id = "keep_in_circle";
        else if (talk.flag) id = "stay_quiet";
        else if (st && (st.motion_delta ?? 0) > 10) id = "hold_still";
        if (id && scanThrottle.allow(id, now)) { if (id === "stay_quiet") talk.flag = false; scanThrottle.mark(id, now); await this.say(id); }
        else if (id === null) {
          for (const [at, mid] of marks) {
            if (elapsed >= at && !done.has(mid) && scanThrottle.allow(mid, now)) {
              done.add(mid); scanThrottle.mark(mid, now); await this.say(mid); break;
            }
          }
        }
        await sleepMs(250);
      }
    } finally { stopTalk(); }

    // ---- finish ----
    await this.say("done");
    const last = this.lastStatus;
    if (!this.demo) { await generateReport(); await releaseCamera(); }
    const weak = !this.demo && !(last && last.bpm > 0);
    if (weak && this.attempt < 1) {
      this.attempt++;
      await this.sayAll(["weak_signal", "retry"]);
      if (!this.demo) void startWebcam();
      return "retry";
    }

    // ---- register ----
    this.ui.setStep("registering");
    this.ui.setNotice(null);
    const result = this.demo ? this.demoResult() : await triageRun({
      patient_name: this.details.name, age: this.details.age, sex: this.details.sex, complaint: this.details.complaint,
    });
    if (!result || !result.success) {
      this.ui.setStep("problem");
      this.ui.setNotice("I couldn't complete your registration. Please see the reception desk.");
      await this.say("no_token");
      return "done";
    }
    const esi = result.queue?.esi_level ?? result.clinical_assessment?.esi_level ?? 3;
    const urgent = esi <= 2 || !!result.clinical_assessment?.is_shock;
    this.ui.setResult(result, urgent);
    this.ui.setStep("token");
    await this.announce(result, urgent);
    return "done";
  }

  private demoResult(): TriageResult {
    return {
      success: true, token: "A-017", db_saved: true,
      queue: { token: "A-017", position: 4, ahead: 3, est_wait_min: 40, status: "waiting", esi_level: 3 },
      clinical_assessment: { esi_level: 3, is_shock: false },
    };
  }

  /** Speak the token, then the situation. The patient's number is read in the SAME voice, from
   *  pre-rendered clips; beyond the clips we have, the number is on screen. */
  private async announce(r: TriageResult, urgent: boolean): Promise<void> {
    const seq = r.token ? parseInt(r.token.replace(/\D/g, ""), 10) : NaN;
    const ids: string[] = [];
    if (Number.isFinite(seq) && seq >= 1 && seq <= MAX_SPOKEN_TOKEN) ids.push(`tok_A${String(seq).padStart(3, "0")}`);
    if (urgent) {
      ids.push("handoff_urgent");
    } else {
      const q = r.queue;
      if (q) {
        if (q.ahead >= 0 && q.ahead <= 15) ids.push(`ahead_${q.ahead}`);
        const w = q.est_wait_min;
        if (w <= 5) ids.push("wait_soon");
        else if (w <= 120) ids.push(`wait_${Math.ceil(w / 5) * 5}`);
        else ids.push("wait_long");
      }
      ids.push("please_take_seat", "stay_seated");
    }
    await this.sayAll(ids);
  }
}
