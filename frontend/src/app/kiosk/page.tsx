"use client";

// Patient-facing, voice-guided intake kiosk. Open /kiosk on the screen the patient stands at
// (/kiosk?demo=1 runs the whole experience with a scripted camera for rehearsals/testing).
// All logic lives in src/lib/kiosk/flow.ts — this file is presentation only.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./kiosk.css";
import { BACKEND_URL, SCAN_SECONDS, TOKEN_SCREEN_SECONDS } from "@/lib/kiosk/config";
import { getStatus, type TriageResult } from "@/lib/kiosk/api";
import { EMPTY_DETAILS, KioskFlow, type Details, type Heard, type Step, type UI } from "@/lib/kiosk/flow";
import type { Issue } from "@/lib/kiosk/guidance";
import { loadCaptions } from "@/lib/kiosk/voice";

const ISSUE_TEXT: Record<Issue, { title: string; hint: string }> = {
  no_face: { title: "I can't see you yet", hint: "Please look at the screen." },
  closer: { title: "A little closer", hint: "Move a little towards the screen." },
  farther: { title: "A little further back", hint: "Move back from the screen a little." },
  move_left: { title: "Move a little to your left", hint: "Keep your face inside the oval." },
  move_right: { title: "Move a little to your right", hint: "Keep your face inside the oval." },
  keep_in_circle: { title: "Keep your face in the oval", hint: "Sit upright and look at the screen." },
  more_light: { title: "It's a little dark", hint: "Face a light, or move somewhere brighter." },
  too_bright: { title: "It's very bright", hint: "Turn a little away from the light." },
  hold_still: { title: "Hold still for a moment", hint: "Try not to move your head." },
  ok: { title: "Perfect", hint: "Hold just like that." },
};

const STEP_ORDER: { id: Step[]; label: string }[] = [
  { id: ["name", "age", "sex", "complaint", "review"], label: "Your details" },
  { id: ["position", "countdown"], label: "Get ready" },
  { id: ["scan"], label: "Health check" },
  { id: ["registering", "token", "problem"], label: "Your number" },
];

function Icon({ name }: { name: "mic" | "keyboard" | "check" }) {
  const p = { width: 20, height: 20, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  if (name === "mic") return (<svg {...p}><rect x="9" y="2" width="6" height="12" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v4" /></svg>);
  if (name === "keyboard") return (<svg {...p}><rect x="2" y="6" width="20" height="12" rx="2" /><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10" /></svg>);
  return (<svg {...p}><path d="M20 6 9 17l-5-5" /></svg>);
}

/** Stand-in for the camera in ?demo=1: a silhouette that moves/resizes with the scripted guidance. */
function DemoPerson({ issue }: { issue: Issue }) {
  const t: Record<Issue, { s: number; x: number; o: number }> = {
    no_face: { s: 1, x: 0, o: 0 }, closer: { s: 0.5, x: 0, o: 1 }, farther: { s: 1.5, x: 0, o: 1 },
    move_left: { s: 1, x: 105, o: 1 }, move_right: { s: 1, x: -105, o: 1 }, keep_in_circle: { s: 1, x: 0, o: 1 },
    more_light: { s: 1, x: 0, o: 0.6 }, too_bright: { s: 1, x: 0, o: 1 }, hold_still: { s: 1, x: 0, o: 1 }, ok: { s: 1, x: 0, o: 1 },
  };
  const v = t[issue];
  return (
    <svg viewBox="0 0 400 300" className="absolute inset-0 w-full h-full" aria-hidden>
      <rect width="400" height="300" fill="#2a1d15" />
      <g style={{ transform: `translateX(${v.x}px) scale(${v.s})`, transformOrigin: "200px 150px", transition: "transform 0.8s ease, opacity 0.5s ease", opacity: v.o }}>
        <ellipse cx="200" cy="140" rx="46" ry="58" fill="#8a5a3c" />
        <path d="M110 300c4-62 40-84 90-84s86 22 90 84z" fill="#6b4329" />
      </g>
      <text x="200" y="288" textAnchor="middle" fill="rgba(255,240,228,0.5)" fontSize="13">Demo camera view</text>
    </svg>
  );
}

/** Under the orb: what the microphone just picked up, so the patient can see they were heard. */
function HeardPanel({ heard, listening, level }: { heard: Heard; listening: boolean; level: number }) {
  if (listening) {
    return (
      <div className="kiosk-heard live" role="status" aria-live="polite">
        <span className="kiosk-meter" aria-hidden>
          {[0, 1, 2, 3, 4, 5, 6].map((i) => (
            <span key={i} style={{ transform: `scaleY(${Math.max(0.15, Math.min(1, level * (1.6 - Math.abs(i - 3) * 0.28)))})` }} />
          ))}
        </span>
        <span>Listening… speak now</span>
      </div>
    );
  }
  if (!heard) return <div className="kiosk-heard empty" aria-hidden />;
  if (heard.kind === "processing") return <div className="kiosk-heard" role="status" aria-live="polite"><span className="kiosk-dots" aria-hidden><i /><i /><i /></span>Understanding what you said…</div>;
  if (heard.kind === "none") return <div className="kiosk-heard warn" role="status" aria-live="polite">I didn&apos;t catch that. Please speak a little louder, or use the screen.</div>;
  if (heard.kind === "offline") return (
    <div className="kiosk-heard warn" role="status" aria-live="polite">
      {heard.why === "mic" ? "The microphone isn't available, so please use the screen." : "Voice answers aren't available right now, so please use the screen."}
    </div>
  );
  return (
    <div className="kiosk-heard said" role="status" aria-live="polite">
      <span className="kiosk-heard-k">I heard</span>
      <span className="kiosk-heard-q">“{heard.text}”</span>
    </div>
  );
}

/** The answers collected so far, so the patient can watch them being recorded. */
function AnswersSoFar({ d }: { d: Details }) {
  const rows: [string, string][] = [];
  if (d.name) rows.push(["Name", d.name]);
  if (d.age != null) rows.push(["Age", String(d.age)]);
  if (d.sex) rows.push(["Sex", d.sex === "unspecified" ? "Prefer not to say" : d.sex[0].toUpperCase() + d.sex.slice(1)]);
  if (d.complaint) rows.push(["Reason", d.complaint]);
  if (!rows.length) return null;
  return (
    <dl className="kiosk-answers">
      {rows.map(([k, v]) => (
        <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
      ))}
    </dl>
  );
}

function Orb({ state, level }: { state: "idle" | "speaking" | "listening"; level: number }) {
  return <div className={`kiosk-orb ${state}`} style={{ ["--lvl" as string]: level }} aria-hidden />;
}

function TypedField({ placeholder, type = "text", onSubmit, cta = "Continue", multiline = false }: {
  placeholder: string; type?: string; onSubmit: (v: string) => void; cta?: string; multiline?: boolean;
}) {
  const [v, setV] = useState("");
  const go = () => { if (v.trim()) onSubmit(v.trim()); };
  return (
    <div className="flex flex-col gap-3">
      {multiline ? (
        <textarea className="kiosk-input" rows={3} value={v} placeholder={placeholder} onChange={(e) => setV(e.target.value)} />
      ) : (
        <input className="kiosk-input" type={type} inputMode={type === "number" ? "numeric" : undefined} value={v} placeholder={placeholder}
               onChange={(e) => setV(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") go(); }} />
      )}
      <button className="kiosk-btn primary" onClick={go} disabled={!v.trim()}>{cta}</button>
    </div>
  );
}

function ScanRing({ elapsed, total, calibrated }: { elapsed: number; total: number; calibrated: boolean }) {
  const r = 54, c = 2 * Math.PI * r;
  const frac = calibrated ? Math.min(1, elapsed / total) : 0;
  const left = Math.max(0, Math.ceil(total - elapsed));
  return (
    <div className="relative w-44 h-44" role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={Math.round(elapsed)}>
      <svg viewBox="0 0 128 128" className="w-full h-full -rotate-90">
        <circle cx="64" cy="64" r={r} fill="none" stroke="var(--line-2)" strokeWidth="9" />
        <circle cx="64" cy="64" r={r} fill="none" stroke="url(#kg)" strokeWidth="9" strokeLinecap="round"
                strokeDasharray={c} strokeDashoffset={c * (1 - frac)} style={{ transition: "stroke-dashoffset 0.4s linear" }} />
        <defs><linearGradient id="kg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#ff9a3d" /><stop offset="1" stopColor="#f26a1b" /></linearGradient></defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className="text-5xl font-semibold tabular-nums">{calibrated ? left : "…"}</div>
        <div className="kiosk-dim text-sm mt-1">{calibrated ? "seconds" : "getting ready"}</div>
      </div>
    </div>
  );
}

export default function KioskPage() {
  const [step, setStep] = useState<Step>("idle");
  const [details, setDetails] = useState<Details>({ ...EMPTY_DETAILS });
  const [caption, setCaption] = useState("");
  const [speaking, setSpeaking] = useState(false);
  const [listening, setListening] = useState(false);
  const [level, setLevel] = useState(0);
  const [issue, setIssue] = useState<Issue>("no_face");
  const [candidate, setCandidate] = useState<string | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [scan, setScan] = useState({ elapsed: 0, calibrated: false, faceLost: false });
  const [result, setResult] = useState<TriageResult | null>(null);
  const [urgent, setUrgent] = useState(false);
  const [tokenAt, setTokenAt] = useState(0);
  const [now, setNow] = useState(() => Date.now());
  const [notice, setNotice] = useState<string | null>(null);
  const [voiceMode, setVoiceMode] = useState<"voice" | "typing">("voice");
  const [heard, setHeard] = useState<Heard>(null);
  const [feedKey, setFeedKey] = useState(0);
  const [feedOk, setFeedOk] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);
  const [demo, setDemo] = useState(false);
  const [skip, setSkip] = useState(false);

  const flowRef = useRef<KioskFlow | null>(null);

  const ui = useMemo<UI>(() => ({
    setStep, setCaption, setSpeaking, setListening, setLevel, setIssue, setCandidate,
    setDetails: (d) => setDetails({ ...d }),
    setCountdown, setScan,
    setResult: (r, u) => { setResult(r); setUrgent(u); setTokenAt(Date.now()); },
    setFeedKey: () => { setFeedOk(false); setFeedKey((k) => k + 1); },
    setNotice, setVoiceMode, setHeard,
  }), []);

  const goIdle = useCallback((keepNotice = false) => {
    flowRef.current?.cancel();
    flowRef.current = null;
    setStep("idle"); setCaption(""); setSpeaking(false); setListening(false); setLevel(0); setCandidate(null); setHeard(null);
    setCountdown(null); setResult(null); setUrgent(false); setDetails({ ...EMPTY_DETAILS });
    setScan({ elapsed: 0, calibrated: false, faceLost: false });
    if (!keepNotice) setNotice(null);
  }, []);

  const start = useCallback(() => {
    if (flowRef.current) return;
    setNotice(null);
    const flow = new KioskFlow(ui, { demo, skipIntake: skip, scanSeconds: demo ? 12 : SCAN_SECONDS });
    flowRef.current = flow;
    void flow.run().then((outcome) => {
      if (flowRef.current !== flow) return;                  // already reset by the user / timer
      if (outcome === "walkaway") goIdle(true);
    });
  }, [demo, skip, ui, goIdle]);

  useEffect(() => {
    void loadCaptions();
    // Reading the URL once on load is a sync with an external system (the address bar), not derived
    // state; initialising from window during render would cause a server/client hydration mismatch.
    /* eslint-disable react-hooks/set-state-in-effect */
    const q = new URLSearchParams(window.location.search);
    setDemo(q.get("demo") === "1");
    setSkip(q.get("skip") === "1");
    /* eslint-enable react-hooks/set-state-in-effect */
    return () => { flowRef.current?.cancel(); };
  }, []);

  // backend reachability, shown on the attract screen so staff notice a dead backend before a patient does
  useEffect(() => {
    if (step !== "idle") return;
    let alive = true;
    const check = async () => { const s = await getStatus(); if (alive) setOnline(!!s); };
    void check();
    const iv = setInterval(check, 5000);
    return () => { alive = false; clearInterval(iv); };
  }, [step]);

  // token screen: tick once a second, and return to the attract screen when the time is up
  useEffect(() => {
    if (step !== "token") return;
    const iv = setInterval(() => {
      const n = Date.now();
      setNow(n);
      if (n - tokenAt >= TOKEN_SCREEN_SECONDS * 1000) goIdle();
    }, 1000);
    return () => clearInterval(iv);
  }, [step, tokenAt, goIdle]);
  const resetIn = Math.max(0, TOKEN_SCREEN_SECONDS - Math.floor((now - tokenAt) / 1000));

  const submit = (v: string) => flowRef.current?.submit(v);
  const orbState = listening ? "listening" : speaking ? "speaking" : "idle";
  const showCamera = step === "position" || step === "countdown" || step === "scan";
  const activeIdx = STEP_ORDER.findIndex((s) => s.id.includes(step));

  // ───────────────────────────── attract screen ─────────────────────────────
  if (step === "idle") {
    return (
      <main className="kiosk-root relative flex items-center justify-center p-8 cursor-pointer select-none" onClick={start}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") start(); }} tabIndex={0} role="button" aria-label="Touch to begin">
        <div className="flex flex-col items-center text-center gap-10 kiosk-fade-in max-w-3xl">
          <Orb state="idle" level={0} />
          <div className="flex flex-col gap-4">
            <div className="kiosk-eyebrow">VITAL · Contactless health check</div>
            <h1 className="kiosk-title text-6xl md:text-7xl">Welcome</h1>
            <p className="kiosk-dim text-2xl md:text-3xl">Touch the screen to begin. It takes about three minutes, and you won&apos;t need to touch anything else.</p>
          </div>
          <div className="kiosk-btn primary flex items-center justify-center text-2xl px-10" style={{ minHeight: "4.5rem" }}>Touch to begin</div>
          {notice && <div className="kiosk-chip warn text-lg" role="status">{notice}</div>}
          <p className="kiosk-dim text-base max-w-xl">
            Your camera and microphone are used only during your check. Your name and answers are used only to register you for care.
          </p>
          {online === false && <div className="kiosk-chip warn" role="alert">The system is starting up or offline. Please see the reception desk.</div>}
          {demo && <div className="kiosk-chip">Demo mode — scripted camera, nothing is saved</div>}
        </div>
      </main>
    );
  }

  // ───────────────────────────── guided screens ─────────────────────────────
  return (
    <main className="kiosk-root flex flex-col">
      <header className="flex items-center justify-between px-8 pt-6">
        <div className="flex items-center gap-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/vital_logo.png" alt="" width={40} height={40} className="rounded-lg" />
          <span className="kiosk-title text-2xl">VITAL</span>
        </div>
        <ol className="hidden md:flex items-center gap-2" aria-label="Progress">
          {STEP_ORDER.map((s, i) => (
            <li key={s.label} className={`kiosk-chip ${i < activeIdx ? "good" : i === activeIdx ? "" : "kiosk-dim"}`} aria-current={i === activeIdx ? "step" : undefined}
                style={i === activeIdx ? { borderColor: "var(--brand)", color: "var(--brand)" } : undefined}>
              {i < activeIdx ? <Icon name="check" /> : <span className="tabular-nums">{i + 1}</span>}{s.label}
            </li>
          ))}
        </ol>
        <div className="flex items-center gap-3">
          <span className={`kiosk-chip ${voiceMode === "voice" ? "good" : ""}`}>
            <Icon name={voiceMode === "voice" ? "mic" : "keyboard"} />{voiceMode === "voice" ? "Voice on" : "Using the screen"}
          </span>
          <button className="kiosk-btn" style={{ minHeight: "2.8rem", fontSize: "1rem" }} onClick={() => goIdle()}>Start over</button>
        </div>
      </header>

      <section className="flex-1 grid lg:grid-cols-[1.1fr_1fr] gap-8 p-8 items-center">
        {/* ── left: orb or camera ── */}
        <div className="kiosk-glass rounded-[2rem] p-6 flex flex-col items-center justify-center min-h-[22rem]">
          {showCamera ? (
            <div className="kiosk-frame w-full">
              {demo ? <DemoPerson issue={issue} /> : (<>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img key={feedKey} className="kiosk-feed" src={`${BACKEND_URL}/video_feed?clean=1&t=${feedKey}`} alt="Your camera view"
                     onLoad={() => setFeedOk(true)} onError={() => setFeedOk(false)} />
                {!feedOk && <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-xl" style={{ color: "#f7e9de" }}><span className="kiosk-dots" aria-hidden><i /><i /><i /></span>Starting the camera…</div>}
              </>)}
              <svg viewBox="0 0 400 300" className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden>
                <path fillRule="evenodd" fill="rgba(28,18,12,0.5)" d="M0 0H400V300H0Z M118 140a82 108 0 1 0 164 0a82 108 0 1 0 -164 0Z" />
                <ellipse className={`kiosk-oval ${issue === "ok" || step === "scan" ? "ok" : "adjust"}`} cx="200" cy="140" rx="82" ry="108" fill="none" strokeWidth="4" />
                {step === "scan" && (
                  /* scan progress travels round the oval, starting at the top */
                  <ellipse cx="200" cy="140" rx="82" ry="108" fill="none" stroke="var(--brand)" strokeWidth="7" strokeLinecap="round"
                           pathLength={100} strokeDasharray={`${scan.calibrated ? Math.min(100, (scan.elapsed / (demo ? 12 : SCAN_SECONDS)) * 100) : 0} 100`}
                           transform="rotate(-90 200 140)" style={{ transition: "stroke-dasharray 0.4s linear" }} />
                )}
              </svg>
              {step === "scan" && (
                <div className="kiosk-live-tag" aria-hidden><span />Measuring</div>
              )}
              {step === "countdown" && countdown !== null && (
                <div className="absolute inset-0 flex items-center justify-center" style={{ background: "rgba(28,18,12,0.42)" }}>
                  <div key={countdown} className="kiosk-token kiosk-fade-in" style={{ fontSize: "12rem" }}>{countdown}</div>
                </div>
              )}
            </div>
          ) : step === "registering" ? (
            <div className="flex flex-col items-center gap-6">
              <svg width="96" height="96" viewBox="0 0 50 50" className="kiosk-spin" aria-hidden>
                <circle cx="25" cy="25" r="20" fill="none" stroke="var(--line-2)" strokeWidth="4" />
                <path d="M25 5a20 20 0 0 1 20 20" fill="none" stroke="var(--brand)" strokeWidth="4" strokeLinecap="round" />
              </svg>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-6 w-full">
              <Orb state={orbState} level={level} />
              <div className="flex items-center gap-3 kiosk-dim text-lg">
                <span className={`kiosk-bars ${speaking ? "on" : ""}`} aria-hidden><span /><span /><span /><span /></span>
                {listening ? "Your turn to speak" : speaking ? "Speaking…" : voiceMode === "voice" ? "Your turn" : "Please use the screen"}
              </div>
              <HeardPanel heard={heard} listening={listening} level={level} />
              <AnswersSoFar d={details} />
            </div>
          )}
        </div>

        {/* ── right: the current step ── */}
        <div className="flex flex-col gap-6 kiosk-fade-in" key={step}>
          {step === "name" && (<>
            <div className="kiosk-eyebrow">Step 1 of 4 · Your details</div>
            <h2 className="kiosk-title text-5xl">What&apos;s your name?</h2>
            {candidate ? (
              <div className="kiosk-glass rounded-3xl p-6 flex flex-col gap-5">
                <div className="kiosk-dim text-lg">I heard</div>
                <div className="kiosk-title text-5xl">{candidate}</div>
                <div className="flex gap-3">
                  <button className="kiosk-btn primary flex-1" onClick={() => submit("yes")}>Yes, that&apos;s right</button>
                  <button className="kiosk-btn flex-1" onClick={() => submit("no")}>No, try again</button>
                </div>
              </div>
            ) : (<>
              <p className="kiosk-dim text-xl">Say it out loud, or type it below.</p>
              <TypedField placeholder="Your full name" onSubmit={submit} />
            </>)}
          </>)}

          {step === "age" && (<>
            <div className="kiosk-eyebrow">Step 1 of 4 · Your details</div>
            <h2 className="kiosk-title text-5xl">How old are you?</h2>
            <p className="kiosk-dim text-xl">Say your age, or type it.</p>
            <TypedField placeholder="Age in years" type="number" onSubmit={submit} />
          </>)}

          {step === "sex" && (<>
            <div className="kiosk-eyebrow">Step 1 of 4 · Your details</div>
            <h2 className="kiosk-title text-5xl">Which applies to you?</h2>
            <p className="kiosk-dim text-xl">Say it, or choose below.</p>
            <div className="grid grid-cols-2 gap-3">
              {[["male", "Male"], ["female", "Female"], ["other", "Other"], ["unspecified", "Prefer not to say"]].map(([v, l]) => (
                <button key={v} className="kiosk-btn" style={{ minHeight: "4.5rem" }} onClick={() => submit(v)}>{l}</button>
              ))}
            </div>
          </>)}

          {step === "complaint" && (<>
            <div className="kiosk-eyebrow">Step 1 of 4 · Your details</div>
            <h2 className="kiosk-title text-5xl">What brings you in today?</h2>
            <p className="kiosk-dim text-xl">Tell me in your own words, or type it. You can skip this.</p>
            <TypedField placeholder="For example: chest pain, fever, headache…" multiline onSubmit={submit} />
            <button className="kiosk-btn" onClick={() => submit("__skip__")}>Skip</button>
          </>)}

          {step === "review" && (<>
            <div className="kiosk-eyebrow">Please check</div>
            <h2 className="kiosk-title text-5xl">Is this all correct?</h2>
            <div className="kiosk-glass rounded-3xl p-6 flex flex-col gap-4">
              <label className="flex flex-col gap-1"><span className="kiosk-dim">Name</span>
                <input className="kiosk-input" value={details.name} onChange={(e) => flowRef.current?.editDetails({ name: e.target.value })} /></label>
              <div className="grid grid-cols-2 gap-4">
                <label className="flex flex-col gap-1"><span className="kiosk-dim">Age</span>
                  <input className="kiosk-input" inputMode="numeric" value={details.age ?? ""}
                         onChange={(e) => { const n = parseInt(e.target.value, 10); flowRef.current?.editDetails({ age: Number.isFinite(n) ? Math.min(120, Math.max(0, n)) : null }); }} /></label>
                <label className="flex flex-col gap-1"><span className="kiosk-dim">Sex</span>
                  <select className="kiosk-input" value={details.sex || "unspecified"} onChange={(e) => flowRef.current?.editDetails({ sex: e.target.value })}>
                    <option value="male">Male</option><option value="female">Female</option><option value="other">Other</option><option value="unspecified">Prefer not to say</option>
                  </select></label>
              </div>
              <label className="flex flex-col gap-1"><span className="kiosk-dim">What brings you in</span>
                <textarea className="kiosk-input" rows={2} value={details.complaint} onChange={(e) => flowRef.current?.editDetails({ complaint: e.target.value })} /></label>
            </div>
            <button className="kiosk-btn primary" style={{ minHeight: "4.5rem", fontSize: "1.35rem" }} onClick={() => submit("ok")}>Yes, continue</button>
          </>)}

          {step === "position" && (<>
            <div className="kiosk-eyebrow">Step 2 of 4 · Get ready</div>
            <h2 className="kiosk-title text-5xl">{ISSUE_TEXT[issue].title}</h2>
            <p className="kiosk-dim text-2xl">{ISSUE_TEXT[issue].hint}</p>
            <div className="flex flex-wrap gap-2">
              <span className={`kiosk-chip ${issue === "ok" ? "good" : "warn"}`}>{issue === "ok" ? "Position looks good" : "Adjusting…"}</span>
              <span className="kiosk-chip">Sit upright · look at the screen · stay quiet</span>
            </div>
          </>)}

          {step === "countdown" && (<>
            <div className="kiosk-eyebrow">Step 2 of 4 · Get ready</div>
            <h2 className="kiosk-title text-5xl">Here we go</h2>
            <p className="kiosk-dim text-2xl">Breathe normally and stay as still as you can.</p>
          </>)}

          {step === "scan" && (
            <div className="flex flex-col gap-6 items-start">
              <div className="kiosk-eyebrow">Step 3 of 4 · Health check</div>
              <h2 className="kiosk-title text-5xl">Breathe normally.<br />Stay still.</h2>
              <ScanRing elapsed={scan.elapsed} total={demo ? 12 : SCAN_SECONDS} calibrated={scan.calibrated} />
              <div className="flex flex-wrap gap-2">
                {scan.faceLost
                  ? <span className="kiosk-chip warn" role="status">Please keep your face in the oval</span>
                  : <span className="kiosk-chip good" role="status">Measuring · looking good</span>}
                <span className="kiosk-chip">Please don&apos;t talk during the check</span>
              </div>
            </div>
          )}

          {step === "registering" && (<>
            <div className="kiosk-eyebrow">Step 4 of 4 · Your number</div>
            <h2 className="kiosk-title text-5xl">Thank you.<br />Just a moment…</h2>
            <p className="kiosk-dim text-2xl">I&apos;m registering you.</p>
          </>)}

          {step === "problem" && (<>
            <h2 className="kiosk-title text-5xl">Please see the desk</h2>
            <p className="kiosk-dim text-2xl">{notice ?? "Something went wrong. A member of staff will help you."}</p>
            <button className="kiosk-btn primary" onClick={() => goIdle()}>Start over</button>
          </>)}
        </div>
      </section>

      {/* ── token screen takes the whole stage ── */}
      {step === "token" && result && (
        <div className="fixed inset-0 z-20 flex items-center justify-center p-8 kiosk-root kiosk-solid kiosk-fade-in">
          <div className="kiosk-glass rounded-[2.5rem] p-12 max-w-4xl w-full flex flex-col items-center text-center gap-8">
            <div className="kiosk-eyebrow">Registered · Your number</div>
            <div className="kiosk-token" style={{ fontSize: "min(22vw, 13rem)" }} aria-label={`Your number is ${result.token ?? "not available"}`}>{result.token ?? "—"}</div>
            {urgent ? (
              <p className="text-3xl kiosk-title">Please stay where you are.<br /><span className="kiosk-dim text-2xl font-normal">A member of our team is on their way to you.</span></p>
            ) : (<>
              <div className="flex flex-wrap justify-center gap-3">
                {result.queue && <span className="kiosk-chip good text-xl">{result.queue.ahead === 0 ? "You're next" : `${result.queue.ahead} ahead of you`}</span>}
                {result.queue && <span className="kiosk-chip text-xl">{result.queue.est_wait_min <= 5 ? "Seen very shortly" : `About ${result.queue.est_wait_min} min`}</span>}
              </div>
              <p className="kiosk-dim text-2xl">Please take a seat. Watch the waiting-room screen — it will show when it&apos;s your number.</p>
            </>)}
            <p className="kiosk-dim text-base">Wait times are estimates and change when more urgent cases arrive.</p>
            <p className="text-xl min-h-[2rem]" aria-live="polite">{caption}</p>
            <div className="flex items-center gap-4">
              <span className="kiosk-dim">This screen resets in {resetIn}s</span>
              <button className="kiosk-btn" onClick={() => goIdle()}>Done</button>
            </div>
          </div>
        </div>
      )}

      {/* ── subtitles: exactly what is being said ── */}
      <footer className="px-8 pb-6">
        <div className="kiosk-glass rounded-2xl px-6 py-4 flex items-center gap-4 min-h-[4.2rem]" aria-live="polite">
          <span className={`kiosk-bars ${speaking ? "on" : ""}`} aria-hidden><span /><span /><span /><span /></span>
          <span className="text-xl md:text-2xl">{caption || " "}</span>
        </div>
      </footer>
    </main>
  );
}
