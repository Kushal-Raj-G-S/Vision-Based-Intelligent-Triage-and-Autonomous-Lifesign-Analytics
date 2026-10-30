"use client";

/* Optical bedside monitor — the staff "Vitals monitor" view, modelled on the instrument clinicians
   already read every day: dark glass, one colour per parameter, huge condensed numerals with the
   alarm limits printed beside them, a sweep trace over ECG millimetre paper, and an alarm strip
   where *** marks a physiological alarm and INOP a technical one.

   Nothing here decides anything clinical. Alarm messages come from the backend's own
   classifications; the printed limits mirror core/clinical.py (keep them in sync). */

import { useEffect, useRef, useState } from "react";

// Mirrors core/clinical.py — display only.
export const LIMITS = { HR_HI: 100, HR_LO: 50, RR_HI: 20, STRESS_HI: 150, STRESS_ELEV: 80, SQI_MIN: 40 } as const;

// Parameter colours follow bedside-monitor convention: pulse green, respiration amber, others distinct.
export const PARAM = {
  hr: "var(--sig-hr)",
  rr: "var(--sig-rr)",
  stress: "var(--sig-stress)",
  sqi: "var(--sig-sqi)",
} as const;

/** Canvas can't read CSS variables; resolve "var(--x)" against an element. */
function resolveColor(el: Element, c: string): string {
  const m = /^var\((--[\w-]+)\)$/.exec(c.trim());
  return m ? (getComputedStyle(el).getPropertyValue(m[1]).trim() || "#4ef0a0") : c;
}

/* ───────────────────────────── alarm strip ───────────────────────────── */

export interface AlarmInput {
  bpm: number; hrClass: string;
  rr: number; rrClass: string;
  stress: number;
  warnings: string[];
  faceDetected?: boolean;
  hasSource?: boolean;
}

interface Alarm { level: "high" | "med" | "inop"; text: string }

export function deriveAlarms(a: AlarmInput): Alarm[] {
  const out: Alarm[] = [];
  const hc = (a.hrClass || "").toUpperCase();
  const rc = (a.rrClass || "").toUpperCase();
  if (a.bpm > 0 && hc === "TACHYCARDIA") out.push({ level: "high", text: `HR ${a.bpm.toFixed(0)} > ${LIMITS.HR_HI}` });
  if (a.bpm > 0 && hc === "BRADYCARDIA") out.push({ level: "high", text: `HR ${a.bpm.toFixed(0)} LOW` });
  if (a.rr > 0 && rc === "TACHYPNEA") out.push({ level: "med", text: `RR ${a.rr.toFixed(0)} > ${LIMITS.RR_HI}` });
  if (a.rr > 0 && rc === "BRADYPNEA") out.push({ level: "med", text: `RR ${a.rr.toFixed(0)} LOW` });
  if (a.stress > LIMITS.STRESS_HI) out.push({ level: "med", text: `STRESS ${a.stress.toFixed(0)} > ${LIMITS.STRESS_HI}` });
  if (a.hasSource && a.faceDetected === false) out.push({ level: "inop", text: "NO FACE IN VIEW" });
  for (const w of a.warnings || []) out.push({ level: "inop", text: w.toUpperCase().slice(0, 48) });
  return out;
}

export function MonitorBar(props: {
  patient: string; source: string; live: boolean;
  alarms: Alarm[]; scanLeft: number | null; scanTotal: number;
  onDetails: () => void; detailsBadge: number;
}) {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    const tick = () => setNow(new Date());
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  const top = props.alarms[0];
  const more = props.alarms.length - 1;
  return (
    <div className="bm-bar">
      <div className="bm-bar-id">
        <span className={`bm-dot ${props.live ? "on" : ""}`} />
        <div className="min-w-0">
          <div className="bm-bar-name">{props.patient || "Unregistered patient"}</div>
          <div className="bm-bar-sub">{props.source} · adult limits</div>
        </div>
      </div>

      <div className={`bm-alarm ${top ? top.level : "quiet"}`} role="status" aria-live="polite">
        {top ? (
          <>
            <span className="bm-alarm-tag">{top.level === "inop" ? "INOP" : top.level === "high" ? "***" : "**"}</span>
            <span className="truncate">{top.text}</span>
            {more > 0 && <button className="bm-alarm-more" onClick={props.onDetails}>+{more} more</button>}
          </>
        ) : (
          <span className="bm-alarm-none">No active alarms</span>
        )}
      </div>

      <div className="bm-bar-right">
        {props.scanLeft !== null && (
          <div className="bm-scan" title="Scan window remaining">
            <div className="bm-scan-track"><div className="bm-scan-fill" style={{ width: `${(1 - props.scanLeft / props.scanTotal) * 100}%` }} /></div>
            <span>{props.scanLeft}s</span>
          </div>
        )}
        <button className="bm-chip-btn" onClick={props.onDetails}>Details{props.detailsBadge ? ` · ${props.detailsBadge}` : ""}</button>
        <div className="bm-clock" suppressHydrationWarning>
          {now ? now.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit", hour12: true }) : "--:--:--"}
        </div>
      </div>
    </div>
  );
}

/* ───────────────────────────── trend ───────────────────────────── */

/** Session trend of a parameter, drawn between its alarm limits so the band is visible. Values
 *  are collected client-side from successive readings (display only). */
function Trend({ value, lo, hi, color }: { value: number; lo?: number | null; hi?: number | null; color: string }) {
  const [pts, setPts] = useState<number[]>([]);
  const last = useRef<number | null>(null);
  useEffect(() => {
    if (!(value > 0) || value === last.current) return;
    last.current = value;
    setPts(p => (p.length >= 120 ? [...p.slice(1), value] : [...p, value]));
  }, [value]);
  if (pts.length < 2) return <div className="bm-trend" aria-hidden />;
  const vmin = Math.min(...pts, lo ?? Infinity) - 3, vmax = Math.max(...pts, hi ?? -Infinity) + 3;
  const y = (v: number) => 46 - ((v - vmin) / (vmax - vmin || 1)) * 42;
  const d = pts.map((v, i) => `${i ? "L" : "M"} ${(i / (pts.length - 1)) * 100} ${y(v).toFixed(2)}`).join(" ");
  return (
    <svg className="bm-trend" viewBox="0 0 100 50" preserveAspectRatio="none" aria-label={`trend of ${pts.length} readings`}>
      {hi != null && <line x1="0" x2="100" y1={y(hi)} y2={y(hi)} stroke={color} strokeOpacity="0.25" strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />}
      {lo != null && <line x1="0" x2="100" y1={y(lo)} y2={y(lo)} stroke={color} strokeOpacity="0.25" strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />}
      <path d={d} fill="none" stroke={color} strokeWidth="1.6" vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
      <circle cx="100" cy={y(pts[pts.length - 1])} r="2.4" fill={color} />
    </svg>
  );
}

/* ───────────────────────────── parameter block ───────────────────────────── */

export function ParamBlock(props: {
  tag: string; unit: string; color: string;
  value: string; hasValue: boolean;
  hi?: number | null; lo?: number | null;
  status: string; conf?: number | null; foot?: string;
  alarm?: boolean; pulseSec?: number; breatheSec?: number; size?: "xl" | "lg"; trend?: number;
}) {
  const { tag, unit, color, value, hasValue, hi, lo, status, conf, foot, alarm, pulseSec, breatheSec, size = "xl", trend } = props;
  return (
    <div className={`bm-param ${alarm ? "alarm" : ""}`} style={{ ["--pc" as string]: color }}>
      <div className="bm-param-head">
        <span className="bm-param-tag">{tag}</span>
        <span className="bm-param-unit">{unit}</span>
        {pulseSec && <span className="bm-beat" style={{ animationDuration: `${pulseSec}s` }} aria-hidden />}
        {breatheSec && <span className="bm-breath" style={{ animationDuration: `${breatheSec}s` }} aria-hidden />}
      </div>
      <div className="bm-param-body">
        <div className="bm-limits" aria-label="alarm limits">
          <span>{hi ?? "—"}</span>
          <span>{lo ?? "—"}</span>
        </div>
        {trend !== undefined && <Trend value={trend} lo={lo} hi={hi} color={color} />}
        <div className={`bm-num ${size} ${hasValue ? "" : "empty"}`}>{value}</div>
      </div>
      <div className="bm-param-foot">
        <span className="bm-status">{status}</span>
        {conf != null && (
          <span className="bm-conf" title="Signal confidence">
            <span className="bm-conf-bar"><span style={{ width: `${Math.max(0, Math.min(100, conf))}%` }} /></span>
            {conf.toFixed(0)}%
          </span>
        )}
        {foot && <span className="bm-foot-extra">{foot}</span>}
      </div>
    </div>
  );
}

/* ───────────────────────────── sweep trace ───────────────────────────── */

/** Pleth-style sweep: the newest samples are written at a moving cursor that wraps across the
 *  screen, with an erase gap ahead of it — the way a bedside monitor draws, not a scrolling chart.
 *  Drawn on ECG paper (1 mm / 5 mm grid). */
export function SweepTrace({ samples, color, label, rate }: { samples: number[]; color: string; label: string; rate: string }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const latest = useRef<number[]>([]);
  latest.current = samples;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    const COLS = 900;                       // ring buffer = screen width in columns
    const ring = new Float32Array(COLS).fill(NaN);
    let cursor = 0;
    let lastLen = 0;
    let lastFirst = NaN;
    let lastT = performance.now();
    let pending: number[] = [];
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const frame = (t: number) => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // pick up new samples (the backend sends a rolling window; take what's new since last poll)
      const s = latest.current;
      if (s.length && (s.length !== lastLen || s[0] !== lastFirst)) {
        const fresh = Math.max(1, Math.min(s.length, Math.round(s.length / 10)));   // ~1 s of a 10 s window
        pending = pending.concat(s.slice(-fresh));
        lastLen = s.length; lastFirst = s[0];
      }
      if (reduce) {
        // reduced motion: no sweep — show the latest window as a still trace
        const n = s.length;
        for (let i = 0; i < COLS; i++) {
          if (n < 2) { ring[i] = NaN; continue; }
          const x = (i / (COLS - 1)) * (n - 1), k = Math.floor(x), f = x - k;
          ring[i] = k + 1 < n ? s[k] * (1 - f) + s[k + 1] * f : s[n - 1];   // linear interpolation
        }
        cursor = 0; pending = [];
      }
      // advance the cursor at a steady speed (~ one screen per 6 s), spending pending samples
      const dt = Math.min(0.1, (t - lastT) / 1000); lastT = t;
      const step = Math.max(1, Math.round((COLS / 6) * dt));
      for (let i = 0; i < (reduce ? 0 : step); i++) {
        ring[cursor] = pending.length ? (pending.shift() as number) : NaN;
        cursor = (cursor + 1) % COLS;
      }
      if (pending.length > COLS) pending = pending.slice(-COLS);

      // paper
      ctx.clearRect(0, 0, w, h);
      const mm = Math.max(4, h / 30);
      ctx.lineWidth = 1;
      for (let x = 0, i = 0; x < w; x += mm, i++) {
        ctx.strokeStyle = i % 5 === 0 ? "rgba(255,110,140,0.13)" : "rgba(255,110,140,0.05)";
        ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, h); ctx.stroke();
      }
      for (let y = 0, i = 0; y < h; y += mm, i++) {
        ctx.strokeStyle = i % 5 === 0 ? "rgba(255,110,140,0.13)" : "rgba(255,110,140,0.05)";
        ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(w, y + 0.5); ctx.stroke();
      }

      // scale from the finite values on screen
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < COLS; i++) { const v = ring[i]; if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
      if (Number.isFinite(lo) && hi > lo) {
        const pad = h * 0.14, span = hi - lo;
        const gap = reduce ? 0 : Math.round(COLS * 0.035);
        ctx.lineJoin = "round"; ctx.lineCap = "round";
        const ink = resolveColor(canvas, color);
        ctx.strokeStyle = ink; ctx.lineWidth = 2;
        ctx.beginPath();
        let pen = false;
        for (let i = 0; i < COLS; i++) {
          const ahead = (i - cursor + COLS) % COLS;
          const v = ring[i];
          if (ahead < gap || !Number.isFinite(v)) { pen = false; continue; }
          const x = (i / (COLS - 1)) * w;
          const y = h - pad - ((v - lo) / span) * (h - 2 * pad);
          if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
        }
        ctx.stroke();
        // write head
        if (!reduce) {
        const cx = (cursor / (COLS - 1)) * w;
        ctx.fillStyle = ink;
        ctx.globalAlpha = 0.9; ctx.fillRect(cx - 1, pad * 0.5, 2, h - pad); ctx.globalAlpha = 1;
        }
      }
    };
    // full motion: redraw every frame; reduced motion: a still redraw twice a second
    let timer = 0;
    const loop = (t: number) => {
      frame(t);
      if (reduce) timer = window.setTimeout(() => { raf = requestAnimationFrame(loop); }, 500);
      else raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(raf); window.clearTimeout(timer); };
  }, [color]);

  return (
    <div className="bm-trace" style={{ ["--pc" as string]: color }}>
      <div className="bm-trace-label"><span>{label}</span><span className="bm-trace-rate">{rate}</span></div>
      <canvas ref={ref} className="bm-trace-canvas" />
      {samples.length < 8 && <div className="bm-trace-empty">Waiting for pulse signal</div>}
    </div>
  );
}
