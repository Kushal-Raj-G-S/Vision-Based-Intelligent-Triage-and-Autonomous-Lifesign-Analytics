"use client";

/** Radial instrument cluster for the staff dashboard: heart rate, respiration, stress.
 *  Pure presentation of the numbers the backend already computes — nothing is derived here
 *  except where a value falls inside its arc. Normal bands are standard adult reference
 *  ranges drawn for orientation only; the clinical engine (core/clinical.py) decides acuity. */

interface Props {
  bpm: number;
  bpmConf: number;
  bpmClass: string;
  rr: number;
  rrConf: number;
  rrClass: string;
  stress: number;
  hrv: number;
  sqi: number;
  calibrating: boolean;
  calibrationProgress: number;
  ppg: number[];
  compact?: boolean;
}

const R = 88;
const START = 135;   // degrees, gauge sweeps 270 clockwise from bottom-left
const SWEEP = 270;

function arcPath(): string {
  const p = (deg: number) => {
    const a = (deg * Math.PI) / 180;
    return [110 + R * Math.cos(a), 110 + R * Math.sin(a)];
  };
  const [x0, y0] = p(START);
  const [x1, y1] = p(START + SWEEP);
  return `M ${x0} ${y0} A ${R} ${R} 0 1 1 ${x1} ${y1}`;
}
const ARC = arcPath();

function toneFor(cls: string, fallback: string): string {
  const c = (cls || "").toUpperCase();
  if (c === "NORMAL" || c === "OPTIMAL") return "#34d399";
  if (c === "TACHYCARDIA" || c === "TACHYPNEA" || c === "CRITICAL") return "#fb7185";
  if (c === "BRADYCARDIA" || c === "BRADYPNEA") return "#38bdf8";
  if (c === "ELEVATED") return "#fbbf24";
  return fallback;
}

function Gauge(props: {
  label: string; unit: string; value: number; display: string;
  min: number; max: number; bandLo: number; bandHi: number;
  tone: string; chip: string; sub: string; anim?: string; animSec?: number; id: string; compact?: boolean;
}) {
  const { label, unit, value, display, min, max, bandLo, bandHi, tone, chip, sub, anim, animSec, id, compact } = props;
  const has = value > 0;
  const pct = has ? Math.max(0, Math.min(1, (value - min) / (max - min))) * 100 : 0;
  const b0 = ((bandLo - min) / (max - min)) * 100;
  const b1 = ((bandHi - min) / (max - min)) * 100;
  return (
    <div className={`relative flex flex-col items-center rounded-2xl border border-white/[0.06] bg-white/[0.02] px-3 min-w-0 ${compact ? "pt-2.5 pb-2" : "pt-4 pb-3"}`}>
      <div className="text-xs font-medium text-zinc-400 self-start px-1">{label}</div>
      <div className={`relative w-full aspect-square ${compact ? "max-w-[150px]" : "max-w-[230px]"}`}>
        <svg viewBox="0 0 220 220" className={`absolute inset-0 w-full h-full ${anim ?? ""}`} style={animSec ? { animationDuration: `${animSec}s` } : undefined}>
          <defs>
            <linearGradient id={`g-${id}`} x1="0" y1="1" x2="1" y2="0">
              <stop offset="0%" stopColor={tone} stopOpacity="0.55" />
              <stop offset="100%" stopColor={tone} />
            </linearGradient>
            <filter id={`f-${id}`} x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="4" result="b" />
              <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          <path d={ARC} fill="none" stroke="rgba(148,163,184,0.16)" strokeWidth="10" strokeLinecap="round" pathLength={100} />
          <path d={ARC} fill="none" stroke="rgba(52,211,153,0.35)" strokeWidth="10" pathLength={100}
                strokeDasharray={`${Math.max(0, b1 - b0)} 100`} strokeDashoffset={-b0} />
          {has && (
            <path d={ARC} fill="none" stroke={`url(#g-${id})`} strokeWidth="10" strokeLinecap="round" pathLength={100}
                  strokeDasharray={`${Math.max(pct, 0.8)} 100`} filter={`url(#f-${id})`}
                  style={{ transition: "stroke-dasharray 0.6s ease" }} />
          )}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div className={`font-mono font-semibold leading-none tracking-tight text-white ${compact ? "text-3xl" : display.length > 4 ? "text-4xl" : "text-5xl"}`}>{display}</div>
          <div className="mt-1.5 text-[11px] font-medium text-zinc-500">{unit}</div>
        </div>
      </div>
      <div className={`flex flex-wrap items-center justify-center gap-x-2 gap-y-1 ${compact ? "-mt-2" : "-mt-3"}`}>
        <span className="rounded-full px-2.5 py-0.5 text-[11px] font-semibold" style={{ color: tone, background: `${tone}22` }}>{chip}</span>
        <span className="text-[11px] font-mono text-zinc-500">{sub}</span>
      </div>
    </div>
  );
}

function Trace({ ppg, live }: { ppg: number[]; live: boolean }) {
  const pts = ppg.slice(-180);
  if (pts.length < 8) {
    return <div className="h-[84px] rounded-xl border border-white/[0.06] bg-white/[0.02] flex items-center justify-center text-xs text-zinc-500">Waiting for pulse signal…</div>;
  }
  const lo = Math.min(...pts), hi = Math.max(...pts), span = hi - lo || 1;
  const d = pts.map((v, i) => `${i === 0 ? "M" : "L"} ${(i / (pts.length - 1)) * 600} ${70 - ((v - lo) / span) * 60}`).join(" ");
  return (
    <div className="relative rounded-xl border border-white/[0.06] bg-white/[0.02] overflow-hidden">
      <div className="absolute left-3 top-2 text-[11px] font-medium text-zinc-500">Live pulse trace</div>
      {live && <span className="absolute right-3 top-2 flex items-center gap-1.5 text-[11px] font-medium text-rose-300"><span className="h-1.5 w-1.5 rounded-full bg-rose-400 animate-pulse" />live</span>}
      <svg viewBox="0 0 600 80" preserveAspectRatio="none" className="w-full h-[84px]">
        <defs>
          <linearGradient id="trace-g" x1="0" x2="1">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.1" />
            <stop offset="70%" stopColor="#38bdf8" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#f43f5e" />
          </linearGradient>
        </defs>
        <path d={d} fill="none" stroke="url(#trace-g)" strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke"
              style={{ filter: "drop-shadow(0 0 5px rgba(56,189,248,0.55))" }} />
      </svg>
    </div>
  );
}

export default function VitalsHero(p: Props) {
  const hrTone = toneFor(p.bpmClass, "#38bdf8");
  const rrTone = toneFor(p.rrClass, "#38bdf8");
  const stTone = p.stress > 150 ? "#fb7185" : p.stress > 80 ? "#fbbf24" : p.stress > 0 ? "#34d399" : "#64748b";
  const waiting = p.calibrating ? `calibrating ${Math.round(p.calibrationProgress)}%` : "waiting";
  return (
    <div className={`flex flex-col ${p.compact ? "gap-3" : "gap-4"}`}>
      <div className={`grid grid-cols-3 ${p.compact ? "gap-3" : "gap-4"}`}>
        <Gauge compact={p.compact} id="hr" label="Heart rate" unit="beats / min" value={p.bpm} display={p.bpm > 0 ? p.bpm.toFixed(0) : "--"}
               min={40} max={160} bandLo={60} bandHi={100} tone={hrTone} chip={p.bpm > 0 ? p.bpmClass : "NO SIGNAL"}
               sub={p.bpm > 0 ? `conf ${p.bpmConf.toFixed(0)}%` : waiting}
               anim="vital-heartbeat-soft" animSec={p.bpm > 0 ? 60 / p.bpm : undefined} />
        <Gauge compact={p.compact} id="rr" label="Respiration" unit="breaths / min" value={p.rr} display={p.rr > 0 ? p.rr.toFixed(1) : "--"}
               min={4} max={40} bandLo={12} bandHi={20} tone={rrTone} chip={p.rr > 0 ? p.rrClass : "NO SIGNAL"}
               sub={p.rr > 0 ? `conf ${p.rrConf.toFixed(0)}%` : "needs ~30 s of signal"}
               anim="vital-breathe" animSec={p.rr > 0 ? 60 / p.rr : undefined} />
        <Gauge compact={p.compact} id="st" label="Stress index" unit={`HRV ${p.hrv > 0 ? p.hrv.toFixed(0) : "--"} ms`} value={p.stress} display={p.stress > 0 ? p.stress.toFixed(0) : "--"}
               min={0} max={200} bandLo={0} bandHi={80} tone={stTone}
               chip={p.stress > 150 ? "CRITICAL" : p.stress > 80 ? "ELEVATED" : p.stress > 0 ? "OPTIMAL" : "NO SIGNAL"}
               sub={p.sqi > 0 ? `SQI ${p.sqi.toFixed(0)}%` : waiting} />
      </div>
      {!p.compact && <Trace ppg={p.ppg} live={p.bpm > 0} />}
    </div>
  );
}
