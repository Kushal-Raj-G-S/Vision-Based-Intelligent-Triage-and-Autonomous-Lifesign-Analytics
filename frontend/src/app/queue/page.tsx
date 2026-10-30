"use client";

// Waiting-room board. Put /queue on the screen patients can see: it shows TOKENS ONLY (no names,
// vitals or diagnoses). /queue?staff=1 adds Call-next / Seen controls for the front desk.

import { useCallback, useEffect, useState } from "react";
import "../kiosk/kiosk.css";
import { BACKEND_URL } from "@/lib/kiosk/config";
import { getWaitingRoom, type WaitingRoom } from "@/lib/kiosk/api";

export default function QueueBoard() {
  const [data, setData] = useState<WaitingRoom | null>(null);
  const [offline, setOffline] = useState(false);
  const [staff, setStaff] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [clock, setClock] = useState<Date | null>(null);   // set after mount: SSR and browser locales format time differently
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    // reading the URL once on load is an external sync, not derived state
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setStaff(q.get("staff") === "1");
    setApiKey(q.get("key") ?? "");
  }, []);

  const refresh = useCallback(async () => {
    const d = await getWaitingRoom();
    if (d && d.success) { setData(d); setOffline(false); } else setOffline(true);   // keep the last good board on screen
  }, []);

  useEffect(() => {
    const first = setTimeout(() => { void refresh(); setClock(new Date()); }, 0);   // first load, via a callback
    const iv = setInterval(refresh, 4000);
    const ck = setInterval(() => setClock(new Date()), 1000);
    return () => { clearTimeout(first); clearInterval(iv); clearInterval(ck); };
  }, [refresh]);

  const staffPost = async (path: string, body?: unknown) => {
    setBusy(true);
    try {
      await fetch(`${BACKEND_URL}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(apiKey ? { "X-API-Key": apiKey } : {}) },
        body: body ? JSON.stringify(body) : undefined,
      });
    } finally { setBusy(false); void refresh(); }
  };

  const serving = data?.now_serving ?? [];
  const waiting = (data?.waiting ?? []).filter((w) => w.status === "waiting");

  return (
    <main className="kiosk-root flex flex-col p-8 gap-8">
      <header className="flex items-end justify-between">
        <div>
          <div className="kiosk-eyebrow">VITAL · Waiting room</div>
          <h1 className="kiosk-title text-5xl md:text-6xl">Please wait for your number</h1>
        </div>
        <div className="flex items-center gap-5">
          <div className="text-right">
          <div className="text-4xl font-semibold tabular-nums">{clock ? clock.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true }) : "--:--"}</div>
          <div className="kiosk-dim">{data ? `${data.count} waiting` : "…"}</div>
          </div>
        </div>
      </header>

      {offline && <div className="kiosk-chip warn self-start" role="alert">Updating… (showing the last known list)</div>}

      <section className="grid lg:grid-cols-[1fr_1.6fr] gap-8 flex-1">
        <div className="kiosk-glass rounded-[2rem] p-8 flex flex-col gap-6">
          <div className="kiosk-eyebrow">Now calling</div>
          {serving.length === 0 ? (
            <p className="kiosk-dim text-3xl">No one is being called right now.</p>
          ) : (
            <div className="flex flex-col gap-4">
              {serving.map((t) => (
                <div key={t} className="kiosk-serving kiosk-glass rounded-3xl px-8 py-6 flex items-center justify-between">
                  <span className="kiosk-token whitespace-nowrap" style={{ fontSize: "clamp(2.6rem, 5vw, 5rem)" }}>{t}</span>
                  <span className="text-2xl kiosk-dim">Please come to the desk</span>
                  {staff && <button className="kiosk-btn" disabled={busy} onClick={() => staffPost("/api/queue/mark", { token: t, status: "seen" })}>Seen</button>}
                </div>
              ))}
            </div>
          )}
          {staff && (
            <button className="kiosk-btn primary mt-auto" disabled={busy || waiting.length === 0} onClick={() => staffPost("/api/queue/call_next")}>
              Call next patient
            </button>
          )}
        </div>

        <div className="kiosk-glass rounded-[2rem] p-8 flex flex-col gap-5">
          <div className="flex items-center justify-between">
            <div className="kiosk-eyebrow">Waiting</div>
            <div className="kiosk-dim">Estimated wait</div>
          </div>
          {waiting.length === 0 ? (
            <p className="kiosk-dim text-3xl">Nobody is waiting.</p>
          ) : (
            <ul className="flex flex-col gap-3">
              {waiting.slice(0, 9).map((w) => (
                <li key={w.token} className="flex items-center justify-between rounded-2xl px-6 py-4 bg-white/5 border border-white/10">
                  <div className="flex items-center gap-6">
                    <span className="kiosk-board-token text-5xl">{w.token}</span>
                    {staff && <span className={`kiosk-chip ${w.esi_level <= 2 ? "warn" : ""}`}>ESI {w.esi_level}</span>}
                  </div>
                  <div className="flex items-center gap-5">
                    <span className="text-3xl tabular-nums">{w.est_wait_min <= 5 ? "Very soon" : `~${w.est_wait_min} min`}</span>
                    {staff && <button className="kiosk-btn" style={{ minHeight: "2.8rem" }} disabled={busy} onClick={() => staffPost("/api/queue/mark", { token: w.token, status: "left" })}>Left</button>}
                  </div>
                </li>
              ))}
            </ul>
          )}
          {waiting.length > 9 && <p className="kiosk-dim">…and {waiting.length - 9} more</p>}
          <p className="kiosk-dim mt-auto">{data?.note ?? "Wait times are estimates and change when more urgent cases arrive."}</p>
        </div>
      </section>
    </main>
  );
}
