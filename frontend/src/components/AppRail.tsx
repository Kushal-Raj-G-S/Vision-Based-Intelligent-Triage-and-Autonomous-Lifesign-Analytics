"use client";

/* App navigation: a slim floating rail of icons with labels. The active page gets the orange
   marker; the queue count rides on its icon; system status sits at the bottom as small lights. */

export type RailTab = "monitor" | "queue" | "crew" | "chat";

const ICONS: Record<RailTab, React.ReactNode> = {
  monitor: <path d="M3 12h4l2.5-6 4 12 2.5-6H21" />,
  queue: <><path d="M8 6h12M8 12h12M8 18h12" /><circle cx="4" cy="6" r="1.2" /><circle cx="4" cy="12" r="1.2" /><circle cx="4" cy="18" r="1.2" /></>,
  crew: <><circle cx="12" cy="5" r="2.2" /><circle cx="5" cy="18" r="2.2" /><circle cx="19" cy="18" r="2.2" /><path d="M11 7l-5 9M13 7l5 9M7.5 18h9" /></>,
  chat: <path d="M4 5h16v11H9l-5 4z" />,
};

const ITEMS: { id: RailTab; label: string }[] = [
  { id: "monitor", label: "Monitor" },
  { id: "queue", label: "Dispatch" },
  { id: "crew", label: "Crew" },
  { id: "chat", label: "ARIA" },
];

export default function AppRail({ active, onSelect, queueCount, backendOnline }: {
  active: RailTab; onSelect: (t: RailTab) => void; queueCount: number; backendOnline: boolean;
}) {
  return (
    <nav className="ar-rail" aria-label="Main">
      <div className="ar-mark" title="VITAL">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/vital_logo.png" alt="VITAL" />
      </div>

      <ul className="ar-items">
        {ITEMS.map((it) => (
          <li key={it.id}>
            <button
              className={`ar-item ${active === it.id ? "on" : ""}`}
              onClick={() => onSelect(it.id)}
              aria-current={active === it.id ? "page" : undefined}
            >
              <span className="ar-icon">
                <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                  {ICONS[it.id]}
                </svg>
                {it.id === "queue" && queueCount > 0 && <span className="ar-badge">{queueCount > 99 ? "99+" : queueCount}</span>}
              </span>
              <span className="ar-label">{it.label}</span>
            </button>
          </li>
        ))}
      </ul>

      <div className="ar-status" aria-label="System status">
        <span className={`ar-light ${backendOnline ? "ok" : "bad"}`} title={backendOnline ? "Backend connected" : "Backend offline"} />
        <span className="ar-status-text">{backendOnline ? "Online" : "Offline"}</span>
      </div>
    </nav>
  );
}
