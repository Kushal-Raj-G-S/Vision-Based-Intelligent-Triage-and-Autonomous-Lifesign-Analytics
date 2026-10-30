import { BACKEND_URL } from "./config";
import type { FaceBox } from "./guidance";

export interface KioskStatus {
  bpm: number;
  rr: number;
  status: string;
  is_live: boolean;
  calibration_done: boolean;
  calibration_progress: number;
  face_detected: boolean;
  face_box?: FaceBox | null;
  estimated_lux: number;
  motion_delta: number;
  warnings: string[];
}

export interface QueueInfo {
  token: string;
  position: number;
  ahead: number;
  est_wait_min: number;
  status: string;
  esi_level: number;
}

export interface TriageResult {
  success: boolean;
  token: string | null;
  queue: QueueInfo | null;
  db_saved: boolean;
  message?: string;
  clinical_assessment?: { esi_level: number; is_shock: boolean; needs_manual_review?: boolean };
}

export interface WaitingEntry {
  token: string;
  esi_level: number;
  status: string;
  position: number;
  est_wait_min: number;
}
export interface WaitingRoom {
  success: boolean;
  now_serving: string[];
  waiting: WaitingEntry[];
  count: number;
  avg_service_min: number;
  note: string;
}

async function j<T>(path: string, init?: RequestInit, timeoutMs = 15000): Promise<T | null> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${BACKEND_URL}${path}`, { ...init, signal: ctrl.signal });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

const post = (body?: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
});

export const getStatus = () => j<KioskStatus>("/status", undefined, 4000);
export const startWebcam = () => j<{ success: boolean }>("/start_webcam", post({}), 20000);
export const scanBegin = () => j<{ success: boolean }>("/api/scan/begin", post());
export const releaseCamera = () => j<{ success: boolean }>("/release_camera", post());
export const generateReport = () => j<{ success: boolean }>("/api/generate_report", post(), 20000);
export const getWaitingRoom = () => j<WaitingRoom>("/api/waiting_room", undefined, 8000);

export interface Intake {
  patient_name: string;
  age?: number | null;
  sex?: string;
  complaint?: string;
}

/** fast:true => the backend saves the record + token right after the quick deterministic
 *  assessment and writes the AI narrative later, so the patient is not kept waiting. */
export const triageRun = (intake: Intake) =>
  j<TriageResult>("/api/triage_run", post({ source: "live", fast: true, ...intake }), 60000);

export interface SttResult {
  success: boolean;
  text: string;
  field: unknown;
  confidence: number;
  note?: string;
}

/** null = the speech server is not available (the kiosk then falls back to typing). */
export async function transcribe(wav: Blob, hint: string): Promise<SttResult | null> {
  const fd = new FormData();
  fd.append("audio", wav, "clip.wav");
  fd.append("hint", hint);
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 30000);
  try {
    const res = await fetch(`${BACKEND_URL}/api/stt`, { method: "POST", body: fd, signal: ctrl.signal });
    if (!res.ok) return null;
    return (await res.json()) as SttResult;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}
