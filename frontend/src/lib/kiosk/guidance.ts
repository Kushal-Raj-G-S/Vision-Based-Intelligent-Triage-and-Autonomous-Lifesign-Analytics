// Pure decision logic for the patient kiosk: "is this person positioned well enough to
// measure, and if not, what is the ONE thing to tell them?" Kept free of React and the DOM
// so it can be unit-tested (guidance.test.ts) — a wrong instruction here is worse than none.

export interface FaceBox { x: number; y: number; w: number; h: number }   // fractions of the frame, RAW camera orientation

export interface StatusLike {
  face_detected?: boolean;
  face_box?: FaceBox | null;
  estimated_lux?: number;
  motion_delta?: number;
}

export type Issue =
  | "no_face" | "closer" | "farther" | "move_left" | "move_right"
  | "keep_in_circle" | "more_light" | "too_bright" | "hold_still" | "ok";

// Thresholds, with where they come from:
//  - lux < 50 / > 210 and motion are the same limits the backend already warns on
//    (web/routes.py generate_frames).
//  - motion > 8 is stricter than the backend's 15: the rPPG signal degrades well before
//    the backend's "excessive motion" warning, and speech/movement is this project's
//    worst accuracy condition.
//  - face width 17-52 % of the frame: a face narrower than that has too few pixels in the
//    forehead/cheek ROI boxes for a clean pulse signal; wider than that clips the edges.
export const THRESH = {
  minW: 0.17, maxW: 0.52,
  centerTol: 0.13,
  minCy: 0.28, maxCy: 0.68,
  luxLow: 50, luxHigh: 210,
  motionHigh: 8,
};

/** x-centre of the face as the PATIENT sees it. The kiosk shows a mirrored preview, so the
 *  raw camera x is flipped. */
export function mirroredCenterX(b: FaceBox): number {
  return 1 - (b.x + b.w / 2);
}

/** The single most important thing to fix right now (priority order matters: there is no
 *  point talking about light while the person is out of frame). */
export function assessPosition(s: StatusLike): Issue {
  const b = s.face_box;
  if (!s.face_detected || !b) return "no_face";
  if (b.w < THRESH.minW) return "closer";
  if (b.w > THRESH.maxW) return "farther";
  const cx = mirroredCenterX(b);
  // In a mirror, a face shown on the RIGHT of the screen means the person is too far to their
  // right, so they should move LEFT (and vice versa).
  if (cx > 0.5 + THRESH.centerTol) return "move_left";
  if (cx < 0.5 - THRESH.centerTol) return "move_right";
  const cy = b.y + b.h / 2;
  if (cy < THRESH.minCy || cy > THRESH.maxCy) return "keep_in_circle";
  const lux = s.estimated_lux ?? 0;
  if (lux > 0 && lux < THRESH.luxLow) return "more_light";
  if (lux > THRESH.luxHigh) return "too_bright";
  if ((s.motion_delta ?? 0) > THRESH.motionHigh) return "hold_still";
  return "ok";
}

/** Issue -> the pre-rendered prompt id (public/voice/<voice>/<id>.mp3). */
export const ISSUE_PROMPT: Record<Exclude<Issue, "ok">, string> = {
  no_face: "no_face", closer: "closer", farther: "farther",
  move_left: "move_left", move_right: "move_right", keep_in_circle: "keep_in_circle",
  more_light: "more_light", too_bright: "too_bright", hold_still: "hold_still",
};

/** Decides whether a spoken instruction may play now. Guidance that nags is as bad as guidance
 *  that is silent: the same line is not repeated within `sameGapMs`, and no two lines are closer
 *  than `anyGapMs`. Injectable clock so it can be tested. */
export class PromptThrottle {
  private last = new Map<string, number>();
  private lastAny = -Infinity;
  private sameGapMs: number;
  private anyGapMs: number;
  constructor(sameGapMs = 8000, anyGapMs = 3500) {
    this.sameGapMs = sameGapMs;
    this.anyGapMs = anyGapMs;
  }

  allow(id: string, now: number): boolean {
    if (now - this.lastAny < this.anyGapMs) return false;
    if (now - (this.last.get(id) ?? -Infinity) < this.sameGapMs) return false;
    return true;
  }
  mark(id: string, now: number): void {
    this.last.set(id, now);
    this.lastAny = now;
  }
  reset(): void { this.last.clear(); this.lastAny = -Infinity; }
}

/** Requires the position to stay good for `holdMs` before we call it good, so a single lucky
 *  frame doesn't start the scan. A bad frame resets the clock. */
export class StableTimer {
  private since: number | null = null;
  private holdMs: number;
  constructor(holdMs = 2000) {
    this.holdMs = holdMs;
  }
  update(ok: boolean, now: number): boolean {
    if (!ok) { this.since = null; return false; }
    if (this.since === null) this.since = now;
    return now - this.since >= this.holdMs;
  }
  reset(): void { this.since = null; }
}
