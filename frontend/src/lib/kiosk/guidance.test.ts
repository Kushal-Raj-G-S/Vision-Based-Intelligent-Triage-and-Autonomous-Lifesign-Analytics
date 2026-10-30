// Run: node --test src/lib/kiosk/guidance.test.ts   (Node 22.18+/24 strips the types itself)
import test from "node:test";
import assert from "node:assert/strict";
import { assessPosition, PromptThrottle, StableTimer, mirroredCenterX, type StatusLike } from "./guidance.ts";

// A well-placed face: 30 % of the frame wide, centred, a little above middle, decent light, still.
const good = (over: Partial<StatusLike> = {}): StatusLike => ({
  face_detected: true, face_box: { x: 0.35, y: 0.25, w: 0.30, h: 0.40 },
  estimated_lux: 120, motion_delta: 1, ...over,
});

test("good position is ok", () => assert.equal(assessPosition(good()), "ok"));

test("no face -> no_face, whichever way it is missing", () => {
  assert.equal(assessPosition({ face_detected: false, face_box: null }), "no_face");
  assert.equal(assessPosition({ face_detected: true, face_box: null }), "no_face");
  assert.equal(assessPosition({}), "no_face");
});

test("too small / too large", () => {
  assert.equal(assessPosition(good({ face_box: { x: 0.42, y: 0.35, w: 0.12, h: 0.16 } })), "closer");
  assert.equal(assessPosition(good({ face_box: { x: 0.10, y: 0.10, w: 0.70, h: 0.80 } })), "farther");
});

test("left/right is judged on the MIRRORED preview", () => {
  // Raw camera x = 0.05..0.35 => on the raw frame the face is at the LEFT, which a mirrored
  // preview shows on the RIGHT => the person should move LEFT.
  const rawLeft = good({ face_box: { x: 0.05, y: 0.25, w: 0.30, h: 0.40 } });
  assert.ok(mirroredCenterX(rawLeft.face_box!) > 0.5);
  assert.equal(assessPosition(rawLeft), "move_left");
  const rawRight = good({ face_box: { x: 0.65, y: 0.25, w: 0.30, h: 0.40 } });
  assert.equal(assessPosition(rawRight), "move_right");
});

test("a face a little off-centre inside the tolerance is still ok", () => {
  assert.equal(assessPosition(good({ face_box: { x: 0.40, y: 0.25, w: 0.30, h: 0.40 } })), "ok");
});

test("vertical position", () => {
  assert.equal(assessPosition(good({ face_box: { x: 0.35, y: 0.0, w: 0.30, h: 0.30 } })), "keep_in_circle");
  assert.equal(assessPosition(good({ face_box: { x: 0.35, y: 0.62, w: 0.30, h: 0.38 } })), "keep_in_circle");
});

test("light and motion", () => {
  assert.equal(assessPosition(good({ estimated_lux: 30 })), "more_light");
  assert.equal(assessPosition(good({ estimated_lux: 240 })), "too_bright");
  assert.equal(assessPosition(good({ motion_delta: 12 })), "hold_still");
  assert.equal(assessPosition(good({ estimated_lux: 0 })), "ok");   // 0 = unmeasured, not "dark"
});

test("priority: framing problems are reported before light and motion", () => {
  const tooSmallAndDark = good({ estimated_lux: 20, motion_delta: 30, face_box: { x: 0.42, y: 0.35, w: 0.10, h: 0.14 } });
  assert.equal(assessPosition(tooSmallAndDark), "closer");
});

test("throttle: same line not repeated quickly, and no two lines back to back", () => {
  const t = new PromptThrottle(8000, 3500);
  assert.equal(t.allow("closer", 0), true);
  t.mark("closer", 0);
  assert.equal(t.allow("closer", 5000), false);      // same line, too soon
  assert.equal(t.allow("hold_still", 1000), false);  // any line, too soon after the last
  assert.equal(t.allow("hold_still", 4000), true);   // different line, gap respected
  assert.equal(t.allow("closer", 9000), true);       // same line again after its own gap
});

test("stable timer needs a continuous good stretch", () => {
  const s = new StableTimer(2000);
  assert.equal(s.update(true, 0), false);
  assert.equal(s.update(true, 1500), false);
  assert.equal(s.update(false, 1800), false);        // a bad frame resets it
  assert.equal(s.update(true, 2000), false);
  assert.equal(s.update(true, 3900), false);
  assert.equal(s.update(true, 4000), true);
});
