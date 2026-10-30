"""Self-healing NVIDIA NIM model selection.

Why this exists (measured, 2026-08/10): NVIDIA's hosted endpoints are volatile
for a given API key. In ONE evening the model that worked changed several times:
`meta/llama-3.1-*`, `qwen3.5`, `dracarys` went 410 (retired); `nemotron-3-super`
worked, then returned 404 "Function not found for account"; `gpt-oss-120b` and
`mistral-nemotron` flipped to 410; `llama-3.2-11b-vision` went from ~1 s to
timing out at 40 s. Any hardcoded model list eventually becomes a dead list.

So instead of trusting a list, this module:
  * probes every candidate in the background with a real (small) request,
  * ranks the ones that answered by quality tier, then latency,
  * marks a model dead for a while the moment it returns 404/410/5xx/timeout,
  * hands both LLM call sites (CrewAI crew, ARIA chat) the live ranking.

The deterministic triage (core/clinical.py) never depends on any of this — a
fully dead pool only degrades the AI narrative to the deterministic text.
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

load_dotenv()

NIM_BASE = "https://integrate.api.nvidia.com/v1"

# (model id, quality tier — lower is better). Tier is a judgement about narrative
# quality; a faster model of a worse tier never outranks a working better tier.
CANDIDATES = [
    ("nvidia/nemotron-3-super-120b-a12b", 0),
    ("openai/gpt-oss-120b", 0),
    ("meta/llama-3.2-90b-vision-instruct", 0),
    ("google/gemma-4-31b-it", 1),
    ("nvidia/nemotron-3.5-lightning-30b-a3b", 1),
    ("openai/gpt-oss-20b", 1),
    ("meta/llama-3.2-11b-vision-instruct", 1),
    ("mistralai/mistral-nemotron", 1),
    ("poolside/laguna-xs-2.1", 2),
]
_TIER = dict(CANDIDATES)

PROBE_TIMEOUT_S = 25      # a model slower than this on a one-line prompt is not usable live
MAX_OK_LATENCY_S = 20     # ...and one slower than this is skipped even if it answers
DEAD_TTL_S = 600          # how long a failed model is skipped before being retried
REPROBE_EVERY_S = 600

_lock = threading.Lock()
_latency = {}             # model -> seconds, only for models that recently answered
_dead_until = {}          # model -> unix time until which it is skipped
_probe_started = False
_first_probe_done = threading.Event()
_DEAD_MARKERS = ('404', '410', 'not found', 'gone', 'end of life', 'timed out',
                 'timeout', '502', '503', '500', 'connection')


def _key():
    return os.getenv("NVIDIA_API_KEY")


def _probe_one(model):
    """Return latency in seconds if the model answers a small request with real
    content, else None."""
    key = _key()
    if not key:
        return None
    t0 = time.time()
    try:
        r = requests.post(
            f"{NIM_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "max_tokens": 400,
                  "messages": [{"role": "user", "content": "In one short sentence, what is a normal adult resting heart rate?"}]},
            timeout=PROBE_TIMEOUT_S)
        if r.status_code != 200:
            return None
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
        return (time.time() - t0) if content else None
    except Exception:
        return None


def probe_all():
    """Probe every candidate concurrently and refresh the ranking."""
    models = [m for m, _ in CANDIDATES]
    with ThreadPoolExecutor(max_workers=len(models)) as ex:
        results = list(ex.map(_probe_one, models))
    now = time.time()
    with _lock:
        for m, lat in zip(models, results):
            if lat is None or lat > MAX_OK_LATENCY_S:
                _latency.pop(m, None)
                _dead_until[m] = now + DEAD_TTL_S
            else:
                _latency[m] = lat
                _dead_until.pop(m, None)
    _first_probe_done.set()
    alive = ranked()
    print(f"[LLM-POOL] probe done — usable now: {alive[:4] if alive else 'NONE'}")
    return alive


def start_background_probe():
    """Probe once now (in a daemon thread) and then keep the ranking fresh."""
    global _probe_started
    with _lock:
        if _probe_started or not _key():
            return
        _probe_started = True

    def _loop():
        while True:
            try:
                probe_all()
            except Exception as exc:
                print(f"[LLM-POOL] probe failed: {exc}")
            time.sleep(REPROBE_EVERY_S)

    threading.Thread(target=_loop, daemon=True, name="llm-pool-probe").start()


def mark_dead(model, ttl=DEAD_TTL_S):
    with _lock:
        _latency.pop(model, None)
        _dead_until[model] = time.time() + ttl


def ranked():
    """Usable models, best first: proven-working ones by (tier, latency), then models
    not yet probed (tier order) so a cold start still has something to try."""
    now = time.time()
    with _lock:
        working = sorted((m for m in _latency if _dead_until.get(m, 0) <= now),
                         key=lambda m: (_TIER.get(m, 9), _latency[m]))
        untested = [m for m, _ in CANDIDATES
                    if m not in _latency and _dead_until.get(m, 0) <= now]
    return working + untested


def wait_ready(timeout=30):
    """Block (up to `timeout` s) until the first probe has finished, so a request that
    arrives right after startup uses the real ranking instead of guessing."""
    start_background_probe()
    return _first_probe_done.wait(timeout)


def is_dead_error(exc):
    """True if an exception looks like the model itself is unavailable (retired, not
    entitled, overloaded, timing out) rather than a problem with one particular prompt."""
    msg = str(exc).lower()
    return any(m in msg for m in _DEAD_MARKERS)
