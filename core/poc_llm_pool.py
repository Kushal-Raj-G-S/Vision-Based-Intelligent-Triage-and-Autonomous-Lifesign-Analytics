"""
Self-healing NVIDIA NIM model pool for the proof-of-concept conclusion polish layer (core/poc_llm_polish.py).

Separate from agents/llm_pool.py (the triage crew's pool) on purpose: its own API key (NVIDIA_API_KEY_POC, falls
back to NVIDIA_API_KEY if unset), its own candidate list, and its own on-disk "recently dead" cache — so this
layer's usage, failures and rate limits never compete with or affect the triage crew's pool. Runs inside the
isolated venv (tools/poc_llm/venv), never inside the main app process.

How the candidates were chosen (2026-11-04): every model visible to the account was benchmarked on the real
polish prompt and on the real self_check_facts guardrail call (8+ trials each: correct text must be ALLOWed,
text with one number deliberately changed must be BLOCKed). Full numbers: data/evolution/26-poc-llm-polish.md.
  * nvidia/nemotron-3.5-bolt-e4b and poolside/laguna-xs-2.1 — fast (~0.3-2s), 8/8 correct as BOTH generator
    and guardrail-checker. These are the only two CHECKER candidates: a checker that blocks correct text, or
    passes wrong numbers, is worse than no guardrail at all.
  * nvidia/nemotron-3.5-lightning-30b-a3b — fast and 6/6 correct as a GENERATOR, but ONLY with
    chat_template_kwargs={"thinking": False} (it is a reasoning model; without that flag it burns its whole
    token budget on hidden reasoning and returns empty, exactly the failure already seen in the live triage
    crew logs with this same model). Even with the flag it failed as a CHECKER under a small token budget
    (the flag did not reliably reach that code path), so it is listed as a GENERATOR fallback only.
  * google/diffusiongemma-26b-a4b-it — correct every time it answered, but much slower (2-7.5s) and timed
    out once as a lone call. Last-resort generator only.
  * Every other model on the account's /v1/models listing (85 total) returned 404 "Function not found for
    account" on this key for this task, or burned its whole budget on hidden reasoning. Do not assume a model
    being listed means it is usable — see agents/llm_pool.py's docstring; the same volatility applies here.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

NIM_BASE = "https://integrate.api.nvidia.com/v1"
DEAD_CACHE = Path(__file__).resolve().parent.parent / "tools" / "poc_llm" / "_dead_models.json"
DEAD_TTL_S = 600
PROBE_TIMEOUT_S = 20

# (model id, role: "both" | "generator_only", extra request body / model_kwargs, tier — lower tried first)
CANDIDATES = [
    ("nvidia/nemotron-3.5-bolt-e4b", "both", None, 0),
    ("poolside/laguna-xs-2.1", "both", None, 0),
    ("nvidia/nemotron-3.5-lightning-30b-a3b", "generator_only", {"chat_template_kwargs": {"thinking": False}}, 1),
    ("google/diffusiongemma-26b-a4b-it", "generator_only", None, 2),
]


def api_key() -> Optional[str]:
    return os.environ.get("NVIDIA_API_KEY_POC") or os.environ.get("NVIDIA_API_KEY")


def _load_dead() -> dict:
    try:
        return json.loads(DEAD_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _mark_dead(model: str) -> None:
    try:
        DEAD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        d = _load_dead()
        d[model] = time.time() + DEAD_TTL_S
        DEAD_CACHE.write_text(json.dumps(d), encoding="utf-8")
    except Exception:
        pass


def generators() -> list:
    """Candidates usable as the text generator, ranked tier then order, skipping anything recently dead."""
    dead = _load_dead()
    now = time.time()
    return [(m, extra) for m, role, extra, tier in sorted(CANDIDATES, key=lambda c: c[3])
            if role in ("both", "generator_only") and dead.get(m, 0) <= now]


def checkers() -> list:
    """Candidates allowed to act as the fact-checking guardrail. Only models proven reliable in BOTH directions
    (allow correct text, block corrupted text) under a small token budget qualify — see the module docstring."""
    dead = _load_dead()
    now = time.time()
    return [(m, extra) for m, role, extra, tier in sorted(CANDIDATES, key=lambda c: c[3])
            if role == "both" and dead.get(m, 0) <= now]


def probe_raw(model: str, extra: Optional[dict] = None, prompt: str = "Say OK.", max_tokens: int = 10) -> bool:
    """Quick liveness probe: does this model answer at all right now? Marks it dead on a clear failure."""
    key = api_key()
    if not key:
        return False
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0.0, "messages": [{"role": "user", "content": prompt}]}
    if extra:
        body.update(extra)
    try:
        r = requests.post(f"{NIM_BASE}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=PROBE_TIMEOUT_S)
        if r.status_code != 200:
            _mark_dead(model)
            return False
        content = (r.json()["choices"][0]["message"].get("content") or "").strip()
        if not content:
            _mark_dead(model)
            return False
        return True
    except Exception:
        _mark_dead(model)
        return False
