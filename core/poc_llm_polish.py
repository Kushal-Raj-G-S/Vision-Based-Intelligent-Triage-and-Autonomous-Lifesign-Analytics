"""
Polish the wording of one proof-of-concept run's conclusions with an LLM, guarded by NeMo Guardrails so a
misquoted number can never reach the page.

Runs ONLY inside the isolated venv (tools/poc_llm/venv — has nemoguardrails/langchain; the main app venv does
not and must never need to). Invoked as a subprocess by core/poc_jobs.py, after the plain (deterministic)
report for a run already exists — so if this step is slow, disabled, or every model is down, the plain report
the viewer sees is already complete and correct; this only adds a nicer-reading paragraph on top, never
replaces or re-decides anything.

    tools/poc_llm/venv/Scripts/python.exe -m core.poc_llm_polish --run "poc/patients/<name>/<run>"

What it does, per stage (and the top "In short" card):
  1. Takes the ALREADY-GENERATED plain facts (core/poc_conclusions.py's "did" + "found" sentences — themselves
     built only from measured numbers, never from a model).
  2. Asks a small fast model to rewrite them as one smooth paragraph, with an explicit instruction to copy
     every number unchanged.
  3. Asks a SECOND, independent model (NeMo Guardrails' self_check_facts action) whether every number in that
     paragraph still appears in the original facts. Only if it says yes is the paragraph kept; otherwise (or on
     any error, timeout, or missing model) the plain version stays and nothing is written.
Both roles try a short ranked list of models and fall back automatically — see core/poc_llm_pool.py for why
each one is on that list and what it is and is not trusted for.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GEN_TIMEOUT_S = 25
CHECK_MAX_TOKENS = 20
PER_RUN_BUDGET_S = 90          # give up on the whole run past this; the plain report already stands on its own

CHECK_PROMPT = """You are a strict fact-checker. You are given EVIDENCE (the only true numbers) and a RESPONSE someone wrote from it.
Check ONLY the numbers: does every number that appears in the RESPONSE also appear, completely unchanged, somewhere in the EVIDENCE?
Ignore wording, style and sentence order - check numbers only. If even one number in the RESPONSE does not appear in the EVIDENCE, answer "no".

EVIDENCE: {evidence}

RESPONSE: {response}

Does every number in the RESPONSE appear unchanged in the EVIDENCE? Answer with exactly one word, "yes" or "no"."""

GEN_PROMPT = """Rewrite the following facts as ONE smooth, friendly paragraph (2-4 sentences) for a non-technical reader.
Copy every number EXACTLY as given - same digits, same decimal places. Do not invent, round, or drop any number. Do not add any fact that is not already here.
Do not add commentary before or after the paragraph.

Facts:
{evidence}

Paragraph:"""


def _generate(evidence: str) -> tuple:
    """Try each generator candidate in order. Returns (text, model) or (None, None) if every one failed."""
    import requests
    from core import poc_llm_pool as pool
    key = pool.api_key()
    if not key:
        return None, None
    # stage 3's evidence (every spectral peak at 3 time points x 2 algorithms) is far denser in numbers than any
    # other stage; a fixed small budget was measured to truncate it mid-number (found 2026-11-04 — see
    # data/evolution/26-poc-llm-polish.md), which the guardrail correctly rejected as not fully matching the
    # evidence. Scale the budget with how much there is to restate instead of a fixed guess.
    max_tokens = max(250, min(700, len(evidence) // 2))
    for model, extra in pool.generators():
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0.3, "messages": [{"role": "user", "content": GEN_PROMPT.format(evidence=evidence)}]}
        if extra:
            body.update(extra)
        try:
            r = requests.post(f"{pool.NIM_BASE}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=GEN_TIMEOUT_S)
            if r.status_code != 200:
                pool._mark_dead(model)
                continue
            text = (r.json()["choices"][0]["message"].get("content") or "").strip()
            if text:
                return text, model
            pool._mark_dead(model)
        except Exception:
            pool._mark_dead(model)
    return None, None


def _make_rails(model: str, extra):
    from nemoguardrails import RailsConfig, LLMRails
    from core import poc_llm_pool as pool
    # nemoguardrails' "openai" engine builds a langchain ChatOpenAI, which reads its key from OPENAI_API_KEY at
    # construction time (not from NVIDIA_API_KEY_POC) — without this, every call fails 401 "authorization was
    # missing" (found 2026-11-04 testing this exact function; see data/evolution/26-poc-llm-polish.md).
    os.environ["OPENAI_API_KEY"] = pool.api_key() or ""
    config = RailsConfig.from_content(yaml_content=f"""
models:
  - type: main
    engine: openai
    model: {model}
    parameters:
      base_url: {pool.NIM_BASE}
""", colang_content="")
    rails = LLMRails(config)
    if extra:
        try:
            rails.llm.model_kwargs = {**(getattr(rails.llm, "model_kwargs", None) or {}), **extra}
        except Exception:
            pass
    return rails


class _FixedPrompt:
    """Feeds a pre-rendered prompt straight to self_check_facts, skipping Colang's prompt-template lookup
    (there is no shipped default self_check_facts prompt) and its dialogue-flow overhead."""

    def __init__(self, rendered: str):
        self._rendered = rendered

    def render_task_prompt(self, task, context):
        return self._rendered

    def get_stop_tokens(self, task):
        return None

    def get_max_tokens(self, task):
        return CHECK_MAX_TOKENS

    def has_output_parser(self, task):
        return False

    def parse_task_output(self, task, output, forced_output_parser=None):
        from nemoguardrails.llm.output_parsers import is_content_safe
        return is_content_safe(output)


async def _check_one(model: str, extra, evidence: str, response: str):
    from nemoguardrails.library.self_check.facts.actions import self_check_facts
    rails = _make_rails(model, extra)
    rendered = CHECK_PROMPT.format(evidence=evidence, response=response)
    outcome = await self_check_facts(llm_task_manager=_FixedPrompt(rendered), context={}, llm=rails.llm, config=rails.config,
                                     relevant_chunks=[evidence], bot_message=response)
    return outcome.decision.value == "allow"


async def _check(evidence: str, response: str) -> tuple:
    """Try each checker candidate in order (only models proven reliable both ways — see poc_llm_pool.py).
    Returns (allowed: bool, model_used: str | None). No reachable checker => NOT allowed (fail closed).

    Must run inside the SAME asyncio event loop as every other check call this process makes — nemoguardrails
    caches its LLM client keyed in a way that breaks ("stale event loop binding") if a later call runs in a
    freshly created loop, which is what happens if each check gets its own asyncio.run(). polish_run() below
    opens exactly one loop for the whole run and awaits every check inside it."""
    from core import poc_llm_pool as pool
    for model, extra in pool.checkers():
        try:
            ok = await _check_one(model, extra, evidence, response)
            return ok, model
        except Exception:
            pool._mark_dead(model)
    return False, None


async def _polish_run_async(run_dir: Path, t0: float, log) -> dict:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "done":
        return {"status": "skipped", "reason": "run_not_done"}
    from core import poc_conclusions
    con = poc_conclusions.build(result)

    items = [("headline", con["headline"] + " " + con["verdict"])]
    for i, st in enumerate(con["stages"]):
        evidence = st["did"] + " " + " ".join(st["found"])
        items.append((f"stage{i + 1}", evidence))

    out = {"built": time.strftime("%Y-%m-%d %H:%M:%S")}
    for key, evidence in items:
        if time.time() - t0 > PER_RUN_BUDGET_S:
            log(f"{key}: skipped — over the {PER_RUN_BUDGET_S}s per-run time budget, plain text stays")
            continue
        text, gen_model = _generate(evidence)             # plain requests call — fine to block the loop briefly
        if not text:
            log(f"{key}: no generator model answered — plain text stays")
            continue
        allowed, check_model = await _check(evidence, text)
        if allowed:
            out[key] = {"text": text, "generated_by": gen_model, "checked_by": check_model}
            log(f"{key}: OK (generated by {gen_model}, checked by {check_model}, {time.time() - t0:.1f}s so far)")
        else:
            log(f"{key}: guardrail did not confirm the numbers (checker: {check_model or 'none reachable'}) — plain text stays, discarding: {text[:80]!r}")
    out["_n_items"] = len(items)
    return out


def polish_run(run_dir: Path, log=print) -> dict:
    from core import poc_llm_pool as pool
    t0 = time.time()
    if not pool.api_key():
        log("no API key for the POC LLM pool — skipping (plain report stands)")
        return {"status": "skipped", "reason": "no_api_key"}
    out = asyncio.run(_polish_run_async(run_dir, t0, log))
    if out.get("status") == "skipped":
        return out
    n_items = out.pop("_n_items", 0)
    (run_dir / "llm_polish.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    try:
        from core import poc_site
        poc_site.build_run_page(run_dir)
        poc_site.build_index()
    except Exception as exc:
        log(f"page rebuild after polish failed (plain page is unaffected): {exc}")
    n_polished = len(out) - 1                            # minus the "built" timestamp key
    log(f"done in {time.time() - t0:.1f}s, {n_polished}/{n_items} stages polished")
    return {"status": "done", "polished": n_polished, "of": n_items, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    res = polish_run(Path(a.run))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
