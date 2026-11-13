# Era 26 — AI-polished conclusions for the POC reports, guarded by NeMo Guardrails

2026-11-04. User request: make the proof-of-concept conclusions (Era 24) read more smoothly, using an LLM, but
with a guardrail so a misquoted number can never reach the page — specifically asked for NeMo Guardrails, and to
find the best/fastest NVIDIA NIM model for it by testing, not guessing.

## Isolation decision (made before any code)

Installing `nemoguardrails` into the main app venv would have pulled in its own pinned `langchain`/`openai`
versions, right next to CrewAI's own pins — exactly the kind of dependency collision that could quietly break
the triage crew. Built a **separate venv**, `tools/poc_llm/venv` (nemoguardrails 0.24.1, langchain-openai,
langchain-nvidia-ai-endpoints), used **only** by `core/poc_llm_polish.py`, invoked as its own subprocess by
`core/poc_jobs.py` — never imported by the Flask app or the main venv. Its own API key var,
`NVIDIA_API_KEY_POC` (reuses the same key as `NVIDIA_API_KEY` for now per the user — swap it for a separate key
whenever ready), so this layer's usage and any rate limiting never touches the triage crew's pool
(`agents/llm_pool.py`) or its quota.

## Model selection — tested, not assumed

`GET /v1/models` on this key listed **85 models**. Benchmarked every plausible small/fast candidate on the real
polish prompt (restate given BPM facts in one paragraph, every number must appear unchanged):

- **73 of 85** returned 404 "Function not found for account" the instant they were called — confirms
  `agents/llm_pool.py`'s own warning that a *listed* model is not necessarily a *usable* one.
- **4 reasoning models** (`nemotron-3.5-lightning-30b-a3b`, `glm-5.3-flash`, `minimax-m3.1-flash`,
  `openai/gpt-oss-20b`) burned their whole token budget on hidden `reasoning_content` and returned empty —
  **the same failure already seen live in the triage crew's narrative step.** `nemotron-3.5-lightning` becomes
  usable with `"chat_template_kwargs": {"thinking": false}` in the request body (6/6 correct once disabled) —
  **this fix is not yet applied to `agents/llm_pool.py`**, which is a separate, real opportunity worth raising
  with the user, out of scope for this change.
- **4 models answered correctly**: `nvidia/nemotron-3.5-bolt-e4b` (~2s), `poolside/laguna-xs-2.1` (~1.6s),
  `nemotron-3.5-lightning` (with the flag above, ~1.5-2s), `google/diffusiongemma-26b-a4b-it` (2-7.5s, and timed
  out once as a lone call — too flaky to trust).

## Guardrail — the real `self_check_facts` action, not the full Colang dialogue

NeMo Guardrails' Colang `generate()` path runs extra NLU calls (`generate_user_intent` etc.) even for a single
turn — pure overhead for this use case. Instead, `core/poc_llm_polish.py` calls
`nemoguardrails.library.self_check.facts.actions.self_check_facts` **directly**, with a custom prompt ("does
every number in RESPONSE appear unchanged in EVIDENCE — yes/no") and a minimal fake task manager, skipping the
whole dialogue runtime. One LLM call per check.

Tested with 8 trials per checker candidate — 4 giving the correct paragraph (must ALLOW) and 4 with one number
deliberately changed (must BLOCK):

- `nvidia/nemotron-3.5-bolt-e4b` — **8/8 correct**, 0.2-4.5 s.
- `poolside/laguna-xs-2.1` — **8/8 correct**, 0.26-1.4 s.
- `nemotron-3.5-lightning` (thinking disabled) — **failed even on the correct text** under the small token
  budget this role needs: the `thinking:false` flag did not reliably reach this code path, the model burned the
  budget anyway, and `is_content_safe`'s empty-string default (`[False]`, i.e. "block") made it block
  everything. **Excluded from the checker role entirely** — see `core/poc_llm_pool.py`'s `CANDIDATES`
  (`role="generator_only"`). A checker that can't tell correct from wrong is worse than no guardrail.

Final pool (`core/poc_llm_pool.py`): tier 0 `nemotron-3.5-bolt-e4b` / `laguna-xs-2.1` (both roles), tier 1
`nemotron-3.5-lightning` (generator only), tier 2 `diffusiongemma` (generator only, last resort). A tiny on-disk
dead-cache (`tools/poc_llm/_dead_models.json`, 600 s TTL, git-ignored) lets a dead candidate be skipped quickly
on the next run without re-probing from scratch.

## Two real bugs found and fixed while wiring this up (not in the original design)

1. **"stale event loop binding" on the second guardrail call in the same process.** `asyncio.run()` was being
   called once per check; nemoguardrails caches part of its LLM client in a way that breaks once the event loop
   that created it is closed. Fixed by running the **whole** per-run polish loop (all 5 items, generate +
   check) inside **one** `asyncio.run()` call (`_polish_run_async`), not one per check.
2. **401 "authorization was missing" on every single guardrail call.** `core/poc_llm_polish.py` passed the key
   to the generator via `Authorization` header (plain `requests`), but the guardrail path builds a
   `langchain_openai.ChatOpenAI` through nemoguardrails' `openai` engine, which reads its key from the
   `OPENAI_API_KEY` **environment variable** at construction time — never set it. Fixed in `_make_rails()`.
3. (Found and fixed before those two broke it twice:) a **fixed 250-token generation budget** truncated stage
   3's paragraph mid-number on the one stage with far more numbers than any other (3 time points x 2 algorithms
   x 3 peaks x 2 values each) — the guardrail correctly rejected the truncated fragment (`"166 (0."` matches
   nothing). Not a guardrail false positive: a real defect, fixed by scaling the token budget with how much
   evidence there is to restate (`max(250, min(700, len(evidence)//2))`), confirmed to fix that exact stage.
4. **`core/poc_jobs.py` didn't load `.env` itself**, only relying on some other already-imported module having
   done it first — worked by accident in the live app (`web/__init__.py` loads it early) but made
   `llm_polish_enabled()` silently return `False`, and the whole polish step silently never ran, under any
   standalone invocation. Fixed by calling `load_dotenv()` directly in `poc_jobs.py` too, same as
   `poc_llm_pool.py` already does — a module should not depend on import order to read its own environment.

## Verified end to end (the real automatic path, not a script)

`poc_jobs.submit_video()` -> report subprocess (`core/poc_report.py`, ~60-70 s, unchanged from Era 24) -> on
success, a **second** subprocess in the isolated venv (`core/poc_llm_polish.py`) -> `llm_polish.json` ->
`core/poc_site.py` rebuilds the run's page with a clearly labelled blue "AI-polished, fact-checked" box under
the plain, deterministic "In short" card and under each stage's plain conclusion — never replacing them,
attributed ("Written by X; checked by Y"). Measured: polish step added **16-42 s** after the report finished (5
items, one generate + one check call each), comfortably under the triage crew's own observed 60-90+ s runs
(which include its own empty-response retries — see Era 8 / live logs). One real run: 5/5 stages polished and
passed the guardrail on the first clean attempt after the fixes above.

## What this is NOT, and other honest limits

- The guardrail checks ONE thing: do the numbers in the polished paragraph match the facts. It does not check
  tone, length, or whether the paragraph is actually a good summary — that's still on the generator model and
  the fixed prompt.
- If every candidate model is down, or the isolated venv/key is missing, nothing breaks: `llm_polish_enabled()`
  returns `False` and the plain, deterministic report (Era 24) is the whole page, same as before this feature
  existed.
- Not yet tested: a live video upload through the running app's actual browser flow (only `poc_jobs`'s own
  Python entry points, which is what the app calls, were exercised directly).
- The `chat_template_kwargs: {"thinking": false}` fix for `nemotron-3.5-lightning` is proven here but **not
  applied to `agents/llm_pool.py`** (the triage crew's own pool, which hits the identical empty-response
  failure on this same model live) — flagged to the user, not done, since it's a change to a different,
  safety-relevant file than the one asked for here.
