"""
VITAL — Lightweight Guideline & Literature Retrieval
=====================================================
Honest scope note (read this before citing this module as "RAG")
──────────────────────────────────────────────────────────────────
This is deliberately NOT a vector-embedding RAG pipeline. It's local,
dependency-free term-frequency cosine retrieval over two small corpora:

  1. A handful of hand-written ESI/Compensated-Shock guideline snippets
     (original phrasing, not copied from any single source).
  2. The 30 IEEE paper summaries already curated in research/*.csv
     (Key Findings / Limitations / Future Scope columns).

Why not FAISS + OpenAIEmbeddings (as implementation_plan.md once implied)?
Because that was never built, and claiming it without the code behind it
is exactly the kind of claims/code mismatch this project's own README was
called out for. Term-frequency cosine similarity is a real, citable
information-retrieval technique (classic tf-idf/vector-space-model IR) —
just a much simpler one than dense embeddings. If a future revision adds
real embeddings, this module is the natural place to swap the similarity
function without touching call sites (`retrieve()` / `retrieve_for_assessment()`).

Used by
────────
agents/crew.py builds a small "grounding context" string from
`retrieve_for_assessment()` and injects it into the Diagnostic Agent's task
input, so the LLM's narrative explanation is actually anchored in retrieved
text instead of purely generated from the model's parametric knowledge —
directly addressing "sparse narrative data" by supplying *relevant literature
context* the agent can reference, per this project's own novelty claim in
implementation_plan.md.
"""

from __future__ import annotations

import csv
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

_TOKEN_RE = re.compile(r"[a-zA-Z]+")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_RESEARCH_DIR = os.path.join(_PROJECT_ROOT, "research")

# ── Static ESI / Compensated Shock guideline snippets ─────────────────────────
# Original phrasing summarising the publicly-documented ESI acuity framework
# and this project's own Triage Rule Table (project details.pdf) — not
# reproduced verbatim from any single copyrighted source.

_STATIC_GUIDELINES = [
    {
        "id": "esi-1",
        "title": "ESI Level 1 — Resuscitation",
        "text": (
            "ESI Level 1 covers patients requiring immediate life-saving "
            "intervention — cardiac arrest, severe trauma, or physiological "
            "collapse. Vitals showing severe, combined cardiorespiratory "
            "compromise (marked tachycardia with marked tachypnea, or "
            "critical autonomic stress load) should route here."
        ),
    },
    {
        "id": "esi-2-shock",
        "title": "ESI Level 2 — Emergent / Compensated Shock",
        "text": (
            "Compensated Shock is the stage where the cardiovascular system "
            "is still maintaining blood pressure despite reduced perfusion, "
            "by increasing heart rate (tachycardia, >100 bpm) and increasing "
            "respiratory rate (tachypnea, >20 breaths/min) to blow off CO2 "
            "and correct metabolic acidosis. Because compensation masks the "
            "underlying deterioration, it is frequently missed in crowded "
            "waiting rooms until the patient decompensates. Any patient "
            "meeting both criteria simultaneously should be treated as "
            "Emergent (seen within 10-15 minutes), escalating to "
            "Resuscitation if the severity of both signs is pronounced."
        ),
    },
    {
        "id": "esi-3",
        "title": "ESI Level 3 — Urgent",
        "text": (
            "Urgent cases show a single abnormal vital sign (isolated "
            "tachycardia or isolated tachypnea) or moderate stress-index "
            "elevation without the combined shock pattern. These should be "
            "seen within 30-60 minutes; abdominal pain and moderate fever "
            "are typical examples."
        ),
    },
    {
        "id": "esi-4-5",
        "title": "ESI Levels 4-5 — Semi-Urgent / Non-Urgent",
        "text": (
            "Patients with vitals in the normal resting range, or only mild "
            "autonomic variability changes (slightly reduced HRV without "
            "tachycardia/tachypnea), fall into Semi-Urgent or Non-Urgent "
            "bands — routine care that can safely wait."
        ),
    },
    {
        "id": "hrv-stress",
        "title": "HRV and the Baevsky Stress Index as secondary evidence",
        "text": (
            "Reduced heart rate variability (low RMSSD) and an elevated "
            "Baevsky Stress Index both indicate reduced autonomic reserve. "
            "Neither is sufficient on its own to declare Compensated Shock, "
            "but both should raise clinical suspicion when combined with "
            "borderline heart rate or respiratory rate readings."
        ),
    },
    {
        "id": "rppg-limits",
        "title": "Known rPPG signal-quality limitations",
        "text": (
            "Remote photoplethysmography accuracy degrades under motion, "
            "poor or flickering illumination, and — per the melanin-"
            "absorption literature in this project's own research corpus — "
            "may be less reliable across the full range of skin tones "
            "without explicit stratified validation. Low Signal Quality "
            "Index readings should lower diagnostic confidence and trigger "
            "manual review rather than a confident automated classification."
        ),
    },
]


def _tokenize(text: str) -> list:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _tf_vector(tokens: list) -> Counter:
    return Counter(tokens)


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str  # "guideline" | "research"


@lru_cache(maxsize=1)
def _load_corpus() -> tuple:
    """Build the document set once per process: static guidelines + research CSVs."""
    docs = [
        Document(doc_id=g["id"], title=g["title"], text=g["text"], source="guideline")
        for g in _STATIC_GUIDELINES
    ]

    if os.path.isdir(_RESEARCH_DIR):
        for fname in sorted(os.listdir(_RESEARCH_DIR)):
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join(_RESEARCH_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    name_col = next(
                        (k for k in fieldnames if k and "paper" in k.lower()),
                        fieldnames[0] if fieldnames else None,
                    )
                    for i, row in enumerate(reader):
                        paper_name = (row.get(name_col) if name_col else None) or f"{fname}#{i}"
                        findings = row.get("Key Findings (At least 3 main findings)", "") or ""
                        limitations = row.get("Limitations", "") or ""
                        future = row.get("Future Scope", "") or ""
                        text = " ".join([findings, limitations, future]).strip()
                        if not text:
                            continue
                        docs.append(Document(
                            doc_id=f"{fname}:{i}",
                            title=f"{paper_name} ({fname.replace('.csv', '')})",
                            text=text,
                            source="research",
                        ))
            except Exception:
                # Corpus is a nice-to-have for grounding, never a hard dependency
                continue

    vectors = {d.doc_id: _tf_vector(_tokenize(d.title + " " + d.text)) for d in docs}
    return tuple(docs), vectors


def retrieve(query: str, top_k: int = 3, min_score: float = 0.05) -> list:
    """Return the top_k documents most similar to `query` by TF cosine similarity."""
    docs, vectors = _load_corpus()
    q_vec = _tf_vector(_tokenize(query))
    if not q_vec:
        return []

    scored = []
    for doc in docs:
        score = _cosine(q_vec, vectors[doc.doc_id])
        if score >= min_score:
            scored.append((score, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {"title": doc.title, "text": doc.text, "source": doc.source, "score": round(score, 3)}
        for score, doc in scored[:top_k]
    ]


def retrieve_for_assessment(assessment, top_k: int = 3) -> list:
    """
    Build a retrieval query from a core.clinical.ClinicalAssessment and return
    grounding snippets — guideline text plus relevant literature findings.
    """
    query_parts = [assessment.primary_diagnosis, assessment.esi_category]
    if assessment.is_shock:
        query_parts.append("compensated shock tachycardia tachypnea")
    query_parts.extend(assessment.reasoning)
    return retrieve(" ".join(query_parts), top_k=top_k)


def grounding_context(assessment, top_k: int = 3) -> str:
    """Render retrieved snippets as plain text for injection into an LLM prompt."""
    hits = retrieve_for_assessment(assessment, top_k=top_k)
    if not hits:
        return "No directly relevant guideline/literature snippets retrieved."
    lines = ["RETRIEVED GROUNDING CONTEXT (local term-overlap retrieval, top matches):"]
    for h in hits:
        lines.append(f"  [{h['source']}] {h['title']} (score={h['score']}): {h['text'][:400]}")
    return "\n".join(lines)
