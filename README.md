<div align="center">

<img src="frontend/public/vital_logo.png" width="120" alt="VITAL Logo"/>

# VITAL: Vision-Based Intelligent Triage and Autonomous Lifesign Analytics

### 🔬 Towards contactless health screening, serverless telemetry analytics, and AI-driven clinical coordination.

<p align="center">
  VITAL extracts physiological vitals from standard webcams via remote photoplethysmography (rPPG) and coordinates emergency triage through a CrewAI 3-Agent clinical swarm, a serverless Neon database, and the voice-enabled ARIA clinical assistant.
</p>

<br/>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-Frontend-000000?style=for-the-badge&logo=nextdotjs&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-API%20Backend-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/PostgreSQL-Neon%20DB-336791?style=for-the-badge&logo=postgresql&logoColor=white"/>
  <img src="https://img.shields.io/badge/NVIDIA%20NIM-Llama%203.3-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/MediaPipe-Face%20Mesh-007acc?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/TailwindCSS-v4-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white"/>
</p>

<p align="center">
  <a href="#-why-vital">Why VITAL</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#%EF%B8%8F-the-triage-crew-swarm">Triage Swarm</a> •
  <a href="#-dsp-signal-processing-pipeline">DSP Pipeline</a> •
  <a href="#-aria-assistant">ARIA Chatbot</a> •
  <a href="#-local-setup">Setup</a> •
  <a href="#-database-schema">Database Schema</a>
</p>

</div>

---

# 💡 Why VITAL?

Conventional triage relies on physical sensors and static **First-In, First-Out (FIFO)** queuing. In high-surge emergency departments, this causes delays that contribute to patient complications. 

VITAL addresses three major limitations identified in emergency medicine literature (such as the 30 papers in our `research/` directory):

* **Tabular Note Sparsity**: The system evaluates patients through polynomial vital cross-correlations (e.g., elevated heart rate paired with tachypnea to identify Compensated Shock), filling information gaps dynamically.
* **Clinical Alert Fatigue**: Instead of raw numbers, VITAL translates biometrics into structured, explainable classifications grounded in official **Emergency Severity Index (ESI)** and **WHO** guidelines.
* **Ineffective Static Queuing**: The system maintains a dynamic queue that automatically prioritizes high-acuity patients (especially those in active shock) at the top of the waitlist.

---

# ✨ Features

## 📷 Contactless rPPG Acquisition
* **Clinically-anchored landmarks**: MediaPipe 468-point Face Mesh isolates forehead (151) and malar-cheek (117/346) landmarks per R2I-rPPG (Nagar et al., 2024), with yaw-aware cheek gating. *(Previous revisions of this README referenced hairline-boundary landmarks `[10,338,297,332,284]` that were already replaced in code — this doc had drifted from the implementation; see `core/camera.py`.)*
* **Environmental Guards**: Automatically measures BT.601 luminance (lux) and centroid drift (motion delta) to flag sensor noise or target motion.
* **Haar Fallback**: Operates fully on Haar Cascades if MediaPipe components are missing.
* **Skin-tone robustness — status**: CHROM's melanin-robust chrominance projection is designed to reduce (not eliminate) skin-tone sensitivity, but this project has **not yet run a stratified accuracy evaluation across skin tones** — see [Validation](#-validation) below. Treat any inclusivity claim as a design goal pending evidence, not a proven result.

## 🤖 Multi-Agent Triage Crew
* **Perception Specialist Agent**: Consolidates raw telemetry from live streams or video files.
* **Clinical Diagnostician Agent**: Narrates and explains a **deterministic** ESI/Compensated-Shock assessment (`core/clinical.py`) grounded in retrieved guideline context (`core/guidelines.py`) — see [Deterministic Clinical Engine](#-deterministic-clinical-engine) below. The agent does not compute or override the ESI level; it explains it.
* **Flow Coordinator Agent**: Restates the deterministic priority score (1-100) and produces the queue-facing summary.

## 💬 ARIA AI Chatbot
* **Context Ingestion**: Inherits read-only privileges to query Neon PostgreSQL patient records.
* **Multimodal OCR**: Allows uploading patient intake charts or clinical sheets, parsing the images directly.
* **Voice-Activated**: Integrates speech-to-text (STT) voice recognition and text-to-speech (TTS) playback in Next.js.

## 🗄️ Serverless Postgres Persistence
* All telemetry data, ESI levels, diagnostic summaries, and coordinator prioritizations are saved to a Neon database.
* The centralized queue loads records automatically sorted by urgency.

---

# 🏗 Architecture

```mermaid
graph TD
    User([User]) --> FE[Next.js Frontend]
    FE --> API[Flask Backend API]

    subgraph Perception["📷 Perception Layer (camera.py & rppg.py)"]
        API --> Cam[Camera Capture]
        Cam --> ROI[Face Mesh ROI Extraction]
        Cam --> Lux[Pseudo-Lux Estimation]
        Cam --> Motion[Centroid Drift Tracking]
        
        API --> DSP[rPPG DSP Engine]
        DSP --> Normalization[RGB Temporal Normalization]
        DSP --> Projection[CHROM Projections]
        DSP --> Filtering[Butterworth Bandpass Filter]
        DSP --> Spectral[Welch PSD Peak Finding]
        DSP --> HRV[Cubic Spline 250Hz HRV]
        DSP --> Respiration[RSA Respiration Filter]
    end

    subgraph Swarm["🤖 Multi-Agent Triage Crew (crew.py)"]
        API --> Crew[CrewAI Orchestrator]
        Crew --> PerceptionAgent[Perception Agent]
        Crew --> DiagnosticAgent[Diagnostic Agent]
        Crew --> CoordinatorAgent[Coordinator Agent]
        
        PerceptionAgent --> Tool[Vitals Extraction Tool]
        DiagnosticAgent --> ESI[ESI/WHO Guidelines Check]
        CoordinatorAgent --> JSON[Prioritized Queue Triage JSON]
    end

    subgraph Storage["🗄️ Neon Serverless Postgres (neon_db.py)"]
        API --> DB[(triage_records)]
    end

    FE --> ARIA[ARIA Assistant]
    API --> ARIA
    ARIA --> NIM[NVIDIA NIM Llama 3.3]
```

**Known constraint — one active live-camera session at a time.** `web/__init__.py` holds a single global `camera`/`rppg_engine`/`session`, matching the physical reality of one workstation with one webcam. The "dynamic triage queue" shown on the dashboard is a **sequential** log of completed triage runs sorted by ESI/priority after the fact, not concurrent live monitoring of multiple simultaneous patients — a real ED waiting room has dozens of patients at once, which this architecture does not model. Uploaded-video triage runs (`/api/triage_run` with a file, as opposed to `source: "live"`) are now safe to execute concurrently from multiple requests (see `build_triage_crew()` below), but the live-camera path is inherently single-stream.

---

# 🕵️ The Triage Crew Swarm

VITAL utilizes a sequential multi-agent task execution pipeline structured as follows:

| Agent | Role | Goal | Backstory |
| :--- | :--- | :--- | :--- |
| **Perception Agent** | Sensor Specialist | Restate the already-extracted heart rate, respiratory rate, and HRV metrics as a structured summary. | Expert in computer vision pipelines and rPPG signal analysis. |
| **Diagnostic Agent** | Clinical Diagnostician | Explain the deterministic ESI level and shock flag in clinical language, grounded in retrieved guideline context — does not compute or override the numbers. | Trained emergency diagnostician specializing in guideline-grounded triage. |
| **Coordinator Agent** | Flow Coordinator | Restate the deterministic priority score and produce the queue-facing summary. | Flow manager dedicated to optimizing patient throughput and safety. |

*Why the LLM doesn't compute the ESI level anymore*: earlier revisions let the Diagnostic/Coordinator agents freely generate the ESI level and shock flag as JSON, parsed with regex, falling back to a hardcoded `esi_level=3` on any parse failure. That made the single most safety-critical number in the system non-deterministic. See [Deterministic Clinical Engine](#-deterministic-clinical-engine).

Concurrency note: each `/api/triage_run` call now builds a **fresh** Agent/Task/Crew set (`agents/crew.py::build_triage_crew()`) rather than reusing one shared module-level `Crew` object — CrewAI `Task` objects store their own output on `self` and are not safe to re-enter concurrently from multiple Flask threads on the same instance.

---

# 🧠 Deterministic Clinical Engine

`core/clinical.py` computes the ESI level, Compensated Shock flag, and priority score directly from extracted vitals — in plain Python, not sampled from an LLM. The LLM agents above narrate this result; they do not produce it.

**Compensated Shock as a real polynomial cross-correlation, not an AND-gate.** A boolean `HR>100 and RR>20` treats a barely-over-threshold reading identically to a severe crisis. Instead:

```
tachy_hr = ramp(HR, threshold=100, span=40)      # 0 → 1 as HR climbs from 100 to 140 bpm
tachy_rr = ramp(RR, threshold=20,  span=15)      # 0 → 1 as RR climbs from 20 to 35 br/min
shock_interaction = tachy_hr * tachy_rr           # multiplicative, not additive
```

Multiplying (rather than OR-ing or averaging) means risk only accumulates when **both** vitals are elevated together — an isolated tachycardia with normal respiration scores near zero on this term, matching the physiological definition of *compensated* shock as a combined cardiorespiratory response. The literal `HR>100 AND RR>20` threshold from this project's own Triage Rule Table still gates the boolean `is_shock` flag exactly as specified; the interaction term grades *severity* on top of that (ESI 1 vs. ESI 2, not shock vs. no-shock).

HRV (RMSSD) and the Baevsky Stress Index are folded in as secondary, lower-weight terms — they shift the composite risk score but cannot trigger the shock flag alone. Low signal-quality (SQI) readings are explicitly flagged `needs_manual_review` rather than confidently classified.

# 📚 Guideline & Literature Retrieval (honest scope)

`core/guidelines.py` retrieves relevant ESI/WHO guideline snippets and findings from the 30-paper `research/` corpus using **local term-frequency cosine similarity** — a real, classic vector-space-model IR technique, but a much simpler one than dense embeddings. Earlier project documentation implied a FAISS/OpenAIEmbeddings-style RAG pipeline; that was never built. This module is what actually grounds the Diagnostic Agent's narrative today. Swapping in real embeddings later is a drop-in change at `core/guidelines.py::retrieve()`.

---

# 🧬 DSP Signal Processing Pipeline

1. **Color Normalization**: Normalizes skin color channels to offset melanin levels:
   $$R_n(t) = \frac{R(t)}{\mu_R}, \quad G_n(t) = \frac{G(t)}{\mu_G}, \quad B_n(t) = \frac{B(t)}{\mu_B}$$
2. **Chrominance Projection**: Projects colors onto specular-free vectors:
   $$X = 3R_n - 2G_n, \quad Y = 1.5R_n + G_n - 1.5B_n$$
3. **Adaptive Scaling (CHROM)**: 
   $$Pulse(t) = X - \left(\frac{\sigma(X)}{\sigma(Y)}\right) \cdot Y$$
4. **Respiration via RIV Fusion**: Extracts breathing rate from *respiratory-induced variations* measured per heartbeat — amplitude (RIAV) and baseline intensity (RIIV) — each resampled onto a uniform 4 Hz grid via cubic spline, detrended, bandpass-filtered (0.10–0.50 Hz / 6–30 breaths/min), spectrally estimated, and fused by median. Resampling to 4 Hz places Nyquist at 2 Hz, which removes the pulse by construction so it cannot contaminate the respiratory band. Beat detection for the underlying RIV series uses a filter narrowed around the currently-confirmed heart rate (not a static wide band), which prevents a heart rate's own 2nd harmonic from being double-counted as extra beats. A third RIV (beat-interval/RSA, RIFV) is extracted but deliberately excluded from fusion — measured to be near-noise on the validation dataset (see Validation section). Following Pimentel et al. (2016) and Karlen et al. (2013). This replaced a direct bandpass of the raw waveform, which validation showed had **no correlation with real respiration**.
5. **Cubic Spline Upsampling**: Upsamples the PPG signal to 250 Hz (4 ms quantization) for high-accuracy peak timing used to calculate RMSSD HRV.

**POS algorithm — present but not currently wired into the live product.** `core/rppg.py` also implements POS (Wang et al., 2017) as a hot-swappable alternative projection, but every instantiation site in this codebase (`web/__init__.py`, `web/routes.py`, `scripts/analyze_video.py`) constructs `AdvancedRPPG` with the default `algorithm="CHROM"` — there is no API route or UI control that switches it. Validating it (see below) surfaced a real accuracy issue, which is why it stays opt-in rather than default.

---

# 💬 ARIA Assistant

**ARIA (Adaptive Real-time Intelligence for Acute-care)** is integrated directly into the dashboard.

* **Database Context**: Automatically receives active patient telemetry to answer operational queries.
* **OCR Support**: Users can upload document screenshots (base64 payload). ARIA runs multimodal document parsing to evaluate blood pressure, history, and intake info.
* **Strict Formatting**: Constrained to return standard capitalized headers and simple bullet points without markdown bold stars or hashtags, keeping TTS output clean.

---

# 🛠️ Local Setup

Make sure you have Python 3.9+ and Node.js v18+ installed.

### 1. Backend Server Setup
```bash
# Clone the repository
git clone https://github.com/User-s22/Opti-Screen.git
cd Opti-Screen

# Setup virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

Create a `.env` file in the root folder:
```env
NVIDIA_API_KEY=your_nvidia_api_key
DATABASE_URL=your_neon_postgres_db_url
FLASK_DEBUG=false
```

Start the Flask server:
```bash
python app.py
```
*API Backend launches on `http://localhost:5002`*

### 2. Next.js Dashboard Setup
```bash
cd frontend
npm install
npm run dev
```
*SaaS Dashboard launches on `http://localhost:3000`*

---

# 🗃️ Database Schema

VITAL persists patient records to the `triage_records` table:

| Field | Type | Description |
| :--- | :--- | :--- |
| **id** | `VARCHAR(50) [PK]` | Unique patient record UUID string |
| **name** | `VARCHAR(100)` | Patient's full name |
| **timestamp** | `VARCHAR(30)` | Acquisition date-time string |
| **video_path** | `VARCHAR(255)` | Path to file, or 'live' |
| **esi_level** | `INTEGER` | Urgency classification (1 = critical, 5 = non-urgent) |
| **priority_score**| `INTEGER` | Emergency department triage priority (1 to 100) |
| **primary_diagnosis**| `VARCHAR(255)`| Extracted primary clinical assessment |
| **is_shock** | `BOOLEAN` | True if criteria for Compensated Shock is met |
| **triage_summary** | `TEXT` | Concise summary of triage logic |
| **agent_output** | `TEXT` | Full raw multi-agent diagnostic logs |
| **heart_rate** | `DOUBLE` | Computed resting BPM |
| **respiration** | `DOUBLE` | Computed breaths per minute |
| **hrv** | `DOUBLE` | Upsampled RMSSD value in milliseconds |
| **stress_index** | `DOUBLE` | Calculated Baevsky Stress Index score |

---

# ✅ Validation

Several independent validation modes — see `scripts/validate_dataset.py` and `scripts/validate_rr_bidmc.py`. Do not conflate them: some validate the DSP math, others validate end-to-end real-world accuracy. **Heart rate and respiratory rate are validated separately, on different datasets, to different standards** — the sections below keep them apart deliberately.

### 1. Synthetic signal validation (runs today, no external data needed)
A known-BPM RGB trace (physiologically-modelled: green channel carries the strongest AC component, plus realistic sensor noise and a slow illumination drift) is fed directly into `AdvancedRPPG`, bypassing camera capture and face detection. This isolates the DSP pipeline itself.

```bash
python scripts/validate_dataset.py
```

Measured results (7 known-BPM trials, 50-140 bpm, 2% sensor noise):

| Algorithm | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- |
| **CHROM** (shipped default) | **0.29 BPM** | 0.53 BPM | Accurate across the full tested range under noise. |
| **POS** (present, not wired into the app) | 16.00 BPM | 28.88 BPM | Traced (unlike the original 26.71 number, which was measured but never root-caused): POS's cold-start estimate overshoots to roughly double the true rate — the mirror image of CHROM's undershoot bug — and the adaptive filter then reinforces it. A mirrored "demote to the half if it has comparable power" check (`core/rppg.py::_select_dominant_freq_hps`) fixed this specific pattern (one synthetic case went from 74.0 to 1.0 BPM error) without touching CHROM's numbers, but two of the seven synthetic cases still fail for a *different*, not-yet-root-caused reason, and real-video testing (first time ever run) still shows harmonic errors in both directions on some subjects. Full story: [`data/evolution/14-pos-subharmonic-demotion.md`](data/evolution/14-pos-subharmonic-demotion.md). Still flagged **experimental**, still not wired into the app. |

### 1b. Respiratory rate validation — 53 recordings, human-annotated breaths (BIDMC)

Respiratory rate is validated on a **different dataset from heart rate**, because none of the video datasets used for HR ship a respiration reference. The **BIDMC PPG and Respiration Dataset** (PhysioNet, Open Data Commons Attribution v1.0) provides 53 × 8-minute recordings with breath annotations from **two independent human annotators**.

```bash
python scripts/validate_rr_bidmc.py --baseline --stride 3.0
```

The photoplethysmogram is resampled 125 Hz → 30 fps first, so the test exercises the frame-rate regime the product actually runs at. A window is scored only when both annotators saw ≥3 breaths **and** agreed within 2 breaths/min.

| | Original implementation | Current (RIV fusion, 2-channel) |
| :--- | :--- | :--- |
| Per-window MAE | 11.88 br/min | **2.21 br/min** |
| **Session-level MAE** | 4.56 (from the 1 recording that produced a number) | **1.60 br/min** |
| Session median error | — | **0.37 br/min** |
| Sessions within 2 br/min | 0% | **81.1%** |
| Correlation with annotated breaths | **+0.048 (none)** | **+0.389** |
| Readings suppressed by the confidence gate | **99.7%** | 35.8% |

The original implementation **had no correlation with real respiration at all**. Root cause: a shared filter helper applied heart-rate band guards to every caller, silently rewriting the requested 0.10–0.50 Hz respiration band into **0.70–1.00 Hz (42–60 breaths/min)** — a filter centred on the heart-rate fundamental, *passing* the pulse rather than rejecting it. Verified directly with `sosfreqz`.

The first RIV-fusion rewrite (3-channel, linear interpolation) reached session MAE 2.14 — already a large jump from non-functional. Two further, independently-verified bugs pushed it to the current 1.60: **(1)** `rr_history` was silently appending each true estimate ~15 times due to a throttle/append mismatch, letting a single wrong-but-confident reading dominate the session median purely by being duplicated more; **(2)** RIV beat detection used a static wide pulse filter that let a heart rate's 2nd harmonic through unattenuated, over-counting beats by ~55% on a synthetic case with a *known, confirmed* heart rate and *no respiration at all* — proving it wasn't respiration-specific. Also: RIFV (the RSA/beat-interval channel) was dropped from fusion after measuring it as near-noise (slope 0.09–0.14 alone, likely blunted RSA in this ICU/ventilated population), and linear interpolation between beat-spaced samples was replaced with cubic spline, which measurably reduced systematic shrinkage toward the mean (regression slope of estimate-on-truth: 0.339 → 0.541; 1.0 = unbiased). Full story with every measurement: [`data/evolution/19-respiratory-rate-ground-truth.md`](data/evolution/19-respiratory-rate-ground-truth.md).

**Scope limits, stated plainly:**
- BIDMC is **contact pulse oximetry, not webcam video**. This validates the RR extraction DSP on a clean signal. It does **not** validate the camera/ROI/face-detection layer, and does **not** establish that the same accuracy holds for a webcam-derived waveform, which is far noisier. There is still **no end-to-end webcam RR accuracy number** — obtaining one requires COHFACE or a respiration-belt pilot recording.
- **The estimator still systematically shrinks toward the population mode** (~17–18 br/min) rather than tracking the true rate — improved but not solved. This biases toward under-reading high RR, the clinically unsafe direction, since the tachypnea threshold is 20 br/min and a suppressed tachypnea also suppresses the Compensated Shock interaction term. The dominant remaining cause is unresolved — gate selection, search-band width, and Welch sub-windowing were all tested as explanations and refuted.
- A handful of individual recordings (`bidmc_32`, `bidmc_33`, `bidmc_36`, `bidmc_43`) remain large outliers (6–16 br/min session error) even after all fixes.
- Synthetic RR MAE (~6.7–6.9 br/min) is **worse** than the real-data figure — unresolved, and reported rather than hidden.

### 2. Real video validation — 56 real sessions across three independent datasets
This repository does not ship with dataset video data — each requires either a public-link download or a manual access request (see below). `scripts/build_manifest.py` turns a downloaded folder tree into a manifest automatically, handling three different ground-truth formats (UBFC-rPPG's `gtdump.xmp`/`ground_truth.txt`, and UBFC-Phys's raw BVP waveform, peak-detected into a BPM via `bpm_from_waveform()`):

```bash
python scripts/build_manifest.py path/to/downloaded/data/ --out manifest.csv
python scripts/validate_dataset.py --manifest manifest.csv --skip-synthetic
```

**Real measured results, 56 real sessions** — full pipeline (camera → MediaPipe face mesh → CHROM → adaptive filter), not the synthetic signal above:

| Source | Sessions | Notes |
| :--- | :--- | :--- |
| [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg) (public Drive link, no request needed) | 41 | Math-game stress protocol |
| [UBFC-Phys](https://ieee-dataport.org/open-access/ubfc-phys-2) (free IEEE DataPort account, instant) | 15 | 5 subjects × 3 tasks (rest/speech/arithmetic) |

| Metric | Value |
| :--- | :--- |
| Overall MAE | **8.4 BPM** (n=56) |
| Overall RMSE | 15.6 BPM |
| Status | CONSUMER GRADE overall (MAE ≥ 8 BPM) — **UBFC-rPPG subset alone (n=41) is 7.4 BPM, crossing into RESEARCH GRADE** |

**Stratified by condition:**

| Condition | MAE | n |
| :--- | :--- | :--- |
| Speech (talking) | **23.4 BPM** (worst — untouched by any fix so far, see why) | 5 |
| Rest | 6.5 BPM | 5 |
| Arithmetic (mental stress) | 3.5 BPM | 5 |
| UBFC-rPPG baseline (unlabeled) | 7.4 BPM | 41 |

Speech staying flat while Rest/Arithmetic improved dramatically (from a prior pass of 18.4/12.7) is itself informative, not a failure — it confirms the speech-condition gap is really about mouth movement disrupting the pulsatile signal, not the two bugs fixed below (neither of which had anything to do with talking).

**A second, unrelated bug found the same way as the first — by tracing an individual failing subject instead of guessing.** `test/dataset_2/subject25` showed a *completely different* failure signature from subject41: rock-stable, high-confidence, wrong the entire clip (not oscillating between two states). Checking its actual video file found the cause: it's recorded at **23.2 fps**, but every `AdvancedRPPG` construction site in this codebase hardcoded `fps=30` regardless of the source's real frame rate. Analyzing a 23.2fps video as if it were 30fps deterministically scales every frequency by (30/23.2 ≈ 1.29×) — 91.5 real bpm × 1.29 ≈ 118, matching the observed error almost exactly. Fixed by reading the actual FPS from the video file (`Camera.get_source_fps()`, `core/camera.py`) and using it everywhere an `AdvancedRPPG` engine is constructed — both the file-analysis path (`scripts/analyze_video.py`) and the live-webcam path (`web/routes.py::_reset`). Verified immediately on the two known-affected subjects: `subject25` 23.5→**3.5** BPM error, `subject26` 25.5→**2.5** BPM error.

Combined effect of the fixes so far across all 56 sessions: **MAE 17.4 → 11.4 → 8.4 BPM (52% total reduction)**, RMSE 24.1 → 18.1 → 15.6. (A third bug, found after this section — see below — moved the last step from 8.7 to 8.4.)

**The harmonic-ambiguity bug — root-caused, then genuinely (partially) fixed.** First found on `test/dataset_2/subject41`: the CHROM peak-picker was locking onto a spurious frequency at roughly half or double the true heart rate for sustained multi-second stretches, reproduced identically across three independent datasets (UBFC-rPPG, UBFC-Phys, rPPG-10 below).

*Two earlier fix attempts genuinely failed and are kept in the code history/comments rather than erased, because that's real information:*
1. A debounced BPM lock/candidate hysteresis (`AdvancedRPPG._resolve_bpm_lock`) — verified to work exactly as designed via trace, but produced **no aggregate improvement**, because the false lock isn't noise-driven flip-flopping, it's sustained and spectrally confident.
2. A naive upward-summing Harmonic Product Spectrum — verified via trace to also produce **no change**, and reasoned through why: for a signal locked at half the true rate, the true signal's own energy sits exactly where that candidate's "2nd harmonic" would be, so naive HPS reads it as supporting evidence *for* the wrong candidate rather than against it. Wrong tool for this specific failure direction.

*What actually worked*: an asymmetric subharmonic-promotion check (`_select_dominant_freq_hps` in `core/rppg.py`, despite the name — see its docstring) — for the top raw candidate frequency, explicitly check whether its **double** has comparable-or-greater spectral power; if so, promote to the double instead of trusting the raw peak. This targets the specific, evidenced failure direction (undershooting to ~half-rate) rather than a generic harmonic-support heuristic. Measured effect across all 56 real sessions:

| Subject | Before | After |
| :--- | :--- | :--- |
| subject41 | 52.6 | **8.2** |
| subject46 | 41.2 | **1.3** |
| subject49 | 36.5 | **0.4** |
| subject9 | 60.9 | **2.8** |
| subject34 | 63.0 | **0.5** |

**Honest downside — the subharmonic-promotion fix is not a universal win, even after the FPS fix recovered some of its casualties.** The FPS fix incidentally fixed one subharmonic-fix regression outright (UBFC-Phys `s4` rest: was 15.9 pre-fixes, 43.3 after subharmonic-promotion alone, 3.7 with both fixes — it turned out to also be a non-30fps video). But several sessions remained worse than the original, pre-any-fix baseline: `dataset_1/after-exercise` (11.0→32.9), `subject22` (41.0→46.8), `subject24` (5.9→20.4), `subject32` (6.0→19.3), `subject37` (4.8→21.0, since fixed — see below), and UBFC-Phys `s4` speech (9.6→22.5) — promoting to double the rate helps when the lock was genuinely a half-rate error, but occasionally misfires on a session that was already tracking correctly. These regressions were real, open, and not root-caused the way subject41/subject25 were — until one of them was traced.

**A third, independent bug — found by tracing `subject37`'s regression instead of assuming it was more subharmonic-promotion fallout.** Its per-cycle BPM readings converged and held rock-stable near 58 BPM (correct — ground truth 63.0) for nearly the whole clip, yet the session's final reported value was 84 BPM. Root cause was not in frequency estimation at all — it was in `get_final_summary()`'s history aggregation: the function only accepts a new reading into its history if it's within 18 BPM of the median of the *last 5 accepted* readings, to keep single-frame noise out of the final answer. But if the first handful of accepted readings happen to land during a brief wrong burst early in cold-start convergence, every later — correct, stable — reading gets rejected forever, because the rejection anchor itself never updates. `subject37`'s entire session-summary history ended up with just 7 entries, all from a wrong 82-89 BPM burst in the first couple of seconds; the correct ~58 BPM signal that dominated the rest of the two-thousand-frame clip was silently discarded the whole time.

The first fix attempt (allow the anchor to "unstick" and re-anchor after any sufficiently long streak of sustained disagreement) genuinely fixed `subject37` — but broke `subject41`, dropping it from the celebrated 8.2 BPM error back to 41.8. Tracing *why* revealed something more interesting than a tuning mistake: `subject41`'s per-cycle readings are **still genuinely bimodal across the whole clip** even after subharmonic-promotion (in this trace, roughly 983 low-cluster vs. 933 high-cluster readings) — so the original "fixed, 8.2 BPM error" result was never a real resolution of subject41's ambiguity. It was this exact freeze bug accidentally anchoring on the correct cluster early and then never letting go — a second, unrelated bug propping up the reported result of the first fix. Unsticking unconditionally simply let the still-unresolved ambiguity resurface.

The corrected fix narrows the escape hatch: re-anchoring is only allowed while the frozen anchor is still small (≤10 entries — genuinely untested, cold-start-fluke territory), never once it has accumulated substantial support. Verified: `subject37` stays fixed (21.0 → **6.0**), `subject41` is fully restored (**8.2**, unchanged), and the other pre-existing regressions (`subject22/24/32`, `after-exercise`, UBFC-Phys `s4` speech) are untouched either way — confirming they have a different, still-open root cause, not this one.

Combined effect of all three fixes across all 56 sessions: **MAE 17.4 → 11.4 → 8.7 → 8.4 BPM**, RMSE 24.1 → 18.1 → 15.8 → 15.6 — a small but real, individually-verified improvement, not a guess. Full story, including the failed naive version and its regression trace, in [`data/evolution/09-history-freeze-bug.md`](data/evolution/09-history-freeze-bug.md).

No skin-tone stratification exists yet — none of the three datasets used ship per-subject Fitzpatrick metadata (rPPG-10 does, but see below for why it can't run through this pipeline). The Fitzpatrick-inclusivity design goal stated earlier in this README remains unvalidated, not disproven — it just hasn't been tested.

### 3. DSP-core-only validation (rPPG-10) — a different, narrower test
[rPPG-10](https://data.mendeley.com/datasets/bx8982xgwt/1) (Mendeley, direct download, no request) ships pre-cropped 64×64 forehead/cheek patches, not full-face frames — MediaPipe's face mesh cannot find a face in these, so this dataset **cannot run through the standard camera+face-detection pipeline above**. `scripts/validate_cropped_roi.py` bypasses face detection entirely (each frame IS already the ROI, spatially averaged directly into `AdvancedRPPG`) to test the DSP core alone against real camera footage and real ECG-derived ground truth (`bpm_from_waveform()` again, this time on a 1000 Hz ECG signal):

```bash
python scripts/validate_cropped_roi.py path/to/extracted/Dataset_rPPG-10
```

| Metric | Value |
| :--- | :--- |
| MAE | 24.5 BPM (n=26, 1 subject failed on a corrupt file) |
| RMSE | 33.0 BPM |

Worse than the full-pipeline number above, on lower-resolution, more heavily compressed footage — and again shows the same harmonic pattern (e.g. one subject at exactly 0.50× the true rate). **Do not average this into the full-pipeline number above** — it's deliberately testing something different (no face detection/ROI-selection involved at all) and isn't directly comparable.

**Performance note**: these are 10-minute clips (18,000 frames); the DSP core originally re-ran a full Welch FFT + cubic-spline HRV fit on every single frame, which took over 13 minutes with zero results for even one subject. `analyze_video_complete()` and the cropped-ROI script now support a `process_stride` (skip N-1 of every N expensive recomputations while still buffering every frame) — cut this specific run from an hours-long trajectory to ~10 minutes total. The live product path keeps `process_stride=1` (unchanged, real-time frame-by-frame updates); only batch validation over long clips uses a higher stride.

### 4. Hybrid classical-DSP + ML correction layer

Sections 1-3 above are all pure classical signal processing (CHROM/POS) —
fixed equations, no learning. `scripts/build_ml_dataset.py`,
`scripts/train_correction_model.py`, and `scripts/validate_with_correction.py`
add a second stage on top: a small gradient-boosted model trained to
correct the classical pipeline's output, using dense per-frame/near-per-
frame ground truth (not just one scalar BPM per session) extracted from
the same real UBFC-rPPG/UBFC-Phys videos — 242,026 labeled cycles across
71 real sessions (46→**56** unique people once 5 already-downloaded but
unused UBFC-Phys subjects were found and added).

```bash
python scripts/build_ml_dataset.py test/ --out test/ml_dataset.csv
python scripts/train_correction_model.py test/ml_dataset.csv --folds 5 --save-fold-models models/cv_folds
python scripts/validate_with_correction.py test/manifest.csv models/cv_folds
```

**The one rule that makes this honest**: every number below is measured
with the model corrected by evaluated **only on subjects it never trained
on** (`GroupKFold` by `subject_id`, and session-level validation using a
model trained on the *other* folds for each specific subject — see
`models/cv_folds/`). A model trained on all 71 people and then evaluated
on those same 71 people would be measuring memorization, not accuracy.

| | MAE | RMSE |
| :--- | :--- | :--- |
| Classical DSP only (sections 1-3) | 9.43 BPM | 16.51 |
| **Hybrid (DSP + ML correction)** | **6.48 BPM** | **9.66** |

(The correction model uses Huber loss, not squared-error — measured to be
a small but real improvement over squared-error at both the per-cycle
level, 8.80→8.67 MAE, and the session level, 6.65→6.48 MAE, on held-out
subjects either way. Squared-error over-weights a handful of extreme-
outlier sessions; Huber is more robust to them. See
[`data/evolution/12-huber-loss.md`](data/evolution/12-huber-loss.md).)

Per-condition, the correction model's biggest win is exactly the failure
mode three rounds of DSP-level fixes never touched:

| Condition | Classical | Hybrid | n |
| :--- | :--- | :--- | :--- |
| Speech (talking) | 21.0 (worst, untouched by every DSP fix above) | **7.8** | 10 |
| Arithmetic | 10.4 | **7.0** | 10 |
| UBFC-rPPG (unlabeled) | 7.4 | 6.1 | 41 |
| Rest | 5.1 | 6.1 (slightly worse) | 10 |

**Honest, not a free lunch**: the model helps every condition except
already-clean "rest," where it's correcting for bias patterns that aren't
present. Net effect across all 71 sessions is unambiguously positive
(both MAE and RMSE improve), but every group's number is
reported here, not just the aggregate.

**Bigger models were tested and did not win** — a larger, more heavily
regularized neural net, a larger gradient-boosted model, and an ensemble
of both were compared the same GroupKFold way against the small deployed
model. None beat it; the neural net was measurably worse. The real
constraint is 56 unique people, not the 242,026 rows — a small model
already extracts most of what that many people can teach it. Full story,
including the real per-fold numbers and a computational trap caught by
timing small samples before committing to a multi-hour run:
[`data/evolution/10-bigger-models-dont-help.md`](data/evolution/10-bigger-models-dont-help.md).

Adding 5 already-downloaded-but-unused UBFC-Phys subjects (46→56 people,
56→71 sessions) *widened* the hybrid model's advantage rather than
narrowing it — full story:
[`data/evolution/11-more-subjects.md`](data/evolution/11-more-subjects.md).

### 5. Agent narration faithfulness — does the LLM layer actually do what it's told?

Everything above validates the rPPG **signal** pipeline. None of it says anything about whether the multi-agent LLM layer (`agents/`) actually behaves as instructed — specifically, whether the Diagnostic/Coordinator agents faithfully **restate** the deterministic `core/clinical.py` assessment, or drift/hallucinate a different number, despite `agents/config/tasks.yaml` explicitly instructing them to restate exact given values and nothing else.

`scripts/evaluate_agent_reliability.py` runs the real CrewAI pipeline (actual NVIDIA NIM calls, not a mock) against 8 clinical scenarios spanning the full severity range (normal, isolated tachycardia/tachypnea, mild/severe shock, bradycardia, low-signal-quality, no-signal), each run twice, and checks whether the LLM's final JSON output exactly matches the deterministic assessment on every field:

```bash
python scripts/evaluate_agent_reliability.py --repeats 2
```

**Real measured results, 16 runs, `meta/llama-3.1-8b-instruct`:**

| Field | Faithfulness rate |
| :--- | :--- |
| JSON parses at all | 15/16 (93.8%) |
| ESI level restated correctly | 13/16 (81.2%) |
| **Priority score restated correctly** | **1/16 (6.2%)** |
| Shock flag restated correctly | 15/16 (93.8%) |
| Diagnosis text — exact wording | 6/16 (37.5%) |
| Diagnosis text — correct meaning, different words | 12/16 (75.0%) |

**Why the 6.2% number is the most important finding in this whole project, not just a QA metric.** Even when explicitly instructed "restate this exact number, do not recompute it," the LLM almost never produced a clean, correct integer for `priority_score` — it repeatedly stuffed descriptive text into the field instead (e.g. `"esi_level": "2 (Emergent)"`, `"priority_score": "ESI Level 2 — Emergent / Compensated Shock (score=0.311)"`). This is **empirical, measured proof** — not a defensive-programming assumption — that the project's core safety rule (the deterministic engine is the sole source of truth; the LLM narrates, it never computes or holds the authoritative number — see [The one rule that matters most here](CLAUDE.md)) was necessary, not merely cautious. Had this system trusted the LLM's own JSON for `priority_score`, as many "agentic AI in healthcare" systems do, it would have assigned the **wrong queue priority to a patient over 90% of the time.**

A related finding from the same runs: the LLM occasionally attempted to call tools that don't exist at all (`structured_summary_tool`, `clinical_summary_tool` — "Tool not found") — a second, independent reliability gap in the agent layer.

**A real mistake happened building this evaluation, and it's documented rather than hidden**: see [`data/EVOLUTION.md`](data/EVOLUTION.md), Era 8, for the full story of a 2-hour-20-minute infinite loop this caused, its root cause (a `Camera` dummy-mode fallback with no termination condition, combined with the LLM not reliably following a "don't re-call this tool" instruction), and the three independent layers of fix now in place against it recurring.

For the complete, chronological story of every bug found and every fix attempted across this entire project — including the ones that failed — see **[`data/EVOLUTION.md`](data/EVOLUTION.md)**.

---

# 🔒 Security Notes

* **CORS** is restricted to an explicit origin allowlist (`CORS_ORIGINS` env var, defaults to the local Next.js dev server) — previously `origins: "*"` allowed any website to call this API from a browser.
* **`DELETE /api/triage_queue`** (wipes every patient record, irreversible) is gated behind an `X-API-Key` header when `API_KEY` is set in `.env`. Unset by default so local development is unaffected — set it before exposing this beyond localhost.
* No other endpoints have authentication. This is a research/demo system, not a HIPAA-compliant deployment — treat it accordingly if extending it.

---

# ⚙️ CLI Utilities

* **Offline Vitals Extraction**:
  ```bash
  python scripts/analyze_video.py uploads/sample_face.mp4
  ```
* **Validation Harness** (synthetic self-test + optional real-dataset manifest — see [Validation](#-validation)):
  ```bash
  python scripts/validate_dataset.py
  python scripts/validate_dataset.py --manifest path/to/manifest.csv
  ```
* **Standalone NLP Toolkit** (`scripts/open.py`): a general-purpose text-processing utility (tokenization, bag-of-words, n-grams) kept for reference — it is not part of the triage pipeline and is not called anywhere in `app.py`/`web/`/`agents/`.
