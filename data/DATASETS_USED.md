# Datasets used in this project — what they are, and why

Plain-language reference so it doesn't need to be re-discovered later. For
the detailed measured numbers and methodology, see
[`RR_PAPER_NOTES.md`](RR_PAPER_NOTES.md) (paper-shaped) and
[`evolution/19-respiratory-rate-ground-truth.md`](evolution/19-respiratory-rate-ground-truth.md)
(full chronological debugging story).

---

## BIDMC PPG and Respiration Dataset

**What it is**: 53 recordings, 8 minutes each, from critically-ill patients
at Beth Israel Deaconess Medical Center (Boston) — pulled from a larger
hospital ICU database (MIMIC II). Each recording has the patient's pulse
signal (PPG, from a finger clip) plus a breathing signal, and — the
important part — **two different human experts independently marked every
single breath by listening/watching**, so we have real, trustworthy
"this many breaths happened, at this exact moment" ground truth.

**Where it lives**: `dataset/bidmc-ppg-and-respiration-dataset-1.0.0/`

**Why we used it**: It was the very first dataset with real respiratory-rate
ground truth we could get **instantly**, no permission needed, no waiting —
we downloaded it directly from PhysioNet the same day. It let us discover
that our old RR-measuring code was actually broken (measuring almost nothing
real), and it's what we used to build and test the replacement.

**Population**: ICU patients — sicker than average, several on ventilators
(breathing machines). This matters because it means their bodies may behave
a bit differently than a healthy person sitting at a webcam, so it's a good
first test but not a perfect stand-in.

**License**: Open Data Commons Attribution — free to use for research.

---

## CapnoBase IEEE TBME Respiratory Rate Benchmark

**What it is**: 42 recordings, 8 minutes each, from patients under
anesthesia during surgery (mostly children, from a Vancouver hospital's
pediatric anesthesia research program). Instead of humans marking breaths by
ear/eye, this one measures breathing directly and very precisely via
**capnography** — a sensor that measures the CO2 you breathe out, which
gives an extremely reliable breath count. It also comes with the **original
published algorithm's own results already included**, so we can directly
compare "how did we do" vs. "how did the actual research paper's algorithm
do," on the exact same recordings.

**Where it lives**: `dataset/doi-10.5683-sp2-nlb8it/`

**Why we used it**: A *second, independent* way to check our results —
different hospital, different patients (mostly kids under anesthesia vs.
BIDMC's sick ICU adults), different way of measuring truth (CO2 sensor vs.
human ear). If our numbers hold up on both, that's much stronger evidence
than one dataset alone. It also gives us a fair, apples-to-apples comparison
against the actual published research our method is based on
(Karlen et al.'s "SmartFusion" algorithm).

**Two file formats inside**: a `data/csv/` folder (easy to read, but its
"comparison to the published algorithm" columns turned out to be empty/broken
in the CSV export) and a `data/mat/` folder (the original MATLAB format,
which DOES have the full comparison numbers — we had to install a small
Python library, `h5py`, to read this format, since it's a newer MATLAB
version that needs special handling).

**Result** (`scripts/validate_rr_capnobase.py`): we traced the worst-failing
recordings and found two separate causes — one is a hard physical limit (a
few patients breathe faster than our system was told to even look for,
since it was tuned for adult rates), and the other was a genuine bug where
the system locked onto exactly double the true breathing rate while being
*confident* it was right. We fixed the second one. After the fix, on the
*typical* recording we now **beat** the original published algorithm —
better median error (0.07 br/min vs. their 0.35) and more recordings landing
within 2 br/min of truth (83.3% vs. their 78.6%). Our overall average is
still a bit behind theirs (2.51 vs. 1.59) because of the still-unfixed
first cause (the patients breathing faster than our range covers) — that's
the honest remaining gap, and the same fix improved our BIDMC number too
(1.60 → 1.59), so it wasn't a trade-off between the two datasets.

**License**: Free for research use, credit the original 2013 paper.

---

## VitalDB

**What it is**: Real hospital monitor data from **6,388 actual surgeries**
at Seoul National University Hospital — heart rate, breathing, oxygen,
blood pressure, brain activity, and more, all continuously recorded during
real operations. Released publicly by the hospital's research team
specifically so people can build and test medical monitoring algorithms
like ours.

**Where it lives**: nowhere on disk yet — unlike the two above, this one is
accessed live over the internet (an API), not downloaded as a file. We pull
exactly the pieces we need directly, on demand.

**Why we're using it**: 
1. **Scale** — ~150x more patients than BIDMC and CapnoBase combined, which
   makes any "this works in general, not just on 50-ish special cases"
   claim much stronger.
2. **A third independent check** — different country, different hospital,
   different patients (general surgery, not ICU or pediatric anesthesia),
   different equipment. If our RR algorithm still works here too, that's
   real evidence it generalizes rather than being tuned to one dataset's
   quirks.
3. **It genuinely has what we need** — confirmed directly (not assumed):
   it has the same pulse signal our algorithm reads (`SNUADC/PLETH`) and a
   CO2-sensor-derived respiratory rate ground truth (`Solar8000/RR_CO2`,
   `Primus/RR_CO2`), the same kind of solid ground truth CapnoBase has.

**Access**: Genuinely free and open — no account, no application, no
waiting. Creative Commons Attribution license. This is why we didn't need
your help downloading it, unlike BIDMC/CapnoBase which needed a manual
download (or in COHFACE/MAHNOB's case, a permission request that never came
through / a dead website).

---

## Quick summary table

| Dataset | Patients | Population | Ground truth from | Access | Used for |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BIDMC | 53 | ICU (adults, some ventilated) | 2 human annotators listening | Instant download | First real RR validation; found + fixed the original broken estimator |
| CapnoBase | 42 | Anesthesia (mostly children) | CO2 sensor (capnography) | Instant download | Second independent check + direct comparison to the original published algorithm |
| VitalDB | 6,388 | General surgery (all ages) | CO2 sensor (capnography) | Live API, no permission needed | Large-scale third check — does it generalize beyond 2 small hospital datasets |

## What we tried and could NOT use

- **COHFACE** — actual webcam video + a real respiration belt (would have
  validated our camera/face pipeline, not just the algorithm on a clean
  signal) — requires a permission request that was submitted but never
  approved.
- **MAHNOB-HCI** — same idea, webcam video + respiration belt — the entire
  website is down (every page except the homepage returns "not found"),
  confirmed by trying it directly.
- **RR-rPPG** — a brand new dataset built for exactly this problem — the
  paper says the data "will be released upon publication," meaning it does
  not exist publicly yet.

None of the three above are webcam video, which is the one type of
validation we still don't have — that gap is intentionally left for your
own live webcam testing later.
