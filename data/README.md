# data/ — the evidence archive

This folder exists so the full history of this project's validation and
debugging work survives independent of any one conversation, laptop, or
session. Start here:

- **[EVOLUTION.md](EVOLUTION.md)** — index into the full chronological
  story, now split into one file per era under
  [`evolution/`](evolution/): every bug found, every fix attempted
  (including the ones that failed and why), every real measured number, in
  the order it actually happened. Written so a person or an AI with zero
  prior context can read it and fully reconstruct what was done and why.

The raw evidence EVOLUTION.md references lives in [`test/`](../test/), not
duplicated here, to avoid two copies of the same file drifting apart:

| What | Where |
| :--- | :--- |
| Real 56-session validation results | `test/validation_run5_final.log`, `test/validation_run6_fpsfix.log` |
| Harmonic lock-on root-cause traces (3 fix attempts) | `test/trace_subject41*.log` |
| Hardcoded-FPS bug root-cause trace | `test/trace_subject25.log` |
| Agent narration faithfulness evaluation (raw + aggregated) | `test/agent_reliability_run2.log`, `test/agent_reliability_results.jsonl` |
| Manifest of every real video used, with ground truth | `test/manifest.csv` |

For the current, final numbers (not the history of how they were reached),
see [README.md](../README.md)'s Validation section. For guidance on working
in this codebase, see [CLAUDE.md](../CLAUDE.md).
