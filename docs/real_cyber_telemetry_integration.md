# Phase 8 — Real Cyber Telemetry Baseline (Proposed / Partially Implemented)

## 1. Objective

Establish a real, labeled automotive CAN intrusion dataset as an explicit
comparison baseline against this project's existing synthetic cyber
telemetry, without merging the two, so any future claim about
real-world detection performance is traceable to real-world data.

## 2. Status

| Item | Status |
|---|---|
| Dataset selection | Decided — HCRL Car-Hacking dataset |
| Ingestion + release-gate script (v2) | **PASS — 26/26 checks, executed against the real downloaded dataset on 2026-09-15.** v1 assumed a header row and failed (0/5 files); v2 fixed the parsing to match the real headerless, variable-width format. |
| Real evidence | **16,569,475 real CAN frames ingested** — see table below |
| Phase 8.2 baseline detection model (code) | Implemented, smoke-tested on dummy data (pipeline runs end-to-end; dummy-data metrics are meaningless by design — no real signal was in the dummy rows) |
| Phase 8.2 run against real data | **Not yet executed** — pending local run |
| Feature-parity mapping to the existing 59-input model | Not yet started — deferred until Phase 8.2's standalone baseline is evaluated |

### Phase 8.1 real ingestion results (2026-09-15)

| File | Rows | Attack label | Attack count | Normal count |
|---|---|---|---|---|
| DoS_dataset.csv | 3,665,771 | DOS | 587,521 | 3,078,250 |
| Fuzzy_dataset.csv | 3,838,860 | FUZZY | 491,847 | 3,347,013 |
| gear_dataset.csv | 4,443,142 | GEAR_SPOOFING | 597,252 | 3,845,890 |
| RPM_dataset.csv | 4,621,702 | RPM_SPOOFING | 654,897 | 3,966,805 |
| **Total** | **16,569,475** | | | |

## 2a. Phase 8.2 — Baseline Detection Model

- **Objective:** independent, real-data intrusion-detection baseline (Layer 3: detection), using standard CAN-IDS features — inter-arrival time (`delta_t`), rolling 1s frequency (`freq_1s`), DLC, Hamming distance from the previous same-ID frame, and a DLC-change flag. Deliberately independent of the existing 59-dim vehicle-state schema and the frozen Transformer checkpoint.
- **Model:** RandomForestClassifier, deliberately bounded (`n_estimators=50`, `max_depth=20`) for tractability on 11M+ training rows — a documented simplification, not a tuned/optimized configuration.
- **Split:** time-based, 70/30, per source file (not randomly shuffled — CAN traffic is sequential and a random split would leak information into `delta_t`/`freq_1s`).
- **Code:** `scripts/real_cyber_telemetry/phase8_2_hcrl_feature_baseline_model.py`
- **Evidence (once run):** `experiments/real_cyber_telemetry/phase8_2_baseline_model_report.json`
- **Model artifact (once run):** `models/real_cyber_telemetry_baseline/rf_baseline_model.joblib`

## 3. Dataset

- **Name:** HCRL Car-Hacking Dataset
- **Source:** https://ocslab.hksecurity.net/Datasets/car-hacking-dataset/
- **Vehicle:** Hyundai YF Sonata (real vehicle, OBD-II CAN capture — not simulated)
- **Attack types:** DoS, Fuzzy, Gear spoofing, RPM spoofing
- **Scale (per published documentation):** ~988,872 attack-free frames;
  ~16.6 million total frames across all four attack files combined.
- **License / access:** publicly documented and widely cited in
  peer-reviewed literature; verify current access terms on the source
  page before use, since this was not re-verified by direct download
  in this session.

## 4. Why this dataset first (of the candidates discussed)

Simpler, well-documented tabular format (timestamp, CAN ID, DLC, 8 data
bytes, R/T flag) integrates faster than ROAD's raw `candump` format.
ROAD (Oak Ridge National Laboratory, via Zenodo) remains the stronger
follow-on candidate — it includes real masquerade attacks, which HCRL
does not — but is heavier to integrate and is deferred to Phase 8.2.

## 5. Data-integrity discipline

The canonical output is tagged `data_class = REAL_CYBER_TELEMETRY_BASELINE`
on every row. This project's existing synthetic cyber telemetry must
never be silently combined with this data. Any future fusion step must
be an explicit, separately-reviewed operation, not a default join.

## 6. Experiment definition (per project template)

- **Initial state:** four raw HCRL CSV files placed under
  `data/raw/real_cyber_telemetry/hcrl_car_hacking/`.
- **Stimulus:** run `phase8_1_hcrl_car_hacking_ingest_and_gate.py`.
- **Expected behavior:** script detects the real header, ingests all
  four files, and either passes or fails each check explicitly —
  it does not silently coerce malformed rows.
- **Observed signals:** per-file row counts, label distribution,
  timestamp-monotonicity violations, invalid CAN ID / DLC counts.
- **Pass/fail criteria:** all checks pass AND at least one file
  ingests successfully AND zero exact-duplicate rows.
- **Evidence:** `experiments/real_cyber_telemetry/phase8_1_hcrl_ingest_gate.json`

## 7. Explicit limitations

- This ingest step does **not** evaluate your existing Transformer
  model against real data — that is Phase 8.2+ and requires mapping
  HCRL's 11-field schema onto (a subset of) the 59-dimensional feature
  space, which is a nontrivial follow-on task, not a trivial column
  rename.
- HCRL is a single vehicle, single make/model. Passing this gate does
  not establish cross-vehicle generalization.
- "Real data" here means real CAN traffic and real injected attacks —
  it does not by itself validate real-world safety consequences; that
  still requires your existing CRSS-based safety/risk layer.

## 8. Next decision point

Once you've run this locally and it passes against your actual
downloaded files, the next concrete choice is: (a) map a subset of
HCRL's fields onto your existing feature schema for a same-model
comparison, or (b) train/evaluate a separate baseline model on HCRL
alone before attempting feature-level fusion. Recommend (b) first —
it's the cleaner scientific comparison and doesn't risk contaminating
your existing frozen-checkpoint evidence chain.
