# Phase 8 — Real Cyber Telemetry Baseline
## Evidence Milestone — 2026-09-18

## 1. Objective

Establish a real, labeled automotive CAN intrusion dataset as an explicit
comparison baseline against this project's existing synthetic cyber
telemetry, without merging the two, so any future claim about real-world
detection performance is traceable to real-world data.

---

## 2. Status

| Item | Status |
|---|---|
| Dataset selection | **Decided — HCRL Car-Hacking Dataset** |
| Phase 8.1 ingestion + release gate | **PASS — 26/26 checks** |
| Real CAN frames ingested | **16,569,475** |
| Phase 8.2 baseline detection | **COMPLETED — 2026-09-16** |
| Phase 8.3 CAN-ID-aware detection | **COMPLETED — 2026-09-18** |
| Feature-parity mapping to existing 59-input model | **Not started — deliberately deferred** |
| Phase 8.4 residual-gap refinement | **Deferred — future work** |

### Phase 8.1 real ingestion results

| File | Rows | Attack label | Attack count | Normal count |
|---|---:|---|---:|---:|
| DoS_dataset.csv | 3,665,771 | DOS | 587,521 | 3,078,250 |
| Fuzzy_dataset.csv | 3,838,860 | FUZZY | 491,847 | 3,347,013 |
| gear_dataset.csv | 4,443,142 | GEAR_SPOOFING | 597,252 | 3,845,890 |
| RPM_dataset.csv | 4,621,702 | RPM_SPOOFING | 654,897 | 3,966,805 |
| **Total** | **16,569,475** | | | |

---

## 3. Phase 8.2 — Baseline Detection Model

### Objective

Establish an independent real-data intrusion-detection baseline using
standard CAN-IDS traffic features.

Features:

- `delta_t`
- `freq_1s`
- `dlc`
- `hamming_dist`
- `dlc_changed`

The model deliberately excludes CAN ID and remains independent of the
existing 59-dimensional vehicle-state schema and frozen Transformer
checkpoint.

### Model

`RandomForestClassifier`

- `n_estimators = 50`
- `max_depth = 20`
- `n_jobs = -1`
- `random_state = 42`

### Split

Time-based 70/30 split per source file.

No random shuffle was used.

### Train/test

- Training: **11,598,631**
- Testing: **4,970,844**

### Results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| DOS | 0.9998 | 0.9985 | 0.9992 | 21,164 |
| FUZZY | 0.9970 | 0.9895 | 0.9932 | 43,423 |
| GEAR_SPOOFING | 0.0010 | 0.0016 | 0.0012 | 69,037 |
| NORMAL | 0.9844 | 0.9662 | 0.9752 | 4,754,568 |
| RPM_SPOOFING | 0.1335 | 0.2023 | 0.1609 | 82,652 |

Overall:

- Accuracy: **94.0439%**
- Macro F1: **62.5937%**
- Weighted F1: **94.8407%**

Binary attack-vs-normal detection:

- TPR: **66.3490%**
- FPR: **3.3804%**
- Precision: **47.1687%**

### Diagnosis

Phase 8.2 showed severe confusion between GEAR_SPOOFING and
RPM_SPOOFING.

GEAR_SPOOFING was predicted as RPM_SPOOFING in:

**62,566 / 69,037 cases = 90.6%**

Feature importance was dominated by:

| Feature | Importance |
|---|---:|
| `delta_t` | 0.5081 |
| `freq_1s` | 0.3859 |
| `hamming_dist` | 0.1052 |
| `dlc` | 0.0007 |
| `dlc_changed` | 0.0001 |

The model did not receive CAN ID, so it could capture temporal traffic
behavior but could not directly distinguish the specific arbitration IDs
being spoofed.

### Evidence artifacts

- Code:
  `scripts\real_cyber_telemetry\phase8_2_hcrl_feature_baseline_model.py`
- Evidence:
  `experiments\real_cyber_telemetry\phase8_2_baseline_model_report.json`
- Model:
  `models\real_cyber_telemetry_baseline\rf_baseline_model.joblib`

---

# 4. Phase 8.3 — CAN-ID-Aware Detection Model

## 4.1 Hypothesis

Phase 8.2's GEAR/RPM confusion was caused by omitting CAN ID from the
feature set.

Adding CAN ID as a feature should improve GEAR_SPOOFING and
RPM_SPOOFING detection without requiring a new data source.

## 4.2 Implementation

The real training data contains **2,048 distinct CAN IDs**.

A full one-hot encoding attempt was not retained because it attempted to
allocate an approximately 22 GB array.

The implemented solution uses:

- Top 30 most frequent CAN IDs as individually encoded categories
- Remaining IDs bucketed as `OTHER`
- Total encoded categories: **31**

`OTHER` represented:

- Training: **3.8091%**
- Test: **0.8602%**

The same 70/30 time-based split and the same Random Forest configuration
were retained for comparison with Phase 8.2.

## 4.3 Results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| DOS | 1.0000 | 1.0000 | 1.0000 | 21,164 |
| FUZZY | 0.9984 | 0.9882 | 0.9933 | 43,423 |
| GEAR_SPOOFING | 0.9952 | 0.7271 | 0.8403 | 69,037 |
| NORMAL | 0.9955 | 0.9899 | 0.9927 | 4,754,568 |
| RPM_SPOOFING | 0.6285 | 0.9774 | 0.7651 | 82,652 |

Overall:

- Accuracy: **98.6064%**
- Macro F1: **91.8273%**
- Weighted F1: **98.6830%**

Binary attack-vs-normal detection:

- TPR: **90.1899%**
- FPR: **1.0107%**
- Precision: **80.2336%**

---

## 4.4 Verified Confusion Matrix

Labels:

`[DOS, FUZZY, GEAR_SPOOFING, NORMAL, RPM_SPOOFING]`

| Actual \ Predicted | DOS | FUZZY | GEAR | NORMAL | RPM |
|---|---:|---:|---:|---:|---:|
| DOS | 21,164 | 0 | 0 | 0 | 0 |
| FUZZY | 0 | 42,910 | 0 | 513 | 0 |
| GEAR_SPOOFING | 0 | 0 | 50,199 | 18,838 | **0** |
| NORMAL | 0 | 67 | 241 | 4,706,513 | 47,747 |
| RPM_SPOOFING | 0 | 0 | **0** | 1,866 | 80,786 |

### Targeted GEAR↔RPM crossover

The specific failure mode targeted by Phase 8.3 was eliminated:

- GEAR_SPOOFING → RPM_SPOOFING: **0**
- RPM_SPOOFING → GEAR_SPOOFING: **0**

This confirms that the CAN-ID-aware change directly addressed the
specific GEAR/RPM crossover observed in Phase 8.2.

This conclusion is based on the measured confusion matrix, not inferred
from aggregate F1 or accuracy.

---

## 4.5 Residual Errors

Phase 8.3 did not solve all detection errors.

### GEAR_SPOOFING recall gap

18,838 GEAR_SPOOFING samples were classified as NORMAL.

This represents approximately **27.3%** of the GEAR_SPOOFING test support.

The remaining errors therefore occur primarily between GEAR_SPOOFING
and the normal traffic associated with that signal.

### RPM_SPOOFING precision gap

47,747 genuinely NORMAL samples were classified as RPM_SPOOFING.

This is the largest individual error component in the Phase 8.3
confusion matrix.

A possible explanation is that normal RPM traffic may exhibit substantial
legitimate byte-level variability during vehicle operation. This is only
a hypothesis at this stage and has not been independently verified against
decoded physical RPM values.

---

# 5. Phase 8.2 vs Phase 8.3

| Metric | Phase 8.2 | Phase 8.3 | Change |
|---|---:|---:|---:|
| GEAR_SPOOFING F1 | 0.0012 | 0.8403 | +0.8391 |
| RPM_SPOOFING F1 | 0.1609 | 0.7651 | +0.6042 |
| Binary TPR | 0.6635 | 0.9019 | +0.2384 |
| Binary FPR | 0.0338 | 0.0101 | -0.0237 |
| Binary precision | 0.4717 | 0.8023 | +0.3306 |
| Overall accuracy | 0.9404 | 0.9861 | +0.0456 |

The principal scientific result is not simply the increase in aggregate
accuracy.

Phase 8.3 specifically tested the diagnosed Phase 8.2 failure mode and
eliminated the GEAR↔RPM crossover in both directions.

---

# 6. Dataset

## HCRL Car-Hacking Dataset

Source:

https://ocslab.hksecurity.net/Datasets/car-hacking-dataset/

Vehicle:

**Hyundai YF Sonata**

The dataset contains real vehicle OBD-II CAN captures with four attack
categories:

- DoS
- Fuzzy
- Gear spoofing
- RPM spoofing

The implemented Phase 8 pipeline contains **16,569,475 real CAN frames**.

Other NHTSA datasets investigated for the broader project, including CISS,
FARS, SCI, NiTS/NTS, and ADAS/ADS SGO data, are not part of the implemented
Phase 8 modeling and validation pipeline unless separately documented.

---

# 7. Experimental Integrity

Phase 8 is an independent real-cyber-telemetry baseline track.

The HCRL data is not silently combined with the project's existing
synthetic cyber-physical telemetry.

Phase 8 models are not used to:

- modify the frozen Transformer checkpoint
- retrain the frozen Transformer checkpoint
- overwrite the existing model
- alter the existing evidence chain
- claim validation of the existing cyber-physical prediction model

Any future fusion with vehicle-state features must be treated as a new,
separately evaluated experiment.

---

# 8. Evidence Boundary

"Real data" in Phase 8 means real CAN traffic and real injected attacks
within the HCRL dataset.

It does not establish:

- cross-vehicle generalization
- production fleet performance
- real-world safety consequence validation
- validation of the project's cyber-physical Transformer
- physical plausibility of decoded RPM or gear signals

The Phase 8 models are intrusion-detection baselines, not
cyber-physical state prediction models.

---

# 9. Limitations

1. HCRL represents a single vehicle platform and therefore does not
   establish cross-vehicle generalization.

2. The time-based split tests chronological generalization within the
   available source files rather than cross-platform generalization.

3. Phase 8.2 excludes CAN ID deliberately.

4. Phase 8.3 uses the top-30 CAN-ID representation plus `OTHER`, rather
   than individually encoding all 2,048 observed IDs.

5. Phase 8.3 still misses 18,838 GEAR_SPOOFING samples as NORMAL.

6. Phase 8.3 produces 47,747 NORMAL → RPM_SPOOFING false positives.

7. The explanation that normal RPM traffic variability causes the
   RPM precision gap is a plausible hypothesis, not a confirmed mechanism.

8. No DBC-based physical signal decoding has been verified for this exact
   HCRL capture.

9. Phase 8 results should not be generalized beyond the evaluated dataset
   and split without further validation.

---

# 10. Phase 8.4 — Deferred Future Work

Phase 8.4 residual-gap refinement was deliberately deferred on
2026-09-18.

Two candidate approaches have been identified.

## 10.1 Per-ID rolling statistical baseline

A low-cost approach would calculate a rolling statistical baseline for
each CAN ID, such as a z-score for `hamming_dist` or related features.

Purpose:

Allow the detector to distinguish an ID that naturally exhibits high
variability from one whose behavior is anomalous.

## 10.2 Physical signal decoding

A deeper approach would decode actual RPM and gear signal values using a
DBC file for the specific vehicle.

This could enable physical plausibility checks, such as detecting an
implausible RPM change over a short time interval.

However, a reliable DBC for the exact HCRL capture has not been verified.
Therefore, DBC availability and applicability remain unconfirmed.

---

# 11. Decision — Stop at Phase 8.3

On **2026-09-18**, the project decision was to stop the real-data baseline
track at Phase 8.3 for now.

Reason:

- Phase 8.2 diagnosed a specific failure mode.
- Phase 8.3 tested a targeted hypothesis.
- The targeted GEAR↔RPM crossover was measured and eliminated in both
  directions.
- The remaining errors have been explicitly identified rather than
  hidden inside aggregate metrics.
- Further optimization is not necessary to establish this evidence
  milestone.

Phase 8.4 remains future work, not abandoned work.

---

# 12. Reproducibility Artifacts

## Phase 8.1

- `scripts\real_cyber_telemetry\phase8_1_hcrl_car_hacking_ingest_and_gate.py`
- `experiments\real_cyber_telemetry\phase8_1_hcrl_ingest_gate.json`

## Phase 8.2

- `scripts\real_cyber_telemetry\phase8_2_hcrl_feature_baseline_model.py`
- `experiments\real_cyber_telemetry\phase8_2_baseline_model_report.json`
- `models\real_cyber_telemetry_baseline\rf_baseline_model.joblib`

## Phase 8.3

- `scripts\real_cyber_telemetry\phase8_3_hcrl_can_id_feature_model.py`
- `experiments\real_cyber_telemetry\phase8_3_canid_feature_model_report.json`
- `models\real_cyber_telemetry_baseline\rf_v2_canid_model.joblib`

---

# 13. Phase 8 Milestone

**PHASE 8 REAL-DATA BASELINE TRACK: COMPLETED THROUGH PHASE 8.3**

The project now has:

- A real labeled automotive CAN intrusion dataset
- A reproducible ingestion and release gate
- 16,569,475 real CAN frames
- An independent CAN-IDS baseline
- A CAN-ID-aware follow-up experiment
- A measured diagnosis of the Phase 8.2 failure mode
- A targeted Phase 8.3 improvement
- Zero GEAR↔RPM crossover errors in the Phase 8.3 confusion matrix
- Explicitly documented residual errors
- Explicit experimental boundaries
- Reproducibility artifacts
- Deferred future-work options

**Next research step:** do not begin Phase 8.4 automatically. First determine
whether feature-parity analysis with the existing 59-input vehicle-state
schema is required for the project's research question.
