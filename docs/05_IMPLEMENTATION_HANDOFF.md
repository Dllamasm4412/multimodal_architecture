# 05 — Implementation Handoff: Milestone 1

**DEAP audit, leakage-proof evaluation harness, control features, classical baselines**

This file is self-contained. You do not need the research conversation or the other reports to build this milestone.

## 1. Why this milestone exists

The project asks whether topological (persistent-homology) features and modality-aware fusion improve emotional-arousal estimation from physiological signals for **participants never seen in training**. Published DEAP results are frequently inflated by evaluation leakage, and realistic participant-independent performance is close to chance. Nothing later in the project can be interpreted until we know, on our own harness:

1. the chance level and the noise floor (how large a difference between two models is detectable with 32 participants);
2. what simple spectral and peripheral features already achieve;
3. how much of the apparent signal comes from stimulus identity or from non-neural channels.

**In scope:** data audit, splits, features, classical models, statistics, reports.
**Out of scope:** neural networks, persistent homology, STFT/wavelet encoders, fusion models. Do not add them.

## 2. Data

DEAP preprocessed Python release (`data_preprocessed_python/s01.dat … s32.dat`). The data must already be on disk; read its location from the environment variable `DEAP_ROOT`. Never download, copy into the repository, or commit DEAP data.

Documented layout — **verify each item in the audit, do not assume it**:

| Item | Documented value |
|---|---|
| File format | Python 2 pickle; load with `pickle.load(f, encoding="latin1")` |
| `data` | `(40 trials, 40 channels, 8064 samples)`, 128 Hz, 63 s per trial |
| `labels` | `(40, 4)` = valence, arousal, dominance, liking; continuous 1–9 |
| Pre-trial baseline | first 3 s (384 samples); stimulus period is the remaining 60 s (7680 samples) |
| Channels 0–31 | EEG (already 4–45 Hz band-passed, average-referenced, EOG-cleaned) |
| Channels 32–39 | hEOG, vEOG, zygomaticus EMG, trapezius EMG, GSR, respiration belt, plethysmograph (BVP), temperature |
| Trial order | reordered to video (`Experiment_id`) order, so trial index *i* is the same video for every participant |
| Recording sites | participants 1–22 and 23–32 were recorded at different sites; GSR format/units may differ |

There is no ECG. Cardiac information comes only from the plethysmograph.

## 3. Repository layout

```
pyproject.toml
configs/default.yaml          # every tunable constant lives here
src/pf/
  data/deap.py                # loader, channel map, audit
  data/synthetic.py           # DEAP-shaped synthetic fixture
  splits.py                   # grouped splits, manifest, leakage assertions
  features/eeg.py             # spectral features
  features/peripheral.py      # BVP, EDA, respiration, temperature, EMG, EOG
  features/nonlinear.py       # amplitude and entropy control features
  models/classical.py         # sklearn pipelines
  eval/metrics.py  eval/stats.py  eval/run_cv.py
tests/
reports/                      # generated Markdown reports
results/                      # generated per-trial predictions (parquet/csv), split manifests
```

Dependencies: Python ≥ 3.10, `numpy`, `scipy`, `scikit-learn`, `pandas`, `pyyaml`, `pytest`. Nothing else. All randomness is seeded from the config.

## 4. Components

### 4.1 Audit (`reports/stage0_audit.md`)

Report, per participant and overall: array shapes and dtypes; NaN/Inf counts; flat-lined or saturated channels; per-channel amplitude range, split by recording site (1–22 vs 23–32); rating histograms; fraction of trials with arousal > 5 and valence > 5; number of participants with fewer than 5 trials in either class; correlation between a participant's ratings and the across-participant mean rating per video. Confirm the trial-index ↔ video mapping against `participant_ratings.csv` if that file is present, and state clearly if it could not be confirmed.

### 4.2 Synthetic fixture

`make_synthetic_deap(n_participants, effect_size, seed)` writes files with the exact DEAP layout. Signals are coloured noise plus a pulse-like BVP and a slow EDA trace. A latent per-trial arousal variable drives (a) EEG alpha power in a subset of channels, (b) heart rate and (c) the rating, with strength `effect_size`; each participant gets a random rating offset and gain. All tests run on this fixture, so the test suite needs no real data.

### 4.3 Splits

* Outer: leave-one-participant-out (32 folds).
* Also write a fixed, seeded **8-fold participant-grouped** manifest (4 test participants per fold) to `results/splits_v1.json`. Later neural experiments will reuse it unchanged.
* Inner model selection: `GroupKFold(5)` over training participants only.
* Secondary design: crossed hold-out, 8 participant folds × 5 video folds; train on (train participants × train videos), test on (test participants × test videos).
* `assert_no_leakage(train_idx, test_idx, meta)` is called on every fold: participant sets disjoint (and video sets disjoint in the crossed design). Splitting always happens at trial level, before any windowing.

### 4.4 Targets

* `y_raw`: arousal rating (primary). Valence is run identically as a secondary target.
* `y_bin = y_raw > 5` (fixed in advance; never re-thresholded).
* Regression training target: rating z-scored **within each training participant** using only that participant's own ratings. Test-participant labels are never transformed or used.

### 4.5 Features (stimulus period only; computed per trial)

| Group | Features |
|---|---|
| `eeg_spec` | Welch PSD (2 s Hann, 50 % overlap) per channel → log band-power in θ 4–8, α 8–13, β 13–30, γ 30–45 Hz (32 × 4 = 128) |
| `bvp` | band-pass 0.5–8 Hz; peaks by `scipy.signal.find_peaks`; mean heart rate, heart-rate slope, SDNN, RMSSD of inter-beat intervals, mean and coefficient of variation of pulse amplitude, fraction of physiologically implausible intervals (quality flag) |
| `eda` | low-pass 1 Hz, resample to 4 Hz; mean, slope, standard deviation; number and mean prominence of skin-conductance responses |
| `resp` | band-pass 0.05–1 Hz; breathing rate (spectral peak), mean and coefficient of variation of breath amplitude and of breath interval |
| `temp` | mean, slope |
| `confound` | zygomaticus and trapezius EMG log-RMS; hEOG/vEOG variance and blink-rate proxy |
| `ctrl_nonlinear` | per EEG channel and per peripheral signal: variance, peak-to-peak, line length, kurtosis, Hjorth mobility and complexity, zero-crossing rate, spectral entropy, permutation entropy (order 3). EEG values are averaged within five scalp regions (frontal, central, parietal, occipital, temporal) |

Also provide every feature in a **baseline-relative** variant (stimulus value minus pre-trial 3 s value where the feature is defined on 3 s; otherwise omit). Feature extraction must be a pure function of one trial's samples — no statistics from other trials.

### 4.6 Models

sklearn `Pipeline(StandardScaler → estimator)`, so scaling is fitted on training folds only:

* ridge regression (α grid) and L2 logistic regression with `class_weight="balanced"` (C grid);
* `HistGradientBoostingRegressor` with a small fixed grid;
* hyper-parameters chosen by the inner grouped CV only.

Feature sets to run: `eeg_spec`; peripheral (`bvp+eda+resp+temp`); EEG + peripheral; `confound` alone; each of the above with `ctrl_nonlinear` added.

### 4.7 Reference baselines (all mandatory)

| ID | Baseline | Purpose |
|---|---|---|
| R0 | constant / majority class | exposes class imbalance |
| R1 | permutation null: shuffle labels **within participant**, re-run the complete pipeline ≥ 200 times (ridge, EEG + peripheral) | empirical chance distribution |
| R2 | stimulus prior: predict each video's mean rating over training participants (uses video identity, not physiology) | how much is explainable by stimulus identity alone |
| R3 | `confound` features only | how much is explainable by eye and muscle activity |
| R4 | per-participant unsupervised feature z-scoring (uses the test participant's unlabeled trials) | quantifies the benefit of label-free calibration; always reported separately and labelled "transductive" |

### 4.8 Metrics and statistics

Computed **per held-out participant** from trial-level predictions, then averaged over participants:

* primary: Spearman correlation between prediction and `y_raw`;
* secondary: ROC-AUC against `y_bin` (skip and count participants with < 3 trials in a class); balanced accuracy and macro-F1 (decision threshold fixed from training data); concordance correlation and MAE on the raw scale.

Uncertainty: participant-level bootstrap 95 % interval (10 000 resamples). Model comparison: paired per-participant differences, sign-flip permutation test. From the observed standard deviation of paired differences, report the **minimum detectable effect** at 80 % power, α = 0.05, n = 32. Save every trial-level prediction with participant, video, fold, model and feature-set identifiers.

## 5. Tests that must pass

1. **Planted signal:** fixture with `effect_size > 0` → primary metric significantly above the R1 null; with `effect_size = 0` → interval covers 0 (Spearman) and 0.5 (AUC).
2. **Leakage canary:** adding a participant-ID one-hot to the features must not raise leave-one-participant-out performance; a deliberately leaky random-trial split must be rejected by `assert_no_leakage`.
3. **Scaler isolation:** changing test-fold feature values does not change fitted scaler parameters.
4. **Determinism:** two runs with the same config produce identical predictions.
5. **Feature sanity:** a 10 Hz sinusoid yields maximal α band-power; a synthetic 72 bpm pulse train yields heart rate 72 ± 1.

## 6. Definition of done

* All tests pass on the synthetic fixture; the full pipeline runs end-to-end on real DEAP from one command.
* `reports/stage0_audit.md` and `reports/stage1_baselines.md` exist. The second contains one table (rows = model × feature set and R0–R4; columns = metrics with intervals), the permutation-null histogram summary, the minimum detectable effect, and a list of every deviation from this specification.
* No DEAP data, credentials or absolute user paths in the repository.

## 7. Rules

* Never fit anything (scaler, selector, threshold, hyper-parameter) on test participants, except in R4, which is labelled as such.
* Never create window-level labels. One trial has one label.
* Report results as they are. Near-chance performance is an expected and acceptable outcome of this milestone; do not tune to escape it.
* If a documented DEAP property in Section 2 turns out to be false, stop and report it instead of working around it.

## 8. What comes next (not part of this milestone)

Milestone 2 is a small, independently tested topology module (lower-star H0 on waveforms; delay-embedding Vietoris–Rips H0/H1), validated on synthetic signals and then scored against the `ctrl_nonlinear` and spectral features produced here. It has no dependency on DEAP access and may be started in parallel.
