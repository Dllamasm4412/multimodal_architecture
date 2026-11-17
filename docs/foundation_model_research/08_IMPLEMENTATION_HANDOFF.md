# 08 — Implementation Handoff: Milestone 1 (revised)

**Unified physiological data layer, input-policy-enforcing benchmark harness, and reference baselines**

This file is self-contained. You do not need the research conversation or the other reports. It supersedes the earlier `docs/05_IMPLEMENTATION_HANDOFF.md` and the first version of this file.

## 1. Purpose

The project will pretrain multimodal physiological encoders (ECG, PPG, respiration, EDA, EEG) without labels and test whether the learned representations transfer to new participants, datasets and tasks. Before any pretraining, four things must exist and be trusted:

1. one on-disk format that records, for every signal, where it came from, how it was timed and how well it is synchronised with the others;
2. an evaluation harness that scores **any function mapping signal windows to vectors**, under participant-disjoint protocols, and that **refuses inputs from which the target was constructed**;
3. reference results from conventional features, so later models have a bar to clear and each task's noise floor is known;
4. measured preprocessing and data-loading speed.

**In scope:** data conversion, synchronisation audit, splits, input policy, hand-crafted features, linear and gradient-boosted models, statistics, data-layer benchmark, reports.
**Out of scope:** neural networks, self-supervised training, pretrained checkpoints, GPU code. Do not add them.

## 2. Datasets and their roles

All three are open downloads, already on disk. Read their locations from environment variables; never download inside the code; never modify or commit data.

| Variable | Dataset | Role in this milestone | Documented content — verify in the audit |
|---|---|---|---|
| `WESAD_ROOT` | WESAD | stress classification, **chest device and wrist device as two separate tasks** | 15 participants. Chest device, 700 Hz: ECG, EDA, EMG, respiration, temperature, accelerometer. Wrist device: BVP 64 Hz, EDA 4 Hz, temperature 4 Hz, accelerometer 32 Hz. Condition labels. The two devices have independent clocks, aligned by the dataset authors |
| `DALIA_ROOT` | PPG-DaLiA | heart-rate regression from **wrist signals only** | 15 participants, same two devices. Reference heart rate = mean instantaneous rate from chest-ECG R-peaks in 8 s windows shifted by 2 s. Devices aligned by a double tap at start and end |
| `HMC_ROOT` | HMC sleep staging database | sleep staging from **ECG alone** and from **EEG alone** | 151 overnight recordings (EDF). Four EEG derivations, two EOG, chin EMG, one ECG lead. **No respiration and no PPG.** Stage per 30 s epoch |

HMC cannot test cardiorespiratory fusion and must not be described as doing so. The wrist BVP channel is a processed signal from the device, not raw PPG. If a documented property is false, stop and report it.

## 3. Repository layout

```
pyproject.toml
configs/default.yaml                 # every constant below lives here
src/physio/
  io/format.py                       # unified format: write, read, validate
  io/adapters/{wesad,dalia,hmc}.py   # dataset → unified format
  io/sync.py                         # synchronisation classes and empirical offset check
  io/synthetic.py                    # synthetic multimodal recordings for tests
  preprocess.py                      # band-limit, resample, validity and gap detection
  windows.py                         # window index with labels
  splits.py                          # participant-level splits, manifests, overlap registry
  policy.py                          # per-task input policy and its enforcement
  features/{cardiac,resp,eda,eeg,motion,coupling,scale}.py
  embed.py                           # Embedder protocol, HandcraftedEmbedder, RandomProjectionEmbedder
  eval/{probes,metrics,stats,label_efficiency,run}.py
  bench/data_throughput.py
tests/   reports/   results/
```

Dependencies: Python ≥ 3.10, `numpy`, `scipy`, `pandas`, `scikit-learn`, `pyyaml`, `pyedflib`, `pytest`. Nothing else. All randomness seeded from the config.

## 4. Unified recording format

**Layer 0, raw.** The original dataset files. Read-only. Referenced by relative path and SHA-256; never copied into the store, never altered.

**Layer 1, canonical.** One directory per recording, fully reproducible from layer 0 and the recorded configuration:

```
meta.json
<signal>.npy          float32 [channels, samples] at the canonical rate, native units, no normalisation
<signal>.valid.npy    uint8 validity mask at 1 Hz
labels.csv            start_s, end_s, task, value, source
```

`meta.json` must contain:

| Field | Content |
|---|---|
| identity | `dataset`, globally unique `participant_id` (`<dataset>:<id>`), `recording_id` |
| per signal | `modality`, channel names, sensor model and body site, **original sampling rate**, canonical rate, native units and whether they are physical (`units_comparable`), whether the signal is device-processed (for example wristband BVP) |
| timing | `t0`: absolute start time if the dataset gives one, otherwise recording-relative with the origin stated; `clock_group` (signals digitised by the same device share an id) |
| synchronisation, per clock group relative to a reference group | `method` (`same_device`, `single_recorder`, `event_marker`, `dataset_aligned`, `nominal_timestamps`), `class` (A–D, section 6), `estimated_offset_s`, `offset_uncertainty_s`, `estimated_drift_ppm`, how they were estimated, and `beat_level_ok` (true only for class A or B) |
| gaps | list of missing-data intervals with start, end and reason (absent in source, flat-line, saturation, sensor off) |
| provenance | raw file paths and hashes, adapter name and version, configuration hash, filter and resampling specification, creation date |

A validator checks shapes, rates, finite values, label intervals, that every signal has a clock group and a synchronisation record, and that hashes match.

## 5. Preprocessing

| Modality key | Band (zero-phase) | Canonical rate |
|---|---|---|
| `ecg` | 0.5–40 Hz | 128 Hz |
| `ppg` | 0.1–8 Hz | 64 Hz |
| `resp` | 0.05–1 Hz | 10 Hz |
| `eda` | low-pass 1 Hz | 4 Hz |
| `eeg` | 0.3–45 Hz | 128 Hz |
| `acc` (wrist) | low-pass 15 Hz | 32 Hz |

Anti-aliased polyphase resampling. **No amplitude normalisation is stored.** Normalisation happens per window inside embedders, and the quantities it removes are kept as scale descriptors (section 8). Beat detection for hand-crafted features runs on the layer-0 signal at its original rate, so timing precision is not limited by the canonical rate.

## 6. Synchronisation audit

Classes: **A** same device or converter; **B** single recorder with a possible constant latency in one channel; **C** separate devices aligned by an event marker or by the dataset authors; **D** nominal timestamps only or documented as unreliable.

1. Assign each clock-group pair a class from the dataset documentation (table in the config): within a WESAD or PPG-DaLiA device = A; chest versus wrist = C; HMC channels = B.
2. Empirical check where two cardiac signals exist in different clock groups (WESAD and PPG-DaLiA: chest ECG versus wrist BVP): cross-correlate their beat-interval series in consecutive 5-minute blocks; record the median offset, its spread across blocks, and the fitted drift.
3. Store the estimates in `meta.json`. They include pulse arrival time and therefore cannot be used to study it. An empirical check may lower a class; it never raises one above C.
4. Nominal timestamps alone never justify `beat_level_ok`.

Using the PPG-DaLiA chest ECG for this audit is permitted: a per-recording constant is metadata, not a model input.

## 7. Tasks, splits and input policy

| Task | Dataset | Window | Target (source) | Allowed inputs | Prohibited inputs | Evaluation | Primary metric |
|---|---|---|---|---|---|---|---|
| `stress_chest` | WESAD | 60 s, step 30 s, inside one condition segment | stress versus baseline + amusement (protocol) | chest `ecg`, `resp`, `eda` | accelerometer, temperature, any clock-time or position-in-session feature, all wrist signals | leave-one-participant-out | macro-F1, AUROC |
| `stress_wrist` | WESAD | same | same | wrist `ppg`, `eda` | accelerometer, temperature, time features, all chest signals | same | same |
| `hr_wrist` | PPG-DaLiA | 8 s, step 2 s | reference heart rate (**chest ECG**) | wrist `ppg`, wrist `acc` | **chest ECG and anything derived from it**; all other chest signals | leave-one-participant-out | mean absolute error |
| `sleep_ecg` | HMC | 30 s epochs | five stages (scored from EEG, EOG, EMG) | `ecg` | EEG, EOG, EMG, epoch index, clock time | 5-fold grouped by participant | Cohen's κ, macro-F1 |
| `sleep_eeg` | HMC | 30 s epochs | same | `eeg` (flagged: label-defining signal) | epoch index, clock time | same | same |

Declared variants, run and reported as separate rows: `stress_chest` without respiration (speaking during the stress condition disturbs breathing); `hr_wrist` with BVP only; `hr_wrist` plus wrist EDA and temperature.

**Enforcement (`policy.py`).** Each task declares `target_sources`, `allowed`, `prohibited` and its variants. The runner hands an embedder only allowed signals and raises an error if a prohibited one is requested. Every feature carries the list of signals it was computed from; a feature whose sources include a prohibited or target-source signal is rejected. There is no "all available modalities" mode.

**Splits.** By participant, before windowing; every window inherits its recording's partition; `assert_disjoint` on every fold. Manifests written to `results/splits/<task>.json` with a hash of the participant lists. `splits.py` also holds an overlap registry of known dataset derivations (for example "BIDMC is drawn from MIMIC-II") and refuses any listed pretraining/evaluation pair; it only needs to exist and be tested. Windows with more than 20 % invalid samples in a required signal are dropped and counted.

## 8. Embedders and reference models

`Embedder` protocol: `fit(train_windows)` (optional; training participants only) and `transform(windows) -> array [n_windows, dim]`. Later milestones plug neural encoders into this protocol unchanged.

* **`HandcraftedEmbedder`**, per allowed signal:
  * cardiac (`ecg` or `ppg`): mean heart rate, heart-rate slope, SDNN, RMSSD, fraction of implausible intervals; for `ppg` also pulse-amplitude mean and coefficient of variation, rise time;
  * `resp`: rate from the spectral peak; mean and coefficient of variation of breath amplitude and breath interval;
  * `eda`: tonic level, slope, number and mean prominence of skin-conductance responses;
  * `eeg`: log band-power per channel in 0.3–4, 4–8, 8–13, 13–30, 30–45 Hz;
  * `acc`: magnitude mean and variance, dominant movement frequency;
  * **scale descriptors** for every signal: log inter-quartile range and median in native units, `units_comparable` flag, fraction of invalid, flat or clipped samples, a simple quality index;
  * coupling, only between signals whose pair is class A or B and both allowed by the task: respiratory sinus arrhythmia amplitude and a cardiorespiratory phase-synchronisation index (in this milestone: WESAD chest ECG with respiration only). Pulse arrival time is implemented and unit-tested on synthetic data but used by no task here.
* **`RandomProjectionEmbedder`**: fixed random 1-D convolution features on per-window robustly z-scored signals, concatenated with the scale descriptors; a no-learning control. Run with and without the descriptors.
* For `hr_wrist`, two classical references: spectral-peak heart-rate tracking on BVP; the same with accelerometer-informed suppression of movement peaks.

Probes, each as `Pipeline(StandardScaler → model)` so scaling is fitted on training folds only: ridge or logistic regression; `HistGradientBoosting`. Hyper-parameters by inner grouped cross-validation over training participants. For the sleep tasks, one fixed post-processing step for all models: median smoothing over five epochs.

## 9. Statistics

* Metrics per test participant, then mean with participant-level bootstrap 95 % interval (10,000 resamples).
* Model comparison: paired per-participant differences, sign-flip permutation test.
* Permutation null (labels shuffled within participant, whole pipeline re-run ≥ 200 times) for both stress tasks.
* Minimum detectable effect per task at 80 % power, α = 0.05.
* Label efficiency: sub-sample **training participants** (2, 5, 10, all), five nested random subsets each, fixed test participants.
* Save every window-level prediction with participant, fold, task, variant, embedder and probe identifiers.

## 10. Data-layer benchmark

`bench/data_throughput.py` reports: preprocessing speed (recording-hours per CPU-hour) per dataset; stored bytes per recording-hour per modality; random-access window loading from the canonical store in windows per second for 0, 2, 4 and 8 workers, for 30 s windows with (a) ECG + PPG + respiration, (b) the same plus 4 EEG channels. Results go to `reports/data_benchmark.md` with the hardware and disk type stated. No model is involved.

## 11. Tests that must pass (synthetic recordings; no real data)

1. **Format:** write → read → validate; hashes detect a changed raw file; a 10 Hz sinusoid keeps its frequency after resampling; gaps and clock groups survive the round trip.
2. **Signal processing:** a synthetic 72 beats-per-minute pulse train gives heart rate 72 ± 1; with an imposed 250 ms ECG-to-PPG delay, pulse arrival time 250 ± 15 ms; a 0.1–0.5 Hz modulation of a synthetic pulse wave survives the PPG band.
3. **Input policy:** requesting ECG for `hr_wrist` raises; a feature derived from ECG is rejected for `hr_wrist`; accelerometer is rejected for both stress tasks; EEG is rejected for `sleep_ecg`.
4. **Synchronisation:** a synthetic two-device recording with a 0.8 s offset and 40 ppm drift is recovered to within 0.1 s and 15 ppm; coupling features are not computed across a class C or D pair; `beat_level_ok` is false for it.
5. **Planted effect:** labels that shift synthetic heart rate are classified above the permutation null; with no effect, intervals cover chance.
6. **Leakage:** a random-window split is rejected; a participant-ID one-hot does not improve participant-disjoint results; test-fold values do not change fitted scaler parameters; a listed pretraining/evaluation pair raises.
7. **Scale descriptors:** a ten-fold amplitude change alters the descriptor and leaves the normalised waveform unchanged.
8. **Determinism:** identical configuration gives identical predictions.

## 12. Definition of done

* All tests pass; each dataset converts with one command and passes the validator; the benchmark suite runs with one command.
* `reports/audit.md`: per dataset, participants, hours per modality, invalid fraction, gaps, label distribution per participant, synchronisation class and empirical offset results, and every deviation from section 2.
* `reports/baselines.md`: one table per task and variant (rows: embedder × probe × input set, plus classical trackers; columns: metrics with intervals), permutation-null summary, minimum detectable effect, label-efficiency curves, and the effect of including scale descriptors.
* `reports/data_benchmark.md`.
* No data, credentials or absolute user paths in the repository.

## 13. Rules

* Nothing is fitted on test participants.
* The target source of a task is never a model input, directly or through derived features.
* Raw data stay unchanged.
* One label per window, from the dataset's own annotation.
* Report results as they are; a task where conventional features do not beat the null is a finding.
* No additional datasets, models or dependencies.

## 14. What comes next (not part of this milestone)

* **Milestone 1b, resource benchmark:** a PyTorch skeleton of the planned model (student, teacher, prediction head, optimiser) run on synthetic tensors to measure throughput and peak memory for 0–32 EEG channels, before any pretraining budget is fixed.
* **Milestone 2, synthetic coupling benchmark:** simulated signals with known cross-modal lags, the fusion variants including an explicit pairwise lag module, a delay-resolution test, and the pairing-perturbation controls, all evaluated through the same `Embedder` protocol and statistics.
