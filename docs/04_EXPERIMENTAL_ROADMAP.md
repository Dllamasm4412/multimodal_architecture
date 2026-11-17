# 04 — Experimental Roadmap

Staged plan with controls, ablations and go/no-go gates. No calendar is given: durations depend on data access and compute that have not been measured. Model names (A0, A1, B-gated, B-xattn, C, C-Q) are defined in `03_ARCHITECTURE_RECOMMENDATION.md`; falsification experiments F0–F6 in `02_CRITICAL_ASSESSMENT.md`.

## 1. Changes to the proposed stage order

| Proposed | Revised | Reason |
|---|---|---|
| 0 Audit → 1 Synthetic topology → 2 Physiological topology → 3 Baselines → 4 Encoders → 5 Fusion → 6 Topology integration | **0 Audit and harness → 1 Classical baselines and noise floor → 2 Synthetic topology (parallel with 0–1) → 3 Physiological topology: the decisive test → 4 Neural encoders → 5 Fusion and robustness → 6 Conditional topology integration → 7 Confirmatory run** | The topology feasibility stage needs the evaluation harness, the control features and the measured noise floor, so baselines must precede it. The decisive topology test is then run on low-variance classical models *before* any neural architecture exists, instead of last. Synthetic validation has no dependency on DEAP and runs in parallel. A final frozen confirmatory stage is added. |

## 2. Protocol that applies to every stage

**Independent unit.** The participant. All inference is over the 32 participants; windows and trials are never treated as independent samples.

**Splits.**
* Classical models: nested leave-one-participant-out; inner 5-fold grouped cross-validation for all tuning.
* Neural models: a fixed, seeded 8-fold participant-grouped split (per fold: 24 train, 4 validation for early stopping and selection, 4 test), each participant tested once, 5 seeds.
* Secondary design: crossed hold-out of participants *and* videos (8 × 5), because all participants watched the same 40 videos and a model may decode the stimulus rather than the response.
* Splitting precedes windowing. Every window, surrogate and augmented copy of a trial inherits that trial's partition.

**Fitting.** Scalers, feature selection, dimensionality reduction, thresholds, embedding parameters and hyper-parameters are fitted on training participants only. The one exception is the *calibrated* regime (per-participant standardisation from that participant's unlabeled trials), which is always reported in a separate column and labelled transductive.

**Target and metrics.** Trial-level self-reported arousal (valence secondary).
* Primary: Spearman correlation between predictions and ratings **within each held-out participant**, averaged over participants. It uses the full rating scale, is defined for every participant, and is unaffected by individual differences in scale use, which no physiological model can know for an unseen person.
* Secondary, for comparability with the literature: balanced accuracy, macro-F1 and ROC-AUC for the fixed split rating > 5; concordance correlation and mean absolute error on the raw scale.
* Always shown beside: constant predictor, permutation null, stimulus-identity prior, eye/muscle-channel model.

**Uncertainty and testing.** Participant-level bootstrap 95 % intervals. Model comparisons use per-participant paired differences (averaged over seeds first) with a sign-flip permutation test. Seed-to-seed spread is reported separately from participant-to-participant spread. Cross-validation folds share training data, so these intervals are approximate and somewhat optimistic; this is stated in every report.

**Minimum detectable effect (MDE).** With n = 32 paired observations, 80 % power and two-sided α = 0.05, the detectable mean difference is about 0.5 × the standard deviation of per-participant paired differences. Stage 1 measures that standard deviation; "the MDE" below means the resulting number. Differences smaller than the MDE are reported as unresolved, not as wins or ties.

**Confirmatory comparisons (fixed now, Holm-corrected).** Everything else is exploratory and labelled so.

| ID | Comparison | Stage |
|---|---|---|
| K1 | Topology increment: control features + topology vs control features + placebo features | 3 |
| K2 | Modality-specific vs homogeneous encoders | 4 |
| K3 | Multimodal B-gated vs best unimodal model | 5 |
| K4 | Robustness under corruption: topological quality gating vs conventional quality-index gating | 6 |

**Reproducibility.** Config-driven runs, fixed seeds, cached features keyed by dataset version and parameters, saved trial-level predictions, a log of every deviation from this document.

## 3. Stages

### Stage 0 — Data and evaluation audit

* **Hypothesis.** None; this stage establishes facts. Working assumption to check: the DEAP release matches its documentation (40 × 40 × 8064 arrays at 128 Hz, 3 s pre-trial segment, channel order, trials in video order).
* **Components.** Loader and channel map; audit of shapes, missing values, flat or saturated channels, amplitude ranges by recording site (participants 1–22 vs 23–32, especially GSR); rating distributions and class balance per participant; split manifests; leakage assertions; DEAP-shaped synthetic fixture.
* **Evaluation.** Automated tests on the fixture: planted effect recovered, zero effect gives null results, a participant-ID feature gives no gain, a leaky split is rejected.
* **Deliverables.** Data dictionary, audit report, split manifest, tested harness.
* **Success criteria.** All tests pass; every documented property confirmed or its discrepancy recorded; number of participants with fewer than 5 trials in either class known.
* **Go/no-go.** Go when the harness passes its tests. If the trial-to-video mapping cannot be confirmed, the crossed design is dropped and that limitation is stated.

### Stage 1 — Classical baselines and noise floor

* **Hypothesis (F0).** Simple spectral and peripheral features predict within-participant arousal ordering in unseen participants better than chance.
* **Components.** Control feature library (EEG log band-power; BVP, EDA, respiration and temperature features; amplitude, Hjorth and entropy features); ridge, logistic and gradient-boosted models; reference baselines (constant, permutation null with ≥ 200 within-participant label shuffles, stimulus prior, eye/muscle channels, calibrated regime).
* **Evaluation.** Nested leave-one-participant-out; EEG-only, peripheral-only, combined; crossed participant × video design.
* **Deliverables.** Baseline table with intervals; permutation-null distribution; measured MDE; per-modality contribution; estimate of how much is attributable to stimulus identity and to eye/muscle activity.
* **Success criteria.** Best baseline exceeds the permutation null at p < 0.01 with the lower interval bound above zero (Spearman).
* **Go/no-go.**
  * *Pass* → continue as planned.
  * *Fail in the strict regime but pass in the calibrated regime* → continue with the calibrated regime as the declared primary setting, described as "label-free calibration required".
  * *Fail in both* → DEAP participant-independent arousal cannot support incremental claims. Options, in order: (a) run the same pipeline on a second affect dataset with EEG and peripheral signals (AMIGOS is a candidate) and continue there; (b) restrict later stages to robustness and signal-quality questions that do not depend on arousal labels; (c) use a within-participant, trial-blocked protocol and state explicitly that the results say nothing about new users.

### Stage 2 — Topology primitives on synthetic signals (parallel with Stages 0–1)

* **Hypothesis.** The implementations are correct and their behaviour under controlled changes matches theory.
* **Components.** Lower-star H0; delay embedding with max-min subsampling; Vietoris–Rips H0/H1 via an existing library; the scalar descriptors of report 03; IAAFT and phase-randomised surrogate generators.
* **Evaluation: known-answer and stress tests.**

| Signal | Expected behaviour |
|---|---|
| Pure sinusoid, any frequency or phase | lower-star pairs coincide, inertia ≈ 0; one dominant H1 class |
| Amplitude-modulated sinusoid | inertia grows with modulation depth; H1 persistence falls as the envelope minimum falls |
| Two incommensurate tones | two prominent H1 classes |
| White and 1/f noise | no dominant H1 class |
| Same signal resampled or time-warped | lower-star diagram unchanged |
| Signal scaled by *a* | persistences scale by *a*, inertia by *a*² |
| Added noise at graded levels | bottleneck distance bounded by noise amplitude; unthresholded counts and entropy degrade |
| Synthetic pulse wave with beat-to-beat variability and artefact bursts | loop thickness tracks variability; loop dominance drops under bursts |
| Rössler or Lorenz series vs their IAAFT surrogates (positive control) | descriptors differ from surrogates |
| Autoregressive Gaussian series vs its surrogates (negative control) | descriptors do not differ beyond the nominal false-positive rate |

* **Deliverables.** Tested module; sensitivity report over embedding dimension, delay, window length, point budget and normalisation; runtime per diagram versus point count.
* **Success criteria.** All known-answer tests pass; both surrogate controls behave as expected; subsampling error characterised.
* **Go/no-go.** Go when both controls pass. Without a working positive and negative control, Stage 3's surrogate test is uninterpretable.

### Stage 3 — Physiological topology: stability, redundancy, surrogates, incremental value

This stage decides whether topology proceeds as a predictor of arousal, as a quality descriptor, both, or neither. It uses classical models only.

* **Hypotheses.** (F1) Descriptors are not reconstructible from spectral, amplitude and entropy features. (F2) Their predictive content depends on structure that surrogates destroy. (F3) They add predictive value beyond controls and beyond a placebo. (F4) They are not predominantly participant fingerprints. (F6-pre) The loop-dominance score detects corrupted BVP and respiration windows.
* **Components.** Descriptors for BVP, respiration, EDA (lower-star only) and EEG (lower-star; delay-embedding and connectivity variants exploratory), each in amplitude-preserving and shape-only form; the Stage 1 harness.
* **Evaluation.**

| Analysis | Procedure | Outcome measure |
|---|---|---|
| 3a Reliability | Agreement between adjacent windows of a trial; variation across the embedding-parameter grid | intraclass correlation; rank stability across settings |
| 3b Redundancy (F1) | Predict each descriptor from the control features with gradient boosting, participant-grouped cross-validation | cross-validated R² |
| 3c Surrogates (F2) | (i) Rank test of each descriptor against 19 IAAFT surrogates on a stratified sample of windows; (ii) replace descriptors by surrogate-derived ones for the whole dataset (5 realisations) and re-run prediction | (i) rejection rate vs nominal 5 %; (ii) paired performance difference, real minus surrogate |
| 3d Increment (F3, K1) | Control vs control + topology vs control + equally many placebo features (drawn from a generic time-series feature set) | paired difference in the primary metric |
| 3e Identity (F4) | Classify participant from each feature family; between- to within-participant variance ratio | identification accuracy relative to control features |
| 3f Quality (F6-pre) | Inject graded artefacts into clean BVP and respiration windows; detect corrupted windows with the loop-dominance score vs autocorrelation peak, spectral purity, template correlation | detection AUC |

  Interpretation note for 3c: IAAFT surrogates also remove non-stationarity, so "real better than surrogate" means "more than a stationary spectrum plus amplitude distribution", not specifically "nonlinear dynamics".
* **Deliverables.** One feasibility report per modality; a list of descriptors retained, with reasons.
* **Success criteria and gates.**
  * **Prediction gate.** Topology proceeds to Stage 6a only if all hold: (1) some retained descriptors have cross-validated R² < 0.9 against controls; (2) the K1 increment has a 95 % interval excluding zero and is at least the MDE; (3) the increment is larger with real than with surrogate-derived descriptors. If (1)–(2) hold but (3) fails, topology is a convenient re-encoding of spectral information; it may be kept as an engineering feature but the "non-spectral information" claim is dropped.
  * **Quality gate.** Topology proceeds to Stage 6b only if the loop-dominance score is non-inferior to the best conventional index (AUC within 0.02) and improves detection when combined with it.
  * **Neither gate passed** → topology is closed for this project's emotion-recognition aim and written up as a controlled negative result. Stages 4, 5 and 7 continue without it.

### Stage 4 — Compact neural models and modality-specific encoders

* **Hypotheses.** (K2) Encoders matched to each modality's timescale and structure outperform one homogeneous encoder applied to all modalities. The EEG front end matters less than the evaluation regime.
* **Components.** A1; unimodal encoders of B; homogeneous control (the same 1-D convolutional encoder on every modality resampled to one rate, matched in parameters); EEG front-end variants at matched size: log band-power, STFT log-spectrogram, Morlet scalogram, learnable filter bank on raw EEG, input-conditioned band weighting; EEGNet as an external anchor.
* **Evaluation.** Fixed 8-fold grouped split, 5 seeds, unimodal and late-averaged predictions.
* **Deliverables.** Unimodal table per modality and front end, with parameter counts and training cost.
* **Success criteria.** Each neural unimodal model is non-inferior to its classical counterpart (difference smaller than the MDE or in its favour).
* **Go/no-go.** For each modality keep the simplest variant within one MDE of the best. Where a learned encoder is inferior to hand-crafted features, that modality uses feature tokens (the A1 stream). Adaptive spectral weighting is retained only if it exceeds fixed band-power by at least the MDE; otherwise it is dropped and reported.

### Stage 5 — Multimodal fusion and robustness

* **Hypotheses.** (K3) Fusion exceeds the best unimodal model. (F5) Cross-attention adds nothing over gated fusion once capacity is matched. Auxiliary unimodal losses and modality dropout improve performance with missing modalities at little cost on complete data.
* **Components.** Concatenation; averaging of unimodal predictions; B-gated; B-xattn and its capacity-matched control; with and without auxiliary unimodal losses; with and without modality dropout; alignment loss as an exploratory arm.
* **Evaluation.**
  * Clean data: primary metric.
  * Missing modalities: all 15 non-empty subsets of {EEG, BVP, EDA, respiration} at test time.
  * Corruption, applied to test participants at graded severity: additive noise (20, 10, 0, −5 dB); motion-like bursts covering 10, 30, 50 % of the trial; flat-line segments; clipping; baseline drift (EDA, respiration); loss of 4, 8, 16 EEG channels. Summarised as the area under the performance-versus-severity curve.
  * A separate arm trains with corruption as augmentation and is labelled as such.
* **Deliverables.** Fusion comparison table; missing-modality matrix; degradation curves; gate-weight diagnostics.
* **Success criteria.** K3 interval excludes zero. Robustness results are reported only for models whose clean performance is above the permutation null; a model at chance cannot degrade.
* **Go/no-go.** The fusion mechanism going forward is the simplest one within one MDE of the best on clean data and on the robustness summary. Cross-attention is adopted only if it beats its capacity-matched control by at least the MDE on one of them.

### Stage 6 — Topology integration (conditional)

* **6a Predictive (only if the Stage 3 prediction gate passed).**
  * *Hypothesis.* Descriptors injected at the modality token improve B beyond a placebo of equal size.
  * *Components.* C; B + placebo adapter; C with surrogate-derived descriptors; per-modality variants (BVP only, respiration only, EDA only, EEG only, all).
  * *Success criterion.* C exceeds both B + placebo and the surrogate variant by at least the MDE, in the same direction as in Stage 3.
* **6b Quality gating (only if the Stage 3 quality gate passed).**
  * *Hypothesis (K4, F6).* Supplying a topological quality score to the fusion gate improves robustness more than conventional indices.
  * *Components.* Gate input: none; conventional index; topological score; both. Evaluated on the Stage 5 corruption suite. Because DEAP has no quality annotations, one dataset with real motion artefacts and a reference signal should be added before any general claim (PPG-DaLiA is a candidate; its contents must be checked first).
  * *Success criterion.* Larger area under the degradation curve than the conventional-index gate, by at least the MDE, without loss on clean data; and no systematic down-weighting of physiologically unusual but clean windows.
* **Deliverables.** Incremental-value report; where relevant, robustness report.
* **Go/no-go.** Topology is part of the final model only where its criterion is met. Otherwise the final model is the Stage 5 model and the topology results are reported as they are.

### Stage 7 — Frozen confirmatory evaluation

* **Hypothesis.** The effects selected during development replicate when nothing further can be tuned.
* **Components.** Final configuration frozen and recorded; confirmatory comparisons K1–K4 re-run with leave-one-participant-out and at least 5 seeds; crossed participant × video design; valence as secondary target. If feasible, the frozen pipeline is applied unchanged to a second dataset.
* **Evaluation.** As in section 2, Holm-corrected across K1–K4.
* **Deliverables.** Final report with every stage's outcome, including failed gates; code, configurations, split manifests and predictions.
* **Success criteria.** Each claim made in the write-up corresponds to a confirmatory comparison whose corrected interval excludes zero. Anything else is described as exploratory.

## 4. Claim boundaries

| Claim type | Supported by | Not supported by |
|---|---|---|
| Emotion (arousal) estimation for new users | Stages 1, 4, 5, 6a, 7 under participant-disjoint evaluation | any within-participant or window-shuffled result |
| Sensor-quality assessment | Stage 3f and 6b, synthetic corruption; real-artefact dataset for generality | arousal labels |
| Physiological event detection | not addressed; needs event-annotated data (seizure, arrhythmia, apnoea) | DEAP trial ratings, which are not event annotations |

## 5. Risks carried through the plan

| Risk | Where it is caught |
|---|---|
| Baselines at chance | Stage 1 gate and its contingencies |
| Stimulus decoding mistaken for emotion recognition | crossed design in Stages 1 and 7; stimulus-prior baseline |
| Eye and muscle activity mistaken for neural signal | eye/muscle-channel baseline |
| Topology re-encodes amplitude or spectrum | Stage 3b, 3c |
| "More features" mistaken for "topological information" | placebo arms in Stages 3 and 6 |
| Features encode identity | Stage 3e |
| Capacity mistaken for mechanism | parameter-matched controls in Stages 4–6 |
| Selective reporting | four pre-declared confirmatory comparisons; Stage 7 freeze |
