# 06 — Transfer-Learning Evaluation Framework

How to decide whether a pretrained multimodal physiological encoder has learned something reusable, and whether cross-modal learning is the reason. Model and objective names (Architectures A, B, C; objectives O1, O2; fusion variants) are defined in `04_ARCHITECTURE_OPTIONS.md` and `05_SELF_SUPERVISED_LEARNING_STRATEGY.md`. Dataset facts and their verification status are in `03_DATASET_AND_PRETRAINING_STRATEGY.md`. The statistical conventions extend those of the earlier `docs/04_EXPERIMENTAL_ROADMAP.md`.

## 1. What is being evaluated

The object of study is the **frozen representation**, not a tuned end-to-end system. Fine-tuned accuracy on one task can come from the downstream head, from capacity, or from overlap between pretraining and test data. The framework is built to separate those sources.

Four questions, in order of importance:

1. Does pretraining help at all, relative to the same network trained from scratch and to conventional features, especially with few labelled participants?
2. Does *cross-modal* pretraining help beyond unimodal pretraining of the same encoders?
3. Does *interaction-based fusion* help beyond late fusion at equal capacity?
4. Do the answers hold across datasets, devices and tasks, or only where pretraining and evaluation data resemble each other?

## 2. Three evaluation settings, never mixed

| Setting | What the model may use from evaluation participants | Reported as |
|---|---|---|
| **Inductive** (primary) | Nothing. Normalisation constants, probes and thresholds come from training participants only. Per-window self-normalisation is allowed because it uses only the window itself | "new user, no calibration" |
| **Transductive** | Unlabelled recordings of the evaluation participants: per-participant standardisation of embeddings, Riemannian recentering, test-time adaptation, continued self-supervised training | "label-free calibration" |
| **Personalised** | A stated number of labelled calibration windows per evaluation participant, disjoint in time from that participant's test windows | "k-shot calibration" |

Every results table has one setting per column group. A method that needs unlabelled test data is never compared with an inductive number without that being stated in the same row.

## 3. Downstream task suite

Tasks are chosen to span state versus trait, sleep versus wake, clinical versus wearable sensors, and classification, regression and event detection. Arousal is one task among several.

| ID | Task | Candidate dataset(s) | Modalities exercised | Type | Primary metric (per participant, then averaged) |
|---|---|---|---|---|---|
| T1 | Sleep staging, 5 classes | held-out participants of the pretraining cohort; a second, unseen sleep dataset | ECG or PPG + respiration; EEG variant | state, sequence | Cohen's κ, macro-F1 |
| T2 | Stress versus baseline | WESAD | chest ECG + respiration + EDA; wrist BVP + EDA | state, wake, wearable | macro-F1, AUROC |
| T3 | Continuous emotional arousal | CASE (continuous ratings); DEAP and AMIGOS (trial ratings, with EEG) | ECG/BVP + EDA + respiration; EEG | state, regression | within-participant Spearman, concordance correlation |
| T4 | Rhythm classification from one lead | PTB-XL, one lead, using its patient-level folds. (PhysioNet/CinC 2017 provides no documented patient identifiers, so participant-disjoint splits cannot be guaranteed there; it is exploratory only) | ECG only | clinical, unimodal | macro-AUROC per fold |
| T5 | Physiological regression probes: heart rate under motion; respiratory rate | PPG-DaLiA; BIDMC | wrist PPG and wrist accelerometer; PPG. The reference signals (chest ECG; impedance respiration) are target-only | content probe | mean absolute error |
| T6 | Event detection: sleep arousals or apnoea events | PhysioNet/CinC 2018; apnoea annotations of the sleep cohorts | EEG + cardiorespiratory | event | AUPRC, event-level F1 |
| T7 | Trait and identity probes: age, sex, participant identification | sleep cohorts | each stream separately | trait, diagnostic | MAE / AUROC / top-1 identification |

Notes:
* T1 across datasets and T2 are the two **primary transfer tasks** for confirmatory tests: one near the pretraining domain, one far from it (wake, wearable, different hardware).
* T5 and T7 are diagnostics. T5 asks whether basic physiology is linearly readable from the embedding. T7 measures how much of the embedding is identity: high identifiability with poor state performance signals a representation dominated by who the person is.
* T3 on DEAP and AMIGOS is deliberately a far-transfer stress test. The earlier review found participant-independent DEAP arousal at chance for EEG-only models; a null here is informative, not a failure of the framework.
* Any task dataset that also contributes pretraining data is split by participant *before* pretraining, and its test participants are excluded from every pretraining run.

### 3.1 Input policy: privileged signals and shortcut risks

A benchmark is invalid if the model is given the signal from which its target was constructed, or a signal that reveals the experimental condition by a route other than physiology. Each task therefore declares its target source, and the harness refuses prohibited inputs.

| Task | Target constructed from | Prohibited as model input | Allowed inputs (primary benchmark) | Other shortcut risks and handling |
|---|---|---|---|---|
| T5 heart rate, PPG-DaLiA | R-peaks of the chest ECG, manually corrected; mean heart rate per 8 s window | Chest ECG and everything derived from it: beats, heart-rate variability, pulse arrival time, any coupling index, ECG-based phase encoding | **Wrist BVP and wrist accelerometer.** Declared variants: BVP alone; plus wrist EDA and temperature | Chest respiration and chest accelerometer are not wrist signals: excluded from the primary benchmark, allowed only in a labelled variant. No part of this dataset enters any pretraining phase |
| T5 respiratory rate, BIDMC | Breaths annotated by two annotators on the impedance respiration signal | Impedance respiration and anything derived | PPG (primary); ECG as a separate variant | Drawn from MIMIC-II: excluded when MIMIC data are in pretraining; inherits MIMIC's inter-signal delay problem |
| T1 sleep staging | Technician scoring of EEG, EOG and EMG | For *cardiorespiratory* staging: EEG, EOG, EMG and anything derived | ECG, PPG, respiration, as defined per evaluation cell (3.2). EEG-based staging is a separate task, flagged as using the label-defining signal | Time of night predicts stage: epoch index and clock time are not inputs; a variant that adds them is labelled. Same sequence head for all models |
| T6 arousal events | Scoring of EEG and EMG | EEG and EMG, for the autonomic-detection version | ECG, PPG, respiration | — |
| T6 respiratory events | Scoring of airflow, effort and oxygen saturation | Airflow, effort, saturation, for the cardiac-detection version | ECG, PPG | With respiration as input the task reduces to reading the label source; reported separately |
| T2 stress, WESAD | Protocol condition | Accelerometer (conditions differ in posture and movement); clock time or position in the session | Per device: chest {ECG, respiration, EDA}; wrist {BVP, EDA} | The stress condition includes speaking, which disturbs breathing: results with and without respiration. Skin temperature drifts with session time: excluded from the primary benchmark. Condition order is nearly fixed |
| T3 continuous arousal, CASE | Joystick operated by the participant | Muscle-activity channels (may reflect joystick movement) | ECG, BVP, respiration, EDA | Each video has a designed emotion: evaluated with held-out videos as well as held-out participants. Ratings lag physiology: a lag-tolerant metric is reported as secondary |
| T3 DEAP, AMIGOS | Self-report per trial | Eye and muscle channels | EEG, BVP or ECG, EDA, respiration | Shared stimuli; see the earlier reports |
| T4 rhythm | Cardiologist reading of the ECG | None: the input is the label-defining signal by design | One ECG lead | Requires patient-level folds |
| T7 traits, identity | Metadata | — | Each stream separately | Identity probe needs time-separated segments per person |
| Coupling probes (8.4) | Computed from the same signals the model sees | — | — | Content probes, never reported as task performance |

Rules that apply everywhere:
1. A feature or encoding derived from a prohibited signal is itself prohibited. This includes hand-crafted coupling features and physiological-phase encodings.
2. Privileged signals of evaluation participants never appear in any pretraining phase, labelled or not. Seeing the ECG–PPG pairs of the people on whom PPG heart rate is later tested would teach the target relation transductively.
3. A variant that relaxes a rule is a separate, labelled row; it is never the headline number.
4. Using a prohibited signal to *estimate recording metadata* (for example a clock offset) is allowed, because a per-recording constant carries no target information; it is logged.

### 3.2 Task–modality compatibility and evaluation cells

Datasets with different modality sets are not interchangeable. Every comparison is made inside one **evaluation cell**: a fixed dataset, task and input set.

| Cell | Dataset | Task | Input set, identical for all models compared | Synchronisation of inputs (report 03, 5a) | Role |
|---|---|---|---|---|---|
| E1 | MESA, held-out participants | sleep staging | ECG + PPG + respiration | one recorder | within-corpus, full cardiorespiratory triad |
| **E2** | CinC 2018 | sleep staging | ECG + respiration (the dataset has no PPG) | one recorder | **primary**: cross-dataset, modality subset |
| **E3** | WESAD, chest device | stress | ECG + respiration | one device | **primary**: wake, far transfer |
| E4 | WESAD, wrist device | stress | BVP | — | far transfer of the PPG stream alone |
| E5 | CASE | continuous arousal | ECG + BVP + respiration | one acquisition unit | wake, full triad; the only verified dataset outside pretraining with all three on one clock |
| E6 | HMC | sleep staging | ECG alone; EEG alone (two separate cells) | one recorder | harness development and unimodal checks. HMC has no respiration and no PPG, so it takes no part in the cardiorespiratory fusion experiment |
| E7 | PPG-DaLiA | heart rate | wrist BVP + wrist accelerometer | one device | content probe |
| E8 | DREAMT | sleep staging | wrist BVP (+ EDA) | one device | wearable sleep staging; cross-device fusion with the sleep-lab signals is not attempted |

* Results from cells with different input sets are never averaged into one score.
* When a model was pretrained with more modalities than a cell provides, the absent ones are simply missing at test time, and the cell is declared a missing-modality evaluation.
* The value of ECG–PPG coupling for fused use can be seen only in E1 and E5; in E4 and E7 it can appear only as an improved PPG stream.
* EEG remains a first-class modality. Cells that add the EEG stream (to E1, E2, E6) are defined now and reported once EEG joins cross-modal pretraining; they are flagged because EEG is the label-defining signal for sleep stages.

## 4. Adaptation protocols

Each pretrained model is evaluated under the same four protocols with the same tuning budget (identical number of hyper-parameter configurations, selected on validation participants only):

| Protocol | Trainable parts | Purpose |
|---|---|---|
| Linear probe | ridge or logistic regression on mean-pooled frozen embeddings | Is the information linearly accessible? Lowest variance; primary for label-efficiency |
| Attentive probe | one attention-pooling layer + linear head on frozen token embeddings | Allows the probe to select tokens; standard for latent-prediction models, whose information is spread over tokens |
| Partial fine-tuning | low-rank adapters or the top two encoder layers | Realistic adaptation at small data |
| Full fine-tuning | everything | Upper bound; least attributable to the representation |

Sequence tasks (T1, T6) add the same small temporal head over window embeddings for every model, including baselines.

## 5. Reference models every table must contain

| Reference | Isolates |
|---|---|
| Hand-crafted features + linear model and gradient boosting (band-power, heart-rate variability, pulse morphology, skin-conductance responses, breathing features, classical coupling indices) | whether learning is needed at all |
| Compact supervised network trained from scratch on the downstream data (1-D residual network per modality, late fusion) | value of pretraining versus a well-tuned small model |
| The evaluated architecture with **random frozen weights** + the same probes | contribution of the architecture's inductive bias alone |
| The evaluated architecture trained **from scratch** end to end | contribution of pretraining to fine-tuned performance |
| Publicly released pretrained encoders, frozen, same probes (one per modality where a usable checkpoint exists) | whether our pretraining is competitive with reuse |

A pretrained model is credited only with what it achieves beyond the best of these under the same protocol.

## 6. Core comparisons

All comparisons hold fixed: architecture family, parameter count (within 5 %), number of pretraining gradient steps, number of pretraining windows seen, augmentation set, downstream protocol and tuning budget. Where two variants cannot be parameter-matched exactly, the simpler one is widened to match.

| ID | Comparison | Held fixed | Primary read-out |
|---|---|---|---|
| **K1** | Pretrained versus random initialisation | architecture, downstream protocol | label-efficiency curves on T1 (cross-dataset) and T2 |
| **K2** | Cross-modal pretraining versus unimodal pretraining of the same encoders | encoders, steps, data; fusion module trained only downstream in the unimodal arm | linear and attentive probes, T1 and T2 |
| **K3** | Interaction fusion versus late gated fusion versus capacity-matched no-exchange control; plus late fusion with the explicit pairwise lag module (report 04, section 4.4) | encoders, objective | T1, T2; synthetic coupling benchmark |
| **K4** | True pairing versus perturbed pairing during cross-modal pretraining: within-window circular shift and within-state swap, each read against the unimodal arm (section 8.2) | everything except the pairing | T1, T2; coupling probes |
| **K5** | Modality-specific encoders versus one shared encoder versus shared trunk with specific front ends | total parameters, objective | per-modality probes; T1, T2 |

K1–K5 are the pre-declared confirmatory comparisons (Holm-corrected within each primary task). "T1" and "T2" in this table mean the primary evaluation cells E2 and E3 of section 3.2. Further comparisons are exploratory and labelled so:

* simple versus advanced signal representations (one substitution at a time; see report 01);
* frozen versus fine-tuned;
* pretraining with and without modality masking;
* pretraining-data size (0, 10 %, 30 %, 100 % of prototype hours);
* within-dataset versus cross-dataset evaluation;
* **surrogate-pretraining control:** pretrain on phase-randomised surrogates of the same recordings, which keep each signal's spectrum but destroy waveform shape and cross-modal timing. If transfer is unchanged, the encoder has learned spectral statistics only.

## 7. Label-efficiency protocol

* Sub-sample **participants**, not windows: 2, 5, 10, 25 and all training participants, five random subsets each (nested, so smaller sets are contained in larger ones).
* Evaluate on a fixed set of test participants never used in any subset.
* Report the full curve and its area; the confirmatory read-out for K1 is the difference at the smallest two sizes, where pretraining should matter most.
* Tasks with fewer than about 30 participants (WESAD, CASE) use leave-one-participant-out for testing and sub-sample the remaining participants.

## 8. Cross-modal learning: dedicated evaluation

### 8.1 Questions and experiments

| Question | Experiment | Evidence against the hypothesis |
|---|---|---|
| 1. Does cross-modal pretraining improve representations? | K2, plus the *unimodal* downstream test: probe each modality's private stream alone after cross-modal versus unimodal pretraining | No gain on multimodal probes and no gain on unimodal probes |
| 2. Does explicit interaction beat late fusion? | K3 with three arms; the control has the same block with attention restricted to the token's own modality | Interaction ≤ control |
| 3. Does fusion help when a modality is noisy or missing? | Section 9 suite; compare fused model with the best single remaining modality | Fused model below best single modality under corruption (the failure CoRe-Sleep reported for early and cross-attention fusion) |
| 4. Are learned interactions robust across environments? | Train probes on one dataset or device, test on another (finger versus wrist PPG; clinical versus wearable ECG); compare degradation of fused versus late-fusion models | Larger cross-environment drop for the interaction model |
| 5. Do alignment and lag modelling matter? | K4; ablate the relative-time attention bias; the pairwise lag-module control; test-time rotation ladder and predictability curve (section 8.2) | Performance insensitive to all of these |
| 6. Does hierarchical fusion beat flat fusion? | Cardiorespiratory-first grouping versus flat versus a random grouping of the same shape | Hierarchical ≤ random grouping |
| 7. Is modality-specific information preserved? | Modality-specific probes on private streams before and after joint training (QRS morphology class, pulse-shape indices, skin-conductance-response count, EEG band-power) | Probe performance falls after joint training |
| 8. Does one modality dominate the shared representation? | Drop each modality at test time and measure the change; centred-kernel-alignment similarity between the fused embedding and each private embedding; unimodal heads' accuracy | One modality accounts for nearly all of the fused model's performance and similarity |

Attention maps and learned lag kernels are reported as descriptions. They count as evidence only when the same analysis recovers known structure on the synthetic benchmark and when an intervention (time shift, modality removal) changes performance as the map predicts.

### 8.2 Pairing-perturbation controls (K4)

A plain time shift of seconds to minutes changes several things at once: fine timing, but also physiological state (a sleep-stage transition may lie between the two windows) and, across recordings, identity and recording conditions. Mismatched pairs also make the cross-modal objective unsatisfiable, which can harm training for reasons unrelated to lost information. The controls below change one factor at a time.

**Factors.** F: fine temporal correspondence inside the window (beat- and breath-level). W: window-specific shared content not captured by state labels. S: slow physiological state (sleep stage, level of heart rate and breathing rate, posture). P: participant identity. R: recording conditions (device, site, night).

**Pretraining-time perturbations of the partner modality.**

| Arm | Construction | F | W | S | P | R |
|---|---|---|---|---|---|---|
| P0 true pairing | — | kept | kept | kept | kept | kept |
| **P1 within-window circular shift** | The partner modality's own 30 s window is rotated in time by a random amount, drawn afresh for every sample. Ladder of maximum rotation: ±0.5 s, ±2 s, ±8 s, full window | destroyed above the ladder's scale | kept | kept exactly | kept | kept |
| **P2 within-state swap** | Partner taken from another window of the same recording, at least 5 min away, in the same scored state, with mean heart rate within ±3 beats/min and breathing rate within ±1.5 per min. Windows within 30 s of a stage transition, scored arousal, respiratory event or position change are excluded on both sides | destroyed | destroyed | kept, as far as matched | kept | kept |
| P3 different state, same recording | Partner from a different scored state | destroyed | destroyed | destroyed | kept | kept |
| P4 different participant, matched state | Partner from another participant of the same dataset and site, same state, matched rates | destroyed | destroyed | kept (category) | destroyed | kept |
| P5 different participant, different state | — | destroyed | destroyed | destroyed | destroyed | dataset level only |
| U unimodal reference | No cross-modal term; its prediction targets are reassigned to the within-modality term so that compute and number of targets match | — | — | — | — | — |

*Why the circular shift.* The rotated partner window contains exactly the same samples. Every marginal statistic of that modality in that window is preserved (amplitude distribution, spectrum, heart rate and its variability, waveform shape), and state, identity and recording are identical by construction. The seam where the rotated window wraps is masked and excluded from the loss; a seam mask at a random position is applied in **all** arms, including P0, so the seam reveals nothing about the arm. The rotation is random, not fixed, so the model cannot learn a new constant offset. One caveat: for a nearly periodic signal a rotation by a whole number of cycles is almost invisible, so rate-level correspondence survives by design. What is destroyed is the correspondence of individual beats and breaths.

State annotations are used only to construct P2–P5, never as model input, and only for pretraining participants.

**Training-difficulty confound.** In P1–P5 the cross-modal targets are partly or wholly unpredictable. Three safeguards:
1. Every arm is read against both P0 and U through the retained fraction ρ = (arm − U) / (P0 − U).
2. An arm that falls *below* U shows that mismatch harms optimisation. Its result is then not interpreted as information loss.
3. A dose-response run trains on a 50 % mixture of true and perturbed pairs; if the effect is informational it should fall between the pure arms.

**Test-time perturbations, which have no training confound.** Take the model pretrained on true pairs and perturb only at evaluation:
* apply the P1 ladder to one modality of the evaluation windows, probes unchanged;
* measure cross-modal prediction error on held-out pretraining-domain data as a function of rotation size (a predictability curve).

These show whether the trained model *uses* fine timing, separately from whether training *needed* it.

**What each result would and would not establish.**

| Result | Establishes | Does not establish |
|---|---|---|
| P0 > U | Cross-modal training improves the representation at matched targets and compute (this is K2) | Which aspect of the correspondence is responsible |
| P1 at full-window rotation ≈ P0 | The gain needs no timing inside the window; correspondence at the level of the 30 s window is enough, and the lag term is unnecessary for these tasks | That the signals lack fine coupling, or that tasks sensitive to it would not benefit |
| P1 ≈ U from some ladder step upward, ≈ P0 below it | The gain depends on correspondence at that timescale: beat-level if lost at ±0.5 s, breath-level if lost at ±2–8 s | That the correspondence is physiological rather than a shared artefact (motion, common electrical noise). Checked on artefact-free segments and on the synthetic benchmark |
| P2 < P1 at full-window rotation | Content specific to the window, beyond labelled state and matched rates, contributes | What that content is; state labels are coarse |
| P3 < P2 | Shared slow state contributes | — |
| P4 < P2 | Person-specific correspondence contributes | Whether that is desirable; it may be identity leakage, to be read with the identity probe |
| P5 ≈ P4 ≈ U | Nothing beyond dataset-level regularities is used when person and window are wrong | — |
| Any arm < U | Mismatched targets harm optimisation | Anything about information content |
| Test-time ladder changes probe performance | The trained model's outputs depend on fine timing and the task benefits from it | That pretraining required it |
| Predictability curve flat | The model does not exploit fine timing for cross-modal prediction | — |

**Confirmatory and exploratory parts.** K4 is P0 versus P1 at full-window rotation, and P0 versus P2, each read with U. The P1 ladder, P3–P5, the mixture run and the test-time analyses are exploratory. Every perturbation experiment requires a dataset whose synchronisation supports the timescale under test (report 03, section 5a); in particular, beat-level conclusions cannot be drawn from recordings with documented variable inter-signal delay.

### 8.3 Synthetic coupling benchmark (positive and negative control)

Simulated multimodal recordings in which the coupling is known and the label depends on the coupling, not on any single modality's statistics.

| Generator | Signals | Ground-truth coupling | Label |
|---|---|---|---|
| G1 cardiac–pulse | ECG-like beat train with realistic interval variability; pulse wave following each beat | pulse arrival delay, 150–350 ms, slowly varying | level or trend of the delay |
| G2 cardiorespiratory | breathing oscillator with variable rate; heart rate modulated by breathing | modulation gain and phase lag | gain or lag class |
| G3 central–autonomic | sparse latent events producing an EEG band-power change, a heart-rate acceleration 2–6 s later and a skin-conductance response 1–4 s later through a slow impulse response | event-locked lags; coupled versus independent events at identical marginal rates | coupled versus independent; lag value |

Nuisance factors: additive noise, random gain, sensor dropout, sampling-clock offset. For every generator a **marginal-matched null** version has identical per-modality distributions and no coupling.

Uses:
1. Supervised: which fusion mechanisms can learn each label, as a function of lag size and jitter, at matched capacity.
2. Self-supervised: after pretraining on unlabelled synthetic data, is the coupling parameter linearly decodable from the fused embedding, and not from concatenated unimodal embeddings?
3. Mechanism check: does the learned relative-time bias peak at the true lag?
4. Calibration of 8.2: the pairing-perturbation controls must behave as predicted when the truth is known.

A fusion mechanism or objective that fails here is not carried to real data. Success here shows capability, not that real physiology rewards it.

### 8.4 Real-data coupling probes

Classical signal processing supplies coupling quantities that no single modality contains: pulse arrival time (ECG and PPG), respiratory sinus arrhythmia amplitude (ECG and respiration), cardiorespiratory phase-synchronisation index. Linear probes regress these from (a) the fused embedding and (b) concatenated private embeddings. A fused representation that has learned coupling should be clearly better on (a). The same quantities, fed directly to a linear model, form the hand-crafted coupling baseline for T1 and T2.

## 9. Missing-modality and corruption tests

* **Missing:** every non-empty subset of the available modalities at test time. Arms: pretraining and fine-tuning with versus without modality masking.
* **Corruption,** graded, applied to evaluation participants only: additive noise at 20, 10, 0, −5 dB; motion-like band-limited bursts covering 10, 30, 50 % of the window; flat-line and clipping segments; baseline drift (EDA, respiration); EEG channel loss; sampling-clock drift between modalities.
* **Real artefacts:** wrist PPG during activity (PPG-DaLiA) as a naturally corrupted condition.
* Read-outs: area under the performance-versus-severity curve; the fused model relative to the best single clean modality; gate weights versus injected corruption (does the gate turn away from the corrupted sensor?).
* Robustness is reported only where clean performance exceeds the permutation null.

## 10. Leakage and overlap audit

| Risk | Control |
|---|---|
| Evaluation participants seen during pretraining | Participant-level manifest for every dataset; test participants of every downstream task excluded from all pretraining runs; manifest hash stored with each checkpoint |
| Target-source or privileged signals used as input, directly or through derived features | Per-task input policy (section 3.1); the harness raises an error on a prohibited input |
| Datasets without participant identifiers | Not used for confirmatory participant-disjoint evaluation |
| Derived datasets sharing patients with a pretraining source (for example a benchmark extracted from a larger clinical waveform database) | Listed in report 03; such pairs are never used as pretraining source and evaluation target together |
| Public checkpoints pretrained on our evaluation data | For each reused checkpoint, its pretraining datasets are checked against our task list; conflicting tasks are marked and excluded from comparisons with that checkpoint |
| Window overlap across splits | Splits at participant level before windowing; overlapping windows inherit the split |
| Tuning on test participants | Validation participants are a separate group; test participants are evaluated once per frozen configuration |
| Same-night or same-session leakage in personalised setting | Calibration and test windows separated in time by a guard interval |

A "leave-one-pretraining-dataset-out" run (pretrain without the dataset most similar to a downstream task) quantifies how much apparent transfer is domain proximity.

## 11. Statistics

* Unit of analysis: the participant. Metrics are computed per participant and averaged.
* Intervals: participant-level bootstrap, 95 %.
* Comparisons: paired per-participant differences, sign-flip permutation test; Holm correction over K1–K5 within each primary task.
* Seeds: at least three **pretraining** seeds for every arm of a confirmatory comparison at prototype scale, and five downstream seeds per pretraining seed. Variance across pretraining seeds is reported separately; it is routinely omitted in the literature and can exceed the effects being claimed.
* Minimum detectable effect is computed per task from the observed paired-difference spread; smaller differences are reported as unresolved.
* A claim of general transfer requires the same direction of effect on both primary tasks; a single-task effect is reported as task-specific.
* Permutation null (labels shuffled within participant) for every task with fewer than 50 participants.

## 12. Attribution checklist for any reported improvement

| Possible source | Ruled out by |
|---|---|
| Downstream head or tuning | identical probes and tuning budgets; linear probe as primary |
| Added capacity | parameter- and step-matched controls; widened baselines |
| Architecture prior | random-frozen-weights reference |
| More data rather than better objective | pretraining-size curves at equal data |
| Dataset overlap or domain proximity | overlap audit; leave-one-dataset-out |
| Spectral statistics only | surrogate-pretraining control |
| Shared slow state, window content, identity or recording conditions rather than coupling | pairing-perturbation controls P1–P5 |
| Optimisation harm from mismatched pairs | unimodal reference U; mixture run; test-time perturbations |
| Privileged input or target leakage | input policy (section 3.1), enforced by the harness |
| Participant identity | T7 identification probe; transductive results reported separately |
| Chance | permutation null; pretraining-seed variance |
