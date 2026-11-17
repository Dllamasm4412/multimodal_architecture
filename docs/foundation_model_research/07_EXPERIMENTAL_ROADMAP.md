# 07 — Experimental Roadmap

Staged plan from harness to confirmatory evaluation. No calendar is given; compute is stated per stage as an estimate. Names used: Architectures A, B, C (report 04); objectives O1, O2 (report 05); tasks T1–T7, evaluation cells E1–E8 and confirmatory comparisons K1–K5 (report 06). "Primary tasks" always means the primary cells E2 and E3; prototype and research corpora (report 03).

## 1. Hypotheses and what would invalidate them

| ID | Hypothesis | Invalidated if | Tested in |
|---|---|---|---|
| H1 | Self-supervised pretraining on unlabelled physiological recordings improves downstream learning with few labelled participants | Pretrained encoders do not beat the same network from scratch, or hand-crafted features, at the smallest label sizes (K1) | Stage 2 |
| H2 | Encoders designed per modality transfer better than one shared encoder at equal total size | Shared encoder matches or beats modality-specific ones (K5) | Stage 2 |
| H3 | Cross-modal pretraining yields better representations than unimodal pretraining of the same encoders | No gain on multimodal or unimodal probes (K2) | Stage 3 |
| H4 | Token-level interaction fusion beats late fusion beyond what added capacity explains | Interaction fusion ≤ capacity-matched no-exchange control (K3) | Stages 1, 3 |
| H5 | The benefit of cross-modal learning depends on fine temporal correspondence (physiological coupling and lags) | Transfer is unchanged when the partner modality is rotated in time within its window, or swapped for another window in the same physiological state, during pretraining (K4) | Stage 3 |
| H6 | The effects persist with more data and across datasets, devices and tasks | Effects fall below the detectable size at research scale, or appear on one dataset only | Stage 5 |
| H7 | What is learned exceeds spectral statistics | Pretraining on phase-randomised surrogates transfers equally well | Stage 3 (exploratory control) |

H3–H5 are the scientific core. H5 can fail while H3 holds; that outcome is still a clear, publishable result ("cross-modal pretraining helps through shared slow state, not through coupling") and it simplifies the architecture.

## 2. Resource tiers

| Tier | Pretraining data | Model | Compute (estimate) | Purpose |
|---|---|---|---|---|
| **Prototype** | ≈ 4,000 h from ≈ 500 participants; ≈ 17 GB after preprocessing | 4–13 M parameters | one accelerator; 7–27 GPU-hours per pretraining run and days per confirmatory comparison are hypotheses, replaced by the Stage 0b measurement | Stages 0–4: decide whether the hypotheses hold |
| **Research** | 50,000–100,000 h; ≈ 150–300 GB after preprocessing, several hundred GB raw | 13–50 M | 1–2 GPUs; 2–5 GPU-days per run; weeks for an ablation table | Stage 5: does it scale and generalise |
| **Foundation (aspirational)** | ≥ 500,000 h, multi-institution | ≥ 100 M | multi-GPU for weeks | Not planned; listed to mark the boundary |

No research-tier run is started before the prototype result that motivates it.

## 3. Stages

### Stage 0 — Data layer, benchmark harness, reference baselines

* **Hypothesis.** None. Establishes that the downstream tasks are solvable above chance under participant-disjoint evaluation and how far conventional features and public pretrained encoders already go.
* **Components.** Unified on-disk format that keeps original sampling rates, timestamp origins, device clock groups, estimated clock offsets and drift with their method and confidence, missing-data intervals and raw-to-processed provenance (report 08); dataset adapters; a synchronisation audit that assigns each dataset a class from its documentation and an empirical check (report 03, section 5a); per-task input policy with enforcement (report 06, section 3.1); evaluation cells (report 06, section 3.2); participant-level split manifests and the pretraining–evaluation overlap audit; hand-crafted feature library including scale descriptors and classical coupling indices; compact supervised networks; probe code; statistics (bootstrap, paired permutation, permutation null, minimum detectable effect); data-loading and preprocessing throughput measurement.
* **Datasets.** Open-access sets first, as specified in report 08: WESAD (cells E3, E4), PPG-DaLiA (E7) and HMC (E6: sleep staging from ECG alone and from EEG alone; HMC has no respiration or PPG and is not part of the cardiorespiratory fusion experiment). PTB-XL (T4) and the cross-dataset sleep cell E2 are added once the harness is stable. Access requests for the prototype pretraining corpus are submitted in parallel.
* **Design.** Participant-disjoint cross-validation per task; inductive setting; label-efficiency curves for baselines.
* **Evaluation.** Metrics of report 06, section 3, with intervals; frozen public encoders with linear and attentive probes where a compatible checkpoint exists.
* **Acceptance.** Harness tests pass on synthetic fixtures, including refusal of prohibited inputs; each primary cell (E2, E3) has a baseline above its permutation null; minimum detectable effect known per cell; overlap manifest complete; every dataset has a synchronisation class with its evidence recorded.
* **Dependencies.** Dataset access.
* **Go/no-go.** A task whose best baseline is not above the null loses its confirmatory role and is reported as exploratory. If neither primary task clears the null, replace the task before any pretraining.

### Stage 0b — Resource benchmark (before any pretraining)

* **Hypothesis.** The compute figures of report 04, section 4.6 (0.5–2 thousand windows/s; batch 256 in about 4.5 GB with a sleep EEG montage and about 14 GB with 32 EEG channels, in mixed precision) are within a factor of two of reality on the available hardware.
* **Components.** A skeleton of Architecture B with student, teacher update, prediction head and AdamW, run on synthetic tensors of the true shapes; the Stage 0 data loader on the preprocessed store.
* **Dataset.** Synthetic tensors; a few hours of preprocessed real data for the loader test.
* **Design.** The sweep of report 04, section 4.6: batch size, EEG channel count (0, 2, 3, 6, 14, 32), precision, masked tokens dropped or kept, pairwise module on or off; loader workers.
* **Evaluation.** Training throughput, peak accelerator memory, data-loading throughput, preprocessing speed, stored bytes per recording-hour.
* **Acceptance.** A measured table exists and the pretraining budget has been recomputed from it.
* **Dependencies.** Stage 0 data layer.
* **Go/no-go.** Data loading below 1.5 × training throughput → fix storage first. Below 300 windows/s at two EEG channels → shorten the schedule or reduce width before Stage 2. The number of arms and seeds in Stage 3 is then fixed from the measured budget, in the priority order given there.

### Stage 1 — Synthetic coupling benchmark (parallel with Stage 0)

* **Hypotheses.** (a) H4 in a setting where the truth is known: with labels that depend on cross-modal lag, interaction fusion with a relative-time term succeeds where late fusion and the no-exchange control do not, at matched capacity. (b) After O1 pretraining on unlabelled synthetic data, coupling parameters are linearly decodable from the shared tokens and not from concatenated private embeddings. (c) The pairing-perturbation controls behave as theory predicts. (d) Sub-second ECG-to-pulse delay survives the fusion bottleneck, or the explicit pairwise lag module is needed to carry it.
* **Components.** Generators G1–G3 with marginal-matched null versions (report 06, section 8.3); small versions of the tokenisers, private encoders and all fusion variants, including the pairwise lag-interaction module (report 04, section 4.4); O1 and O2; the pairing perturbations of report 06, section 8.2.
* **Dataset.** Synthetic only.
* **Design.** Sweep lag magnitude, lag jitter, noise and coupling strength; three seeds; laptop-scale models.
* **Evaluation.** Label accuracy per fusion variant; probe R² for coupling parameters; recovered lag kernel versus true lag; false-positive rate on null data; **delay-resolution test** on G1: window-mean pulse arrival delay recovered from the frozen shared tokens to within 10 ms mean absolute error, and within-window variation tracked with correlation ≥ 0.5.
* **Acceptance.** Positive and negative controls both pass for at least one interaction variant; for second-scale generators (G2, G3) the lag kernel peaks at the true lag; the delay-resolution test passes for one of: default fusion, fusion tokens every 0.25 s, default plus pairwise module (tried in that order; the first that passes is adopted); the pairing perturbations behave as predicted on data with known coupling.
* **Dependencies.** None.
* **Go/no-go.**
  * Interaction variant cannot beat the no-exchange control even here → fix or replace the fusion mechanism before Stage 3.
  * Late fusion solves every generator at matched capacity → interaction fusion has no demonstrated mechanism advantage at these timescales; H4 is tested on real data only as exploratory and the fusion claim is narrowed to the pretraining objective.
  * O1 fails to make coupling decodable but O2 succeeds → O2 becomes the primary objective.

### Stage 2 — Unimodal pretraining at prototype scale (H1, H2)

* **Hypotheses.** H1 (K1) and H2 (K5).
* **Components.** Architecture A; the private encoders of B; O2 and the within-modality part of O1; shared-encoder and shared-trunk variants for K5.
* **Dataset.** Prototype corpus, each modality using all recordings that contain it. EEG is pretrained here as a first-class stream (sleep montage), although it joins cross-modal training only after the cardiorespiratory experiment of Stage 3.
* **Design.** Per modality: {A with O2, B-encoder with O2, B-encoder with O1} × pretraining size {0, 10 %, 30 %, 100 %} × three pretraining seeds. K5: separate encoders versus shared trunk with separate tokenisers versus one shared encoder, at equal total parameters.
* **Evaluation.** Label-efficiency curves on the primary cells E2 and E3, on unimodal cells (E4, E6, E7) and on T4, with linear and attentive probes and full fine-tuning, each under its input policy; with and without scale-descriptor tokens; identity probe T7; comparison with Stage 0 references.
* **Acceptance.** K1: at the two smallest label sizes, pretrained beats from-scratch and hand-crafted with interval excluding zero on both primary tasks, without loss at full labels. A monotone benefit with pretraining size.
* **Dependencies.** Stage 0.
* **Go/no-go.**
  * K1 passes → Stage 3 with the better of O1 and O2.
  * K1 fails for our encoders but public checkpoints do show the benefit on our tasks → the limit is scale; switch the unimodal streams to reused encoders with adapters and continue to Stage 3 on that basis.
  * K1 fails for our encoders *and* for public checkpoints → pretraining does not help these tasks at this label regime; the premise of the project is not supported for this task set. Revisit task selection once; otherwise stop and report.
  * K5: if the shared encoder is within the detectable effect of modality-specific encoders, adopt the shared trunk (simpler, more data per parameter) and record H2 as unsupported.

### Stage 3 — Cross-modal pretraining and fusion: the decisive experiment (H3, H4, H5)

* **Hypotheses.** K2, K3, K4; H7 as an exploratory control.
* **Components.** Full Architecture B; fusion variants; whole-modality masking; pairing-perturbation machinery (report 06, section 8.2); contrastive-alignment baseline.
* **Dataset.** The synchronised part of the prototype corpus (ECG + PPG + respiration recorded through one recorder; EEG added afterwards). Recordings with documented variable delay between signals are not used for this stage.
* **Design.** All arms use the same private encoders, parameter count, gradient steps and windows seen; three pretraining seeds each.

| Arm | Pretraining | Fusion |
|---|---|---|
| 1 | unimodal | late gated, trained downstream |
| 2 | unimodal | bottleneck, trained downstream |
| 3 | **cross-modal (O1)** | **bottleneck with lag bias** |
| 4 | cross-modal (O1) | no-exchange control |
| 5a | cross-modal (O1), partner modality **rotated within its own window** by a random amount up to the full window (P1) | as arm 3 |
| 5b | cross-modal (O1), partner modality taken from **another window in the same state**, matched heart and breathing rates (P2) | as arm 3 |
| 5c (exploratory) | P1 ladder at ±0.5 s, ±2 s, ±8 s; different state (P3); other participant, matched state (P4); other participant, other state (P5); 50 % mixture | as arm 3 |
| 9 | cross-modal (O1) | late gated + pairwise lag-interaction module, parameters matched to arm 3 |
| 6 | cross-modal contrastive alignment | late gated |
| 7 | arm 3 on phase-randomised surrogate signals | as arm 3 |
| 8 (exploratory) | cross-modal (O1) | per-token delay trained against pulse arrival time, ECG–PPG only (the published alternative) |

* **Evaluation.** Primary cells E2 (cross-dataset sleep staging from ECG + respiration) and E3 (stress from chest ECG + respiration); secondary cells E1 (held-out pretraining-cohort participants, full triad), E4 (wrist BVP alone) and E5 (continuous arousal, full triad on one clock); coupling probes; unimodal probes on private streams; dominance analysis; missing-modality matrix; test-time rotation ladder and predictability curve on arm 3. Arm 1, with within-modality targets increased to match, serves as the unimodal reference U.
* **Priority under a limited budget.** Arms 1, 3, 4, 5a, 5b with three pretraining seeds come first (15 runs). Arms 9 and 2 next, then 6 and 7, then 5c and 8.
* **Acceptance.**
  * K2: arm 3 exceeds arms 1 and 2 by at least the detectable effect on both primary tasks, or on unimodal probes of at least one stream.
  * K3: arm 3 exceeds arm 4.
  * K4: arms 5a and 5b retain clearly less of the gain of arm 3 over the unimodal reference than arm 3 itself, and neither falls below the reference (which would indicate optimisation harm, not information loss).
* **Dependencies.** Stages 1 and 2.
* **Go/no-go.**

| Outcome | Decision |
|---|---|
| K2, K3 pass; rotation within the window (5a) loses most of the gain | Full hypothesis supported: the benefit needs temporal correspondence inside the window. Use the ladder (5c) to say at which scale. Proceed to Stages 4–6 with B |
| K2 passes; 5a keeps the gain, 5b loses it | Cross-modal learning helps through content shared by the same window, not through timing inside it. Drop the lag term, prefer the simplest fusion that keeps the gain, and state the finding as such |
| K2 passes; 5a and 5b both keep the gain | The benefit comes from shared slow state or identity. Window-level alignment is sufficient; read the exploratory arms (P3–P5) and the identity probe to say which |
| Arm 9 matches arm 3 | Explicit pairwise lag features are sufficient; token-level mediated fusion is not needed for these tasks |
| K2 passes; K3 fails | Benefit comes from the pretraining objective, not from interaction fusion. Use late gated fusion downstream |
| K2 fails | One scale-up attempt at 3–5 × data (still prototype tier). If it fails again, adopt A, report a controlled negative result on cross-modal pretraining, and redirect effort to Priority A |
| Arm 7 matches arm 3 | The encoder learns spectral statistics; tighten masking and targets (report 05) before any scale-up |

### Stage 4 — Signal-processing extensions (Architecture C)

* **Hypothesis.** Each shortlisted component (report 01) adds to B independently: physiological-phase encoding; physically parametrised front end or scattering stem; Riemannian EEG token; coupling targets.
* **Components.** One addition at a time on the Stage 3 model; probe table for higher-order spectral, recurrence and topological descriptors.
* **Dataset.** Prototype corpus; multichannel EEG downstream sets for the Riemannian token.
* **Design.** B versus B + one component, with the component's specific control (phases from another recording; cross-device probe transfer; tangent-space classifier alone), at two pretraining sizes.
* **Evaluation.** Primary tasks and the component's own read-out (report 01, criterion 12).
* **Acceptance.** Gain of at least the detectable effect on a primary task or on the component's targeted read-out, not reproduced by its control.
* **Dependencies.** Stage 3 (any outcome that keeps B or A as a base).
* **Go/no-go.** Components that fail are dropped and reported. A descriptor is promoted from probe to target only if it is not linearly decodable from the embedding and predicts a downstream label beyond the embedding.

### Stage 5 — Scale, heterogeneous modalities, far transfer (H6)

* **Hypothesis.** The Stage 3 effects persist with roughly ten times the data, when datasets with different modality subsets are mixed, and on tasks far from the pretraining domain.
* **Components.** Research corpus; training with partially overlapping modality sets (report 03, curriculum); EDA stream; public EEG encoder plugged in as an alternative EEG stream for multichannel tasks.
* **Dataset.** Research corpus; downstream T1–T6 including continuous arousal (T3) and wake wearable tasks.
* **Design.** Repeat arms 1, 3 and 4 of Stage 3 at research scale (two pretraining seeds); leave-one-pretraining-dataset-out; cross-device probes.
* **Evaluation.** As Stage 3, plus T3 and T6; transductive and personalised settings reported separately.
* **Acceptance.** Direction and approximate size of the K2 and K3 effects reproduced on both primary tasks; no reliance on a single pretraining dataset.
* **Dependencies.** Stage 3 with K2 passed.
* **Go/no-go.** Effects shrinking below the detectable size with scale are reported as low-data benefits. Far-transfer nulls (for example arousal from sleep-pretrained encoders) are reported as limits of the pretraining domain, with the wake-data requirement that would be needed.

### Stage 6 — Robustness: missing and corrupted modalities

* **Hypothesis.** Modality masking in pretraining and reliability-aware gating keep the fused model at or above the best single clean modality under sensor loss and corruption.
* **Components.** Corruption suite; quality head; conventional quality indices; naturally corrupted wrist recordings.
* **Dataset.** Downstream sets of Stage 5.
* **Design.** Fused model versus best single modality across the missing-modality matrix and corruption severities; gate input: none, learned quality, conventional index.
* **Evaluation.** Area under the degradation curve; gate response to injected corruption.
* **Acceptance.** Fused model never below the best single clean modality by more than the detectable effect; learned quality gating not worse than conventional indices.
* **Dependencies.** Stage 3; clean performance above the null.
* **Go/no-go.** If the fused model falls below a single modality under corruption, the fusion stage is reworked (stronger modality masking, gating) before any robustness claim.

### Stage 7 — Frozen confirmatory evaluation and release

* **Hypothesis.** The selected effects replicate when nothing can be tuned.
* **Components.** Frozen configurations; K1–K5 re-run with fresh seeds on test participants untouched during development; one downstream dataset held out of all earlier stages.
* **Evaluation.** Holm-corrected confirmatory tests; full table of every stage's outcome including failed gates.
* **Acceptance.** Every claim in the write-up maps to a confirmatory comparison whose corrected interval excludes zero on both primary tasks, or is labelled task-specific or exploratory.
* **Deliverables.** Code, configurations, split manifests, checkpoints where licences allow, per-participant predictions.

## 4. Dependency graph

```
Stage 0 (harness, baselines) ─► Stage 0b (resource benchmark)
Stage 0 (harness, baselines) ──┬──► Stage 2 (unimodal: H1, H2) ──┐
                               │                                  ├──► Stage 3 (cross-modal: H3–H5)
Stage 1 (synthetic coupling) ──┴──────────────────────────────────┘          │
                                                    ┌────────────────────────┼───────────────┐
                                                    ▼                        ▼               ▼
                                          Stage 4 (extensions)     Stage 5 (scale, H6)   Stage 6 (robustness)
                                                    └────────────────────────┴───────────────┘
                                                                             ▼
                                                                   Stage 7 (confirmatory)
```

## 5. The smallest convincing experiment

Stage 3, arms 1, 3, 4, 5a and 5b, on cardiorespiratory signals only (ECG, PPG, respiration from one recorder), evaluated with frozen linear probes on cross-dataset sleep staging from ECG + respiration (cell E2) and on stress detection from chest ECG + respiration (cell E3). It needs the Stage 0 harness, the Stage 0b measurement, about 4,000 hours of data and a 13 M-parameter model; the hypothesised cost is 4–17 GPU-days including three seeds per arm. EEG is not dropped: it is pretrained unimodally in Stage 2 and joins cross-modal training immediately after this experiment. A positive result shows that cross-modal learning improves transferable representations for a stated reason; a negative result shows, with controls, that it does not at this scale. Either is a defensible thesis chapter.

## 6. Risks carried through the plan

| Risk | Where it is caught |
|---|---|
| Downstream tasks at chance | Stage 0 gate |
| Pretraining adds nothing over small supervised models | Stage 2 gate, with public checkpoints as a second opinion |
| Cross-modal gain is capacity | no-exchange control, Stages 1 and 3 |
| Cross-modal gain is shared slow state, window content, identity or recording conditions | pairing perturbations P1–P5 read against the unimodal reference; identity probe |
| Mismatched pairs harm optimisation and are mistaken for information loss | unimodal reference; mixture run; test-time perturbations |
| Privileged inputs or target leakage in a benchmark | input policy enforced by the harness (report 06, section 3.1) |
| Lag conclusions drawn from poorly synchronised recordings | synchronisation classes (report 03, section 5a) |
| Compute estimates wrong | Stage 0b |
| Gains are spectral statistics | surrogate-pretraining arm |
| Sleep-pretrained encoders do not transfer to wake tasks | far-transfer tasks reported separately in Stage 5 |
| Pretraining–evaluation overlap | manifest audit, Stage 0; leave-one-dataset-out, Stage 5 |
| Seed noise mistaken for an effect | three pretraining seeds per confirmatory arm |
| Accumulating unproven components | one-at-a-time admission, Stage 4 |
