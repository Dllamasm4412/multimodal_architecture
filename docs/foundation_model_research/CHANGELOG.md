# Changelog — technical revision before implementation

Revision of the eight foundation-model reports in response to the pre-implementation review. Structure and scientific conclusions are unchanged except where stated. Date: 2026-11-16.

## 1. What changed, by review item

| # | Review item | Change | Where |
|---|---|---|---|
| 1 | Benchmark leakage | PPG-DaLiA heart rate now uses wrist BVP and wrist accelerometer only; chest ECG is target-only, and anything derived from it (beats, variability, pulse arrival time, coupling indices, phase encodings) is prohibited. All other benchmarks audited: sleep staging (EEG/EOG/EMG are the label source), event detection, WESAD (accelerometer, session time and temperature reveal the condition; speaking disturbs breathing), CASE (muscle channels may reflect joystick use; video identity), BIDMC (impedance respiration is the label source), rhythm classification. Privileged signals of evaluation participants are barred from pretraining. The harness enforces the policy | 06 §3.1, §10, §12; 08 §7, §11; 05 §9 |
| 1 | Participant-disjointness | PhysioNet/CinC 2017 has no documented patient identifiers; the rhythm task moves to PTB-XL, whose folds keep each patient together | 06 §3; 03 §3 |
| 2 | Lag-aware fusion | Stated precisely: the lag term is a per-modality temporal receptive field, not a pairwise coupling term; it captures population-average, second-scale offsets and cannot represent window-specific or sub-token delays or absolute physiological delay. All cross-modal interaction is mediated by fusion tokens. ECG–PPG delay (150–400 ms) is about one 250 ms token wide, so it must be carried in token content; input resolution is adequate, survival through one-second fusion tokens is unproven. No redesign. Added a ≈ 17 k-parameter pairwise lag-interaction module as a capacity-matched control and optional component, and a delay-resolution test with thresholds | 04 §4.4, §4.7; 05 §5, §8; 01 §3(1); 07 Stage 1, Stage 3 arm 9; 06 K3 |
| 3 | Time-shift controls | The 5 s / 60 s / 10 min shifts are replaced by pairing perturbations that change one factor at a time: within-window circular shift (same samples, state, person and recording; only timing changes), within-state swap with matched heart and breathing rates, different state, different participant with and without matched state. A unimodal reference with matched targets, a mixture run and test-time perturbations separate lost information from optimisation harm. A table states what each result does and does not establish | 06 §8.2; 07 H5, Stage 3 arms 5a–5c, outcomes, §5 |
| 4 | Synchronisation | Each dataset's documented synchronisation reviewed and assigned a class (A same converter; B one recorder with possible channel latency; C event-marker or author alignment; D unreliable). Raw-device timing is distinguished from provided aligned forms. The unified format now records original rates, timestamp origin, clock groups, estimated offset and drift with method and uncertainty, synchronisation class, gap intervals and raw-to-processed provenance. Beat-level use requires class A or B and is never inferred from nominal timestamps | 03 §5a; 08 §4, §6; 04 §2 |
| 4 | New finding | MIMIC-III waveforms document a changing delay of up to 500 ms between signals. They are excluded from lag-sensitive pretraining and from the perturbation controls; the open substitute for ECG–PPG timing becomes VitalDB. This also weakens timing claims in prior work pretrained on that corpus | 03 §4, §5a, §7; 02 §4 |
| 5 | Preprocessing loss | Per-window normalisation is kept for the waveform but paired with a scale-descriptor token per modality (amplitude, baseline or tonic level, band-power, pulse amplitude, quality), with dropout and an ablation. PPG band widened to 0.1–8 Hz so breathing-related baseline modulation is kept; EEG to 0.3–45 Hz for slow oscillations. Three normalisation modes are defined and reported separately. No normalisation is stored; raw files are never modified | 04 §2; 08 §4, §5, §8; 05 §5 |
| 6 | Task–modality compatibility | Evaluation cells (dataset, task, fixed input set) replace loosely named tasks; comparisons are made only within a cell. HMC is restricted to sleep staging from ECG alone and EEG alone; it has no respiration or PPG and takes no part in the cardiorespiratory fusion experiment. The primary cells are cross-dataset sleep staging from ECG + respiration and stress from chest ECG + respiration. EEG stays first-class: in the format and harness from Milestone 1, pretrained unimodally in Stage 2, joined to cross-modal training right after Stage 3 | 06 §3.2; 03 §5b, §7; 07 Stages 0, 2, 3, §5 |
| 7 | Compute | All figures relabelled as hypotheses. Memory model now itemises student, teacher, prediction head, gradients and optimiser state (≈ 0.27 GB fixed) and activations for 0–32 EEG channels (≈ 3.6–14 GB at batch 256 in mixed precision). Throughput hypothesis revised downward to 0.5–2 thousand windows/s. A resource benchmark with decision rules is specified and inserted as Stage 0b; a data-layer benchmark is part of Milestone 1 | 04 §4.6; 07 §2, Stage 0b; 08 §10, §14 |

## 2. Conclusions that changed

* The lag term is no longer described as handling ECG–PPG delay. That depends on token content or on the pairwise module, to be decided on synthetic data.
* The hypothesised cost of the decisive experiment rises from "about two GPU-weeks" to 4–17 GPU-days for its first tier of 15 runs, pending measurement.
* MIMIC-III is no longer an acceptable substitute corpus for the lag-sensitive experiment.
* Everything else stands: Architecture B as primary, A as fallback, objective O1 with O2 as fallback, and controlled evidence on what cross-modal pretraining learns as the main contribution.

## 3. Remaining methodological blockers

None for Milestone 1. Before later stages:

| Blocker | Affects | Resolution |
|---|---|---|
| Access to MESA Sleep. No open corpus offers ECG, PPG and respiratory effort together with trustworthy timing | Stage 3 | NSRR request |
| MESA's oximeter latency and possible auto-gain are undocumented | beat-level analyses; PPG amplitude descriptors | synchronisation audit on the data; treat absolute delays as uninterpretable |
| Whether enough stable, rate-matched windows exist for the within-state swap | K4 arm 5b | count eligible windows in the MESA audit; relax matching tolerances if needed, and report them |
| Whether sub-second delay survives the fusion bottleneck | fusion configuration | Stage 1 delay-resolution test |
| Actual hardware and its throughput | every budget | Stage 0b |
| WESAD's synchronisation method and file layout were not confirmed from a primary source | Milestone 1 audit | read the dataset readme during the audit |
| Atrial-fibrillation label in PTB-XL not confirmed | T4 | check the statement table |

## 4. Is Milestone 1 ready to implement?

Yes. It depends only on three open datasets and CPU code, and its specification now contains the input policy, the format fields and the synchronisation audit that the review asked for.

## 5. Corrected scope of Milestone 1

In scope:
1. Unified format with raw files untouched and referenced by hash; canonical arrays without stored normalisation; original rates, timestamp origin, clock groups, synchronisation method, class, estimated offset and drift, gap intervals and provenance in the metadata.
2. Adapters for WESAD, PPG-DaLiA and HMC.
3. Synchronisation audit: class from documentation plus an empirical beat-series check across devices.
4. Five tasks with enforced input policy: stress from the chest device; stress from the wrist device; heart rate from wrist BVP and wrist accelerometer; sleep staging from ECG alone; sleep staging from EEG alone. Declared variants as separate rows.
5. Hand-crafted features including scale descriptors and within-device coupling indices; random-projection control; classical heart-rate trackers; linear and gradient-boosted probes.
6. Participant-disjoint evaluation, permutation null, bootstrap intervals, paired tests, minimum detectable effect, label-efficiency curves, overlap registry.
7. Data-layer benchmark: preprocessing speed, bytes per recording-hour, window-loading throughput.
8. Reports: audit, baselines, data benchmark.

Out of scope: neural networks, pretraining, public checkpoints, the model resource benchmark, synthetic coupling generators, any additional dataset.

## 6. Questions deferred to later milestones

| Question | Decided in |
|---|---|
| Model throughput and memory; final pretraining budget | Milestone 1b (Stage 0b) |
| Fusion-token rate; whether the pairwise lag module joins Architecture B | Milestone 2 (Stage 1) |
| Whether scale-descriptor tokens and which normalisation mode help | Stage 2 |
| Modality-specific versus shared encoders; O1 versus O2 | Stage 2 |
| Cross-modal versus unimodal pretraining; interaction fusion versus controls; pairing perturbations | Stage 3 |
| EEG in cross-modal training; brain–body coupling in sleep | immediately after Stage 3 |
| Signal-processing extensions (phase encoding, physical front ends, Riemannian EEG token, coupling targets) | Stage 4 |
| EDA stream and wristband pretraining data; wake-state transfer; scale-up | Stage 5 |
| Reliability-aware gating under corruption | Stage 6 |
| Overlap between public checkpoints' pretraining data and our evaluation cells | when a checkpoint is first used as a reference |
