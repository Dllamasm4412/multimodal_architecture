# 05 — Self-Supervised Learning Strategy

Which pretraining objective to use for Architecture B (report 04), why, and what could go wrong. Model facts cited here are documented, with verification status, in `02_FOUNDATION_MODEL_PRIOR_ART.md`. **[analysis]** marks my reasoning.

## 1. Recommendation

| Role | Objective | In one line |
|---|---|---|
| **Primary — O1** | **Masked latent prediction within and across modalities**, with a slowly updated teacher | Predict the *representation* of hidden signal, using the same modality's context and the other modalities |
| **Fallback — O2** | **Masked reconstruction of fixed signal descriptors with whole-modality masking** | Predict a deterministic target (spectrum or waveform) of hidden signal; no teacher, cannot collapse |
| Baseline to beat | Cross-modal contrastive alignment | The approach of the large multimodal sleep and wearable models |
| Exploratory add-on | Cross-modal time-shift prediction | Predict the offset artificially applied between two modalities |

O1 is primary because the project's evaluation centres on *frozen* representations with few labels, and the available evidence, though not controlled across papers, points the same way: models trained to predict codes or samples tend to need full fine-tuning, while models trained to predict latent representations report usable frozen features (section 3). O2 is the fallback because it is simpler, has no collapse mode, and decides Stage 1 and Stage 2 if O1 proves unstable.

## 2. What each objective family rewards

| # | Objective | The encoder is rewarded for | Typical failure | Physiological examples (report 02) |
|---|---|---|---|---|
| 1 | Masked waveform reconstruction | Reproducing samples, including noise and sensor detail | Local interpolation; copying neighbouring cycles; features that need fine-tuning to be useful | CBraMod, LUNA, REVE, FEMBA (EEG); xMAE (ECG from PPG) |
| 2 | Masked code or token prediction | Predicting a discrete summary from a separately trained tokenizer | Two-stage training; tokenizer quality caps the model; weak linear probes | LaBraM (spectral codes), HeartLang (heartbeat vocabulary), HuBERT-ECG |
| 3 | Masked latent prediction with a teacher (joint-embedding predictive) | Predicting abstract content that is predictable from context; unpredictable detail is ignored | Representational collapse; targets that encode position only | EEGPT, ECG-JEPA, CardioState-JEPA |
| 4 | Contrastive, augmentation-based | Invariance to the chosen augmentations; discrimination between instances | Encodes whatever separates instances most easily, usually participant identity; needs large batches | BENDR, BIOT, Apple wearable model |
| 5 | Cross-modal reconstruction | Information in one modality that determines another | Dominated by the easiest pair; ignores modality-unique content | xMAE, PhysioME |
| 6 | Cross-modal latent prediction | As 5, at the level of abstract content | As 3 and 5 | CardioState-JEPA |
| 7 | Cross-modal contrastive alignment | Information *shared* by all aligned modalities; everything else is discarded | Loss of modality-specific information; slow-state or identity shortcuts | SleepFM, Brant-X, SA-CLIP |
| 8 | Temporal forecasting | Predictable dynamics | Trivial for quasi-periodic signals; equivalent to span masking at the window end | — |
| 9 | Teacher–student distillation | Whatever the teacher encodes | Inherits the teacher's biases; needs a good teacher | used inside 3; cross-modal distillation from a rich to a poor modality |

**State versus trait [analysis].** The objective determines whether the embedding describes *who* is recorded or *what state* they are in. Contrastive learning with participant-level positive pairs, as used in the Apple wearable model, deliberately encodes stable individual characteristics, which suits health-condition prediction. Our downstream tasks are states that vary within a person (sleep stage, stress, arousal). Objectives that predict hidden content inside a window from its context have no incentive to separate participants and are the better match. This is checked, not assumed, by the identity probe (report 06, T7).

## 3. Evidence that bears on the choice

* **Frozen features from code and sample prediction are weak for EEG.** LaBraM's own ablation shows linear probing far below full fine-tuning (balanced accuracy 0.346 versus 0.641 on one task). Independent evaluations of EEG foundation models report that linear probing "severely underperforms" full tuning, that gains over small specialist networks are around one percentage point, and that the benefit appears mainly on long-context tasks such as sleep staging (EEG-FM-Compass 2026; Lee et al. 2025; Kommineni et al. 2026).
* **Latent-prediction models report frozen evaluation as their main protocol:** EEGPT (linear probing), CardioState-JEPA (frozen encoder, 25 tasks). These claims are the authors' own and partly verified at abstract level only.
* **Contrastive multimodal sleep pretraining gives strong linear probes on sleep tasks** (SleepFM: 0.88 versus 0.72 AUROC for a supervised CNN). Cross-modal correspondence is therefore a useful training signal in this domain. What it does not show is which aspect of the correspondence matters.
* **A counter-example for cardiac signals.** xMAE, a reconstruction objective, reports good frozen linear probes for PPG under participant-level splits. The weakness of frozen features after reconstruction may be specific to EEG, where samples are barely predictable (section 4). This is one reason O2 is a credible fallback rather than a straw man.
* **Masked reconstruction is not uniformly good.** The EDA model accompanying the EDAMAME corpus reports that a masked-reconstruction variant failed where a contrastive one worked (snippet level). An informal ECG–PPG latent-prediction project reports an interpolation shortcut that disappeared only at a 75 % mask ratio.
* **Outside physiology,** predicting latent or hand-crafted targets rather than pixels improves frozen-feature quality (data2vec, I-JEPA, masked feature prediction).

None of this is a controlled comparison on our signals; Stage 2 runs one.

## 4. The interpolation problem, by modality [analysis]

Whether masked prediction teaches anything depends on how predictable the hidden part is from its surroundings.

| Modality | Predictability of hidden samples | Consequence |
|---|---|---|
| ECG, PPG | Very high: the next beat looks like the last | Reconstructing samples is solved by copying a neighbouring cycle. What is *not* copyable, and is physiologically meaningful, is beat timing and beat-to-beat change. Use long masked spans (several beats), high mask ratios, and cross-modal context |
| Respiration | High over one breath, moderate beyond | Mask whole breaths |
| EDA | Slow drift is trivially interpolated; responses are sparse events | Mask on the deconvolved driver, where the content is; otherwise the loss is dominated by drift |
| EEG | Very low at sample level: phase is essentially unpredictable; band power and transient patterns are partly predictable | Sample reconstruction spends capacity on noise. Predict latents or phase-free spectra |

This is the specific reason cross-modal prediction is attractive for cardiac signals: with ECG hidden and PPG visible, the timing of every hidden beat becomes predictable, but only through the physiological relationship between the two signals.

## 5. Primary objective O1 in detail

**Components.**
* *Student:* tokenisers, private encoders, fusion tokens, and a small prediction head (two Transformer layers, width 128), which is discarded after pretraining.
* *Teacher:* an exponential-moving-average copy of each **private** encoder and its tokeniser. The teacher has no fusion stage and sees one modality at a time, complete and unmasked.

**Targets.** For every token of modality *m*, the average of the teacher's top three layer outputs, normalised per token. Because the teacher is unimodal, a target never contains information from another modality; predicting it from another modality therefore requires real cross-modal inference.

**Masking, applied to the student input.**
1. *Span masking within a modality:* contiguous spans of 2–8 s covering about 60 % of each visible modality's tokens; for EEG, also whole-channel masking.
2. *Whole-modality masking:* each modality is hidden entirely with probability 0.3, at least one remaining visible.
3. *Real gaps:* signal segments that are genuinely missing or flagged as invalid are treated as masked and excluded from the loss, so recordings with absent sensors are used as they are.

**Prediction.** For each masked token, a query built from the modality embedding and the token's timestamp attends to (a) the visible tokens of the same modality, if any, and (b) the fusion tokens. Two loss terms result:
* *within-modality:* target tokens whose modality is partly visible;
* *cross-modal:* target tokens whose modality is entirely hidden, predicted only through the fusion tokens.

**Loss.** Smooth-L1 between prediction and target, averaged per modality and per term so that no modality or pair dominates by token count:

  L = Σ_m [ L_within(m) + λ · L_cross(m) ],  λ = 1 initially.

**Where timing enters.** Queries name the target time; the fusion layer's learned lag term (report 04, section 4.4) gives each modality a typical temporal offset at the scale of seconds. It cannot resolve delays finer than the token spacing or delays that change from window to window: for ECG–PPG, the objective can be solved only if beat timing is carried in token content through the fusion tokens, or by the explicit pairwise lag module. Which of these is needed is decided on synthetic data (report 07, Stage 1). No delay is hard-coded.

**Inputs that are never targets.** Scale-descriptor tokens (report 04, section 2) are inputs only. Physiological-phase encodings and pairwise lag profiles derived from a modality are removed whenever that modality is masked, so they cannot leak a hidden signal into the student.

**Starting hyper-parameters.** Teacher momentum 0.996 rising to 0.9999; AdamW, peak learning rate 3·10⁻⁴, weight decay 0.05, 5 % warm-up, cosine decay; batch 256; mixed precision.

## 6. Fallback objective O2 in detail

Same masking scheme, no teacher. Targets are deterministic descriptors of each hidden token:

| Modality | Target |
|---|---|
| EEG | log-magnitude spectrum of the patch, 1–45 Hz |
| ECG, PPG | the band-limited, window-normalised waveform patch and its first derivative |
| Respiration | waveform patch |
| EDA | deconvolved phasic driver and tonic level |

Loss: L1. With whole-modality masking this is cross-modal reconstruction, and it is the form in which signal-processing descriptors (scattering coefficients, coupling indices) can be added as auxiliary targets (report 01, candidate 5).

**When O2 replaces O1:** O1 collapses or is unstable across seeds; O1's embeddings fail the synthetic coupling probes that O2 passes; or the Stage 2 comparison shows no frozen-probe advantage for O1.

## 7. Is cross-modal prediction worthwhile?

**Arguments for [analysis].**
* It supplies a training signal that unimodal masking cannot: content that is unpredictable within a modality but determined by another.
* It can improve *unimodal* use. A PPG encoder trained to predict EEG-derived or ECG-derived latents is pushed to extract whatever in the pulse wave relates to those systems. This is the mechanism by which lab-grade modalities could improve wearable-grade encoders.
* It trains the model for missing modalities as a by-product.

**Arguments against.**
* The relationships may be learnable only for tightly coupled pairs (ECG–PPG), where two recent papers already work (xMAE, CardioState-JEPA), and may add nothing for loosely coupled systems (EEG–autonomic).
* The gain may come from shared slow state (both signals know the sleep stage) rather than from coupling, in which case simple window-level alignment would do as well.
* The cross-modal loss competes with the within-modality loss for capacity.

**How it is decided.** Comparisons K2 and K4 (report 06): cross-modal versus unimodal pretraining at matched compute and matched number of targets, and true pairing versus within-window rotation and within-state swap of the partner modality. The prior-art search found no published study that runs such a control as a main experiment.

## 8. Critical questions

| Question | Answer |
|---|---|
| Can encoders be pretrained independently before joint training? | Yes, and this is the recommended curriculum: unimodal O1 on every recording that contains the modality, then joint training on synchronised recordings with the private encoders initialised from the first phase. CardioState-JEPA reports the same order. The comparison K2 holds total steps equal between "unimodal only" and "unimodal then joint" |
| Can cross-modal pretraining improve unimodal downstream performance? | Plausible (section 7); tested by probing each private stream alone after both kinds of pretraining |
| How to learn from partially overlapping modality sets? | Every loss term is defined per modality or per co-present pair and is simply absent when its inputs are. Batches are sampled to balance modality *pairs*, not datasets. No relationship is inferred for a pair that never co-occurs (report 03, section 5) |
| How to avoid participant and device identity dominating? | Within-window prediction with no cross-participant negatives; band-limiting and resampling to canonical rates; per-window robust normalisation; random gain, filtering and resampling augmentations; metadata dropout. Monitored by the identity probe. If identity still dominates, add within-recording negatives (other times of the same recording) rather than cross-participant ones |
| How should timing and lag enter the objective? | Through time-stamped queries and the fusion lag term for second-scale relations, and through token content or the pairwise lag module for sub-second ones (section 5). An explicit time-shift prediction task is an exploratory add-on. Random timing perturbation is *not* applied in the main arm because it would destroy the structure under study; it is used deliberately in the pairing-perturbation controls (report 06, section 8.2) and as a robustness arm for wearables whose devices do not share a clock. Lag-sensitive training uses only recordings whose synchronisation supports the timescale (report 03, section 5a) |
| How is collapse prevented? | Section 9 |
| Is a shared/private design beneficial? | Adopted structurally: private streams with their own losses, shared content confined to the fusion tokens. A loss-based factorisation (as in FOCAL) is an ablation, not the default, because orthogonality penalties add hyper-parameters without a clear physiological meaning |
| Which approaches have credible transfer evidence? | Frozen-probe transfer: contrastive multimodal sleep pretraining (SleepFM), morphology-aware PPG pretraining (PaPaGei, 20 tasks), latent prediction for cardiac signals (CardioState-JEPA). Fine-tuning transfer: masked EEG models, with small margins over specialists. Cross-dataset, participant-independent evidence is thin throughout |

## 9. Shortcuts and safeguards

| Risk | Safeguard | Diagnostic |
|---|---|---|
| Collapse of teacher and student to a constant or low-rank solution | Stop-gradient teacher with high momentum; asymmetric prediction head; per-token target normalisation; high mask ratio. If needed, a variance–covariance penalty on pooled student embeddings | Per-dimension standard deviation and effective rank of embeddings, logged every epoch; a small frozen-probe task evaluated during training |
| Targets that encode only position in the window | Teacher sees randomly cropped windows, so absolute position is uninformative | Probe for token index from the target |
| Copying neighbouring cycles | Long spans, 60 % or higher mask ratio; cross-modal term | Loss on masked spans versus span length |
| Loss dominated by the easiest pair | Per-pair averaging; per-pair loss curves; cap on any pair's share of the gradient | Per-pair loss and probe table |
| Shared artefacts taken for coupling (motion appears in every sensor) | Corruption augmentation applied independently per modality; quality head | Lag kernel concentrated at exactly zero lag for all pairs is a warning sign |
| Learning line noise, filter edges or device signatures | Canonical band-limiting and rates; augmentation | Device or dataset classification probe |
| Spectral statistics only | — | Surrogate-pretraining control (report 06) |
| Seed dependence | Three pretraining seeds per confirmatory arm | Across-seed spread reported |
| A privileged signal of an evaluation dataset seen in pretraining (for example the ECG that defines a PPG heart-rate target) | Evaluation datasets, including their unlabelled signals, are excluded from every pretraining phase | Manifest audit |
| Timing structure learned from recordings whose signals are not reliably aligned | Synchronisation class per dataset; lag-sensitive terms restricted accordingly | Stability of the estimated inter-signal offset within recordings |

## 10. Transfer to downstream tasks

* **Exposed representations.** Per-second shared tokens; pooled private embedding per modality; their gated combination.
* **Adaptation ladder.** Linear probe → attentive probe → low-rank adapters → full fine-tuning, with identical tuning budgets across models (report 06, section 4).
* **New modality or sensor at fine-tuning time.** Add a tokeniser and a shallow private encoder for it; train those and the "absent" flags while the rest is frozen.
* **Reused public encoders.** A frozen external encoder can stand in for a private stream through a linear adapter to width 192 and synthetic timestamps for its tokens; the fusion stage is then trained with O1 on top. This is how a multichannel EEG model is attached for tasks where sleep-montage pretraining is inadequate.
* **Long recordings.** Window embeddings are computed once and cached; sequence heads operate on the cache.
