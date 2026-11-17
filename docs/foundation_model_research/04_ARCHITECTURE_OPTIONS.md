# 04 — Architecture Options

Three competing designs for transferable multimodal physiological representation learning, sized for one 24 GB GPU. Parameter counts are computed from the stated layer sizes; throughput and memory figures are estimates to be measured. Objectives O1 and O2 are specified in `05_SELF_SUPERVISED_LEARNING_STRATEGY.md`; datasets in `03_DATASET_AND_PRETRAINING_STRATEGY.md`; checkpoint reuse in `02_FOUNDATION_MODEL_PRIOR_ART.md`, section 5.

## 1. Recommendation

| Role | Design | Size |
|---|---|---|
| **Primary** | **Architecture B**: modality-specific tokenisers and encoders that stay private, joined by a small set of time-stamped fusion tokens with lag-aware cross-attention; pretrained with masked latent prediction within and across modalities (O1) | ≈ 13 M |
| **Fallback** | **Architecture A**: one compact 1-D residual encoder per modality, unimodal masked reconstruction (O2), gated late fusion | ≈ 4–7 M |
| Conditional extensions | **Architecture C**: B plus up to three signal-processing components, each admitted only by its own ablation | ≈ +0.3 M |

A is not a formality. It is the model against which B must demonstrate, at equal data and compute, that cross-modal pretraining and interaction fusion are worth their complexity. If B does not beat A on the confirmatory comparisons, A is the system to carry forward and the result is a controlled negative finding on cross-modal learning.

**Training strategy: hybrid.** Train compact encoders from scratch for the controlled experiments, because comparing unimodal with cross-modal pretraining requires pretraining the same encoders both ways. Reuse public checkpoints in two places: as frozen reference models in every table, and as the EEG stream when a downstream task has many EEG channels, since spatial EEG structure cannot be learned from two- or three-channel sleep recordings.

## 2. Conventions shared by all architectures

**Window.** 30 s for pretraining (one sleep-scoring epoch; long enough for six or more breaths). Downstream tasks use 30–60 s windows; longer contexts (sleep staging over minutes) are handled by a small sequence head over window embeddings, outside the pretrained model.

**Sampling rates and filtering.** Each modality is band-limited to a common physiological band and resampled to one canonical rate, so that device-specific filter edges and rates do not reach the model.

| Modality | Band | Canonical rate | Reason |
|---|---|---|---|
| EEG | 0.3–45 Hz | 128 Hz | covers sleep and wake rhythms including slow oscillations; within every candidate dataset's native rate |
| ECG (single lead) | 0.5–40 Hz | 128 Hz | QRS shape and 8 ms timing resolution |
| PPG / BVP | 0.1–8 Hz | 64 Hz | pulse morphology, plus the breathing-related baseline modulation that a 0.5 Hz cut-off would delete; equals the native rate of common wristbands |
| Respiration (effort belt or airflow) | 0.05–1 Hz | 10 Hz | breathing band |
| EDA | < 1 Hz | 4 Hz | native rate of common wristbands |

**Amplitude: normalised waveform plus a scale descriptor.** Each window and channel is robustly z-scored (median, inter-quartile range) for the tokeniser, which needs no statistic from other recordings. Normalisation alone would delete physiology that downstream tasks use: slow-wave amplitude in sleep EEG, pulse-amplitude change with vasoconstriction, breathing depth, tonic skin-conductance level. What normalisation removes is therefore kept in a compact **scale descriptor** per window and channel. It enters as one extra token per modality (small MLP to width 192), is masked together with its modality, and is randomly dropped (p = 0.5) in pretraining so the model also works without it.

| Modality | Scale-descriptor contents |
|---|---|
| all | log inter-quartile range and median in native units; a flag saying whether native units are physical and comparable across devices; fraction of invalid, flat or clipped samples; a signal-quality index |
| EEG | log absolute band-power in five bands per channel |
| ECG | log R-wave amplitude |
| PPG | log pulse amplitude; ratio of slow (0.1–0.5 Hz) to pulsatile power |
| Respiration | log breath amplitude |
| EDA | tonic level (log), slope |

Amplitudes in arbitrary or auto-gained units (most PPG and respiration channels) are flagged as not comparable; for those, only within-recording changes are meaningful. Three normalisation modes are distinguished and never mixed in one results table: per-window (default; inductive); causal running statistics over the preceding five minutes (inductive, deployable); whole-recording statistics (label-free calibration, transductive). Whether the descriptor helps is an ablation (with versus without), not an assumption.

**Raw data.** Original files are never modified. Everything above is derived, versioned and reproducible from them (report 08, format).

**Time.** Every token carries its centre time in seconds from the window start. Nothing in the model assumes two modalities share a token grid. Token times are only as good as the recording's synchronisation: each dataset is assigned a synchronisation class (report 03, section 5a), and lag-sensitive components and experiments are restricted to the classes that support them.

**Metadata.** Modality embedding; optional sensor-site embedding (PPG: finger, wrist, ear; ECG: limb-lead-like, chest strap; respiration: thoracic, abdominal, airflow); EEG electrode position as 3-D coordinates through a small MLP. Every metadata field has an "unknown" value and is randomly replaced by it during pretraining, so the model works without metadata.

## 3. Architecture A — compact transferable baseline

```
ECG  [B,1,3840]  ─ ResNet1D ──► z_ecg  [B,256]
PPG  [B,3,1920]  ─ ResNet1D ──► z_ppg  [B,256]     (PPG + 1st + 2nd derivative)
RESP [B,1,300]   ─ ResNet1D-s ► z_resp [B,256]
EDA  [B,2,120]   ─ ResNet1D-s ► z_eda  [B,256]     (phasic + tonic)
EEG  [B,C,3840]  ─ ResNet1D per channel, mean over channels ─► z_eeg [B,256]
                         │
        masked gate: g = softmax(score(z_m)) over present modalities
                         ▼
                   z = Σ g_m z_m ──► task head
```

* **Encoders.** Four residual stages (widths 32–64–128–192, kernel 5–7, stride 2 per stage); global average pooling; linear projection to 256. ≈ 1–2 M parameters for EEG, ECG and PPG each; ≈ 0.3 M for the slow signals. Total ≈ 4–7 M.
* **Pretraining.** Unimodal masked reconstruction (O2) per encoder with a light convolutional decoder that is discarded afterwards. No cross-modal term.
* **Fusion.** Learned at fine-tuning only; gate masked for missing modalities; modality dropout during fine-tuning.
* **Fine-tuning.** Linear probe on concatenated or gated embeddings; optional full fine-tuning.
* **Compute.** Expected to be several times cheaper than B; measured in the same resource benchmark (section 4.6).
* **Weaknesses.** No token-level cross-modal information; pooled embeddings discard relative timing; channel averaging ignores EEG spatial structure.
* **Rationale.** Convolutional encoders remain strong for cardiac signals (several public ECG and PPG models are convolutional; report 02). If this model with simple pretraining already delivers the label-efficiency gain, the added machinery of B has to justify itself against it.

## 4. Architecture B — modality-specific encoders with time-aware bottleneck fusion (primary)

### 4.1 Block diagram

```
            tokeniser T_m            private encoder E_m           
EEG  ──► per-channel 1 s patches ──► Transformer (6 layers) ──► H_eeg  [B, C·30, 192]
ECG  ──► 0.25 s patches          ──► Transformer (6 layers) ──► H_ecg  [B, 120, 192]
PPG  ──► 0.25 s patches          ──► Transformer (6 layers) ──► H_ppg  [B, 120, 192]
RESP ──► 1 s patches             ──► Transformer (3 layers) ──► H_resp [B, 30, 192]
EDA  ──► 2 s patches             ──► Transformer (3 layers) ──► H_eda  [B, 15, 192]
                                           │  (each stream usable alone)
                                           ▼
        fusion tokens U [B, 30, 192], one per second, each with its own timestamp
        3 fusion layers:  U ← U + CrossAttn(U → all present modality tokens,
                                             bias = b_m(t_key − t_query))
                          U ← U + SelfAttn(U);  U ← U + MLP(U)
                                           │
              ┌────────────────────────────┴───────────────────────────┐
     shared output U (per second)                      private outputs pool(H_m)
              └──────────────► task head (probe / adapter) ◄───────────┘
```

### 4.2 Modality-specific tokenisers (Priority A)

| Modality | Input per 30 s | Token | Tokeniser | Physiological reason |
|---|---|---|---|---|
| **EEG** | [C, 3840] | 1 s per channel → C × 30 tokens | Two paths summed: (i) multi-scale 1-D convolutions on the raw patch (kernels ≈ 16, 32, 64 samples); (ii) linear map of the patch's log-magnitude spectrum, 1–45 Hz. Plus electrode-position encoding | EEG information is mainly band power and its topography; raw path keeps transients (spindles, K-complexes, artefacts). Channel-wise tokens make the encoder independent of montage |
| **ECG** | [1, 3840] | 0.25 s → 120 tokens | Multi-scale convolutions (≈ 30, 100, 300 ms) then linear patch projection, which keeps position inside the patch | QRS (tens of ms), P and T waves (hundreds of ms); beat timing must survive to the fusion stage at sub-second resolution |
| **PPG / BVP** | [3, 1920] | 0.25 s → 120 tokens | Same design as ECG on the signal and its first two derivatives; optional gain control | Pulse upstroke and notch are derivative features; amplitude is in arbitrary units and device-dependent |
| **Respiration** | [1–2, 300] | 1 s → 30 tokens | Single-scale convolution + linear projection | Slow cyclic signal; breath phase and depth matter, fine shape does not |
| **EDA** | [2, 120] + level scalar | 2 s → 15 tokens | Sparse deconvolution into phasic driver and tonic component offline, then linear projection | Skin conductance is sparse nerve bursts through a slow impulse response; the driver is the physiologically meaningful signal and is nearly sensor-independent |

Token duration is the one place where modalities are treated most differently, and it follows from the signal: a token should hold less than one cycle or event, so that cycles are sequences rather than points.

### 4.3 Private encoders

Pre-norm Transformer, width 192, 3 heads, feed-forward width 768, continuous-time positional encoding. About 0.445 M parameters per layer.

| Stream | Layers | Parameters |
|---|---|---|
| EEG | 6, with attention factorised into within-channel (temporal) and within-second (spatial) steps | ≈ 2.9 M |
| ECG | 6 | ≈ 2.8 M |
| PPG | 6 | ≈ 2.8 M |
| Respiration | 3 | ≈ 1.4 M |
| EDA | 3 | ≈ 1.4 M |

**Why Transformers here and not state-space models or deeper CNNs.** After patching, sequences are 15–120 tokens per stream; attention cost is negligible and masking is trivial to implement. State-space models pay off on long raw sequences, which patching removes. Convolution is kept where its inductive bias matters, in the tokeniser.

**What is shared.** Architecture and hyper-parameters are shared across streams; weights are not. Whether weights of the upper layers can be shared between ECG and PPG, or across all streams, is comparison K5, not an assumption. Published models go both ways (one shared Transformer with light tokenisers in BIOT and CardioState-JEPA; separate encoders in PhysioOmni, CroSSL and FOCAL), and no located paper compares them at matched size.

**Preserving modality-specific information.** Streams are never projected into a common aligned space and no loss forces H_ecg to resemble H_ppg. Width 192 is shared only so that the fusion layer can read all streams. Each stream has its own within-modality prediction loss and can be probed and used alone.

### 4.4 Fusion (Priority B)

**Mechanism.** Thirty learned fusion tokens, one per second, each with a timestamp. In every fusion layer they cross-attend to all tokens of all present modalities, then self-attend. The attention logit between fusion token *q* and modality token *k* is

  logit(q, k) = ⟨W_q u_q, W_k h_k⟩ / √d + b_{m(k), head}(t_k − t_q)

where *b* is a small learned function of the time difference (an MLP on sinusoidal features of the lag), separate for each modality and head. After training, *b* can be plotted as a kernel per modality; what such a plot does and does not mean is stated below.

**Why this design.**
* All cross-modal exchange passes through 30 tokens. What is shared is therefore measurable, and private streams are untouched: the shared/private split is structural rather than imposed by a loss.
* Cost is linear in the number of modality tokens, and any subset of modalities is handled by leaving absent tokens out of the key set.
* Different token rates need no alignment: a one-second fusion token can read four ECG tokens, one respiration token and half an EDA token, each at its own lag.
* The lag term gives typical second-scale delays an explicit place, instead of forcing synchrony. Its limits are analysed below.

**Relation to the closest prior work (report 02, section 3.6).** CardioState-JEPA predicts one bounded delay per token and trains it against physiological anchors such as pulse arrival time; it uses a single shared Transformer and is evaluated one modality at a time. xMAE lets ECG queries attend to PPG without any lag term and keeps only the PPG encoder. The design here differs in three ways: the lag is a learned *distribution over time differences* in the attention, needing no anchor, so it applies to pairs with no well-defined delay (respiration, EEG, EDA), although it does not by itself resolve sub-second ECG–PPG delay; private encoders are kept separate from a small shared bottleneck; and the fused model is used, and evaluated, with any subset of modalities. These are extensions of existing ideas, not new principles; the contribution rests on the controlled comparisons.

**What the lag term can and cannot represent.**

*It is a per-modality temporal receptive field, not a pairwise coupling term.* The bias depends only on the key's modality and on the time difference between a fusion token and that key. It lets a fusion token at time τ read, for example, ECG around τ − 0.1 s and EDA around τ + 2 s. A lag *between two modalities* is represented only as the difference between their two kernels, and that difference is the same for every window and every person.

| The lag term can represent | It cannot represent |
|---|---|
| A population-average offset between modalities and its spread (a kernel with width) | A delay that varies between windows, beats or people; that has to come from content-based attention |
| Distributed, smooth lags of seconds (breathing to heart period; event to skin-conductance response) | Delays finer than the token spacing, because its argument is a difference of token-centre times |
| Lead or lag in either direction, several patterns per head | A physiological delay in absolute terms: the kernel absorbs sensor latency, filter delay and clock offset. Only differences between kernels within one acquisition system are interpretable |
| | Relations longer than the window |

*All cross-modal interaction is mediated.* No ECG token ever attends to a PPG token. A fusion token aggregates content from each stream; a relation between two streams exists only as something the fusion token's MLP, or a later layer's content-dependent query, computes from what was aggregated. Sample-specific alignment is possible from the second fusion layer on (layer 1 reads beat times from one stream; layer 2 queries the other stream conditioned on them), but nothing in the design makes it easy, and attention into a one-second token averages over one to two beats. The fusion tokens therefore hold a joint state; they are not a table of pairwise couplings, and the learned kernels are not measurements of physiological delay.

**Resolution for each coupling of interest.**

| Coupling | Typical lag | Variation of interest | Token spacing | Resolved by the lag term? | What must carry it |
|---|---|---|---|---|---|
| ECG → PPG pulse arrival | 150–400 ms | 5–30 ms | 250 ms in both streams | **No**: the whole delay is about one token wide | Position of the R-peak and of the pulse upstroke *inside* tokens, encoded in token content |
| Breathing → heart period and pulse amplitude | 0–2 s within a 3–5 s cycle | ≈ 0.25 s | 0.25 s cardiac, 1 s respiration | Partly | Lag term plus breath phase inside respiration tokens |
| EEG event → heart rate | 2–6 s | ≈ 1 s | 1 s and 0.25 s | Yes | Lag term |
| Event → skin-conductance response | 1–4 s | ≈ 0.5 s | 2 s EDA | Coarsely | Lag term plus response onset inside EDA tokens |

*Is the input resolution adequate for ECG–PPG delay?* Yes at the input, unproven at the output. Timing precision of a band-limited signal is set by bandwidth and noise, not by the sampling grid: ECG at 128 Hz locates R-peaks to a few milliseconds with interpolation, and a PPG upstroke band-limited to 8 Hz and sampled at 64 Hz can be located to roughly 5 ms at 20 dB signal-to-noise [analysis; to be confirmed on synthetic data]. The linear patch projection keeps within-patch position, so the information reaches the private encoders. Whether it survives the fusion bottleneck is not established: 250 ms tokens and one-second fusion tokens are adequate in principle for a window's *mean* delay and unproven for beat-level delay.

*Consequence.* This is a limit of the lag term, not a fundamental limit of the architecture, so no redesign is made. Three changes follow: the claim is corrected (the lag term serves second-scale coupling; ECG–PPG delay depends on token content); a deciding test is added; and an explicit pairwise control is specified.

**Pairwise lag-interaction module (control, and optional component).** For a chosen pair (a, b), take the two tokenisers' convolutional feature maps before patching, at a common fine rate, project each to 16 channels and normalise. For each fusion-token second s and lag ℓ compute C[s, ℓ] = mean over t in s of ⟨f_a(t), f_b(t + ℓ)⟩. A two-layer MLP maps each row C[s, ·] to width 192 and adds it to fusion token s, or, pooled over s, to the late-fusion vector.

| Pair | Feature rate | Lag range and step | Lags |
|---|---|---|---|
| ECG–PPG | 64 Hz | ±0.5 s in steps of 15.6 ms | 65 |
| Respiration–cardiac | 10 Hz | ±5 s in steps of 0.1 s | 101 |

About 17 k parameters per pair, under 1 % of the fusion stage. It represents exactly what the lag term cannot: an explicit, per-window, sub-token lag profile for one pair. Because it is small, parameter matching is exact: the arm without it is widened by the same count. It is used only between signals whose synchronisation class supports the lag scale (report 03, section 5a).

**Deciding test (Stage 1, synthetic cardiac–pulse generator).** Pulse arrival delay varies between windows (150–350 ms) and slowly within windows (±20 ms). From the frozen shared tokens a linear probe must recover the window-mean delay with mean absolute error ≤ 10 ms and follow within-window variation with correlation ≥ 0.5. Configurations are tried in this order: default (one-second fusion tokens); fusion tokens every 0.25 s; default plus the pairwise module. The first that passes is adopted. If only the pairwise module passes, it becomes part of B for tightly coupled pairs.

**Parameters.** 3 layers × (cross-attention + self-attention + MLP) ≈ 1.8 M. **Total B ≈ 13 M** plus a prediction head used only in pretraining (≈ 0.5 M) and a non-trainable teacher copy of the private encoders.

**Missing modalities.** Absent streams contribute no keys; a learned per-modality "absent" flag is added to the fusion tokens so the model knows what it is not seeing. Whole-modality masking during pretraining makes this the normal case, not an exception.

**Reliability-aware gating.** Each stream predicts a per-token quality score from H_m, trained without labels by injecting synthetic corruption during pretraining and predicting where it was injected. The score is added to the attention bias (low quality → less attention). Compared in report 06 against no quality input and against conventional quality indices.

**Fusion variants to compare (same encoders, same objective, matched parameters).**

| Variant | Description | Role |
|---|---|---|
| Late gated | pool each stream, gate, sum (as in A) | simplest alternative |
| **Bottleneck with lag bias** | section 4.4 | primary |
| Bottleneck without lag bias | same, *b* removed | tests lag modelling |
| Flat joint attention | all tokens in one sequence with self-attention | maximal interaction |
| Hierarchical | fusion tokens first read ECG + PPG + respiration; a second set reads those plus EEG and EDA; also a random grouping of the same shape | tests physiological grouping |
| **Late gated + pairwise lag module** | explicit pairwise lag profiles only, no token-level exchange; parameters matched to the bottleneck arm | **pairwise control**: is explicit coupling sufficient without mediated attention? |
| Bottleneck + pairwise lag module | both | adopted for tightly coupled pairs only if the Stage 1 resolution test requires it |
| Anchor-supervised delay | per-token delay predicted and trained against pulse arrival time, for ECG–PPG only | the published alternative; tests whether anchor-free lag learning loses anything where anchors exist |
| **No-exchange control** | identical layers, but each modality has its own fusion tokens that read only that modality; outputs gated | isolates cross-modal exchange from capacity |

Directional versus bidirectional attention: the bottleneck is bidirectional by construction (every modality writes to and, through the prediction head, reads from the shared tokens). A directional variant (only the cardiac streams write, EEG only reads, and the reverse) is an exploratory ablation for asymmetric coupling.

### 4.5 Self-supervised objective and fine-tuning

* **Pretraining.** O1: a student sees span-masked and modality-masked input; a slowly updated teacher copy of each private encoder sees the complete signal of its own modality; the student predicts the teacher's token representations at masked positions, within a modality and from other modalities through the fusion tokens. Details and collapse safeguards in report 05.
* **Outputs for transfer.** Per-second shared tokens U; pooled private embeddings; both exposed to probes separately and jointly.
* **Fine-tuning ladder.** Linear probe → attentive probe → low-rank adapters → full fine-tuning (report 06). Task heads: window classification or regression; per-second heads on U for event detection; a small sequence model over window embeddings for sleep staging.

### 4.6 Compute: hypotheses, and the benchmark that replaces them

Every figure in this section is a hypothesis derived from layer sizes. None has been measured, and none should be used for planning once the benchmark below has run.

**Tokens per window:** 315 + 30·C including fusion tokens, where C is the number of EEG channels (375 for C = 2; 1,275 for C = 32).

**Memory model.**

| Component | Estimate |
|---|---|
| Student weights (13.3 M + 0.5 M prediction head), 32-bit | 55 MB |
| Gradients | 55 MB |
| AdamW state (two moments) | 110 MB |
| Teacher weights (private encoders and tokenisers, ≈ 11.5 M), no gradients | 46 MB |
| **Fixed total, independent of batch size** | **≈ 0.27 GB** |
| Activations kept for back-propagation | ≈ 12 kB per token per layer at 32-bit (about 16 × width values), plus attention maps; this term dominates |
| Teacher forward pass | a few MB per window, transient |

| EEG channels | Token-layers per window | Per window, 32-bit | Per window, mixed precision | Batch 256, mixed precision |
|---|---|---|---|---|
| 0 (cardiorespiratory only) | ≈ 1,700 | ≈ 25 MB | ≈ 14 MB | ≈ 3.6 GB |
| 2–3 (sleep montage) | ≈ 2,000–2,200 | ≈ 30 MB | ≈ 17 MB | ≈ 4.5 GB |
| 6 | ≈ 2,700 | ≈ 39 MB | ≈ 21 MB | ≈ 5.5 GB |
| 14 | ≈ 4,200 | ≈ 56 MB | ≈ 31 MB | ≈ 8 GB |
| 32 | ≈ 7,400 | ≈ 96 MB | ≈ 53 MB | ≈ 14 GB |

These could be wrong by a factor of two in either direction. At 32-bit precision, batch 256 with 32 EEG channels would not fit in 24 GB. Dropping masked tokens from the student's private encoders, instead of replacing them with a mask token, should roughly halve the activation term.

**Throughput hypothesis.** About 4 G multiply–accumulate operations per training window at C = 2 (student forward and backward, teacher forward). At an effective 5–15 TFLOP/s this gives 0.5–2 thousand windows per second, and roughly a third of that at C = 32. This revises the earlier figure of 1–3 thousand downward. A prototype run of ≈ 50 M window-steps would then take 7–27 GPU-hours, and the first tier of the decisive experiment (15 runs) 4–17 GPU-days.

**Resource benchmark (run before any pretraining; takes minutes).** Synthetic tensors of the true shapes; a complete training step with student, teacher update, prediction head and AdamW.

| Measured | Sweep |
|---|---|
| Training throughput (windows/s) and peak accelerator memory | batch 32–512; EEG channels 0, 2, 3, 6, 14, 32; mixed precision on and off; masked tokens dropped or kept; with and without the pairwise module |
| Data-loading throughput (windows/s) from the preprocessed store on the disk that will be used | 0–16 loader workers; windows with 0, 2 and 32 EEG channels |
| Preprocessing speed (recording-hours per CPU-hour) and stored bytes per recording-hour | per dataset |

Output: one table, and a revised budget computed as steps = available GPU-hours × 3600 × measured throughput ÷ batch size. Decision rules:
* data loading must exceed 1.5 × training throughput, otherwise storage is fixed first;
* below 300 windows/s at C = 2, the schedule is shortened or the width reduced before anything is launched;
* a configuration whose peak memory exceeds 80 % of the device runs at a smaller batch with gradient accumulation;
* the number of pretraining seeds and arms in Stage 3 is set from the measured budget, in the priority order given in report 07.

The benchmark runs on whatever accelerator is actually available. Its result, not this section, sets the plan.

### 4.7 Anticipated weaknesses

* Assumes modalities share a clock to within the lag resolution being modelled. Wearable datasets with separately clocked devices violate this at the beat level, and nominal timestamps are not accepted as evidence of beat-level synchronisation.
* The lag term cannot resolve sub-second delays; ECG–PPG coupling depends on token content surviving the fusion bottleneck, which Stage 1 must demonstrate.
* The cross-modal loss may be dominated by the easiest pair (ECG ↔ PPG beat timing); loss terms need per-pair monitoring and balancing.
* Fusion tokens can ignore a weak or rarely present modality (EEG, EDA). Checked by the dominance analysis in report 06.
* EDA has little openly licensed pretraining data (about 3,800 h from 190 people; report 03); its encoder will be the least mature.
* At prototype scale the private encoders will not match large public unimodal models on unimodal benchmarks. The claim under test concerns cross-modal learning at matched scale, not unimodal state of the art.

## 5. Architecture C — signal-processing extensions of B

Each component is added alone to B and kept only if its own ablation succeeds (report 01, shortlist).

| Component | Change to B | Added parameters | Admission test |
|---|---|---|---|
| **Physiological-phase encoding** | Cardiac and respiratory phase (sin, cos) added to every token's positional features; masked whenever the source modality is masked | ≈ 2 k | Improves cross-modal probes and a primary task over B; fails with phases from another recording |
| **Hz-parametrised filter bank with gain control, or fixed scattering stem** | Replaces the convolutional part of the EEG, ECG and PPG tokenisers | −/+ few k | Smaller loss when probes are trained at one device or sampling rate and tested at another |
| **Riemannian EEG token** | For multichannel EEG at fine-tuning: tangent-space vector of the window's spatial covariance, projected to width 192 and appended to the EEG stream | montage-dependent, ≈ 0.1 M for 32 channels | Gain on multichannel EEG tasks over B and over the tangent-space classifier alone |
| **Coupling targets** | Auxiliary heads on U predict classical coupling indices (pulse arrival time, respiratory sinus arrhythmia amplitude) computed offline | ≈ 40 k | Improves transfer beyond O1 alone; otherwise the indices remain probes |

Higher-order spectral, recurrence and topological descriptors do not enter C as inputs. They are probes; promotion requires the two conditions stated in report 01, section 4.

```
B  ──►  + phase encoding           (side information on tokens)
   ──►  + physical front end       (tokeniser swap)
   ──►  + Riemannian EEG token     (extra EEG token, fine-tuning only)
   ──►  + coupling targets         (extra pretraining heads)
```

**Weakness.** Each addition is another degree of freedom for the researcher; hence one-at-a-time admission against B with the same confirmatory statistics.

## 6. Comparison

| | A | B | C |
|---|---|---|---|
| Encoders | 1-D residual networks | patch tokenisers + small Transformers | as B |
| Cross-modal learning in pretraining | none | masked latent prediction across modalities | as B, plus coupling targets |
| Fusion | late gated | time-stamped bottleneck with lag bias | as B, plus phase encodings |
| Missing modality | gate masking | native | native |
| Parameters | ≈ 4–7 M | ≈ 13 M | ≈ 13.3 M |
| Prototype pretraining (hypothesis) | few GPU-hours | 7–27 GPU-hours per run | as B plus offline feature extraction |
| Main risk | too little interaction to test the hypothesis | benefit may come from shared slow state, not coupling | accumulation of unproven parts |

## 7. Alternatives considered

* **One universal encoder for all modalities** (single trunk, common patching). Simpler and data-efficient if it works; it is the "shared" arm of K5 rather than a separate architecture.
* **Contrastive alignment of modality embeddings into one space** (the approach of several multimodal sleep and wearable models). Keeps only information shared by all modalities and requires large batches; kept as a baseline objective, not as the design.
* **Scaling up a single large Transformer over all raw samples.** Outside the compute budget and not needed to answer the research questions.
* **Building on frozen public encoders only.** Fast, and included as a reference system, but cannot answer whether cross-modal pretraining changes what unimodal encoders learn.
