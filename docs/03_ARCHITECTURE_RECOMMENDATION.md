# 03 — Architecture Recommendation

Three competing designs, sized for DEAP (32 participants, 1,280 labelled trials). Parameter counts are computed from the layer sizes given and are approximate. Reasoning behind the choices is in `02_CRITICAL_ASSESSMENT.md`.

## 1. Recommendation

| Role | Design | Status |
|---|---|---|
| **Primary** | **Architecture B with gated fusion** ("B-gated"): modality-specific compact encoders, masked gated fusion, auxiliary unimodal heads, modality dropout. ≈ 40k parameters | Build after baselines |
| **Fallback** | **Architecture A**: per-modality hand-crafted features with a regularised linear model (A0), or its ≈ 6k-parameter neural counterpart (A1) | Build first; also the vehicle for all topology tests |
| Ablation only | B with one cross-attention exchange block ("B-xattn") | Adopt only if it beats a capacity-matched control |
| Conditional | **Architecture C** = B-gated + topology adapters (≈ 2k extra parameters) | Build only if topology passes the feasibility gate in Stage 3 |

If B-gated does not exceed A by the measured minimum detectable effect, A becomes the reference model for every later comparison. That outcome is likely enough that A should be treated as a real candidate, not a formality.

## 2. Shared conventions

**Unit of prediction.** One trial → one arousal estimate. Windows are tokens inside a trial; pooling over tokens happens inside the model, so no window ever receives its own label. Splits are made at participant level before any windowing.

**Inputs.** The 60 s stimulus period of each DEAP trial (128 Hz preprocessed release). The 3 s pre-trial segment is used only for baseline-relative features.

| Modality | Channels | Conditioning | Rate | Token window / hop | Tokens per trial |
|---|---|---|---|---|---|
| EEG | 32 | as released (4–45 Hz, average reference) | 128 Hz | 4 s / 2 s | 29 |
| BVP (plethysmograph) | 1 | band-pass 0.5–8 Hz | 32 Hz | 8 s / 4 s | 14 |
| EDA (GSR) + temperature | 1 + 1 | low-pass 1 Hz; temperature as mean and slope | 4 Hz | 20 s / 10 s | 5 |
| Respiration | 1 | band-pass 0.05–1 Hz | 8 Hz | 20 s / 10 s | 5 |

Token grids differ by modality because the signals' timescales differ; each token carries its centre time. EOG and EMG channels are **excluded from the main models** and used as a confound probe (how much arousal variance eye and facial-muscle activity explain).

**Normalisation.** Two regimes, always reported separately:
* *Strict* (primary): scaling constants from training participants only; amplitude-arbitrary signals (BVP, respiration, EDA) expressed relative to the same trial's pre-trial segment.
* *Calibrated* (secondary): features additionally standardised per participant using that participant's own unlabeled trials. Label-free but transductive.

**Target and loss.** Arousal rating, z-scored within each training participant; Huber loss. Valence is a secondary target trained the same way. Test-participant labels are never used.

## 3. Architecture A — compact baseline

**A0, classical.** Trial-level features: EEG log band-power (32 channels × 4 bands = 128); BVP heart rate, rate slope, short-term variability, pulse amplitude; EDA level, slope, response count and prominence; respiration rate, depth and regularity; temperature mean and slope (≈ 20 peripheral features). Standardise → ridge or L2 logistic regression. Fusion by feature concatenation, and alternatively by averaging standardised unimodal predictions. ≈ 150 coefficients. Trains in seconds.

**A1, compact neural.** Same information per window, pooled by the network:

| Stream | Input tensor | Encoder | Output |
|---|---|---|---|
| EEG | [B, 29, 160] (32 ch × 5 bands, log power) | Linear 160→32, GELU, dropout | mean over tokens → 32 |
| BVP | [B, 14, 6] beat features | Linear 6→16 | mean → 16 |
| EDA + temp | [B, 5, 6] | Linear 6→16 | mean → 16 |
| Respiration | [B, 5, 5] | Linear 5→16 | mean → 16 |
| Fusion | concat → 80 | Linear 80→1 | arousal |

≈ 5.6k parameters. A literature anchor on raw EEG (EEGNet, a few thousand parameters; Lawhern et al. 2018) is run in the same harness.

*Advantages:* lowest variance; fastest; interpretable; every later claim can be tested against it. *Risks:* fixed band edges; no learned temporal structure; concatenation has no mechanism for a missing modality (handled by imputing training means, which is the baseline to beat in robustness tests).

## 4. Architecture B — modality-aware encoders with coordinated fusion

### 4.1 Encoders

| Modality | Input tensor | Window encoder | Temporal stage | Output | Params |
|---|---|---|---|---|---|
| EEG | [B, 29, 32, 5] log band-power | flatten → Linear 160→64, GELU, dropout 0.5, Linear 64→32 | 1 Transformer layer (d = 32, 2 heads, FF 64) + attention pooling | z_eeg ∈ R³² | ≈ 21k |
| BVP | [B, 14, 256] waveform | Conv1d 1→8 (k7, s2) → 8→16 (k5, s2) → 16→32 (k3, s2), norm, global average | same Transformer layer design + attention pooling | z_bvp ∈ R³² | ≈ 11k |
| EDA + temp | [B, 5, 80] waveform + [B, 5, 2] | same 3-layer Conv1d (k9, k5, k3); Linear 2→32 for temperature, summed | attention pooling only | z_eda ∈ R³² | ≈ 2.5k |
| Respiration | [B, 5, 160] waveform | same 3-layer Conv1d | attention pooling only | z_resp ∈ R³² | ≈ 2.4k |

Design reasoning:
* **EEG is spectral, peripheral signals are temporal.** EEG information for affect is carried by band power and its topography; BVP, EDA and respiration carry it in waveform shape and event timing. This is the "different mathematical representation per modality" hypothesis in its simplest testable form.
* **The EEG front end is a variable, not a commitment.** Stage 4 compares, at matched size: log band-power (default); a 1 Hz-resolution STFT log-spectrogram with a small 2-D convolution; a Morlet scalogram; a learnable filter bank on raw EEG (EEGNet-style); and an input-conditioned band weighting (squeeze-and-excitation over frequency bins), which is the FFTNet idea applied where it is physically meaningful. The simplest variant within one detectable effect of the best is kept.
* **Slow modalities get no sequence model.** Five tokens do not justify one.

### 4.2 Fusion (default: gated)

```
z_m ∈ R^32 for each available modality m ∈ {eeg, bvp, eda, resp}
s_m = MLP_gate(z_m + e_m)            # shared 32→16→1, e_m = learned modality embedding
g   = softmax(s over available m)    # missing modalities are masked out
z   = Σ_m g_m · z_m
ŷ   = MLP_head(z)                    # 32→16→1
ŷ_m = MLP_head_m(z_m)                # one auxiliary head per modality
```

≈ 3.4k parameters. Total B-gated ≈ **40k**.

**Objective.** L = Huber(ŷ, y) + λ · Σ_m Huber(ŷ_m, y), λ ∈ {0, 0.5, 1} chosen on inner validation participants. During training each modality is dropped with probability 0.25 (at least one kept), so the same model serves every modality subset at test time. These two elements, auxiliary unimodal supervision and training on incomplete inputs, are what CoRe-Sleep's ablations credit for its robustness.

**Not included by default:** CoRe-Sleep's alignment loss. It pulls unimodal representations of the same time window together, which is sensible when both modalities observe the same latent state (sleep stage in EEG and EOG). EEG and skin conductance during music videos are more plausibly complementary than redundant, so forcing agreement may remove useful modality-specific information. It is an exploratory ablation.

### 4.3 Cross-attention variant (ablation: B-xattn)

Before pooling, all 53 tokens (29 + 14 + 5 + 5) receive modality and time embeddings; one shared block lets each modality's tokens attend to the other modalities' tokens (d = 32, 2 heads, FF 64; ≈ 8.5k parameters shared, ≈ 34k if separate per modality). Total ≈ 49k–74k.

**Capacity-matched control:** the identical block with attention masked to the token's own modality. Parameter count and depth are the same; the only difference is whether information crosses modalities. Cross-attention is adopted only if it beats this control.

*Advantages of B:* representation matched to each signal; graceful handling of missing modalities; unimodal predictions available for diagnosis. *Risks:* about 30 parameters per labelled trial, so heavy regularisation and early stopping on held-out participants are essential; learned peripheral encoders may not beat beat-level features; gate weights can collapse onto one modality (monitored through the unimodal heads).

## 5. Architecture C — topology-enhanced model

C is B-gated plus pre-computed topological descriptors. It exists to test a hypothesis and is built only if Stage 3 shows the descriptors are stable, non-redundant and incrementally predictive on classical models.

### 5.1 Descriptors (computed offline, cached per participant, trial, window and parameter set)

**Lower-star H0** (waveform sublevel filtration; EDA also superlevel). After discarding pairs with persistence below a noise threshold (a fixed multiple of the window's median absolute first difference), 10 scalars: number of pairs; total, maximum, mean and standard deviation of persistence; median and 90th percentile; persistence entropy; Gravot inertia; mean birth.

**Delay-embedding Vietoris–Rips H0/H1.** Embedding dimension 3, delay ≈ one quarter of the modality's nominal period (fixed in advance; sensitivity grid d ∈ {2, 3, 5}, delay × {0.5, 1, 2}); at most 250 points per window (max-min subsampling if more). 8 scalars: maximum H1 persistence; ratio of the two largest H1 persistences (loop dominance); total H1 persistence; number of H1 classes above threshold; H1 persistence entropy; mean and maximum H0 death; total H0 persistence.

Persistence images, landscapes and learned vectorisations are deliberately not used at this stage: they add hundreds of input dimensions per channel on a dataset of 1,280 trials.

| Modality | Descriptors per token | Dim | Diagrams for all of DEAP | Estimated cost |
|---|---|---|---|---|
| BVP | lower-star + Rips | 18 | 17,920 Rips | minutes |
| Respiration | lower-star + Rips | 18 | 6,400 Rips | minutes |
| EDA | lower-star only | 10 | none | seconds |
| EEG | lower-star per channel [29, 32, 10] → shared Linear 10→8 → mean over channels | 8 | none | minutes |
| EEG, exploratory only | Rips per channel, or of the 32-node connectivity network | — | ≈ 614k per band (per channel) | CPU-hours per band |

Both amplitude-preserving and shape-only (per-window z-scored) versions are cached, because they answer different questions.

### 5.2 Where topology enters: at the modality token, before fusion

```
t_{m,w}  = descriptor vector of modality m, window w
z_{m,w} ← z_{m,w} + W_m · LayerNorm(t_{m,w})     # W_m initialised to zero
```

≈ 2k added parameters; total C ≈ **42k**. Reasons for this placement:

* **Before fusion, not after.** A descriptor belongs to its modality. If it were concatenated after fusion, a "missing" BVP channel would still leak BVP-derived topology into the prediction and the missing-modality experiments would be invalid. Attached to the modality token, it is dropped together with its parent signal.
* **After the window encoder, not as raw input.** Descriptors are order-free summaries of a window; they complement, rather than feed, the temporal encoder.
* **Zero-initialised.** C starts numerically identical to B, so any difference is attributable to the descriptors rather than to initialisation.

**Controls with identical structure and size:** (i) *placebo* — the adapter receives the same number of non-topological nonlinear features (Hjorth parameters, entropies, line length); (ii) *surrogate* — descriptors computed from IAAFT surrogates of the same windows. C is credited only with what it achieves beyond both.

**Quality-gating variant (C-Q, separate hypothesis).** The loop-dominance score of BVP and respiration windows, averaged per trial, is appended to the gate input (s_m = MLP_gate([z_m + e_m, q_m])). It is compared with conventional quality indices in the same position and with no quality input.

*Advantages of C:* tiny capacity increase; clean, interpretable ablation. *Risks:* descriptors may duplicate what the encoders already extract (expected for EDA); amplitude-preserving descriptors may encode participant identity; the approach inherits every sensitivity listed in report 02, section 2.4.

## 6. Comparison

| | A0 | A1 | B-gated | B-xattn | C |
|---|---|---|---|---|---|
| Parameters | ≈ 150 | ≈ 6k | ≈ 40k | ≈ 49–74k | ≈ 42k |
| Pre-computation | features, minutes | features, minutes | band-power, minutes | same | + descriptors, minutes (CPU-hours only for exploratory EEG Rips) |
| Training, per fold | seconds | < 1 min CPU | minutes, CPU or laptop GPU | minutes | minutes |
| Missing modality | mean imputation | mean imputation | native | native | native |
| Main risk | under-fits temporal structure | same | over-fits 32 participants | over-fits; gain unproven | redundancy |

A full evaluation of one neural variant is 8 participant folds × 5 seeds = 40 short runs, so the whole ablation table fits on one workstation.

## 7. Alternatives considered and set aside

* **State-space sequence models (S4/Mamba-type) and raw-waveform Transformers.** Their advantage is long raw sequences. After windowing, sequences here have 5–29 tokens. Modelling 7,680-sample raw trials end-to-end needs far more participants than DEAP has.
* **FFT-domain token mixing (FFTNet as published).** Solves a cost problem this project does not have; retained only as the band-weighting ablation in 4.1.
* **Persistence images with convolutional encoders, learned diagram vectorisation.** Too many inputs for the data; revisit only if scalar descriptors show incremental value.
* **Domain-adversarial or subject-adaptive training.** The review literature suggests transfer methods are what most improves cross-subject results. Out of scope for the architecture question, but the *calibrated* normalisation regime is a cheap proxy and is reported throughout.
* **Riemannian covariance features for EEG.** A strong classical baseline in cross-subject brain–computer-interface work; an optional addition to A0 if a dependency on a Riemannian-geometry package is acceptable.
