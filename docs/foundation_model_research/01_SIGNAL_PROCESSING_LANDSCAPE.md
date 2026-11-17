# 01 — Signal-Processing Landscape

Which signal-processing ideas, including ones from audio, vibration analysis, sonar and vision, are worth testing in a transferable multimodal physiological model, and where each belongs. Statements marked **[analysis]** are my reasoning; citations and their verification status are in the evidence index (section 7). The earlier `docs/02_CRITICAL_ASSESSMENT.md` already analysed persistent homology and delay embedding in depth and is referenced, not repeated.

## 1. How to judge a representation

**A deterministic transform adds no information [analysis].** Every representation below is a function of the waveform. Its value can only be one of:

1. **Sample efficiency:** it hands the network something that would otherwise have to be learned from limited data.
2. **Invariance:** it removes nuisance variation (gain, sampling rate, sensor, heart rate) that hurts transfer.
3. **Access:** it exposes structure that the chosen pipeline otherwise discards. Two cases matter here. Magnitude spectrograms discard phase, so phase-coupling information is unrecoverable downstream. Pooled unimodal embeddings discard fine timing, so cross-modal lag information is unrecoverable by late fusion.

**A representation can enter a model in four places,** and the right place differs by method:

| Role | Meaning | Cost at inference |
|---|---|---|
| Front end | replaces or augments the learned input stem | always paid |
| Side information | added to tokens as positional or contextual features | small |
| Prediction target | what the self-supervised model must predict for masked tokens | none |
| Probe | a quantity we test for in the learned embedding | none |

Using a method as a target or probe lets us ask "does the encoder already capture this?" before paying to feed it in. That is the main change of approach relative to the earlier DEAP plan, where topology was to be an input feature.

**The scaling expectation [analysis].** Fixed, well-designed transforms help most when pretraining data are scarce and their advantage shrinks as data grow, because convolutional stems learn comparable filters. Every "advanced representation" experiment should therefore be run at two or more pretraining sizes; a benefit that vanishes with data is an efficiency result, not a representational one.

## 2. Assessment by family

| Family | What it captures | Verdict | Role |
|---|---|---|---|
| **Wavelet scattering** | First order: band energies averaged over a time scale. Second order: how each band's envelope is modulated. Provably stable to noise and small time-warping | **Shortlist (3)** | front end, target |
| **Synchrosqueezing, reassigned spectrograms** | Same information as the underlying transform, re-concentrated onto instantaneous-frequency ridges. Unstable at low signal-to-noise | Use as a tool to extract instantaneous heart rate, breathing rate and phase; not as network input | utility for (2) |
| **Learnable filter banks (SincNet, LEAF), adaptive wavelets** | Band-pass filters with centre frequency and bandwidth as parameters in Hz | **Shortlist (3)** | front end |
| **Multiresolution convolution stems** | Parallel kernels of several lengths | Adopt as the default learned stem; it is the baseline, not a contribution | front end |
| **Bispectrum, bicoherence, higher-order cumulants** | Quadratic phase coupling between frequencies; identically zero for linear Gaussian signals | Probe only. High estimator variance, needs long windows, dominated by artefacts | probe |
| **Phase–amplitude and cross-frequency coupling** | Slow-rhythm phase modulating fast-rhythm amplitude; physiologically established in sleep EEG | Probe; second-order scattering captures the amplitude part | probe |
| **Cyclostationary analysis** | Statistics that repeat with a cycle; in machinery, analysed after resampling to the rotation angle | Motivates **Shortlist (2)**; the full spectral-correlation map is not needed | concept |
| **Empirical mode decomposition, Hilbert–Huang, variational mode decomposition** | Data-adaptive oscillatory modes | **Reject as input.** Modes are not consistent across windows or people, decomposition is unstable to noise, iterative and non-differentiable; on broadband signals it behaves like a wavelet filter bank | — |
| **Riemannian covariance geometry** | Spatial second-order structure of multichannel EEG with a metric invariant to linear channel mixing | **Shortlist (4)**, EEG only | side token, baseline |
| **Graph signal processing, connectivity graphs, graph networks** | Inter-channel relations as a graph | Defer. Attention over channel tokens already learns a data-dependent graph; the supporting evidence is mostly from subject-dependent emotion benchmarks | — |
| **Delay embedding, recurrence plots, recurrence quantification, persistent homology** | Geometry and recurrence of the reconstructed trajectory | Probe only; see the earlier reports. No evidence found then for participant-independent gains | probe |
| **Cross-modal coupling analysis** (cross-spectrum, coherence, phase synchronisation, delay estimation) | Relations between two signals that neither contains alone | **Shortlist (1)** | fusion layer, target, probe, baseline |
| **Sparse deconvolution of EDA** | Skin conductance as sparse nerve bursts passed through a slow impulse response | Adopt as the EDA front end (modality-specific, physiologically grounded) | front end |
| **Automatic gain control (PCEN) from speech** | Divides by a smoothed running energy | Part of **Shortlist (3)** | front end |
| **Learned tokenisation (vector-quantised codecs)** | Discrete codes for patches | Defer. Adds a training stage and codebook-collapse risk; latent prediction gives a similar abstraction without a tokenizer | — |
| **Multiscale patching** | Token duration matched to each signal's timescale | Adopt as the default (report 04) | tokenisation |
| **Masked modelling, latent prediction** | — | See report 05 | objective |
| **Signal-processing-defined targets** (as in masked feature prediction in vision) | The encoder must predict a chosen descriptor of hidden signal | **Shortlist (5)** | target, probe |

## 3. Shortlist

Each candidate is assessed on the twelve required criteria.

### (1) Lag-resolved cross-modal coupling

*Origin: time-delay estimation in sonar and microphone arrays (generalised cross-correlation); cost volumes in stereo vision and optical flow; "network physiology".*

1. **Mathematics.** Cross-correlation as a function of lag, R_ab(τ); its whitened form (phase transform) gives a sharp delay peak; coherence and phase-synchronisation indices quantify coupling strength. The neural equivalent is a correlation volume C(t, τ) = ⟨f_a(t), f_b(t + τ)⟩ between two modalities' feature maps, or an attention bias that is a learned function of the time difference between query and key.
2. **Preserved / discarded.** Keeps lag and strength of coupling; discards each signal's own content.
3. **Complementarity.** By construction absent from any unimodal feature. Pulse arrival time needs ECG and PPG; respiratory sinus arrhythmia needs respiration and heart period.
4. **Modalities.** ECG–PPG (hundreds of milliseconds), respiration–heart rate (seconds), EEG–heart rate and EEG–EDA (seconds, event-driven).
5. **Evidence.** Coupling between organ systems changes with sleep stage and is characterised by stable time delays (Bashan et al. 2012); ECG-derived cardiopulmonary coupling tracks sleep stability (Thomas et al. 2005); distinct forms of cardiorespiratory coupling change across sleep stages (Bartsch et al. 2014). These establish that the structure exists and is state-dependent. For tightly coupled cardiac signals, one 2026 preprint reports that removing its delay component lowers downstream ECG performance (CardioState-JEPA; report 02, section 3.6). For loosely coupled systems (respiration, EEG, EDA) a benefit to learned models is untested.
6. **Cost.** Linear in the number of lags; negligible beside the encoders.
7. **Robustness.** Whitened correlation degrades at low signal-to-noise; coupling estimates need several cycles (tens of seconds for breathing-related measures). Shared motion artefacts create spurious zero-lag coupling.
8. **Sampling and window.** Needs a common clock; nominal timestamps are not sufficient evidence of one. Beat-level lags are finer than a 0.25 s token, so they must be carried either inside token content or by a correlation over fine lags computed before patching (report 04, section 4.4). Clock drift between devices is a real hazard in wearables, and one large clinical waveform database documents changing delays of up to 500 ms between its signals (report 03, section 5a).
9. **Self-supervision.** Usable as a fusion layer (differentiable), as a pretext (predict an applied time shift), and as a probe target.
10. **Transfer.** Lags are in seconds and therefore device-independent in principle, but sensor site changes them (finger versus wrist pulse).
11. **Learnable anyway?** Token-level cross-attention can represent it. Late fusion of pooled vectors cannot represent fine lags. The explicit lag axis is an inductive bias with a readable output.
12. **Decisive ablation.** Synthetic coupling benchmark with a delay-resolution test, then on real data: fusion with versus without the relative-time term, against a capacity-matched block without cross-modal exchange and against late fusion with an explicit pairwise lag module, plus the pairing-perturbation controls (report 06, K3 and K4).

### (2) Cycle-synchronous (physiological-phase) encoding

*Origin: order tracking and time-synchronous averaging in rotating-machinery diagnostics; pitch-synchronous analysis in speech; cyclostationarity.*

1. **Mathematics.** Define cardiac phase φ_c(t), rising 0 → 2π between successive beats, and respiratory phase φ_r(t) between successive breaths. Attach (sin φ, cos φ) of both to every token of every modality as additional positional features. Averaging any signal over repeated cycles at fixed phase isolates its cycle-locked component.
2. **Preserved / discarded.** Adds timing relative to the body's own cycles; removes nothing. Wrong when beat or breath detection fails.
3. **Complementarity.** Clock-time positions cannot align events across people with different heart or breathing rates; phase positions can. Makes cycle-locked effects directly addressable: heartbeat-evoked EEG responses, breathing-related modulation of pulse amplitude and heart period.
4. **Modalities.** Any modality recorded together with a cardiac or respiratory signal.
5. **Evidence.** Heartbeat-locked cortical responses carry behaviourally relevant information (Park et al. 2014). Cycle-domain resampling is standard and effective in machinery diagnostics (Antoni 2009). In neural models the search found only an R-peak-distance positional encoding inside ECG classifiers and one model conditioned on a learned cardiac phase; nothing that attaches cardiac or respiratory phase to the tokens of *other* modalities (report 02, section 3.6).
6. **Cost.** Peak detection, linear time.
7. **Robustness.** Inherits the peak detector's failures (motion, arrhythmia). Mitigate by randomly dropping the phase features in training and by a confidence flag.
8. **Sampling and window.** Phase is dimensionless, hence invariant to sampling rate and to heart rate itself.
9. **Self-supervision.** Available without labels; phase can also be a prediction target.
10. **Transfer.** Device-independent by definition.
11. **Learnable anyway?** Only if the network learns beat detection in one stream and routes it to another. Supplying it is nearly free.
12. **Decisive ablation.** With versus without phase encodings; control with phases taken from a different recording (same statistics, wrong alignment). Caveat: phase encodings are derived from the cardiac or respiratory channel and must be masked whenever that channel is treated as missing, otherwise they leak it.

### (3) Physically parametrised, gain-robust front ends

*Origin: speech and audio front ends (SincNet, LEAF, per-channel energy normalisation); wavelet scattering.*

1. **Mathematics.** Band-pass filters whose centre frequencies and bandwidths are parameters in Hz, so the same filter bank is instantiated at any sampling rate. Per-channel energy normalisation divides each band by a smoothed running estimate of its own energy (automatic gain control). Scattering: S₁ = |x ∗ ψ_λ| ∗ φ and S₂ = ||x ∗ ψ_λ₁| ∗ ψ_λ₂| ∗ φ with fixed wavelets ψ and averaging window φ.
2. **Preserved / discarded.** Band energies and (S₂) envelope modulations; phase and timing finer than the averaging window are discarded, so the window must not exceed the token length. Gain normalisation deliberately discards absolute amplitude.
3. **Complementarity.** First order duplicates a spectrogram. Second order adds modulation structure that averaged spectrograms lose (for example breathing-rate modulation of pulse amplitude, burst rhythms in EEG).
4. **Modalities.** EEG, ECG, PPG; less relevant to slow EDA.
5. **Evidence.** Scattering matches or exceeds mel-spectral features in audio classification with little data (Andén & Mallat 2014); learnable Hz-parametrised front ends are established in speech (Ravanelli & Bengio 2018; Zeghidour et al. 2021); gain normalisation improves robustness to level changes in far-field speech (Wang et al. 2017). Evidence on physiological transfer across devices is missing.
6. **Cost.** A few convolutions; fixed scattering has no trainable parameters.
7. **Robustness.** Scattering has stability guarantees; gain control suppresses slow amplitude drift and sensor-gain differences.
8. **Sampling and window.** The point of the design: parameters are in physical units.
9. **Self-supervision.** Works as a stem under any objective; scattering coefficients are also stable, phase-free targets.
10. **Transfer.** The expected benefit is across sampling rates, amplifier gains and sensor types, which is where physiological models usually break.
11. **Learnable anyway?** Yes, given enough data and augmentation. Expect the gap to close as pretraining grows.
12. **Decisive ablation.** Swap only the stem (plain learned convolution; Hz-parametrised filter bank with gain control; fixed scattering) and evaluate probes trained at one sampling rate or device and tested at another, at two pretraining sizes.

### (4) Riemannian covariance geometry for multichannel EEG

*Origin: brain–computer interfaces; radar and array processing.*

1. **Mathematics.** Each window's spatial covariance is a symmetric positive-definite matrix. With the affine-invariant metric, distances are unchanged by any invertible linear re-mixing of channels. Matrices are mapped to a tangent-space vector at a reference point; re-centring each participant's matrices to the identity aligns participants without labels.
2. **Preserved / discarded.** Keeps power and zero-lag connectivity jointly; discards temporal order within the window, phase lags and non-Gaussian structure.
3. **Complementarity.** Inter-channel structure is absent from per-channel spectral tokens.
4. **Modalities.** EEG with roughly eight or more channels. Little to gain on two- or three-channel sleep montages.
5. **Evidence.** Among the strongest classical methods for cross-session and cross-participant EEG decoding (Barachant et al. 2012; Zanini et al. 2018).
6. **Cost.** Cubic in channel count; trivial for 32 channels.
7. **Robustness.** High-amplitude artefacts dominate covariances; average-referenced or component-cleaned data are rank-deficient and need shrinkage.
8. **Sampling and window.** Insensitive to sampling rate after band-limiting; needs windows of a few seconds for stable estimates.
9. **Self-supervision.** Vector size depends on the montage, so it is not channel-agnostic. Use as a dataset-specific side token at fine-tuning or on a fixed channel subset.
10. **Transfer.** Strong within a montage; re-centring is label-free calibration and belongs to the transductive setting.
11. **Learnable anyway?** In principle through attention over channel tokens, but that needs multichannel pretraining data, which sleep recordings do not provide.
12. **Decisive ablation.** Add the tangent-space token to the EEG stream on multichannel downstream tasks; compare with the tangent-space classifier alone as a baseline.

### (5) Signal-processing quantities as prediction targets and probes

*Origin: masked feature prediction in vision; spectral-code prediction in EEG models; morphology-aware PPG pretraining.*

1. **Mathematics.** Instead of predicting raw samples of a hidden patch, predict a descriptor of it: log-magnitude spectrum, scattering coefficients, deconvolved EDA driver, instantaneous heart or breathing rate, a coupling index. As probes: regress bicoherence, phase–amplitude coupling, recurrence measures or persistence statistics from the frozen embedding.
2. **Preserved / discarded.** The target decides what the encoder is rewarded for keeping; phase-free targets stop the model spending capacity on unpredictable sample-level detail.
3. **Complementarity.** Tests directly whether "advanced" quantities are already encoded. If they are linearly decodable, feeding them in is redundant. If they are not decodable *and* they predict downstream labels, they become auxiliary targets.
4. **Modalities.** All.
5. **Evidence.** Hand-crafted targets rival learned ones in vision (Wei et al. 2022). In physiology, LaBraM predicts spectral codes and PaPaGei supervises with pulse-morphology indices (report 02).
6. **Cost.** Offline computation of targets; none at inference.
7. **Robustness.** Deterministic targets cannot collapse.
8. **Sampling and window.** Targets defined in physical units carry across datasets.
9. **Self-supervision.** This *is* the objective design; see report 05, fallback objective.
10. **Transfer.** Untested; plausible for phase-free targets.
11. **Learnable anyway?** Not applicable; this changes what is learned, not what is supplied.
12. **Decisive ablation.** Same model with raw-waveform targets versus spectral or scattering targets versus learned latent targets; probe table for higher-order, dynamical and topological descriptors.

## 4. Ranking

| Rank | Candidate | Research value | Feasibility | Serves | Where it goes |
|---|---|---|---|---|---|
| 1 | Lag-resolved cross-modal coupling | High: the central question, largely untested with proper controls | High | cross-modal fusion | fusion layer; probes; baseline features |
| 2 | Physiological-phase encoding | Medium–high: cheap, mechanistically clear, apparently unexplored | High, given reliable beat detection | cross-modal fusion | token side information |
| 3 | Physically parametrised front ends with gain control; scattering | Medium: an engineering gain for cross-device transfer, likely to shrink with data | High | modality-specific encoding | input stems |
| 4 | Riemannian EEG covariance | Medium: well evidenced but confined to multichannel EEG | Medium (montage-dependent) | modality-specific encoding | EEG side token; baseline |
| 5 | Signal-processing targets and probes | Medium: mostly a methodological lens; settles the higher-order and topology questions cheaply | High | objective design | targets; probes |

Persistent homology, recurrence analysis and higher-order spectra are retained **only as probes** under candidate 5. Promotion to a target or input requires that the descriptor is (a) not linearly decodable from the pretrained embedding and (b) predictive of a downstream label beyond the embedding.

## 5. What ordinary networks cannot recover

Two cases where a standard pipeline loses information irreversibly [analysis], which is why candidates 1 and 2 rank highest:

* **Late fusion after temporal pooling** cannot use relative timing between modalities at scales finer than the pooled window.
* **Magnitude-only time-frequency inputs** cannot represent phase coupling between frequencies or between signals.

The recommended architecture avoids both by keeping time-stamped tokens up to the fusion stage and by embedding waveform patches rather than magnitude spectra for cardiac and respiratory signals.

## 6. Set aside, with reasons

* **Adaptive decompositions as inputs:** inconsistent mode identity across windows and people makes them unsuitable as network channels.
* **Image encodings of time series (recurrence plots, Gramian fields) with 2-D networks:** quadratic memory in window length and no established advantage over 1-D models.
* **Synchrosqueezed or reassigned spectrograms as inputs:** sharper pictures of the same information, at the price of noise sensitivity.
* **Graph networks over EEG channels:** subsumed by channel attention until evidence under participant-independent protocols says otherwise.
* **Full spectral-correlation (cyclostationary) maps:** large, and their useful content for these signals is captured by phase encodings and second-order scattering.

## 7. Evidence index

The combined, status-tagged reference list for this report is section 8 of `02_FOUNDATION_MODEL_PRIOR_ART.md`. Every reference named above was verified there; claims about physiological applications of scattering and filter-bank front ends beyond the cited audio papers are my assessment, not reviewed literature.
