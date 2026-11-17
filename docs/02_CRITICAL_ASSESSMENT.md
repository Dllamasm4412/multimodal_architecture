# 02 — Critical Assessment

Independent critique of the research concept and of the supplied plan (`Modality_Aware_Topological_Fusion_Research_Plan.docx`). Sources are cited by author and year; full references and verification status are in `01_EVIDENCE_REVIEW.md`. Statements marked **[analysis]** are my own mathematical or methodological reasoning, not findings from a paper.

## 1. Bottom line

1. **The direction is plausible as a hypothesis test, not as an expected performance gain.** There is no participant-independent evidence that persistent homology improves emotion recognition, and there are concrete mathematical reasons to expect its descriptors to be largely redundant with amplitude and spectral features in these signals.
2. **The largest risk is not topology; it is DEAP.** A rigorous cross-subject benchmark reports DEAP arousal accuracy at chance (about 49–51 %, LibEER 2024). If our own baselines cannot beat a permutation null, no incremental claim about topology, attention or robustness can be tested on this dataset under this protocol.
3. **Neither topology variant should be assumed superior.** Lower-star H0 measures the distribution of oscillation amplitudes. Delay-embedding Vietoris–Rips H1 measures how loop-like (periodic) a signal is. Both are cheap enough to test; both must pass a redundancy and surrogate test before entering a neural model.
4. **Cross-attention is not supported as a default.** CoRe-Sleep's own ablation shows that early, mid-late and cross-attention architectures perform alike until the auxiliary unimodal losses are added, and that was with 5,791 participants rather than 32.
5. **The best-supported use of topology in this project is as a periodicity/quality descriptor for quasi-periodic peripheral signals** (BVP, respiration), and for event detection on other datasets. Those are different claims from emotion recognition and need different data.

## 2. The two topology approaches

### 2.1 Approach A — lower-star H0 on the waveform

**What it computes [analysis].** Sweeping a threshold upward through a sampled signal, each local minimum starts a connected component; at each local maximum two components merge and the one with the higher (younger) minimum ends. Each finite point of the diagram is therefore a (valley height, merging peak height) pair, and its persistence is the depth of that valley. This is the same pairing used for peak *prominence* in peak detection and is closely related to rainflow cycle counting in fatigue analysis. The diagram is the multiset of oscillation amplitudes with all timing removed.

**Invariances [analysis].** The diagram is unchanged by any order-preserving change of the time axis: frequency change, phase shift, time-warping, and resampling that keeps the extrema. This is the property Gravot et al. advertise. It is *not* invariant to amplitude: scaling the signal by *a* scales every birth and death by *a*, and an offset shifts them.

**What Gravot's inertia is [analysis].** Inertia is the mean squared distance of diagram points to their centroid, which equals the variance of the births plus the variance of the deaths. It is zero for any constant-amplitude oscillation of any frequency and grows with amplitude modulation and baseline wander, quadratically in signal scale. In physiological terms it is an envelope-variability measure.

**Why it worked for transmission-line faults, and why that does not transfer.**
* The method discards every diagram point whose lifetime is below 0.9 × the pre-fault amplitude. A healthy phase therefore yields inertia exactly zero by construction and a faulted phase a positive value. The classifier is in effect asking "did this phase's current oscillate with larger amplitude than before the fault?"
* The three inertias are normalised to sum to one, which exploits three-phase geometry with no physiological analogue.
* K-means was fitted and scored on the same 3,300 simulations; only the 15 field records were held out.
* The paper's comparison table lists test conditions of other methods, not accuracies on common data. No amplitude-ratio or RMS baseline was run, so the paper gives **no evidence that topology adds anything beyond amplitude** even in its own domain.
* Invariance to frequency is an asset when recorders have different sampling rates. In physiology it means discarding heart rate, breathing rate and EEG band structure, which carry most of the established arousal information.

### 2.2 Approach B — delay embedding, Vietoris–Rips, H0/H1

**What it computes.** A window of the signal is mapped to points (x_t, x_{t+τ}, …, x_{t+(d−1)τ}). A periodic signal traces a closed curve; a quasi-periodic signal with two incommensurate rhythms fills a torus; broadband noise fills a blob. The maximum H1 persistence measures how round and how empty the loop is, and Perea & Harer (2015) developed exactly this as a *periodicity score*. Emrani et al. (2014) used the first Betti number of delay embeddings to detect wheezes, which are near-sinusoidal segments. In both cases the detected property is periodic or harmonic structure.

**Consequences [analysis].**
* H1 max persistence is closely related to spectral peakedness (how much power sits in one rhythm) and to envelope stability (the loop collapses whenever the amplitude dips). H0 statistics of the point cloud track point spacing, i.e. signal variance and bandwidth.
* A point cloud forgets the order of its points. Timing is retained only indirectly through the fixed delay, so heart rate and breathing rate are encoded weakly and nonlinearly.
* **Narrow-band filtering manufactures loops.** Any band-passed noise is an amplitude-modulated oscillation whose embedding is an annulus. H1 features of alpha-band EEG therefore largely reflect the filter bandwidth and the band's envelope statistics. Per-band delay-embedding topology of EEG, as in Yan et al. (2022), must be read with this in mind.
* Vietoris–Rips persistence is fragile to outliers: a single artefact sample creates points far from the attractor and distorts the diagram. Artefact handling or subsampling is not optional.

### 2.3 When can either approach carry information beyond the power spectrum? [analysis]

A zero-mean stationary Gaussian process is completely specified by its power spectrum. The probability distribution of *any* quantity computed from it, including any persistence diagram from either approach, is therefore a function of the spectrum alone. For such signals topology can only be a different parametrisation of second-order information; it cannot contain label-relevant information that the spectrum lacks. (For Approach A this is visible directly: the expected rate of extrema of a Gaussian process is given by its spectral moments, a classical result of Rice.)

Topology can add information only to the extent that the signals are nonlinear, non-Gaussian or non-stationary within the analysis window:

* **Scalp EEG at rest or during passive viewing** is, by the classical surrogate-data literature, only weakly distinguishable from linearly filtered noise, with clear nonlinearity mainly in seizures (Theiler & Rapp 1996; Stam 2005). The verified successes of topological EEG analysis are in seizure detection (Piangerelli et al. 2018; Wang, Ombao & Chung 2018), which fits this picture. DEAP's preprocessed EEG is additionally restricted to 4–45 Hz.
* **BVP and respiration** are produced by nonlinear oscillators and are strongly non-Gaussian, so non-spectral structure certainly exists. Whether it relates to arousal is an open question.
* **EDA** is a sparse, asymmetric, event-driven signal. Approach A applied to it recovers skin-conductance-response prominences, which are the standard phasic features under another name.

This gives a direct experimental handle: features computed on amplitude-adjusted phase-randomised surrogates (IAAFT; Schreiber & Schmitz 1996), which keep the spectrum and the amplitude distribution but destroy everything else, should predict arousal *worse* than features from the real signals if topology contributes non-spectral information. The literature search found no study that has run this test for physiological topology features.

### 2.4 Sensitivity

| Factor | Approach A | Approach B |
|---|---|---|
| Embedding dimension, delay | not applicable | Strong. Data-driven choices (mutual information, false nearest neighbours) are unstable on noisy broadband signals and make features incomparable across participants. Fix per modality from physiology (window ≈ one dominant period) and report a sensitivity grid. Yan et al. report two different (d, τ) settings in one paper. |
| Window length | Sets how much non-stationarity leaks into the diagram | Must contain several cycles; point count drives cost |
| Normalisation | Decisive. Per-window z-scoring removes the amplitude information that is most of what A measures | Decisive. Per-window z-scoring leaves shape only; without it, features track amplitude |
| Additive noise | Diagram is provably stable in the bottleneck sense (Cohen-Steiner et al. 2007), but counts, entropy and inertia over low-persistence points are not. A noise threshold is required; Gravot's is physics-specific | Stable to small perturbations; fragile to outliers and bursts |
| Between-participant amplitude differences (skull, electrode, sensor placement) | Transfer directly into the features | Same unless normalised |

### 2.5 Suitability by modality

| Modality | Approach A | Approach B | Prior expectation for arousal |
|---|---|---|---|
| BVP | Pulse-amplitude distribution, baseline wander | Clean loop; thickness = beat-to-beat morphology and amplitude variability | **Most suitable.** Pulse amplitude is an established sympathetic marker, but conventional beat features capture it |
| Respiration | Breath-depth distribution | Loop regularity = breathing regularity | Suitable; about 12–18 breaths per trial limits precision |
| EDA | Equivalent to response prominences | No rationale (no recurrence) | A only; expected to duplicate standard features |
| EEG, per channel | Envelope variability per band | Loops largely induced by band-pass filtering | **Least suitable**; lowest prior |
| EEG, across channels | — | Persistent homology of a 32-node connectivity network (as in Wang et al. 2024) costs almost nothing and is the only variant using spatial information | Exploratory only; protocol of the prior work unverified |

A third "hybrid" approach is not justified as a core component. The cross-channel connectivity filtration is the only variant that taps a genuinely different information source, and it is worth one exploratory arm, nothing more.

### 2.6 Computation [analysis; order-of-magnitude estimates to be measured]

* Approach A is a union-find pass, O(n log n). All of DEAP is seconds to minutes.
* Approach B with H1 scales roughly with the cube of the number of points. Windows of ≤ 250 points (8 s of BVP at 32 Hz; 20 s of respiration at 8 Hz) take tens of milliseconds each; about 24,000 such diagrams cover BVP and respiration for the whole dataset in minutes.
* Per-channel EEG is the expensive case: 1,280 trials × 32 channels × 15 non-overlapping 4 s windows ≈ 614,000 diagrams per frequency band. With subsampling to ~200 points this is a few CPU-hours per band, one-off and parallelisable. Feasible, but it buys the least-motivated features.
* H0/H1 is therefore practical offline for every modality. It is not practical, and not needed, inside the training loop.

### 2.7 Emotion recognition, event detection, signal quality

| Use | Evidence | Assessment |
|---|---|---|
| Emotion recognition, unseen participants | None found. Yan et al. (2022) report > 99 % on DEAP and DREAMER, but explicitly subject-dependent, on shuffled overlapping 1 s windows | Unsupported |
| Event or regime detection (seizure, arrhythmia, wheeze, sleep stage) | Piangerelli 2018; Emrani 2014; Ignacio 2019; Dindin 2020; Chung et al. 2021 (cross-database) | Best supported. A change of dynamical regime changes attractor geometry. Needs event-annotated data, which DEAP lacks |
| Signal quality of periodic signals | Recent preprints on PPG (Shao et al. 2025) | Plausible and mechanistically clear: a clean pulse makes a clean loop. Untested against conventional quality indices; DEAP has no quality annotations |

## 3. The ten assessment questions

1. **Modality-specific encoders: justified?** Yes for *input representation and timescale*. The signals differ by two orders of magnitude in bandwidth (EDA < 0.5 Hz, respiration 0.1–0.5 Hz, BVP ≈ 1–8 Hz, EEG 4–45 Hz) and in structure (multichannel oscillatory, quasi-periodic, event-driven). That argues for per-modality sampling rate, window length and front end. It does not by itself argue for different, larger learned architectures per modality on 32 participants.
2. **Can delay embedding plus persistent homology be complementary?** In principle only through nonlinear or non-stationary structure (2.3). Plausible for BVP and respiration, weak for scalp EEG.
3. **Will topology remain useful beside strong temporal/spectral encoders?** Probably not for prediction accuracy. Its descriptors are deterministic functions of the same waveform, and its most distinctive property, invariance to timing, removes established arousal markers. Graff et al. (2021) found topological heart-rate-variability indices "at least as useful" as classical ones, which is parity.
4. **Is H0/H1 practical for multichannel signals?** Yes offline (2.6).
5. **Which modalities suit topology?** BVP and respiration, then EDA (Approach A only), then EEG.
6. **Is cross-attention necessary?** Not on current evidence. Arousal-relevant coupling between EEG and peripheral signals is slow (seconds), CoRe-Sleep's gain came from its training objectives rather than from attention, and the data are too few to fit fine-grained cross-modal interactions. Gated fusion with auxiliary unimodal losses is the defensible default; cross-attention is an ablation.
7. **Will adaptive Fourier filtering beat STFT, wavelets or CNNs?** Unlikely. FFTNet applies the FFT across the *token sequence* as a cheap substitute for attention on long sequences; our sequences have at most a few dozen tokens. Its evidence is a single-author preprint with Long Range Arena and ImageNet results, no reported variance, no replication found, and no physiological data. Applied to the actual signal spectrum, the same idea reduces to a learned, input-conditioned band weighting, which a linear layer on log band-powers or a learnable filter bank already provides.
8. **Is DEAP large enough?** For compact models and for effects of roughly half a standard deviation of the per-participant paired difference (my estimate: about 0.02–0.03 in AUC, to be measured), yes. For ranking architectures that differ by less, no. The independent unit is the participant (n = 32); windowing multiplies samples, not independent labels. All participants saw the same 40 videos, which adds a stimulus confound.
9. **Simpler alternatives to test:** constant predictor; stimulus-identity prior; eye/muscle channels alone; log band-power with ridge or logistic regression; peripheral beat, response and breath features; EEGNet; unimodal models averaged; label-free per-participant feature standardisation. The last is likely the single highest-leverage intervention for cross-subject performance and must be reported separately because it uses the test participant's unlabeled data.
10. **Interesting but unsupported hypotheses:** H1 loops in single-channel EEG reflect emotion-relevant dynamics; topological features generalise across participants better than spectral ones; topology-informed quality gating beats conventional quality indices; cross-attention between EEG and peripheral signals helps on DEAP; adaptive spectral filtering helps on physiological signals.

## 4. Critique of the supplied plan

**Sound and worth keeping:** trial-level labels; participant-disjoint evaluation; the explicit statement of what the three source papers do *not* show; separating topology-as-feature (M4) from topology-as-quality-gate (M6); staged gates.

**Weaknesses:**

1. **Gates are not falsifiable.** "Evidence of nonredundant behaviour sufficient to justify further experiments" and "gain survives fair comparison" have no thresholds, no pre-specified primary metric, and no detectable-effect calculation. The task (classification or regression) is left open.
2. **The decisive topology test comes last and in the noisiest setting.** Topology's incremental value is first tested as M4 = M3 + TDA, on top of a trained cross-attention network, where seed variance is largest. The same question can be answered earlier, more cheaply and with lower variance using classical models.
3. **The first milestone optimises for the appealing part.** Reproducing Gravot's figures on synthetic sinusoids verifies code, but its outcome is known in advance from 2.1 and it cannot inform the DEAP question. The evaluation harness and noise floor should come first.
4. **The mathematics of the method is not examined.** The plan does not note that lower-star H0 is extrema pairing, that inertia is envelope variance, or that for linear-Gaussian signals no topological statistic can exceed the spectrum's information. These determine the experimental controls.
5. **Approach B is absent.**
6. **Missing threats to validity:** the shared-stimulus confound (only listed as a secondary check); eye and facial-muscle contamination of EEG, especially above 20 Hz; class imbalance and the majority-class baseline; individual rating-scale use; whether normalisation of an unseen participant may use that participant's unlabeled data; the limits a 60 s trial places on heart-rate-variability and tonic EDA analysis; possible site differences in the GSR channel.
7. **The model ladder M0→M6 is cumulative,** so each step changes capacity as well as mechanism. M4 has no placebo (equally many non-topological features), and M6 has no conventional quality-index comparator in the table.
8. **CoRe-Sleep is read too architecturally.** What transfers is its objective (auxiliary unimodal losses, training with dropped modalities), not its cross-attention blocks or its size. Its alignment loss assumes the modalities observe the same latent state, which holds for EEG and EOG in sleep and is doubtful for EEG and skin conductance in emotion.
9. **No control of researcher degrees of freedom.** Seven models, several modalities and several topology variants on 32 participants will produce spurious "wins" without a short pre-registered list of confirmatory comparisons.
10. **Robustness claims rest on synthetic corruption only,** and are uninterpretable if clean performance is near chance.

## 5. Strongest arguments

**For the direction**
* Modality-appropriate representation is physiologically well motivated.
* Topological descriptors are low-dimensional, cheap, interpretable and come with stability guarantees, which suits a 32-participant regime better than high-capacity learned features.
* The rigorous incremental-value test (participant-independent, surrogate-controlled, placebo-controlled) does not exist in the literature, so either outcome is publishable.
* Periodicity/quality descriptors for peripheral signals address a real problem in wearable monitoring and connect to the longer-term goals.

**Against**
* For near-Gaussian signals topology is mathematically a re-encoding of the spectrum; scalp EEG is close to that regime.
* Topology's invariance to timing discards heart rate, breathing rate and band structure.
* The only emotion-recognition evidence is subject-dependent and probably leaky.
* DEAP cross-subject arousal performance is at or near chance in a rigorous benchmark, leaving little room to measure increments.
* The concept stacks five hypotheses (modality-specific encoding, adaptive spectra, topology, cross-attention, robustness) on a dataset that can resolve one or two.

## 6. Experiments that can falsify the main hypotheses

| ID | Hypothesis under test | Experiment | Falsified if |
|---|---|---|---|
| F0 | DEAP supports participant-independent arousal estimation at all | Classical baselines vs within-participant label-permutation null | Best baseline not above the null |
| F1 | Topological features are not redundant | Predict each topological feature from spectral + amplitude + entropy features with a flexible regressor, cross-validated across participants | Cross-validated R² ≥ 0.9 for all retained features |
| F2 | Topology carries non-spectral, label-relevant information | Same features from IAAFT surrogates; compare predictive value | Surrogate-derived features predict as well as real ones |
| F3 | Topology adds predictive value | Control vs control + topology vs control + equally many placebo features, paired by participant | Gain interval includes zero, or does not exceed the placebo gain |
| F4 | Topological features generalise across people | Participant-identification accuracy and between/within-participant variance ratio of each feature family | Features are mostly participant fingerprints and gains vanish across participants |
| F5 | Cross-attention is needed | Cross-attention vs gated fusion vs capacity-matched block without cross-modal exchange | No gain over the capacity-matched control |
| F6 | Topological quality gating improves robustness | Gating by topological score vs conventional quality indices vs learned gating, under graded corruption | Not better than conventional indices |

Thresholds and procedures are specified in `04_EXPERIMENTAL_ROADMAP.md`.

## 7. Uncertainties in this assessment

* The effect-size estimate in Q8 is assumption-based until Stage 1 measures the variance.
* The claim that scalp EEG is nearly linear-Gaussian comes from older surrogate studies on resting data; emotion-evoked EEG in DEAP has not been tested this way, which is why F2 is an experiment rather than a conclusion.
* Several cited papers were verified only at abstract or snippet level (flagged in report 01), including the only connectivity-topology emotion paper and the multimodal DEAP comparisons.
* The DEAP layout details used in the architecture and handoff are from the dataset documentation as I recall it and must be confirmed in the audit.
