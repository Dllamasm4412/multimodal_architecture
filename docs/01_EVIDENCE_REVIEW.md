# 01 — Evidence Review

Focused review of the four supplied sources and of external literature on persistent homology for physiological signals, DEAP evaluation rigor, and multimodal fusion. Searches were run on 2026-11-16.

## 1. How to read this document

**Verification status** of each external source:

| Tag | Meaning |
|---|---|
| **READ** | Supplied PDF, read in full or in its relevant sections |
| **VERIFIED** | Abstract, publisher page or full text was seen during the search |
| **PARTIAL** | Search-result snippet or secondary listing only; details may be wrong |
| **UNVERIFIED** | From recollection; not confirmed in this session |

Numbers are quoted only with the evaluation protocol they belong to. "Subject-dependent" means data from the test participant, or from the same trial, were available in training.

## 2. Supplied sources: what they actually show

### 2.1 Gravot et al. (2025), transmission-line fault classification — READ

* **Method.** Lower-star filtration on each phase current; H0 persistence diagram (minimum–maximum pairs) computed with Ripser. Pairs with lifetime below 0.9 × pre-fault amplitude are discarded, as are pairs with positive birth or negative death. Feature: diagram *inertia* (mean squared distance to the diagram centroid), normalised so the three phases sum to one. K-means with k = 7; a second K-means (k = 2) on the sigmoid-transformed residual-current inertia separates phase-to-phase from phase-to-phase-to-ground faults.
* **Data and protocol.** 3,300 simulated faults (11 types × 10 locations × 5 resistances × 6 inception angles, 100 kHz). K-means is fitted and scored on the same simulations. 15 field records are classified with the simulation centroids.
* **Results.** 100 % on noise-free simulations; above 95 % down to 5 dB signal-to-noise ratio for threshold margins of 5–20 %; 15/15 field records.
* **Limitations.**
  * The comparison table lists other methods' test conditions, not their accuracy on common data. No amplitude or RMS baseline is run.
  * The threshold makes a healthy phase's inertia zero by construction, so the feature acts as an amplitude-increase detector.
  * The authors state the method "lacks temporal segmentation capabilities" and needs windowing for evolving faults; that extension is shown on one record.
  * The stated invariance to frequency, phase and sampling rate is real; invariance to amplitude scale is not a property of the method.
* **Our interpretation, not a finding of the paper.** Success depends on three-phase geometry, a known pre-fault reference amplitude, and faults that are large amplitude changes. None of these has a counterpart in emotion recognition.

### 2.2 CoRe-Sleep, Kontras et al. (2024) — READ

* **Method.** EEG and EOG as STFT time-frequency inputs in 30 s epochs. Per modality: an inner Transformer over within-epoch features and an outer Transformer over a sequence of 21 epochs (4 layers each, width 128, 8 heads). Multimodal encoders add pairwise cross-attention. Loss: multimodal cross-entropy + unimodal cross-entropy on each modality (multi-supervised loss) + a contrastive alignment loss between unimodal representations (weight 0.1).
* **Data and protocol.** SHHS-1, 5,791 participants, random 70/30 participant split with 100 recordings for validation.
* **Findings relevant here.**
  * Without the extra losses, early fusion, mid-late fusion and the cross-attention model "performed similarly". The multi-supervised loss helped all three; the alignment loss helped only the models with cross-modal attention.
  * With both losses, CoRe-Sleep was about equal to early fusion on complete data and best when a modality was missing.
  * On recordings with a heavily corrupted modality, the early and cross-attention models did worse than using the clean modality (EOG) alone; the paper contrasts this with mid-late fusion, which has fewer cross-modal interactions.
  * Training can use recordings that contain only one modality.
* **Limitations.** Two modalities that observe the same latent state; a dataset about 180 times larger than DEAP in participants; one random split.
* **Our interpretation.** The transferable elements are the training objective and training on incomplete inputs. The paper does not show that cross-attention itself is needed.

### 2.3 FFTNet, Fein-Ashley (2025), arXiv:2502.18394v3 — READ

* **Method.** Replaces self-attention by: FFT along the token sequence → multiplication by a filter whose scale and bias are produced by an MLP from the mean token → modReLU on the complex coefficients → inverse FFT. Cost O(n log n).
* **Evidence.** Long Range Arena average 58.31 vs 55.83 (Transformer) and 55.32 (FNet); ImageNet top-1 79.6 vs 79.4 (base size) at lower FLOPs; component ablation on ImageNet only. No variance or repeated runs reported.
* **Limitations.** Single-author preprint; no peer-reviewed version, independent replication or physiological-signal experiment was found (PARTIAL search). The filter has one parameter set per frequency bin, tying it to sequence length. Multiplication in the frequency domain is a circular convolution, so sequence ends wrap around.
* **Our interpretation.** It is a token-mixing efficiency method for long sequences, not a spectral feature extractor for signals. The transferable idea is an input-conditioned weighting of frequency bands.

### 2.4 DEAP, Koelstra et al. (2012) — citation VERIFIED, details UNVERIFIED

IEEE Trans. Affective Computing 3(1):18–31, DOI 10.1109/T-AFFC.2011.15. Confirmed from the abstract: 32 participants, 40 one-minute music-video excerpts, EEG and peripheral signals, ratings of arousal, valence, like/dislike, dominance and familiarity. **Not confirmed in this session:** the preprocessed file layout, channel order, site differences, and the paper's single-trial classification scores (my recollection is a within-participant scheme with arousal accuracy near 0.6). All of these are checked in Stage 0 before anything relies on them.

## 3. External literature

### 3.1 Persistent homology for EEG emotion recognition

| Source | Method | Data and protocol | Reported result | Limitations | Implication |
|---|---|---|---|---|---|
| Yan et al. 2022, "Topological EEG Nonlinear Dynamics Analysis for Emotion Recognition", IEEE Trans. Cogn. Dev. Syst., DOI 10.1109/TCDS.2022.3174209, arXiv:2203.06895 — **VERIFIED** (full text) | Delay embedding per rhythm band → persistent homology → diagram features → random forest | DEAP, DREAMER. **Subject-dependent**: 10-fold on shuffled 1 s windows with 25 % overlap | DEAP arousal 99.37 %, valence 99.35 %; DREAMER above 99.9 % | Overlapping windows of one trial fall on both sides of the split; authors state subject-independent evaluation was not addressed; two inconsistent embedding settings in the text; no spectral baseline or surrogate test seen; entropy and recurrence comparisons exist but their numbers were not seen | Direct prior art for Approach B on DEAP. Its accuracy says nothing about new participants |
| Wang, Li, Zhang, Liang 2024, "Emotion recognition based on phase-locking value brain functional network and topological data analysis", Neural Comput. Appl. 36:7903–7922, DOI 10.1007/s00521-024-09479-3 — **PARTIAL** | Persistent homology of phase-locking-value connectivity networks; landscapes, Betti curves, persistent entropy | DEAP, DREAMER, SEED; protocol not verified | Accuracies of roughly 83–90 % in a secondary summary | Protocol unknown | Prior art for cross-channel topology. Read in full before citing |

No paper was found that combines topological features with a deep multimodal fusion model for emotion recognition, or that applies persistent homology to skin conductance or respiration for emotion. A preprint on eye-movement topology for emotion (arXiv:2507.17450) was seen at snippet level only.

### 3.2 Persistent homology for EEG more generally

| Source | Method | Data and protocol | Finding | Limitations |
|---|---|---|---|---|
| Wang, Ombao, Chung 2018, Ann. Appl. Stat. 12(3):1506–1534, DOI 10.1214/17-AOAS1119 — **VERIFIED** (abstract) | Fourier-series denoising, persistence landscapes, permutation test | Single-patient seizure recording | Detects pre- vs during-seizure differences | Statistical test, not a classifier; one patient; no spectral baseline |
| Piangerelli et al. 2018, BMC Res. Notes 11:392, DOI 10.1186/s13104-018-3482-7 — **VERIFIED** (abstract) | Persistent entropy with a linear classifier | 33 seizure and 33 non-seizure recordings (CHB-MIT) | AUC 97.2 % vs 62.0 % for sample entropy | Small sample; protocol not verified; comparison with one entropy measure only |
| Xu, Drougard, Roy 2021, Front. Neurosci. 15:761703, DOI 10.3389/fnins.2021.761703 — **VERIFIED** | Mini-review | — | Argues topology is under-used in EEG; no new data | Not empirical |

### 3.3 Persistent homology for cardiac and pulse signals

| Source | Method | Data and protocol | Finding | Limitations |
|---|---|---|---|---|
| Chung, Hu, Lo, Wu 2021, Front. Physiol. 12:637684, DOI 10.3389/fphys.2021.637684 — **VERIFIED** (abstract) | Both sublevel-set persistence and Vietoris–Rips persistence of the delay embedding, on heart-rate-variability series | Sleep–wake and REM/NREM classification; cross-database validation on three datasets | Features transfer across databases | Numbers and non-topological baselines not seen |
| Graff et al. 2021, PLoS ONE 16(7):e0253851, DOI 10.1371/journal.pone.0253851 — **VERIFIED** (abstract) | Topological indices of RR-interval series | Healthy vs post-stroke | Indices "at least as useful as" classical ones | Parity, not gain; protocol not verified |
| Dindin, Umeda, Chazal 2020, arXiv:1906.05795 — **PARTIAL** | Topological channel inside a modular neural network | Arrhythmia detection, patient-independent focus | Reaches, does not exceed, state of the art | Prior art for topology inside a deep model; no spectral ablation seen |
| Ignacio et al. 2019, ICMLA, arXiv:1911.12253 — **PARTIAL** | Delay embedding, persistence features, random forest | PhysioNet/CinC 2017 single-lead ECG | Numbers not seen | — |
| Shao et al. 2025, arXiv:2509.12510 — **VERIFIED** (abstract; preprint) | Sublevel-type persistence on a learned embedding of PPG; clustering | PPG signal quality | Silhouette 0.72 | Quality, not emotion; no comparison with conventional quality indices |
| Aston et al. 2018, Physiol. Meas. 39:024001, DOI 10.1088/1361-6579/aaa93d — **VERIFIED** (citation) | Delay-coordinate attractor reconstruction of cardiovascular waveforms (not persistent homology) | Blood-pressure and pulse waveforms | Waveform-shape features beyond heart-rate variability | Non-topological; shows the embedding of pulse signals is informative |

### 3.4 Foundations of delay-embedding topology and of surrogate testing

| Source | Relevance |
|---|---|
| Perea & Harer 2015, Found. Comput. Math. 15:799–838, DOI 10.1007/s10208-014-9206-z — **VERIFIED** | Maximum H1 persistence of the sliding-window embedding quantifies periodicity. Establishes what Approach B measures |
| Emrani, Gentimis, Krim 2014, IEEE Signal Process. Lett. 21(4):459–463, DOI 10.1109/LSP.2014.2305700 — **VERIFIED** | First Betti number of delay embeddings detects wheezes, i.e. near-sinusoidal segments |
| Myers, Munch, Khasawneh 2019, Phys. Rev. E 100:022314, DOI 10.1103/PhysRevE.100.022314 — **VERIFIED** | Persistent homology of ordinal-partition networks for dynamic-state detection (a network construction, not delay-embedding Rips) |
| Cohen-Steiner, Edelsbrunner, Harer 2007, Discrete Comput. Geom. 37:103–120, DOI 10.1007/s00454-006-1276-5 — **VERIFIED** | Stability theorem: diagrams move at most as far as the function is perturbed |
| Theiler et al. 1992, Physica D 58:77–94, DOI 10.1016/0167-2789(92)90102-S — **VERIFIED** | Surrogate-data method for testing nonlinearity |
| Schreiber & Schmitz 1996, Phys. Rev. Lett. 77:635, DOI 10.1103/PhysRevLett.77.635 — **VERIFIED** (DOI from search result) | IAAFT surrogates preserving spectrum and amplitude distribution |
| Theiler & Rapp 1996, Electroencephalogr. Clin. Neurophysiol. 98:213–222, DOI 10.1016/0013-4694(95)00240-5 — **VERIFIED** (abstract) | Linear Gaussian surrogates account for resting EEG; earlier low-dimensional findings attributed to autocorrelation |
| Stam 2005, Clin. Neurophysiol. 116:2266–2301, DOI 10.1016/j.clinph.2005.06.011 — **VERIFIED** (citation) | Review of nonlinear EEG analysis |
| Lubba et al. 2019 (catch22), Data Min. Knowl. Discov. 33:1821–1852, DOI 10.1007/s10618-019-00647-x — **VERIFIED** | Generic time-series feature set; source of placebo features |

**Search for sceptical evidence.** No study was found that benchmarks persistent-homology time-series features against spectral or generic feature sets and reports redundancy, and none that applies phase-randomised surrogates to physiological topology features. A finance preprint (arXiv:2602.00383, PARTIAL) uses Fourier surrogates as nulls for persistence-landscape norms. Ichinomiya (Sci. Rep. 2025; VERIFIED by fetch) combines recurrence plots with persistent homology, runs no non-topological baseline, and describes the advantage as explainability rather than accuracy. The absence of such benchmarks is a gap, not evidence in either direction.

### 3.5 DEAP evaluation rigor

| Source | Finding | Implication |
|---|---|---|
| LibEER, Liu et al. 2024, arXiv:2410.09767 — **VERIFIED** (results table) | Cross-subject DEAP, participants split about 60/20/20, 1 s windows. Accuracy, valence / arousal: CDCN 57.78 / 49.73; DBN 55.02 / 50.99; BiDANN 53.49 / 50.07; GCBNet 53.58 / 50.05; EEGNet 52.36 / 48.94; SVM 49.58 / 51.48 | Under a controlled protocol, EEG-only arousal is at chance. Sets the realistic expectation for our baselines |
| Brookshire et al. 2024, Front. Neurosci., DOI 10.3389/fnins.2024.1373515 — **VERIFIED** | Segment-level splits put one person's data in train and test and inflate deep-learning EEG results; shown by reproducing published models (clinical EEG, not emotion) | The leakage mechanism that applies to window-shuffled DEAP studies |
| Kukhilava et al. 2025, arXiv:2505.18175 — **PARTIAL** | Review of 216 EEG emotion papers: no common protocol for splits, metrics or labels | Published DEAP numbers are not comparable with each other |
| Apicella et al. 2024, Neurocomputing 604:128354, DOI 10.1016/j.neucom.2024.128354 — **PARTIAL** | Systematic review of cross-subject and cross-session methods; transfer-learning approaches rank highest on average | Normalisation and adaptation matter more than encoder design |

### 3.6 Multimodal fusion on DEAP

| Source | Finding | Limitations |
|---|---|---|
| Ramaswamy & Palaniswamy, Appl. Comput. Inform., DOI 10.1108/ACI-03-2022-0080 — **PARTIAL** | Subject-independent comparison: EEG better than peripheral for arousal by about 7 points; peripheral better for valence by about 3.5 points | Details and protocol not verified |
| TACOformer, arXiv:2306.13592 — **PARTIAL** | Cross-attention fusion on DEAP and DREAMER; claims subject-independent state of the art | Protocol not verified |
| LibEMER, arXiv:2509.19330 — **PARTIAL** | Benchmark of multimodal emotion methods on DEAP with participant-level 3:1:1 split | Results not read |

No paper was found that compares EEG-only, peripheral-only and fused models under one leave-one-participant-out protocol with a permutation null.

### 3.7 Candidate datasets and baselines mentioned in the plan

| Source | Status |
|---|---|
| EEGNet, Lawhern et al. 2018, J. Neural Eng. 15:056013, DOI 10.1088/1741-2552/aace8c | **VERIFIED** |
| AMIGOS, Miranda-Correa et al. 2021, IEEE Trans. Affect. Comput. 12(2):479–493, DOI 10.1109/TAFFC.2018.2884461 — 40 participants; EEG, ECG, GSR | **VERIFIED** |
| PPG-DaLiA, Reiss et al. 2019, Sensors 19(14):3079, DOI 10.3390/s19143079 | Citation **VERIFIED**; that it contains wrist PPG under motion with an ECG reference is **UNVERIFIED** |

## 4. Prior art and novelty

| Element of the proposal | Prior art | Remaining novelty |
|---|---|---|
| Delay-embedding persistent homology for EEG emotion on DEAP/DREAMER | Yan et al. 2022 | Participant-independent evaluation; spectral, surrogate and placebo controls |
| Topology of EEG connectivity for emotion | Wang et al. 2024 | Same, pending a full reading |
| Sublevel-set and Rips persistence on cardiac series | Chung et al. 2021; Graff et al. 2021 | Application to BVP waveforms and to arousal |
| Topological channel inside a neural network | Dindin et al. 2020 | Multimodal, missing-modality setting |
| Persistent homology as PPG quality descriptor | Shao et al. 2025 (preprint) | Comparison with conventional quality indices; use as a fusion gate |
| Cross-attention fusion of EEG and peripheral signals on DEAP | TACOformer and others | Little; capacity-matched comparison with gated fusion is the missing piece |
| Lower-star H0 with inertia on physiological signals | None found | New, but expected to be largely an amplitude-variability measure |

The most defensible novel contribution is methodological: the first controlled, participant-independent test of whether topological descriptors carry arousal information beyond spectral, amplitude and entropy features. "Topological emotion recognition on DEAP" in itself is not new.

## 5. Methodological lessons

1. **Very high DEAP accuracies come from subject-dependent, window-shuffled protocols.** Under participant-level splits, EEG-only arousal is at chance in the one benchmark whose table was read. Any result of ours far above that should first be treated as a suspected leak.
2. **What delay-embedding topology measures is periodicity**, by the foundational papers' own account. Its relation to the spectrum is therefore close and has to be tested, not assumed away.
3. **Verified successes of physiological topology are in regime or event discrimination** (seizure, wheeze, arrhythmia, sleep stage), where the dynamics change qualitatively. Emotion during video viewing is not obviously such a change.
4. **Where topological and classical indices were compared on cardiac data, the outcome was parity.**
5. **In CoRe-Sleep, robustness came from the loss and the training regime**, with far more data than DEAP offers.
6. **The decisive control experiments do not exist in the literature.** Running them is the contribution.

## 6. Gaps in this review

* Not searched or not found: Fernández & Mateos (topological seizure biomarkers); learnable spectral filters for EEG beyond EEGNet; independent evaluations of FFTNet.
* Seen only at snippet level and to be read before citing: Wang et al. 2024; Dindin et al. 2020; Ignacio et al. 2019; Ramaswamy & Palaniswamy; TACOformer; LibEMER; Kukhilava et al. 2025; Apicella et al. 2024.
* Bibliographic check needed: one index lists Yan et al. under the journal's former name with a 2023 volume; confirm on IEEE Xplore.
* The classical result that extrema rates of Gaussian processes depend only on spectral moments (Rice) is cited in report 02 as textbook knowledge and was not looked up.
* The search was bounded (about 30 queries). It is sufficient to establish the main patterns above; it is not a systematic review.
