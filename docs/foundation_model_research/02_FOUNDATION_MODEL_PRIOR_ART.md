# 02 — Foundation-Model Prior Art

Review of pretrained models for physiological signals, of the evidence on whether they transfer, and of what this leaves open. Searches were run on 2026-11-16 (about 110 queries and page fetches). Section 8 is the combined reference index for this report and for `01_SIGNAL_PROCESSING_LANDSCAPE.md`.

## 1. How to read this document

| Tag | Meaning |
|---|---|
| **V** | abstract, full text, repository or dataset page was read |
| **P** | search snippet or secondary listing only; details may be wrong |
| **nv** | not verified |

Reported results are the authors' own, under their protocols. Numbers are quoted only where the protocol is known. Preprints from 2026 were identified by arXiv number in search results and, where tagged V, read; they have not been peer reviewed unless a venue is given.

## 2. Bottom line

1. **A multimodal physiological foundation model is not novel in itself.** Large sleep models (SleepFM, more than 585,000 hours), multimodal wearable models, and missing-modality models already exist.
2. **Cross-modal latent prediction with delay modelling exists for cardiac signals.** CardioState-JEPA (August 2026) pretrains ECG, PPG and heart sounds with latent prediction and a learned, anchor-supervised delay. xMAE (ICML 2026) reconstructs ECG from PPG. The mechanism proposed in report 04 is therefore an extension, not an invention.
3. **What no located paper does:** (a) test whether cross-modal gains depend on fine temporal correspondence, by deliberately shifting or mis-pairing modalities; (b) compare interaction fusion with capacity-matched alternatives; (c) evaluate fused use with missing modalities across EEG, ECG, PPG, EDA and respiration; (d) model lags for loosely coupled systems where no physiological anchor exists.
4. **The field's own audits are sobering for EEG.** Independent benchmarks find EEG foundation models gain about one percentage point over small specialist networks, need full fine-tuning, and help mainly on long-context tasks such as sleep staging.
5. **Frozen-probe transfer is better evidenced for cardiac, PPG and sleep signals** than for EEG, which supports starting with cardiorespiratory modalities.

## 3. Models

### 3.1 EEG

| Model | Pretraining data | Architecture, size | Objective | Evaluation | Code / weights | Limitations |
|---|---|---|---|---|---|---|
| **BENDR**, Kostas et al., Front. Hum. Neurosci. 2021, arXiv:2101.12037 — V | Temple EEG corpus, "over 10,000 people"; hours not reported | Convolutional encoder + 8-layer Transformer, width 1536; size nv | wav2vec 2.0-style masked contrastive | Motor imagery, error potentials, P300, sleep; leave-participants-out; no cross-dataset test; fine-tuning best | Yes | Small downstream sets |
| **BIOT**, Yang et al., NeurIPS 2023, arXiv:2305.10351 — V | 6,478 clinical EEG recordings; 5,445 SHHS recordings; 21,264 ECG recordings | 4-layer Transformer, per-channel tokens; size nv | Contrastive between a perturbed and the original view | Seizure, abnormality, events, ECG, activity; patient-grouped splits; transfer across datasets reported; fine-tuned | Code yes; weights nv | Shared encoder across EEG and ECG, no cross-modal objective |
| **LaBraM**, Jiang et al., ICLR 2024, arXiv:2405.18765 — V | ≈ 2,535 h from ≈ 20 datasets; downstream sets excluded | 5.8 M / 46 M / 369 M | Vector-quantised tokenizer trained on Fourier amplitude and phase, then masked code prediction | Abnormality, events, emotion, gait; **linear probe far below fine-tuning (0.346 vs 0.641 balanced accuracy on one task)** | Code yes; weights nv | Needs full fine-tuning |
| **EEGPT**, Wang et al., NeurIPS 2024 — P | Mixed motor-imagery-heavy sets; hours nv | ≈ 10 M | Masked prediction in representation space with a momentum encoder, plus reconstruction | Linear probing on several tasks; splits nv | Code yes; official weights nv | Small, narrow pretraining data |
| **CBraMod**, Wang et al., ICLR 2025, arXiv:2412.07236 — V (abstract), P (details) | Temple EEG corpus; hours nv | Criss-cross Transformer (separate spatial and temporal attention), ≈ 4.0 M | Masked patch reconstruction | Up to 10 tasks on 12 datasets; fine-tuned in independent benchmarks | Yes | — |
| **LUNA**, Döner et al., NeurIPS 2025, arXiv:2510.22257 — P | Temple corpus + Siena, > 21,000 h | Learned queries map any electrode layout to a fixed latent; size nv | Masked patch reconstruction | Abnormality, emotion, artefact and slowing detection; protocol nv | Code yes | — |
| **REVE**, El Ouahidi et al., NeurIPS 2025, arXiv:2510.21585 — P | > 60,000 h, 92 datasets, 25,000 participants | Spatio-temporal Transformer with 4-D positional encoding; several sizes | Masked autoencoding | 10 tasks; details nv | Code and weights | — |
| NeuroLM (ICLR 2025, arXiv:2409.00101 — V); FEMBA (arXiv:2502.06438 — P) | ≈ 25,000 h; ≈ 21,000 h | Language-model based, up to 1.7 B; bidirectional state-space model, 7.8–386 M | Text-aligned tokens; masked reconstruction | — | — | Listed for completeness |

### 3.2 Independent evaluations of EEG foundation models

| Source | Design | Finding |
|---|---|---|
| "EEG Foundation Models: Progresses, Benchmarking, and Open Problems" (EEG-FM-Compass), arXiv:2601.17883, Natl. Sci. Rev. 2026 — V (abstract) | 12 open models versus specialist baselines on 13 datasets, 9 paradigms; leave-one-participant-out and few-shot | Linear probing often insufficient; specialists trained from scratch remain competitive; larger models do not reliably generalise better |
| Lee et al., "Are Large Brainwave Foundation Models Capable Yet?", arXiv:2507.01196, ICML 2025 — P | Fine-tuned LaBraM and another model versus small convolutional networks, participant-independent | Gains of about 0.9–1.2 % with orders of magnitude more parameters; freezing the backbone costs about 8–10 %; low-rank adaptation matches full fine-tuning |
| EEG-FM-Bench, Xiong et al., arXiv:2508.17742 — P | 14 datasets, 10 paradigms, 7 EEG models | Generalisation gap with frozen backbones; pretraining efficiency matters more than data scale |
| Kommineni et al., arXiv:2605.28563, 2026 — V (abstract) | Three models, six datasets | Help on long-context tasks (sleep staging, mental state); on short-window tasks small supervised models are comparable; fragile with few channels |

### 3.3 ECG

| Model | Pretraining data | Architecture, size | Objective | Evaluation | Code / weights | Limitations |
|---|---|---|---|---|---|---|
| **ECG-FM**, McKeen et al., JAMIA Open 2025, arXiv:2408.05178 — V | ≈ 1.5 M 12-lead ECGs | wav2vec 2.0 encoder + Transformer, 90.9 M | Masked span reconstruction + contrastive between adjacent segments + lead masking | Interpretation labels, reduced ejection fraction; patient-level splits; external cohort; fine-tune, scratch and linear probe compared | Yes | 12-lead, 10 s clinical ECG |
| **HuBERT-ECG**, medRxiv 2024.11.14.24317328 — P | 9.1 M 12-lead ECGs | base 93.1 M | Masked cluster prediction (inferred from name; nv) | 21 datasets, fine-tuned | Weights, non-commercial licence | Preprint |
| **ECGFounder**, Li et al., NEJM AI 2025, arXiv:2410.04133 — P | 10.77 M ECGs, 1.82 M people | Convolutional; size nv | **Supervised** on 150 labels | — | Code and a checkpoint; a single-lead model was trained, release nv | Not self-supervised |
| **ECG-JEPA**, Kim, arXiv:2410.08559 — V (snippet level) | ≈ 180,000 public ECGs | Transformer | Masked latent prediction | About 22 h training on one consumer GPU | Code | Single author, 12-lead |
| **HeartLang**, Jin et al., ICLR 2025, arXiv:2502.10707 — V | public ECG sets | Transformer | Heartbeats segmented at QRS become "words"; vector-quantised vocabulary; masked sentence modelling | 6 downstream sets | Code | Beat-synchronous tokenisation depends on QRS detection |

### 3.4 PPG and wearables

| Model | Pretraining data | Architecture, size | Objective | Evaluation | Code / weights |
|---|---|---|---|---|---|
| **PaPaGei**, Pillai et al., ICLR 2025, arXiv:2410.20542 — V | > 57,000 h, ≈ 20 M segments, public data | Residual 1-D network with expert mixture, 512-d output; size nv | Self-supervision with positives defined by pulse morphology | 20 tasks, 10 datasets, **linear or shallow heads on frozen embeddings**; splits nv | Weights released (BSD-3) |
| **Pulse-PPG**, Saha et al., IMWUT 2025, arXiv:2502.01108 — P | Wrist PPG from 120 participants over 100 days (≈ 55,000 h by my arithmetic) | 28.5 M | nv | Beats PaPaGei on 10 of 11 tasks | Weights released |
| **Abbaspourazad et al.** (Apple), ICLR 2024, arXiv:2312.05409 — V (abstract) | PPG and ECG from ≈ 141,000 participants | Convolutional | Contrastive with **participant-level positive pairs** | Health conditions and demographics | Release nv; believed not released |
| **LSM-2** (Google), arXiv:2506.05321 — P | 40 M hours of multimodal wearable data | — | Masked modelling in which real gaps and artificial masks share one mechanism | — | nv |

### 3.5 Multimodal physiological models

| Model | Modalities | Data | Objective | Evaluation | Notes |
|---|---|---|---|---|---|
| **SleepFM**, Thapa et al., ICML 2024, arXiv:2405.17766 — P | Brain activity, ECG, respiration | > 100,000 h, > 14,000 participants, one clinic | Leave-one-out contrastive: each modality against the mean of the others | Linear probe 0.88 vs supervised CNN 0.72 AUROC on sleep staging | Code released |
| SleepFM, Nature Medicine 32:752–762, 2026, doi:10.1038/s41591-025-04133-4 — P | EEG, ECG, EMG, respiration | > 585,000 h, > 65,000 participants | Contrastive variant for varying montages | 130 diseases predicted with concordance ≥ 0.75 | Size, splits, weights nv |
| **wav2sleep**, Carter & Tarassenko, PMLR 259, arXiv:2411.04644 — P | Any subset of ECG, PPG, respiration | > 10,000 recordings from 6 public sets including SHHS and MESA | **Supervised** sleep staging | — | Checkpoints released; strong reference for cardiorespiratory sleep staging |
| **PhysioOmni**, Jiang et al., arXiv:2504.19596 — V (abstract) | EEG, ECG, EOG, EMG | — | Decoupled tokenizer; masked modelling with modality-invariant and modality-specific objectives | Emotion, sleep, motor, workload with arbitrary missing modalities | No PPG, EDA or respiration; code not found |
| **PhysioME**, Lee et al., arXiv:2510.11110 — P | several | — | Within-modality reconstruction + reconstruction of missing modalities + cross-modal contrastive | — | Details nv |
| **Brant-X**, Zhang et al., KDD 2024, arXiv:2409.00122 — P | EEG model aligned with EOG, ECG, EMG | — | Contrastive alignment at patch and sequence level | Sleep, emotion, arrhythmia, gait | Code released |
| **NormWear**, Luo et al., arXiv:2412.09758 — P | PPG, ECG, EEG, EDA, motion | ≈ 14,900 h | Wavelet-transform front end, channel-aware attention; objective nv | 18 applications, 11 datasets | Code released |
| **CroSSL**, Deldari et al., WSDM 2024, doi:10.1145/3616855.3635795 — V | generic multimodal time series | — | Masks latent embeddings of modality-specific encoders; cross-modal aggregator; no negatives | Tolerates missing modalities | Code released |
| **FOCAL**, Liu et al., NeurIPS 2023, arXiv:2310.20071 — V | generic multimodal sensing | — | Shared and private latent spaces per modality with orthogonality; temporal constraint | 4 sensing datasets | Code released |

### 3.6 Closest prior work to this proposal

| | **CardioState-JEPA** — V (full text) | **xMAE** — V (most of the text) | **CTAF** — V (full text) |
|---|---|---|---|
| Reference | Shafiq et al., arXiv:2608.12944, Aug 2026; preprint | Zhou et al., arXiv:2605.00973, May 2026; ICML 2026 listing seen | Khorasani & Demazure, arXiv:2602.02784 |
| Modalities | ECG, PPG, heart sounds | PPG encoder; ECG as pretraining target only | EEG and peripheral signals |
| Encoder | One shared Transformer (12 layers, width 768), no modality-specific layers; a light convolutional tokenizer per modality | 6.5 M | — |
| Objective | Masked latent prediction; unimodal stage, then paired stage | Almost all ECG hidden, PPG visible; ECG reconstructed through **directional** cross-attention | Contrastive with alignment regularisation |
| Treatment of lag | **Per-token delay** predicted by a small head, bounded, used to gather the target with a Gaussian kernel. **Supervised by physiological anchors** (pulse arrival time; R-peak to first heart sound) where reliable | None explicit; states it works "without assuming a fixed ECG–PPG delay" | Sinusoidal time features on inputs; no lag term in attention |
| Data | Public ECG, PPG and heart-sound sets; paired ECG–PPG from VitalDB and others; hours not given | ≈ 9,400 h paired ECG–PPG from ≈ 2,400 MIMIC-III patients | K-EmoCon |
| Cross-modal vs unimodal pretraining | Yes, same architecture and budget: PPG classification 80.4 vs 78.5; PPG regression error 9.1 vs 10.7; heart sounds 82.3 vs 60.2; ECG within noise | Beats unimodal and multimodal baselines of similar size on 15 of 19 tasks | — |
| Effect of delay modelling | Removing it lowers ECG from 90.9 to 85.3 | — | Removing time features is the only ablation |
| Shift or mis-pairing control | **No** | **No** | No |
| Capacity-matched fusion comparison | Not applicable (single-modality downstream use) | Not applicable (PPG-only at inference) | **No** fusion baselines |
| Missing-modality evaluation | **No** | Not applicable | — |
| Evaluation | Frozen encoder, 25 tasks | Frozen linear probes, participant-level 5-fold, 19 tasks, mostly private data | Leave-one-participant-out |
| Release | Code and weights listed | Code; no weights | — |

Also seen: an informal ECG–PPG latent-prediction repository with a learned lag term, reporting an interpolation shortcut that needed a 75 % mask ratio; ECG-guided contrastive PPG pretraining (arXiv:2608.01620, P); an R-peak-distance positional encoding for ECG classification (P); a model using learned cardiac phase as a conditioning variable (arXiv:2608.10011, P). No respiratory-phase encoding was found.

## 4. Cross-cutting observations

1. **Scale is out of reach, and is not the gap.** Public models train on 10⁴–10⁵ hours, industry on 10⁶–10⁷. An academic project competes on design and evidence.
2. **Objectives are converging on masked prediction, but evaluation habits differ.** EEG models are judged after full fine-tuning; cardiac, PPG and sleep models increasingly by frozen probes. The second is the right standard for small labelled datasets.
3. **Multimodal models mostly align whole windows.** Contrastive alignment treats a 30 s epoch as the unit of correspondence. Only the cardiac papers address timing inside the window.
4. **Shared versus modality-specific encoders is unresolved.** BIOT and CardioState-JEPA share one Transformer across modalities with light tokenizers; PhysioOmni, CroSSL and FOCAL use modality-specific encoders. No located paper compares the two at matched size.
5. **Cross-modal pretraining can improve unimodal use.** CardioState-JEPA and xMAE both report it for PPG. This is the best existing support for hypothesis H3 and makes its extension to respiration and EEG a reasonable next question.
6. **Controls are missing.** No located study reports pretraining-seed variance, a shift control, or a capacity-matched fusion baseline.
7. **Timing claims rest partly on poorly aligned data.** The MIMIC-III waveform documentation warns of "a (changing) delay of up to 500ms" between waveforms and that pulse transit time measured across them "may be unreliable" (read on the PhysioNet page). xMAE pretrains on that corpus. Any conclusion about beat-level ECG–PPG timing needs data whose synchronisation is established, which is a further reason for the controls proposed here.

## 5. Reuse assessment

| Stream | Candidate checkpoints | Fit to this project | Recommendation |
|---|---|---|---|
| EEG, multichannel wake | CBraMod (≈ 4 M, small, public); LaBraM; REVE | Channel-flexible; trained on clinical EEG; independent audits say fine-tuning is needed | Reuse as the EEG stream for multichannel tasks, with low-rank adaptation; also as a frozen reference |
| EEG, sleep montage | same | Audits report fragility with few channels | Train the compact private encoder; compare with reuse |
| ECG, single lead | ECG-FM, HuBERT-ECG (12-lead, 10 s, ≈ 90 M; non-commercial licence for one); ECGFounder single-lead release unconfirmed; CardioState-JEPA weights | Lead count, window length and size mismatch | Reference models only; train the compact encoder |
| PPG | PaPaGei, Pulse-PPG; CardioState-JEPA | Good fit; frozen-probe evidence. PaPaGei's pretraining sets likely overlap our candidate corpora (nv) | Frozen reference in every table; overlap must be checked before comparing |
| Respiration, EDA | none found | — | Train from scratch |
| Fusion | none reusable | — | Train from scratch; this is the research object |

**Strategy: hybrid.** Controlled experiments use compact encoders trained from scratch, because the central comparison requires pretraining the same encoders with and without cross-modal learning. Public checkpoints serve as reference systems and as a plug-in EEG stream. A purely reuse-based system is the contingency if our own unimodal pretraining shows no benefit (report 07, Stage 2).

## 6. Novelty map

| Candidate contribution | Strongest prior work | Assessment |
|---|---|---|
| New signal-processing representation | Wavelet front ends (NormWear); spectral targets (LaBraM); morphology-aware PPG (PaPaGei) | Weak. Physiological-phase encoding across modalities appears unexplored but is a component, not a thesis |
| Better modality-specific encoder | A crowded field per modality | Weak as a headline; the matched comparison of shared versus specific encoders is a useful secondary result |
| Better fusion mechanism | CoRe-Sleep; attention bottlenecks; time-aware attention (CTAF); delay alignment (CardioState-JEPA) | Moderate. Anchor-free lag modelling for loosely coupled systems with native missing-modality use is an extension |
| Physiologically informed objective | CardioState-JEPA; xMAE | Moderate and shrinking. Not safe as the sole claim |
| More reliable transfer across datasets and tasks | Every foundation-model paper claims this | Weak without controls |
| Better missing-modality strategy | PhysioOmni; PhysioME; CroSSL; LSM-2; wav2sleep | Weak as a headline |
| **Controlled evidence on what cross-modal pretraining learns** — coupling, shared slow state, or capacity — across tightly and loosely coupled systems | None located | **Strongest.** A combination of known techniques whose contribution is the experimental design: shift controls, capacity-matched fusion, synthetic ground truth, pretraining-seed variance, cross-dataset frozen probes |

The recommended framing: one mechanism (lag-aware bottleneck fusion with cross-modal latent prediction, extended beyond cardiac pairs) and one rigorous question (does the benefit depend on physiological coupling?). The second carries the contribution whichever way the result falls.

## 7. Not verified

Parameter counts of BENDR, BIOT, LUNA, REVE, PaPaGei, PhysioME, NormWear, the second SleepFM, ECGFounder; CBraMod's pretraining hours; evaluation protocols of EEGPT, LUNA, REVE; objectives of Pulse-PPG, NormWear, HuBERT-ECG; release of the Apple model and of an ECGFounder single-lead checkpoint; the pretraining sets of PaPaGei; whether the unread final part of the xMAE text contains further controls. Entries tagged P should be read in full before citation.

## 8. Combined reference index (reports 01 and 02)

Models and benchmarks are referenced in section 3 with arXiv numbers or DOIs. Additional references:

| Reference | Used for | Status |
|---|---|---|
| Andén & Mallat, "Deep scattering spectrum", IEEE Trans. Signal Process. 62:4114–4128, 2014, doi:10.1109/TSP.2014.2326991 | scattering | V |
| Daubechies, Lu, Wu, "Synchrosqueezed wavelet transforms", Appl. Comput. Harmon. Anal. 30:243–261, 2011, doi:10.1016/j.acha.2010.08.002 | synchrosqueezing | V |
| Ravanelli & Bengio, "Speaker recognition from raw waveform with SincNet", IEEE SLT 2018, doi:10.1109/SLT.2018.8639585 | Hz-parametrised filters | V |
| Zeghidour et al., "LEAF: a learnable frontend for audio classification", ICLR 2021, arXiv:2101.08596 | learnable front end | V |
| Wang et al., "Trainable frontend for robust and far-field keyword spotting", ICASSP 2017, doi:10.1109/ICASSP.2017.7953242 | gain normalisation | V |
| Barachant et al., "Multiclass brain–computer interface classification by Riemannian geometry", IEEE Trans. Biomed. Eng. 59:920–928, 2012, doi:10.1109/TBME.2011.2172210 | Riemannian EEG | V |
| Zanini et al., "Transfer learning: a Riemannian geometry framework with applications to brain–computer interfaces", IEEE Trans. Biomed. Eng. 65:1107–1116, 2018, doi:10.1109/TBME.2017.2742541 | re-centring | V |
| Bashan et al., "Network physiology reveals relations between network topology and physiological function", Nat. Commun. 3:702, 2012, doi:10.1038/ncomms1705 | state-dependent coupling, time-delay stability | V |
| Thomas et al., "An electrocardiogram-based technique to assess cardiopulmonary coupling during sleep", Sleep 28:1151–1161, 2005, doi:10.1093/sleep/28.9.1151 | cardiopulmonary coupling | V |
| Bartsch et al., "Three independent forms of cardio-respiratory coupling: transitions across sleep stages", Computing in Cardiology 2014 (PMID 25664348) | cardiorespiratory coupling | V |
| Greco et al., "cvxEDA: a convex optimization approach to electrodermal activity processing", IEEE Trans. Biomed. Eng. 63:797–804, 2016, doi:10.1109/TBME.2015.2474131 | EDA deconvolution | V |
| Antoni, "Cyclostationarity by examples", Mech. Syst. Signal Process. 23:987–1036, 2009, doi:10.1016/j.ymssp.2008.10.010 | cycle-domain analysis | V |
| Knapp & Carter, "The generalized correlation method for estimation of time delay", IEEE Trans. Acoust. Speech Signal Process. 24:320–327, 1976, doi:10.1109/TASSP.1976.1162830 | delay estimation | V |
| Nagrani et al., "Attention bottlenecks for multimodal fusion", NeurIPS 2021, arXiv:2107.00135 | bottleneck fusion | V |
| Korbar et al., "Cooperative learning of audio and video models from self-supervised synchronization", NeurIPS 2018, arXiv:1807.00230 | synchronisation pretext | V |
| Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243 | latent prediction | V |
| Baevski et al., data2vec, ICML 2022, arXiv:2202.03555 | latent targets | V |
| Wei et al., "Masked feature prediction for self-supervised visual pre-training", CVPR 2022, arXiv:2112.09133 | hand-crafted targets | V |
| Park et al., "Spontaneous fluctuations in neural responses to heartbeats predict visual detection", Nat. Neurosci. 17:612–618, 2014, doi:10.1038/nn.3671 (magnetoencephalography) | heartbeat-locked cortical responses | V |
| Cheng et al., "Subject-aware contrastive learning for biosignals", arXiv:2007.04871, 2020 | identity in contrastive learning | V |
| Kontras et al., CoRe-Sleep, IEEE TNSRE 32:840–849, 2024, doi:10.1109/TNSRE.2024.3354388 | fusion and missing modalities | read in the earlier review |
