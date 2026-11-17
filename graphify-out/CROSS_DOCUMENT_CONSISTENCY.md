# Cross-Document Consistency

Reconciliation of concepts duplicated or redefined between the DEAP-era reports (`docs/01`–`05`, here **D01–D05**) and the foundation-model reports (`docs/foundation_model_research/`, here **FM01–FM08**, **FMCH** = CHANGELOG). Machine-readable version: `reconciliation_overlay.json` (3 nodes, 85 edges; 68 VERIFIED, 17 INFERRED). `graph.json` is unchanged.

**Principle.** Concepts are merged (`same_as`) only when both nodes describe the same thing in the same report generation. Where a concept changed between generations it is linked by `evolved_into`, `superseded_by` or `same_entity_role_changed`. Where only the label is shared it is linked by `name_collision`.

## 1. Structural finding

`graph.json` contains **0 edges** between the two report generations. The extraction agents processed D01–D05 and FM01–FM08 in separate batches, so the graph presents two disconnected research programmes. The overlay adds 34 cross-generation edges.

## 2. True duplicates (same concept, same generation) → `same_as`

| Node 1 | Node 2 | Evidence |
|---|---|---|
| FM05 "Architecture B (private encoders plus fusion tokens)" | FM04 "Architecture B: private encoders with time-aware bottleneck fusion" | FM05 §1; FM04 §4 |
| FM05 "O1 masked latent prediction" | FM04 "Objective O1" | FM04 §4.5 points to FM05 |
| FM04 "Resource benchmark before pretraining" | FM07 "Stage 0b resource benchmark" | FM07 Stage 0b cites FM04 §4.6 |
| FMCH "MIMIC-III variable inter-signal delay finding" | FM03 "MIMIC-III Waveform dataset" (`same_entity_as`) | CHANGELOG row 4 → FM03 §5a |

Related but **not** duplicates:

| Pair | Relation | Why not merged |
|---|---|---|
| FM03 "Synchronisation classes A–D" / FM08 "Synchronisation audit and classes A–D" | `implements` | Definition vs audit procedure. Class definitions agree word for word in substance (FM03 §5a, FM08 §6) |
| FM04 "Scale descriptor token" / FM08 "Scale descriptors" | `implements` | Same quantities; a model input token in B, a feature vector in Milestone 1 |
| FM04 "Stage 1 synthetic cardiac-pulse deciding test" / FM07 "Stage 1" | `part_of` | One test inside the stage |

## 3. Concepts that evolved between generations

### 3.1 Architectures (same letters, different designs)

| DEAP-era (D03) | Foundation-model era (FM04) | Relation | Evidence |
|---|---|---|---|
| **A**: A0 hand-crafted features + ridge/logistic; A1 ≈ 6 k-parameter supervised network | **A**: per-modality residual encoders, unimodal masked reconstruction (O2), gated late fusion, 4–7 M | `name_collision` (V); D-A0 `evolved_into` the hand-crafted reference embedder (I 0.85) | D03 §3; FM04 §3; FM08 §8 |
| **B-gated**: supervised modality encoders, masked gated fusion, auxiliary heads, modality dropout, ≈ 40 k; primary | **B**: private Transformer encoders, time-stamped bottleneck with lag bias, O1 pretraining, ≈ 13 M; primary | `evolved_into` (I 0.85, role succession) | D03 §1, §4; FM04 §1, §4 |
| D-B's gated fusion | FM-A's fusion and FM-B's "late gated" variant | `evolved_into` (I 0.95) | D03 §4.2; FM04 §3, §4.4 |
| **B-xattn** with capacity-matched own-modality control | "No-exchange control" and interaction variants | `evolved_into` (I 0.95) | D03 §4.3; FM04 §4.4 |
| **C**: B-gated + zero-initialised topology adapters (lower-star H0, delay-embedding Rips) | **C**: B + phase encoding, physical front ends, Riemannian EEG token, coupling targets. **Topology excluded as input** | `superseded_by` (V) | FM04 §5 last paragraph |
| Topology adapter | Persistent homology as **probe only** | `evolved_into` (V) | FM01 §4; FM04 §5 |
| **C-Q** topological quality gating | Reliability-aware gating (learned quality vs conventional indices) | `evolved_into` (I 0.75) | D03 §5.2; FM04 §4.4 |

**Rule for readers and coding models:** never write "Architecture A/B/C" without the report number. In current work, B means FM04 §4.

### 3.2 Experimental stages (same numbers, different experiments)

| No. | DEAP-era (D04) | Foundation-model era (FM07) | Relation |
|---|---|---|---|
| 0 | Data and evaluation audit (DEAP) | Data layer, harness, baselines (multi-dataset) | `evolved_into` (I 0.95) |
| 0b | — | Resource benchmark | new |
| 1 | Classical baselines and noise floor | Synthetic coupling benchmark | **`name_collision`**; D-Stage 1 content `evolved_into` FM-Stage 0 |
| 2 | Topology primitives on synthetic signals | Unimodal pretraining (H1, H2) | **`name_collision`**; D-content → FM-Stage 4 probes |
| 3 | Physiological topology decisive test | Cross-modal pretraining decisive experiment | **`name_collision`**; D-content → FM-Stage 4 |
| 4 | Compact neural models, modality-specific encoders | Signal-processing extensions | **`name_collision`**; D-content `evolved_into` FM-Stage 2 (K5) (V) |
| 5 | Multimodal fusion and robustness | Scale and far transfer | **`name_collision`**; D-content → FM-Stages 3 and 6 |
| 6 | Conditional topology integration | Robustness | **`name_collision`**; D-content → FM-Stage 4 |
| 7 | Frozen confirmatory evaluation | Frozen confirmatory evaluation | `evolved_into` (I 0.95) |

### 3.3 Confirmatory comparisons (same labels, redefined)

| Label | D04 §2 (DEAP) | FM06 §6 (current) | Where the D04 question lives now |
|---|---|---|---|
| K1 | Topology increment over controls and placebo | Pretrained vs random initialisation | FM07 Stage 4, exploratory |
| K2 | Modality-specific vs homogeneous encoders | Cross-modal vs unimodal pretraining | FM-K5 |
| K3 | Multimodal B-gated vs best unimodal | Interaction vs late gated vs no-exchange (+ pairwise module arm) | FM07 Stage 3 acceptance; FM06 §8.1 Q3 (not a K) |
| K4 | Topological vs conventional quality gating under corruption | True vs perturbed pairing (within-window rotation, within-state swap) | FM07 Stage 6, not confirmatory |
| K5 | — | Modality-specific vs shared encoder | — |

The FM-K definitions also changed within the current generation: the latest revision added the pairwise-module arm to K3 and replaced K4's fixed 5 s / 60 s / 10 min shifts with within-window rotation and within-state swap (CHANGELOG §1 rows 2–3; overlay `revises`). Only FM06 §6 and FM07 Stage 3 carry the current definitions.

### 3.4 Other evolved concepts

| DEAP-era | Current | Relation | Evidence |
|---|---|---|---|
| D05 Milestone 1 (DEAP harness) | FM08 Milestone 1 (multi-dataset harness with input policy and synchronisation audit) | `supersedes` (V) | FM08 line 5 |
| D05 "Milestone 2: topology module" | FM08 "Milestone 2: synthetic coupling benchmark"; topology → probe | `superseded_by` (V); `evolved_into` probe (I 0.85) | FM08 §14; FM01 §4 |
| D05 feature groups | Hand-crafted embedder | `evolved_into` (I 0.95) | D05 §4.5; FM08 §8 |
| D03 strict vs calibrated normalisation | Inductive / transductive / personalised settings; three normalisation modes | `evolved_into` (V) | FM06 §2; FM04 §2 |
| D04 participant-level protocol | FM06 §11 statistics (adds pretraining-seed variance, Holm over K1–K5) | `evolved_into` (V) | — |
| D02 falsification experiments F0–F6 | FM07 hypotheses H1–H7 and stage gates | `evolved_into` (I 0.85) | F0 → Stage 0 gate; F1–F4 → Stage 4; F5 → H4/K3; F6 → Stage 6 |
| D01 DEAP (primary benchmark) | FM03 DEAP (far-transfer cell only) | `same_entity_role_changed` (V) | FM03 §1, §7 |

## 4. Review of existing graph edges

| Edge in `graph.json` | Verdict | Reason |
|---|---|---|
| Gated fusion ↔ FFTNet (`semantically_similar_to`, 0.55) | **Reject** | Modality gating and spectral gating share a word, not a mechanism |
| SleepFM ↔ SHHS (`conceptually_related_to`, 0.65) | **Reject** | SleepFM was pretrained on one clinic's data (FM02 §3.5); the model trained on SHHS and MESA is wav2sleep, which has no node |
| PaPaGei ↔ MESA (AMBIGUOUS, 0.3) | Keep, relabel *possible pretraining overlap* | Stated from recollection only (FM02 §5; FM03 §6) |
| Architecture B ↔ MESA (INFERRED 0.75) | Upgrade to verified `pretrained_on` | FM03 §7; FM07 Stage 3 |
| MIMIC delay finding → synchronisation audit (INFERRED 0.75) | Upgrade to verified | CHANGELOG row 4 |
| Data-throughput benchmark → Stage 0b (INFERRED 0.75) | Partly right | Belongs to Stage 0 / Milestone 1; it feeds Stage 0b |
| Pairing perturbation ↔ interpolation problem (0.65) | Keep as thematic only | Not evidence of a dependency |

Isolated nodes in `graph.json`: the FM02 and FM03 report nodes, and **CinC 2018, which is the dataset of a primary evaluation cell**. The overlay connects CinC 2018.

## 5. Current terminology

| Term | Use the definition in | Ignore the definition in |
|---|---|---|
| Architecture A, B, C | FM04 | D03 |
| Stage 0–7, Stage 0b | FM07 | D04 |
| K1–K5 | FM06 §6, FM07 Stage 3 (latest revision) | D04 §2; any pre-revision copy |
| Milestone 1 | FM08 | D05 |
| Milestone 1b, 2 | FM08 §14 | D05 §8 |
| H1–H7 | FM07 §1 | — (D02 uses F0–F6) |
| Synchronisation classes A–D | FM03 §5a (definitions), FM08 §6 (audit) | — |
| Evaluation cells E1–E8 | FM06 §3.2 | — |
