# Architecture B — Dependency Trace

Trace of the current foundation-model Architecture B through the research graph, with every relationship tied to its source section. Built from `graph.json` (unchanged, SHA-1 `663b8417…`) plus `reconciliation_overlay.json`. No research report was modified.

## Evidence classes

| Tag | Meaning |
|---|---|
| **G-EXT** | Edge in `graph.json`, marked EXTRACTED by the extraction agent; spot-checked against the cited section |
| **G-INF** | Edge in `graph.json`, marked INFERRED by the extraction agent |
| **O-V** | Added in the overlay; the relationship is stated in the cited section |
| **O-I** | Added in the overlay; reviewer's inference from comparing two sections (confidence given) |

Section references use the report number and section, for example `FM04 §4.4` (= `docs/foundation_model_research/04_ARCHITECTURE_OPTIONS.md`, section 4.4) or `D03 §4.2` (= `docs/03_ARCHITECTURE_RECOMMENDATION.md`). Line numbers refer to the files as of 2026-11-17.

## 1. Which node is Architecture B

| Node | Report | Community | Graph degree | Status |
|---|---|---|---|---|
| `…04_architecture_options_architecture_b` — "Architecture B: private encoders with time-aware bottleneck fusion" | FM04 §4 | 2 (Cross-Modal Prior Art and Fusion) | 11 | **Canonical.** Highest-degree node in the graph |
| `…05_self_supervised_learning_strategy_architecture_b` — "Architecture B (private encoders plus fusion tokens)" | FM05 §1 | 0 (Self-Supervised Objectives) | 1 | Same concept, extracted twice. Overlay: `same_as` (O-V) |
| `…03_architecture_recommendation_b_gated` — "Architecture B-gated" | D03 §4 | 1 | 4 | **Different design** (DEAP-era, supervised, gated fusion, ≈ 40 k parameters). Overlay: `evolved_into` FM04 B (O-I 0.85) |

The two foundation-model nodes split B across two communities, which is why the graph reports it as a bridge: part of its centrality is a duplication artefact, not a structural finding.

## 2. Relationships by topic

### 2.1 Modality-specific signal representations

| Relationship | Evidence | Class |
|---|---|---|
| B → modality-specific tokenisers (EEG per-channel 1 s raw + spectrum; ECG/PPG 0.25 s; respiration 1 s; EDA 2 s on deconvolved driver) | FM04 §4.2 (lines 95–105) | G-EXT |
| B → scale-descriptor token (amplitude, baseline, tonic level, quality per modality) | FM04 §2 "Amplitude" (line 31) | G-EXT |
| cvxEDA deconvolution → tokenisers | FM01 §2 table; FM04 §4.2 EDA row | G-EXT |
| tokenisers → tested in Stage 2 (K5: specific vs shared encoders) | FM07 Stage 2 Hypotheses, Components | O-V |
| scale descriptors implemented as hand-crafted features in Milestone 1 | FM08 §8 | O-V |
| B → Architecture C (signal-processing extensions: phase encoding, physical front ends, Riemannian token, coupling targets) | FM04 §5; FM01 §3–4 | G-EXT |

### 2.2 Temporal synchronisation and physiological coupling

| Relationship | Evidence | Class |
|---|---|---|
| B → lag-aware attention bias on fusion tokens | FM04 §4.4 "Mechanism" (line 127) | G-EXT |
| Lag-resolved cross-modal coupling → rationale for the lag bias and the pairwise module | FM01 §3 candidate 1 | G-INF 0.85 |
| Lag-term clarification revises the lag bias (per-modality receptive field, not pairwise coupling; cannot resolve sub-token ECG–PPG delay) | CHANGELOG §1 row 2; FM04 §4.4 (line 141) | O-V |
| B → pairwise lag-interaction module (control; component only if Stage 1 requires it) | FM04 §4.4 (line 167) | O-V |
| lag bias and pairwise module → Stage 1 delay-resolution test (MAE ≤ 10 ms, r ≥ 0.5) | FM04 §4.4 (line 176); FM07 Stage 1 | G-EXT |
| B depends on synchronisation classes A–D | FM04 §2 "Time" (line 46); FM03 §5a | O-V |
| Synchronisation classes → pairwise module (only within classes A/B) | FM03 §5a "Consequences" | G-EXT |
| MIMIC-III waveforms excluded from the decisive experiment (changing delay up to 500 ms) | FM07 Stage 3 Dataset (line 86); FM03 §5a (line 98) | O-V |

### 2.3 Cross-modal attention and fusion

| Relationship | Evidence | Class |
|---|---|---|
| B → time-stamped fusion tokens (bottleneck) | FM04 §4.4 | G-EXT |
| CardioState-JEPA and xMAE → B (closest prior work; B extends them to anchor-free lags and > 2 modalities) | FM04 §4.4 (line 139); FM02 §3.6 | G-EXT |
| CardioState-JEPA → fusion variants (anchor-supervised delay arm) | FM04 §4.4 variant table | G-EXT, O-V |
| Core contribution ("controlled evidence on what cross-modal pretraining learns") → fusion variants | FM02 §6 | G-INF 0.85 |
| DEAP-era gated fusion → FM Architecture A and the "late gated" variant | D03 §4.2 vs FM04 §3, §4.4 | O-I 0.95 |
| DEAP-era B-xattn capacity-matched control → "no-exchange control" | D03 §4.3 vs FM04 §4.4 | O-I 0.95 |

### 2.4 O1 masked latent prediction

| Relationship | Evidence | Class |
|---|---|---|
| B → objective O1 | FM04 §4.5 | G-EXT |
| O1 implements B (FM05 node) | FM05 §1, §5 | G-EXT (on the duplicate node) |
| FM04 "Objective O1" = FM05 "O1 masked latent prediction" | FM04 §4.5 refers to FM05 | O-V (`same_as`) |
| O1 → masking scheme (span, whole-modality, real gaps) → interpolation problem | FM05 §5 lines 64–67; §4 | G-EXT |
| O1 → collapse safeguards (failure mode) | FM05 §9 | G-EXT, O-V |
| O2 fallback → interpolation (failure mode) | FM05 §4 | O-V |
| Stage 1 can switch the primary objective to O2 | FM07 Stage 1 Go/no-go (line 65) | O-V |

### 2.5 Unimodal and cross-modal pretraining

| Relationship | Evidence | Class |
|---|---|---|
| Unimodal phase P0 then joint phase P1 (curriculum) | FM03 §8; FM05 §8 first row | G-EXT (curriculum → MESA) |
| B pretrained on MESA Sleep (prototype tier, ≈ 500 participants) | FM03 §7; FM07 Stage 3 Dataset | G-INF 0.75 → **O-V** |
| Cross-modal vs unimodal pretraining = K2, in Stage 3 | FM06 §6; FM07 Stage 3 | O-V |
| Hybrid reuse strategy (public EEG checkpoints) motivated by EEG foundation-model audits | FM02 §2 point 4, §5 | O-V |
| Stage 2 failure can switch to reused checkpoints | FM07 Stage 2 Go/no-go (lines 78–79) | O-V |

### 2.6 Missing-modality handling

Not represented in `graph.json` at all. Added in the overlay:

| Relationship | Evidence | Class |
|---|---|---|
| B → missing-modality handling (absent streams give no keys; learned "absent" flags) | FM04 §4.4 (line 180) | O-V |
| trained by whole-modality masking and real-gap masking | FM05 §5 (lines 66–67) | O-V |
| B → reliability-aware gating (learned per-token quality score) | FM04 §4.4 (line 182) | O-V |
| both tested in Stage 6 (missing-modality matrix, corruption suite, gate-input comparison) | FM07 Stage 6; FM06 §9 | O-V |
| DEAP-era topological quality gating C-Q → reliability gating | D03 §5.2 vs FM04 §4.4 | O-I 0.75 |

### 2.7 Dataset selection

| Relationship | Evidence | Class |
|---|---|---|
| MESA → prototype tier | FM03 §7 | G-EXT |
| B → MESA (pretraining) | as 2.5 | O-V |
| B evaluated in cells E1–E8 (primary E2 CinC 2018, E3 WESAD chest) | FM07 Stage 3 Evaluation; FM06 §3.2 | O-V |
| CinC 2018 → cell E2 | FM06 §3.2; FM03 §7 | O-V (node was isolated) |
| CinC 2018 also listed for pretraining (P2, open substitute, research tier) | FM03 §7–8 | O-V `conflicts_with` (see consistency report) |

### 2.8 Transfer-learning evaluation

| Relationship | Evidence | Class |
|---|---|---|
| B → evaluation cells; adaptation protocols (linear → attentive probe → adapters → full fine-tuning) | FM06 §3.2, §4 | O-V (cells); graph links only via the FM06 document node |
| Per-second shared tokens and pooled private embeddings are B's transfer outputs | FM04 §4.5 (line 203); FM05 §10 (line 142) | not in graph |
| Milestone 1 `Embedder` returns window vectors only, so attentive probes on B's tokens are not supported | FM08 §8 (line 122) vs FM06 §4 (line 97) | O-V `conflicts_with` |

### 2.9 Comparisons K1–K5

| Relationship | Evidence | Class |
|---|---|---|
| K1 (pretrained vs random), K5 (specific vs shared encoders) tested in Stage 2 | FM07 Stage 2 | O-V |
| K2 (cross- vs unimodal), K3 (interaction vs late vs no-exchange, + pairwise arm), K4 (true vs perturbed pairing) tested in Stage 3 | FM07 Stage 3; FM06 §6 | O-V, partly G-EXT via the Stage 3 hyperedge |
| H1–H7 map to K1–K5 (H1–K1, H2–K5, H3–K2, H4–K3, H5–K4) | FM07 §1 | O-V |
| K3 and K4 definitions revised in the latest revision | CHANGELOG §1 rows 2–3 | O-V |
| B → K1–K5: **no direct edge in the graph**; the link runs only through the Stage 3 hyperedge | — | — |

### 2.10 Implementation milestones

| Relationship | Evidence | Class |
|---|---|---|
| B → resource benchmark (= Stage 0b = Milestone 1b) | FM04 §4.6; FM07 Stage 0b | G-EXT, O-V `same_as` |
| Milestone 1 (FM08) implements a subset of Stage 0 | FM07 Stage 0 Datasets (line 35) | O-V |
| Milestone 1 provides the `Embedder` protocol into which B will be plugged | FM08 §8 | not in graph |
| Milestone 2 = synthetic coupling benchmark (Stage 1) tests small versions of B's fusion variants | FM08 §14; FM07 Stage 1 | O-V |
| Milestone 1 does **not** depend on any open decision about B | FM08 §1 "Out of scope" | — |

## 3. What the trace shows

1. **B's decision chain is complete in the source reports but not in the graph.** In `graph.json` B reaches its components, prior work and MESA, but not its tests (K1–K5, Stages 1–3, 6), its missing-modality mechanism, its evaluation cells or its synchronisation dependency. The overlay supplies 12 direct edges.
2. **Every B component has a deciding experiment except four minor ones** (see the dependency audit, issue M6): EEG two-path tokeniser, factorised EEG attention, metadata embeddings, absent flags.
3. **Go/no-go decisions that can remove parts of B:**
   * Stage 1: adopt the pairwise module, or switch to O2.
   * Stage 2: switch to a shared encoder (K5) or to reused checkpoints.
   * Stage 3: drop the lag term, or fall back to Architecture A.

   All six are now explicit `decides` edges.
4. **B's centrality is partly an artefact** of the duplicate node and of the missing links between the two report generations: the graph had 0 edges between DEAP-era and foundation-model reports.
