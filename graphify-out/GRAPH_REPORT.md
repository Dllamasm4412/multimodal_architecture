# Graph Report - multimodal_architecture  (2026-11-16)

## Corpus Check
- Corpus is ~43,291 words - fits in a single context window. You may not need a graph.

## Summary
- 142 nodes · 155 edges · 19 communities (10 shown, 9 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 1% AMBIGUOUS · INFERRED: 23 edges (avg confidence: 0.81)
- Token cost: 365,432 input · 0 output

## Community Hubs (Navigation)
- Self-Supervised Objectives and Controls
- Topology Approaches and DEAP Architectures
- Cross-Modal Prior Art and Fusion
- DEAP Review and Roadmap
- Transfer Evaluation Framework
- Lag Coupling and Synchronisation
- Milestone 1 Data Layer
- Signal-Processing Shortlist
- DEAP Milestone 1 Baseline
- Checkpoint Reuse Strategy
- Hypotheses and Resource Tiers
- DEAP Evaluation Protocol
- Missing-Modality EEG Models
- EEG Foundation Model Audits
- Large Sleep Corpora
- PPG Hub and EDA Data
- Prior Art Report
- Dataset Strategy Report
- CinC 2018 Dataset

## God Nodes (most connected - your core abstractions)
1. `Architecture B: private encoders with time-aware bottleneck fusion` - 11 edges
2. `06 Transfer-Learning Evaluation Framework` - 9 edges
3. `Stage 3 cross-modal pretraining decisive experiment` - 9 edges
4. `O1 Masked latent prediction with EMA teacher (primary objective)` - 8 edges
5. `Architecture C (B-gated plus topology adapters)` - 6 edges
6. `Synchronisation classes A-D` - 6 edges
7. `Changelog technical revision before implementation` - 6 edges
8. `Falsification experiments F0-F6` - 5 edges
9. `Stage 1 Classical baselines and noise floor` - 5 edges
10. `Stage 3 Physiological topology decisive test` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Pairing-perturbation controls P0-P5 (K4)` --semantically_similar_to--> `Interpolation problem by modality`  [INFERRED] [semantically similar]
  docs/foundation_model_research/06_TRANSFER_LEARNING_EVALUATION.md → docs/foundation_model_research/05_SELF_SUPERVISED_LEARNING_STRATEGY.md
- `Masked gated fusion with auxiliary unimodal heads and modality dropout` --semantically_similar_to--> `FFTNet (Fein-Ashley 2025)`  [INFERRED] [semantically similar]
  docs/03_ARCHITECTURE_RECOMMENDATION.md → docs/01_EVIDENCE_REVIEW.md
- `PaPaGei` --conceptually_related_to--> `MESA Sleep dataset`  [AMBIGUOUS]
  docs/foundation_model_research/02_FOUNDATION_MODEL_PRIOR_ART.md → docs/foundation_model_research/03_DATASET_AND_PRETRAINING_STRATEGY.md
- `05 Implementation Handoff (Milestone 1)` --conceptually_related_to--> `04 Experimental Roadmap`  [INFERRED]
  docs/05_IMPLEMENTATION_HANDOFF.md → docs/04_EXPERIMENTAL_ROADMAP.md
- `05 Implementation Handoff (Milestone 1)` --implements--> `Stage 0 Data and evaluation audit`  [INFERRED]
  docs/05_IMPLEMENTATION_HANDOFF.md → docs/04_EXPERIMENTAL_ROADMAP.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Controls that test whether topology adds non-spectral information** — docs_02_critical_assessment_falsification_f0_f6, docs_01_evidence_review_iaaft_surrogates, docs_04_experimental_roadmap_stage3, docs_05_implementation_handoff_feature_groups [INFERRED 0.85]
- **Architecture ladder A to B-gated to C** — docs_03_architecture_recommendation_arch_a, docs_03_architecture_recommendation_b_gated, docs_03_architecture_recommendation_b_xattn, docs_03_architecture_recommendation_arch_c [EXTRACTED 1.00]
- **Milestone 1 harness, fixture and baselines** — docs_05_implementation_handoff_harness, docs_05_implementation_handoff_synthetic_fixture, docs_05_implementation_handoff_reference_baselines, docs_05_implementation_handoff_feature_groups [EXTRACTED 1.00]
- **Architecture C signal-processing extensions admitted individually** — docs_foundation_model_research_01_signal_processing_landscape_physiological_phase_encoding, docs_foundation_model_research_01_signal_processing_landscape_parametrised_front_ends, docs_foundation_model_research_01_signal_processing_landscape_riemannian_eeg_covariance, docs_foundation_model_research_01_signal_processing_landscape_sp_targets_and_probes [EXTRACTED 0.95]
- **Closest prior work on cross-modal cardiac pretraining** — docs_foundation_model_research_02_foundation_model_prior_art_cardiostate_jepa, docs_foundation_model_research_02_foundation_model_prior_art_xmae, docs_foundation_model_research_02_foundation_model_prior_art_ctaf [EXTRACTED 0.90]
- **Architecture B core components** — docs_foundation_model_research_04_architecture_options_tokenisers, docs_foundation_model_research_04_architecture_options_fusion_tokens, docs_foundation_model_research_04_architecture_options_lag_bias_attention, docs_foundation_model_research_04_architecture_options_objective_o1 [EXTRACTED 0.95]
- **Stage 3 decisive cross-modal experiment (K2, K3, K4 arms)** — docs_foundation_model_research_07_experimental_roadmap_stage3, docs_foundation_model_research_06_transfer_learning_evaluation_k1_k5, docs_foundation_model_research_06_transfer_learning_evaluation_pairing_perturbation, docs_foundation_model_research_05_self_supervised_learning_strategy_o1_masked_latent_prediction [INFERRED 0.85]
- **Milestone 1 harness: format, sync audit, policy, embedders, splits** — docs_foundation_model_research_08_implementation_handoff_unified_format, docs_foundation_model_research_08_implementation_handoff_sync_audit, docs_foundation_model_research_08_implementation_handoff_policy_py, docs_foundation_model_research_08_implementation_handoff_embedder_protocol, docs_foundation_model_research_08_implementation_handoff_splits_py [EXTRACTED 1.00]

## Communities (19 total, 9 thin omitted)

### Community 0 - "Self-Supervised Objectives and Controls"
Cohesion: 0.12
Nodes (19): 05 Self-Supervised Learning Strategy, Architecture B (private encoders plus fusion tokens), Collapse and shortcut safeguards, Cross-modal contrastive alignment baseline, Interpolation problem by modality, Span, whole-modality and real-gap masking, Confirmatory comparisons K1-K5, Pairing-perturbation controls P0-P5 (K4) (+11 more)

### Community 1 - "Topology Approaches and DEAP Architectures"
Cohesion: 0.12
Nodes (15): Brookshire et al. 2024 segment-level leakage, CoRe-Sleep (Kontras 2024), FFTNet (Fein-Ashley 2025), Gravot et al. 2025 transmission-line fault classification, Perea and Harer 2015 sliding-window persistence, Shao et al. 2025 PPG topological quality, Yan et al. 2022 topological EEG emotion recognition, Architecture C (B-gated plus topology adapters) (+7 more)

### Community 2 - "Cross-Modal Prior Art and Fusion"
Cohesion: 0.12
Nodes (16): Sparse deconvolution of EDA (cvxEDA), CardioState-JEPA, CTAF, PaPaGei, xMAE, Pretraining curriculum P0-P4, MESA Sleep dataset, Prototype tier: about 500 MESA participants (4,000 h) (+8 more)

### Community 3 - "DEAP Review and Roadmap"
Cohesion: 0.16
Nodes (16): 01 Evidence Review, IAAFT surrogate data (Schreiber and Schmitz 1996), LibEER benchmark (Liu 2024), 02 Critical Assessment, Falsification experiments F0-F6, Shared-stimulus confound (40 videos), 03 Architecture Recommendation, Strict vs calibrated normalisation regimes (+8 more)

### Community 4 - "Transfer Evaluation Framework"
Cohesion: 0.16
Nodes (14): 06 Transfer-Learning Evaluation Framework, Adaptation protocols (linear probe to full fine-tuning), Evaluation cells E1-E8, Inductive, transductive and personalised settings, Input policy for privileged signals (section 3.1), Label-efficiency protocol, Leakage and overlap audit, Downstream task suite T1-T7 (+6 more)

### Community 5 - "Lag Coupling and Synchronisation"
Cohesion: 0.20
Nodes (9): Signal-Processing Landscape Report, CASE dataset, DEAP dataset, MIMIC-III Waveform dataset, VitalDB dataset, WESAD dataset, Lag-aware cross-attention bias on fusion tokens, Pairwise lag-interaction module (+1 more)

### Community 6 - "Milestone 1 Data Layer"
Cohesion: 0.27
Nodes (10): Stage 0 data layer, harness, baselines, Stage 0b resource benchmark, 08 Implementation Handoff Milestone 1, bench/data_throughput.py data-layer benchmark, Datasets WESAD, PPG-DaLiA, HMC, Synchronisation audit and classes A-D, Unified recording format (layer 0 raw, layer 1 canonical), Changelog technical revision before implementation (+2 more)

### Community 7 - "Signal-Processing Shortlist"
Cohesion: 0.29
Nodes (3): Persistent homology, recurrence and higher-order spectra (probe only), Wavelet scattering, Architecture C: B plus signal-processing extensions

### Community 8 - "DEAP Milestone 1 Baseline"
Cohesion: 0.40
Nodes (5): DEAP dataset (Koelstra 2012), Architecture A (A0 classical, A1 compact neural), Control feature groups (eeg_spec, bvp, eda, resp, temp, confound, ctrl_nonlinear), Leakage-proof evaluation harness, Synthetic DEAP fixture make_synthetic_deap

### Community 10 - "Hypotheses and Resource Tiers"
Cohesion: 0.67
Nodes (3): 07 Experimental Roadmap, Hypotheses H1-H7, Resource tiers (prototype, research, foundation)

## Ambiguous Edges - Review These
- `PaPaGei` → `MESA Sleep dataset`  [AMBIGUOUS]
  docs/foundation_model_research/02_FOUNDATION_MODEL_PRIOR_ART.md · relation: conceptually_related_to

## Knowledge Gaps
- **62 isolated node(s):** `01 Evidence Review`, `Gravot et al. 2025 transmission-line fault classification`, `CoRe-Sleep (Kontras 2024)`, `DEAP dataset (Koelstra 2012)`, `LibEER benchmark (Liu 2024)` (+57 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 70 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Architecture B: private encoders with time-aware bottleneck fusion` connect `Cross-Modal Prior Art and Fusion` to `Lag Coupling and Synchronisation`, `Signal-Processing Shortlist`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **What connects `01 Evidence Review`, `Gravot et al. 2025 transmission-line fault classification`, `CoRe-Sleep (Kontras 2024)` to the rest of the system?**
  _62 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Self-Supervised Objectives and Controls` be split into smaller, more focused modules?**
  _Cohesion score 0.11688311688311688 - nodes in this community are weakly interconnected._
- **Why does `06 Transfer-Learning Evaluation Framework` connect `Transfer Evaluation Framework` to `Self-Supervised Objectives and Controls`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Should `Topology Approaches and DEAP Architectures` be split into smaller, more focused modules?**
  _Cohesion score 0.11578947368421053 - nodes in this community are weakly interconnected._
- **Should `Cross-Modal Prior Art and Fusion` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `PaPaGei` and `MESA Sleep dataset`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._