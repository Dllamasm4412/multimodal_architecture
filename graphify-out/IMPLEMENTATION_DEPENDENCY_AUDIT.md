# Implementation Dependency Audit

Scientific and implementation risks surfaced by tracing the research graph and checking the cited source sections. Every issue below was verified against the report text, except where marked [inference]. No report, architecture or code was changed; recommended fixes are for the author to apply.

**Severity.** *High*: could produce an invalid result or a wrong implementation. *Medium*: weakens or delays a later decision. *Low*: wording, bookkeeping or cost.
**Scope.** **M1**: affects Milestone 1 (FM08). **Later**: affects a later stage only.

## 1. Summary

| ID | Issue | Severity | Scope |
|---|---|---|---|
| H1 | Two complete "Milestone 1" handoffs; the superseded DEAP one carries no notice | High | **M1** |
| H2 | CinC 2018 is both the primary cross-dataset evaluation cell and a pretraining source | High | Later |
| H3 | Labels K1–K4, Stage 1–6, Architecture A/B/C, Milestone 2 mean different things in the two report generations | High | M1 (handoff context) and later |
| M1 | Milestone 1 does not produce the baseline for primary cell E3 (chest ECG + respiration) | Medium | **M1** |
| M2 | `Embedder` protocol cannot expose token embeddings, which attentive probes on B need | Medium | **M1** (interface), used later |
| M3 | B's 30 s window and five-modality design do not match cell E7 (8 s windows, wrist accelerometer) | Medium | Later |
| M4 | No staged hypothesis or decisive test for EEG cross-modal (brain–body) learning | Medium | Later |
| M5 | Primary cells E2 and E3 contain no PPG, so ECG–PPG coupling never enters a confirmatory test | Medium | Later |
| M6 | "Window-level cross-modal learning" on poorly synchronised data has no defined objective | Medium | Later |
| M7 | Resource benchmark omits the 0.25 s fusion-token configuration that Stage 1 may select | Medium | Later (Milestone 1b) |
| L1 | Storage estimate assumes 16-bit; Milestone 1 stores float32 | Low | M1 |
| L2 | Stage 2 dependencies omit Stage 0b | Low | Later |
| L3 | Four Architecture B components have no deciding test | Low | Later |
| L4 | Riemannian-token admission test relies on small, possibly chance-level multichannel EEG sets | Low | Later |
| L5 | Stage 0 acceptance requires cells (E2) that Milestone 1 does not build | Low | M1 (expectation) |

**Milestone 1 is not blocked.** H1, M1, M2, L1 and L5 touch it and are cheap to resolve before coding (section 4). Nothing in Milestone 1 depends on an unresolved scientific decision.

## 2. Issues

### H1 — Superseded handoff without a notice (High, M1)

* **Evidence.** `docs/05_IMPLEMENTATION_HANDOFF.md` is a self-contained "Milestone 1" specification for a DEAP-only harness. FM08 line 5 says it supersedes that file, but D05 itself says nothing. In the graph the two handoffs sat in different communities with no link.
* **Risk.** A coding model given the repository, or asked to "implement Milestone 1", may build the DEAP harness: wrong datasets, no input policy, no synchronisation metadata.
* **Fix.** Give the coding model FM08 only, by path. Add a one-line superseded notice at the top of D05 (and of D03/D04, which are background). This is an edit to a research report, so it is left to you.

### H2 — CinC 2018 in both roles (High, later)

* **Evidence.**
  * FM06 §3.2 makes CinC 2018 primary cell **E2**, the cross-dataset read-out for K1–K4.
  * FM03 §8 (line 166) adds the EEG stream "on MESA, SHHS, CinC 2018" in curriculum phase P2.
  * The open-data substitute in FM03 §7 pretrains on "the labelled half of CinC 2018".
  * The research tier lists it as "optionally" pretraining data.
  * FM03 §6 says such datasets get one role per study, but the plan assigns both.
* **Risk.** E2 stops being cross-dataset. K1–K4 confirmatory results would be inflated by domain proximity or participant overlap.
* **Fix (decision).** Reserve CinC 2018 for evaluation in every phase, and replace it in P2 and in the open substitute with SHHS. Or, if NSRR access fails, choose a different E2 dataset before any pretraining.

### H3 — Label collisions (High)

* **Evidence.** CROSS_DOCUMENT_CONSISTENCY §3. D04-K2 ("modality-specific vs homogeneous encoders") is FM-K5; FM-K2 is "cross-modal vs unimodal pretraining". Stages 1–6 differ in content under the same numbers. "Milestone 2" is a topology module in D05 and the synthetic coupling benchmark in FM08.
* **Risk.** A prompt such as "implement the K2 comparison" or "run Stage 3" is ambiguous. A model that read both generations can implement the wrong experiment and still believe it is correct.
* **Fix.** Always qualify with the report (FM06-K2, FM07-Stage 3). Use the terminology table in CROSS_DOCUMENT_CONSISTENCY §5.

### M1 — E3 baseline missing from Milestone 1 (Medium, M1)

* **Evidence.** The primary cell E3 uses chest ECG + respiration (FM06 §3.2; FM07 Stage 3 Evaluation). FM08 §7 defines `stress_chest` as ECG + respiration + EDA, with only a "without respiration" variant. FM06 T2 (line 33) also lists EDA.
* **Risk.** Stage 3 would lack a conventional baseline on the identical input set, which FM06 §3.2 requires for any comparison in a cell.
* **Fix.** Add one declared variant to FM08 §7: `stress_chest` with ECG + respiration only.

### M2 — `Embedder` returns only window vectors (Medium, M1 interface)

* **Evidence.** FM08 §8: `transform(windows) -> [n_windows, dim]`, and "later milestones plug neural encoders into this protocol unchanged". FM06 §4 requires attentive probes "on frozen token embeddings". FM04 §4.5 and FM05 §10 expose per-second shared tokens.
* **Risk.** The protocol is frozen in Milestone 1 and must change when B arrives, after baselines have been computed through it.
* **Fix.** Add an optional `transform_tokens(windows) -> [n_windows, n_tokens, dim]` with token timestamps. Hand-crafted embedders need not implement it.

### M3 — Window and input-set mismatch with E7 (Medium, later)

* **Evidence.** FM04 §2: 30 s pretraining windows, "downstream 30–60 s", and 30 fusion tokens (§4.4). FM08 `hr_wrist` and FM06 E7 use 8 s windows and wrist accelerometer. B has no accelerometer stream (FM04 §4.2).
* **Risk.** On E7, B and the Milestone 1 baselines would not see an identical input set. Whether B handles 8 s windows is unspecified.
* **Fix (decision, before Stage 2).** Either evaluate B on E7 with BVP only (the declared `hr_wrist` BVP-only variant gives the matched baseline), or specify variable-length windows and an accelerometer tokeniser.

### M4 — EEG cross-modal learning has no decisive test (Medium, later)

* **Evidence.** FM03 §8 P2 targets "brain–body coupling in sleep". FM07 §5 says EEG "joins cross-modal training immediately after this experiment". FM07 has no stage, hypothesis or acceptance criterion for it, and H1–H7 do not mention EEG.
* **Risk.** The modality you asked to keep first-class is the one cross-modal claim without a planned test. It would be evaluated ad hoc.
* **Fix.** Add a Stage 3E (or a K-comparison) mirroring Stage 3 arms 1, 3, 4, 5a and 5b with EEG added, evaluated on EEG-containing cells with the label-source flag.

### M5 — Confirmatory cells exclude PPG (Medium, later) [design observation]

* **Evidence.** E2 is ECG + respiration (CinC 2018 has no PPG). E3 is chest ECG + respiration. ECG–PPG (pulse-arrival) coupling, the best-documented coupling and the target of the pairwise module, appears only in secondary cells E1 (within-corpus) and E5 (CASE, 30 participants).
* **Risk.** K3 and K4 can be confirmed or refuted without the ECG–PPG pair ever being tested. A positive beat-level ECG–PPG claim would rest on exploratory cells.
* **Fix.** Accept, and state that confirmatory claims concern ECG–respiration coupling. Or promote E1 to a third primary cell (within-corpus, so read together with E2).

### M6 — Undefined "window-level" cross-modal objective (Medium, later)

* **Evidence.** FM03 §5a: MIMIC-III remains usable "for window-level cross-modal learning". FM05 §8: lag-sensitive training only on adequately synchronised recordings. O1 is defined only with time-stamped token targets (FM05 §5).
* **Risk.** Mixing class C/D data into O1 in curriculum phase P4 would train lag kernels on misaligned signals. No loss variant or data-loader rule prevents it.
* **Fix (before Stage 5).** Define a pooled-target variant of the cross-modal loss for class C/D recordings and enforce it in the pretraining loader from `meta.json` synchronisation fields. Milestone 1 already stores those fields.

### M7 — Resource benchmark misses a candidate configuration (Medium, Milestone 1b)

* **Evidence.** FM04 §4.4: if one-second fusion tokens fail the delay test, fusion tokens every 0.25 s are tried. That is 120 fusion tokens instead of 30. The FM04 §4.6 sweep varies the pairwise module but not the fusion-token rate.
* **Fix.** Add fusion-token rate (1 s, 0.25 s) to the Milestone 1b sweep.

### L1 — Storage precision (Low, M1)

FM03 §9 (line 174) estimates 4 MB per recording-hour at 16-bit. FM08 §4 stores float32, about 8.6 MB per hour, so the prototype corpus is ≈ 34 GB rather than ≈ 17 GB. Either is fine; the Milestone 1 data benchmark measures the real figure.

### L2 — Stage 2 dependency (Low)

FM07 Stage 0b's go/no-go applies "before Stage 2", but Stage 2 lists only "Stage 0" as a dependency (line 75).

### L3 — Components without a deciding test (Low)

These B components carry no ablation:
* EEG two-path tokeniser (FM04 §4.2);
* factorised EEG attention (§4.3);
* sensor-site and electrode metadata embeddings (§2);
* "absent" flags (§4.4).

None blocks anything. Add ablations only if a result depends on them.

### L4 — Riemannian token test data (Low)

FM07 Stage 4 admits the Riemannian EEG token on "multichannel EEG downstream sets": DEAP (32 participants) and AMIGOS (40), where earlier work found arousal near chance. The admission test may be unpowered. [inference]

### L5 — Milestone 1 vs Stage 0 acceptance (Low, M1 expectation)

FM07 Stage 0 acceptance needs baselines above the null on E2 and E3. Milestone 1 builds neither exactly (E2 is deferred; E3 needs the M1 variant). Completing Milestone 1 therefore does not complete Stage 0. State this when reviewing Milestone 1.

## 3. Checks that passed

| Question | Result |
|---|---|
| Architecture components without a supporting hypothesis | Lag bias (H5/K4), pairwise module (Stage 1, K3), fusion bottleneck (H4/K3), O1 (Stage 1–2), missing-modality handling and reliability gating (Stage 6), scale descriptors (Stage 2 ablation), private encoders (H2/K5): all covered. Exceptions in L3 |
| Hypotheses without a decisive experiment | H1–H7 each map to a stage and comparison; H7 is exploratory by design. Exception: EEG cross-modal (M4) |
| Synchronisation requirements enforced in Milestone 1 | Yes: `meta.json` fields, `beat_level_ok`, coupling only within class A/B, tested (FM08 §4, §6, §11 test 4). Not yet enforced in any pretraining loader (M6) |
| Input-policy enforcement | Specified and tested in Milestone 1 (FM08 §7, §11 test 3) |
| Conflicting synchronisation-class definitions | None: FM03 §5a and FM08 §6 agree |
| Milestone 1 depends on an unresolved scientific decision | No. Bands, rates, windows, tasks and policies are fixed. Open decisions (fusion-token rate, pairwise module, O1/O2, encoder sharing, lag term) all come later |
| Throughput figures consistent across reports | Yes after the latest revision (FM04 §4.6, FM07 §2 and §5, CHANGELOG) |

## 4. Before handing Milestone 1 to a coding model

1. Pass `docs/foundation_model_research/08_IMPLEMENTATION_HANDOFF.md` alone, by path. Do not pass D05 (H1).
2. Optionally amend FM08 §7 with the `stress_chest` ECG + respiration variant (M1).
3. Optionally amend FM08 §8 with an optional token-level `Embedder` method (M2).
4. Expect float32 storage (L1) and record that Milestone 1 completes only part of Stage 0 (L5).

Decisions to make before pretraining, not before Milestone 1: H2 (CinC 2018 role), M3 (E7 inputs), M4 (EEG cross-modal stage), M5 (whether ECG–PPG needs a confirmatory cell), M6 (objective for poorly synchronised data), M7 (benchmark sweep).
