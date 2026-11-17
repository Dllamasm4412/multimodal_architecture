# 03 — Dataset and Pretraining Strategy

Which datasets can support self-supervised pretraining and transfer evaluation, which cross-modal relationships the available data can actually teach, and an incremental plan. Searches were run on 2026-11-16.

**Status tags.** **V** = verified on the dataset's page or paper during this review. **P** = search snippet or secondary source only. **R** = my recollection, not checked. **E** = my estimate from verified counts. Nothing tagged R or E should be relied on before the Stage 0 audit confirms it.

## 1. Bottom line

1. **DEAP should not be a pretraining dataset.** It has 32 participants. It remains one far-transfer evaluation task.
2. **Sleep studies are the only public source with thousands of participants and several synchronised physiological systems.** MESA Sleep is the single most valuable corpus because it records ECG, a finger PPG waveform, respiratory effort and EEG together. SHHS is larger but has no PPG waveform.
3. **The data cannot teach every relationship the project is interested in.** Fine coupling between EEG and autonomic signals can be learned at scale only during sleep and only with two or three EEG channels. Coupling between EEG and EDA during wakefulness exists in public data for a few tens of participants. This limits the claims, not the architecture.
4. **PPG is the hub modality.** It is the only signal present at scale both in clinical sleep recordings (with ECG, respiration, EEG) and in wearable recordings (with EDA). Any bridge between the two worlds runs through it, across a finger-to-wrist sensor shift.
5. **A prototype needs about 4,000 hours, not a foundation-scale corpus.** That is roughly 500 MESA participants and about 17 GB after preprocessing.

## 2. Pretraining candidates

| Dataset | Participants / recordings | Hours | Synchronised signals and rates | Access | Size | Notes |
|---|---|---|---|---|---|---|
| **MESA Sleep** (NSRR) | 2,056 participants with raw sleep recordings (V) | ≈ 17,000–20,000 (E) | ECG 256 Hz; **PPG waveform 256 Hz**; thoracic and abdominal effort 32 Hz; nasal pressure 32 Hz; EEG 3 channels (Fz-Cz, Cz-Oz, C4-M1) 256 Hz; EOG, EMG; SpO₂ 1 Hz (V). No EDA | Online data access and use agreement reviewed by NSRR, free, renewable after three years (V/P). Commercial use of MESA needs a separate request (P) | not verified | In-home recordings; older adults (R). The only large corpus with ECG + PPG + respiration + EEG together |
| **SHHS** (NSRR) | Raw recordings for 5,793 participants at visit 1 and 2,651 at visit 2 (V) | ≈ 70,000 (E) | ECG 125 Hz (visit 1), 250–256 Hz (visit 2); EEG C3-A2, C4-A1 125 Hz; thoracic and abdominal effort 10 Hz; airflow 10 Hz; SpO₂ and heart rate 1 Hz (V). **No PPG waveform (V)**. No EDA | As MESA | not verified | Largest ECG + respiration + EEG corpus |
| **PhysioNet/CinC 2018** | 1,985 participants, 994 with labels (V) | ≈ 15,000 (E) | 13 channels at 200 Hz: six EEG derivations, EOG, chin EMG, abdominal and chest effort, airflow, SaO₂, ECG (V, file header). No PPG waveform | Open (V) | 266.6 GB, about half of it the labelled set (V) | Sleep-clinic population; sleep stages and arousal annotations |
| **VitalDB** | 6,388 surgical cases (V) | ≈ 17,000 (E, R) | ECG lead II and PPG in ≈ 6,200–6,350 cases, arterial pressure in ≈ 3,650, two frontal EEG waveforms in ≈ 5,870 (V, track list); ECG, PPG and pressure at 500 Hz, EEG at 128 Hz (P) | Open, CC BY 4.0 (V) | 94 GB (V) | General anaesthesia: drugs and ventilation alter every signal |
| **MIMIC-III Waveform, matched subset** | 22,317 records, 10,282 patients (P) | tens of thousands (E) | ECG, PPG, impedance respiration, 125 Hz (P) | Open access, ODbL (V); linking to clinical records needs credentialing | 2.4 TB (V) | Intensive care; a subset is enough |
| **Wristband pool (EDAMAME)** | 24 public wristband datasets, 634 users (V) | ≈ 24,700 in total; **≈ 3,800 h from ≈ 190 users in the 11 openly licensed sets** (V, paper tables) | EDA 4 Hz; the device also records BVP 64 Hz, temperature and accelerometer, but whether each dataset's files include BVP is not verified | 11 datasets openly licensed, 11 under one data-sharing agreement, 2 from authors; location of the pooled release not found (P) | not verified | The only route to EDA at scale; daily-life recordings |
| **DREAMT** | 100 participants (V) | ≈ 800 (E) | Wrist BVP 64 Hz, EDA 4 Hz, temperature, accelerometer; sleep-lab EEG, ECG, respiration at 100 Hz; sleep stages (V) | Restricted data-use agreement (V) | not verified | Bridges wristband and sleep-lab signals; better used for evaluation |
| **Temple University EEG corpus** | ≈ 26,800 sessions, ≈ 15,000 patients (P) | 10,000–21,000 depending on counting (P, E) | Clinical multichannel EEG | Signed agreement; approval in days (P) | large | Source of spatial EEG structure; already used by public EEG models, so reuse them instead |

## 3. Downstream evaluation candidates

| Dataset | Participants | Signals | Labels | Access | Role (task in report 06) |
|---|---|---|---|---|---|
| **WESAD** | 15 (V) | Chest: ECG, EDA, EMG, respiration, temperature, accelerometer at 700 Hz. Wrist: BVP 64 Hz, EDA 4 Hz, temperature, accelerometer (V) | baseline / stress / amusement (V) | Open download (V) | T2, primary far-transfer task. Chest and wrist devices have separate clocks (V) |
| **CASE** | 30 (V) | ECG, BVP, EDA, respiration, skin temperature, 3 EMG at 1000 Hz from one acquisition system (V) | Continuous valence and arousal by joystick, 20 Hz (V) | Open (figshare, ≈ 5 GB) (V) | T3. Best arousal dataset for this project: continuous labels, cardiorespiratory + EDA, one clock. No EEG |
| **DEAP** | 32 | 32-channel EEG, BVP, EDA, respiration, temperature | trial ratings | Licence agreement | T3, far transfer with EEG; see earlier reports |
| **AMIGOS** | 40 (V, earlier review) | 14-channel EEG, ECG, EDA | ratings | Licence agreement | T3, far transfer |
| **DREAMER** | 23 (P) | Consumer EEG, ECG (P); rates R | ratings | Request form (P) | optional |
| **PhysioNet/CinC 2018** (if not used for pretraining) | 994 labelled | EEG, ECG, respiratory effort, airflow (V) | sleep stages, arousals | Open | T1 cross-dataset, T6 |
| **DREAMT** | 100 | wrist BVP, EDA + sleep-lab signals | sleep stages | Restricted agreement | T1 on wearable signals |
| **PTB-XL**, one lead | 21,799 records, 18,869 patients (V) | 12-lead ECG, 10 s; any single lead usable | diagnostic and rhythm statements; atrial-fibrillation label to be confirmed in the statement table | Open (V) | T4. Its recommended folds keep each patient in one fold (V) |
| PhysioNet/CinC 2017 | 8,528 recordings (P) | single-lead ECG, 300 Hz | rhythm class | Open (P) | exploratory only: no documented patient identifiers, so participant-disjoint splits cannot be guaranteed (P) |
| **PPG-DaLiA** | 15 (V) | wrist BVP, EDA, temperature, accelerometer; chest ECG, respiration, accelerometer (V) | heart rate from ECG; activity | Open, CC BY 4.0, 2.7 GB (V) | T5; naturally motion-corrupted PPG |
| **BIDMC** | 53 recordings × 8 min (V) | ECG, PPG, impedance respiration, 125 Hz (V) | respiratory rate | Open (V) | T5; **extracted from MIMIC-II (V)**, so not to be combined with MIMIC pretraining |
| Small open sleep sets (HMC: 151 recordings; ISRUC: 118 participants; Sleep-EDF Expanded: 197 recordings) | — | EEG, EOG, EMG confirmed; HMC also has one ECG lead (V); ECG not confirmed for ISRUC; none in Sleep-EDF | sleep stages | Open | HMC: small open set for T1 with ECG and EEG, useful for harness development |

## 4. Which cross-modal relationships can the data teach?

Rows are modality pairs recorded **together on one clock**. Counts are participants, rounded.

| Pair | Sleep / clinical | Wake / wearable | Learnable at scale? |
|---|---|---|---|
| ECG – PPG | MESA 2,000 (one recorder; oximeter latency unknown); VitalDB ≈ 6,200 cases (one converter); MIMIC-III 10,000 patients (**documented changing delay of up to 0.5 s between signals**: usable for slow state, not for beat-level timing) | CASE 30 (one acquisition unit); PPG-DaLiA and WESAD 15 each (separate devices) | **Yes for slow state; for beat-level timing only in MESA, VitalDB and CASE** |
| ECG – respiration | SHHS ≈ 5,800; MESA 2,000; CinC 2018 2,000; MIMIC-III | WESAD, PPG-DaLiA, CASE: tens | **Yes** |
| PPG – respiration | MESA 2,000; MIMIC-III | CASE 30 | **Yes** |
| EEG (2–6 channels) – ECG / respiration | SHHS, MESA, CinC 2018: ≈ 10,000 together | — | **Yes, in sleep only** |
| EEG – PPG | MESA 2,000; VitalDB under anaesthesia | DEAP 32 | Sleep only |
| EDA – PPG | DREAMT 100 (overnight) | wristband pool: ≈ 190 users openly licensed, ≈ 630 with agreements; WESAD, PPG-DaLiA, CASE | **Partly**: a few hundred people, daily life; the largest open set has 16 users |
| EDA – ECG / respiration | — | WESAD 15, CASE 30, AMIGOS 40 | No |
| EDA – EEG | DREAMT 100 (different devices) | DEAP 32, AMIGOS 40 | **No** |
| Multichannel wake EEG – anything autonomic | — | DEAP, AMIGOS, DREAMER: ≈ 100 in total | **No** |

Consequences:

* **No transitivity.** Learning ECG–PPG in one dataset and PPG–EDA in another yields no knowledge of ECG–EDA timing. A shared embedding can align slow state across such datasets through the common modality, but lagged coupling between a pair is learned only where that pair is recorded together. Synchronisation between separately recorded datasets cannot be recovered and is not assumed.
* **The cardiorespiratory triad (ECG, PPG, respiration) is where the central hypothesis can be tested properly.** It has the data, the strongest known physiology (pulse arrival time, respiratory modulation of heart period and pulse amplitude) and wake datasets for transfer.
* **EEG enters as a sleep-montage stream.** Brain–body claims are limited to sleep until wake data with synchronised EEG exist at scale. For multichannel wake EEG, a public EEG model supplies the stream (report 02).
* **EDA is a late, modular addition** whose pretraining depends on obtaining wristband data.

## 5a. Synchronisation: what each dataset actually provides

Beat-level synchronisation is never inferred from nominal timestamps. Each dataset is given a class from its documentation; the Stage 0 audit then checks it empirically (stability, within a recording, of the offset between beat series from two cardiac signals) before any lag-sensitive use.

| Class | Meaning | Supports |
|---|---|---|
| **A** | Signals digitised by one converter or one device | Beat-level timing, relative and (subject to sensor latency) absolute |
| **B** | One recorder, but a signal passes through a separate instrument with unknown, presumably constant latency | Beat-level *changes* within a recording; breath-level and slower. Absolute delays are not interpretable |
| **C** | Separate devices aligned by an event marker or by the dataset authors; residual error not quantified | Breath-level and slower, after an estimated-offset check; not beat-level |
| **D** | Documented as unreliable, or merged by timestamps only | Slow state only (≥ 5 s) |

| Dataset | What is documented | Raw device timing versus provided synchronised form | Class |
|---|---|---|---|
| **MESA Sleep** | One portable sleep recorder; ECG and oximeter pleth in the same file at 256 Hz (P). Oximeter delay or auto-gain: nothing found | Single file per participant; no separate device files | **B** for ECG–PPG; A among directly recorded channels (to be audited) |
| **SHHS, CinC 2018, HMC** | Single sleep recorder per study (R; CinC 2018 resampled to 200 Hz) | Single file | B (to be audited) |
| **VitalDB** | ECG, PPG, arterial pressure and capnography from one patient monitor through one converter at 500 Hz, in most cases; EEG from a separate depth-of-anaesthesia monitor at 128 Hz; all written by one recorder "in a time-synchronized manner"; no latency figure (V) | Tracks share the case start as time zero | **A** among converter tracks; **C** between the EEG monitor and the others |
| **MIMIC-III waveforms; BIDMC** | "a (changing) delay of up to 500ms between any of the other waveforms"; pulse transit time across waveforms "may be unreliable"; ECG leads are aligned with each other (V) | Extraction "not designed for inter-waveform analysis" | **D** for ECG–PPG and ECG–respiration |
| **CASE** | All physiological sensors through one acquisition unit at 1000 Hz; joystick on a separate USB path, both stamped by a common clock on the logging computer; latency not quantified; raw and interpolated versions both provided (V) | Raw streams plus 1 ms and 50 ms interpolated grids | **A** for physiology; **C** between physiology and annotations |
| **PPG-DaLiA** | Chest and wrist devices synchronised by a double tap on the chest at the start and at the end; the end tap corrects drift; residual error not stated (V). Synchronised signals and raw chest data provided | Provided arrays are already aligned; per-device rates retained | **A** within each device; **C** across devices |
| **WESAD** | Same devices and authors; readme refers to "synchronised data"; method and error not found in a primary source (P) | As PPG-DaLiA (P) | A within device; C across devices (assumed, to be audited) |
| **DREAMT** | Wristband and sleep-lab signals "merged by aligning timestamps" with "minor adjustments"; no accuracy figure; an earlier release had a misalignment that was later fixed (P) | Merged tables at 64 Hz and at 100 Hz | A within the wristband; **D** across wristband and sleep lab until audited |
| **AMIGOS** | Wireless EEG headset and wireless ECG/EDA sensors, synchronised by the acquisition computer; described as precise, method not stated (P) | — | C |
| **DEAP** | One acquisition system for EEG and peripheral signals (R) | Single file | A (to be audited) |

Two further caveats:
* **The wristband BVP channel is not raw PPG.** It is the output of a proprietary algorithm combining two light wavelengths at 64 Hz; filtering and latency are undocumented (P). Pulse timing and shape from it are device-specific.
* **Cross-device offsets cannot be separated from physiology.** Aligning a chest ECG with a wrist pulse signal by matching their beat-interval series yields clock offset *plus* pulse arrival time. Such an estimate is good enough for breath-level alignment and useless for studying the delay itself.

Consequences for the plan:
* The decisive cross-modal experiment and every beat-level analysis use MESA (class B: relative timing) and, for evaluation with all three cardiorespiratory signals, CASE (class A).
* MIMIC-III waveforms are not used for lag-sensitive pretraining or for the pairing-perturbation controls. They remain usable for unimodal pretraining and for window-level cross-modal learning.
* The pairwise lag module and pulse-arrival features are computed only within classes A and B.

## 5b. Modality availability by dataset

"✓" means recorded; "1" marks the signals that share one clock group with ECG or PPG at class A or B. This table, not a dataset's reputation, decides which experiment a dataset can serve. Evaluation cells built on it are in report 06, section 3.2.

| Dataset | ECG | PPG / BVP | Respiration | EDA | EEG | Can serve |
|---|---|---|---|---|---|---|
| MESA Sleep | ✓ 1 | ✓ 1 (finger) | ✓ 1 | — | ✓ 3 channels | full cardiorespiratory triad with EEG: pretraining; cell E1 |
| SHHS | ✓ 1 | — | ✓ 1 | — | ✓ 2 | ECG–respiration–EEG pretraining; not the triad |
| CinC 2018 | ✓ 1 | — | ✓ 1 | — | ✓ 6 | cell E2 (ECG + respiration); event detection; EEG cells |
| HMC | ✓ 1 | — | — | — | ✓ 4 | sleep staging from ECG alone or EEG alone (cell E6). **No part in the triad experiment** |
| VitalDB | ✓ 1 | ✓ 1 | airway signals only | — | ✓ 2 (other device) | ECG–PPG under anaesthesia |
| MIMIC-III, BIDMC | ✓ | ✓ | ✓ impedance | — | — | unimodal and window-level use only (class D) |
| CASE | ✓ 1 | ✓ 1 | ✓ 1 | ✓ 1 | — | cell E5: the only verified wake dataset with the triad on one clock |
| WESAD | ✓ chest | ✓ wrist | ✓ chest | ✓ both | — | cells E3 (chest) and E4 (wrist); devices not mixed at beat level |
| PPG-DaLiA | target only | ✓ wrist | chest (variant only) | ✓ wrist | — | cell E7 |
| DREAMT | sleep lab | ✓ wrist | sleep lab | ✓ wrist | sleep lab | cell E8 (wrist signals) |
| DEAP | — | ✓ | ✓ | ✓ | ✓ 32 | far transfer with EEG |
| AMIGOS | ✓ | — | — | ✓ | ✓ 14 | far transfer with EEG |
| PTB-XL | ✓ | — | — | — | — | T4 |


## 5. Domain shift between pretraining and evaluation

| Shift | From → to | Expected effect |
|---|---|---|
| Behavioural state | sleep, supine → awake, seated or moving | Different heart-rate and breathing regimes; EEG rhythms differ entirely. The largest unknown in the plan |
| PPG sensor | finger, transmissive, clinical → wrist, reflective, consumer | Different waveform shape, more motion artefact, lower quality |
| ECG | sleep-lab lead → chest strap or wearable patch | Morphology changes; timing largely preserved |
| Respiration | effort belts → chest-strap sensor or impedance | Amplitude scale arbitrary; timing preserved |
| EEG | 2–3 mastoid-referenced derivations → 14–32 channels, other references | Spatial information unavailable in pretraining |
| Population | older community cohorts → young adult volunteers | Slower, less variable cardiac dynamics in pretraining |
| Clock | one acquisition system → separate devices aligned by hand | Beat-level cross-modal timing unreliable across devices in WESAD, PPG-DaLiA, DREAMT and AMIGOS (section 5a) |
| Clinical context | anaesthesia, intensive care → healthy | Drug and ventilator effects on every signal |

Design responses: canonical band-limiting and rates, per-window normalisation, metadata dropout, augmentation; wake cardiorespiratory data added to pretraining at research scale; clock-jitter robustness arm; leave-one-pretraining-dataset-out analysis.

## 6. Overlap and leakage risks

| Risk | Handling |
|---|---|
| BIDMC is drawn from MIMIC-II | Never used for evaluation when any MIMIC waveform data are in pretraining |
| DREAMT and CinC 2018 as both pretraining and evaluation | Assigned to one role per study; participant-level split fixed before pretraining if both |
| SHHS second visit repeats first-visit participants | Split by participant across visits |
| Public checkpoints pretrained on our evaluation data: PPG models on MESA, VitalDB and MIMIC (R); a supervised sleep model on SHHS and MESA (P); EEG models on the Temple corpus | Each checkpoint's pretraining list is checked against the task list; conflicting comparisons are excluded or flagged |
| Participants shared between cohorts (for example across NSRR studies) | Cannot be audited by identifier; stated as a limitation |

## 7. Recommended datasets

**Prototype tier (Stages 0–4).**
* Pretraining: **≈ 500 MESA participants (≈ 4,000 h)**, chosen at random with the remainder held out. ECG, PPG, respiration, EEG.
* If NSRR access is delayed, an open-data substitute needing no agreement: ECG + PPG from VitalDB (one converter, so beat-level timing is usable; anaesthesia), and ECG + respiration + EEG from the labelled half of CinC 2018. MIMIC-III waveforms are open but have documented, changing inter-signal delays and are therefore unsuitable for the lag-sensitive experiment (section 5a). The substitute loses the single-corpus design, and no open corpus offers ECG, PPG and respiratory effort together with trustworthy timing.
* Evaluation, by cell (report 06, section 3.2): E1 held-out MESA participants; E2 CinC 2018 from ECG + respiration; E3 and E4 WESAD chest and wrist; E5 CASE; E6 HMC (ECG alone, EEG alone) for harness development; E7 PPG-DaLiA from wrist signals; T4 on PTB-XL; T7 on MESA demographics.

**Research tier (Stage 5).**
* Pretraining: all of MESA and SHHS (≈ 90,000 h), optionally CinC 2018; wake cardiorespiratory and wristband data as obtainable; a slice of VitalDB or MIMIC-III for ECG–PPG diversity.
* Evaluation adds: CASE (T3), DREAMT (wearable T1), DEAP and AMIGOS (far transfer), arousal and apnoea events (T6).

**Not recommended for pretraining:** DEAP, DREAMER, AMIGOS, WESAD, CASE (too small; needed for evaluation).

## 8. Pretraining curriculum

| Phase | Data | What is trained | Purpose |
|---|---|---|---|
| P0 | Each modality from every recording that contains it | Private encoders, unimodally | Maximum data per encoder; the unimodal reference for comparison K2 |
| P1 | Recordings with ECG + PPG + respiration on one clock (MESA) | Fusion stage and private encoders, jointly | Core cross-modal experiment |
| P2 | Add the EEG stream on MESA, SHHS, CinC 2018 | EEG encoder and fusion | Brain–body coupling in sleep |
| P3 | Wristband recordings (EDA + BVP) | EDA encoder and its fusion connections, with the PPG stream as the anchor | Bring in EDA through the hub modality |
| P4 | All sources mixed, batches balanced by modality pair | Everything, at low learning rate | Research-scale model with heterogeneous modality sets |

Each phase is a separate, testable step; a later phase is started only if the earlier one passed its gate (report 07). Missing modalities need no special handling: absent streams contribute no tokens and no loss terms (report 05).

## 9. Storage and compute (estimates)

After band-limiting and resampling to canonical rates, stored as 16-bit values: about 4 MB per recording-hour with three EEG channels, ECG, PPG and two respiration channels.

| Corpus | Preprocessed | Raw download |
|---|---|---|
| Prototype, 4,000 h | ≈ 17 GB | the corresponding raw files, tens of GB |
| MESA, ≈ 18,000 h | ≈ 75 GB | not verified; expect several hundred GB |
| MESA + SHHS, ≈ 90,000 h | ≈ 330 GB | not verified; expect of the order of 1 TB |
| CinC 2018 | ≈ 60 GB | 266.6 GB (V) |
| VitalDB | depends on tracks kept | 94 GB (V) |

Preprocessing is a one-off CPU job. Training-time data loading should read from the preprocessed arrays, never from the original files.

## 10. Actions that need you

1. Submit NSRR data access and use agreements for MESA and SHHS (free; reviewed by a committee; ethics-board documentation was described by NSRR staff as preferred but not required for MESA; read the current terms on commercial use and on models trained from the data).
2. Decide whether the Temple EEG corpus is needed at all; the recommendation is to reuse a public EEG model instead.
3. Check the release status of the pooled wristband corpus, or select its openly licensed constituents.
4. Request DREAMT if wearable sleep staging is wanted as a task.

## 11. Not verified

MESA and SHHS download sizes; hours for VitalDB, MIMIC-III and SHHS (estimates only); ECG in ISRUC; VitalDB track rates (from documentation, not the files); current NSRR terms (from staff posts, 2018–2022); pretraining sets of the public PPG checkpoints; whether each wristband dataset ships BVP alongside EDA and where the pooled release lives; DREAMER channel details; download sizes of WESAD and DREAMT; the WESAD synchronisation method and file layout (from the companion dataset and a third-party description only); any oximeter latency or auto-gain in MESA; whether PTB-XL's statement table includes atrial fibrillation; whether CinC 2017 has undisclosed patient grouping.
