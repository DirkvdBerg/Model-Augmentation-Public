# Excitation Design Literature Log

**Purpose:** exact papers and links consulted online for the redesign of
`Matlab-scripts/Augmentation/data/generate_oscillatory_multisine_data.m`
(data generation for ANN augmentation training).

**Logged:** 2026-07-02.

**Status legend:**
- VERIFIED: bibliographic details confirmed online this session (title, authors, venue checked against the source).
- PLAUSIBLE: surfaced in search with consistent metadata, but details not independently confirmed.

---

## 1. Standard excitation signal families (overview sources)

| Reference | Link | Status | Relevance |
|-----------|------|--------|-----------|
| Pintelon, Schoukens, *System Identification: A Frequency Domain Approach*, 2nd ed., Wiley-IEEE, 2012 | https://www.researchgate.net/publication/4378090_System_Identification_A_Frequency_Domain_Approach | VERIFIED (book exists; chapter content from own knowledge) | Canonical comparison of excitation families: stepped sine, swept sine, periodic chirp, multisine, PRBS, random noise. Period length vs frequency grid. |
| Schoukens, Ljung, "Nonlinear System Identification: A User-Oriented Road Map," IEEE Control Systems Magazine 39(6), 2019 | https://arxiv.org/abs/1902.00683 | VERIFIED | Excitation must cover the amplitude AND frequency range of interest for nonlinear models. Primary justification for amplitude ladders and broadband coverage. |
| Schoukens, Vaes, Pintelon, "Linear System Identification in a Nonlinear Setting," IEEE Control Systems Magazine, 2018 | https://arxiv.org/pdf/1804.09587 | VERIFIED (arXiv) | Random-phase multisines and the BLA framework: why independent phase realizations are needed to quantify nonlinear distortions. |
| Vuojolainen, Nevaranta, Jastrzebski, Pyrhonen, "Comparison of Excitation Signals in Active Magnetic Bearing System Identification," Modeling, Identification and Control 38(3):123-133, 2017, DOI 10.4173/mic.2017.3.2 | https://www.mic-journal.no/ABS/MIC-2017-3-2.asp/ | VERIFIED (fetched abstract page) | Practical head-to-head of PRBS, chirp, multisine, stepped sine on a mechatronic (magnetic bearing) system. All workable; multisine preferred for band-selective excitation with low crest factor. |
| "Broadband versus stepped sine FRF measurements" (Schoukens school, IEEE Trans. Instrum. Meas.) | https://www.researchgate.net/publication/3089696_Broadband_versus_stepped_sine_FRF_measurements | PLAUSIBLE (authors/year not confirmed) | SNR vs measurement time trade-off between signal families. |

## 2. Chirp / swept sine

| Reference | Link | Status | Relevance |
|-----------|------|--------|-----------|
| Gloth, Sinapius, "Analysis of swept-sine runs during modal identification," Mechanical Systems and Signal Processing 18 (2004) 1421-1441 | https://www.sciencedirect.com/science/article/abs/pii/S0888327003000876 | VERIFIED (title, authors, volume, year, pages via search) | Sweep-rate limits for lightly damped modes: too-fast sweeps shift and lower the resonance peak and cause ringing/beating after the peak. Directly relevant to any chirp passing near fa = 150 Hz with Q = 10. |
| Spectral Dynamics, "Understanding Sine Test Methodologies" (technical note) | https://www.spectraldynamics.com/support/technical-library/understanding-sine-test-methodologies-swept-sine-stepped-sine-and-resonance-search-and-dwell | PLAUSIBLE (vendor note, not peer reviewed) | Practical guideline: sweep slow enough that the structure completes on the order of ten or more cycles per frequency; more for high-Q modes. |
| Roy, "Sine sweep effect on specimen modal parameters characterization," ECSSMET 2016 | https://www.topmodal.fr/wp-content/uploads/2017/03/roy_ecssmet_2016_sine_sweep_effect.pdf | PLAUSIBLE | Quantifies sweep-rate distortion of identified modal parameters. |
| Schroeder (1970) phase formula, discussed in crest factor literature (Section 3) | see Section 3 links | VERIFIED (as cited within those papers) | A periodic chirp is a multisine with deterministic (Schroeder-like) phases: chirp vs multisine is a phase-spectrum choice within one family, not two different tools. |

**Working conclusions for chirp (from the above):**
1. Chirp is acceptable as a position *reference* sweep through the structural modes (0.1 to 5 Hz), smooth and machine-friendly.
2. Any chirp interacting with the 150 Hz, Q = 10 MSD must respect sweep-rate limits or the resonance response is distorted (peak shift, ringing that looks like bursts/spikes in time plots).
3. A chirp gives no realization averaging: nonlinear distortions are deterministic and hidden inside the response. Random-phase multisines with independent realizations remain the tool for distortion quantification (BLA framework, Section 1).
4. As NN training data, a chirp visits each frequency briefly at one amplitude: it does not solve amplitude coverage. APRBS/space-filling signals do (Section 4).

## 3. Multisine design and crest factor

| Reference | Link | Status | Relevance |
|-----------|------|--------|-----------|
| Schoukens, Guillaume, "Design of multisine excitations" | https://www.semanticscholar.org/paper/Design-of-multisine-excitations-Schoukens-Guillaume/3a0f78b84bf40d23ce9d2fb4eb0f0b826eaae549 | PLAUSIBLE | Amplitude/phase selection for multisines, spectrum shaping. |
| Schoukens, Dobrowiecki, "Design of broadband excitation signals with a user imposed power spectrum and amplitude distribution" | https://www.semanticscholar.org/paper/Design-of-broadband-excitation-signals-with-a-user-Schoukens-Dobrowiecki/bf700da45dd99c9de725da6a8a3cf57603b83594 | PLAUSIBLE | Supports dual-band / shaped spectra: power can be placed exactly where information is needed. |
| "Improved crest factor minimization of multisine excitation signals using nonlinear optimization," Automatica, 2022 | https://www.sciencedirect.com/science/article/abs/pii/S0005109822005180 | PLAUSIBLE | State of the art beyond the best-of-200-random-candidates approach in our script. |
| "Multiphase multisine signals - theory and practice," ISMA 2016 | https://past.isma-isaac.be/downloads/isma2016/papers/isma2016_0570.pdf | PLAUSIBLE | Phase-shifted (orthogonal) multisines across channels for MIMO modal testing: the literature route to per-mode/per-channel excitation design. |
| "Recent Advances in Crest Factor Minimization of Multisine" | https://www.researchgate.net/publication/316462036_Recent_Advances_in_Crest_Factor_Minimization_of_Multisine | PLAUSIBLE | Survey of crest factor algorithms (Schroeder, clipping, optimization). |

## 4. Nonlinear / NN training data: APRBS and space-filling signals

| Reference | Link | Status | Relevance |
|-----------|------|--------|-----------|
| Nelles, *Nonlinear Dynamic System Identification* (Springer book chapter) | https://link.springer.com/chapter/10.1007/978-3-030-47439-3_19 | VERIFIED (chapter exists) | APRBS as the standard excitation for nonlinear model training; dwell time on the order of the dominant time constant. |
| Heinz, Nelles, "Excitation signal design for nonlinear dynamic systems with multiple inputs - a data distribution approach," at-Automatisierungstechnik, 2018 | https://www.degruyterbrill.com/document/doi/10.1515/auto-2018-0027/html | VERIFIED (title/venue via search) | Multi-input excitation design by target data distribution (space filling), OMNIPUS line of work. |
| "Online and Offline Space-Filling Input Design for Nonlinear System Identification: A Receding Horizon Control-Based Approach," arXiv 2504.02653 | https://arxiv.org/html/2504.02653 | PLAUSIBLE | Recent space-filling input design method. |
| "Analysis of space-filling excitation signals ... of a Diesel engine," Control Engineering Practice, 2025 | https://www.sciencedirect.com/science/article/pii/S0967066125002163 | PLAUSIBLE | Applied comparison: APRBS vs OMNIPUS vs sGOATS on a real nonlinear process. |
| Bombois et al., "Least Costly Space-Filling Experiment Design for the Identification of a Nonlinear System," arXiv 2605.02517 | https://arxiv.org/html/2605.02517 | PLAUSIBLE | Formalizes operating-region coverage as an experiment design criterion. |
| "Deep active learning for nonlinear system identification," arXiv 2302.12667 | https://arxiv.org/pdf/2302.12667 | PLAUSIBLE | Ensemble-based informativeness for trajectory selection. |

## 5. Plant-friendly identification (the "spikes never seen before" objection)

| Reference | Link | Status | Relevance |
|-----------|------|--------|-----------|
| Rivera et al., "'Plant-Friendly' system identification: a challenge for the process industries" (IFAC) | https://www.sciencedirect.com/science/article/pii/S1474667017348735 | PLAUSIBLE | Defines plant-friendliness: constraints on span, move size, and variability of inputs AND outputs during identification experiments. |
| Rivera et al., "Constrained multisine input signals for plant-friendly identification of chemical process systems" | https://asu.elsevierpure.com/en/publications/constrained-multisine-input-signals-for-plant-friendly-identifica/ | PLAUSIBLE | Multisine design under time-domain operating constraints. |
| Lee, Rivera, "Constrained minimum crest factor multisine signals for plant-friendly identification of highly interactive systems" | https://www.sciencedirect.com/science/article/pii/S1474667017348772 | PLAUSIBLE | MIMO extension; ill-conditioned interactive systems. |
| "Optimization-based design of plant-friendly multisine signals using geometric discrepancy criteria," Computational Optimization and Applications, 2007 | https://link.springer.com/article/10.1007/s10589-007-9033-0 | PLAUSIBLE | Combines spectral requirements with time-domain friendliness. |

## 6. MIMO informativity and LPV experiment design (verified in the claims-validation session, 2026-07-02)

| Reference | Link | Status | Relevance |
|-----------|------|--------|-----------|
| Ghosh, Bombois, Huillery, Scorletti, Mercere, "Optimal identification experiment design for LPV systems using the local approach," Automatica 87 (2018) 258-266 | https://hal.science/hal-01720097 | VERIFIED | Frozen-scheduling (fixed-Y) local experiments: optimal choice of operating points and input spectra. |
| Colin, Bombois, Bako, Morelli, "Data informativity for the open-loop identification of MIMO systems in the prediction error framework," Automatica 117 (2020) 109000 | https://www.sciencedirect.com/science/article/abs/pii/S0005109820301989 | VERIFIED | Structure-specific MIMO informativity conditions; classical positive-definite-spectrum criterion is sufficient but too restrictive. |
| Colin et al., "Informativity: how to get just sufficiently rich for the identification of MISO FIR systems with multisine excitation?" ECC 2019 | https://ieeexplore.ieee.org/document/8795997/ | VERIFIED | Multisine informativity per channel; relaxed conditions vs classical criterion. |
| Beintema, Toth, Schoukens, "Nonlinear state-space identification using deep encoder networks," L4DC 2021 (PMLR 144) | https://arxiv.org/abs/2012.07697 | VERIFIED | Method introduction (encoder + truncated prediction loss). |
| Beintema, Schoukens, Toth, "Deep subspace encoders for nonlinear system identification," Automatica 156 (2023) 111210 | https://arxiv.org/abs/2210.14816 | VERIFIED (SUBNET acronym coined here, confirmed in full text) | Journal version; defines SUBNET; consistency analysis of the truncated loss. |
| Frequency-domain MIMO motion identification (wafer stage), arXiv 2503.02869 | https://arxiv.org/html/2503.02869v1 | VERIFIED (fetched; uses per-axis multisines, no coordinate-transform excitation design) | Closest motion-control application; confirms no established modal-coordinate excitation argument found (our logical-coordinate design remains a HEURISTIC). |

---

## 7. Data generation in similar augmentation cases (verified 2026-07-03)

What excitation the closest-relative papers actually used to train physics-baseline + ANN augmentation models.

| Reference | Link | Status | Data generation used |
|-----------|------|--------|----------------------|
| Hoekstra, Verhoek, Toth, Schoukens, "Learning-based model augmentation with LFRs," European Journal of Control 86 (2025) 101304 | https://arxiv.org/abs/2404.01901 (arXiv); local copy `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf` | VERIFIED (read from local PDF, pp. 4-6) | 3-DOF MSD study: RK4 simulation, Ts = 0.02 s, ZOH input from a DT multisine with 1666 frequency components on the full grid [0, 25] Hz, uniform random phases; additive white output noise at SNR 20/30/60 dB; separate (independent) data sets for estimation, validation, test. Bouc-Wen: standard benchmark data (below). |
| Hoekstra, Gyorok, Toth, Schoukens, "Encoder initialisation methods in the model augmentation setting," arXiv 2602.13108 (2026) | https://arxiv.org/html/2602.13108 ; local copy `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf` | VERIFIED (fetched full text) | 2-DOF nonlinear MSD: multisine, 1666 components on [0, 5] Hz full grid, uniform random phases; RK4 at 0.01 s, sampled at 0.1 s; white noise to 20 dB SNR; splits 20k/10k/10k samples. No explicit experiment-design recommendations given. |
| Hoekstra, Gyorok, Toth, Schoukens, "Learning-based augmentation of first-principle models: a linear fractional representation-based approach," arXiv 2602.17297 (2026) | https://arxiv.org/abs/2602.17297 | VERIFIED (abstract only) | Case studies: hardening MSD simulation + F1Tenth electric car (real data). Data generation details not extracted yet (PDF not read). |
| Schoukens, Noel, Bouc-Wen benchmark (nonlinearbenchmark.org) | https://www.nonlinearbenchmark.org/benchmarks/bouc-wen | VERIFIED (via search of benchmark description) | Training: random-phase multisine, 8192 samples, full grid 5-150 Hz, 50 N RMS. Tests: (i) sinesweep 40 N, 20-50 Hz at 10 Hz/min; (ii) independent multisine, same band. The de-facto standard in this community: broadband full-grid multisine training + cross-signal-class (sweep) generalization test. |
| Bolderman, Lazar, Butler, "Physics-guided neural networks for inversion-based feedforward control applied to linear motors," IEEE CCTA 2021, pp. 1115-1120 | https://research.tue.nl/en/publications/physics-guided-neural-networks-for-inversion-based-feedforward-co/ ; https://arxiv.org/abs/2103.06092 | VERIFIED (venue/authors; training-data section NOT extracted, PDF unreadable via fetch) | Closest hardware relative (industrial linear motor, TU/e). Training-data details still to be read from the PDF. |

**Working conclusions from Section 7:**
1. In every published case of this exact augmentation framework, training data is an open-loop, full-grid random-phase multisine covering the ENTIRE dynamic range of the system (not a narrowband around the unknown dynamics), with independent realizations per split. This is community-standard practice and supports broadband/dual-band over narrowband-only for our case.
2. Generalization in these papers is tested across signal classes (multisine-trained model evaluated on a sinesweep), not just across phase realizations. We currently have no cross-class test experiment: worth adding (e.g. train on multisine-type excitation, test on chirp or p2p-type motion).
3. No published augmentation case handles our complications: closed-loop excitation through a tracking controller, position references as the primary motion, an LPV scheduling variable to cover, or hardware amplitude limits. The data-generation design for the gantry therefore has no ready-made recipe in the augmentation literature; it must be assembled from the excitation-design principles in Sections 1-6 and justified per choice.
4. All published cases are single-amplitude experiments on simulated or benchmark data; amplitude-ladder design (multiple RMS levels) is not standard in the augmentation papers, but follows from the BLA amplitude-dependence argument (Section 1).

**Open verification tasks:**
- Confirm authors/year of "Broadband versus stepped sine FRF measurements."
- Obtain and read Gloth & Sinapius (2004) for the quantitative sweep-rate criterion before designing any chirp near 150 Hz.
- Confirm exact theorem numbering in Ljung (1999) Ch. 13 for the SISO PE(2n) statement (physical copy).
- Read the training-data section of Bolderman et al. (CCTA 2021) from the arXiv PDF (fetch failed; download and read locally).
- Read the F1Tenth data-generation details in arXiv 2602.17297 (real-data augmentation case).

---

## 8. Closed-loop resonance identification / suppression on precision motion stages (verified 2026-07-06)

Targeted search in response to Section 7 conclusion 3 and brief Section 5.4 (the in-field
reference base for closed-loop, high-frequency resonance excitation on mechatronic motion
stages is thin). Searched for closed-loop identification and resonance treatment specifically
on motion stages / servo hardware, not augmentation papers.

| Reference | Link | Status | Relevance |
|-----------|------|--------|-----------|
| Boukhebouz, Mercere, Grossard, Laroche, "Shaping multisine excitation for closed-loop identification of a flexible transmission," IFAC-PapersOnLine 54(7), 643-648, 2021 | https://www.sciencedirect.com/science/article/pii/S2405896321012076 | VERIFIED (authors, venue, pages via search) | Directly on the closed-loop-shaping open question (brief Sec 6, last bullet). Heuristic: minimize crest factor and shape the multisine spectrum via the REFERENCE spectrum (not the raw plant input) so the realized control signal converges to the desired spectrum despite closed-loop reshaping and input saturation. Case study is a flexible motor-to-joint transmission: a resonance downstream of the actuator, structurally analogous to our payload MSD downstream of the stage. |
| Dee, Natu, HosseinNia, "Active damping control of higher-order resonance mode in positioning systems: application to prototype compliant dual positioning stage," Mechatronics, 2025 | https://www.sciencedirect.com/science/article/pii/S0957415825000248 ; TU Delft repository https://repository.tudelft.nl/record/uuid:e6fe69a6-27ea-4eb5-92bf-05ae8c8b5a87 | VERIFIED (authors via SSRN preprint + TU Delft repository) | Structurally close in-field system: a primary positioning stage carrying a compliant secondary structure with a non-collocated higher-order resonance, i.e. the same "hidden resonance riding on a controlled stage" topology as our payload MSD. Focus is active damping control (HP-PPF), not excitation/training-data design, so it does not answer the data-generation question directly, but confirms the system class is a real, currently-studied precision-mechatronics problem (TU Delft precision mechatronics group). |
| Chen, Gao, "Robust Synergistic Control Architecture for High-Frequency Resonance Suppression in Precision Linear Motion Stages," Electronics 15(1):195, 2026 | https://doi.org/10.3390/electronics15010195 | VERIFIED (authors, journal, volume/issue via search) | Confirms lightly-damped high-frequency resonance on precision linear stages is an active current problem (unified input-shaping + feedforward + notch-filter architecture). Control-focused, not identification/excitation-design; use only as motivation/background, not as a data-generation precedent. |
| van Haren, Mae, Blanken, Oomen, "Lifted frequency-domain identification of closed-loop multirate systems: applied to dual-stage actuator hard disk drives," Mechatronics 108, 2025 | https://arxiv.org/abs/2502.21065 | VERIFIED (authors, venue via arXiv + search) | Different hardware (rotational HDD dual-stage actuator) but the same topology class as our problem: closed-loop identification of a secondary fast/fine dynamic nested inside a coarser stage's control loop, using fast-rate excitation with slow-rate output. Relevant as a precedent for closed-loop identification of a nested secondary dynamic; not directly informative for multisine amplitude/coordinate design. |
| "Closed-Loop Black-Box Identification of Active Magnetic Bearing System Under Decentralized Control," Actuators (MDPI) 15(7):372, 2025 | https://www.mdpi.com/2076-0825/15/7/372 | PLAUSIBLE (journal/volume/issue confirmed via search; author names not independently confirmed this session) | Different domain (rotating AMB, not a translational stage), but reports a concrete quantitative rule not found elsewhere in this log: PRBS excitation at ~10-12% of actuator saturation current as the reported sweet spot for coherence / SNR / FRF-variance trade-off in closed loop. This is a noise/SNR-motivated rule (coherence, variance) -- per brief Section 5.2, do NOT import the number into our noiseless-simulation amplitude reasoning. Logged only as a data point on how much closed-loop authority other groups give an injected signal relative to actuator limits. |

**Working conclusions from Section 8:**
1. The closed-loop-shaping open question (brief Section 6) now has one directly relevant hit: Boukhebouz et al. shape the REFERENCE spectrum rather than the raw force to counteract closed-loop/controller reshaping. This is a candidate alternative to shaping `f_sim` directly in the current generator, worth raising in the design discussion, not yet adopted.
2. The in-field base for "a hidden resonance riding on a controlled precision stage" is less thin than Section 7 assessed: Dee et al. (TU Delft, 2025) is a close structural analog. It still does not resolve amplitude, coordinate, or band questions since it is a control paper, not an excitation-design paper.
3. No new reference gives a broadband-vs-narrowband amplitude/identifiability rule for noiseless-simulation ANN training on a closed-loop motion stage; Section 7 conclusion 3 (no ready-made recipe) still stands after this search.

**New open verification tasks:**
- Independently confirm author list of the MDPI AMB paper (fetch was too large to read this session).
- Read Boukhebouz et al. (2021) in full for the reference-spectrum-shaping algorithm, to assess whether it is applicable to our Cfb/tracking-controller setup.
- Read Dee et al. (2025) methods section to check whether their resonance is collocated or non-collocated relative to the sensor, and whether that maps onto our Y-scheduled MSD (delta_a is not directly measured, only inferred through its reaction force on Y).

---

## 9. Targeted research: closed-loop injection point, modal-coordinate constrained excitation, broadband-vs-narrowband (verified/attempted 2026-07-06)

Follow-up search after Section 8, aimed at the three sharpest open items: (1) whether any
paper designs MIMO excitation in a modal/logical coordinate basis to respect a hard
constraint on one physical mode, (2) reference- vs. disturbance-injection for closed-loop
ANN/augmentation training data, (3) a non-noise argument for narrowband-concentrated vs.
full-grid spectral energy near a known target resonance. Also attempted full-text reads of
two previously-unread papers (Bolderman CCTA 2021, Hoekstra arXiv:2602.17297 F1Tenth
section).

### 9.1 Bolderman, Lazar, Butler, CCTA 2021 (arXiv:2103.06092) -- training data section, read in full

**VERIFIED, full PDF read directly (confirmed independently, not only via subagent report).**

Section III-A ("Training Data Generation") specifies, for their industrial coreless linear
motor (CLM), closed-loop, PID feedback `Cfb` always active (CLMs "cannot be operated in
open-loop" due to drift):
1. Zero-mean white noise dither added directly to the plant input `u` (i.e. in parallel
   with the feedback controller output, at the actuator, not through the reference),
   variance (80 N)^2, held constant at 100 Hz update rate.
2. A separate third-order (jerk-limited) point-to-point reference trajectory (0 to 0.05 m,
   v_max = 0.05 m/s, a_max = 4 m/s^2, jerk_max = 1000 m/s^3, constant-velocity segment 50%
   of the time) for operating-range coverage, per Schoukens & Ljung 2019.
3. Verbatim quote (their stated rationale for the dither placement): "Dithering the CLM
   input directly prevents the signal from being filtered by the feedback controller and
   the lower frequency causes the CLM to explore more frequencies rather than sticking
   close to the reference trajectory."
4. Data from 4 back-and-forth motions, Ts = 1e-4 s, split 70% train / 15% val / 15% test.

Caveat: their identification objective is inversion-based feedforward control for a SISO
linear motor with nonlinear friction (not grey-box LFR/state-space augmentation of a
resonant mode), so the target dynamic class differs from our 157 Hz MSD. The injection-point
rationale (point 3) is nonetheless directly on-topic for the closed-loop-shaping question and
is architecture-relevant independent of the target dynamic.

**Relevance to our generator:** our own `generate_oscillatory_multisine_data.m` already
injects `f_scaled` as an additive force in parallel with the feedback path (`u_ms = lsim(Cfb_ss,
-q_ms) + f_scaled`, superposed on the trajectory-only response), not through the position
reference `r_traj`. This is structurally the same choice Bolderman et al. made deliberately
to defeat controller filtering, and their explicit written rationale is a citable justification
for an architectural decision our generator already takes for other/no stated reasons. It does
not eliminate closed-loop reshaping (the `-Cfb*q_ms` feedback term still couples the delivered
spectrum to loop gain), but confirms the current injection point does not need to change to
match this precedent.

### 9.2 Modal/logical-coordinate constrained excitation design (Q1) -- negative result

No system-ID, LPV experiment-design, or ANN-training-data paper was found that designs a
multisine directly in a modal/transformed coordinate basis to budget a hard physical
constraint on one mode. Closest adjacent work: Johansen & Fossen, "Control allocation -- a
survey," Automatica 2013 (https://www.sciencedirect.com/science/article/abs/pii/S0005109813000368,
PLAUSIBLE relevance only -- control allocation under actuator limits, not excitation design
for identification) and dual-drive gantry decoupling-control papers that use a sum/difference
coordinate transform for control synthesis, not experiment design (e.g.
https://www.researchgate.net/publication/251965622, https://www.researchgate.net/publication/257426888,
PLAUSIBLE, one of these reports a yaw-mode resonance near 126 Hz on a similar dual-drive
gantry, structurally close but not an excitation-design source).
**Conclusion: the logical-coordinate multisine design (brief Section 3.1) remains an
uncited engineering choice.** Motivated by analogy to control-allocation practice (design in
the space where the constraint lives, then map to actuators), not by an identification-theory
source. Should be logged as `# HEURISTIC` if/when implemented.

### 9.3 Reference- vs. disturbance-injection for closed-loop ANN augmentation training (Q2) -- negative result

No paper was found comparing reference-injected vs. disturbance-injected excitation
specifically for training an ANN augmentation block that will later run inside the same
closed loop. One tangential hit: "Neural Network Training Using Closed-Loop Data: Hazards
and an Instrumental Variable (IVNN) Solution," arXiv:2202.05337
(https://arxiv.org/pdf/2202.05337), PLAUSIBLE (title/ID matched, not read in full this
session) -- warns that closed-loop data creates feedback-induced input/output correlations
that bias NN training, and proposes an instrumental-variable correction. Relevant as a
general warning, not as a resolution of the injection-point question. Bolderman (Section 9.1)
remains the closest concrete precedent, and favors disturbance-injection (post-controller-ish,
parallel to feedback) over reference-injection -- consistent with what our generator already
does.

### 9.4 Broadband full-grid vs. narrowband-concentrated spectral energy (Q6) -- negative result, and a correction to the generator's default

No FIR/state-space informativity paper, and no augmentation paper (including our own
strongest precedents, Hoekstra EJC 2025 and the Bouc-Wen benchmark, both full-grid), was
found to argue for concentrating spectral energy near a specific known target mode over
full-grid coverage, on identifiability or bias grounds independent of noise/SNR. One
unattributed search-engine synthesis surfaced the opposite classical argument (avoid exciting
unmodeled resonances to prevent bias) -- explicitly not usable here since it is not
attributable to a real source and, more importantly, does not apply to our case (our
"unmodeled" resonance is the very thing we want the ANN to learn, not a nuisance to avoid).
**Conclusion: the current generator's default `MULTISINE_BAND = 'narrowband'` has no literature
support beyond the in-repo HEURISTIC amplitude reasoning (D-056/D-057). The already-settled
framework precedent (this doc, Section 4: full-grid broadband, Hoekstra EJC 2025 / Bouc-Wen)
argues the other way.** Whether concentrated power is still needed in practice for adequate
delta_a activation under the TELICA force-RMS budget is a quantitative question the
generator's own informativeness diagnostic (delta_a RMS with/without multisine, PSD peak
near fa) can answer empirically once both bands are run and compared -- not a question the
literature answers either way.

### 9.5 Hoekstra et al., arXiv:2602.17297 (2026), F1Tenth section -- still unread

Bibliographic details reconfirmed (submitted to Automatica, under review, arXiv:2602.17297,
19 Feb 2026, CC-BY-NC-SA-4.0). PDF full text was not retrieved in this session (rate-limited
on the research pass; not reattempted). The abstract-level description of the F1Tenth case
(real, measured data) is all we have. **Next step, not yet done:** retry the PDF fetch, or
check the associated code repository https://github.com/JanHHoekstra/Model-Augmentation-Public,
which may document the F1Tenth data-generation maneuvers directly (e.g. in a README or data
script) without needing the paper text.

**New open verification tasks:**
- Fetch and read arXiv:2602.17297 in full (F1Tenth excitation/maneuver details), or check
  the GitHub repo above.
- If the logical-coordinate multisine design is adopted, log it explicitly as `# HEURISTIC`
  per CLAUDE.md's signal-processing labeling rule, citing Section 9.2's negative result as
  the reason no `# THEORY` label applies.
- Run the existing with/without-multisine informativeness diagnostic for both broadband and
  narrowband bands on at least one experiment, to settle the delta_a-activation question
  empirically per Section 9.4, before finalizing the band choice.
