# Data Generation Brief (for a fresh session)

**Purpose:** context handoff for discussing proper data generation for the ANN
augmentation. Read this after `CLAUDE.md` and `tasks/lessons.md`.

**This is a discussion brief, not a spec.** Nothing here is finalized. Section 3
in particular is a set of OPEN working assumptions to be challenged, not settled
decisions.

**Companion docs (do not duplicate, read them):**
- `docs/excitation-design-literature.md` — full research log: every paper found,
  with exact links and a VERIFIED / PLAUSIBLE status per entry. This is the
  record of what was researched (Section 7 below indexes the key ones).
- `docs/data-generation-plan.md` — the earlier, more detailed plan (predates the
  corrections in Section 5; treat its noise-based and narrowband-vs-broadband
  arguments as superseded).
- `Matlab-scripts/Augmentation/data/generate_oscillatory_multisine_data.m` — the
  current generator being redesigned.

---

## 1. Goal and system

- Grey-box: augment a trusted LPV first-principles (FP) model of a 3-axis linear
  motor gantry with an ANN that learns dynamics the FP model misses.
- Hidden dynamics to learn: a payload mass-spring-damper, resonance fa ≈ 157 Hz,
  ζ ≈ 0.05 (Q ≈ 10). This is the primary target of the augmentation.
- Scheduling variable: Y (payload position) in [-0.30, +0.30] m. Quasi-LPV model.
- States: 6 physical (positions + velocities in logical coords X_sym, X_anti, Y)
  + 2 augmented (delta_a, delta_a_dot).
- Identification method: SUBNET (subspace encoder + truncated prediction /
  multi-step loss). The encoder must reconstruct all 8 states from short I/O
  windows.
- **Current phase is NOISELESS simulated data** (MATLAB/Simulink ground truth =
  FP + hidden MSD). This is critical: noise-motivated reasoning does not apply
  (see Section 5).

## 2. Current generator and why it is being redesigned

`generate_oscillatory_multisine_data.m` produces 14 experiments (osc + p2p),
each a fixed-frequency sinusoidal or jerk-limited position reference with a force
multisine overlaid. Known problems:

- **Multisine is per-channel in STAGE coordinates** (independent F_X1, F_X2, F_Y).
  Consequence: the anti-symmetric/yaw component (F_X1 - F_X2)/2 is an
  uncontrolled random ~0.71x of the per-channel amplitude, aimed at the one mode
  with a hard 6 mm |X1-X2| limit; sym-vs-anti energy split is random per
  realization.
- **One 1 s realization per split, tiled ~11x** to fill each 10 s experiment,
  and shared across all experiments in a split. In noiseless simulation the tiled
  repeats add zero information.
- **Constant-amplitude, constant-frequency references**: each experiment traces
  one closed orbit for 10 s. Supervisor objection: "the oscillations are all the
  same." Every experiment looks alike because only (amplitude, frequency, Y) change.
- **A sudden unexplained spike** was seen in one experiment's plots (supervisor
  objection). Not root-caused. Candidate sources: feedback-only tracking of
  aggressive p2p moves (no feedforward in the sim loop), MSD resonance ringing
  after acceleration edges, or the narrowband multisine envelope during standstill
  holds. Diagnosing this is worthwhile before trusting any p2p data.
- Band is currently narrowband [130, 180] Hz (targets the MSD only); the
  structural band [1, 10] Hz and the mid-band are barely excited.

## 3. OPEN working assumptions (discuss, do not assume settled)

These are leanings from prior discussion, deliberately left open:

1. **Channels designed in logical (modal) coordinates** [F_sym, F_anti, F_Y],
   each with its own amplitude, transformed to stage forces
   (F_X1 = F_sym + F_anti, F_X2 = F_sym - F_anti) before simulation, with F_anti
   sized directly from the 6 mm yaw budget. Rationale is constraint control and
   even mode coverage, NOT a literature result (see Section 4). Open: is this the
   right call, or does the ANN seeing mixed (x_logical, u_stage) inputs make it moot?
2. **Multisine period = experiment length** (no tiling), independent realization
   per experiment. Open: how many experiments, how long.
3. **All 3 channels excited simultaneously** every experiment (MIMO informativity).
4. **Amplitude of the force multisine** set so the MSD resonance response
   (delta_a) is strongly activated for training gradients. Open: what fraction,
   and whether broadband dilutes the 157 Hz activation vs the current narrowband.

## 4. What has precedent vs what is ours

| Element | Status | Basis |
|---------|--------|-------|
| Random-phase multisine as training excitation | VERIFIED precedent | Hoekstra EJC 2025; Bouc-Wen benchmark; SUBNET papers |
| Full-grid broadband over ALL dynamics (not narrowband around the unknown mode) | VERIFIED precedent | Hoekstra MSD study uses full grid [0,25] Hz; Bouc-Wen trains on full grid 5-150 Hz covering its resonance |
| Independent realizations per data split | VERIFIED precedent | Hoekstra EJC 2025; BLA literature |
| Fixed-scheduling (frozen-Y) local LPV experiments | VERIFIED precedent | Ghosh et al. 2018, Automatica 87 |
| Sweep as a held-out TEST class (not training) | VERIFIED precedent | Bouc-Wen benchmark test set |
| Logical-coordinate channel design + yaw budgeting | OURS (no citation) | engineering choice; flag `# HEURISTIC` |
| 40%-of-trajectory-RMS amplitude fraction | OURS | heuristic |
| Position-level APRBS (randomized jerk-limited setpoints) | PARTIAL | APRBS is standard (Nelles); position-level jerk-limited adaptation is ours |
| Y-scheduling rate coverage as a design requirement | OURS / weak | principle-based; no verified LPV experiment-design paper prescribing it found yet |

## 5. Corrections — mistakes made in prior discussion, do NOT repeat

1. **A cross-class test for the augmentation must still excite near 157 Hz.**
   A previous outline proposed a 0.1-5 Hz reference chirp as the "generalization
   test" and cited Bouc-Wen's "20-50 Hz sinesweep." That band is Bouc-Wen's
   resonance, NOT ours. A test that never reaches ~157 Hz cannot test our MSD
   augmentation at all. Transfer a precedent's FUNCTION (sweep through the
   system's own resonant band), never its literal numbers.
2. **Noise-based arguments are void this phase.** SNR budgets, averaging over
   periods/realizations, BLA variance estimation, crest-factor-for-SNR: all
   assume measurement noise. The data is noiseless simulation. Here realizations
   matter only for split independence / coverage diversity, and repeated content
   adds nothing.
3. **Chirp is not standard TRAINING practice in this field.** In every verified
   augmentation/SUBNET case the training signal is a full-grid random-phase
   multisine; a sweep appears only as a held-out test. Swept sine as a primary
   signal is standard in experimental modal analysis (a different paradigm), not
   here.
4. **The in-field reference base is thin and was rightly questioned.** The
   augmentation-specific precedents (Hoekstra / SUBNET / Bouc-Wen) are solid, but
   references for high-frequency resonance excitation on closed-loop mechatronic
   motion stages are weak. Finding better-matched in-field papers is an OPEN task,
   not a solved one. Do not present the current log as authoritative.

## 6. Open questions to drive the discussion

- How to test the 157 Hz augmentation on a different signal class while still
  exciting the resonance (the correct version of the mistake in 5.1).
- Does broadband [1,200] Hz at a fixed force budget dilute MSD identifiability
  relative to concentrated power near 157 Hz? (In noiseless sim this is about
  delta_a activation level and encoder identifiability, not SNR.)
- The 10-130 Hz mid-band: excite it (probe lines) or accept it as a gap if the
  FP model is accurate there?
- How to fix the "all oscillations look the same" problem: randomized setpoint
  sequences vs other coverage strategies, and how to cover off-orbit states the
  ANN sees in real operation.
- Root-cause the spike before trusting p2p data (Section 2).
- Closed-loop shaping: the excitation enters through a tracking controller whose
  sensitivity reshapes the injected spectrum. None of the precedent papers are
  closed-loop. Does this need compensation?

## 7. Key references (compact index; full log with links in `docs/excitation-design-literature.md`)

Augmentation / method (most relevant, VERIFIED):
- Hoekstra, Verhoek, Tóth, Schoukens, "Learning-based model augmentation with
  LFRs," European Journal of Control 86 (2025) 101304. Data gen for its MSD study
  read directly: DT multisine, 1666 components, full grid [0,25] Hz, uniform
  random phase; independent estimation/validation/test sets.
- Beintema, Tóth, Schoukens, "Nonlinear state-space identification using deep
  encoder networks," L4DC 2021 (PMLR 144). SUBNET method introduction.
- Beintema, Schoukens, Tóth, "Deep subspace encoders for nonlinear system
  identification," Automatica 156 (2023) 111210. SUBNET (name coined here).
- Bouc-Wen benchmark (Schoukens & Noël), nonlinearbenchmark.org/benchmarks/bouc-wen.
  Train: multisine full grid 5-150 Hz, 50 N RMS. Test: sinesweep 20-50 Hz +
  independent multisine. The template for "train multisine, test cross-class."

Experiment design (VERIFIED):
- Ghosh, Bombois, Huillery, Scorletti, Mercère, "Optimal identification
  experiment design for LPV systems using the local approach," Automatica 87
  (2018) 258-266. Frozen-scheduling local experiments.
- Colin, Bombois, Bako, Morelli, "Data informativity for the open-loop
  identification of MIMO systems in the prediction error framework," Automatica
  117 (2020) 109000. MIMO informativity; classical full-rank-spectrum criterion
  is sufficient but too restrictive for multisines.
- Schoukens & Ljung, "Nonlinear System Identification: A User-Oriented Road Map,"
  IEEE Control Systems Magazine 39(6), 2019. Excite full amplitude + frequency
  range of interest.

Excitation signals (VERIFIED / mixed):
- Pintelon & Schoukens, "System Identification: A Frequency Domain Approach,"
  2nd ed., Wiley-IEEE, 2012. Multisine design, period vs frequency grid.
- Nelles, "Nonlinear Dynamic System Identification," Springer. APRBS as standard
  nonlinear-sysid training signal.
- Gloth & Sinapius, "Analysis of swept-sine runs during modal identification,"
  Mechanical Systems and Signal Processing 18 (2004) 1421-1441. Sweep-rate limits
  for lightly damped (high-Q) resonances — directly relevant to any chirp near
  157 Hz.
- Vuojolainen, Nevaranta, Jastrzebski, Pyrhönen, "Comparison of Excitation
  Signals in Active Magnetic Bearing System Identification," Modeling,
  Identification and Control 38(3):123-133, 2017.

PLAUSIBLE (not fully verified — check before citing): Rivera et al.
plant-friendly identification; Schoukens & Dobrowiecki broadband user-imposed
spectrum; Heinz & Nelles multi-input space-filling design; Bolderman/Lazar/Butler
PGNN linear motors (CCTA 2021, closest hardware relative — training-data section
not yet read).

---

**Bottom line for the new session:** the framework-level precedent says train on a
broadband full-grid random-phase multisine covering all dynamics including 157 Hz,
independent realization per split, and test on a different signal class that still
hits the resonance. Everything about coordinates, amplitudes, Y-scheduling
coverage, closed-loop shaping, and off-orbit state coverage is open and only
weakly supported by literature — that is the discussion to have.
