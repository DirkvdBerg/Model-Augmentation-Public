# Step 0 FRAME: experiment design for Y-scheduled LPV grey-box augmentation

Source: `tasks/handoffs/2026-09-23-experiment-design-research.md` (read in full, 2026-09-24).
Skill: `.claude/skills/deep-research/SKILL.md`. Session decisions are logged in `DECISIONS.md`.

## The question
Supervisor (Maarten, to Roland) implied the data's "p excitation" is not properly designed. Two
readings, neither assumed: (1) persistency of excitation (PE) / informativity; (2) excitation of the
scheduling variable `p = Y`. Györök 2026 Condition 4 is the theorem that motivates the question.

## System facts the agents need (verified in this session from the generator and project docs)
- Plant: dual-gantry, 3 logical channels `[X_sym, X_anti(yaw), Y]`, `M(Y)` affine/quadratic in `Y`,
  hidden absorber (mass-spring-damper, 8 states in the truth) on the payload. Baseline: 10
  identifiable physical combinations (`kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d`).
- Closed loop: `u_total = Cfb (r - y) + f_ms`, per-record frozen-`Y` ruleOfThumb controller at 100 Hz
  bandwidth, fs = 20 kHz, record 12 s (0.5 s hold, 10 s active, 0.5 s hold). Controller known.
- Excitation: random-phase flat multisine on exact FFT bins in `[140, 230]` Hz, three independent
  realisations in logical channels (not orthogonal by construction), crest factor best-of-30,
  RMS `A_sym = 240 N`, `A_Y = 180 N`, `A_anti` sized by a 2 mm yaw budget (as generated, D-207).
- Reference `r`: T1 to T5 standstill at `Y = -0.30, -0.15, 0, 0.15, 0.30` m; T6 to T8 sinusoidal
  Y sweeps (0.2 / 0.7 / 0.35 Hz, 0.30 m); T9 to T12 jerk-limited random setpoint sequences
  ("APRBS"); T13, T14 Lissajous; validation V1 to V4 at held-out Y (0.10, -0.22, 0.10, -0.10);
  TP1 to TP4 Telica cycloidal profiles with no multisine (D-206).
- Training: SUBNET encoder + multi-step simulation loss; orthogonal-by-construction projection of
  the learned component against the baseline sensitivity `J` over a reference set of one-step tuples.
- Measured: `rho* = ||P Delta*|| / ||Delta*|| = 0.205`; `cond(J) = 766`; `cg1`/`cg2` columns 97.7 %
  correlated; bias 98.8 % along one direction (`m_diff`), caused by the absorber offset `ma L0`.
- Telica: real closed-loop, point-to-point ILC motion, no designed excitation.

## Sub-questions (one per gap, one output row each)
| ID | sub-question | agent |
|-|-|-|
| E1 | PE / informative data for grey-box physical parameters in closed loop, known controller, exogenous `r` plus injected force | A |
| E2 | Excitation requirements for learned nonlinear state-space models (SUBNET, neural SS): PE for nonlinear/neural models, state-space coverage, space-filling design | A |
| E3 | LPV: excitation of the scheduling signal `p`, local vs global design, identifiability conditions, position-dependent motion systems | B |
| E4 | Györök Condition 4 as input design (Remark 5) and relatives: calibration design with model discrepancy, bias-distribution design, application-oriented and least-costly design, design robust to unmodelled dynamics | A |
| E5 | Multisine design for MIMO closed-loop motion systems: band, amplitudes, orthogonal multisines across channels, crest factor, periodicity/leakage under feedback | B |
| E6 | Informativity of operational non-periodic point-to-point / ILC motion profiles on production machines; admissible added excitation | B |

## Seed DOIs / arXiv IDs from the document and held PDFs
- Györök et al. 2026, arXiv:2511.01321 (held): Eqs. (6), (7), (16), (17), (21), (30); Condition 4,
  Remark 5, Theorem 7, Condition 14, Sect. 5.1, 5.2 (read this session).
- Gevers, Bazanella, Bombois, Miskovic 2009 IEEE TAC "Identification and the information matrix: how
  to get just sufficiently rich?" (held `gevers2009_...`); Gevers et al. 2008 CDC informative data
  (held); Bazanella et al. 2010 EJC closed-loop MIMO identifiability (held); Colin et al. 2020
  closed-loop MIMO informativity (held).
- Gevers, Bombois, Hildebrand, Solari 2011 "Optimal experiment design" (held); Ghosh, Bombois et al.
  2018 Automatica LPV local approach (held); Bombois et al. 2019 IJC LPV local approach (held).
- Dankers et al. 2011 CDC informative data LPV-ARX (held); Dreef et al. 2022 LCSS excitation
  allocation (held); Liu et al. 2025 LCSS space-filling (held, arXiv:2502.17042); Hassaballa and
  Lazar 2026 PE (held, arXiv:2606.04290); Beintema, Schoukens, Toth 2023 Automatica SUBNET (held).
- Tuo and Wu 2015 AoS; Plumlee 2017 JASA; Brynjarsdottir and O'Hagan 2014 (held; calibration with
  discrepancy). "Design variables for bias distribution in transfer function estimation" (held).
- Toth 2010 LPV book (held, `literature/books/`); Verhoek 2025 PhD thesis (held); Cox 2018 PhD thesis
  (held); Voorhoeve et al. 2021 TCST position-dependent wafer stage (held, arXiv:1807.06942).
- Pintelon and Schoukens book (held, `literature/BLA/`); "Fast FRF Measurement of Multivariable" (held).
- Kon et al. 2022, 2023 (held; orthogonal projection on trajectory data, feedforward); van Haren
  2024 ECC finite-time ILC (held).

## Entry points (author IDs, venues; never keyword strings)
- Author IDs to resolve and enumerate: Gevers, Bazanella, Bombois, Hjalmarsson, Rojas, Colin (E1, E4);
  Beintema, Ljung, Nelles, Lazar, Bombois (E2); Toth `A5088619613`, Bachnas, Laurain, Verhoek,
  Oomen `A5050892930`, van der Maas (E3); J. Schoukens `A5076996530`, Pintelon, Dobrowiecki, Oomen,
  Evers (E5); Oomen, Bolderman, Kon, Blanken (E6).
- Venues: Automatica, IEEE TAC, IEEE TCST, IFAC SYSID (IFAC-PapersOnLine `S2898405271`), CDC, ECC,
  ACC (dblp `venue:`), Mechatronics, MSSP, IEEE TIM (E5), L4DC / PMLR (E2), JASA / SIAM-ASA JUQ (E4
  statistics vocabulary).

## Disqualification filter
1. Open-loop-only results are reported, but marked "open loop, transfer to our closed loop not shown".
2. Results that need stochastic noise to be meaningful (FIM-based optimality) are reported with the
   note that our simulated data are noiseless; the rank/conditioning part still applies.
3. LTI-only conditions are marked as applying per frozen `Y` only.
4. Anything whose condition was read from an abstract or summary only: listed, equation marked
   "not read".

## Out of scope
Generating data, training, the orthogonality derivation (session `orthogonal-addition`), noise
injection (`closed-loop-noise`), encoder work, black-box work, the meaning of "p excitation" by
assumption.

## Vocabularies to search (at least three)
control / system identification ("informative experiment", "sufficiently rich", "persistently
exciting", "least costly"); statistics / uncertainty quantification ("calibration", "model
discrepancy", "design of experiments", "identifiability"); machine learning ("coverage",
"distribution shift", "active learning", "space-filling"); structural dynamics / modal testing
("force appropriation", "multiple-input multisine"); process industry ("plant-friendly",
"routine operating data").

## Access
TU/e browser route: UNAVAILABLE in this session, layer 1 (no `mcp__claude-in-chrome__*` tools are
loaded in this session, so the bridge cannot be reached). Items that need it are marked
`needs-browser-route`.
