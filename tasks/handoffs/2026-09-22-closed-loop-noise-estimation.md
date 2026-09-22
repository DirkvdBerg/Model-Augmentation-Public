# Handoff: what does a noise floor measured from closed-loop ILC data actually represent, and how is the source noise recovered from it
**From**: session of 2026-09-22 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Establish, from the system-identification literature, what a noise spectrum measured on the
tracking error of a **closed-loop** machine represents in terms of the loop's sensitivity
functions, and what methods exist to recover the **source** noise (sensor noise at the encoder,
and process/input disturbance) from it. The supervisor's objection is the whole reason this task
exists: the Telica logs are closed-loop, so a variance measured on the logged error has already
been shaped by the sensitivity function, and it is **not** the same quantity as a noise source
injected inside a simulated loop. Injecting the measured number back into a loop applies the
sensitivity twice. Produce a written synthesis naming the candidate methods, their assumptions,
and which of them survive contact with this specific dataset (properties in section 4). Use the
`deep-research` skill; per D-121 that is mandatory for literature work and ad-hoc `WebSearch` is
not an acceptable substitute.

## 2. Out of scope

- **Do not estimate the noise from the data yet.** The user will open that as a second phase in
  the same session, after reading the literature synthesis, and wants to discuss the choice of
  method before any number is produced. Producing an estimate now pre-empts that decision.
- **Do not modify `scripts/gantry/gantry_dynamic/data.py`** or any part of the training pipeline
  to change where noise is injected. That change is designed but deliberately not made.
- **Do not touch the OBC arm work** (`gantry_interconnect_dynamic.py`, `controller.py`,
  `augmentation_ma50_z03_b140-230_a6_telica/`). It is staged for submission and unrelated.
- **Do not re-open the training-loss weighting question** (per-record NRMS, channel
  normalisation). Measured and closed on 2026-09-21; see section 6.
- **Do not attempt to resolve `_S_FORCE`** (D-112). It is an open question to Kamtin and blocks
  only force-unit work, not this task.

## 3. Where things stand

Branch `Augmentation`, last commit `ccd2fee`. Tree is dirty across many directories from
unrelated prior work. No run in flight. Nothing about noise has been implemented; everything in
section 4 is measurement only, produced by throwaway probe scripts that no longer exist.

## 4. Established and verified

All measured this session from `kamtin-data/Data Telica/06 40 mm XL 80 mm YL/`, at the native
20 kHz. The loader conventions are in `docs/kamtin-telica-schema.md`.

**Dataset properties that constrain which methods apply:**
- Closed loop, with an ILC feedforward that changes every iteration.
- The reference `M0` is **bit-identical across every ILC iteration** at an operating point (max
  difference 0.0000 um once aligned on motion start). The reference is therefore known and
  noise-free.
- ILC iterations are **not repeated realisations**: they differ by feedforward, so
  realisation-averaging and periodicity-based variance estimation do not directly apply.
- There is **exactly one true repeat pair** in the whole campaign:
  `train/xpos_-60_ypos120/iter6.log` and `iter6_1.log` (a redo of iter6).
- The feedback controller is **known and verified**: `kamtin-data/dFeedbackControllersTelica_ba.mat`
  (num/den export for scipy, Ts = 5e-5 s, 6x6 diagonal). Replay against the data gives
  correlation 0.96 to 0.97 (D-074). So `S` and `T` are computable, not assumed.
- `iter0` is a pure-feedback run (feedforward off, `MF30 == MF230` bit-identically), i.e. an
  exact input/output pair of the feedback controller.

**Measured error levels, rms on the tracking error `M2`:**

| axis | standstill (444 ms pre-motion, iter0) | move, iter0 | move, iterETEL | move, iter8 |
|-|-|-|-|-|
| BHL_GTRX1 | 8.69 nm | 3598 nm | 557 nm | 137 nm |
| BHL_GTRX2 | 10.63 nm | 4559 nm | 692 nm | 158 nm |
| BHL_GTRY | 5.94 nm | 1191 nm | 658 nm | 113 nm |

- ILC converges but the `iter8` move residual is still about **15x the standstill floor**, so
  there is non-repeatable content during motion well above the standstill level.
- Naive SNR (move error against standstill floor, iter0): **52.3 / 52.6 / 46.0 dB**.

**The standstill floor is not quantisation-limited:**
- Encoder resolution `q = 9.765625e-10 m/cnt` (from `Telica 1.mat`), so `q/sqrt(12) = 0.282 nm`
  rms and a flat equivalent PSD of `7.947e-24 m^2/Hz`.
- Measured standstill PSD in the noisiest bands is `1e-20` to `3e-20 m^2/Hz`, i.e. roughly
  **31 to 36 dB above** the quantisation floor in amplitude.
- The spectrum is **not white**: it rises into the kHz and most of the variance sits above
  200 Hz.

**Independent consistency check:** the ETEL datasheet
(`literature/gantry/telica-xyz-0750-0800-data.pdf`, ASME-YGNN-08-0750-0800W3) specifies position
stability +/-150 nm at 1 kHz, 3 sigma. The measured 8.69 nm rms broadband is 3 sigma = 26 nm,
comfortably inside spec.

**What the simulation currently does, and what the user wants instead:**
- `scripts/gantry/gantry_dynamic/data.py`, in `load_datasets`, adds output noise **post-hoc and
  open loop**: `sd.y = sd.y + N(0, sigma_n)`, with a hard-coded table `SNR 50 -> 4.5e-4`,
  `55 -> 2.5e-4`, `60 -> 1.4e-4` m (Jan's ECC convention, D-078). This noise never reaches the
  controller.
- The user has decided the noise should instead be injected **inside the loop**, at the encoder,
  because that is what the real machine does and it is what ASMPT cares about. That decision is
  made; this task is about what number to inject, not whether.
- Consequence that makes this task load-bearing: if the simulator applies `S` itself, the
  injected quantity must be the **source** noise, not the measured closed-loop error.

## 5. Assumed but not verified

- **That the standstill error is dominated by sensor noise rather than process disturbance.**
  Unverified, and the 31 to 36 dB gap above quantisation argues against a purely-sensor
  explanation. Settled by fitting a two-source model (section 8) or by an independent
  measurement of the stage at rest with the loop open, which is not available.
- **That the rising-into-kHz shape is the source spectrum rather than the loop shaping it.**
  This is precisely the confound the task exists to resolve. `S` rises toward 1 at high
  frequency, so a white sensor noise would already produce a rising error spectrum.
- **That `M2` is a clean error signal.** `MF230` lags `K(M2)` with a broad correlation maximum
  around 2.5 ms, which cannot be a physical loop delay and is attributed to the logging path
  (D-074, `diag_cl_correctness.py` check 4). Any method that uses `M2` and `MF230` jointly in
  the frequency domain must handle this.
- **Force units.** `_S_FORCE = [3.469, 3.469, 3.202]` on top of Kt is HEURISTIC with an open
  question to Kamtin (D-112). Any method needing forces in Newtons inherits that uncertainty.

## 6. Tried and failed

- **Proposing that the measured standstill variance be injected directly into the simulated
  loop** -> would apply the sensitivity twice, overstating the noise by `|S|^2` -> the
  simulator's own loop reproduces the shaping that the measurement already contains -> this is
  the supervisor's objection and the reason for this handoff. Do not re-derive it; take it as
  the premise.
- **Proposing to inject only the encoder quantisation noise** (0.282 nm rms, white) -> would
  understate the real floor by about three orders of magnitude in power -> the measured PSD sits
  31 to 36 dB above the quantisation floor, so quantisation is not the dominant source ->
  measured, section 4.
- **A hypothesis that the training loss starves the absorber-bearing records**, which motivated a
  per-record loss reweighting -> refuted -> per-record gradient norms at true physical parameters
  are uniform across all 14 production records (loss spread 1.14x, augmentation-gradient spread
  1.4x) -> measured 2026-09-21. Listed here only so it is not revived; it is out of scope.

## 7. Achieved

Nothing implemented. The contribution of this session on the noise question is the set of
measurements in section 4 and the identification of the confound in section 5. Treat section 4 as
inputs, not as results to reproduce.

## 8. The open question

**What does the measured standstill error spectrum represent, and which inversion recovers the
source noise?** Three candidate framings, and the evidence that would choose between them:

1. **Single sensor-noise source.** Model the measured error as `S * n` with `n` white at the
   encoder, and invert with the known `S`. Cheap. Chosen if the de-shaped `n` comes out
   approximately white and near the quantisation floor. Current evidence argues against it: the
   measured level is far above quantisation.
2. **Two sources, sensor plus input disturbance.** Fit the measured standstill PSD as
   `|S|^2 * sigma_n^2 + |S*G|^2 * Phi_d(omega)`, with `sigma_n` pinned at the quantisation floor.
   The two terms are separable by spectral shape, since `S*n` and `S*G*d` have different
   roll-offs. Chosen if the fit residual is small and `Phi_d` comes out physically plausible.
3. **Iteration-domain estimation from the ILC residual.** ILC removes the repeatable part, so the
   converged `iter8` residual bounds the non-repeatable content during motion, which standstill
   cannot observe. Whether the ILC literature gives a principled estimator here, rather than just
   a bound, is one of the things the literature search must answer.

A fourth possibility the literature may surface: the single genuine repeat pair
(`iter6` / `iter6_1`) gives `2 * sigma^2` directly on their difference, but from one pair only.

**A related question the user has flagged as the second phase:** given the answer above, what is
the *optimal* estimator for this data, as opposed to merely a valid one. Do not pre-empt it.

## 9. Next action

Invoke the `deep-research` skill on the question in section 1, framed around these four
sub-questions, which are independent enough to warrant one subagent each:

1. **Closed-loop identification and noise estimation.** Direct, indirect and joint input-output
   methods; what each assumes about the controller being known; how the noise model is identified
   in closed loop (Box-Jenkins and related). Ljung, Forssell & Ljung, Van den Hof & Schrama.
2. **Nonparametric noise analysis with a known controller.** The local polynomial method and
   related frequency-domain approaches for separating noise from the plant in closed loop; what
   they require in terms of excitation and repetitions. Pintelon & Schoukens; note that their
   standard machinery assumes periodic excitation, which this dataset does not have.
3. **Sensitivity-function inversion and disturbance source attribution.** How the literature
   separates sensor noise from input/process disturbance given only closed-loop output data and a
   known controller; identifiability limits of that separation.
4. **ILC residual as a noise estimator.** Whether iteration-domain convergence gives a principled
   estimator of the non-repetitive disturbance, or only a bound.

Carry one hypothesis into the search rather than leaving it to be rediscovered: the project's own
`telica_residual_spectrum.py` argues that the **feedback current is the better channel** for
seeing what the loop could not reject, because the tracking error is sensitivity-suppressed while
the feedback effort is not. Whether the noise-estimation literature supports estimating the source
noise from the feedback signal, or from `e` and `i_fb` jointly, is a specific thing to look for.

Return the skill's mandatory Research Log, plus a synthesis that states for each named method:
its assumptions, whether this dataset satisfies them (against the properties in section 4), and
what it would yield. Report every method found, including ones judged inapplicable, with the
reason for exclusion. Do not filter for relevance in the same pass as the search.

## 10. Acceptance criterion

Done when the synthesis lets the user choose an estimator without further reading: each candidate
method named with a citation, its assumptions checked one by one against section 4, and an
explicit statement of which fail and why. The criterion is coverage and traceability, not a
number, because no estimate is produced in this phase. Any method that survives must come with
the quantity it actually estimates, stated as a formula in terms of `S`, `T`, `G` and `K`, so
that the follow-on phase knows what to inject where.

## 11. Read these first

1. `docs/kamtin-telica-schema.md` — the only permitted reference for the data's columns, units,
   structure and split. The raw folder is blocked by `.claudeignore`; this file replaces reading it.
2. `scripts/gantry/real-data-verification/telica_residual_spectrum.py` — **prior art on exactly
   this confound**, and the single most useful file here. Its header already argues that the
   tracking error is suppressed below crossover by the sensitivity so a low-frequency feature is
   invisible in `e`, and that the feedback current `i_fb` is not attenuated the same way and is
   "the loop's own estimate of what the model does not explain". It also states the limit that
   every peak in closed-loop data is a property of plant AND controller together. Read this
   before the literature search, so the search extends it rather than rediscovers it.
3. `scripts/gantry/real-data-verification/telica_loader.py` — the sanctioned way to read the
   blocked data. See section 13 for the API.
4. `D-074` in `docs/decisions.md` — controller verification, the 0.96 to 0.97 replay correlation,
   and the 2.5 ms `MF230` logging lag.
5. `docs/control-reasoning.md` — the project's control stance, including the loop and
   noise-setting checks that this task is an instance of.

Pointers, not reading assignments: `D-112` (the `_S_FORCE` open question, section 5),
`scripts/gantry/gantry_dynamic/data.py::load_datasets` (where noise is injected today, section 4),
`scripts/gantry/real-data-verification/telica_controller.py` (steppable controller, section 13).

## 12. Do not

- Do not use ad-hoc `WebSearch` for the literature sweep; D-121 requires the `deep-research` skill.
- Do not read anything under `kamtin-data/Data Telica/` with the file tools, and do not
  `cat`/`head`/`grep` it. It is blocked by `.claudeignore`; `docs/kamtin-telica-schema.md` is the
  substitute. A Python script may open those files, but ask the user before the first such run
  and have it print derived quantities only. See section 13.
- Do not touch `kamtin-data/Telica.mat`. Off-limits per user instruction, and it is an AeroPro
  machine rather than Telica (D-069). `Telica 1.mat` is the correct hardware-constant source.
- Do not write a new log parser or a new controller stepper. Both exist and are verified
  (section 13).
- Do not inject the measured closed-loop variance into the simulated loop (section 6).
- Do not treat the ILC iterations as repeated realisations; they differ by feedforward.
- Do not produce a noise estimate in this phase.

## 13. Operational

No run to launch. Literature work only. Everything below is for the second phase, so that it does
not have to be reconstructed.

**How to reach the data, which is blocked by `.claudeignore`.** The block stops the file tools and
shell inspection (`cat`, `head`, `grep`) on `kamtin-data/Data Telica/`. It does not stop a Python
script from opening those files, and in this session the user authorised exactly that: a script
that reads the logs and prints only derived quantities. Ask before the first such run; do not read
the folder with tools.

**Use the existing loader rather than parsing logs by hand.**
`scripts/gantry/real-data-verification/telica_loader.py`:

- `load_telica_log(path)` -> `(u [N], q1 [m], fs)` for open-loop use.
- `load_telica_log_cl(path)` -> `{r, q1, u_ff, i_fb, fs}` for closed-loop use. This is the one
  this task wants: `r` is the reference in m, `q1` the measured position in m, `u_ff` the ILC
  feedforward force in N, `i_fb` the logged feedback current in A. Tracking error in metres is
  `r - q1`, which equals `M2 * 1e-6`.
- It trims to 50 ms of standstill before motion start by default, so a script that needs the full
  444 ms pre-motion window must not rely on the trimmed output.
- It applies `_S_FORCE` to every force it returns. `i_fb` is left in amperes, so a method working
  on the feedback signal can avoid the D-112 uncertainty entirely by staying in amperes.

**Controller.** `scripts/gantry/real-data-verification/telica_controller.py` is a steppable
version of the verified controller (input tracking error in m, output feedback current in A,
20 kHz, BHL uses rows LX1/LX2/LY). Use it rather than re-exporting from the `.mat`. Source file is
`kamtin-data/dFeedbackControllersTelica_ba.mat`; `export_controllers_ba.m` regenerates it.

**Related scripts in the same folder**, worth knowing exist before writing anything new:
`telica_plant_frf.py`, `show_controller_mismatch.py`, `run_telica_param_recovery.py`,
and the `diagnostics/` subfolder.

**Other data files.** `kamtin-data/Telica 1.mat` holds the hardware constants (Kt, encoder
resolution, sampling time). `kamtin-data/Telica.mat` is off-limits per user instruction and is an
AeroPro machine, not Telica (D-069). The ETEL datasheet is
`literature/gantry/telica-xyz-0750-0800-data.pdf`.

**Environment.** conda env `GraduationProject`. The live-output convention in `CLAUDE.md` applies
to anything running more than a few seconds, and `conda run python -c` cannot take a multi-line
argument, so write snippets to a scratchpad file and run that.

## 14. Delegation

One Explore subagent per sub-question in section 9, so four, run through the `deep-research`
skill's own fan-out rather than spawned directly. That is above the project default of one and is
justified because the four are genuinely independent literatures (closed-loop ID, frequency-domain
nonparametrics, disturbance attribution, ILC). Do not exceed four. Do not spawn a subagent to
review or verify the synthesis.
