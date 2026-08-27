# Handoff: run the random-init and fitted-init arms, and settle whether the augmented states can learn at all

**From**: session of 2026-08-22 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Phase A of `scripts/gantry/BLA-Augmentation/DESIGN.md` is written: D1 through D8 all carry a filled
`Decision:` block, `EVIDENCE.md` holds 28 machine-verified claims, and `RESULTS.md` holds the
measurements. **Your task is Phase B: implement BOTH initialisations and make the BLA one reach
`e-7`.** Restore one file, write the augmented block as new code, implement the random init and the
BLA init, and run them against a control.

**The two arms are the main goal. Neither is optional.**

**The bar, in one sentence: the BLA arm must reach `e-7` WITHOUT HEURISTICS.** That is the whole
point of the exercise and it is what makes it a result rather than a repeat. The mechanism being
replaced already reached `3.795974e-07`, but it did so with **four unexplained constants**
(`over_floor_db > 10`, `AUG_LRU_B = 0.377`, the phase-arc restriction, and the `[min,max]`-scatter
band rule), a threshold measured to be unsafe on coloured backgrounds, and a random pole draw whose
spread over five seeds was `20.1 %`. Matching that number with a construction in which **every choice
is cited or derived** is the deliverable. Matching it by reintroducing a tuned constant is not, and
neither is missing it and reporting that something was learned.

**A result obtained using an unresolved constant does not count.** Two are outstanding and are named
in section 8: D5's `eps`, and the `2.0x` ablation threshold. Resolve each by derivation or by a
`CONFIRMED` citation, or make the code refuse. Do not let either slip into an arm and then quote the
arm's RMS.

**Read the literature; do not merely cite it.** `EVIDENCE.md` is a starting point, not a substitute.
Anything D9 or D10 needs that is not already a `CONFIRMED` claim requires opening the PDF and running
`verify_pdf_quote.py`, and both outstanding constants above are more likely to be solved by finding
the right source than by inventing an argument.

### STOP AND REPORT RATHER THAN DESIGNING AROUND IT

**This is the instruction the 2026-08-22 session failed, and it is why that night produced a design
instead of a result.** The prompt said: if claims 1-3 come back REFUTED, stop and report, that is a
success not a setback. They came back CONFIRMED *as statements about the papers*, and the session
then discovered they did **not** support the design's central step. It read "REFUTED" as a verdict
field rather than as intent, wrote the gap into a footnote, and built seven more decisions on top.

**The same failure is available to you. These are the triggers. On any of them, stop, write the
verdict row, and report - do not build around it.**

* the pre-flight `dL/dp` at step 1 is zero, so `DESIGN.md` D7's amendment is wrong;
* the Telica-portability audit turns up a `NOT available` entry that the construction depends on and
  for which no substitute exists;
* D5's `eps` or the `2.0x` threshold cannot be derived or cited, and refusing them makes an arm
  unrunnable;
* a `CONFIRMED` claim turns out not to support the step that cites it - **the exact 2026-08-22
  failure**;
* the Hankel spectrum does not exist at any order where the fit is usable, so `nx_aug` cannot be
  derived.

Reporting a blocked night with the block located is worth more than a night of work built on
something that does not hold. **A negative that is legible is the deliverable when the positive is
not available.**

### THE FILES ARE THE PRODUCT

`EVIDENCE.md`, `DESIGN.md`, `RESULTS.md` and the verdict file are the deliverable. **Do not restate
them elsewhere, and do not produce analysis in conversation that belongs in them.** The 2026-08-22
session spent eight exchanges reasoning in chat about `rho`, the gradient chain, the routing and the
`PL.FA` error, and filed none of it until told to. Write findings into the file that owns them at the
moment you have them; if a finding contradicts something already written there, amend that text
rather than appending a note beside it.

### EXTENDABILITY TO NOISE AND THE REAL SYSTEM IS A DELIVERABLE, NOT A CAVEAT

The user has raised this three times and the previous session let it slip twice. It is not enough to
note at the end that a result might not transfer. **Make it checkable:**

**1. Every arm runs both noise conditions.** Noiseless and 1x Telica sigma
(`8.544e-9, 7.762e-9, 6.539e-9` m) with `CL_NOISE_CONSISTENT=1`, which correlates `u` with the noise
by construction and is therefore the case that matters. Four training runs, not two. An arm reported
noiseless only is an incomplete arm.

**2. Produce a Telica-portability audit as part of D9**, listing every quantity the implementation
consumes and marking it `available` / `NOT available` / `substituted by <what>`. The previous session
built a residual construction requiring an `x0` that Telica does not have and did not notice for
several hours. The audit is what catches that class of error. Known entries to start from:

| quantity | on Telica |
|-|-|
| `u`, `y` | available |
| `M0` setpoint, as the instrument or regressor | available, `docs/kamtin-telica-schema.md` |
| `C_fb` | available, bit-exact, D-074 |
| **`x0` as a true state** | **NOT available.** Positions measured, velocities differentiated from noisy positions |
| **baseline parameters** | **estimates, not truth.** So `rho = missing dynamics + parameter mismatch + noise` |
| **`157.8937 Hz`, `zeta = 0.05276`, `deriv8`** | **NOT available.** Simulation only |
| sample rate, identifiable band | **different**: 20 kHz against our 4 kHz; below 83 Hz on X and 55 Hz on Y |

**3. Nothing band-dependent may be hard-coded to the simulation.** `DEC`, any order range, any
frequency window must be parameters derived from the record's own rate and content, because
`157.89 Hz` is above Telica's identifiable band on both axes and could not be found there even if it
existed.

**4. The refusal condition must be live and exercised.** On Telica the likely outcome is that **no
clean mode is identifiable at all** - there is no absorber there, and the missing physics is
friction, stick-slip, cable forces and drift. D3 requires the estimator to return nothing rather than
a defaulted band. Demonstrate that it does, by feeding it a case where there is nothing to find.

The user's constraint, stated three times: **no heuristics, and nothing that only works noiseless.**

## 2. Out of scope

* **Do not modify** `kamtin-fp-model/`. Do not commit or push `model_augmentation/` or
  `gantry_dynamic/`.
* **Do not run the wave-1 ablation factorial.** Its band arms test a mechanism that has been deleted.
  The runners stay on disk as a record.
* **Do not re-verify** the 28 claims in `EVIDENCE.md`. Each carries its `MATCH OK` and its page. The
  four that carry a correction say so in their own text.
* **Do not implement a Telica arm.** `kamtin-data/Data Telica/` is blocked by policy. Telica is a
  design constraint, not a dataset to touch.
* **Do not rewrite Phase A.** D1-D8 are decided. Three of them carry an explicit not-yet-measured
  item, listed in section 5; those are measurements, not redecisions.

## 3. Where things stand

Branch `Augmentation`, last commit `55980b7`. **`model_augmentation/` and `kamtin-fp-model/` are
clean** - `git status --porcelain` returns zero lines for both. Phase A touched no production file.
`closed_loop.py` has **not** been restored.

New this session, all untracked, all under `scripts/gantry/BLA-Augmentation/`: `EVIDENCE.md`,
`DESIGN.md`, `RESULTS.md`, `probe_d1_residual_identity.py`, `probe_d8_residual_fit.py`,
`probe_pole_gate.py`, and seven JSON artefacts in `runs/`. Five PDFs were fetched into
`literature/` (`marconato2014init`, `relan2017lpm_bla`, `forgione2024mor`, `pontesduff2019tlbt`,
`bachlechner2020rezero`) plus four for D4 and D6.

**Nothing is in flight.** The last run, `probe_pole_gate.py`, completed and its artefact is
`runs/pole_gate.json`, 72 rows. Its result is in section 4: **the planted simulation mode is
recoverable at Telica-magnitude noise with a stable realisation**, so arm A2 has a construction and
is not blocked. Read the scope note under that table before quoting it: it is a result about the
simulation, not about Telica.

## 4. Established and verified

| fact | evidence |
|-|-|
| **`residual_for` returns `S*rho`, not the residual.** The baseline is simulated inside a tracking loop. Suppression `rms(rho)/rms(r)` spans `10.1` to `642.0` across three records; `\|r - S rho\|/\|r\|` is `0.24 %` to `52 %` | `runs/d1_residual_identity_*.json`, D-153 |
| **`r` is loop-limited, not disturbance-limited.** `rms(rho)` varies over three orders of magnitude across records while `rms(r)` is nearly constant at `~3.7e-07` (X) and `~3.7e-06` (Y) in all nine record-channels | same |
| Controller replay is clean on standstill and y-sweep records (at or below the float32 storage floor once the `z = 1` ramp is removed) and **`6x` to `18x` over the floor on APRBS records**, at `~10^-3` relative to `rms(u_fb)` | `verify_cfb_against_records.py`, `RESULTS.md` D1a |
| **The undifferenced residual model is unstable at every order that fits.** `rho(A_r) > 1` from `na = 8` up, so the Hankel spectrum does not exist and D5's order rule is unevaluable | `runs/d8_residual_fit.json` |
| **No over-fitting.** In-sample and out-of-sample VAF agree to `0.003` at every order; the turnover is in the p10 across record-channels (`0.8195` at `na=8` falling to `0.6654` at `na=28`), not in the median | same |
| **8.5 nm of sensor noise becomes 3 to 33 mm of open-loop residual.** `-C_fb(v)` injected as force noise through the rigid-body double integrators | `runs/d8_residual_fit_noisy.json` |
| Differencing by `(1-q^-1)^2` removes it: `rms(rho)` on X falls from `2.4e-03..3.3e-02` m to `2.6e-08..1.1e-07` m | `runs/d8_residual_fit_noisy_diff_iv.json` |
| **`PL.FA = 150.0 Hz` is the ISOLATED absorber, not the coupled mode at `157.8937 Hz`**, a 5 % difference. `cl_residual_spectrum.py` prints `plant.FA` as its "VALIDATION TARGET (simulation only)" | `probe_pole_gate.py:54-63` |
| 28 claims read at the PDF: 23 CONFIRMED, 2 PARTIAL, 1 REFUTED, 1 UNREADABLE | `EVIDENCE.md` verification log |

### THE POLE GATE PASSED. A2 has something to install.

`runs/pole_gate.json`, 72 rows, sweeping `{clean, noisy} x {differenced, not} x {ARX, IV}` over nine
orders. Best **installable** row (`rho(A_r) < 1`) per combination, scored against the coupled mode
`157.8937 Hz` / `zeta = 0.05276`:

| noisy | differenced | est | `na` | `f` [Hz] | `zeta` | `f` err |
|-|-|-|-|-|-|-|
| no | no | ARX | 4 | `157.730` | `0.05461` | `-0.104 %` |
| no | no | IV | 28 | `162.764` | `0.12472` | `+3.085 %` |
| no | yes | ARX | 12 | `158.105` | `0.05275` | `+0.134 %` |
| no | yes | IV | 16 | `157.999` | `0.05269` | `+0.067 %` |
| **yes** | **yes** | **ARX** | **12** | **`158.091`** | **`0.05274`** | **`+0.125 %`** |
| **yes** | **yes** | **IV** | **28** | **`157.710`** | **`0.05208`** | **`-0.116 %`** |

**At Telica-magnitude noise the planted mode is recovered to about `0.12 %` in frequency and `0.04 %`
in damping, with a stable realisation.** That is the result A2 needs for the simulation arms.

**SCOPE, and do not quote the table without it. "1x Telica sigma" is a NOISE LEVEL, not Telica.**
`TELICA_SIGMA` sets the magnitude of the measurement noise added to a **simulation**. The system is
still `deriv8`, and `deriv6` differs from it **only by the absorber**. **Telica has no absorber.**
So this table establishes that the estimator chain can recover a *known, planted, lightly damped*
mode from 14 records at that noise magnitude. It establishes **nothing** about what is recoverable on
the real machine, for three reasons already recorded in `DESIGN.md` D3:

* on Telica `rho = missing dynamics + parameter mismatch + noise`, because the baseline parameters
  are estimates, whereas here the residual is a near-pure modal signature at 65-168 dB over floor;
* the missing physics there is friction, stick-slip, cable forces and drift, which there is no reason
  to expect presents as a clean lightly damped pole pair at all;
* Telica's identifiable band is **below 83 Hz on X and 55 Hz on Y**, so `157.89 Hz` could not be
  identified there even if it existed.

**What transfers is the procedure and its refusal condition, not the number.** D3's requirement that
the estimator return nothing rather than a defaulted band when no mode is identifiable is what makes
the method runnable on Telica; this table is the evidence that the procedure works when there *is*
something to find.

Three readings that change the plan:

* **Differencing is REQUIRED under noise, not optional.** The `noisy, undifferenced` combinations
  produced **no installable rows at all**. So the `8.7 dB` SNR cost is real and is worth paying; the
  earlier objection to route (b) was correct in magnitude and wrong in conclusion.
* **ARX at `na = 12` is not worse than IV here.** Its `zeta` is `0.05274` against a truth of
  `0.05276`, better than IV's `0.05208` at `na = 28`, and at less than half the order. The
  "use IV under noise" mandate comes from a 10x/100x sigma measurement; **at 1x sigma it is not
  established that IV wins.** Do not assume it. Both are implemented behind `--iv`.
* **Order matters more than estimator.** Nothing below `na = 12` is trustworthy under noise.

**The four literature results Phase B depends on**, all `MATCH OK`:

* **claim 24**: `hoekstra2026lfrfp` Table 1's `S-DP` class has **no output equation**. `C_r = 0` is a
  named class in the framework's own paper, used by its MSD examples whenever the missing dynamics
  are in the plant. Output classes appear only in config (c), the output-LPF case, which the paper
  calls "an identifiability problem left to future research".
* **claim 26**: `f_base(x_b,k, u_k)` and `h_base(x_b,k, u_k)` take **no `x_a`** in any class. Our code
  enforces this with `selection_matrix(PHY_IX, nxd)` at `gantry_dynamic/model.py:135` and `:139`.
* **claim 27**: learning functions are ResNets and **only the nonlinear component is zeroed**; the
  linear component stays live. That is how Jan's augmented states are trainable at all.
* **claim 28**: ReZero, `x_{i+1} = x_i + alpha_i F(x_i)`, "Initially the gradients for all parameters
  defining F vanish, but dynamically evolve to suitable values during initial stages of training."

### CONTESTED, DO NOT INHERIT AS SETTLED: D1's move to an OPEN-LOOP residual

**The user rejected this on 2026-08-22 and was right. Treat `DESIGN.md` D1b's decision as open, not
as a Phase A result.**

The gantry is **closed-loop operated and open-loop unstable** - the baseline carries rigid-body
double integrators. Simulating `P0` open loop over a 12 s record is a condition the machine is never
in. Every downstream problem the previous session then spent hours patching is that category error
reporting itself, and each one was treated as a defect to fix rather than as evidence the
construction was wrong:

* the residual drifts without bound (`x0` error times `t`);
* 8.5 nm of sensor noise becomes 3 to 33 mm of residual;
* stabilising it needs `(1-q^-1)^2` differencing, which costs `8.7 dB` at the mode;
* `x0` does not exist as a true state on Telica at all.

**What D1b actually established is narrower and still stands**: the in-loop construction returns
`S*rho`, so `S` filters the residual and contributes its own poles, and the recorded numbers were
measured through that filter. **That is an argument for accounting for `S`, which is exactly known
since `P0` and `C_fb` both are, not for leaving the loop.**

**The closed-loop-native construction is already cited in `EVIDENCE.md` and was never used.** Claim 6,
`pintelon2020bla_feedback`: the BLA of a system **operating in feedback** is
`G_BLA = E{Y R*}/E{U R*}`, a ratio of two projections onto the **reference**, with no open-loop
simulation anywhere in it. Claim 11 is why the reference is the right regressor. Claim 5's SNR
passage says the same from a third direction. The project also holds roughly thirty papers in
`literature/closed-loop-id/`. **Start there rather than repairing the open-loop route.**

### PREREQUISITE THE PREVIOUS SESSION TREATED AS A CAVEAT: the baseline is not properly fitted

In simulation `deriv6` is exact, which is what let the previous session ignore this. **On the real
system the baseline parameters are estimates**, so the residual is
`missing dynamics + parameter mismatch + noise` and there is no way to separate them from one record
set. Fitting an augmentation on top of an unfitted baseline means the augmentation **absorbs
parameter error**, which is the negation failure mode (`CLAUDE.md`, Control Engineering Stance item
6) and the thing the orthogonal-projection contribution exists to prevent.

**So the Telica story has an ordering the previous session never stated: fit the baseline, then
augment.** Nothing in this handoff's simulation arms depends on it, because `deriv6` is exact there.
Everything in the transferability claim does. Do not write a Telica portability argument that steps
over it.

## 5. Assumed but not verified

* **"The plateau was learning but not through the augmented states."** Half measured: the movement
  from `2.1866011034e-06` to `1.3933793e-06` is real. **"Not through the augmented states" has no
  ablation behind it**, and D-130's zero-gradient measurement covers `B_a`, `nu_log`, `theta_log`,
  which are `AUG_LRU` parameters **not present at `4cdb7c1`**. A competing measured explanation
  exists: `cl_reachability.py` reports a `34.83x` barrier with `cos(-grad, w_planted - w_trained)`
  exactly `0.0000`. **An ablation on the plateaued checkpoint settles it. That is arm A0.**
* **That ReZero plus a random `W^a` unblocks the gradient.** Derived in `DESIGN.md` D7's amendment,
  not measured. The pre-flight in section 9 measures it and is a hard gate.
* **That anything is recoverable from the residual ON TELICA.** The pole gate settled the simulation
  case and is in section 4. It did **not** settle this one and cannot: **Telica has no absorber.**
  `deriv6` and `deriv8` differ only by it, so the simulation residual is a near-pure modal signature;
  Telica's is friction, stick-slip, cable forces and drift on top of parameter mismatch, in a band
  below 83 Hz on X and 55 Hz on Y. **Nothing measured so far bears on whether a mode exists there.**
  What would settle it is running the estimator on Telica logs and seeing whether D3's refusal
  condition fires, which is out of scope here because `kamtin-data/Data Telica/` is blocked.
* **`nx_aug`.** `8` must not be assumed. D5's rule exists but its tolerance is flagged as underived
  (section 8). `hoekstra2026lfrfp` config (a) uses `nx_aug = 2` as "the minimum number of states
  required" for one missing DOF; the user reports `8` beat `2` empirically in the unclean build, a
  result they describe as possibly confounded.

## 6. Tried and failed

- **Gating Arm B on free-run VAF of the residual model** -> declared a gate failure, then retracted
  it -> the design installs `A_r`, a pole set, and never uses free-run prediction of `rho`; worse,
  the differenced noisy target is dominated by a floor (Y pinned at `~2.0e-06` m on **every** record
  to three figures, which is `C_fb(v)` through `1/(m s^2)`), so VAF near zero is what a perfect model
  of the signal part also returns -> `DESIGN.md` D8, `probe_pole_gate.py` docstring.
- **`(1-q^-1)^2` differencing as the D4 fix** -> removes the drift, five orders of magnitude, but
  costs `8.7 dB` of SNR at the mode -> derivable: at `fs_eff = 1000 Hz` the mode sits at
  `w = 0.9927`, so the signal picks up `|1-e^-jw|^4 = 0.818` while white noise picks up
  `E[(2-2cos w)^2] = 6` -> `DESIGN.md` D1 amendment. **Route (a), a stability-constrained fit that
  leaves the data alone and costs no SNR, was never tried.**
- **The open-loop residual as the Telica route** -> `x0` is not a true state there (velocities from
  differentiation of noisy positions) and `P0`'s parameters are estimates; open loop **accumulates**
  both over 12 s where the in-loop version decays them -> `DESIGN.md` D1 amendment.
- **Scoring the pole probe against `PL.FA`** -> reports the wrong pole, 5 % off -> `PL.FA` is the
  isolated absorber.
- **Two background runs writing one output file** -> interleaved, produced an internally inconsistent
  table that briefly looked like a result -> use a distinct output path per run.
- **A patch script using `str.replace` without asserting the match** -> two replacements silently
  no-opped and the run died on `NameError` after 50 s of setup -> assert each replacement.

## 7. Achieved

**Phase A, complete as a design.** `EVIDENCE.md` 28 claims. `DESIGN.md` D1-D8 filled, with a premise
section that states the gap this work exists to fill and which the skeleton never argued: **their
known part is fitted to data and already carries unanticipated linear dynamics, so a random added
block suffices; ours is physics-derived and carries only what was derived, so missing *linear*
dynamics must come from the added block and a random draw has no reason to find a mode at 158 Hz.**
`RESULTS.md` carries D1a, D1b and D8 with a "Not measured" table.

**Implemented and validated**: `probe_d1_residual_identity.py` (the `r = S rho` identity, three
records), `probe_d8_residual_fit.py` (shared-denominator fit, per-record `x0`, Hankel spectrum,
`--diff`, `--iv`, refuses rather than crashing).

**Implemented and validated**: `probe_pole_gate.py`, 72 rows, `runs/pole_gate.json`. Validated in
the sense that it recovers a known planted mode; see the scope note in section 4 for what that
does and does not license.

## 8. The open question

**Can the augmented states be made to learn at all, and is that enough?**

Three candidate answers, and arm A0 plus A1 in section 9 choose between them:

1. **The dead zone was the whole story.** A1 unsticks the plateau and reaches the target band.
   Then the BLA line closes and the thesis contribution is the trainability analysis.
2. **The dead zone is real but insufficient.** A1 unblocks the gradient (pre-flight confirms) yet
   the RMS does not move. Then a fitted initialisation has its measured justification and A2 matters.
3. **The plateau was never the dead zone.** A0's ablation shows `x_a` was already load-bearing at
   `1.3934e-06`. Then D7's amendment is wrong and the `34.83x` barrier is the explanation.

**If the pole gate says the current construction cannot deliver an installable `A_r`, this is the
route.** Do not drop A2. The open-loop residual `rho = y - P0 u` is the weak link: it needs an `x0`
that Telica does not have, it accumulates parameter error over 12 s, and stabilising it by
differencing costs `8.7 dB` at the mode (section 6). `marconato2014init`, `EVIDENCE.md` claim 10,
**never simulates**: it estimates the state sequence as a trade-off between the linear model and the
data fit (its eq. (8), so the data term pins it) and then regresses what the state equation is
missing pointwise (eqs. (9)-(10)). No initial-condition error propagates, nothing integrates, no
differencing, therefore none of the three failures. The PDF is held and its quotes are verified. The
adaptation to be explicit about: Marconato's missing term is a static nonlinearity and ours is extra
states.

**Two constants are still unresolved and the code must refuse rather than default on both**: D5's
`eps`, which currently ties an `H-infinity` tolerance to a VAF difference with no derivation, and the
`2.0x` ablation threshold, still carried as `# HEURISTIC:` from C8. The user has ruled out heuristics
explicitly. `eps` proposal not yet written out: set the tolerance from the measured noise
contribution in the same norm, which is data-derived and computable on Telica.

## 9. Next action

**Restore `model_augmentation/fit_systems/closed_loop.py` from
`tasks/snapshots/2026-08-22-working-implementation/`, run the pre-flight, then launch arm A0.**

The restore is the one file measured as necessary: 345 lines carrying the D-147 rollout, `xc = 0`
windowing, and `window_starts` / `make_window_tensors`, which do not exist at `4cdb7c1` and which
`cl_train.py` imports by name. Nothing else from the archive comes back.

**Pre-flight, about 20 minutes, and it is a hard gate.** `probe_d072_matrix.py` for bit-identical
baseline equality with `W^a` Xavier and ReZero on; then print `dL/dp` for `W^a` and the augmented
rows at step 0 and step 1. **Step 1 must be non-zero.** If it is zero, D7's amendment is wrong and
no arm should run.

Then the arms, all at 4 epochs, full routing, `nx_aug` held identical, noiseless and at 1x Telica
sigma, ablation on each. Write the D-090 row before each launch.

| arm | change | falsifier fixed before launch |
|-|-|-|
| **A0** control | `4cdb7c1` config, post-restore | ablation ratio near `1.0` means `x_a` was dead, confirming D7; a high ratio refutes it |
| **A1** random | `W^a` Xavier + ReZero + `A_aa` Orvieto ring, no band | gradient non-zero at step 1 but RMS not below A0 means the dead zone was not the limit |
| **A2** fitted | `A_aa <- A_r`, `B_a` `u`-columns `<- B_r` with the `x_b` path kept, `C_r` discarded | A2 within noise of A1 closes the BLA line |

Each arm is **two runs**, noiseless and at 1x Telica sigma with `CL_NOISE_CONSISTENT=1`, plus its
ablation. Write the Telica-portability audit into D9 **before** launching A2, not after: its job is to
catch a construction that cannot transfer, and it only does that if it comes first.

**Run unattended.** Block W (write everything, run nothing) then the pre-flight then the runs in
section 13's order, applying section 13's drop order without asking. **10 hours of script time, 4
epochs maximum.** The budget does not close; five of the six runs is the realistic number, and
section 13 says which one goes first.

**A2's construction is settled by the pole gate: `rho`, `(1-q^-1)^2` differenced, shared-denominator
fit at `na = 12` to `28`, realised and reduced per D5.** That combination recovers the mode to
`0.12 %` in frequency and `0.04 %` in damping at 1x Telica sigma with `rho(A_r) < 1`. Run ARX and IV
both; the gate does not establish a winner at this noise level.

Section 8's Marconato route is **no longer the first move**. Keep it as the answer if A2's trained
result disappoints for a reason traceable to the residual construction, and as the Telica story if
the `x0` requirement becomes binding there.

## 10. Acceptance criterion

**The bar for A2, set by the user and not negotiable: `e-7` WITHOUT HEURISTICS.**

Free-run validation RMS on V1-V4, 4 epochs, `na_nb = 17`, serial validation. Concretely `< 1.0e-06`,
with the target band `3.80e-07` to `4.89e-07`, the discarded mechanism's measured spread over its
only two draws. `< 1.3933793e-06` merely beats the plateau and is the floor for "something is
happening", not the bar.

**The result is void if any of these is true**, because the point is the *without heuristics* half,
not the number:

* D5's `eps`, or the `2.0x` ablation threshold, was used unresolved rather than derived, cited, or
  refused;
* a constant was tuned to reach the number;
* `nx_aug` was picked rather than derived, or the band restriction came back in any form;
* **the arm was run noiseless only**, rather than both conditions;
* **the Telica-portability audit is missing**, or an entry in it is `NOT available` and was used
  anyway without a stated substitute;
* a band, order range or rate was hard-coded to the simulation rather than derived from the record;
* the refusal condition was never exercised on a case with nothing to find.

**A1 reaching `e-7` and A2 not is a failure of A2**, not a finding about initialisation. Report both
either way.

**Under noise**: absolute `e-7` is not reachable and demanding it is the wrong test. Report RMS
relative to the measured noise floor, plus the ablation ratio. **The `2.0x` ratio threshold is a
heuristic and must be replaced** by the measured spread of the ratio over repeated noise draws before
it is used as a criterion.

**Read the ablation ratio before the RMS.** A good RMS with a ratio near `1.0` is a negative.

## 11. Read these first

1. `scripts/gantry/BLA-Augmentation/DESIGN.md`, the premise section and D7's amendment. The premise
   is why this work exists; D7's amendment is the mechanism the arms test.
2. `runs/pole_gate.json`, with the scope note in section 4. It settles A2's construction and is the
   most misquotable result in this handoff.
3. `scripts/gantry/BLA-Augmentation/RESULTS.md`, including its "Not measured" table.
4. `EVIDENCE.md` claims 24, 26, 27, 28. The four that Phase B's wiring rests on.
5. `docs/decisions.md` D-153, D-154, D-155, and D-072.

## 12. Do not

* Do not relax `gantry_dynamic/model.py:135` or `:139`. Those two `selection_matrix(PHY_IX, nxd)`
  calls keep the physics and the output blind to `x_a`, which is claim 26 and the precondition for
  parameter interpretability.
* Do not place poles at `157.8937 Hz` as an initialisation. Scoring with the truth transfers to
  Telica because on Telica you do not score; initialising with it does not.
* Do not apply `gamma` to a fitted `B_r`: `0.161` at `rho = 0.987` scales a fitted gain down 6x.
* Do not install `B_r` on `u` alone without resolving `DESIGN.md` D9's open question. It deletes the
  `x_b -> x_a` path that all six of Jan's classes have.
* Do not compare arms at different epoch counts or different `nx_aug`.
* Do not run with `CL_PROBE=0` unless `CL_CONCURRENT=0` is also set.
* Do not overwrite `runs/cl_residual_spectrum.json` without `CL_RS_OUT`.
* **Do not invent an arm.** A0, A1 and A2 are the three. If one is not runnable, write
  "not runnable, <reason>" as its verdict row and move on rather than substituting something else.
* **Do not retry a unit that produced a bad number.** One retry only, and only for an infrastructure
  failure.
* **Do not redefine a boundary after seeing a result.** Section 13's pre-registered verdicts are
  fixed.
* Do not pipe a running job through `grep` or `head`: the pipe buffers and nothing appears until the
  process exits. Read the `.output` file. **One output file per background run** - two runs sharing
  one path produced an internally inconsistent table this session.
* Do not stop to ask. The night is unattended; where this file gives a drop order or a fallback,
  apply it and record it.

## 13. Operational

### Run unattended. 10 hours of script time, 4 epochs maximum.

**Block W first: write everything, run nothing. Untimed, no GPU, about 1 h.** D9's block, the
Telica-portability audit, and the resolution or refusal of both outstanding constants are authored
**before** any timed run starts. Do not fold authoring into a run's timebox: a box that must contain
both writing and running a novel experiment times out, and the two arms are what the night is for.

**Budget in updates, not epochs, and do not trust any wall-clock figure in this file.** The recorded
`~85 min` and `6.5 s/it` are from a different configuration. **Time epoch 1 of the first arm, compute
the wall clock for the full arm, then apply the drop order.** Cap at **4 epochs**; if 4 epochs does
not fit, reduce and record the update count in the run row. `CL_STRIDE` changes updates per epoch by
the same factor, so if it moves the epoch count must move inversely. **Do not compare arms at
different update counts.**

**The budget does not close, and that is planned for.** Six training runs plus six ablations is
roughly 11 h against 10. Five runs is the realistic number.

**Run order, and the drop order applied without asking:**

| # | run | droppable |
|-|-|-|
| 1 | A0 noiseless + ablation | **never.** The restore changes `closed_loop_free_run_rms`, so nothing is comparable to `1.3933793e-06` until this re-establishes it on the restored harness |
| 2 | A2 noiseless + ablation | **never.** The deliverable |
| 3 | A1 noiseless + ablation | **never.** Without it A2 has nothing to be better than |
| 4 | A2 noisy + ablation | drop third; truncate in updates before dropping |
| 5 | A1 noisy + ablation | drop second |
| 6 | A0 noisy | **drop first** |

**Never drop the pre-flight or block W to make room for a run.** The pre-flight is the gate; the
arms only test what it clears.

**Timeboxes.** Pre-flight 30 min. Each training run gets its measured estimate plus 20 %; on
overrun, kill, write the verdict row with what was measured, move on. **One retry only, and only for
an infrastructure failure** (crash, OOM, dead kernel), never for a bad number. Nothing is left
running at the end.

### The overview file, appended after every unit

`tasks/overnight-2026-08-23-verdicts.md`. **Authorised explicitly here despite the no-new-files
rule; create it without asking and create no other new document.** Append after each unit, never
compose at the end, so the file is correct if the session dies at 03:00.

```
BLA e-7 WITHOUT HEURISTICS?  <yes / no, with the number>
RANDOM e-7?                  <yes / no, with the number>
HEURISTICS OUTSTANDING:      <none, or which and why it could not be resolved>
TELICA AUDIT:                <complete / incomplete, and which entry blocks>
NOISE ARMS:                  <which ran, which were dropped and why>
PRE-FLIGHT dL/dp AT STEP 1:  <number>  (zero here means no arm should have run)
```

Then one row per unit: hypothesis, what ran, artefact path, number, verdict, what it eliminated.
**Every unit gets a row whether it succeeds, fails, aborts or is dropped**, reading
"dropped for time, <what ran instead>" where applicable. **An absent row is the only real failure
mode**, because it cannot be told apart from a unit that was forgotten. End with one recommended
next action, not a menu.

### Pre-registered, fixed now, not to be redefined after seeing a result

Write each run row into `docs/gantry-augmentation-problem-log.md` §12 **before** its launch, per
D-090.

* **Stop condition, every arm**: epoch 1 worse than `2.1866011034177349e-06` kills that arm. Write
  the row and move on.
* **A0's verdict**: ablation ratio near `1.0` means `x_a` was dead at the plateau, confirming
  `DESIGN.md` D7's amendment. A high ratio refutes it and means A1 and A2 are testing the wrong
  thing; say so in the overview rather than continuing quietly.
* **A1's verdict**: gradient non-zero at step 1 but RMS not below A0 means the dead zone was real and
  not the limit.
* **A2's verdict**: the bar in section 10. A2 within noise of A1 closes the BLA line.
* **Primary criterion for every arm is the ablation ratio, not the RMS.** A good RMS with a ratio
  near `1.0` is a negative.

### Evidence that does not count

A unit reporting only these is logged inconclusive, per the 2026-08-20 night's findings:
`RMS(x_a)` or any gauge-dependent quantity; parameter-movement counts, since under Adam any non-zero
gradient moves everything by about `lr`; `rho` reported without its gradient; `rho(A_aa) > 0.5`;
gradient coherence; and any oracle- or model-derived threshold.

### Launch

Env `GraduationProject`. Live-output convention per `CLAUDE.md`.

```
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
  -n GraduationProject python -u scripts/gantry/BLA-Augmentation/probe_pole_gate.py
```

Training ~85 min per run locally at 6.5 s/it, ~30 min on `kauai` at 1.55 s/it; ablation ~25 min
without `PROBE_PERPAIR`. The user has approved **9 hours locally**. Checkpoints land in
`C:\Users\20203253\AppData\Local\deepSI\checkpoints\`; `cl_train.py` records the path under
`checkpoint.best`. Server deployment: `runners/DEPLOY-wave1.md`; nothing is pushed, the user copies
by hand.

Use a distinct output path per background run. Two runs sharing one file produced an inconsistent
table this session.

## 14. Delegation

**None for the next action.** Every file is named above and the work is targeted implementation plus
runs. If a source is needed that the repo does not hold, one `deep-research` subagent per decision,
ceiling two concurrent, and never for `EVIDENCE.md` - verifying a quote is the one thing that must be
done by whoever writes the design.
