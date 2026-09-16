# What remains, and how to run it

This investigation ran inside the handoff's four-hour budget on 15 September 2026. The
stages below are the ones it did not finish, with the exact command or design for each.
Nothing here is a blocker for the recommendations in `decision-report.md`; each item
either strengthens an existing conclusion or opens a question the recommendations already
treat as open.

## Not run

### 1. Multi-start parameter recovery, reduced versus raw coordinates

Designed in `experiment-plan.md` Stage 2 test 7 and not run. The structural part is
settled (rank 10, four exact null directions, exact invariance), so what this would add
is the *practical* behaviour of an optimizer in the two coordinate systems: whether the
raw split wanders, how far, and whether the ten combinations are recovered equally well
from both. Build it as a variant of `s3c_competition.py` with the learning component
switched off and several random starts inside the admissible box.

Expected cost: about 15 minutes for 3 starts times 2 coordinate systems at the sizes used
in `s3c_competition.py`.

### 2. Closed-loop versus open-loop recovery on a correctly specified synthetic plant

Handoff Stage 5, second half. Not run. The audit already establishes the *structural*
point, correction C3 in `gyorok-assumption-audit.md`: Theorem 12 assumes the input is
independent of the noise, which closed-loop data violate, so the consistency chain is
unavailable regardless of what a simulation shows. What the experiment would add is the
size of the effect, which is a different and useful question.

Design, if run: reuse `scripts/gantry/gantry_dynamic/controller.py`'s per-operating-point
controller so the loop is the one the project actually uses, generate noiseless records
first and then records with output noise at a stated SNR, and run the same one-step
estimator open loop and closed loop with matched excitation. Report the ten combinations,
not the raw fourteen. Three seeds expose failure; they cannot prove consistency, and the
report must say so.

### 3. Assumption 8 of G26, `‖Phi theta_b*‖ > ‖Delta‖`, on real records

Row 8 of the audit is marked unknown and is cheap to settle: stack the baseline write and
the truth-minus-baseline write over the training records and compare their norms. It
needs the dataset, so it was left out of a data-free investigation.

### 4. MATLAB independent check

`matlab.exe` R2025a and R2021b are installed and were confirmed present, but no MATLAB
job was run in this investigation. Every symbolic result here was obtained with sympy
1.14.0 in Python, and every numerical result with torch 2.5.1 in float64. The existing
MATLAB symbolic probe `code/gantry_euler_regressor_structure.m` covers the Euler-fidelity
structural claims and its stored log was used as a cross-check for the null-space result,
which this investigation independently reproduced at full RK4 fidelity in Python
(agreement to below `4e-08 rad` on the principal angles). So the structural claims do have two
independent implementations behind them; what is missing is a MATLAB check of the
*projection* prototype, which has only the Python implementation. That is stated as a
limitation rather than papered over.

To add it: a Symbolic Math Toolbox script mirroring `s3_projection_prototype.py` T1 to T3
on the same surrogate the existing `surrogate_obc_symbolic.m` uses, comparing the
pseudo-inverse convention and the derivative identity in exact arithmetic.

### 5. Free-run and multiple-shooting behaviour of the construction

Everything in this investigation is at the one-step level, which is the level the
orthogonality condition is stated at. The production estimator is free-run with multiple
shooting. The gap between the two is named in row 17 of the audit and in OBC-1, and it is
the reason no horizon-level claim is made anywhere here. Closing it is a training-campaign
question, explicitly out of scope for this handoff.

## Corrections this investigation had to make to its own earlier steps

Recorded so the same mistakes are not repeated from the logs.

1. **`s3_projection_prototype.py` T1 used a metric that inverts the conclusion.**
   `‖Phi^T G̃‖` is dominated by the offset column, whose singular value is about a
   thousand times the largest parameter column (`s3b` section A), so the raw residual
   reported a 5x *degradation* off the reference set while the scale-free quantity
   `rho = ‖P G‖ / ‖G‖` reports a genuine 34 percent *improvement*. Both numbers are in
   the logs. Any implementation must report `rho`, not `‖Phi^T G̃‖`.

2. **`s4_latent_and_encoder.py` T4 ran on a 10 ms window.** The plant's yaw period is
   197 ms and its two integrator axes have viscous timescales of 1546 ms and 1010 ms,
   all computed in `s4b_encoder_window.py` from the governing matrices. The 95 to 99.9
   percent encoder-parameter overlap that T4 reported is therefore a property of the
   window, not of the encoder, and `s4b` supersedes it.

3. **The "exact dependency" between the parameter span and the initial-state span was a
   rank-threshold artifact.** `s4` reported joint rank 15 of 16 independent directions at
   a relative tolerance of `1e-10`; at `1e-12` the rank is 16 and there is no dependency.
   The joint matrix has a condition number near `1.9e9`, so a single threshold cannot
   settle a rank question on it. No exact dependency is claimed.

4. **The null-space decomposition block at the end of `s4b_encoder_window.py` is wrong
   and must not be quoted.** It prints "of the 4 null directions, 0 are the known
   parameter null directions and 4 involve the initial state", and then prints four
   vectors whose initial-state parts are all below 0.004 against a parameter part
   normalized to 1.0, that is, numerically zero. The bug is the test: it thresholds the
   singular values of a residual matrix against that residual's own largest singular
   value, which cannot distinguish "all residuals are tiny" from "all residuals are
   large". The correct statement, which the ranks give directly and unambiguously, is in
   `decision-report.md` Priority 4: `rank(dy/dtheta) = 10`, `rank(dy/dx0) = 6` and
   `rank(joint) = 16` at every window length, so the two spans intersect trivially and
   the joint null space is exactly the four known parameter null directions. To fix the
   block, replace the residual threshold with principal angles between `span(known)` and
   `span(N_joint)`.

5. **`torch.autograd.functional.jacobian` in reverse mode was the wrong tool** for a
   Jacobian with 14 or 20 columns and thousands of rows. Switching to
   `torch.func.jacfwd` / `jacrev` as appropriate turned jobs that had not finished in 15
   minutes into jobs of well under a minute. Recorded because the same trap will appear
   in any implementation that builds `Phi` on a long record.
