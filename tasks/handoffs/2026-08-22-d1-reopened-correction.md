# Mid-session correction: D1 reopened

CORRECTION, mid-session. D1's DECISION is reopened. Its MEASUREMENT stands.

Stands, do not re-derive: r = S*rho, suppression 10x to 642x across records,
r is loop-limited not disturbance-limited, controller replay clean except on
APRBS records at ~1e-3. Those numbers are correct and they do invalidate the
pre-2026-08-22 BLA results.

Does NOT stand: the inference that the fix is an OPEN-LOOP residual. The
gantry is closed-loop operated and open-loop unstable (rigid-body double
integrators). Simulating P0 open loop over a 12 s record is a condition the
machine is never in. The four things the previous session then patched --
unbounded drift, 8.5 nm of noise becoming 3-33 mm, needing (1-q^-1)^2
differencing at a derived 8.7 dB cost at the mode, and x0 not existing as a
true state on Telica -- are not four problems. They are one construction
reporting that it does not apply. S = (I + P0 C)^-1 is exactly known, since
P0 and C_fb both are, so the response to "S filters the residual and adds
its own poles" is to account for S, not to leave the loop.

Where to go instead, already cited in EVIDENCE.md and never used: claim 6,
pintelon2020bla_feedback, G_BLA = E{Y R*}/E{U R*}, a ratio of projections
onto the REFERENCE, defined for a system operating in feedback, with no
open-loop simulation in it. Claim 11 is why the reference is the right
regressor. literature/closed-loop-id/ holds 28 further papers that this
project has not opened.

Second correction, an ordering not a caveat: on the real system the baseline
parameters are ESTIMATES, so the residual is missing dynamics + parameter
mismatch + noise. Augmenting on top of an unfitted baseline lets the
augmentation absorb baseline error, which is the negation failure mode and
the thing the orthogonal-projection contribution exists to prevent. FIT THE
BASELINE, THEN AUGMENT. This does not affect the simulation arms, where
deriv6 is exact. It does affect every transferability claim.

WHAT DOES NOT CHANGE: continue. The two arms, the e-7-without-heuristics
bar, the pre-flight gate, the 10 h / 4 epoch budget, the drop order and the
verdict file are all unaffected. A2's A_r still comes from the open-loop
fit, and the pole gate says that recovers the mode to 0.12 % at
Telica-magnitude noise, so it has something installable and should run. But
label its provenance honestly in D9 and RESULTS.md: A2 is a simulation
result whose residual construction is contested, and it is NOT evidence
that the method transfers. Do not write a Telica portability argument on
top of it.

DESIGN.md D1 and the handoff have been amended with both corrections; the
status table now reads "measurement stands, decision REOPENED".

---

When this file is read: apply the correction above and continue whatever
work was in progress before this note, incorporating it.
