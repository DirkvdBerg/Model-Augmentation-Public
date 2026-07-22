# Opening prompt for the ARTBP implementation session

Paste the block below into a fresh session.

---

We are implementing and verifying ARTBP for the gantry augmentation. This is a fresh, focused
session. Do not reopen prior investigations.

READ FIRST, in this order (CLAUDE.md already loaded; it points you to tasks/lessons.md — read that too):
1. `tasks/lessons.md` — active project rules (live-output, seed-must-reach-build,
   instrument-at-the-called-method, verify-before-concluding, no-scaffolding). These are constraints.
2. `tasks/handoff.md` — the CURRENT ACTIVE THREAD block at the top is this task's state.
3. `scripts/gantry/ARTBP/README.md` — the strategy: goal, paper formula-reuse table, estimator
   design, pre-registered bias/variance verification plan, and the 5 phases. This is your plan.
4. `literature/stability-training/Unbiasing Truncated Backpropagation Through Time.pdf` — the source
   (Tallec & Ollivier 2017). Equation numbers in the README refer to it.
5. `scripts/gantry/gantry-zero-mean/v12_artbp.py` (the diagnostic you are replacing) and
   `scripts/gantry/drift-demo/demo_common.py` (`build_pipeline`, which you reuse).
6. The v12 row in `docs/gantry-augmentation-problem-log.md` §12 (the diagnostic result + numbers).

STATE (do not re-derive or re-open):
- The ANN's DC-drift on the K=0 dY row is CLOSED as a cause: it is the truncated-BPTT bias, proven
  by intervention (v12: fixed nf=400 locks dY DC to -4.5e-6 sign-locked; ARTBP at matched cost
  collapses it into the +/-3e-7 noise band, scattered sign). Offset/physics/normalization are
  refuted. Do NOT reinvestigate them.
- Goal: implement ARTBP properly and VERIFY it (bias + variance vs a ground-truth gradient), then
  compare truncation distributions. Our v12 used a geometric (constant-c) distribution = the
  high-variance case the paper warns against; Eq. 14 (poly-tail) is the target improvement. Our
  loss-term reweighting is unbiased for our independent-window setting (README §2).

FIRST TASK — Phase B (ground truth + biased reference), per the README:
- `true_grad(T)`: exact full-BPTT DC-direction gradient over T steps (no truncation). Pick the
  largest T where full BPTT is feasible on our model (do the memory/compute check); T = the ARTBP cap.
- `fixed_grad(nf)`: the standard truncated-window DC-direction gradient (the biased baseline).
- A figure: `fixed_grad(nf)` vs `true_grad(T)` across nf, showing the ~1/nf bias gap.
Definition of done for Phase B: both functions work, the figure shows the bias, and T is fixed.

GUARDRAILS:
- Follow the phases in order. Do NOT skip to training (Phase E) before the Phase D bias/variance
  verification passes. Keep the pre-registered success criteria in the README; write results back to it.
- As your first sanity check, reproduce the v12 fixed control (dY DC ~ -4.5e-6 sign-locked) so you
  know your harness is correct before trusting any new estimator.
- Require K >= 2 in any truncated rollout (a length-1 window has only the detached encoder-init
  output, so no gradient — this crashed v12). State order [X,Theta,Y,dX,dTheta,dY,delta_a,vdelta_a],
  dY = index 5.
- Runs > a few seconds: launch with the live-output convention (unbuffered background) and read the
  .output; `tee` masks Python exit codes, so grep logs for tracebacks rather than trusting "exit 0".
- Reuse `demo_common.build_pipeline`. Do NOT touch `kamtin-fp-model/`. Label every numerical
  formula/threshold with `# THEORY: <paper eq.>` or `# HEURISTIC: <reason>`. No em-dashes.
- Plan mode for the build (3+ steps); check in before large runs; log non-trivial choices to
  `docs/decisions.md` and give each training run a D-090 row before launch.

Start by reading the files above, then propose the Phase B implementation (do not write code until
you have read the README and the paper and reproduced the fixed-control sanity check plan).
