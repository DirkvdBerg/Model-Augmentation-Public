# Write-up and figures

The supervisor-facing write-up of the method and every figure that belongs to it.
Everything here compiles with `pdflatex <name>.tex` run from this directory.

## Style

| File | Role |
|---|---|
| [figure-style.md](figure-style.md) | House figure style. Read before drawing anything new. |
| [jan-writeup-guideline.md](jan-writeup-guideline.md) | Rules for writing and editing the write-up itself. |

## Write-up

| File | Role |
|---|---|
| `jan-augmentation-writeup.tex` | The write-up: what the system, baseline and augmentation are, with equations. |
| [jan-formula-writeup-outline.md](jan-formula-writeup-outline.md) | Agreed outline and section status. |

## Figures

Five figures covering five different questions. They share a notation, so a change to one
is a change to all five.

| Figure | Answers | Notes |
|---|---|---|
| `jan-blockscheme-v4.tex` | What is the model? | Topology: blocks, routing, `h_aug = 0`. Included by the write-up. |
| `training-objective-v1.tex` | What is trained, and against what? | Input, output, loss, and the 1/120 train-vs-select horizon ratio drawn to scale. Covers checkpoint **selection** only, not the validation strategy: see the gap below. |
| `coordinates-normalisation-v1.tex` | Where do the numbers live? | Where the normalisation constants come from, where the velocities come from, and the three places data-derived scaling and the model do not line up. |
| `closed-loop-form-v4.tex` | Why can we drive the model with `u_data + Cfb(y_data - y_model)`? | **Current.** Three rows: the machine loop, the same drawing with the model in place of the plant, and the form we run. Each row carries the equation it is a picture of. Answers the ASMPT supervisors' question about the subtraction. |

**Caption for `closed-loop-form-v4`**: *The closed-loop training form in three lines. Rows 1 and 2
are the same drawing with the block swapped: the machine's controller reacted to `r - y` and that
reaction is inside the recorded `u`, while the simulated controller must react to `r - y_hat`. The
two errors differ by `y - y_hat`, so linearity of `Cfb` alone turns row 2 into row 3. The
correction is the force the controller would have added because of the model's error, and it
vanishes for a perfect model. `Cfb` is frozen at the record's `Y_op` and is therefore exogenous.
Schematic, not data.*

### Earlier versions, kept for history

v4 is the current one. The three before it are kept because they record three different failed
attempts at the same communication problem, and the reasons are worth not repeating:

- **v1**, two panels with the identity as an equation between them. It *asserted* the equality in
  a label. Also the only version carrying the `xc = 0` annotation (D-142), which v2 to v4 drop as
  a separate claim that dilutes this one.
- **v2**, four panels drawing the derivation as a rewiring, with the plant/model position as one
  opaque block `F`. Better, but it keeps the plant on the page in every panel, which is
  self-defeating for a figure whose message is that the plant is not involved.
- **v3**, a superposition table with no plant and no loop at all. Logically the cleanest, but too
  abstract on its own: it never shows the reader what a "loop" is here.

**The lesson v4 encodes:** let the equations carry the proof and let the pictures only say what
each equation is a picture of. v1 to v3 all failed by asking a block diagram to prove an equality,
which it cannot do. A block diagram shows topology; this argument is arithmetic on signals, and in
a block diagram a signal is a line, which has no identity.

Earlier block-scheme versions (`jan-blockscheme.tex`, `-v2`, `-v3`) are kept for history.
v4 is the current one; it is the version the write-up includes.

### Known gap: validation

Of the four questions a supervisor asks (input, output, loss, validation), the first three are
covered. **Validation is not.** `training-objective-v1` panel (a) shows how the checkpoint is
*selected*, which is one part of it. Nothing in any figure shows the held-out test records
E1 to E4, the held-out Y positions, the encoder-init versus true-x0 baselines, the FP+MSD
oracle, or NRMS. That is a fourth figure and it does not exist yet (D-129, amendment b).

## Shared notation

Set by `jan-blockscheme-v4` and followed by the other three. Do not diverge without changing
all four.

| Symbol | Meaning |
|---|---|
| `x_tilde`, `x_bar` | physical (6) and augmented (2) state partition |
| `x`, `x^phys` | normalised state, SI state |
| `f_base`, `h_base` | baseline state and output blocks |
| `phi_aug`, `S` | augmentation ANN and its router |
| `f_aug`, `g_aug` | router output into the physical rows, into the augmented row |
| `psi` | encoder |
| `A_n, B_n, C_n, D_n` | normalised matrices (subscript, not a bar: the bar is taken by `x_bar`) |

Closed-loop symbols, added by `closed-loop-form-v1`:

| Symbol | Meaning |
|---|---|
| `F` | the plant/model position drawn as ONE opaque block (`closed-loop-form-v2` only): the plant in panel (a), the augmented model elsewhere. Its whole job is to be a block the derivation never opens, so do not gloss it with internals |
| `r_k` | reference. Recorded, but not read by the training path |
| `u^ff_k` | feedforward. Cancels in the subtraction, so also not read. **Not** `f_k`: `f` is taken by `f_base` and `f_aug` |
| `u_k`, `y_k` | the recorded plant input and output, i.e. what the loader returns |
| `y_hat_k` | model output, as in `jan-blockscheme-v4` |
| `u_hat_k` | the input applied to the MODEL, `u_k + u^fb_k` |
| `C_fb` | the feedback controller as an operator. Distinct from `C_n`, the normalised output matrix: subscript `fb` versus `n` is the only thing separating them, so never write a bare `C` |
| `x^c`, `u^fb_k`, `e_k` | controller state, controller output, controller error `y_k - y_hat_k` |
| `tau` | training-window start index, the instant at which the encoder sets `x` and `x^c = 0` |

## Not here

`docs/coulomb-friction-formulation.tex` and `docs/Gantry-Augmentation-Formula-Derivation.tex`
are still in `docs/`. The first is a sibling formulation note, the second is a stale
code-annotated derivation superseded by `jan-augmentation-writeup.tex`. Move or retire them
deliberately, not by accident.
