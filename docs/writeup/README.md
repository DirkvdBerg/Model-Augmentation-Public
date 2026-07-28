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

Three figures covering three different questions. They share a notation, so a change to one
is a change to all three.

| Figure | Answers | Notes |
|---|---|---|
| `jan-blockscheme-v4.tex` | What is the model? | Topology: blocks, routing, `h_aug = 0`. Included by the write-up. |
| `training-objective-v1.tex` | What is trained, and how is it validated? | Input, output, loss and the 1/120 train-vs-select horizon ratio, drawn to scale. |
| `coordinates-normalisation-v1.tex` | Where do the numbers live? | Where the normalisation constants come from, where the velocities come from, and the three places data-derived scaling and the model do not line up. |

Earlier block-scheme versions (`jan-blockscheme.tex`, `-v2`, `-v3`) are kept for history.
v4 is the current one; it is the version the write-up includes.

## Shared notation

Set by `jan-blockscheme-v4` and followed by the other two. Do not diverge without changing
all three.

| Symbol | Meaning |
|---|---|
| `x_tilde`, `x_bar` | physical (6) and augmented (2) state partition |
| `x`, `x^phys` | normalised state, SI state |
| `f_base`, `h_base` | baseline state and output blocks |
| `phi_aug`, `S` | augmentation ANN and its router |
| `f_aug`, `g_aug` | router output into the physical rows, into the augmented row |
| `psi` | encoder |
| `A_n, B_n, C_n, D_n` | normalised matrices (subscript, not a bar: the bar is taken by `x_bar`) |

## Not here

`docs/coulomb-friction-formulation.tex` and `docs/Gantry-Augmentation-Formula-Derivation.tex`
are still in `docs/`. The first is a sibling formulation note, the second is a stale
code-annotated derivation superseded by `jan-augmentation-writeup.tex`. Move or retire them
deliberately, not by accident.
