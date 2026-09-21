# Writing

Self-contained LaTeX project for the 12-page IEEE paper. Compile `main.tex`,
nothing else. No `.tex` file here may contain a `..` path: everything LaTeX
needs lives under `Writing/`, so this directory can be zipped and handed to
Overleaf or a supervisor as is.

| Path | Role |
|-|-|
| `main.tex` | class, inputs, title block. Roughly 70 lines, no content. |
| `util/include.tex` | packages only. Swap templates by touching this file. |
| `util/format.tex` | macros, notation, draft switch. Notation matches `state-level-obc-derivation.tex`. |
| `sections/NN_*.tex` | one file per section, flat, numeric prefix. Each header states its page budget and what it owns. |
| `figures/` | generated PDFs, `\graphicspath` points here and only here. Never hand-edited. |
| `tables/` | generated tabular fragments, `\input` from a section. Numbers are never retyped. |
| `refs.bib` | own entries, cite keys matching `docs/references.md`. |
| `IEEEtran.cls`, `IEEEtran.bst`, `IEEEabrv.bib` | from the shipped IEEE template, kept at project root so the build is standalone. |
| `build/` | aux, log and output PDF. |
| `reference/` | the original untouched IEEE template and BST distribution, for lookup only. Not part of the build. |

## Figure and table pipeline

One stem, three folders:

```
Code/Figures/obc_pareto.py   reads   Results/obc_pareto.csv
                             writes  Writing/figures/obc_pareto.pdf
```

Run `make figures` from `Thesis-writeup/` to regenerate, `make paper` to build
the PDF end to end. Scripts import `_fig_common.py` for the IEEE column widths
and body font size, so no figure ever needs rescaling in LaTeX.

## Draft switch

`util/format.tex` sets `\drafttrue`. Flip it to `\draftfalse` for the
submission build and every `\todo{}` and `\note{}` disappears.
