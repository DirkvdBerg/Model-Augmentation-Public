# Guideline: Jan augmentation writeup (self-accountability)

Rules I hold myself to when writing or editing `docs/jan-augmentation-writeup.tex`.
Reread this before every edit pass. Verify against the self-check at the bottom before
telling Dirk it is done. This exists because an earlier draft was rejected as messy and
noisy; these rules remove my discretion so the same failures cannot recur.

## Content: only what was agreed

- Write only content that is in the plain-text spec Dirk approved. No sentence, clause,
  matrix, or parenthetical enters the `.tex` unless it was in that approved spec.
- I do not decide on my own what counts as "noise" and delete it, and I do not add
  explanatory asides on my own judgment. Every content change is proposed in text and
  approved first.
- Banned content (all previously rejected): "verified in MATLAB" style asides;
  implementation detail (RK4 substeps, expansion matrices, row indices, `STIFF_IX`);
  parameter values repeated in prose; mass-conservation commentary; any sentence that is
  not needed to check an equation or a matrix.

## Structure

- Section order: Baseline before System. Simplest and known object first, then its
  extension (baseline plus the added absorber physics).
- No "Final Result" summary section. No Appendix. (Dirk removed both.)

## Format: house style (`LPV/LFR-derivation-supervisor.tex`)

- Preamble taken from the house doc: `\documentclass[11pt]{article}`, `geometry`,
  `fontenc`/`inputenc`/`lmodern`, `amsmath,amssymb,mathtools,bm`, `hyperref[hidelinks]`,
  macros `\R,\adj,\diag`.
- Every displayed equation is numbered with a `\label`, and referenced with `\eqref`.
  No `\[ ... \]`, no `$$`, no starred `equation*`/`align*`.
- Matrices are full and explicit, every entry shown. Shorthand scalar constants
  (`\alpha,\beta,\gamma`) are defined in one compact list under the matrix. Use `\dfrac`
  and `\\[2mm]` row spacing. No `blkdiag`, no block-concatenation of a referenced
  sub-block, no highlighting (`\hl`, `\colorbox`), no text or aside inside the equation.
- No em-dashes anywhere: not the Unicode character, not `---`, not a spaced `--`.
- The baseline stays mechanical: `M(Y)\ddot q + C\dot q + K q = P^\top u`, with `M(Y)^{-1}`
  appearing in `f_base`. Do not show the LFR realization or the rational rewrite here; it
  is only an inversion trick and lives in the companion LFR note.

## Figure

- The block scheme must be legible at the width it is embedded. Strip the equation legend
  out of `jan-blockscheme-v2.tex` (the prose carries those equations) so the boxes fill
  the text width, then embed the trimmed PDF.

## Self-check before declaring done

Reread the whole `.tex` and confirm, item by item, stating the result of each:

1. Every sentence traces to the approved spec. No self-added content.
2. Baseline section precedes System section.
3. Every displayed equation is numbered and has a `\label`.
4. Every matrix is full-explicit: no `blkdiag`, no highlighting, no in-equation text.
5. No em-dashes.
6. No Final Result section, no Appendix.
7. Figure is legible at embed width.
8. Compiles with `pdflatex`, no new warnings.

Only after every item is confirmed do I report the writeup as done, and I state which
items I checked.
