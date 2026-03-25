# Drenth Verification Notes

This folder groups the Drenth-based working notes for the dual-gantry CT LPV-LFR reformulation.

## Primary Source

- `literature/books/drenth2025_lpv-lfr-thesis.pdf`

## What Is Here

- `docs/drenth/ch2-sec21-source.md`
  Source-only extraction of Drenth Section `2.1`.
- `docs/drenth/ch2-sec211-source.md`
  Source-only extraction of Drenth Section `2.1.1`.
- `docs/drenth/ch2-sec22-source.md`
  Source-only extraction of Drenth Section `2.2`.
- `docs/drenth/ch2-generalized-recipe.md`
  Generalized CT LPV-LFR recipe built from the source-only notes.
- `docs/drenth/ch2-dual-gantry-mapping.md`
  Step-by-step mapping of that recipe onto the dual-gantry model.
- `docs/drenth/claude-handoff.md`
  Handoff note and copy-paste verification prompt for Claude.
- `docs/drenth/claude-paste-prompt.md`
  Direct prompt to paste into Claude for a critical external verification pass.
- `LPV/LFR-derivation-verification.tex`
  Consolidated internal verification document in LaTeX form.

## Recommended Reading Order

1. `docs/drenth/ch2-sec21-source.md`
2. `docs/drenth/ch2-sec211-source.md`
3. `docs/drenth/ch2-sec22-source.md`
4. `docs/drenth/ch2-generalized-recipe.md`
5. `docs/drenth/ch2-dual-gantry-mapping.md`
6. `LPV/LFR-derivation-verification.tex`
7. `docs/drenth/claude-handoff.md` for the full handoff context
8. `docs/drenth/claude-paste-prompt.md` for the direct paste prompt

## Role in the Repo

These notes are the audit trail behind:

- `LPV/LFR-derivation.tex` for the cleaner derivation, and
- `LPV/LFR-derivation-verification.tex` for the single consolidated internal check document.

They are intentionally more explicit than the eventual supervisor-facing version.
