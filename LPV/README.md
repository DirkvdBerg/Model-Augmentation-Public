# LPV-LFR Files

This folder is organized so the current main derivation stays easy to find.

## Main file

- `LPV/LFR-derivation-supervisor.tex`
  Current supervisor-facing LPV-LFR derivation. This is the primary file to work in.

## Supporting material

- `LPV/supporting/verification/`
  Internal verification document and its generated LaTeX artifacts.
- `LPV/supporting/derivations/`
  Earlier derivation drafts, companion proofs, and older LPV derivation notes.
- `LPV/supporting/supervisor-notes/`
  Drafting aids for the supervisor-facing version, such as the outline, inclusion guide, and prompt.

## Build output convention

The top level of `LPV/` is reserved for `LFR-derivation-supervisor.tex` and its build output.
Supporting documents keep their source files and generated artifacts inside the matching `supporting/` subfolders.
