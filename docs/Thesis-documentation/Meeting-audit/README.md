# Meeting audit

This folder is a standalone evidence audit of the TU/e and ASMPT meeting material in
`docs/Thesis-documentation`. It is deliberately **not** part of the thesis writing guide. Its job is
to preserve the development history, recover decisions and warnings that may otherwise be missed,
and point a later writing session to evidence that still needs to be checked.

## What was scanned

- `Meetings ASMPT`: 80 files (37 PowerPoint decks, 40 text notes, 3 Markdown notes).
- `Meetings Tue`: 74 files (34 PowerPoint decks, 37 text notes, 3 PDFs).
- Date span: February through September 2026.
- All text and Markdown notes were read. Text, speaker notes, image counts and chart labels were
  extracted from every PowerPoint deck. The three PDFs were converted to text and read.
- Exact duplicate content was detected before synthesis. Duplicate copies were not treated as
  independent confirmation.

The reusable PowerPoint extractor is
`scripts/thesis/extract_pptx_text.ps1`. It keeps slide boundaries and speaker notes, and reports
image/chart presence so visually important slides can be revisited in PowerPoint.

## How to use this audit

For each thesis section:

1. Read the relevant row in `candidate-thesis-impact.md`.
2. Read the corresponding theme in `thematic-thesis-audit.md`.
3. Use `meeting-context-ledger.md` only when the chronological context or the origin of a decision
   matters.
4. Check the claim against `docs/decisions.md` and the current code before writing it as present
   tense. Meeting material is a discovery source, not the final authority.
5. Check any literature claim against the actual paper. A paper name in a slide is not a verified
   reference and must not be added to the bibliography on that basis alone.

## Evidence hierarchy

Use the following order when sources conflict:

1. current code and run configuration;
2. current generated data/results and their provenance;
3. `docs/decisions.md`, especially later decisions that explicitly supersede earlier ones;
4. dated supervisor or company meeting notes;
5. dated presentation content;
6. undated notes, copied prompts and exploratory assistant-generated analysis.

The lower levels remain useful for finding questions and forgotten context, but they do not
establish what the final pipeline does.

## Status vocabulary

- **Current**: corroborated by current code or a later decision.
- **Historical**: true of an earlier project phase, useful only as development context.
- **Superseded**: explicitly replaced or contradicted by a later decision or implementation.
- **Proposed**: discussed but not established as implemented.
- **Needs verification**: potentially important, but the meeting record is not enough to support a
  thesis claim.

## Known duplicate or copied material

- `27-07-2026-meeting-Jan-and-ASMPT.txt` occurs in both meeting folders.
- `different Y positions.txt` is identical to
  `13-05-2026-joint-meeting-multisine-problem.txt`.
- `26-02-2026.txt` is identical to `OMNIPUS separate topic.txt`.
- `30-03-2026-meeting-roland-no-algebraic-loop-and-solution-with-lfr-structure-intact.txt` is
  identical to `could just calculate the invertible.txt`.
- Several presentation templates and explicit `Copy`/`orig` files occur. They were treated as
  copies unless their extracted content differed.

## Important reading warning

Some later decks contain pasted exploratory AI analyses alongside actual meeting feedback. This
audit treats such passages as hypotheses unless the same point is supported by a dated human note,
a decision record, current code, or a reproducible result. This is especially important for claims
about window-length theory, optimizer behaviour, orthogonality guarantees and causal explanations
of training failure.

## Boundaries of this pass

- No thesis prose or writing-guide section was changed as part of this meeting audit.
- The research-plan document is intentionally not merged here. It deserves a separate, line-by-line
  integration pass after this audit, as requested.
- No bibliography entry is accepted or rejected here. The audit only flags where a verified source
  will later be needed.

