# Claude Handoff for Drenth-Based Verification

## Purpose

This note is a handoff for an external reviewer such as Claude. Its purpose is
to preserve the context behind the Drenth-based verification workflow and to
provide a ready-to-use prompt for checking the dual-gantry CT LPV-LFR
derivation against Drenth's **thesis**.

## What Was Built

We built a layered verification trail for the dual-gantry CT LPV-LFR
reformulation. The work was intentionally split so that the source extraction
 from Drenth is separated from the dual-gantry-specific algebra.

The key files are:

- `docs/drenth/ch2-sec21-source.md`
  Source-only extraction of Drenth thesis Section `2.1`.
- `docs/drenth/ch2-sec211-source.md`
  Source-only extraction of Drenth thesis Section `2.1.1`.
- `docs/drenth/ch2-sec22-source.md`
  Source-only extraction of Drenth thesis Section `2.2`.
- `docs/drenth/ch2-generalized-recipe.md`
  Generalized CT LPV-LFR recipe built from the source-only notes.
- `docs/drenth/ch2-dual-gantry-mapping.md`
  Step-by-step mapping of that recipe onto the dual-gantry model.
- `LPV/supporting/verification/LFR-derivation-verification.tex`
  Consolidated internal verification document.
- `LPV/LFR-derivation-supervisor.tex`
  Current supervisor-facing derivation document.
- `LPV/supporting/derivations/LFR-derivation.tex`
  Cleaner derivation document.
- `LPV/supporting/derivations/M-invertibility.tex`
  Plant-specific invertibility proof used for well-posedness.

## Important Source Boundary

The primary source for the generic LPV-LFR framework is:

- `literature/books/drenth2025_lpv-lfr-thesis.pdf`

The IFAC paper is **not** the primary source for the continuous-time
interconnection. It is treated as supporting discrete-time context only.

This means the reviewer should check the derivation primarily against
Drenth's **thesis**, especially:

- Section `2.1`
- Section `2.1.1`
- Section `2.2`

## Most Important Interpretation Rule

The verification work deliberately separates three classes of statements:

- `Direct from Drenth`
- `Generalized from Drenth`
- `Own dual-gantry derivation`

This distinction matters a lot.

Drenth provides:

- the generic CT LPV-LFR framework,
- the affine-vs-rational trade-off discussion,
- the generic sufficient well-posedness route.

Drenth does **not** directly provide the dual-gantry realization. In
particular, the following are dual-gantry-specific derivation steps:

- decomposition `M(Y) = M_0 + Y M_1 + Y^2 M_2`,
- latent-variable choice `v`, `v_1`, `v_2`,
- signal choice `z = [v; v_1]`, `w = [v_1; v_2]`,
- choice `\Delta(Y) = Y I_6`,
- explicit formulas for `A`, `B_w`, `B_u`, `C_z`, `D_{zw}`, `D_{zu}`,
- reduction of the algebraic loop to `M(Y) v = f_gen`.

Those steps should be checked for algebraic correctness and for consistency
with Drenth's framework, but they should **not** be treated as if Drenth
already derived them.

## Well-Posedness Split

The verification chain also distinguishes two well-posedness routes:

1. Drenth's generic sufficient route from Section `2.2`
2. the sharper plant-specific route based on:
   - reduction of the loop to `M(Y) v = f_gen`
   - invertibility proved in `LPV/supporting/derivations/M-invertibility.tex`

The plant-specific route is the stronger and more relevant proof for the
current fixed baseline model.

## What Claude Should Verify

The review should focus on the following questions:

1. Is the generic CT LPV-LFR framework stated in a way that is faithful to
   Drenth's thesis?
2. Are the claims marked as `Direct from Drenth` really directly supported by
   the thesis text/equations?
3. Are the claims marked as `Generalized from Drenth` reasonable and not too
   strong?
4. Are the dual-gantry-specific derivation steps algebraically correct?
5. Is the well-posedness discussion careful enough about the distinction
   between Drenth's generic theorem and the sharper plant-specific proof?
6. Does any line in the consolidated verification `.tex` overclaim what
   Drenth actually provides?

## Which File to Give Claude

### Minimal input

If you want the smallest useful bundle, give Claude:

- `LPV/supporting/verification/LFR-derivation-verification.tex`
- `literature/books/drenth2025_lpv-lfr-thesis.pdf`

This is usually enough for a direct verification pass, because the `.tex`
already contains the source boundaries and the `Direct / Generalized / Own`
labels.

### Better input

If you want a more reliable audit, give Claude:

- `LPV/supporting/verification/LFR-derivation-verification.tex`
- `docs/drenth/ch2-sec21-source.md`
- `docs/drenth/ch2-sec211-source.md`
- `docs/drenth/ch2-sec22-source.md`
- `docs/drenth/ch2-dual-gantry-mapping.md`
- `literature/books/drenth2025_lpv-lfr-thesis.pdf`

This makes it easier for Claude to verify not just the final wording, but also
the internal reasoning chain.

## Copy-Paste Prompt for Claude

Use the following prompt directly if desired.

```text
Please verify the dual-gantry CT LPV-LFR derivation against Drenth's THESIS, not primarily against the IFAC paper.

Primary source to use:
- literature/books/drenth2025_lpv-lfr-thesis.pdf

Files to review:
- LPV/supporting/verification/LFR-derivation-verification.tex
- docs/drenth/ch2-sec21-source.md
- docs/drenth/ch2-sec211-source.md
- docs/drenth/ch2-sec22-source.md
- docs/drenth/ch2-dual-gantry-mapping.md
- LPV/supporting/derivations/M-invertibility.tex

Review goal:
I want a careful verification-oriented review of whether the consolidated verification document is faithful to Drenth's thesis and whether the dual-gantry-specific derivation steps are correct.

Important context:
- Drenth's thesis is the primary continuous-time LPV-LFR source.
- The IFAC paper is only supporting discrete-time context.
- The document intentionally separates:
  - Direct from Drenth
  - Generalized from Drenth
  - Own dual-gantry derivation
- The dual-gantry realization is not claimed to come directly from Drenth.

What I want you to do:
1. Read LPV/supporting/verification/LFR-derivation-verification.tex as the main document.
2. Check the cited Drenth-based claims against the thesis, especially Sections 2.1, 2.1.1, and 2.2.
3. Verify whether the lines labeled or described as “Direct from Drenth” are really directly supported.
4. Verify whether the “Generalized from Drenth” steps are reasonable and not overstated.
5. Check the dual-gantry-specific derivation for algebraic correctness, especially:
   - M(Y) = M_0 + Y M_1 + Y^2 M_2
   - the choice of latent variables v, v_1, v_2
   - z = [v; v_1], w = [v_1; v_2]
   - Delta(Y) = Y I_6
   - the derivation of A, B_w, B_u, C_z, D_zw, D_zu
   - the reduction of the loop to M(Y) v = f_gen
6. Check the well-posedness discussion carefully and distinguish between:
   - Drenth's generic sufficient Section 2.2 route
   - the sharper plant-specific M(Y)-invertibility route
7. Flag any line that overclaims what Drenth actually proves or states.
8. Flag any line that is ambiguous, insufficiently sourced, or mathematically weak.
9. Give the result as a structured review with:
   - Confirmed statements
   - Potential overclaims
   - Algebraic issues or points needing checking
   - Suggested wording corrections

Please be critical and precise. I do not want a friendly summary only. I want a real verification pass.
```

## Recommended Use

For the strongest result, ask Claude to review `LPV/supporting/verification/LFR-derivation-verification.tex`
first, then use the source-only notes only when he needs to check how a given
claim was constructed.
