# Session Handoff

Written by the finishing agent when context is running low or work is paused.
The receiving agent reads this as step 3 of their session start sequence.

---

**Written by**: Claude (Sonnet 4.6)
**Date**: 2026-03-24
**Handed to**: Next Claude session or Codex

## Session Goal

Write `LPV/supporting/derivations/LFR-derivation.tex` -- a standalone LaTeX document deriving the LPV-LFR
structure of the gantry FP model from the MATLAB CT state-space, suitable for
TU/e supervisor review.

## Status

**`LPV/supporting/derivations/LFR-derivation.tex` is written and structurally complete.**
Citation strategy is resolved (see below). The document needs one revision pass
to update citations and verify the Section 2 LFR definition against Tóth Ch. 3.

Important caveat: the derivation is currently ahead of the implementation plan.
The algebraic existence of a valid LPV-LFR realization is now much clearer than
the exact code-level boundary in the repo. The remaining uncertainty is no
longer "can we derive an LFR?" but "how should that LFR be carried into the
stage-coordinate RK4 baseline used by the codebase?"

## What Was Done

### LFR-derivation.tex written

File: `LPV/supporting/derivations/LFR-derivation.tex`

Seven sections:
1. CT state-space from MATLAB (`getss.m`), with stage coordinate remark
2. LFR definition -- {G, Δ(p)} notation from Drenth IFAC eq. (6)--(7)
3. M(Y) = M0 + Y*M1 + Y^2*M2 decomposition (explicit entry-level)
4. Latent variable construction: v, v1=Yv, v2=Y^2v; z=[v;v1], w=[v1;v2]=Yz, Δ(Y)=YI6
5. Constant G matrix -- derived subsection by subsection (state, loop, output)
6. Algebraic verification -- 3-step: solve loop (M(Y)v=f_gen), recover ẋ, check output
7. Well-posedness -- loop reduces to M(Y)v=f_gen by construction of Dzw; cite `LPV/supporting/derivations/M-invertibility.tex`

The verified G matrix:
- A   = [0, I3; -M0^{-1}K, -M0^{-1}C]
- Bw  = [0, 0; -M0^{-1}M1, -M0^{-1}M2]
- Bu  = [0; M0^{-1}P]
- Cz  = [M0^{-1}[-K,-C]; 0]
- Dzw = [-M0^{-1}M1, -M0^{-1}M2; I3, 0]
- Dzu = [M0^{-1}P; 0]
- Cy  = [I3, 0], Dyw = 0, Dyu = 0

### Citation research -- all sources evaluated

Extensive source review was conducted this session. Conclusions:

**Use:**
- `drenth2025rational` (Drenth IFAC) -- for the {G, Δ(p)} definition (eq. 6-7)
  and well-posedness condition. This is the right source. Cite in Section 2.
- `garcia2013model` -- for EOM and M(Y) structure. Already in Section 1 and 3.
- `LPV/supporting/derivations/M-invertibility.tex` -- for well-posedness (positive definiteness of M(Y)). Section 7.

**Do NOT use (with reasons):**
- `toth2010modeling` Ch. 7 ("LPV Modeling of Physical Systems"):
  Produces LPV-KR (kernel) representations via a decision-tree algorithm,
  NOT LFR. The LFR is mentioned in one paragraph (Sec. 7.3.5) as a different
  approach using external tools. The notation (behavioral framework) is
  incompatible with {G, Δ(p)}. Do not cite for LFR derivation.
- `toth2010modeling` Ch. 3 ("LPV Systems and Representations"):
  NOT yet checked -- Section 3.1.2 "Representations of CT LPV Systems" (p. 49)
  may or may not contain the LFR/LFT form. Worth one read to confirm before
  deciding whether to cite for the definition.
- Alkhoury, Petreczky, Mercère (2016) "Structural properties of LPV to LFR
  transformation" -- newly added to `literature/lpv-lfr/`. Covers ALPV→LFR
  for **affine** models only (A(p) = A0 + sum Ai*pi). Our model has rational
  dependence (M(Y)^{-1}), so this paper does not apply. Also discrete-time.
  Also about structural properties (minimality, identifiability), not
  construction.
- `drenth2025lpvlfr` (Drenth thesis) Ch. 2 assumptions -- Assumption 2.2
  (unit ball scheduling), Dzw = e^{-N} parameterization: these are
  learning-specific. Do not state in baseline realization document.

**Not available (but would be ideal):**
- Zhou, Doyle & Glover (1996) "Robust and Optimal Control" -- cited by Drenth
  IFAC for SVD realization of affine→LFR. Classical LFT source. We only have
  the title page (incomplete upload).

**Bottom line on methodology citation:**
No paper covers "construct LFR from rational first-principles model by latent
variable introduction." This is direct algebra. No methodology citation needed.
Drenth IFAC covers the definition; the derivation stands on its own.

### New file added to literature

`literature/lpv-lfr/Structural properties of LPV to LFR transformation.pdf`
(Alkhoury et al. 2016). NOT added to `docs/references.md` yet -- not needed
for the document.

## Files Modified This Session

- `LPV/supporting/derivations/LFR-derivation.tex` -- created (new file)

## Decisions Made

No new decisions logged to `docs/decisions.md`. The citation strategy above
is the relevant decision for this document.

## Critical Reading Notes From Follow-up Review

These are not decisions yet. They are the places where the repository can be
read as more settled than it really is.

### 1. Coordinate-system boundary is still an implementation decision

The derivation in `LPV/supporting/derivations/LFR-derivation.tex` is written in logical coordinates
`[X, Theta, Y]`, which is the cleanest match to the MATLAB equations.
The implementation path in the repository is stage-coordinate centric by D-006:
states, inputs, outputs, and data all live in `[X1, X2, Y]`.

This is not a contradiction, but it is still a design checkpoint:
- Option A: keep the baseline LFR internally in logical coordinates and only
  transform the output to stage coordinates
- Option B: derive in logical coordinates, then similarity-transform the full
  realized baseline to stage coordinates before implementation
- Option C: redo the derivation directly in stage coordinates

Recommended reading of the current state:
- the derivation document is fine in logical coordinates for explanation
- the implementation will probably want Option B, because it preserves the
  clean derivation while keeping runtime states aligned with measured data
- this has not yet been written down as an explicit project decision

### 2. The derivation proves existence of one valid LFR, not uniqueness or minimality

The latent-variable construction using `v`, `Yv`, and `Y^2v` gives a valid
constant `G` with `Delta(Y) = Y I_6`. That is a strong result and likely enough
for implementation.

What it does not prove:
- that this realization is minimal
- that it matches a canonical textbook LFT realization
- that the chosen repetition count is the smallest possible one

This is not a blocker unless the supervisor explicitly asks for minimality or a
canonical realization. The safe claim is: "we have an explicit valid
realization", not "we have the unique or best realization".

### 3. The repo should distinguish theoretical LFR compatibility from runtime implementation form

Current documents can be read as if "baseline uses LFR" fully determines the
runtime code structure. That is stronger than what is currently justified.

There are still two distinct questions:
- Theoretical representation question: can the baseline be written in LPV-LFR
  form compatible with Drenth's notation? The derivation now says yes.
- Runtime implementation question: must the baseline be simulated through the
  explicit LFR algebraic loop at every RK4 stage, or is it acceptable to use
  the derived LFR as the representation proof and evaluate the equivalent CT
  vector field directly during RK4?

This should be made explicit before coding `CT_RK4_State_Block`, otherwise the
implementation may inherit unnecessary complexity from the document form.

### 4. The statement "affine LPV does not exist; LFR is required" should be read narrowly

The current write-up says that because `M(Y)^{-1}` gives rational dependence, a
standard affine LPV realization is not available and an LFR is required.
The practical project point is right, but the safest mathematical reading is:

"an exact ordinary affine state-space in `Y` is not available in the current
coordinates, while the latent-variable LFR gives an exact realization."

This wording avoids overstating the claim as if LFR were the only imaginable
representation.

## Exact Next Steps

### Step 1 (30 min): Check Tóth Ch. 3 for LFR definition

Read `literature/books/Toth_2010_[12]_LPVModelingIdentificationBook.pdf`
pages 49-63 (Section 3.1.2 "Representations of CT LPV Systems").
Question: does it contain the LFR/LFT interconnection structure in the
{G, Δ(p)} or equivalent form?
- If yes: add as secondary cite for Section 2 alongside Drenth IFAC.
- If no: use only Drenth IFAC for Section 2.

### Step 2: Update Section 2 citation in `LPV/supporting/derivations/LFR-derivation.tex`

Current Section 2 cites `drenth2025lpvlfr` (thesis) and `drenth2025thesis`.
Replace/verify citations based on Step 1 outcome:
- Primary cite: Drenth IFAC (`drenth2025rational`) for eq. (6)--(7)
- Secondary: Tóth (2010) Ch. 3 only if Step 1 confirms LFR is covered there

### Step 3: Update docs/references.md

Add Alkhoury et al. (2016) entry if needed.
Verify all paths and cite keys in the document match references.md.

### Step 4: Final review pass of `LPV/supporting/derivations/LFR-derivation.tex`

Check:
- No em-dashes anywhere (lessons.md rule)
- All mathematical implications justified construction-specifically (lessons.md rule)
- Stage coordinate remark is present and correct (Cy = P^T*[I3,0])
- Well-posedness note box explicitly states the reduction is construction-specific
- Compile check (LaTeX)

### Step 5 (after supervisor review): Implement CT_RK4_State_Block

Use the verified G matrix to implement the CT_RK4_State_Block in Python.
This was D-018. The G matrix is now fully derived and verified.

### Step 6 (before coding): Decide the implementation boundary explicitly

Before implementing the baseline LFR in Python, make one explicit written
decision covering both of these:
- In which coordinates will the implemented baseline state live?
- Will runtime RK4 evaluate the explicit LFR loop, or the equivalent collapsed
  CT vector field obtained from the derivation?

Without this checkpoint, "baseline uses LFR" remains underspecified.

## Open Questions / Blockers

- **Blocker A (LFR discretization paper)**: Still not found. Now less critical
  since RK4 does not require separate DT-LFR theory.
- **M0 choice**: Document uses M0 = M(0). Could use M(Y_nom=0.3). State
  explicitly in write-up (noted in document already).
- **Coordinate boundary**: Derivation is in logical coordinates, while the repo
  and data contract are stage-coordinate based. Need an explicit implementation
  decision: transform the realized baseline to stage coordinates before coding,
  or keep internal logical states and transform around them.
- **Runtime form of the baseline**: Still unclear whether the code should
  simulate the explicit LFR loop directly or treat the LFR as a proof of
  representability and integrate the equivalent CT ODE instead.
- **Minimality / repetition count**: Current derivation gives a valid LFR but
  does not prove minimality of the chosen latent dimension. Probably acceptable,
  but worth stating explicitly if asked by the supervisor.
- **April 9 meeting**: Confirm with supervisor whether trainable inertia
  parameters affect Delta^b structure during training (D-017).
- **Sample rate**: D-012 -- 16 kHz (main.m) vs 20 kHz (ETEL spec), unresolved.

## Proposed Improvements for Claude

None at this time.

## Proposed Improvements for Codex

None at this time.
