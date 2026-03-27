# Prompt For Drafting The Clean Supervisor Version

Use the following prompt when drafting the clean supervisor-facing LPV-LFR
derivation.

---

Write a new clean supervisor-facing derivation of the dual-gantry CT LPV-LFR
realization.

Use these two files as the main references:

- `LPV/supporting/supervisor-notes/LFR-supervisor-inclusion-guide.md`
- `LPV/supporting/supervisor-notes/LFR-supervisor-outline.md`

Use this file as the detailed mathematical source of truth:

- `LPV/supporting/verification/LFR-derivation-verification.tex`

Requirements:

1. The wording should be concise and supervisor-facing, but do not sacrifice
context or skip necessary justification.

2. The mathematics should show all important intermediate steps. Do not compress
away derivation steps just to save space.

3. Cite or justify based on Drenth where appropriate, especially for:
   - the CT LPV-LFR framework
   - the role and block form of `G`
   - the repeated block-diagonal structure of `Delta(p)`
   - the exact generic loop-solvability condition

4. Make the boundary explicit between:
   - what comes from Drenth
   - what is plant-specific in the dual-gantry derivation

5. Define rational dependency before using it, and justify it by stating that
   `M(Y)^{-1} = adj(M(Y)) / det(M(Y))`, without explicitly expanding the full
   inverse.

6. Clearly justify why the physical scheduler is only `Y`, not independent `Y`
   and `Y^2`.

7. Clearly justify the chosen repeated block:
   - `Delta(Y) = Y I_6`
   - explain why the repetition count is `6` for the chosen realization

8. Define the latent variables clearly and explicitly:
   - `v = M(Y)^{-1} f_gen`
   - `v = ddot(q) = [ddot(X), ddot(Theta), ddot(Y)]^T`
   - `v_1 = Y v`
   - `v_2 = Y^2 v = Y v_1`
   - state that `v_1` and `v_2` are derived latent vectors, not physical states

9. Explicitly derive the constant LPV-LFR blocks:
   - `A`, `B_w`, `B_u`
   - `C_z`, `D_zw`, `D_zu`
   - `C_y`, `D_yw`, `D_yu`
   Show the intermediate coefficient-matching steps.

10. Explicitly justify the output choice `y = q` and show that it matches the
MATLAB output map `C_c = [I_3 0]`.

11. Explicitly show the collapse of the LFR back to the original MATLAB-derived
CT state-space model.

12. In the well-posedness section, explicitly write the bridge:
    `(I - D_zw Delta(Y)) z = C_z x + D_zu u`
    and then show how the chosen realization reduces this to:
    `M(Y) v = f_gen`, `v_1 = Y v`.

13. State well-posedness carefully:
    - for the chosen realization, after reduction to `M(Y) v = f_gen`,
      invertibility of `M(Y)` implies uniqueness of `v`, then of `v_1`, `v_2`,
      hence of `z` and `w`, and therefore well-posedness
    - do not state or imply that invertibility of `M(Y)` is a generic LPV-LFR
      well-posedness result in isolation

14. Make clear that the well-posedness proof used here is the plant-specific
exact reduction, not Drenth's generic sufficient theorem from Section 2.2.

15. State explicitly that the realization is exact, but no minimality claim is
made.

Structure the document according to `LPV/supporting/supervisor-notes/LFR-supervisor-outline.md`.

The tone should be mathematically clear, calm, and direct. Avoid the internal
audit style of the verification note.

---
