# Claude Paste Prompt

Paste the following into Claude if you want a strong external verification pass.

```text
Please use `docs/drenth/claude-handoff.md` as the primary instruction and context file for this review.

I want a critical verification-oriented review of the dual-gantry CT LPV-LFR derivation against Drenth's THESIS. Do not primarily rely on the IFAC paper for the continuous-time LPV-LFR framework.

Main source to compare against:
- `literature/books/drenth2025_lpv-lfr-thesis.pdf`

Main document to review:
- `LPV/supporting/verification/LFR-derivation-verification.tex`

Additional supporting files you may use if needed:
- `docs/drenth/ch2-sec21-source.md`
- `docs/drenth/ch2-sec211-source.md`
- `docs/drenth/ch2-sec22-source.md`
- `docs/drenth/ch2-generalized-recipe.md`
- `docs/drenth/ch2-dual-gantry-mapping.md`
- `LPV/supporting/derivations/LFR-derivation.tex`
- `LPV/supporting/derivations/M-invertibility.tex`

Review instructions:
1. First read `docs/drenth/claude-handoff.md` and follow its source boundaries and interpretation rules.
2. Then read `LPV/supporting/verification/LFR-derivation-verification.tex` as the main verification document.
3. Check the Drenth-based claims against the thesis, especially Sections 2.1, 2.1.1, and 2.2.
4. Verify whether statements marked or described as `Direct from Drenth` are truly directly supported by the thesis.
5. Verify whether statements marked or described as `Generalized from Drenth` are reasonable and not too strong.
6. Check the dual-gantry-specific derivation for algebraic correctness, especially:
   - `M(Y) = M_0 + Y M_1 + Y^2 M_2`
   - the choice of latent variables `v`, `v_1`, `v_2`
   - `z = [v; v_1]`, `w = [v_1; v_2]`
   - `Delta(Y) = Y I_6`
   - the derivation of `A`, `B_w`, `B_u`, `C_z`, `D_zw`, `D_zu`
   - the reduction of the algebraic loop to `M(Y) v = f_gen`
7. Check the well-posedness discussion carefully and keep the two routes distinct:
   - Drenth's generic sufficient Section 2.2 route
   - the sharper plant-specific `M(Y)`-invertibility route
8. Flag any line that overclaims what Drenth actually proves or states.
9. Flag any line that is ambiguous, insufficiently sourced, or mathematically weak.
10. If you see a better formulation, a clearer ordering, or an actual mathematical mistake, propose concrete improvements.

Please do not give me only a friendly summary. I want a real verification pass.

Please structure your response as:
- Confirmed statements
- Potential overclaims
- Algebraic issues or points needing manual checking
- Suggested wording or structural improvements
- If useful, proposed revised text snippets

If possible, reference the relevant line or local passage in `LPV/supporting/verification/LFR-derivation-verification.tex` and the corresponding section/equation in Drenth's thesis.
```
