# Meeting with Jan — figure walkthrough (answers to the meeting notes)

Maps each point from the meeting notes to the figure(s) that answer it and the one-line finding.
Figures live in `scripts/gantry/drift-visual/figures/` (f0x) and `scripts/gantry/gantry-zero-mean/figures/`
(v1*, v3*, v4, v5, v6). Full numbers: `RESULTS-2026-07-17-dc-drift-diagnosis.md` and README section 6/8.

Order below follows Jan's note themes: D (nf / first steps) -> A (is the system zero-mean?) ->
C (where is the DC born?) -> B (estimator zero-mean world) -> the f04b paradox -> what's NEW since.

---

## Theme D — nf / first steps / accumulation / no fading memory
Notes: "nf naar beneden", "eerste 100 stappen", "explodeert het na de eerste 100 stappen",
"geen fading memory want geen damper", "error accumuleert over de nf range, eerste stappen belangrijk",
"constant er naast zit, accumulate over lange tijdstap (nf 400)".

| figure | shows / answer |
|---|---|
| `v4_perstep_error.png` | Per-step error WITHIN the nf=400 window (full / DC-muted / ANN-off). **It does NOT explode after 100 steps** — it is a steady LINEAR ramp on the K=0 axes (X, Y). |
| `v4_growth_law.png` | Growth-law fit: X linear R²=0.995, Y R²=0.994 (a constant-VELOCITY-offset ramp, not quadratic/force, not exponential/instability); Theta has no growth law. Answers "explodeert het?" = **no, linear ramp**. |
| `v4_reference_subtraction.png` | **No fading memory is K=0-specific**: late/early envelope ratio X 7.6, Y 7.0, Theta 1.0 (Theta parks because it has a spring; X/Y ramp because K=0). The encoder-init error dominates the within-window ramp. |
| `f09_horizon.png` | Why training is blind: drift grows ~T², only 0.79× the floor at the 0.1 s window but 784× at 12 s. The first steps carry the clean information. |
| `v3nf800_perstep_seed0.png` (+ truncation sweep) | **"nf down" tested**: the DC scales as ~1/nf (400→800 halved it), so lowering nf REDUCES the DC but never removes it. The earlier full sweep (nf up to 3200) still drifted -> **longer/shorter nf is not a fix**, the integrator turns any residual into drift. |

---

## Theme A — is the system / MSD itself non-zero-mean? (Jan's primary hypothesis)
Notes: "signalen voor en na het toevoegen van de extra mass, nog steeds zero mean?",
"de MSD heeft een offset alleen al kijkend naar de formules", "systeem niet zero mean zou het verklaren",
"baseline loopt op tot een nieuw stabiel punt, oscilleert, niet zero mean", "verschil baseline / true system",
"msd 50/50 split", "gantry met een zero mean signal zou heen en weer moeten bewegen".

| figure | shows / answer |
|---|---|
| `v1b_dmean_positions.png` / `..._velocities.png` / `..._forces.png` | With-MSD vs without-MSD signal MEANS (MATLAB, 12 matched records). `mean(with) - mean(without)` is zero to within the error bars on EVERY channel (positions, velocities, forces). **The measured signals are zero-mean before and after adding the mass.** |
| `v1d_T1_standstill_Ym30_difference.png`, `v1d_T8_ysweep_xmix_difference.png` | Open-loop SAME-INPUT truth-minus-baseline difference: a tiny slow deterministic component (the one-sided MSD content), **3–4 orders of magnitude too small** to be the ANN's DC. |
| `v1f_Y_dcac_response.png` (+ `_X_`, `_Theta_`) | DC + 150 Hz excitation, per axis. **Static-gain DC = 0** (spring unpreloaded); the MSD's rectification DC (delta_a² through the Theta inertia) = 3e-10, and the largest physics DC anywhere ~1e-7 — **5+ orders below the ANN's DC**. So Jan's "MSD offset" intuition is REAL but negligibly small. |
| `f03_baseline.png` | The baseline error PARKS at a bounded offset and **nothing drifts without the ANN**. A constant force would RAMP on a K=0 axis, not park — so "correct it with a constant force" is exactly what the ANN does and exactly what drifts. |
| `f04b_target_logical.png` / `f04c_target_stage.png` | The REQUIRED correction (truth − baseline_step) is zero-mean: \|mean\|/rms = 0.000 on every row. |

**Answer to Theme A:** the physics is zero-mean at the level that matters. There is a genuine but
tiny second-order rectification DC from the MSD (confirmed, delta_a²->Theta), so the intuition is
correct in principle — but it is ~5 orders too small to explain the ANN's constant. **The system does
not demand the ANN's DC.**

---

## Theme C — where / when is the constant born during training?
Notes: "geen duidelijke reden waarom de ANN weg duwt / non-zero-mean leert", "kan lr omlaag",
"kijk naar de eerste epoch, de steps", "voor de eerste update step nog zero mean, waarom leert het
non-zero-mean", "verschillende seeds, xavier weights", "monte carlo, seed niet vastzetten",
"kleine lr, niet explodeert, eerste stappen".

| figure | shows / answer |
|---|---|
| `v3b_perstep_seed0.png` / `seed1` / `seed2` | The DC per optimizer step, epoch 1, lr=1e-7, **3 unfixed seeds**. The DC is born by ~step 13 and has the **SAME SIGN across all seeds** -> **systematic, not random wander** (diffusion would scatter). |
| `v3b_multiseed_dc.png` | The three seeds overlaid: same-sign DC on the K=0 rows; the loss gradient along a constant correction (dLoss/dbias) reproduces in sign and matches the DC direction -> the loss weakly but CONSISTENTLY rewards the DC. |
| `v3x0sgd_encoder_multiseed.png` | **The answer to "why non-zero-mean": it is Adam's implicit bias.** Same setup, only Adam→SGD: SGD builds ~2000× LESS DC at the SAME loss. Adam (≈ sign-descent) amplifies the tiny consistent gradient in the loss-flat DC direction into a steady walk; SGD takes a vanishing step. |
| `v3joint_multiseed_dc.png` | Broadband 1–200 Hz excitation: DC essentially identical to narrowband -> **not an identifiability/excitation problem**. |
| `v3x0true_true_multiseed.png` | Feeding the TRUE initial state (encoder bypassed): the DC still forms -> **not the encoder init**. |

**Answer to Theme C:** the DC is born in the first ~13 steps, systematically (same sign across seeds),
because the windowed loss is nearly FLAT in the DC direction but carries a tiny consistent gradient,
and **Adam amplifies it** (SGD at the same loss does not). Small lr (1e-7) confirmed nothing explodes
in the first steps — the DC still appears. Refutes excitation and encoder-init as the cause.

---

## Theme B — the estimator is built for a zero-mean world
Notes: "cost function non-zero-mean gaussian, alle normalizaties op zero-mean gaussian",
"xavier verwacht zero mean std 1", "impact van een constante waarde die naar één kant trekt",
"train/test mismatch, cost landscape niet goed geïnitialiseerd".

| figure | shows / answer |
|---|---|
| `v3x0sgd_encoder_multiseed.png` | The concrete "impact of a constant in a zero-mean-designed estimator": in the loss-flat (constant) direction, **Adam's implicit bias walks off center**; a non-adaptive optimizer does not. This is the estimator-side mechanism, made empirical. |
| `f09_horizon.png` | The train/test mismatch, quantified: the windowed training objective is ~neutral to the drift (0.79× floor at the window) while deployment is 784× — the objective improves while the free-run explodes. |

**Answer to Theme B:** the zero-mean/std-1 assumptions (Xavier, normalization, cost) are indeed
violated by the operating-point constants, and the train/test mismatch is real (f09) — but the
DECISIVE, measured mechanism is Adam's implicit bias in the flat direction (v3x0sgd), not the
normalization pathway alone.

---

## The f04b paradox (the thing that was confusing)
Note: "f04b/f04c: de correctie die de ANN moet maken is zero-mean, maar de correctie die ik nodig heb
is zero-mean, hoe kan dat dan? maar de constante kracht kan verklaard worden als de MSD non-zero-mean is."

| figure | shows / answer |
|---|---|
| `f04b_target_logical.png` / `f04c_target_stage.png` | Learned vs REQUIRED correction side by side: **required is zero-mean (0.000), learned is not**, and the ANN supplies <1% of the required dY amplitude. |
| `f05_counterfactual.png` | Causality: subtract the ANN's measured mean -> the drift mostly collapses. The DC is what drifts. |
| `v5_dc_null_counterfactual.png` | **The deeper resolution (new).** Nulling the K=0 DC brings X drift back to baseline but makes the dominant **Y drift WORSE** — so the dominant drift is NOT the DC, it is the ANN's state-dependent (dynamic) output destabilizing the marginal axis. |

**Resolution:** the required correction IS zero-mean; there is no contradiction. The baseline ERROR
parks non-zero-mean, but the CORRECTION that prevents it is zero-mean (a constant force would RAMP on
a K=0 axis, not park). The ANN's non-zero-mean is therefore NOT demanded by the data — it is an
optimizer artifact (Adam), and the dominant long-horizon drift is the ANN destabilizing the free-run
(v5), not the MSD being non-zero-mean (v1f: physics DC is 5 orders too small).

---

## What's NEW since the last meeting (this chat) — the reframe and the direction

| figure | shows |
|---|---|
| `v3x0sgd_encoder_multiseed.png` | **Adam is the amplifier** (SGD 2000× less DC at same loss) — mechanism of "why non-zero-mean". |
| `v3nf800_perstep_seed0.png` | Truncation sweep: DC ~ 1/nf; longer windows reduce but do not fix (refutes "nf down" as a fix). |
| `v3pole1k_perstep_seed0.png` | Pole-perturbation (add stiffness to move z=1): inconclusive — stiffness wrecks the fit without moving the damping-limited decay. Diagnostic, not a fix. |
| `v5_dc_null_counterfactual.png` | **The reframe**: the dominant drift is the ANN DESTABILIZING the long free-run (Y ~50× the physics baseline), not a DC. The problem is stability, not a constant. |
| `v6_lipschitz_sweep.png` | First stability-preserving prototype: a by-construction Lipschitz cap on the ANN. Control diverges (Y 114×); L=1 was non-binding (need a tighter, binding sweep). |

**Direction (for discussion with Jan):** the clean, real-data-transferable fix is stability-preserving
augmentation BY CONSTRUCTION, not a DC pin. We already have WELL-POSEDNESS (Drenth `D_zw=e^{-N}`); the
missing half is STABILITY. Routes: Györök contraction (own group, same LFR) vs port-Hamiltonian
passivity (Moradi, handles the marginal z=1 mode natively) vs the SDP/ISS route (Ghanipoor). Open
research question = preserving the genuine z=1 integrator while guaranteeing the augmentation can't
diverge. (Decisions D-117/D-118; literature `literature/stability-training/`.)

---

## Action items from the notes (not figures)
- **Jan's parallel-batch script (batch id)** for the seed Monte Carlo — the V5 statistical closure
  (v3b is the 3-seed preliminary; the batch script scales it). Awaiting Jan.
- **Simulink-side V1 confirmation** (signal means at 20 kHz native) — open, Jan's explicit ask.
- **"MSD 50/50 split"** — could be checked, but v1b/v1f already show the offset's DC is 5 orders too
  small, so the split is not expected to change the conclusion.
