# Reference Map

Maps LaTeX cite keys to PDF file locations and brief descriptions.

PDFs are organized in `literature/` subfolders:
- `literature/gantry/` - system model and hardware specs
- `literature/lpv-lfr/` - LPV and LFR methodology papers
- `literature/augmentation/` - model augmentation framework papers
- `literature/books/` - textbooks and theses
- `literature/math/` - mathematical references

---

## Gantry system

| Cite key | File | Description |
|----------|------|-------------|
| `garcia2013model` | `literature/gantry/garcia2013_gantry-decoupling-control.pdf` | Garcia-Herreros et al. - dual-gantry FP model (Euler-Lagrange, stage coordinates). Ground truth for the baseline. |
| - | `literature/gantry/telica-xyz-0750-0800-data.pdf` | ETEL Telica datasheet - XYZ dual gantry specs. Y stroke = 800 mm, operational range: Y = 0.05 to 0.75 m. |
| - | `literature/gantry/AccurET-Oper&Soft-VerV.pdf` | ETEL AccurET manual. Loop rates: PLTI = 50 us (20 kHz), CLTI = 50 us (20 kHz), MLTI = 400 us (2.5 kHz). |

---

## LPV and LFR methodology

| Cite key | File | Description |
|----------|------|-------------|
| `drenth2025lpvlfr` | `literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf` | Drenth, Hoekstra, Schoukens, Toth - "Efficient Learning of Affine and Rational Dependency LPV Models With LFR" (IFAC). Discrete-time companion paper: defines the LPV-LFR pair `{M, Delta(p)}` in DT (eq. 6-9), discusses rational dependency and well-posedness. Use for DT LPV-LFR context and rational-dependency motivation, not as the primary CT source. |
| `toth2010discretization` | `literature/lpv-lfr/toth2010_zoh-discretization-lpv.pdf` | Toth (2010) - ZOH discretization of LPV systems. Assumptions 1 and 2, complete method eq. 9a, footnote 2 on singular `A_c`. |
| `schoukens2018mimo` | `literature/lpv-lfr/Schoukens_2018_LPVRepresentationMIMO.pdf` | Schoukens and Toth (2018) - LPV representation of MIMO nonlinear systems via NLFR embedding and factorization `f(z) ->` scheduling map. |
| `schoukens2020lfr` | `literature/lpv-lfr/Schoukens_2020_LFRInitializationBLA.pdf` | Schoukens (2020) - LFR initialization via best linear approximation. |

---

## Model augmentation framework

| Cite key | File | Description |
|----------|------|-------------|
| `hoekstra2026lfr` | `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf` | Hoekstra, Verhoek, Toth, Schoukens - LFR-based model augmentation (European Journal of Control, 2025). The augmentation framework this project builds on. |
| `kessels2025ai` | `literature/augmentation/kessels2025_ai-control.pdf` | Kessels (2025) - AI in control. |

---

## Books and theses

| Cite key | File | Description |
|----------|------|-------------|
| `drenth2025thesis` | `literature/books/drenth2025_lpv-lfr-thesis.pdf` | Drenth - Master thesis: gradient-based learning of LPV-LFR models (TU/e, 2025). Primary CT LPV-LFR source: Chapter 2 defines the LPV-LFR pair `(G, Delta(p))` in continuous time with `x_dot(t)`, `z(t)`, `w(t)`, `y(t)` and shows the equivalent rational LPV-SS form; Chapter 5 covers augmentation with an LFR baseline. |
| `toth2010modeling` | `literature/books/Toth_2010_[12]_LPVModelingIdentificationBook.pdf` | Toth (2010) - Modeling and Identification of LPV Systems (Springer). LPV-SS / LFR equivalence, quasi-LPV, LFT representations. |
| `tsaigu2013robust` | `literature/books/Robust and Optimal Control.pdf` | Tsai and Gu - Robust and Optimal Control: A Two-port Framework Approach (Springer, AIC series). Generic LFT / LFR background via two-port networks. |
| - | `literature/books/Zhou-Robust_and_optimal_control.pdf` | Zhou, Doyle and Glover (1996) - title page plus ToC only (incomplete). Full book not available locally. |

**Important source split**:
- `drenth2025thesis` (thesis) is the source to cite for continuous-time LPV-LFR definitions used in the gantry derivation.
- `drenth2025lpvlfr` (IFAC paper) is the discrete-time companion and should be cited as such.

---

## Math references

| Cite key | File | Description |
|----------|------|-------------|
| - | `literature/math/positive-definite-matrices.pdf` | Positive definite matrices - used for `M(Y)` invertibility proofs. |

---

## Other literature (not uploaded)

| Cite key | Description |
|----------|-------------|
| `zhou1996robust` | Zhou, Doyle and Glover (1996) - Robust and Optimal Control (Prentice Hall, full book). LFT framework, star products, SVD realization. Not available locally - only title page uploaded. |
| `gyorok2025l4dc` | Gyorok et al. - orthogonal projection regularization (L4DC 2025). Aspect 3. |
| `verhoek2022deeplpv` | Verhoek et al. - deep LPV (2022). |
| `champneys2024baseline` | Champneys et al. - baseline comparison, BFR (2024). |
| `paijmans2008identification` | Paijmans et al. - operating-point LPV identification (2008). |
| `bachnas2014review` | Bachnas et al. - LPV identification review (2014). |
| `kon2022physics` | Kon et al. - physics-informed regularization (2022). |
| `hoekstra2026encoder` | Hoekstra et al. - encoder initialization for augmentation. |

---

## Project documents

| Document | File |
|----------|------|
| Research plan | `Research-Plan/research-plan-dirk-van-den-berg.pdf` |
| Robot identification benchmark | `literature/robot-benchmark-full/` |
