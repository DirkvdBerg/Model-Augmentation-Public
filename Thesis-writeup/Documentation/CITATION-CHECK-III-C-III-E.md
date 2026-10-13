# Citation check: thesis III-C and III-E

Claude, 2026-10-13. Every `\cite` in `Writing/sections/03_augmentation.tex` III-C (`sec:aug_loop`, l.589 to 751)
and III-E (`sec:aug_ps2`, l.896 to end), checked by reading the cited page as an image. Line numbers are those of
2026-10-13 and shift when Dirk edits.

**For Codex:** verify each row independently. Open the PDF at the given page, find the passage, and judge whether it
supports the thesis sentence as written, no stronger. Add your verdict (agree / disagree + reason) in the last column.
Do not edit the `.tex` file or `CITATION-LOG.md`; report disagreements to Dirk.

## Sources

| Key | File | Page convention |
|-|-|-|
| hoekstra2026lfr | `literature/augmentation/hoekstra2026_lfr-augmentation-fp-models.pdf` | arXiv, PDF page = printed page |
| kessels2025ai | `literature/augmentation/kessels2025_ai-control.pdf` | Ch. 5: PDF page = printed page + 27 |
| beintema2023subnet | `literature/experiment-design/Papers/Beintema_Schoukens_Toth_2023_Deep_subspace_encoders.pdf` | PDF page = printed page + 1 |
| forssell1999revisited | `literature/closed-loop-id/forssell1998_cl_revisited_liu2021.pdf` | LiU report LiTH-ISY-R-2021, not the Automatica 1999 paper the bib key names; text layer corrupted, read as image |
| hefny2015supervised | `literature/added-states-training/hefny2015_supervised-learning-dynamical-system-learning_arXiv1505.05310.pdf` | arXiv, PDF page = printed page |
| downey2017psrnn | `literature/added-states-training/1705.09353.pdf` | arXiv, PDF page = printed page |
| venkatraman2017psd | `literature/added-states-training/1709.08520.pdf` | arXiv, PDF page = printed page |

## III-C

| l. | Thesis sentence (short) | Cite | Where (PDF page, printed page) | What the passage must show | Claude | Codex |
|-|-|-|-|-|-|-|
| 605 | Instead of using the recorded input as in Eq. (22), we include the feedback controller | hoekstra2026lfr Eq. (22) | p.8, Eqs. (22a) to (22e), left column bottom | the truncated criterion simulates with the recorded input u, no controller | correct | Agree. Eq. (22b) and Fig. 6 supply the measured input directly; see E1. |
| 609 to 613 | Kessels was not able to identify an open-loop SUBNET model from closed-loop wire-bonder data; noise-induced bias from input/noise correlation and the controller's integral term | kessels2025ai Remark 5.6 | PDF p.201, printed p.174, bottom | open-loop attempt for the wire bonder unsuccessful; explanation hedged ("may be explained"), integrator named | correct; note a | Agree on substance. Tentative explanation, attributed by Kessels to [208]; E2. |
| 615 to 618 | Indirect approach with a known controller fits the closed-loop model from r; r uncorrelated with the noise, so the model can be identified from the output error | forssell1999revisited (no locator) | report Sec. 5.2.1, p.13 last paragraph to p.14 first paragraph, Eqs. (44), (45) | regulator known and r measurable; open-loop problem because r and noise are uncorrelated; output-error models with fixed noise model stay consistent | correct; note e | Qualified agreement: supported by the 1998 report; the cited 1999 journal version is unverified. Output error is justified with H*=1; E3. |
| 624 to 626 | SUBNET assumes the data-generating system is incrementally exponentially output stable | beintema2023subnet Cond. 1 | PDF p.7, printed p.6, left column, Condition 1, Eq. (15) | condition is on the data-generating system | correct; CITATION-LOG row 27 (p.5) wrong, row 106 right | Agree for SUBNET's convergence/consistency analysis. This is a theoretical assumption, not an architectural restriction; E4. |
| 627 to 628 | The identification method we have adapted from Hoekstra relies on this SUBNET assumption | hoekstra2026lfr (no locator) and Sec. 5.5 | p.8 Eq. (22) for the method; p.10, left column, Sec. 5.5 second paragraph | consistency proven under Conditions 2.1 to 2.4 of [5]; 2.1 = incrementally exponentially output stable data-generating system | correct; note b | Agree for the consistency argument. Hoekstra explicitly states the assumption; [5] is Beintema 2023. The numbering mismatch is in Hoekstra; E5. |
| 637 to 639 | Every training window is simulated in closed loop with the known controller, as in Kessels | kessels2025ai Eqs. (5.12), (5.13) | PDF pp.183 to 184, printed pp.156 to 157 | rollout driven by r and u_FF, error r minus model output, FB controller known and part of the simulated loop | correct | Agree. Eqs. (5.13c,d) explicitly close the simulated feedback loop; E6. |
| 677 to 681 | Kessels instead needs the system's controller state, measured directly or calculated by simulating the controller from the record start | kessels2025ai Remark 5.4 | PDF p.184, printed p.157 | FB states from the machine, or reconstructed from y, r and the known controller with zero initial state | correct | Agree, conditional on his assumed zero controller state at the record start. Remark 5.4 offers these initialization routes; E7. |
| 700 to 706 | Output first, then error and input: no algebraic loop through D_fb, as described by Kessels | kessels2025ai Remark 5.1 | PDF p.179, printed p.152 | output augmentation independent of the input, else a non-causal relation through the FB controller | correct in substance; note c; CITATION-LOG row 68 (p.153) wrong, row 105 right | Partial support. Same circular-dependence mechanism, but Remark 5.1 does not describe this D_fb-specific step order. Revise attribution; E8. |

## III-E

| l. | Thesis sentence (short) | Cite | Where | What the passage must show | Claude | Codex |
|-|-|-|-|-|-|-|
| 951 to 952 | PS is adapted from a two-stage regression of predictive-state learning | hefny2015supervised Sec. 2 | p.4, bullets S1A, S1B, S2 (still Sec. 2) | S1 (possibly non-linear) regressions, S2 regression of a linear operator | correct (CITATION-LOG row 38 says Sec. 3, corrected by row 86) | Agree as methodological inspiration, not an equivalence or transferred guarantee; E9. |
| 952 to 953 | it makes use of initialisation before backpropagation through time | downey2017psrnn Sec. 4.2 | p.5, Sec. 4.2 first paragraph | 2SR initialisation, then BPTT refinement | correct | Agree about Downey's procedure; E10. Your S1 also runs alongside V, as the later text explains. |
| 956 to 957 | S1 predicts the model residual instead of features of future observations | hefny2015supervised (no locator) | p.2, last paragraph | predictive state = expectation of future features given the history | correct in substance; note d | Agree. The source target is a feature vector of future observations; the residual target is your adaptation; E11. |
| 958 to 959 | S2 trains the nonlinear state update instead of a linear operator | (same cite context) | p.4, S2 bullet | W is a linear operator | correct | Agree about the contrast in the S2 regression. Hefny's complete filtering update need not be linear; E12. |
| 960 to 962 | first stage used during the initial training of V, instead of before training | downey2017psrnn | p.5 Sec. 4.2 first paragraph; p.6 first paragraph | 2SR completed before BPTT starts | correct | Agree if training means BPTT refinement. 2SR is itself a learning procedure; say before BPTT rather than before training; E13. |
| 971 | Figure layout adapted from Fig. 2 | hefny2015supervised Fig. 2 | p.2 | history, future, extended and shifted future windows with S1A, S1B, S2 | correct | Agree as layout attribution: sample timeline, window brackets and stage arrows. This does not assert algorithmic equivalence; E14. |
| 1064 to 1065 | Hefny et al. separate past and future samples to avoid correlation between history and future noise | hefny2015supervised Sec. 2 | p.3, paragraph on instrumental regression | past observations do not overlap future windows, so they are uncorrelated with the future noise | correct | Agree within Hefny's noise model. Disjoint windows alone do not guarantee decorrelation for your filtered, feedback-affected noise; E15. |

Citations inside `\todo`s (not printed in the submission build):

| l. | Cite | Where | What it must show | Claude | Codex |
|-|-|-|-|-|-|
| 937 | hoekstra2026lfr Eq. (29), Sec. 5.4.3 | pp.9 to 10 | live added-state transition at initialisation; unneeded matrices drawn U(-1,1) | per CITATION-LOG rows 99, 100; p.10 top read 2026-10-13 confirms U(-1,1) | Partial support. Eq. (29) leaves an added-state transition; unused LFR matrices are random. The paper does not directly prescribe random added-state rows of a generic ANN output layer; E16. |
| 1061 | hefny2015supervised Sec. 2 | p.2 | as l.956 | as l.956 | Agree; same evidence as E11. |
| 1071 | venkatraman2017psd Sec. 3 | Eq. (4), p.5 | decoder maps internal state to future observations, trained jointly with the task loss | per CITATION-LOG row 97, not re-read | Agree, more precisely future-observation features/statistics. Eq. (4) is trained together with the main loss; E18. |
| 1181 | hefny2015supervised Sec. 2 | p.3 | overlapping windows give correlated noise and a biased W; S1 denoises, S2 regresses on S1 estimates | correct | Disagree with the unconditional application to your S2. Hefny establishes bias for naive regression of noisy future/extended-future features; your nonlinear encoded-state regression is different. Cite as a bias risk, not a demonstrated result; E19. |

## Notes

a. Kessels does not give the bias explanation as his own; he refers to his [208] ("As discussed in [208]"). The
   thesis "He suggests that this could be explained" is slightly stronger than the source.
b. Hoekstra cites "Conditions 2.1 to 2.4 in [5]"; the Automatica version of Beintema numbers the same stability
   condition "Condition 1". Same content. Check that Hoekstra's [5] is Beintema 2023 (possibly its arXiv version).
c. Kessels speaks of a "non-causal relation" for his output augmentation, not of an algebraic loop through the
   controller feedthrough. Same mechanism; "for the same reason as Kessels" would match the source more closely
   than "as described by".
d. The phrase "features of future observations" (marked `\kw`, a fixed term) is not verbatim; Hefny writes
   "future features".
e. The read source is the LiU report; the bib key `forssell1999revisited` is the Automatica 1999 paper, whose
   section numbering is unchecked (todo l.634).

## Bib entries

- `venkatraman2017psd` added to `Writing/refs.bib` 2026-10-13, checked against p.1 of the PDF (title, eight
  authors, NIPS 2017). CITATION-LOG rows 96 and 97 still say "not in refs.bib".
- `hefny2015supervised` and `downey2017psrnn`: refs.bib comments say the NeurIPS venue is unverified; only the
  arXiv versions were read.

## Independent Codex verification, 2026-10-13

All 19 rows above were checked against the actual thesis wording and the local source PDFs.
The cited pages were rendered and inspected, including the Forssell report whose extracted text
loses mathematical symbols. Quotations below are short verbatim excerpts; equations are transcribed
with simplified indexing where stated. Explanations outside quotation marks are my interpretation.
Repeated claims refer back to the same evidence rather than repeating quotations.
No thesis `.tex`, bibliography, or `CITATION-LOG.md` was changed.

**Result:** most references support their intended background claims. Three rows require substantive
qualification: 700–706, 937, and 1181. The Forssell row has a source-version mismatch rather than a
disproved technical claim. The explanation in Kessels is tentative and credits another reference.

### E1 — III-C, line 605: recorded input in Hoekstra

Source: [Hoekstra PDF, p.8](../../literature/augmentation/hoekstra2026_lfr-augmentation-fp-models.pdf#page=8),
Eq. (22b) and Fig. 6.

The right-hand vector in Eq. (22b) contains the recorded input **`u_{k_i+ell}`** beside the
simulated state. Fig. 6 likewise feeds each measured input directly into the corresponding
rollout step; there is no simulated feedback controller. This directly supports the comparison
with your closed-loop rollout. The evidence here is the displayed equation and diagram.

### E2 — III-C, lines 609–613: failed open-loop identification and possible bias

Source: [Kessels PDF, p.201 / printed p.174](../../literature/augmentation/kessels2025_ai-control.pdf#page=201),
Remark 5.6.

Direct excerpts: **“but was unsuccessful”**, **“As discussed in [208]”**,
**“may be explained”**, and **“integration action”**.

The surrounding passage explicitly identifies the unsuccessful attempt as open-loop SUBNET
identification for the wire bonder using closed-loop input/output data. It connects the possible
bias to correlation between input and output measurement noise and controller integration.
Your tentative wording is supported. Prefer attribution such as: Kessels reports the unsuccessful
attempt and, referring to [208], gives noise-induced bias as a possible explanation.
This passage does not establish that bias was experimentally isolated as the cause.

### E3 — III-C, lines 615–618: indirect identification and output error

Source: [Forssell–Ljung report, pp.13–14](../../literature/closed-loop-id/forssell1998_cl_revisited_liu2021.pdf#page=13),
Sec. 5.2.1, Eqs. (44)–(49). PDF page and printed page agree here.

Direct excerpts: **“the noise and the reference signal are uncorrelated”** (p.13) and
**“output error models with fixed noise models (prefilters) and still guarantee consistency”** (p.14).

The section assumes a known regulator and measurable reference. It first estimates the
reference-to-output closed-loop model, then recovers plant parameters using the regulator.
Equation (45) is a one-step prediction equation with a fixed noise model. Setting `H*=1`
reduces it to simulation of the closed-loop transfer function, so its prediction error is
the output error. With a nonidentity fixed noise model, the error is filtered.
Thus output error is defensible; output prediction error is the more general description.
Consistency still requires the relevant identification/model assumptions; decorrelation alone
does not prove consistency of your nonlinear augmented estimator.

**Citation issue:** this PDF's cover identifies an updated technical report dated 1 April 1998,
LiTH-ISY-R-2021, submitted to Automatica. The bibliography entry names the 1999 Automatica
article. This check verifies the report's passage, not the corresponding passage or section
number in the journal article. Cite the report or verify the journal version before declaring
that bibliographic citation checked. The filename's `liu2021` is the report number, not its year.

### E4 — III-C, lines 624–626: SUBNET stability assumption

Source: [Beintema PDF, p.7 / printed p.6](../../literature/experiment-design/Papers/Beintema_Schoukens_Toth_2023_Deep_subspace_encoders.pdf#page=7),
Condition 1.

Direct quotation: **“The data-generating system (4) is (globally) incrementally exponentially output stable”**.

This is explicitly a condition on the data-generating system. The surrounding section introduces
it for convergence and consistency analysis. Safer precision: SUBNET's consistency analysis
assumes this property. An implementation can still run on a system outside the assumption.

### E5 — III-C, lines 627–628: Hoekstra inherits the stability assumption

Source: [Hoekstra PDF, p.10](../../literature/augmentation/hoekstra2026_lfr-augmentation-fp-models.pdf#page=10),
Sec. 5.5.

Direct quotation: **“Condition 2.1 requires the data-generating system to be incrementally exponential output stable”**.

The paragraph explicitly makes this an assumption on system (1), alongside the other conditions
for convergence and consistency. Reference [5], checked in Hoekstra's bibliography on p.15,
is Beintema, Schoukens and Tóth, Automatica 156 (2023), 111210. However, that published
PDF numbers the stability requirement Condition 1, differentiability Condition 2, predictor
convergence Condition 3, and excitation Condition 6. Hoekstra's numbering does not match
the published numbering; the stability content itself does match.

### E6 — III-C, lines 637–639: closed-loop training windows in Kessels

Source: [Kessels PDF, pp.183–184 / printed pp.156–157](../../literature/augmentation/kessels2025_ai-control.pdf#page=183),
Eqs. (5.12), (5.13c,d).

Direct excerpt: **“known closed-loop FP model”** (p.184).

The displayed equations give, with window indices suppressed here,
`e_hat = r_bar - y_hat`, `x_FB_hat_next = f_FB(x_FB_hat,e_hat)`,
`u_FB_hat = g_FB(x_FB_hat,e_hat)`, and `u_tilde_hat = S_u(u_FB_hat+u_FF_bar)`.
The reference and known feedforward drive each window, and feedback uses that window's
simulated output. This supports the stated comparison directly.

### E7 — III-C, lines 677–681: initializing controller states

Source: [Kessels PDF, p.184 / printed p.157](../../literature/augmentation/kessels2025_ai-control.pdf#page=184),
Remark 5.4.

Direct excerpts: **“directly available”** and **“may be reconstructed”**.

The remark offers machine controller states or reconstruction using measured output, known
reference and known controller. It explicitly assumes `x_FB_bar(k=1)=0` and uses the result
to initialize each truncated window. Simulating the controller from the record start is a
faithful description of that reconstruction, subject to the initial-state assumption.

### E8 — III-C, lines 700–706: algebraic-loop attribution

Source: [Kessels PDF, p.179 / printed p.152](../../literature/augmentation/kessels2025_ai-control.pdf#page=179),
Remark 5.1.

Direct excerpt: **“non-causal relation”**.

Kessels excludes current input from his output augmentation because otherwise output depends
on feedback-controller output, which itself depends on output. This supports the mechanism.
The remark does not explicitly discuss your `D_fb`, state-only output map, or exact evaluation
order. Those follow from your equations. Replace the attribution with, for example:
This avoids an algebraic loop through `D_fb`, for the same reason that Kessels excludes
input dependence from the output augmentation (Remark 5.1).

### E9 — III-E, lines 951–952: two-stage regression

Source: [Hefny PDF, p.4](../../literature/added-states-training/hefny2015_supervised-learning-dynamical-system-learning_arXiv1505.05310.pdf#page=4),
Sec. 2, S1A/S1B/S2 bullets.

Direct excerpt for S1: **“(possibly non-linear) regression model”**.

S1A estimates `E[psi_t | h_t]`; S1B estimates `E[xi_t | h_t]`. S2 fits the mapping between
those estimates, `xi_bar_t = W psi_bar_t`. This establishes the two-stage regression structure.
Your residual target, nonlinear transition and interleaving with `V` are adaptations rather
than the original algorithm. The citation supports the stated inspiration, not its guarantees.

### E10 — III-E, lines 952–953: initialization before BPTT

Source: [Downey PDF, p.5](../../literature/added-states-training/1705.09353.pdf#page=5), Sec. 4.2.

Direct excerpts: **“an initialization procedure followed by gradient-based refinement”** and
**“this model can then be refined via BPTT”**.

The paragraph identifies 2SR as the initialization procedure. This directly supports using
2SR-derived initialization before BPTT as a precedent.

### E11 — III-E, lines 956–957: future features versus model residual

Source: [Hefny PDF, p.2](../../literature/added-states-training/hefny2015_supervised-learning-dynamical-system-learning_arXiv1505.05310.pdf#page=2),
Sec. 2, final paragraph; see also S1 on p.4.

Direct excerpt: **“future features”**.

The paragraph defines `q_t = E[psi_t | o_{1:t-1}]` and `psi_t = psi(o_{t:t+k-1})`.
Thus features of future observations is an accurate paraphrase. It is not a verbatim quotation.
Predicting the model residual instead is your modification, which the source does not evaluate.

### E12 — III-E, lines 958–959: S2 linear operator

Source: [Hefny PDF, p.4](../../literature/added-states-training/hefny2015_supervised-learning-dynamical-system-learning_arXiv1505.05310.pdf#page=4), S2 bullet.

Direct excerpt: **“where W is a linear operator”**.

This confirms the contrast with your nonlinear S2 regression. Be precise about what is linear:
Hefny's S2 operator maps predictive to extended-state expectations. The complete filtering
update also includes conditioning and need not be linear.

### E13 — III-E, lines 960–962: first stage before versus during training

Source: Downey Sec. 4.2, p.5; quotations in E10.

The ordering is initialization followed by BPTT refinement, so the intended comparison holds.
However, before training is imprecise because 2SR itself learns model parameters. Prefer:
We use S1 during the initial optimization of `V`, whereas Downey et al. complete 2SR
initialization before BPTT refinement.

### E14 — III-E, line 971: figure attribution

Source: [Hefny PDF, p.2](../../literature/added-states-training/hefny2015_supervised-learning-dynamical-system-learning_arXiv1505.05310.pdf#page=2), Fig. 2.

The figure places history, future, extended-future and shifted-future windows around a sample
timeline, with S1A/S1B/S2 arrows. Your `figures/tex/ps2_windows.tex` uses a sample timeline,
window brackets and stage arrows for a different target and state-update procedure. Layout
adapted from Fig. 2 is a reasonable attribution; it does not claim the methods are identical.

### E15 — III-E, lines 1064–1065: history and future noise

Source: [Hefny PDF, p.3](../../literature/added-states-training/hefny2015_supervised-learning-dynamical-system-learning_arXiv1505.05310.pdf#page=3), instrumental-regression paragraph.

Direct excerpt: **“not correlated with the noise”**.

The preceding clause explicitly concerns past observations not overlapping future or
extended-future windows. This supports your statement about Hefny's construction within
their noise model. It does not show that disjoint windows decorrelate temporally filtered
noise, or that the residual of your feedback simulation is independent of encoder-history
noise. Your following sentence establishes absence of shared samples; that narrower fact
is defensible. Keep the existing limitation visible in the submitted discussion.

### E16 — todo line 937: random added-state initialization

Source: [Hoekstra PDF, pp.9–10](../../literature/augmentation/hoekstra2026_lfr-augmentation-fp-models.pdf#page=9),
Eq. (29), Sec. 5.4.3.

Direct excerpts: **“All matrices not required”** and **“initialised randomly”** (p.10).

Eq. (29) leaves `x_a,k+1 = phi_aug(theta_aug^0,z_a,k)` while retaining baseline physical
state/output behavior. Sec. 5.4.3 specifies `m ~ U(-1,1)` for matrix elements not needed
to enforce that baseline behavior. However, the end of p.9 explicitly initializes the
nonlinear learning component to zero and retains a linear ResNet component,
`phi_aug(z_a,k)=0+W_a z_a,k`.

Consequently, the paper supports retaining a potentially nonzero added-state transition and
random unused LFR matrix entries. It does **not directly prescribe random added-state rows
of the output layer of your ANN**. That requires a mapping from their ResNet/LFR
parameterization to your implementation. Describe your choice as an adaptation consistent
with Eq. (29), rather than claiming that these exact ANN rows are prescribed by the paper.
The paper also does not guarantee a nonzero added-state trajectory for every initial state.

### E17 — todo line 1061: future-feature target

Same source, quotation and verdict as E11. The repeated citation is supported.

### E18 — todo line 1071: predictive-state decoder jointly trained with task loss

Source: [Venkatraman PDF, p.5](../../literature/added-states-training/1709.08520.pdf#page=5), Sec. 3, Eq. (4).

Direct excerpts: **“a decoder that maps from the internal state”** and
**“weighted total loss function”**.

The displayed objective is `R = sum_t ||F(h_t)-phi([x_{t+1},x_{t+2},...])||_2^2`,
and optimization minimizes `L + lambda R`. This directly confirms a decoder from internal
states to future-observation features/statistics, jointly trained with the task objective.
Predicting future observations is accurate when those features are the observations;
future-observation features is the more general wording.

### E19 — todo line 1181: overlap necessarily biases your nonlinear S2

Source: [Hefny PDF, p.3](../../literature/added-states-training/hefny2015_supervised-learning-dynamical-system-learning_arXiv1505.05310.pdf#page=3),
discussion after Eq. (1), and p.4 S2 bullet.

Direct excerpts: **“a biased estimate of W”** and **“effectively denoising”**.

Hefny considers direct regression of noisy future and extended-future feature samples in
place of conditional expectations. Their overlap correlates the noise, biasing the naive
estimate of `W`. The proposed remedy estimates conditional expectations in S1 and fits
`W` to those estimates in S2.

Your S2 instead fits a nonlinear update to pairs of encoded states from adjacent history
windows. Shared samples motivate a similar errors-in-variables concern, but this source
does not establish that your estimator is necessarily biased, quantify that bias, or show
that this is the same regression problem. The unconditional sentence is too strong.

Suggested replacement: The adjacent encoder windows share samples, so noise can enter
both sides of the S2 regression. This creates a potential bias risk analogous to the
correlated-noise problem discussed by Hefny et al.; their two-stage procedure regresses
first-stage conditional-expectation estimates rather than noisy feature samples directly.
The bias of our nonlinear encoded-state regression has not been established here.

### Bibliographic scope

The checked local versions are Hefny arXiv v2, Downey arXiv v2, Venkatraman arXiv v1,
Hoekstra arXiv, Kessels's thesis, Beintema's published Automatica PDF with a repository
cover, and the 1998 Forssell–Ljung technical report. Official arXiv records were also opened
for [Hefny](https://arxiv.org/abs/1505.05310), [Downey](https://arxiv.org/abs/1705.09353),
[Venkatraman](https://arxiv.org/abs/1709.08520), and [Hoekstra](https://arxiv.org/abs/2602.17297)
to check source identity. This content audit does not independently verify all conference
venue metadata or substitute the 1999 Forssell journal article for the report.
