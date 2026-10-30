# PS figure review

29 October 2026. Independent review against the supplied sources.

**The repeated failure is a mismatch between what the picture claims and what the training actually does.** Early figures label a wave “residual” without constructing it. Later figures construct the residual but make the decoder look like an output correction, or hide the state prediction in a small arrow. Some newer concepts make the explanation attractive by assuming an oscillator geometry or an optimisation trajectory that has not been established. Changing the layout cannot repair those substitutions.

There is enough context to design a defensible explanation. There is no reason to demand another long prompt from Dirk. There are, however, real limits to what this method permits a picture to claim. In particular, PS encourages useful predictive information; it does not identify physical absorber coordinates or guarantee a circular state trajectory.

My choice is **concept 1: construct one training example, then distinguish its prediction route from its target route**. Use separate S1 and S2 slides, preceded by a residual build and followed by the already accepted training overview. Concept 3 is the better alternative if the committee's main question is why the auxiliary prediction helps. Concept 2 is strongest for explaining where the regression data come from.

## Scope and evidence

I inspected all supplied PNGs and the TikZ/HTML sources for all attempt families, including all 15 artboards of attempt 04 and all 12 builds of 03b. I reconstructed the inline SVG/HTML canvases locally, evaluated their `renderVals()` functions where needed, and inspected screenshots. These reconstructions omit the unavailable Claude runtime and remote fonts, so they support judgements about content, geometry and hierarchy, not pixel-identical typography. Inspection sheets are in [review-evidence](review-evidence/). No screenshots from Dirk were necessary.

I read the PS thesis subsection, its windows figure, D-250 and relevant code, Hefny Sec. 2 and Fig. 2, Venkatraman Secs. 2.1 and 3, and Downey Sec. 4.2. I inspected the supplied meeting pages, the storyboard's superseded and replacement PS instructions, and extracted the figures for the neighbouring step/encoder slides from `partC.pptx`. Reactions are those in the handoff; I cannot reliably assign every chronological reaction to an individual later render. Attempt 03b has no known generating brief. The web session's complete output and the figure-guidelines test figure are absent, so I cannot diagnose those unseen images.

Sources below are local, read-only. Earlier handoffs and storyboard rules are evidence to assess, not authority over the current task.

## A. Diagnosis

### Attempt 01: the deck session

**Generating brief:** storyboard “14b-2”, around line 249. **Evidence:** `attempts/01_deck-session_2026-10-27/psb.tex`, `psb_1..4.png`, `psb_steps.png`. **Reaction:** “not a big fan”; request for a dedicated design session and a clean style.

The brief already fixes almost the entire failed architecture: one timeline, past window, encoder, residual wave, S2 below it, then a crossed-out decoder. It asks for the current residual but prohibits an invented decoder output. Consequently the delivered S1 build has one blue wave and an arrow from `p` into it. There is no visible distinction between the target and the prediction. The audience cannot see what is being fitted, even if every label is read.

Specific problems:

- The top trace is labelled “recorded y, u”, although a position and a force are different signals. Its shape supplies atmosphere rather than an observable mathematical relationship.
- The residual below it has no visible relationship to a measured/model gap. The two traces occupy different vertical positions and scales without a construction connecting them.
- In build 3, the shifted bracket is displaced by 0.233 cm across a 7 cm window. That is approximately the intended 1/30 shift, but it is too small to explain itself. A label announcing “one sample later” carries all the meaning.
- `g_aug` receives a long route labelled `xhat_tau, u_tau`, but the target is only the sentence “to match psi_a(chi_tau+1)”. The target's independent construction never appears.
- Each build introduces several ideas at once. Build 2 adds the entire decoder, residual horizon, prediction claim and conclusion about missing dynamics. Counting four pages did not make this a one-element build sequence.
- The blue concluding sentence substitutes for the missing visual argument. “Must hold ... the missing dynamics” also overstates the result.

The clean type and sparse first page are useful. The problem is not simply excess ink: there is substantial empty space while the essential comparison is absent.

### Attempt 02: the PS visual-design session

**Generating brief:** `attempts/02_ps-visual-design_2026-10-27-28/brief_given.txt`. **Evidence:** all eight TikZ sources and their sheets, plus the Main/Contrast/Stages canvases. **Reactions:** separate and motivate the steps; residual prediction unclear; dislike of the past-samples block; “way too messy”; later “really bad”.

| Output | What it explains, and why it falls short |
|---|---|
| `opt1_window_sheet.png` | Makes the windows explicit, but retains the unconstructed residual wave. The second encoder and S2 wiring arrive underneath an already complete S1 picture. Small sample cells explain indices while the curve is supposed to explain dynamics, so attention has no clear centre. |
| `opt2_block_sheet.png` | Correctly tries to reuse the known augmentation block. However, physical-state correction, baseline transition, concatenation and decoder are simultaneously visible. The two loss equations help but do not make the target's origin visible. Most of the diagram explains the model architecture rather than the PS learning task. |
| `opt3_bridge_sheet.png` | The large grey rectangle literally contains the words “past n+1 samples”. It conceals the data the audience needs to see and creates a large, visually important object without an explanatory role. This is the most concrete match to the complaint about the block of past samples. |
| `s1_sheet.png` | Now shows a target and a dashed predictor, an actual improvement. But the residual is still introduced as a standalone curve; the source `y-yhat` remains absent. A large past block, side legend, decoder label and bottom conclusion compete. |
| `s2_sheet.png` | A scalar added-state trace makes the next sample more tangible. Its target is explicitly encoder-derived and its equation includes the full state and force. This is better than the earlier S2 wiring. But the histories are brackets under the trace rather than visibly shifted recorded data; a scalar illustration also needs to be identified as one coordinate of a vector. |
| `ps1_sheet.png` | Removing the encoder and data makes the curve pair readable but destroys provenance. The S2 state points arrive as a new picture with no visible connection to a shifted measurement window. This is the “too little” side of the problem. |
| `sketch_1..3.png`, `concepts_ABC.png` | A compares indirect output supervision with a direct decoder task, but again only labels the wave. B places sample cells under essentially the same decoder-to-wave story. C is a useful stage overview; it should not be expected to teach either regression on its own. |
| Canvas Main | A residual-like curve is extended into the past and fed into the encoder route. It risks teaching that the encoder sees past model residuals; the implementation uses recorded input/output histories. |
| Canvas Contrast | Clarifies why an auxiliary task is being introduced, but the “without/with” comparison leaves encoder training, target generation and S2 to labels. |
| Canvas Stages | An effective recap, consistent with Dirk accepting this closing layout. It does not establish that the detailed method figures work. Its simplified phase descriptions still need the timing and shared-parameter qualifications below. |

The generating brief contributed three problems. It fixed one slide as the target even while asking for several explanations. It relegated S1/V alternation to backup, allowing a misleading “S1 then S2 then normal training” chronology. And it said “real data only, no invented curves”, although Dirk later explicitly allowed concept curves. A measured residual alone cannot show a fitted decoder without either a real decoder checkpoint or an explicitly illustrative prediction. That tension should have been resolved by distinguishing explanatory sketches from result plots.

### Attempt 03: the PS slide-design session

**Generating brief:** [28 October handoff](../2026-10-28-ps-slide-design.md). **Evidence:** bowtie A1/A2, time-strip B1/B2, two-panel C; all later sheets and real-data candidates. **Reactions:** bowtie preferred but still unclean; residual should resemble meeting slide 13; math missing; next-sample prediction and force dependence unclear; improvement when the timeline idea appeared.

The three initial concepts are not three substantially different explanations. All use `history -> encoder -> state -> future`, with S2 as two encoder routes and a transition between them. A bowtie changes the silhouette, a strip changes the arrangement, and panels change the framing. None changes what the audience learns by looking.

Specific failures and improvements:

- **Bowtie A1:** the measured/model gap is a small upper-right inset; the large lower-right residual has a grey area to its own zero line. These areas mean different things. The layout does not show how the first becomes the second. The decoder triangle gives expansion visual prominence, but does not explain the target or learning signal.
- **A2 and B2:** both show `u_tau`, which is good. But the left state is labelled added state, the transition output is labelled full predicted state, and the target is added state. The equation required by the code is full-state input and added-state output. That dimension distinction is central, not decorative notation.
- **B1:** gives the residual the entire future part of a time strip, but no measured/model subtraction appears. It explains when the target is located without explaining what it is.
- **C:** scales two incomplete explanations into panels. This exacerbates the missing math and gives each sample shift less space.
- **`sheet_R.png`:** samples and measured/model gaps finally make a residual vector visible. However, the text says the model is simulated “with the recorded u_k”. The implementation simulates in its own feedback correction loop. Showing the recorded force as the rollout force is a substantive error.
- **`sheet_S1.png`:** the fan of decoder lines lands on the measured/model gap picture. Because its predicted values are represented there as displacements rather than on a separately defined residual axis, it is easy to read the blue result as another output predictor. The fan also adds many lines without explaining more than one ordinary vector arrow would.
- **`sheet_S2.png`:** the sampled state row is small and occupies a different vertical scale from the large output curve. The visually dominant motion profile is not the variable being predicted. The state arc becomes a tiny annotation to the scene.
- **`s1_all.png` / `S1_builds_sheet.png`:** smoother curves and a single decoder arc improve reading. But the near-flat model line suppresses the very dynamics the explanation is meant to relate. The predictor still appears close to the measured output. Correct formulas below do not fully disambiguate that visual claim.
- **`s2_all.png` / `S2_builds_sheet.png`:** explicitly create adjacent encoded points and show the input in the formula. This is a useful improvement. The remaining weakness is scale: the decisive predicted/target pair is tiny under a large signal trace. Force dependence belongs at that arrow as well as in the formula.
- **`ps4_resid.png`:** a clear, simplified gap illustration, but its almost straight simulated curve makes the residual look like “subtract a trend”. That is a special example, not the method.
- **`ps5_cands.png`:** a useful real-data search sheet, not a presentation figure. **`ps5_real.png`** restores real signal structure, but the enlarged residual is floating above the time axis without a strong scale connection. The caption identifies measured Y and the baseline: that makes it an illustration of subtraction, not evidence that this is a current PS training-window target.
- **`web_examples_sheet.png`:** collects potentially reusable teaching devices but does not resolve this method's distinction between output, residual and added state. A literature-derived silhouette cannot supply the missing causal story.

The handoff's diagnosis in section 6 is partly right about clutter and patching. It is incomplete about the reason. “Too many symbols” does not explain why a figure with fewer symbols also fails. More seriously, its instruction not to open rejected attempts prevented the next session from checking exactly what Dirk objected to. It turned a session's interpretation of feedback into the new session's ground truth.

The handoff's verified-method paragraph says the conditional mean makes the added state predictive, and its guiding question says the state “must carry the missing dynamics”. That omits capacity, optimisation, stochastic predictability and the changing target. It invites drawings of a result the objective encourages rather than ensures.

### Attempt 03b: stage builds

**Generating brief:** unknown. **Evidence:** `Main`, `S1_build2..4`, `S2A_build1..4`, `S2B_build1..4`. **Reactions:** cannot identify a unique reaction to each artboard from the supplied chronology.

This is the strongest supplied attempt, not another equally bad failure.

- S1 build 2 places the residual on its own zero-centred axis aligned with the future interval. Build 3 adds the encoder route and build 4 overlays the decoder prediction there. This substantially resolves the output/residual confusion.
- S2A enlarges the current, target and predicted points. The visual now shows a prediction landing near an independently provided next state. This is exactly the operation that was too small in attempt 03.
- S2B likewise makes the step visible in a state plane, without prescribing a perfect ring. It is a plausible optional illustration if clearly identified as schematic coordinates.

Remaining issues are concrete. The S1 encoder triangle and `p` are labelled trained, but the final loss lacks the minimisation variables and a visible learning route. A bottom claim again says that the best predictor given the past means the state “must carry what the model misses”. S2A's smooth latent trace could be mistaken for a measured signal unless explicitly introduced as an encoder-derived coordinate. S2B's two-dimensional geometry is illustrative, not an established result. The full-state and force arguments occur in the small `g` label, but do not have distinct visible routes into the prediction. The fourth S2 builds add many arrows over the trace/plane; repetition demonstrates a dataset but reduces the hierarchy of the one example.

Keep the independent next-state target and its enlarged comparison. Do not discard them merely because the overall attempt was rejected.

### Attempt 04: all 15 approaches

**Generating brief:** `attempts/04_15-approaches_2026-10-29/brief_given.txt`. **Evidence:** `canvas-source/Main.dc.html` and `A02..A15`. **Reaction:** “failing badly each time”. Its named attachments were never uploaded.

| Artboard | Independent assessment |
|---|---|
| 01, gap cut out | Strongest connection to the accepted residual example. But S2 has crossing encoder routes, vertical force input and a wide prediction arc squeezed into one small panel. The encoded full state needed by `g` is not as visible as its formula. |
| 02, state cloud | Uses a transition from an initial blob to a ring as the explanation of learning. Random encoder rows do not imply all windows map near the origin, and the learned state need not form a ring. The angle `2 pi 212/4000` promotes a physical interpretation that has not been established. |
| 03, training table | A genuinely different and useful device. The next row visibly supplies the target. However, identical grey cells stand for both the full-state input and the smaller added-state target; component selection needs to be explicit. Tiny gap icons leave residual construction elsewhere. |
| 04, Hankel matrix | History windows really do shift through the recorded data. The residual block generally is **not** a Hankel block of one fixed error signal: each `tau` starts a fresh encoded rollout, so `yhat_(k|tau)` differs from `yhat_(k|tau+1)`. A nonlinear bottleneck is also not automatically a linear rank factorisation. This imports subspace-identification mathematics the implementation does not provide. |
| 05, curtain game | A different narrative, but “hide, guess, reveal” is the same information restriction as 01, not a separate mechanism. The three panels multiply curves and state cards, and a curtain uses area without showing the current-model target construction. |
| 06, conditional mean | The most useful new explanatory argument: distinguishing histories can make different futures predictable. But the unsorted average need not be zero, the bins are not an implemented training operation, and the S2 field is drawn in added-state coordinates while the real map also depends on physical state and force. |
| 07, gradient paths | Helpful for checking fixed targets and gradient flow. The many sample-to-state edges make it a wiring figure. It explains which parameters are updated better than why this auxiliary task is useful. Suitable as backup. |
| 08, loss landscape | Unsupported scientific story. There is no measured two-dimensional landscape showing a plateau, escape or common basin, and S1/V steps need not be orthogonal or monotonic. It would teach an optimisation guarantee that is absent. |
| 09, spectrum and poles | Conflates fitting the decoder remainder `e-p(xbar)` with reducing the deployed model error `e`. S2 neither fits poles nor guarantees a pole pair at 212 Hz. The Jacobian with respect to added state alone is also not the complete coupled system linearisation. Reject as a method figure. |
| 10, nested zoom | Correctly recognises a scale problem. Its single-sample enlargement is useful. Three nested scenes, however, repeat the same context and shrink the main regression. A drifting free run is also a different object from the current closed-loop training-window simulation. |
| 11, ring cycle | Distinguishes optimiser iteration from signal time, but its main content is training chronology. The residual and prediction are small icons. Good source for the closing overview, weak main explanation of PS. |
| 12, parameter bar | Useful training-status device, but the bar treats `theta_enc` and its subset `theta_enc^a` as separate additive segments. That double counts unless explicitly partitioned. “V updates the rest” is wrong: V also updates the added encoder rows. It also omits that S2 changes shared hidden layers affecting `f_aug`. |
| 13, waterfall | Makes changing windows visible, but depth creates overlap and mixed axes. `yhat+p(xbar)` looks like the proposed deployed predictor; it also misses the `sigma_e` conversion when `p` predicts the scaled residual. The method discards `p`. |
| 14, phasor | Legible and familiar to controls, but a decoder is not implemented by rotating a hand. The learned latent coordinates need not be displacement/velocity quadratures or turn by 19 degrees per sample. Useful only as a separately labelled oscillator analogy, not as this method's explanation. |
| 15, training time-lapse | Shows a different time variable: optimiser progress. Its smooth improvement and stable encoded ring are invented, and the 19-step autonomous spiral is not the S2 objective. It could become an honest empirical training visual with actual checkpoints and full coupled rollouts, but those are not in this package. |

These are more diverse than the initial three layouts. The criticism should not be that all 15 are literally identical. The failure is that several different devices explain something else, and the most direct ones still miniaturise the two regressions. A quota of 15 rewards novelty over fidelity. The shared tiny residual icon, shared equation block and repeated state cards also impose a common template on supposedly different explanations.

The prompt itself preserves avoidable errors: “two stages before normal training”, “must carry”, and the exact meeting-style start on `y`. It simultaneously asks for every dependency, a complete final S1 and S2, and minimal presentation geometry on each artboard. The references that could have corrected those ambiguities were missing. That explains some failure, but does not excuse proceeding as though the attachments had been inspected or letting false pole/landscape claims survive a method check.

### Why the meeting figures work locally, but do not solve the task

Meeting slide 13 (`reference/.../p29.png`) makes a subtraction understandable: one time coordinate, an output continuing across the boundary, a model beginning there, and a visible gap. It succeeds because the geometry itself explains `y-yhat`. The grey panels, extensive explanatory text and absence of sampled window endpoints are not necessary to that success.

Slide 14 (`p32.png`) has a useful explicit fixed residual target and temporary decoder, but its encoder-row box and loss wiring largely require reading. Its norm does not by itself say which parameters solve the regression. Slide 16 (`p40.png`) makes the shifted-window target explicit, but uses old history endpoints and a text-heavy wiring route. The old deck predates D-250: its windows up to `k` and `k+1` must not be copied into the present `tau-1` and `tau` convention. Its timing slide's fixed 350 updates is also a cap in the current implementation, not an inevitable duration.

Maarten's request for math cannot be met by attaching a long formula to the same ambiguous geometry. Conversely, deleting the formulas reproduces the original problem. Each formula needs to identify the variables already made visible by that build.

### Causes across briefs and sessions

1. **The brief specified objects before specifying the inference.** “Window, state, decoder, wave” repeatedly became a checklist. The unasked question was what a committee member should be able to distinguish after each addition.
2. **A different mathematical object was substituted whenever the real one was awkward to show.** Residual became a waveform, encoded state became an oscillator, S1 learning became output correction, S2 became a small autonomous arrow. These substitutions made pictures easier to produce while changing their claim.
3. **Minimalism was treated as removal.** Removing the target route or force dependency reduces ink but increases the explanation the audience must infer. A short, meaningful branch is often clearer than an unlabeled curve.
4. **Builds were counted as frames rather than comprehension steps.** Adding a decoder, horizon, prediction, loss and conclusion on one frame is several new ideas. Several artboards call these one build.
5. **The briefs amplified certainty.** “Encourages prediction” became “must contain missing dynamics”, although the thesis deliberately revised “forces” to “encourages”. A supposedly verified handoff repeated the stronger claim.
6. **Style and scientific semantics were mixed.** Blue sometimes means new, sometimes trainable, sometimes predicted. Grey sometimes means fixed, sometimes merely less important, sometimes a large decorative panel. Those meanings conflict within the same figure.
7. **Feedback was flattened into prohibitions.** “No past samples block” became a ban on useful context; dislike of an entire figure became a reason to discard its successful residual or zoom. “Start fresh” then blocked inspection of the actual evidence.

## B. What a correct figure must make understood

### Ground truth and discrepancies that affect the drawing

The code and the final, unstruck thesis method agree on the essential windows and regression targets:

\[
\chi_\tau=(u^{\rm n},y^{\rm n})_{\tau-30:\tau-1},\quad
\hat x_{\tau|\tau}^{\rm n}=\psi(\chi_\tau),\quad
\bar x_{\tau|\tau}=\psi_a(\chi_\tau).
\]

\[
e_\tau=\frac1{\sigma_e}\operatorname{vec}
\left(y_k^{\rm n}-\hat y_{k|\tau}^{\rm n}\right)_{k=\tau}^{\tau+H-1},
\qquad H=30.
\]

The encoder consumes recorded histories. The **current augmented model**, initialised from its full encoded state, makes the closed-loop rollout used to construct the target. In the residual-form implementation, the plant-model input is recorded force plus a feedback correction based on measured/model output discrepancy and controller state. It is not simply recorded-force replay. The decoder consumes only added states and produces the entire `H*n_y` residual vector in one forward pass. There is no direct force or future-data input to `p`.

S1 trains `theta_p` and `theta_enc^a`, with the target detached for that update. The next V update can change the target. S2 caches full current states, recorded current forces and next added states, then fits the added-state outputs of the network on those fixed pairs. V pauses during that fit.

Material qualifications and disagreements:

- **Shared parameters:** “S2 trains g_aug only” is acceptable as a statement about the supervised output, but false if it means other functions cannot change. The code optimises the augmentation network; shared hidden layers also change `f_aug`. Its physical output-layer rows are not supervised by the S2 loss. S1's own update changes only added encoder rows and decoder; V also updates the shared model parameters between S1 steps.
- **Loss scaling:** S1's minibatch/horizon/output average is a constant factor within its loss. S2 divides by target variance *within each minibatch*. The code uses sample variance, giving `(B-1)/B` times the thesis quotient for a fixed batch. That constant does not change that batch's minimiser. Dropping the entire variance denominator is a slide simplification, not the exact training procedure: batch-dependent weighting changes the aggregate stochastic objective. The sketches retain a short weight `w_B` to avoid silently claiming equivalence.
- **Conditional mean:** for a fixed encoder and target distribution the unrestricted optimal decoder is `E[e|xbar]`. Matching `E[e|chi]` requires the representation to retain the necessary information, sufficient decoder capacity and successful optimisation. Finite-dimensional, alternating, nonlinear training does not guarantee this. Nor does it guarantee sufficient state for arbitrary future force sequences or physical state identification.
- **Start on y:** D-250 prevents the encoder from using `y_tau`. Therefore `yhat_(tau|tau)=y_tau` is not enforced. Starting the illustration with coincident curves is an optional schematic example, not a method property. A small initial gap is valid. No output should be drawn as a copy of measured `y_tau` fed into the initial state.
- **Historical text:** struck or moved passages in the thesis still contain old decoder names and an “up to k+1” history description. Those are revision history, not a competing current algorithm. The current equation, figure, D-250 and `_sample` resolve the endpoints consistently.
- **Real curves:** `res_time_trace.csv` has `t_s,r_Y,y_Y,baseline_Y,final_Y,zoom,status_final`; no record column. Its plotting script identifies I1. The checked time step is 0.00025 s. `y_Y-baseline_Y` is a legitimate example of a model residual, but is not a regenerated rollout from an arbitrary PS window or the current training iterate. Normalise it and label its provenance if used. None of my illustrative curves is presented as this data.
- **Frequency claim in the 28 October handoff:** at 4 kHz and H=30 the unpadded DFT spacing is 133.3 Hz. The reported “dominant FFT bin ... 196 Hz” cannot follow from those settings alone. Zero-padding or another record length would need explanation. A 7.5 ms window covers about 1.59 periods at 212 Hz, but this does not establish a measured 212 Hz peak. Keep frequency-identification evidence on the result slide.
- **Timing:** 30 samples correspond to a 7.5 ms horizon convention; first-to-last sample coordinates span 29 intervals, 7.25 ms. A one-sample S2 step is 0.25 ms, not the whole S1 horizon.

### Criteria and a one-minute comprehension check

| ID | Figure requirement | Grounding and what the audience should be able to say |
|---|---|---|
| C1 | Construct residual from measured output and the current model on a common time axis. | Thesis `eq:ps2_s1`, `_residual`; Dirk's meeting-slide-13 preference. “Those differences are the target.” |
| C2 | Distinguish the H-step residual vector from the one-step added-state target. | `_sample(pair=False/True)`, both objectives; Dirk's next-sample request. “S1 predicts 30 residual values; S2 predicts one next state vector.” |
| C3 | Show recorded `(u,y)` history, its endpoint at `tau-1`, and the information restriction through `xbar` into `p`. | D-250, `_s1_step`; dislike of a box of words and request to show decoder data. “The decoder gets the added state, not the future.” |
| C4 | Give target and prediction independent visual identities and an explicit comparison. | Both squared-error objectives; request to show residual prediction. “The target is constructed from data/model; the dashed curve is the decoder output.” |
| C5 | Make the learning route and its minimisation variables explicit. | `_s1_step`, thesis `ps2_s1`; Maarten's math feedback. “The residual loss trains p and the added encoder rows.” |
| C6 | Show that S1 creates a predictive representation, without inventing physical coordinates. | Thesis's “encourages”, Venkatraman Sec. 3. “Predictable differences in future error give the representation a training signal.” |
| C7 | Derive the next-state target from the same frozen encoder on the one-sample shifted history. | `_fit_added_state_transition`, D-250, thesis `ps2_s2`. “The target is inferred from a window ending at tau.” |
| C8 | Put full current state and recorded `u_tau` into the S2 transition visibly. | Actual `ann(cat(X,U))`; Dirk's concern about input dependence. “The next added state depends on the physical states and input as well.” |
| C9 | Distinguish within-step fixed targets, frozen S2 encoder, S1/V alternation and hand-over. | `after_step`, `ps2_alt`, `ps2_handover`. “Targets are fixed for each regression update; the decoder is temporary; ordinary training continues.” |
| C10 | Each build adds one semantic object/relation at fixed coordinates, with a matching concise equation. | Dirk's repeated acceptance of stepwise builds and formulas. One principal visual task per slide; blue means newly introduced, and explicit labels carry fixed/trained status. |
| C11 | Preserve sample scale and signal identity. | House-style scale rule, D-250, normalisation requirement. No interchangeable `xhat`/`xbar`, force/position composite trace, arbitrary magnified shift or unlabelled “real” data. |

About one minute per S1/S2 slide is reasonable if the encoder and model step have already been introduced, as in the neighbouring deck. It is not reasonable to also derive controller residual form, conditional-expectation theory, minibatch normalisation, stopping rules and shared-layer effects in those two minutes. Keep those qualifications in notes or backup, but do not draw something that contradicts them. A brief S1/V timing label is enough; the stopping threshold need not be on the main figure.

Hefny Fig. 2 is useful for disjoint history/future indexing and the idea that stage-one estimates supply stage-two training data. Its S2 maps predicted future features to extended-future features, not this code's controlled nonlinear next-state map. Venkatraman supplies the temporary decoding supervision idea; its combined regularised recurrent loss is not the present two-stage procedure. Downey supplies the precedent of two-stage initialisation followed by BPTT, not a guarantee for this residual-target adaptation. Cite those specific contributions rather than presenting a borrowed figure as the algorithm.

## C. Concepts

The sketches are deliberately explanatory drawings, not candidate experimental results. All use LaTeX serif math, black/white, meaningful fixed-route grey and one highlight colour. [Six final figures, PDF](sketches/ps-concepts.pdf), [editable TikZ](sketches/ps-concepts.tex), and [concept 1 builds, PDF](sketches/concept-1-builds.pdf). Each PDF page is 16:9. PNG previews are `sketches/concept-1.png` through `concept-6.png`; PDF pages 1/2, 3/4 and 5/6 are the three S1/S2 pairs.

For concise formulas below let `X_tau=psi(chi_tau)`, `a_tau=psi_a(chi_tau)`, and `T_tau=psi_a(chi_tau+1)`. These are shorthand in this review, not proposed replacement notation on the slides. State superscripts n can be suppressed once normalisation is declared. Let

\[
L_1=\sum_{\tau\in\mathcal B}\|p(\psi_a(\chi_\tau))-e_\tau\|^2,
\qquad
L_2=\sum_{\tau\in\mathcal B_{\mathcal P}}w_{\mathcal B_{\mathcal P}}
\|g_{\rm aug}(X_\tau,u_\tau)-T_\tau\|^2,
\]

where `w_B` is the reciprocal summed centred target energy in the thesis quotient. It is fixed for each frozen-target minibatch; the notes can retain the exact code's variance convention. Show the min variables with each final objective. Do not explain the weight verbally during the main talk.

### Concept 1: construct one training example

**Idea:** show the target being constructed independently, then show a learned prediction trying to match it.

**Final sketch:** pages 1 and 2 of the concept PDF. On S1, a measured/model timeline establishes the gap. Below, a separately centred residual trace receives both the fixed target construction and the decoder prediction. The decoder only receives the added-state branch. On S2, two shifted data windows supply two fixed encoder evaluations; a lower route takes the full current state and force into `g_aug`, and ends opposite the next added-state target. Target and prediction have separate endpoints.

The S1 plot uses a small nonzero initial discrepancy deliberately: matching the measurement at `tau` is not enforced. The top and lower plots have different declared normalisations. The lower trace is a residual, never a corrected output. A signal connection and its attached symbol constitute one element; a build does not add a second explanatory mechanism.

**S1 build order and accompanying equation** (rendered as eight pages):

| Build | One new element/relation | Equation introduced or retained |
|---|---|---|
| 1 | Recorded history and marked split; the already familiar encoder window is context. | `chi_tau=(u,y)_(tau-30:tau-1)` |
| 2 | Current-model rollout route and its output, using the encoded full state and known controller. | `X_tau=psi(chi_tau)`; controller dependence is on the route label. |
| 3 | Residual target on its own aligned zero-centred axis. | `e_tau=vec(y^n-yhat^n)_(tau:tau+H-1|tau)/sigma_e` |
| 4 | Added-state branch from the history. | `a_tau=psi_a(chi_tau)` |
| 5 | Temporary decoder attached to that branch. | `p:R^(n_a) -> R^(H n_y)` |
| 6 | Decoder's predicted residual over the target. | `ehat_tau=p(a_tau)` |
| 7 | Comparison objective. | `min_(theta_p,theta_enc^a) L1` |
| 8 | Backward learning route through decoder to added encoder rows. | Retain the objective; highlight `theta_p,theta_enc^a`. |

The residual builds can be placed on a preceding R slide, leaving builds 4 to 8 as the S1 slide. This does not demand eight novel ideas in one minute. Positions stay fixed within each slide; the completed R construction is reused as context.

**S2 build order and accompanying equation** (eight rendered pages):

| Build | One new element/relation | Equation introduced or retained |
|---|---|---|
| 1 | Current history on sampled data. | `chi_tau=(u,y)_(tau-30:tau-1)` |
| 2 | Shift that history exactly one sample. | `chi_tau+1=(u,y)_(tau-29:tau)` |
| 3 | Frozen encoder route supplying the next added-state target. | `T_tau=psi_a(chi_tau+1)` |
| 4 | Frozen full-state encoder route supplying the predictor input. | `X_tau=psi(chi_tau)` |
| 5 | Recorded current input entering the prediction route. | `u_tau=u_tau^recorded` |
| 6 | One-step transition and its predicted added state. | `abar_tau+1|tau^pred=g_aug(X_tau,u_tau)` |
| 7 | Comparison between that prediction and independently encoded target. | `min_(theta_aug) L2` |
| 8 | Learning route; fixed/trained status stated. | Retain objective; `theta_enc` fixed, `theta_aug` trained, `V` paused. |

Use the history context as the opening frame, not as a fresh explanation of the encoder. Eight builds are a detailed review sequence; a presentation can start from already established context, while preserving the one-new-relation rule for subsequent builds.

**Criteria:** strongest on C1 to C5, C7 and C8; the notes and timing label satisfy C9; fixed placements and enlarged sample pitch support C10/C11. **Weakness:** C6 is still principally an information-bottleneck argument accompanying the picture. It shows *how* the state is trained better than it proves *why* that helps. It also repeats the residual in two coordinate systems, costing space. Keep the upper trace as established context after R rather than adding more labels.

### Concept 2: the dataset is the explanation

**Idea:** make one example per row, so S1's input/target pairing and S2's next-row target selection are visible operations on a dataset.

**Final sketch:** pages 3 and 4. S1 has three rows of history snippets, their added states, and future residual snippets with decoder predictions. They are arbitrary examples A/B/C, not asserted shifts of one residual trace. S2 explicitly switches to consecutive cached encoded rows. Each row contains the full current state and force; the added component from the following row supplies its target.

This refines attempt 04's training table. It is a distinct explanatory idea from concept 1: repeated supervised examples and row selection replace the single-example construction narrative. It does not introduce a Hankel assumption. A small residual-construction opening is still required; a table cannot define a difference between outputs by itself.

**S1 builds:**

| Build | One new element/relation | Equation |
|---|---|---|
| 1 | One measured/model gap example in the opening frame. | `e_tau=vec(y^n-yhat^n)/sigma_e` over the stated horizon. |
| 2 | Place the associated past snippet and residual in one aligned training row. | `D1={(chi_tau,e_tau)}` |
| 3 | Insert that row's added-state representation. | `a_tau=psi_a(chi_tau)` |
| 4 | Add the shared decoder prediction over the target. | `ehat_tau=p(a_tau)` |
| 5 | Add a second row, showing another example of the same mapping. | Retain `D1`, add its second member. |
| 6 | Add the third row. | Retain `D1`, add its third member. |
| 7 | Add shared learning through the row comparisons. | `min_(theta_p,theta_enc^a) L1` |

**S2 builds:**

| Build | One new element/relation | Equation |
|---|---|---|
| 1 | Cached consecutive full-state rows, explicitly partitioned into physical/added components. | `X_k=psi(chi_k)=[xtilde_k;abar_k]`, encoder fixed. |
| 2 | Make the one-sample data shift explicit beside the rows. | `chi_k+1=(u,y)_(k-29:k)` |
| 3 | Copy the next row's added component into the current row's target column. | `T_k=psi_a(chi_k+1)` |
| 4 | Add the recorded-input column. | `P={(X_k,u_k,T_k)}` |
| 5 | Fill the prediction column using the same transition for every row. | `abar_k+1|k^pred=g_aug(X_k,u_k)` |
| 6 | Compare the two columns and train the transition. | `min_(theta_aug) L2` |

**Criteria:** strongest on C2/C3/C5/C7/C8 and decoder training-data provenance. C9 can be attached to cached/fixed column labels. **Weaknesses:** C1 relies on an opening illustration or previous slide; C6 remains mostly verbal. Diagonal next-row leaders can compete with the actual regression route, and repeated rows make the samples small. This is a good backup if “where did these targets come from?” is the main committee question. Do not display the full row table and a full timeline simultaneously.

### Concept 3: explain why a state must distinguish histories

**Idea:** two histories with different predictable future residuals cannot be served by one identical added state, while an encoder-derived state sequence still needs a separately learned transition.

**Final sketch:** pages 5 and 6. Two measured histories lead to different conditional mean residual profiles. The same decoder appears in both routes. An initially shared representation would force identical predictions, visibly incompatible with both profiles. S2 then shows two adjacent windows with independently encoded states and an initially missing connection; the full-state/force-dependent transition fills that connection.

This develops the useful argument in attempt 04/06 without asserting a zero unconditional mean, hand-made state clusters, a circular field or learned physical phase. It is conceptually different from showing data rows: the visual argument is a counterexample to an uninformative state.

**S1 builds:**

| Build | One new element/relation | Equation |
|---|---|---|
| 1 | Define the model residual, using the familiar gap opening. | `e_tau=vec(y^n-yhat^n)/sigma_e` |
| 2 | First history and its predictable residual profile. | `mu_A=E[e|chi_A]` |
| 3 | Second history and a different predictable profile. | `mu_B=E[e|chi_B]`, `mu_A != mu_B` |
| 4 | Show both histories mapped to the same candidate added state. | `psi_a(chi_A)=psi_a(chi_B)=a` |
| 5 | Show the consequence for their shared decoder predictions. | `p(a_A)=p(a_B)` |
| 6 | Attach the training loss, encouraging the encoder to distinguish useful histories. | `min_(theta_p,theta_enc^a) L1` |

Final drawing labels the two state outputs separately; the equality line states the counterfactual. No claim is made that the optimiser necessarily resolves it or fits both means exactly.

**S2 builds:**

| Build | One new element/relation | Equation |
|---|---|---|
| 1 | Current window and full encoded state. | `X_tau=psi(chi_tau)` |
| 2 | Shifted window and next added-state target from the frozen encoder. | `T_tau=psi_a(chi_tau+1)` |
| 3 | Mark the missing transition relation between these estimates. | Retain `X_tau,T_tau`; S1's loss contains no `g_aug` term. |
| 4 | Add the recorded current input. | `u_tau=u_tau^recorded` |
| 5 | Add the learned prediction route. | `abar_tau+1|tau^pred=g_aug(X_tau,u_tau)` |
| 6 | Fit that route against the independent target. | `min_(theta_aug) L2`, encoder fixed and V paused. |

**Criteria:** strongest on C6 and the distinct reasons for S1/S2; also supports C3 to C5 and C7/C8. **Weaknesses:** conditional means are conceptual objects, not the sampled targets used directly in code. The caption must say that the two future profiles represent predictable differences, not two arbitrary noisy outcomes. It is more abstract for an audience unfamiliar with conditional prediction, and needs the preceding residual illustration for C1. It should not be sold as a proof of global training success.

### Choice and presentation order

Choose concept 1. Dirk's most persistent unresolved requests concern the residual, decoder training data, one-sample shift and actual regression. Independent target/prediction routes answer all four without assuming an oscillator or an optimisation landscape. The S2 drawing deliberately trades an attractive latent curve for a larger, unambiguous target comparison. If desired, add one isolated next-state point comparison from 03b as a later visual refinement, keeping the full-state/force inputs visible.

Use **R -> S1 -> S2 -> training overview**. R defines the measured/model subtraction. S1 reuses it and adds the decoder task. S2 changes the target explicitly from a future residual vector to a next added-state vector. The overview states “during S1: alternate V and residual updates; S2: fit fixed pairs with V paused; afterwards: discard p and continue V”. The overview is a recap, not a third PS regression stage. The residual-predictability screening/skipped-S2 branch can remain in backup.

The design task is not fundamentally underspecified. What cannot be achieved honestly is a picture that makes “the learned added states are the physical 212 Hz oscillator, and PS must recover it” obvious: the method does not establish that statement. A clean figure can make the two supervised tasks and their motivation understood. The result slide must supply evidence that this initialisation helps recover the mode in this experiment.

## Files and verification

- `sketches/ps-concepts.tex` / `.pdf`: three concept pairs, six final pages, editable TikZ.
- `sketches/concept-1-builds.tex` / `.pdf`: 8 S1 and 8 S2 frames at fixed coordinates.
- `sketches/concept-1.png` through `concept-6.png`: final previews.
- `generate-sketches.py`: regenerates the TikZ sources, with all data explicitly schematic.
- `review-evidence/`: inspection reconstructions and contact sheets, not proposed deck assets.

Both TikZ documents compiled successfully. The final concept PDF has six 16:9 pages; the build PDF has sixteen. These are review sketches, not approved production slides. The thesis, method code, rejected-attempt files and neighbouring deck were not edited.

I visually checked the six final figures and the sixteen-frame contact sheet, and checked extracted text bounds on all 22 PDF pages. No text lies outside a page; the LaTeX logs contain no overfull-box warnings. The compact alternatives have written build specifications rather than separately rendered build sequences. The next refinement should reduce repeated context and captions for presentation placement, not add more mechanisms to the figures.

### Primary source locations

- [Thesis method](../../../Thesis-writeup/Writing/sections/03_augmentation.tex), `sec:aug_ps2`, equations `ps2_s1`, `ps2_alt`, `ps2_s2`, `ps2_handover`.
- [Implementation](../../../Clean/model_augmentation/fit_systems/ps2_opening.py), `_sample`, `_residual`, `_s1_step`, `_fit_added_state_transition`; [closed-loop input construction](../../../Clean/model_augmentation/fit_systems/closed_loop.py), `closed_loop_step`.
- [Decisions](../../../docs/decisions.md), D-250 and D-260; [thesis windows figure](../../../Thesis-writeup/Writing/figures/tex/ps2_windows.tex); [house style](../../../docs/writeup/figure-style.md).
- [Hefny et al. 2015](../../../Thesis-writeup/References/Hefny-2015-Supervised-learning-dynamical-systems.pdf), Sec. 2 and Fig. 2; [Venkatraman et al. 2017](../../../Thesis-writeup/References/Venkatraman-2017-Predictive-state-decoders.pdf), Secs. 2.1 and 3; [Downey et al. 2017](../../../literature/thesis-referenced/downey2017psrnn.pdf), Sec. 4.2.
