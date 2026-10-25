# Citation evidence review — Dynamic Augmentation of the Physics Baseline

Reviewed against the current local `03_augmentation.tex` and local PDFs on 25 October 2026. **All occurrences remain unverified pending your confirmation.** “Supporting” below means that I located relevant evidence; it does not mean you have confirmed it.

Scope: **39 retained citation markers, nine sources**, presented in **38 occurrence blocks** because H10 covers two adjacent markers for the same claim. Deleted wording, comments, TODOs and draft-only blocks are excluded. No thesis text has been edited.

The thesis quotations below give the **retained wording**: tracking/formatting wrappers and citation commands are omitted; bracketed source labels replace citation markers where needed to preserve the sentence's meaning. Mathematical LaTeX and internal cross-reference commands are retained where useful. Each line link opens the exact current source, including its original formatting. Source line wrapping is removed in quotations; line-end word hyphenation is joined. Mathematical symbols are transcribed for readability. Search for the ordinary words at the start of each paper quote if mathematical symbols interfere with PDF search.

Every quoted passage or equation below is a **located passage**, not a suggested search location. Where a passage does not support part of the thesis wording, that gap is stated explicitly. No uninspected search suggestion is presented as supporting evidence.

## Points to discuss before confirming

| Occurrence | What needs attention |
| --- | --- |
| H3 | Table 1 supports the dynamic-parallel structure, but does not give the force-based explanation for correcting positions. |
| H5 | The zero-mean qualification concerns the baseline transformation in Eq. (28); the source also explicitly centres the data. |
| E3 | Local linearisation is required by the reconstruction method discussed there, not by every possible nonlinear reconstruction method. |
| G1 | The cited OBC result concerns additive input–output models; applying it to your dynamic state-space model needs your own construction. |
| E5–E6 | Eq. (33) is a linear fit for an LTI baseline. Your affine fit for a nonlinear baseline is an adaptation, already distinguished in the following text. |
| H10 / S5–S6 | Distinguish assumptions underlying the cited consistency results from properties proved for your modified closed-loop method. |
| P3 | The figure is adapted from Hefny's temporal-window illustration; the arrangements and targets differ. |
| S3 | Separate prose issue: the current retained sentence begins “keeps” without a subject. |

No incorrect equation, section, remark, condition or figure **numbers** were found among the retained citations reviewed here. Correct locators do not establish that all accompanying claims are fully supported. This review does not independently prove the thesis's own model equations, stability, implementation equivalence or identifiability.

## Source 1: `hoekstra2026lfr`

[Open local PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

### H1 — [thesis line 305](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:305>)

**Retained thesis wording**

> The augmented model updates these states using the baseline states, the added states and the actuator forces.

**Located paper evidence — PDF page 3 · printed page 3 · Section 2, Table 1, dynamic parallel (S-DP) row** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> Classes of state model augmentation structures.

**Comparison:** Supporting equation inspected: the added-state update is g_aug(x_b,k, x_a,k, u_k). This directly supports the three argument groups. Interpreting u as actuator forces is your gantry-specific signal definition, not a claim that this paper studies actuator forces.

Source equation, transcribed from Table 1:
\[
x_{\mathrm a,k+1}=g_{\mathrm{aug}}(x_{\mathrm b,k},x_{\mathrm a,k},u_k).
\]

**Status: unverified — awaiting your confirmation.**

### H2 — [thesis line 311](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:311>)

**Retained thesis wording**

> We use the dynamic parallel structure of Hoekstra et al.

**Located paper evidence — PDF page 3 · printed page 3 · Table 1, dynamic parallel (S-DP) row** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> dynamic parallel (S-DP)

**Comparison:** The table directly supplies the additive baseline-state correction and separate added-state update used in your model. Your unchanged output map is a further modelling choice; Table 1 classifies state augmentation.

Source equations, transcribed:
\[
x_{\mathrm b,k+1}=f_{\mathrm{base}}(x_{\mathrm b,k},u_k)+f_{\mathrm{aug}}(x_{\mathrm b,k},x_{\mathrm a,k},u_k),\qquad
x_{\mathrm a,k+1}=g_{\mathrm{aug}}(x_{\mathrm b,k},x_{\mathrm a,k},u_k).
\]

**Status: unverified — awaiting your confirmation.**

### H3 — [thesis line 500](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:500>)

**Retained thesis wording**

> The positions are also corrected, because within one sample period an unmodelled force would change both the velocity and the position.

**Located paper evidence — PDF page 3 · printed page 3 · Table 1, dynamic parallel (S-DP) row** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> dynamic parallel (S-DP)

**Comparison:** Partial support. Table 1 permits correction of the baseline state vector, including position components when these are baseline states. It does not state the force/discretisation rationale in your sentence. That rationale is your own argument about the sampled mechanical dynamics. The table should not be presented as its textual evidence.

Minimal suggestion, not applied: “The positions are also corrected **as in the dynamic parallel structure** [Table 1], because within one sample period an unmodelled force would change both the velocity and the position.” The precise physical claim depends on the discrete-time integration used; for your integrated mechanical baseline, an acceleration perturbation generally changes position as well as velocity over a sampling interval.

**Status: unverified — awaiting your confirmation.**

### H4 — [thesis line 507](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:507>)

**Retained thesis wording**

> The inputs to the neural network are normalised to zero mean and unit variance, which has been shown to improve the estimation of models containing neural networks, as in [Hoekstra, Sec. 5.3].

**Located paper evidence — PDF page 9 · printed page 9 · Section 5.3, opening paragraph** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> For the estimation of model structures containing ANNs, normalisation of the input and output data to zero mean and to a standard deviation of 1 has been shown to improve model estimation [6].

**Comparison:** Directly supports the normalisation practice. Unit standard deviation and unit variance are equivalent here. Their reference [6] is Bishop; the sentence is a background recommendation, not a guarantee that every normalised model improves. See the separate Bishop occurrence below.

**Status: unverified — awaiting your confirmation.**

### H5 — [thesis line 656](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:656>)

**Retained thesis wording**

> We include the means in $f_{\mathrm{base}}^{\mathrm n}$, since we do not assume zero-mean signals as in [Hoekstra, Sec. 5.3].

**Located paper evidence — PDF page 9 · printed page 9 · Section 5.3, immediately before Eq. (28)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> For a baseline model operating around zero mean x̂, u, ŷ, this can be achieved by the transformation

**Comparison:** The locator is correct, but its scope is the scaling-only baseline transformation in Eq. (28). The same section explicitly normalises input/output data to zero mean. Saying that the paper assumes zero-mean signals without mentioning the baseline transformation can obscure that distinction.

Minimal suggestion, not applied: “We include the means in $f_{\mathrm{base}}^{\mathrm n}$, since we do not assume **a baseline model operating around zero mean**, as in [Sec. 5.3].”

**Status: unverified — awaiting your confirmation.**

### H6 — [thesis line 729](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:729>)

**Retained thesis wording**

> The model is simulated in closed-loop with the controller $K_{\mathrm{fb}}$ of \eqref{eq:system_loop}, as in Kessels, instead of using the recorded input as in [Hoekstra, Eq. (22)].

**Located paper evidence — PDF page 8 · printed page 8 · Section 5.1, Eq. (22b) and Figure 6** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> SUBNET structure: the subspace encoder ψθ estimates the initial state at time k based on past inputs and outputs, then it is propagated through φθ multiple times until a simulation length T.

**Comparison:** Equation (22b) was inspected: its rollout receives u_(k_i+ℓ), the input sequence from the dataset, rather than an input recomputed by simulating a feedback controller. Your comparison is supported. This concerns simulation during fitting, not a claim that every dataset in the paper must have been measured without feedback.

**Status: unverified — awaiting your confirmation.**

### H7 — [thesis line 1127](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1127>)

**Retained thesis wording**

> The network parameters $\theta_{\mathrm{aug}}$, the encoder parameters $\theta_{\mathrm{enc}}$ and $\theta_{\mathrm{base}}$ are jointly estimated, as in [Hoekstra, Eq. (22)], while the encoder paper keeps $\theta_{\mathrm{base}}$ fixed.

**Located paper evidence — PDF page 8 · printed page 8 · Section 5.1, Eq. (22) and paragraph immediately below** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> is the joint parameter vector

**Comparison:** Direct support from the actual parameter definition: θ = col(θ_base, θ_aug, θ_LFR, θ_encoder). These parameter groups are estimated together. The source also includes its LFR parameters; your sentence lists the parameter groups used by your model. The fixed-baseline contrast is checked under the encoder paper below.

**Status: unverified — awaiting your confirmation.**

### H8 — [thesis line 1130](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1130>)

**Retained thesis wording**

> The parameters $\vartheta=(\beta,\theta_{\mathrm{aug}},\theta_{\mathrm{enc}})$ are trained to minimise the normalised output error of the closed-loop simulation, with the known controller $K_{\mathrm{fb}}$. We adapt this from [Hoekstra, Eq. (22)] for the closed loop of Section~\ref{sec:aug_loop} and the encoder:

**Located paper evidence — PDF page 8 · printed page 8 · Section 5.1, Eq. (22a–e)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> The initial state of these subsections is estimated by an encoder function ψ using past input-output data,

**Comparison:** The equation supplies truncated output-error minimisation, joint estimation and encoder initialisation. It does not supply your simulated feedback controller or your particular normalisation/parameterisation. Your explicit “We adapt” correctly distinguishes those modifications. Eq. (22a) is the mean squared output error; Eq. (22b) uses the recorded input.

**Status: unverified — awaiting your confirmation.**

### H9 — [thesis line 1179](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1179>)

**Retained thesis wording**

> In \eqref{eq:aug_loss}, the encoder $\psi$ uses the samples up to $\tau-1$, as in [Hoekstra, Eq. (22e)]:

**Located paper evidence — PDF page 8 · printed page 8 · Eq. (22e), plus its following explanation** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> using past input-output data

**Comparison:** Direct support. Both histories in Eq. (22e) end at k−1. Your histories end at τ−1. Your lower endpoint τ−n−1 gives n+1 past samples, whereas the paper separately denotes the two history lengths by n_a and n_b; the citation supports the endpoint convention, not equality of every history-length symbol.

Source equation, transcribed:
\[
\hat x_{k_i|k_i}=\psi\!\left(\theta_{\mathrm{encoder}},
y_{k_i-n_{\mathrm a}}^{k_i-1},
u_{k_i-n_{\mathrm b}}^{k_i-1}\right).
\]

**Status: unverified — awaiting your confirmation.**

### H10 — [thesis line 1248](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1248>)

**Retained thesis wording**

> The identification method we have adapted from [Hoekstra] relies on these SUBNET assumptions [Hoekstra, Sec. 5.5].

**Located paper evidence — PDF page 10 · printed page 10 · Section 5.5, opening paragraph** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> Under Conditions 2.1–2.4 in [5] on the data-generating process (1), model structure (6), and identification criterion (22), the convergence and consistency are proven.

**Comparison:** This covers both adjacent Hoekstra citation markers at lines 1248–1249. The section expressly inherits SUBNET convergence/consistency conditions and names incremental exponential output stability. Input/noise independence is directly stated in Beintema Section 2.1, rather than repeated explicitly here. Treat these as assumptions of the stated guarantees. Simulating your model in closed loop does not by itself prove that those guarantees apply to your modified method.

Further exact passage: “Condition 2.1 requires the data-generating system to be incrementally exponential output stable, which becomes an assumption on the considered system (1).”

**Status: unverified — awaiting your confirmation.**

### H11 — [thesis line 1275](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1275>)

**Retained thesis wording**

> The baseline parameters and the neural network can both correct the same model errors, making the optimal parameter values not unique.

**Located paper evidence — PDF page 9 · printed page 9 · Section 5.2, opening continuation** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> Therefore, the joint identification of all parameters can result in the learning components representing or cancelling out part of the baseline model dynamics.

**Comparison:** Direct conceptual support for overlapping representations and non-unique parameterisations. This passage is not a separate proof of the exact non-uniqueness properties of your gantry parameterisation.

**Status: unverified — awaiting your confirmation.**

### H12 — [thesis line 1361](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1361>)

**Retained thesis wording**

> A fully random initialisation can make the model unstable and lead to poor optimisation results. We therefore start training from a model that behaves as the baseline.

**Located paper evidence — PDF page 9 · printed page 9 · Section 5.4 and Section 5.4.1** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> However, for baseline models, random initialisation can be unstable and result in poor optimisation results.

**Comparison:** Direct support for the motivation and strategy. This is a risk statement (“can”), not a claim that every random initialisation is unstable.

Further exact passage, Sec. 5.4.1: “We propose to initialise the parameters θ so that the LFR-based model structure with the encoder behaves equivalent to the baseline model on initialisation.”

**Status: unverified — awaiting your confirmation.**

### H13 — [thesis line 1377](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1377>)

**Retained thesis wording**

> Hoekstra et al. therefore suggest fitting the encoder to states simulated with the baseline, as their augmentation method does.

**Located paper evidence — PDF page 9 · printed page 9 · Section 5.4.2, Eq. (30); baseline-state generation also described in Section 5.3** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> where x̂b,k is the forward simulated state of the baseline model in the extended dataset D̂N.

**Comparison:** Eq. (30) fits the baseline encoder to simulated baseline states. Correct locator. It is the encoder-initialisation objective, not the main joint-training objective. Your closed-loop data generation and choice to fit only a linear physical map are adaptations, discussed separately below.

**Status: unverified — awaiting your confirmation.**

### H14 — [thesis line 1402](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1402>)

**Retained thesis wording**

> We use the closed-loop input from \eqref{eq:aug_enc_sim}, instead of the recorded input as in [Hoekstra, Eq. (30)].

**Located paper evidence — PDF page 9 · printed page 9 · Section 5.4.2, Eq. (30), with Section 5.3 for dataset generation** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> For this, we simulate the baseline model with θbase0 for the given input sequence resulting in x̂b,k.

**Comparison:** The displayed Eq. (30) uses the input history u and simulated output history ŷ from the extended dataset. The Section 5.3 passage establishes that the baseline simulation uses the given input sequence. Supported with that surrounding definition; Eq. (30) alone is an encoder fit, not a controller simulation.

**Status: unverified — awaiting your confirmation.**

### H15 — [thesis line 1533](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1533>)

**Retained thesis wording**

> We use Xavier initialisation for $W^a$, following [Hoekstra, Eq. (31)]. We choose the uniform distribution with gain one.

**Located paper evidence — PDF page 9 · printed page 9 · Section 5.4.2, text immediately after Eq. (31)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> where the weights and biases of ψaug are initialised by the Xavier approach.

**Comparison:** The reference number is correct; the actual Xavier instruction is in the prose directly after Eq. (31). It supports Xavier initialisation of the added-state encoder. The uniform variant and gain one are your explicit choices; this passage does not prescribe them. It also does not prescribe the different 1/√m distribution mentioned for other layers.

**Status: unverified — awaiting your confirmation.**

### H16 — [thesis line 1538](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1538>)

**Retained thesis wording**

> At the start of training, $\theta_{\mathrm{base}}=\theta^0_{\mathrm{base}}$ and $f^{\mathrm n}_{\mathrm{aug}}=0$. The augmented model and baseline model give the same $\tilde x$ and $\hat y$, given they start from the same physical state. The added states $\bar x$ are still updated by $g^{\mathrm n}_{\mathrm{aug}}$, as in [Hoekstra, Eq. (29)].

**Located paper evidence — PDF page 9 · printed page 9 · Section 5.4.1, Eq. (29)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf>)

> Then the initialisation of θ should realise the following behaviour

**Comparison:** Equation inspected. Its baseline-state update and output equal f_base and h_base at the baseline initial parameters, while its added-state update remains φ_aug. This directly supports your initial-behaviour analogy. Your zero correction is your construction for achieving it; equality over a rollout requires the same input as well as the same physical initial state. In the same closed-loop setup that follows from the same reference/controller initial state.

**Status: unverified — awaiting your confirmation.**

## Source 2: `gyorok2026obc`

[Open local PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Gyorok-2026-Orthogonal-by-construction.pdf>)

### G1 — [thesis line 404](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:404>)

**Retained thesis wording**

> The dynamic-parallel augmentation structure is chosen, because the orthogonal-by-construction augmentation is defined for this additive structure.

**Located paper evidence — PDF page 1 · printed page 1 · Abstract; also PDF page 2, Introduction** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Gyorok-2026-Orthogonal-by-construction.pdf>)

> This paper proposes a novel orthogonal-by-construction parametrization for augmenting physics-based input–output models with a learning component in an additive sense.

**Comparison:** Supports choosing an additive connection as the starting point. Scope: this source develops physics-based input–output model augmentation. It does not itself establish your extension to a dynamic state-space augmentation. Your later construction must justify that extension; this citation supplies the additive precedent.

Further exact passage, PDF page 2: “This approach connects the physics-based and learning components in parallel and the corresponding parameters of the two components are jointly optimized.”

**Status: unverified — awaiting your confirmation.**

## Source 3: `bishop1995neural`

[Open local PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Book-Bishop-Neural Networks for Pattern Recognition.pdf>)

### B1 — [thesis line 507](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:507>)

**Retained thesis wording**

> The inputs to the neural network are normalised to zero mean and unit variance, which has been shown to improve the estimation of models containing neural networks.

**Located paper evidence — PDF page 313 · printed page 298 · Section 8.2, “Input normalization and encoding”, first paragraph; Eqs. (8.1–8.2)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Book-Bishop-Neural Networks for Pattern Recognition.pdf>)

> One of the most common forms of pre-processing consists of a simple linear rescaling of the input variables. This is often useful if different variables have typical values which differ significantly.

**Comparison:** The book supports the motivation and exact zero-mean/unit-standard-deviation transformation. It presents this as often useful, not universally beneficial. Hoekstra Section 5.3 makes the stronger “shown to improve model estimation” statement and cites this book. The local PDF was checked visually because its mathematical text extraction is imperfect.

The displayed Eq. (8.2) subtracts each input's sample mean and divides by its sample standard deviation. Further exact passage: “By applying a linear transformation we can arrange for all of the inputs to have similar values.”

**Status: unverified — awaiting your confirmation.**

## Source 4: `kessels2025ai`

[Open local PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Kessels-2025-AI-based-model-updating.pdf>)

### K1 — [thesis line 729](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:729>)

**Retained thesis wording**

> The model is simulated in closed-loop with the controller $K_{\mathrm{fb}}$ of \eqref{eq:system_loop}, as in Kessels, instead of using the recorded input as in [Hoekstra].

**Located paper evidence — PDF pages 183–184 · printed pages 156–157 · Section 5.2, Eqs. (5.12), (5.13a–d)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Kessels-2025-AI-based-model-updating.pdf>)

> where it is remarked that, since fFB and gFB are assumed known, the FB controller may be regarded as a part of the known closed-loop FP model.

**Comparison:** The cited equations genuinely simulate feedback: Eq. (5.13d) forms reference minus simulated output, updates the controller state and produces its input; Eq. (5.13c) combines feedback and known feedforward. Eq. (5.12) is the fitting objective. Correct broad locator, with (5.13c–d) providing the decisive controller evidence. This is support from the Chapter 5 method, not a claim about every method in the thesis.

**Status: unverified — awaiting your confirmation.**

### K2 — [thesis line 899](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:899>)

**Retained thesis wording**

> Therefore the encoder of Section~\ref{sec:problem} only has to initialise the states of the model. Kessels instead needs the system's controller state at the start of each window, either measured directly or calculated by simulating the controller from the start of the record.

**Located paper evidence — PDF page 184 · printed page 157 · Remark 5.4** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Kessels-2025-AI-based-model-updating.pdf>)

> If feedback control is employed, the true feedback states x̄FB may be directly available on the machine.

**Comparison:** Remark number checked. It says the states can alternatively be reconstructed from measured output, the known reference and the known controller, and used to initialise each truncated window. Your first sentence describes your construction; the source supports the contrast. Reconstruction needs an initial controller state: Kessels assumes zero at k=1. It is not recovery from measurements with an arbitrary unknown initial state.

Further exact passages: “This information can be used to initialize the feedback states for each TW.” The preceding parenthesis states the assumption x̄^FB_(k=1)=0; the full remark is near the bottom of this page.

**Status: unverified — awaiting your confirmation.**

### K3 — [thesis line 961](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:961>)

**Retained thesis wording**

> The model output $\hat y_k$ depends only on the state $\tilde x_k$, not on the input $\hat u_{\mathrm{act},k}$ at the same sample. For each sample we first calculate the output, then the error and the model input, before updating the states. This avoids an algebraic loop through $D_{\mathrm{fb}}$, for the same reason as Kessels.

**Located paper evidence — PDF page 179 · printed page 152 · Remark 5.1** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Kessels-2025-AI-based-model-updating.pdf>)

> This is done for simplicity since otherwise there would be a non-causal relation between

**Comparison:** Correct remark. It explains that allowing the output augmentation to depend on the current input would make the model output depend on the feedback-controller output, which again depends on the model output. This is the same instantaneous circular dependency your state-only output avoids. Kessels calls it a “non-causal relation”; your “algebraic loop” describes the same dependency. Your particular D_fb and evaluation order are established by your own equations.

**Status: unverified — awaiting your confirmation.**

## Source 5: `beintema2023subnet`

[Open local PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Beintema-2023-Deep-subspace-encoders.pdf>)

### S1 — [thesis line 1107](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1107>)

**Retained thesis wording**

> Estimating $\hat x_\tau$ as a separate parameter for each window would increase the number of parameters as the data grows.

**Located paper evidence — PDF page 5 · printed page 4 · Section 3.1, left column, paragraph after Eq. (9)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Beintema-2023-Deep-subspace-encoders.pdf>)

> number of parameters would scale O(N)

**Comparison:** Direct support. The preceding sentence considers making each section's initial state an optimisation parameter. This assumes the number of windows increases with the amount of data, as in their formulation. It does not say a fixed set of windows creates more parameters just because the underlying file grows.

**Status: unverified — awaiting your confirmation.**

### S2 — [thesis line 1110](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1110>)

**Retained thesis wording**

> Following Beintema et al., we use an encoder network that estimates $\hat x_\tau$ from the past inputs and outputs $\chi_\tau$.

**Located paper evidence — PDF page 5 · printed page 4 · Section 3.2, Eq. (12), Eq. (13b), and right-column explanation** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Beintema-2023-Deep-subspace-encoders.pdf>)

> Here, n corresponds to the number of past inputs and outputs, i.e. lag window, considered to estimate the initial state,

**Comparison:** Supports the encoder idea. Important equation distinction: Beintema's displayed encoder includes outputs through t, while inputs end at t−1. Your both-histories-ending-at-τ−1 convention is the Hoekstra Eq. (22e) convention, separately and correctly cited at line 1180. Do not treat Beintema's encoder equation as identical to your indexing.

**Status: unverified — awaiting your confirmation.**

### S3 — [thesis line 1115](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1115>)

**Retained thesis wording**

> keeps the computation per gradient step independent of the record length, for a fixed batch size and window length.

**Located paper evidence — PDF page 5 · printed page 4 · Section 3.1, Eq. (9a–c)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Beintema-2023-Deep-subspace-encoders.pdf>)

> This results in a batch formulation of the loss:

**Comparison:** Eq. (9) was inspected: it averages over N_batch window losses, each of length T. Thus fixed batch size and window length keep the rollout work per gradient update independent of N, assuming fixed architecture/history. This is a consequence of the displayed equation, not an exact prose assertion that every operation including dataset loading is independent of N.

Separate prose issue: the current retained text starts with lowercase “keeps” and has no subject after the preceding full stop. No edit made. Minimal repair would be “**This** keeps …”.

**Status: unverified — awaiting your confirmation.**

### S4 — [thesis line 1117](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1117>)

**Retained thesis wording**

> Using short simulation windows increases the loss function smoothness.

**Located paper evidence — PDF page 4 · printed page 3 · Section 3.1; PDF page 8 · printed page 7 · Section 4.2** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Beintema-2023-Deep-subspace-encoders.pdf>)

> Next, we show that the considered estimation structure and the truncated prediction loss increase the smoothness of the cost function, which potentially makes the optimization process for model estimation more stable and less prone to get stuck in local minima (Ribeiro et al., 2020).

**Comparison:** Both section locators are correct. Section 3.1 states the motivation; Section 4.2 gives the Lipschitz analysis under its stated model assumptions. This supports the stated motivation, not an unconditional guarantee of successful optimisation for every architecture or horizon.

**Status: unverified — awaiting your confirmation.**

### S5 — [thesis line 1243](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1243>)

**Retained thesis wording**

> SUBNET assumes that the input is independent of the noise and that the system that generates the data is incrementally exponentially output stable.

**Located paper evidence — PDF page 3 · printed page 2 · Section 2.1, paragraph after Eq. (1)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Beintema-2023-Deep-subspace-encoders.pdf>)

> u is a quasi-stationary input process independent of e

**Comparison:** Direct support for input/noise independence in the assumed data-generating system. The noise e is the source's i.i.d. white-noise process. The stability half of your sentence is checked separately under Condition 1 below. These hypotheses belong to the model/theory; using an encoder in a computation does not itself establish that the hypotheses hold.

**Status: unverified — awaiting your confirmation.**

### S6 — [thesis line 1243](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1243>)

**Retained thesis wording**

> SUBNET assumes that the input is independent of the noise and that the system that generates the data is incrementally exponentially output stable.

**Located paper evidence — PDF page 7 · printed page 6 · Section 4.1, Condition 1** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Beintema-2023-Deep-subspace-encoders.pdf>)

> The data-generating system (4) is (globally) incrementally exponentially output stable

**Comparison:** The label really is Condition 1. It concerns exponential decay of the effect of differing initial states on output under the specified common input/noise process. It supports the second half of your sentence. Your claim that the gantry violates it and that feedback corrects the offset remains your own model-based reasoning; neither this condition nor feedback simulation alone proves global exponential stability of your closed loop.

**Status: unverified — awaiting your confirmation.**

## Source 6: `hoekstra2026encoder`

[Open local PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-Encoder-initialisation-arXiv-v1.pdf>)

### E1 — [thesis line 1127](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1127>)

**Retained thesis wording**

> The network parameters $\theta_{\mathrm{aug}}$, the encoder parameters $\theta_{\mathrm{enc}}$ and $\theta_{\mathrm{base}}$ are jointly estimated, as in [Hoekstra LFR], while the encoder paper keeps $\theta_{\mathrm{base}}$ fixed.

**Located paper evidence — PDF page 2 · printed page 2 · Footnote 1; Eq. (4) for the fitted parameter vector** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-Encoder-initialisation-arXiv-v1.pdf>)

> It is possible to co-estimate the baseline model parameters with θaug and θenc in the model augmentation setting, see, e.g., (Hoekstra et al., 2025). This is, however, outside the scope of this paper.

**Comparison:** Footnote number checked and direct support for the contrast. Their fitted vector contains augmentation and encoder parameters, excluding baseline parameters. This does not claim co-estimating the baseline is impossible; it expressly excludes it from their scope.

**Status: unverified — awaiting your confirmation.**

### E2 — [thesis line 1209](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1209>)

**Retained thesis wording**

> The encoder uses a linear map with a nonlinear correction:

**Located paper evidence — PDF page 3 · printed page 3 · Eq. (8)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-Encoder-initialisation-arXiv-v1.pdf>)

> We parameterise ψθ as a ResNet architecture in the form

**Comparison:** Eq. (8) was checked visually: a nonlinear encoder term is added to a linear bypass from stacked input/output histories, with separate baseline-state and added-state rows. This directly supports your displayed architecture. Your W^b and W^a consolidate their separate input/output blocks; the notation differs without changing the form. Both histories in Eq. (8) end at k−1.

**Status: unverified — awaiting your confirmation.**

### E3 — [thesis line 1372](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1372>)

**Retained thesis wording**

> A model-based state reconstruction map for the nonlinear baseline needs a linearisation point, so the map is only accurate around that point.

**Located paper evidence — PDF page 4 · printed page 4 · Section 3.1, “Nonlinear baseline models”, Eqs. (30–32)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-Encoder-initialisation-arXiv-v1.pdf>)

> In the general case, when the baseline model is not LTI, an approximated model can be provided based on the local linearisation of the NL model.

**Comparison:** The locator is correct, but your wording generalises the specific method. Their linearisation-based reconstruction requires choosing an equilibrium and is locally accurate. They do not establish that every possible nonlinear model-based reconstruction map needs a linearisation point.

Further exact passage: “Keep in mind that the model-based state reconstruction will only be accurate around the equilibrium point (x∗b, u∗), but the proposed method is only applied for initialising the parameters of the encoder network.” Minimal suggestion, not applied: “**Hoekstra et al.'s** model-based state reconstruction map for the nonlinear baseline needs a linearisation point, so the map is only accurate around that point.”

**Status: unverified — awaiting your confirmation.**

### E4 — [thesis line 1377](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1377>)

**Retained thesis wording**

> Hoekstra et al. therefore suggest fitting the encoder to states simulated with the baseline, as their augmentation method does.

**Located paper evidence — PDF page 5 · printed page 5 · Section 3.2, Eq. (35)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-Encoder-initialisation-arXiv-v1.pdf>)

> We first generate an approximated dataset by forward simulating the baseline model on DN, generating D̂N = {(ŷk, x̂b,k, uk)}N k=1.

**Comparison:** The method is correctly located: Eq. (35) minimises squared encoder-state error against simulated baseline states for a nonlinear baseline. Their proposal fits an ANN, whereas you initially fit only the affine physical rows. Your later text explicitly acknowledges that difference.

Further exact passage, with only line-end hyphenation joined: “However, if the applied baseline model is nonlinear, we propose to fit a randomly initialised ANN encoder ψ̄ on D̂N by minimising the following loss function”

**Status: unverified — awaiting your confirmation.**

### E5 — [thesis line 1420](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1420>)

**Retained thesis wording**

> The physical rows of the encoder are then fitted to the simulated states by least squares:

**Located paper evidence — PDF page 5 · printed page 5 · Section 3.2, Eq. (33)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-Encoder-initialisation-arXiv-v1.pdf>)

> If the baseline model is in LTI-SS form, it is sufficient to fit the initial values of

**Comparison:** Equation number checked: (33) is the linear least-squares fit of the encoder's physical rows to simulated baseline states. It is expressly proposed for LTI baselines. Your baseline is nonlinear and your equation adds a constant offset b. Thus (33) is a precedent for your fitting method, not the identical problem with the same applicability claim. Your following sentence accurately states the LTI/nonlinear distinction.

The quoted sentence continues with the two physical-row weight blocks and “by solving the following Linear Least Squares (LLS) problem:”. Your Eq. (aug_enc_fit) is an affine least-squares adaptation; the source Eq. (33) has no separately fitted offset.

**Status: unverified — awaiting your confirmation.**

### E6 — [thesis line 1441](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1441>)

**Retained thesis wording**

> Hoekstra et al. use this linear fit for LTI baselines and fit a neural network for nonlinear baselines.

**Located paper evidence — PDF page 5 · printed page 5 · Section 3.2, Eqs. (33) and (35)** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hoekstra-2026-Encoder-initialisation-arXiv-v1.pdf>)

> However, if the applied baseline model is nonlinear, we propose to fit a randomly initialised ANN encoder

**Comparison:** Direct support. Eq. (33) is LLS for the LTI case; Eq. (35) is ANN fitting for the nonlinear case. Both explicit equation references are correct. The next thesis sentences distinguish your own restricted linear initial fit and later nonlinear training.

**Status: unverified — awaiting your confirmation.**

## Source 7: `drenth2026lpvlfr`

[Open local PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Drenth-2026-LPV-models-linear-fractional-representation-arXiv-v1.pdf>)

### D1 — [thesis line 1330](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1330>)

**Retained thesis wording**

> Drenth et al. also use Adam followed by L-BFGS-B.

**Located paper evidence — PDF page 4 · Section 4.1 (section starts on PDF page 3), left column, paragraph before Section 4.2** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Drenth-2026-LPV-models-linear-fractional-representation-arXiv-v1.pdf>)

> First, a fixed number of iterations is taken using the Adam algorithm, followed by a number of iterations using the bound-constrained limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS-B) algorithm (Liu and Nocedal (1989)).

**Comparison:** Direct support and correct section number. Your preceding text uses L-BFGS without bounds; “also use Adam followed by L-BFGS-B” accurately names their different second solver rather than claiming the algorithms and selection rules are identical.

**Status: unverified — awaiting your confirmation.**

## Source 8: `hefny2015supervised`

[Open local PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hefny-2015-Supervised-learning-dynamical-systems.pdf>)

### P1 — [thesis line 1626](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1626>)

**Retained thesis wording**

> PS is adapted from a two-stage regression of predictive-state learning, which Downey et al. use to initialise a recurrent network before backpropagation through time (BPTT).

**Located paper evidence — PDF pages 2–4 · printed pages 2–4 · Section 2; Figure 2 on PDF page 2 and regression summary on PDF page 4** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hefny-2015-Supervised-learning-dynamical-systems.pdf>)

> learning to estimate E[ψt | ht] and E[ξt | ht], effectively denoising the training examples, and then use these estimates to compute W by finding the least squares solution to (2).

**Comparison:** The source supplies the two-stage regression principle. Stage 1 contains two supervised regressions (S1A/S1B); stage 2 learns a linear operator from their estimates. Your PS procedure is an adaptation with residual targets and a nonlinear state update, not their identical instrumental-regression estimator. The phrase “is adapted” and your next sentences disclose those changes. The Downey half is checked separately below.

**Status: unverified — awaiting your confirmation.**

### P2 — [thesis line 1630](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1630>)

**Retained thesis wording**

> S1 predicts the residual of the model, instead of features of future observations as in [Hefny]. S2 trains the nonlinear state update instead of a linear map.

**Located paper evidence — PDF page 2 · printed page 2 · Section 2, predictive-state definition; PDF page 4 · printed page 4 · S1/S2 regression summary** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hefny-2015-Supervised-learning-dynamical-systems.pdf>)

> where ψt = ψ(ot:t+k−1) is a vector of future features.

**Comparison:** Direct support for what is being contrasted. Their S1 estimates conditional expectations of future and extended-future features from history; their S2 fits a linear operator. Your residual prediction and nonlinear update are your modifications, so no consistency result transfers automatically.

Further exact passage, PDF page 4: “S2 Regression: Use the feature expectations estimated in S1A and S1B to train a model to predict” followed by ξ̄_t=Wψ̄_t, “where W is a linear operator.”

**Status: unverified — awaiting your confirmation.**

### P3 — [thesis line 1652](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1652>)

**Retained thesis wording**

> Samples used by PS on one record, with the layout of [Hefny, Fig. 2]. In S1 (top), the encoder $\psi$ estimates $\hat x^{\mathrm n}_{\tau|\tau}$ from the window $\chi_\tau$. The decoder $p$ then predicts the residual $e_\tau$ of the next $H$ samples from the added states $\bar x_{\tau|\tau}$ \eqref{eq:ps2_s1}. In S2 (bottom), $g_{\mathrm{aug}}$ predicts the next added state $\bar x_{\tau+1|\tau+1}$ from $\hat x^{\mathrm n}_{\tau|\tau}$ and the recorded $u_\tau$ \eqref{eq:ps2_s2}. The same encoder gives $\bar x_{\tau+1|\tau+1}$ from the window $\chi_{\tau+1}$, shifted by one sample.

**Located paper evidence — PDF page 2 · printed page 2 · Figure 2 and caption** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Hefny-2015-Supervised-learning-dynamical-systems.pdf>)

> Learning and applying a dynamical system with instrumental regression. The predictions from S1 provide training data to S2. At test time, we filter or predict using the weights from S2.

**Comparison:** Figure number and actual diagram checked. It depicts history, future, shifted future and extended future windows with S1/S2 regression arrows. Your figure uses that temporal-window idea but has a different two-row arrangement and your modified targets. The source supports visual inspiration; its caption does not support the details of your own residual decoder/nonlinear state-update algorithm.

Optional precision, not applied: “with the layout **adapted from** [Hefny, Fig. 2]”. This better describes the relationship than implying the layouts are identical.

**Status: unverified — awaiting your confirmation.**

## Source 9: `downey2017psrnn`

[Open local PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Downey-2017-Predictive-state-recurrent-neural-networks.pdf>)

### R1 — [thesis line 1626](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/Writing/sections/03_augmentation.tex:1626>)

**Retained thesis wording**

> PS is adapted from a two-stage regression of predictive-state learning, which Downey et al. use to initialise a recurrent network before backpropagation through time (BPTT).

**Located paper evidence — PDF page 5 · printed page 5 · Section 4.2, first paragraph** · [PDF](<C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation/Thesis-writeup/References/Downey-2017-Predictive-state-recurrent-neural-networks.pdf>)

> There are two components to learning PSRNNs: an initialization procedure followed by gradient-based refinement. We first show how a statistically consistent 2SR algorithm derived for PSRs can be used to initialize the PSRNN model; this model can then be refined via BPTT.

**Comparison:** Direct support and correct section number. Their order is initialisation followed by refinement. Your text later states that you continue minimising V during your first stage, making your procedure an adaptation. The quoted statistical consistency concerns the referenced PSR algorithm; it is not a guarantee for your residual-based nonlinear variant.

**Status: unverified — awaiting your confirmation.**

## Confirmation record

No occurrence has been marked verified in this review. You can confirm individual IDs (for example, H1 and H2), reject one, or ask to discuss a quoted passage. Proposed wording in this document is not an edit to the thesis.
