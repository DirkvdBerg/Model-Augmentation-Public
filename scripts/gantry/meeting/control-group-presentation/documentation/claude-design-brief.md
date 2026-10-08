# Claude Design brief: 15-minute thesis talk for the TU/e Control Systems group

Part A is for Dirk. Parts B and C are pasted into Claude Design unchanged.

## Part A. For Dirk

### How to use
1. Open Claude Design (claude.ai, the Design or Slides starting point).
2. Attach: the TU/e template as **.pptx** (a .potx: open it in PowerPoint, save as .pptx; only
   .pptx upload is documented), all 16 PNG files in `figures/`, and optionally one ASMPT machine
   photo named `asmpt-photo.png` or `.jpg` (slide 2 uses it if present).
3. Paste everything from "Part B" down (Parts B and C).
4. Export PPTX (to present and edit) and PDF (backup). Check that the speaker notes arrived and
   that slides 6 to 9 show the four step figures at exactly the same position and size (they are
   one build). If the notes are missing, they are in Part B per slide.
5. Check before the talk: slide 2 facts come from your thesis introduction
   (`01_introduction.tex`); the "next steps" on slide 16 are my reading of the thesis plan.

### What Claude Design can do (lookup 2026-09-29, official Anthropic sources only)
| Question | Official answer | Source |
|-|-|-|
| Inputs | text, images, DOCX/PPTX/XLSX, a codebase, web capture; "attach an existing slide deck or document with a design style you want to replicate" | [1], [2], [5] |
| Template | a PPTX or PDF that reflects the brand: "Claude extracts colors, layout patterns, and typographic choices"; .potx not mentioned | [3] |
| Charts | "Describe the data you want to visualize and Claude will generate appropriate charts"; exact rendering from supplied numbers not stated | [4] |
| Diagrams, equations, animations | not documented | [2], [5] |
| Speaker notes | not documented for Claude Design | [1], [2], [4] |
| Export | .zip, PDF, PPTX, Google Slides, standalone HTML, Canva and other tools | [2], [6] |
| Upload limits | not documented | [2], [3], [6] |
| Prompting advice | be specific about audience, key messages and context; give slide count and sections; attach real examples | [4], [3] |

Rule applied: a capability no official source confirms counts as unavailable.

| Element | Made by | Reason |
|-|-|-|
| Layout, template styling, titles, bullets, backup table | Claude Design | its core function [1], [2]; template from the PPTX [3] |
| Concept visuals without data (slide 2 strip, slide 3 boxes) | Claude Design, from an exact spec | visual outputs from a prompt [7]; no data to get wrong |
| Data charts (5 main, 1 backup) | us, PNG | charts only "from a description" [4]; a 48 000-sample trace cannot be typed into a prompt; one style across all figures |
| Schematics, augmentation steps, projection, validation loop | us, PNG | geometry and signal flow must be exact; diagram rendering undocumented |
| Equation of motion, orthogonality condition, backup formulas | us, PNG | math rendering undocumented |
| Diagrams and formulas | Claude Design may redraw and improve all of them; our PNGs give the content | user decision 2026-09-29 |
| Speaker notes | us, text in Part B | undocumented; Claude Design is asked to place them, Part B is the fallback |
| Figure format | PNG, white background, 250 to 600 dpi | images documented [1], [2]; SVG not mentioned |

Sources (all read 2026-09-29):
[1] https://www.anthropic.com/news/claude-design-anthropic-labs (17 Apr 2026)
[2] https://support.claude.com/en/articles/14604416-get-started-with-claude-design
[3] https://support.claude.com/en/articles/14604397-set-up-your-design-system-in-claude-design
[4] https://academy.claude.com/tutorials/using-claude-design-for-presentations-and-slide-decks
[5] https://claude.com/product/design
[6] https://support.claude.com/en/articles/14604406-claude-design-admin-guide-for-team-and-enterprise-plans
[7] https://support.claude.com/en/articles/12138966-release-notes (16 Sep 2026: designs and decks in any conversation; Claude Slides as its own starting point)

### Figures
Code: `code/data_figures.py` (data, reads the caches only) and `code/build_tikz.py` (TikZ; the
gantry schematics `\input` the thesis figure). Palette in `code/style.py` and
`code/tikz/preamble.tex`; swap there for TU/e colours. Errors are in metres, scientific notation.
| File | Slide | Content | Source |
|-|-|-|-|
| `gantry-baseline.png` | 4 | thesis schematic, no absorber | `gantry_system_with_absorber.tex`, `\gantryabsorber` 0 |
| `equation.png` | 4 | equation of motion, terms labelled | thesis Eq. (eom) |
| `gantry-absorber.png` | 5 | thesis schematic with absorber, no δa arrow | same, `\gantryabsorber` 1 |
| `augmentation-step1..4.png` | 6 to 9 | v4 block scheme built up in four steps, identical canvas | adapted from `docs/writeup/jan-blockscheme-v4.tex` |
| `closedloop-trace.png` | 10 | V4 (Lissajous, Y = -0.10 m), payload axis, errors in m | `figure_data_81262.npz` |
| `blocksize-curves.png` | 11 | held-out closed-loop error vs epoch, 81262 vs 81265 | `gantry_results_8126x.npz` `loss_val` |
| `orthogonality.png` | 12 | ⟨Φ_j, Δ⟩ = 0, each symbol labelled in words | 25-09-2026 meeting notation |
| `projection-sketch.png` | 13 | exact 3D projection geometry, F, Δ, Φ | new TikZ |
| `overlap-bars.png` | 14 | ρ of the applied correction, 84032 vs 84033, train and held-out | `orthogonality_data.json` (J, applied) |
| `obc-fit-params.png` | 15 | fit [m] and combined parameter error [%] vs epoch | `obc_data.json` |
| `validation-loop.png` | B1 | closed-loop validation diagram | new TikZ |
| `orthogonality-formulas.png` | B2 | Φ, inner product, refit argument, projection, ρ | 25-09-2026 deck + `fig_orthogonality.py` |
| `params-all.png` | B3 | all ten parameter combinations, both arms | `obc_data.json` |

Every number in the figures is checked against the handoff values when the script runs
(`check()` in `data_figures.py`). V4 is drawn because its physics/augmented ratio over the three
measured positions (3.71) is closest to the 3.57 aggregate. Step-scheme simplifications against
v4: no setpoint generator (u_k is the motor-force input), and h_base drawn without its u input.

Numbers you may be asked about: the physics-model error on the V4 payload axis peaks near 223 Hz
in closed loop, not at the 212 Hz open-loop absorber mode, so the notes say "the excited band".
The raw network output of the projected run has ρ = 0.92 (training) and 0.93 (held-out); it is
not in any figure because that part is removed before it reaches the model. The two baselines
stay apart: 2.06·10⁻⁵ m is the nominal physics model (8126x), 2.30·10⁻⁵ m the 10 % detuned start
(8403x).

Notation mismatch to resolve in one place: the talk uses Φ (your 25-09 notation) for the parameter
directions and Δ for the learned correction; thesis Section IV currently writes J and g, and uses
Δ* for the true hidden dynamics. Slide 15 and 16 now cite ρ* = 0.205 (predicted parameter-error
floor 4.96 %), measured on this dataset (`orthogonal-addition/RUNS.md` R-002;
`baseline-and-added-dynamics.md` Sect. 5: the floor matches the error size but not its
per-parameter pattern). Part C gives Claude Design the thesis structure, notation and supervisor
rules; the thesis draft itself is not uploaded because its open todo notes mark unsupported claims.

## Part B. Brief for Claude Design

Build a 16:9 slide deck for a 15-minute master-thesis talk, followed by 5 minutes of questions,
for the Control Systems research group at Eindhoven University of Technology (TU/e). Audience:
control engineers (staff and PhD students). They know state space, feedback, inner products and
system identification; they do not know this project, its acronyms or its code. Speaker: Dirk
van den Berg. Thesis: "Model augmentation for a dual-gantry high-precision motion system", in
collaboration with ASMPT.

### Global rules
1. Use the attached TU/e template (.pptx): its colours, fonts, title and footer layout. 16:9.
2. 16 main slides, then a divider slide "Backup", then 4 backup slides. The times per slide are
   for the speaker: do not show them.
3. Slide titles are full-sentence messages: use them verbatim. Shorten only if the template cannot
   fit them, and keep every number.
4. At most 3 short bullets per slide, verbatim or shorter. No extra text blocks, no footnotes of
   your own, no invented facts or numbers.
5. Figures: the attached PNG named for each slide is our current version and shows the content
   the slide needs. Make better figures wherever you can: redraw, simplify, split into steps or
   restyle them. The only limit: every number and every data curve must come from the attached
   figures or this brief; do not invent data.
6. Formulas: the attached `equation.png` (slide 4), `orthogonality.png` (slide 12) and
   `orthogonality-formulas.png` (backup B2) show what each formula says; typeset them yourself if
   that looks better.
7. Put the "Notes" text of each slide into that slide's speaker notes, verbatim.
8. Never use the acronyms OBC, LPV, LFR, LPV-LFR, SUBNET, FP or ANN. Say "orthogonal projection",
   "physics model", "neural network".
9. No em-dashes or en-dashes as punctuation: use colons or commas.
10. Position errors are always in metres in scientific notation, exactly as written here, for
    example 2.06·10⁻⁵ m. Never convert them to µm or mm.
11. Colours in the figures carry meaning: orange = physics model, blue = augmented model or learned
    part, green = augmented model with projection, light-blue dashed = smaller learned block. Do not
    use these colours for other meanings in text or shapes you add.
12. Where a slide says "Draw:", draw it simply in template colours, with exactly the text given.
13. Result slides 10, 11, 14 and 15 carry a small grey footer: "Noiseless simulation, no friction,
    one training run per setting, earlier pipeline version."
14. Slides 6 to 9 build up one figure step by step: keep the figure in the same place on all four
    so each step visibly adds to the previous one.

### Slide 1 (0:15). Title
- Title: "Model augmentation for a dual-gantry high-precision motion system"
- Subtitle: "Learning what the physics model misses, while keeping its parameters physical"
- Also on the slide: "Dirk van den Berg | MSc thesis, TU/e Control Systems, with ASMPT |
  Supervisors: M. Schoukens, R. Tóth, J. Hoekstra"
- Notes: "I will show how we add a learned part to a physics model of a motion stage, how much
  more accurate that makes it, and how we keep the learned part from taking over the job of the
  physical parameters."

### Slide 2 (0:45). A model of the machine lets ASMPT test a changed controller before it runs on the machine
- Bullets:
  - ASMPT develops semiconductor assembly and packaging equipment with high-precision motion systems
  - Test system: a dual-gantry, two parallel drives carry a cross-arm, a third drive moves the payload along it
  - The model is only useful if it predicts the machine's dynamics accurately
- Draw: a horizontal strip of three rounded boxes joined by arrows: "Change the controller" →
  "Test it on the model" → "Run it on the machine". Under the middle box, small text: "needs an
  accurate model". If an image named `asmpt-photo` is attached, put it on the right half and the
  bullets and strip on the left half; otherwise the strip goes full width under the bullets.
- Notes: "ASMPT develops semiconductor assembly and packaging equipment, with high-precision motion
  systems inside. The system in this thesis is a dual-gantry: two parallel drives carry a
  cross-arm, and a third drive moves the payload along it. The use case is a digital twin: when
  the control engineers change a controller, they try it on a model first, before it runs on the
  machine. That only works if the model predicts the dynamics accurately."

### Slide 3 (1:00). Grey-box modelling keeps the physics and learns only what it misses
- Draw: three boxes side by side, left to right, the middle one highlighted with the template
  accent (thicker outline):
  - "White box: physics only" / "+ physical parameters: mass, stiffness, damping" / "− misses
    dynamics nobody modelled"
  - "Grey box: physics plus a learned part" / "+ keeps the physical parameters" / "+ learns the
    missing dynamics from data"
  - "Black box: data only" / "+ flexible" / "− no physical parameters, usually needs more data,
    harder to trust outside the training data"
- Bullet (under the boxes): For ASMPT: accurate predictions, and parameters that still mean mass,
  stiffness and damping
- Notes: "A white-box model is built from physics: its parameters are masses, stiffnesses and
  dampings, which engineers use for control design, but it misses whatever nobody modelled. A
  black-box model learns everything from data: flexible, but its parameters mean nothing
  physically, it usually needs more data, and it is harder to trust away from its training data.
  Grey-box modelling keeps the physics model and adds a learned part only for what the physics
  misses. For ASMPT that is the point: accurate predictions with meaningful parameters."

### Slide 4 (1:30). The physics model has three coordinates, and its inertia changes with the payload position
- Layout: `gantry-baseline.png` on the left, about 6.2 in wide. On the right, `equation.png`
  about 5.9 in wide at the top, the bullets below it.
- Bullets:
  - Coordinates: X (cross-arm position), Θ (cross-arm yaw), Y (payload position)
  - Moving the payload shifts mass, so the inertia depends on Y
  - No spring to the ground in X and Y: the stage always runs under feedback
- Notes: "This is the physics model, from Garcia-Herreros and colleagues. Three coordinates: the
  cross-arm position X, its yaw angle Theta, and the payload position Y. M of Y is the inertia
  matrix; it depends on where the payload is, because moving the payload moves mass along the
  cross-arm. C is damping in the drives and the flexible joints. K is stiffness, with a single
  nonzero entry: the joints resist yaw. In X and Y there is no stiffness at all, so any force
  error integrates into a position drift; that is why the machine, and all our validation, runs
  with the controller in the loop. From data we can estimate ten combinations of physical
  parameters; for example only the sum of the two joint stiffnesses, not each one."

### Slide 5 (0:45). The simulated machine has a payload resonance that the physics model does not contain
- Layout: `gantry-absorber.png` on the left, about 6.2 in wide; bullets on the right.
- Bullets:
  - Hidden in the simulation: half of the 10.1 kg payload sits on a spring and damper (grey)
  - Resonance at 212 Hz; the excitation, 140 to 230 Hz, covers it
  - Noiseless simulated data, controller in the loop: 14 training and 4 held-out records of 12 s
- Notes: "To test the method where we know the truth, the simulated machine has one thing the
  physics model lacks: half of the payload mass hangs on a spring and damper, a resonance at 212
  hertz. The physics model has six states, the true machine eight, so the learned part must find
  two missing states from input and output data. The data are noiseless, with the controller in
  the loop: 14 training records and 4 held-out records of 12 seconds, sampled at 4 kilohertz."

### Slides 6 to 9 (2:00 in total, 0:30 each). A learned block with its own states corrects the physics model's state update
All four slides use this same title and layout: the step figure about 10 in wide, centred, with
one caption line under it.

Slide 6 (0:30)
- Figure: `augmentation-step1.png`
- Caption: "Step 1: the physics model moves the physical state one time step forward; the output map gives the predicted positions"
- Notes: "Start from the physics model in discrete time. From the motor forces and the physical
  state it computes the next physical state; the state memory hands that back as the current state
  at the next sample. The output map turns the physical state into the predicted positions. In the
  thesis these blocks are f base and h base."

Slide 7 (0:30)
- Figure: `augmentation-step2.png`
- Caption: "Step 2: a neural network adds a correction to the Θ, X and Y updates; the output map stays physical"
- Notes: "Now we add a neural network. It sees the physical state and the motor forces, and a
  router adds its output as a correction to the updates of Theta, X and Y. There is no learned
  output: what we compare with data are still the physical positions. At the start of training
  the network output is exactly zero, so training starts from the physics model."

Slide 8 (0:30)
- Figure: `augmentation-step3.png`
- Caption: "Step 3: the network gets learned states of its own, so it can represent missing dynamics"
- Notes: "Next, the network gets learned states of its own, which it updates every step and reads
  back at the next one. With states it can represent dynamics, such as the hidden resonance, and
  not only a static correction."

Slide 9 (0:30)
- Figure: `augmentation-step4.png`
- Caption: "Step 4: an encoder estimates the initial state from past inputs and outputs"
- Notes: "Finally, the learned states are not measured, so an encoder estimates the initial state,
  physical and learned, from a short window of past inputs and outputs. Network and encoder are
  trained together, and the physical parameters can be trained with them."

### Slide 10 (1:15). On held-out records the augmented model cuts the closed-loop error 3.6 times, from 2.06·10⁻⁵ m to 5.77·10⁻⁶ m
- Layout: `closedloop-trace.png` full width, about 12 in wide; bullets below it.
- Bullets:
  - Criterion: lower error than the physics model on each of the 4 held-out records
  - RMS error over 4 held-out records and 3 measured positions: 2.06·10⁻⁵ m (physics) to 5.77·10⁻⁶ m (augmented), met on all 4
  - Shown: one record, the payload axis, where the resonance acts
- Footer: see global rule 13.
- Notes: "The criterion: on records not used for training, the augmented model must have a lower
  error than the physics model, on every record. Both models run with the controller in the loop,
  as the machine does. The figure shows one held-out record on the payload axis: the physics
  model's error in orange, the augmented model's in blue, and a 40 millisecond zoom. The error is
  a fast oscillation from the excited band, and the augmented model removes most of it. Over all
  four records and three measured positions the error drops from 2.06 times ten to the minus five
  metres to 5.77 times ten to the minus six, 3.6 times lower, and it is lower on every record. So
  the criterion is met. About a quarter of the error remains. If we switch off the update of the
  learned states, the error doubles to 1.17 times ten to the minus five metres: a large part of the
  gain needs the learned dynamics."

### Slide 11 (0:45). A smaller learned block, 2 states instead of 8, ends within 3 % of the larger one
- Layout: `blocksize-curves.png` on the left, about 7.6 in wide; bullets on the right.
- Bullets:
  - Criterion: a larger block is only worth it if it clearly lowers the error
  - 2 states, 4 019 weights: 5.95·10⁻⁶ m; 8 states, 10 871 weights: 5.77·10⁻⁶ m
  - Verdict: 2.7 times the weights buys 3 %; block size is not the bottleneck
- Footer: see global rule 13.
- Notes: "How large must the learned block be? 8 learned states against 2, which is also the number
  of states the physics model misses. Both start at the physics model and drop fast to about one
  times ten to the minus five metres, stay flat, then drop again; the small block stays flat
  longer. The large block has 2.7 times the weights and is only 3 % better, so block size is not
  what limits accuracy. The small block also has a smaller network and a shorter encoder window,
  and it is one training run each, so a 3 % difference is within what a second run could change."

### Slide 12 (1:45). Orthogonal means that no change of the physical parameters can produce the correction, checked on the data
- Layout: `orthogonality.png` full width, about 11 in wide; bullets below it.
- Bullets:
  - Trained together, the network and the physical parameters can explain the same error, and then the parameters lose their meaning
  - Orthogonality: a zero inner product with each of the ten parameter directions Φ_j
  - Data-derived: Φ_j and Δ are evaluated on the training samples, so orthogonality holds on those data
- Notes: "Now interpretability. When we train the physical parameters together with the network,
  both can explain the same error: the network can do what a change of mass or stiffness would have
  done, and then the parameters drift and stop meaning what they should. A penalty that keeps the
  parameters near their start does not prevent this; it only limits how far they move. So we ask
  for orthogonality. For each parameter j we take every training sample and ask: if I change this
  parameter slightly, how does the physics model's one-step prediction change? That gives one
  signal over the data per parameter, Phi j. Delta is the network's correction at the same
  samples. Their inner product sums the product over all samples. Orthogonal means it is zero for
  all ten parameters: then refitting the parameters on these data would not move them, so they
  keep their physical meaning. This is data-derived: it is defined on the samples we have, so it
  holds on the training data by construction, and on new trajectories we have to check it."

### Slide 13 (1:15). Orthogonal projection removes the part of the network output that a parameter change could produce
- Layout: bullets on the left; `projection-sketch.png` on the right, about 6.5 in wide.
- Bullets:
  - Removed: the part of the network output F that lies in the parameter directions
  - Kept: the correction Δ, orthogonal to all ten directions on the training samples
  - Fitted on the training samples and built into the model's state update, no penalty weight to tune
- Notes: "This is how we enforce it. The grey plane stands for everything a change of the ten
  parameters can do; in reality it has ten directions. The network output F is the blue arrow. Its
  shadow in the plane is the part a parameter change could also produce; the model subtracts it.
  Only the green part, perpendicular to the plane, reaches the model: that is the correction Delta.
  The shadow is fitted on the training samples and applied inside the state update, so it is a
  hard constraint, not a penalty with a weight to tune."

### Slide 14 (1:00). Projection removes the overlap on the training data, and cuts it from 0.87 to 0.33 on held-out data
- Layout: bullets on the left; `overlap-bars.png` on the right, about 6.8 in wide.
- Bullets:
  - Overlap ρ: the share of the correction Δ that a parameter change could reproduce (0 = orthogonal, 1 = a parameter change)
  - Criterion: ρ near zero with projection
  - Training records: 0.87 to 8.3·10⁻⁸, met. Held-out records: 0.87 to 0.33, partly met
- Footer: see global rule 13.
- Notes: "We measure the overlap rho: which share of the correction a parameter change could
  reproduce. Without projection it is 0.87: most of what the network does, the parameters could
  have done. With projection, on the training records, it is 8.3 times ten to the minus eight:
  removed, as expected, because that is where the projection is fitted. On held-out records it
  drops from 0.87 to 0.33: on new trajectories the parameter directions differ, and a third of the
  correction overlaps again. That is the data-derived nature showing. If asked: the network output
  itself is not orthogonal; the projection makes the part that reaches the model orthogonal. These
  are the two runs of the next slide."

### Slide 15 (1:15). Projection costs no accuracy, but the parameters do not return to their true values
- Layout: `obc-fit-params.png` full width, about 12 in wide; bullets below it.
- Bullets:
  - Both runs start with all ten parameters 10 % off: 2.30·10⁻⁵ m error, 9.5 % parameter error
  - Criterion 1, no loss of fit: 5.86·10⁻⁶ m without, 5.89·10⁻⁶ m with projection (0.5 % apart): met
  - Criterion 2, parameters closer to the truth than without: 7.7 % with, 7.2 % without projection: not met
- Footer: see global rule 13.
- Notes: "Both runs train the network and the ten physical parameters together, starting with
  every parameter 10 % off, which in the loop gives 2.30 times ten to the minus five metres. Left:
  both reach the same fit, 5.86 and 5.89 times ten to the minus six metres, half a percent apart,
  so the projection costs no accuracy. Right: the parameter error, the root mean square over the
  ten parameters in percent. Both improve from 9.5 %, but neither gets close to zero. With
  projection it reaches 7.0 % at epoch 70 and then rises to 7.7 %; without projection it ends at
  7.2 %. Some single parameters even get worse: without projection one drive damping ends 17 % off,
  with projection the joint stiffness 15.5 % and the payload offset 14 %. Why not zero? Part of
  the answer is known before any training: the hidden absorber's own dynamics overlap the
  parameter directions, rho star is 0.20 on these data, so even a perfect fit shifts the
  parameters, by about 5 % predicted. Both runs end above that, and their per-parameter pattern
  differs from the prediction, so they have not reached that point either. So projection keeps the
  network out of the parameter directions, but it recovers the true parameters only when the
  missing dynamics are themselves orthogonal to them."

### Slide 16 (0:45). The learned block makes the physics model 3.6 times more accurate; keeping the parameters physical is only partly solved
- Bullets:
  - Augmentation: 2.06·10⁻⁵ m to 5.77·10⁻⁶ m on held-out records, in closed loop; a smaller block nearly suffices
  - Orthogonal projection: no loss of fit, overlap removed on training data (0.33 on held-out data); parameters not recovered, partly because the hidden dynamics overlap the parameter directions (ρ* = 0.20)
  - Scope: noiseless simulation, no friction, one training run per setting. Next: noise and friction, and a black-box comparison
- Notes: "To conclude. A learned dynamic block next to the physics model cuts the closed-loop error
  on held-out records 3.6 times, and a small block nearly suffices. Orthogonal projection keeps the
  learned block out of the parameter directions on the training data at no cost in accuracy, and
  partly on new data. It does not bring the physical parameters back here, partly because the
  hidden dynamics themselves overlap the parameter directions. All of this is
  noiseless simulation without friction, one training run per setting, with an earlier version of
  the pipeline. Next are the same comparisons with measurement noise and friction, and against a
  black-box model trained on the same data. Thank you."

### Divider slide: "Backup"

### Backup B1. We validate in closed loop because the stage has no spring to the ground
- Layout: `validation-loop.png` full width, about 11 in wide; bullets below it.
- Bullets:
  - X and Y are free integrators: in an open-loop simulation a small force error integrates into drift
  - The machine always runs under feedback; the validation uses its known controller
  - Model input = recorded input + the controller's response to the model's tracking error
- Notes: "Why closed loop? X and Y have no stiffness to the ground, so in an open-loop simulation
  any small force error integrates twice into a growing position error. Unlike the physics model,
  the augmented model drifts away in a long open-loop simulation; that is a real limitation for
  open-loop use and an open problem in the thesis. The machine itself always runs under feedback,
  so we validate the same way: the model gets the recorded input plus the known controller's
  response to the difference between the recorded output and the model output. The model state is
  never reset to the data."

### Backup B2. Orthogonality and the projection in formulas
- Layout: `orthogonality-formulas.png` centred, about 9.5 in wide. No bullets.
- Notes: "The formulas behind slides 12 to 14. Phi j is the derivative of the physics model's
  one-step prediction with respect to parameter j, at every training sample. Orthogonality is the
  inner product over those samples being zero, for all ten. Why it matters: refitting the
  parameters to data that contain Delta moves them by the least-squares term shown, which is zero
  when Phi transpose Delta is zero. The projection subtracts Phi times c from the network output,
  with c fitted on the training samples; there it is exactly the orthogonal projection. On
  held-out samples c is not refitted, which is why the overlap there is 0.33 and not zero. Rho is
  the norm of the projected part over the norm of Delta."

### Backup B3. All ten physical parameters, with and without projection
- Layout: `params-all.png` full width, about 12.4 in wide.
- Bullet: Error of each parameter combination against its true value, 0 = exact; every parameter starts 10 % off
- Notes: "All ten parameter combinations. The total mass, yaw inertia and payload drive damping
  move towards the truth in both runs. The joint damping does not move at all. Without projection
  one drive damping drifts to 17 %; with projection the joint stiffness and the payload offset
  drift to about 15 %. So the projection redistributes the error over the parameters rather than
  removing it."

### Backup B4. Settings of the four training runs
- Draw: a table with these rows and columns, text exactly as given:

| | 8-state block | 2-state block | no projection | with projection |
|-|-|-|-|-|
| Learned states | 8 | 2 | 8 | 8 |
| Network | 3 layers × 24 | 2 layers × 16 | 3 layers × 24 | 3 layers × 24 |
| Learned weights | 10 871 | 4 019 | 10 871 | 10 871 |
| Encoder window | 29 samples | 17 samples | 29 samples | 29 samples |
| Training epochs | 350 | 350 | 150 | 150 |
| Physical parameters trained | no | no | yes, from 10 % off | yes, from 10 % off |
| Orthogonal projection | no | no | no | yes |

- Line under the table: "All runs: noiseless simulated data, controller in the loop, 4 kHz, 14
  training and 4 held-out records, one training run each, single precision."
- Notes: "The settings, if needed. The block-size comparison and the projection comparison are
  separate pairs of runs; within each pair only the listed settings differ."

## Part C. Thesis context (background for consistency; never put this text on slides)

Use this to keep the deck consistent with the thesis. It adds no new facts for the slides: every
slide fact is already in Part B. Do not use any other source about the thesis, including earlier
conversations, where it conflicts with Part B or C.

### Thesis structure and where each slide belongs
| Thesis section | Content | Slides |
|-|-|-|
| I Introduction | industrial motivation (digital twin, test a changed controller first), modelling gap, research question, four aspects | 2, 3 |
| II Dual-drive gantry and physics baseline | coordinates X, Θ, Y; equation of motion; inertia depending on Y; ten identifiable parameter combinations | 4 |
| III Dynamic augmentation | parallel augmentation with learned states, encoder and state initialisation, closed-loop training | 5 to 9 |
| IV Interpretability by orthogonal construction | the learned block taking over the parameters' job ("negation"), orthogonal projection of the physical state update, condition for true parameter recovery | 12, 13, 15 |
| V Experimental setup | simulated machine with hidden absorber, excitation, training protocol, metrics | 5, B4 |
| VI Results | closed-loop formulation, predictive benefit and learned-state contribution, interpretability and orthogonality, generalisation and black-box comparison | 10, 11, 14, 15, B1 |
| VII Discussion, VIII Conclusion | answers per aspect, limitations, future work | 16 |
| Appendix | parameter identifiability, orthogonality and experiment details | B2, B3 |

Four research aspects: (1) a position-dependent physics baseline, (2) an augmentation structure
that captures dynamics the baseline lacks, (3) keeping the physical parameters interpretable
during augmentation, (4) generalisation against a black-box model. This talk shows (1) to (3) in
noiseless simulation; (4) is future work.

### Notation (the talk uses exactly these; name the idea in words first, the symbol second)
| Symbol | In words |
|-|-|
| X, Θ, Y | cross-arm position, cross-arm yaw angle, payload position |
| X₁, X₂, Y | the three measured positions |
| M(Y), C, K, u | inertia (depends on Y), damping, stiffness, motor forces |
| x̃ | physical state (positions and velocities of X, Θ, Y) |
| x̄ | learned states of the augmentation |
| f_base, h_base | physics model (one time step), physics output map |
| φ_aug, S | neural network, router |
| f_aug, g_aug | correction to the physical state update, update of the learned states |
| h_aug = 0 | no learned output |
| ψ, z⁻¹ | encoder, state memory (one sample) |
| θ | the ten identifiable physical parameter combinations |
| Φ_j | parameter direction j: change of the physics model's one-step prediction when parameter j changes, per training sample |
| F, Δ | network output before projection, learned correction that reaches the model |
| ρ, ρ* | overlap of the correction with the parameter directions; the same for the true hidden dynamics |

### What the supervisors expect
- No formula unless it is explained completely (company supervisor). Formulas only as the
  attached images.
- Every result slide states the criterion first, then the result, then the verdict and why
  (university supervisor).
- Honest scope: simulation only, noiseless, no friction, one training run per setting.

### Figure guide: make each figure as simple as its message allows
The attached figures are detailed versions. For each slide below: the one message the figure has
to carry, what it must show for that, what can go, and the numbers you can use. A simpler figure
that carries the message is better than a complete one. Name each block or idea in words first; a
symbol only after the audience knows what it does. The only limit: every number and data curve
comes from the attached figures or this brief.

**Slide 3, grey box.** Message: the grey box keeps the physics and learns only what is missing.
Must show: three options side by side, the grey one highlighted. Can go: everything but one plus
and one minus per box.

**Slide 4, the machine and its equation.** Message: three coordinates, and the inertia changes
with where the payload is. Must show: two parallel drives (X₁, X₂) carrying a cross-arm; a payload
moving along the arm (Y); the cross-arm can translate (X) and yaw (Θ); the flexible joints resist
only yaw, nothing holds X and Y to the ground. Can go: mass labels m₁, m₂, m_b, m_h, dimension
lines, d, L_b, the xyz frame, the joint parameter labels. The attached `gantry-baseline.png` is the
detailed thesis version. Equation message: one equation, terms in words (inertia depends on Y,
damping, stiffness only in yaw, motor forces).

**Slide 5, what the physics model misses.** Message: the simulated machine has a hidden mass on a
spring and damper on the payload; the physics model does not know it. Must show: the payload with a
smaller mass attached by a spring and a damper, marked as hidden or absent from the physics model.
Numbers: half of the 10.1 kg payload, resonance at 212 Hz, excitation 140 to 230 Hz. Can go: the
rest of the gantry detail.

**Slides 6 to 9, the augmentation in four steps.** One message per step: (1) the physics model
moves the physical state one time step forward and the output map gives the positions; (2) a
neural network adds a correction to that state update, the output stays physical; (3) the network
gets states of its own, so it can represent dynamics such as a resonance; (4) an encoder estimates
the starting state from past inputs and outputs. Must show: the physics block, the added
correction entering the state update (not the output), the network's own state loop, the encoder
feeding the start state. Can go: the router S and the three separate +-nodes (one "+" into the
state update, labelled "correction to Θ, X, Y" is enough), the per-channel split, input and state
bus details, symbols beyond the words.

**Slide 10, closed-loop result.** Message: on held-out records the augmented model's error is 3.6
times smaller than the physics model's. Must show: physics model against augmented model, in
metres. Numbers: RMS over 4 held-out records and 3 positions, 2.06·10⁻⁵ m against 5.77·10⁻⁶ m;
on the shown record's payload axis, 3.55·10⁻⁵ m against 9.61·10⁻⁶ m. A simpler option: two bars
with these numbers; the attached trace can then be smaller or go to the side.

**Slide 11, block size.** Message: 2 learned states do almost as well as 8. Numbers: 5.95·10⁻⁶ m
(2 states, 4 019 weights) against 5.77·10⁻⁶ m (8 states, 10 871 weights); physics model
2.06·10⁻⁵ m. A simpler option: three bars; the learning curves are the detailed version.

**Slide 12, orthogonality.** Message: the correction must be something no change of the physical
parameters can produce, and this is checked on the training data. Must show: the correction and the
parameter directions, and "zero overlap" between them. A picture of "a parameter change could
explain this" against "no parameter change can explain this" carries it better than the formula.

**Slide 13, projection.** Message: the part of the network output that a parameter change could
produce is removed; the rest is kept. Must show: an arrow, its shadow in a plane (removed), the
perpendicular remainder (kept). Keep the right angle exact.

**Slide 14, overlap.** Message: projection removes the overlap on training data and cuts it on
held-out data. Numbers: without projection 0.87 (training) and 0.87 (held-out); with projection
8.3·10⁻⁸ (training) and 0.33 (held-out).

**Slide 15, fit and parameters.** Message: same accuracy with and without projection, but the
parameter error is not better. Numbers: fit 5.86·10⁻⁶ m without, 5.89·10⁻⁶ m with projection,
start 2.30·10⁻⁵ m; parameter error 9.5 % at the start, 7.2 % without, 7.7 % with projection,
predicted floor about 5 %. A simpler option: two small bar pairs (fit, parameter error).

**Backup B1, validation loop.** Message: the model is driven by the recorded input plus the known
controller reacting to the model's own error, like the machine. Must show: model, controller, the
two summing points with signs.
