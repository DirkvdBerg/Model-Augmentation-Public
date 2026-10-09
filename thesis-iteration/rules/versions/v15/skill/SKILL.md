---
name: thesis-section
description: Plans, drafts and restructures a section or subsection of the IEEE-format thesis paper in Thesis-writeup/Writing (sections/NN_*.tex), delivering finished LaTeX in a logical order and a concise expert register, with the section structure modelled on Jan Hoekstra's augmentation papers. Use whenever the user asks to write, draft, rewrite, restructure, expand or improve a thesis section, subsection or paragraph, or says a section is messy, cramped, unstructured, too long or does not read well.
---

# Writing a thesis section

The goal is a finished section that an examiner in system identification reads
once and follows. Three properties make that happen:

1. **Logical order.** The reader always knows why the next paragraph comes
   next, and every argument starts from something already established.
2. **Concise.** Every sentence states something the reader needs and does not
   already have. The equations carry the content; the prose adds only what the
   equations cannot say.
3. **Plain expert register.** Use the correct technical term (LPV, LFR,
   self-scheduled, zero-order hold, observability) and state the point. Do not
   explain standard terms, decorate, or try to sound more than the content is.

Dirk rewrites the sentences afterwards in his own voice, so structure, argument,
equations and citations must be final when you hand the section over.

Jan Hoekstra's papers are the model for **structure**: section order, how a
subsection opens, where citations sit, what goes to the experiment section.
They are not the model for wording. Read
[reference/style-profile.md](reference/style-profile.md) for the structural
patterns.

## Concise: the reference example

This paragraph from an earlier draft (105 words) is the failure to avoid:

> The baseline transition $f_{\mathrm{base}}$ is one step of the classical
> fourth-order Runge-Kutta (RK4) method over the sample period $T_s$, applied
> to the state equation of (eom). The input is held over the step, as the
> actuator force is held between samples. Each RK4 stage evaluates $M(Y)$ at
> its own state. As a result, the scheduling variable $Y$ moves within the
> step, and $f_{\mathrm{base}}$ remains self-scheduled. Drenth formulates the
> augmentation of such a self-scheduled LPV-LFR baseline, including additional
> states. Inputs, outputs and states are normalised to zero mean and unit
> standard deviation with statistics of the training records. The RK4 step is
> applied in these normalised coordinates, which changes the coordinates but
> not the dynamics.

Dirk approved this version (about 40 words) as the target style:

> $f_{\mathrm{base}}$ is one RK4 step of (eom) over $T_s$, with the input held
> over the step. Each stage evaluates $M(Y)$ at its own state, so
> $f_{\mathrm{base}}$ remains self-scheduled. All signals are normalised with
> the statistics of the training records [cite].

What was removed, and the rule each removal follows:

- "classical": decoration. Drop adjectives that add weight, not information.
- why the input is held: obvious to the reader. Do not justify what an expert
  takes for granted.
- "As a result, the scheduling variable moves within the step": restates the
  previous sentence. Say each fact once.
- the Drenth sentence: a different job. It belongs where the structure is
  positioned against the literature.
- "changes the coordinates but not the dynamics": obvious. Same rule.

Concise does not mean fewer, longer sentences. The target version has three
short sentences, each one complete point. Concise means nothing in a sentence
could be deleted without the reader losing information. When cutting words,
do not merge what remains: a semicolon or an "and" that joins two independent
facts makes two sentences.

The second failure to avoid is packing. This sentence from an earlier draft
holds three facts:

> Gy\"or\"ok et al.\ either update the expansion point at every objective
> evaluation or fix it at the nominal parameters [cite]; the first recomputes the
> SVD at every evaluation, and the second is argued valid because nominal
> values typically align with the system.

To the point, the same content reads:

> Gy\"or\"ok et al.\ update the expansion point at every objective evaluation
> or fix it at the nominal parameters [cite]. Updating recomputes the SVD at
> every evaluation. A fixed point is argued valid because nominal values
> typically align with the system.

Cutting and splitting work together: first delete what the reader would not
miss, then give each remaining fact its own sentence.

These examples fix the style, not the facts: a fact in them (the RK4 step
count, a citation) is confirmed in the code or source like any other before
it is written.

## Rules that produce this style

- **Deletion test.** For every sentence, ask what the reader loses if it is
  deleted. If nothing, delete it. Apply the same test to every clause and
  adjective.
- **Say each fact once.** No sentence restates the previous one, an equation,
  an earlier subsection of the same section, or an earlier section. Refer back with a pointer ("Section II-B") instead of
  re-explaining. A pointer replaces the restatement, it does not license one:
  an opening that needs an earlier result names it in one clause with its
  pointer. A sentence that lists properties is not followed by one sentence
  per property: keep either the list or the sentences.
- **Build only on established context.** Every argument starts from something
  an earlier section or an earlier paragraph has stated. If the argument needs
  a fact the thesis has not introduced yet (a system, a dataset, a mode), it is
  in the wrong place or needs that fact introduced first.
- **Argue from the general case, not an assumed instance.** When the thesis
  has established only a general fact, do not state a specific mechanism as
  if it were known. Dirk rejected "An omitted vibration mode requires
  additional model order" at the start of the augmentation section: at that
  point the thesis has only established that the baseline is rigid-body, and
  which dynamics are missing is introduced later (Setup). The argument that
  fits is the general one: "If the omitted effects are dynamic, the six
  baseline states cannot represent them. Refitting $\theta_{\mathrm{base}}$
  adds no states, so the augmentation needs states of its own." Leave the
  specific instance to the section that introduces it.
- **Equations carry the content.** Around a displayed equation, write a lead-in,
  a "where" clause for new symbols, and at most one sentence on its purpose,
  assumption or consequence. Do not paraphrase what the equation shows: a
  sentence that reads the display back (which entries a variable enters, which
  term it adds) is not the consequence and is deleted, and so is a sentence
  that already holds by the display's own lines; when the display fixes an
  implementation choice (a differentiated against a detached coefficient), the
  consequence is what the rejected alternative loses, from its decision. The same holds for an
  inline formula; a read-back list is deleted, not split into sentences.
- **Expert register.** Name the concept with its standard term and do not
  define it. Write the plain verb ("is", "uses", "gives"). No hedged purpose
  clauses ("intended to represent"), no announcements ("In this subsection we
  describe ...", "The learned part rests on five choices"), no repeated pointers to where values are given (one pointer to
  Experiment design per section suffices). The roadmap paragraph (Order) is the
  one exception: it states the goal from what is established and names the
  parts, as in Hoekstra 2026 Sec. 5 ("Given the model (6), our goal is ... To
  this end, ... consists of A, B and C").
- **Justify only what an examiner would question.** A modelling choice that
  differs from the cited method, or that a reader could reasonably do
  differently, gets its reason. A standard choice gets none. The reason names
  what the alternative would lose on the gantry (the property the choice
  keeps), so an examiner can check it; "as derived" or "simpler" is not a
  reason. The reason, also one for a scope or non-transfer claim (a cited
  theorem that does not apply), must
  hold against every alternative the thesis itself uses elsewhere (a
  closed-loop simulation in another section is such an alternative); for a
  non-transfer claim it is the setting difference the cited passage states.
- **No em dashes.** Use a comma, colon, parentheses or a new sentence.

## Order

- Top down. The section opens with one short paragraph that states what it
  delivers and names its subsections in order. A subsection opens with the
  problem it solves (a few sentences, each limitation stated once), then the
  construction, then its consequence. Its first sentence says what the thesis
  needs at this point and lacks, never a construction or a fact about one;
  when the roadmap or an earlier section already states that need, the
  sentence names it in one clause with its pointer and goes on to what this
  subsection adds; a
  `MUST ESTABLISH` item's argument starts after that sentence. Literature positioning sits at the
  component it contrasts, one sentence per cited method, never as a block
  before the construction. A `MUST ESTABLISH` item that names several sources
  for one motivation is argued in this thesis's terms in the problem
  paragraph, each source cited once, on the one sentence it supports. A subsection that adopts or adapts a cited method
  opens with the problem in this thesis's terms; that method's facts (its
  setting, its steps, its lemma) appear once, in the lead-in of the display
  that adopts or adapts them.
- One job per subsection. A subsection answers one question: what the model
  is, what a component computes, what is minimised, or how it starts. A
  construction that the criterion evaluates (an encoder, a closed-loop
  simulation) is a component with its own subsection before the problem. The
  identification-problem subsection holds only, in this order, the
  coordinates and the trained parameter vector, the criterion display with
  its constraints, the optimiser sentence and the selection rule (Hoekstra
  2026 Sec. 5); a consequence of that problem (non-uniqueness of a split, a
  handover to a later section) follows its display, never precedes it. A
  component's refresh or update rule (when its basis or target is rebuilt,
  from what, against which alternatives) is argued in the component's
  subsection; the problem display's "where" clause only names it, with a
  pointer.
  Within a component, a display whose output feeds a later display (data
  tuples feeding a fit) is introduced as that display's input: the lead-in
  names what it feeds before it is built.
- Known before new. Every term, symbol and signal is introduced before it is
  used. A term that only a later section defines (the data-generating system,
  the stroke) is replaced by established wording or carried by the section's
  one pointer to that section; pointers inside `\todo`s and header comments
  do not count against the one-pointer rule, nor does a symbol introduced
  here whose value Section V lists (named in the header).
- Topic sentences. The first sentence of every paragraph states its point. Read
  in order, the first sentences alone tell the section's argument. A
  paragraph that opens with a demonstrative ("These directions") refers to
  an object the previous paragraph named with that noun.

Order by kind of section:

- Method: problem, construction (with its equation), consequence; one component
  per subsection. The consequence says what the construction gives the rest
  of the thesis; when the implementation computes the same model in another
  form (a closed-form inverse instead of the LFR loop), one sentence states
  the equivalence and what the construction is still for, in a paragraph of
  their own, never how the code evaluates the form; "still for" names a
  display or analysis of this thesis that needs the construction's structure
  (coefficient matrices an analysis uses, a block another model takes in),
  never only the section that reuses the model, a property of the form, a
  use in a cited method the thesis does not follow, or a condition the
  construction itself needs (its well-posedness is its consequence); no
  earlier sentence of the section may already meet that need; that use is the subsection's need in its
  first sentence, else a `\todo`. Each model ends complete: what is trained,
  what is fixed, what it starts from, against which cost; a compared model
  that is not trained states its initial state and whether it is simulated
  open loop on the recorded inputs or in a closed loop, else a `\todo`; each phase of a training or
  initialisation states which parameters it changes, shared ones included,
  and so what the next phase starts from; a model that reuses another model's
  phases (the PS2 stages, a polish) states per phase whether it sees the new
  term and what that phase then changes, never only "as for" the other model,
  validation and selection counted as phases; a sentence summarising the
  phases claims only what holds in every one of them;
  a model whose identification problem
  reuses another's shows the substitution in its own display or line, not as
  "(x) with u replaced"; it displays the cost and the lines that change, and
  references the unchanged constraints inside that display by equation number.
  Constructing a component and stating the identification problem of the model
  that uses it are two jobs, so two subsections; the first displays the
  component, the second the model with it substituted (README Math standard
  item 3). The limits of a guarantee are
  the assumptions of its formal statement, written as one list ("Suppose (i)
  ... (ii) ..."), which the packing rule does not split; they are not a run of
  sentences before it. Every listed assumption is used in the proof; a
  condition that only makes another plausible goes in the prose after it. The
  statement concerns the construction the code trains, or names the variant it
  concerns (a restricted tuple set) as such. An assumption the gantry meets only locally (affine,
  linearised, a slice) is named as such in its scope, and every prose sentence
  that states the guarantee carries that qualifier ("to first order", "on the
  tuples") and any rank or excitation assumption it needs, or points to the
  statement that lists it. Scope items that are not assumptions (a surrogate criterion, a
  deliberately omitted term) form one scope paragraph after the proof, holding
  only what an examiner would question about this guarantee, each item once,
  the most consequential first.
- Experiment design: the data-generating system in equations, the data, the
  compared models, the metrics, then the hyperparameter table with an honest
  basis per value (derived, inherited, tuned, compute-limited). It never
  explains training.
- Results: the criterion before the evidence, then the evidence (figure or
  table), then the verdict against the criterion.
- Discussion: the claim, its limits, and what would test it.

## Equations

Follow the README's "Math standard" and "Paper structure and ownership"; they
are authoritative. In short: for every part of the method, the math shows
concisely how it works on the gantry; derivations and proofs only where they
are this thesis's own contribution, and only as far as needed. When the
section must shrink, cut prose, not these equations.

## Content

**The reader test.** A paragraph stays if an examiner needs it to understand
the method, reproduce it, or check a claim. Implementation detail (how the code
stores a signal, when the model is the same either way), optimisation mechanics
(which weights receive a gradient first, step sizes) and development history (earlier
variants, dropped settings) fail the test. Which criteria are minimised at
the same time, over which parameters, and when a phase starts and ends is not
mechanics: it defines what is trained and is stated in the phase's lead-in. Two exceptions stay: a mechanism that
is the problem a component solves gets one sentence as that problem, and a rule
that defines the method (a switching or acceptance rule) gets one sentence with
its condition as the code states it, every compared quantity named (never "a
monitored quantity"), placed after the phase it ends. When the condition
holds more than two facts, the packing rule wins and it takes more sentences;
a threshold gets a symbol only when Experiment design lists its value. A
fallback for a failing case (keep the previous basis, abort after repeated
failures) is mechanics, not a method-defining rule; a rule that an agreed
README sketch names (the PS2 screen) is method-defining, in the form the
sketch gives. A rule that uses the true parameters is named as such
and gets a `\todo`. If anything else matters later, record it
as an open point. Decision IDs and code paths never appear in prose, only in
`\todo`s and header comments.

**The header.** Read the section's `WRITING GUIDE` header as content to
consider, not to cover at equal depth. Items under "Cautions" and
"Research-plan anchor" are requirements: each appears in the section, or the
outline says why not. A Caution whose source no decision entry, meeting
audit or research plan holds is covered as a candidate, with a `\todo`
naming the missing source. When such an item conflicts with another rule here (an
instance not yet introduced, development history), cover it in the form that
rule allows and add a `\todo`. Items under "Choices to justify" are
candidates: each passes the reader test or is left out, except a choice that
differs from a method the section cites (continuous instead of discrete time,
a non-minimal realisation), which gets its reason or a `\todo`; a kept choice that a
display needs goes into that display's lead-in, the others form the one
realisation-choices paragraph of their component after its last display,
opened by what they jointly fix (README "Prose around the math"); the packing
example's three sentences are one such choice inside that paragraph, never a
paragraph per item. A kept item
states its content (the relation, the reason); a sentence that only says the
item exists ("A and B are separate quantities") or disclaims a property
("is not claimed to be minimal") fails the deletion test and becomes a
`\todo` instead. A header item
that the README ownership table gives to another section is left to that
section and named in the header comment.

**Reasons and claims.** State every claim as strongly as its evidence supports,
and no more; a claim about a display is no stronger than the display (an
$O(\cdot)$ remainder makes it a first-order claim). Every stated reason traces
to a cited source, a decision entry in `docs/decisions.md`, the code, or a
documentation file the README source map lists. A `\todo` candidate meets the
same rules. When the reason for a choice is owned by a later section
(a data-design reason in Experiment design), state the choice with a one-clause
pointer to that section; a `\todo` only if no section owns it. A reason you
derive yourself becomes an open point with your reason as the candidate, and
is not also stated as a reason clause in the prose. A decision entry that
records only an instruction or a judgement ("do not discretise first") is
the source of the choice; the loss it guards against is stated with it when
the entry or a display of this section shows that loss, else the loss is a
`\todo` candidate. A pointer to what Results reports, and the purpose
sentence of a compared model, name only what `DOC/THESIS-RESULTS.md` plans
to report, in its terms (a demonstration is not a measurement), else a
`\todo`.
Before writing a `\todo` for a missing reason, search the part's source-map
row (decisions, listed documentation, `citation-log.md`): a reason found there
is stated in the prose with its source, not left as a candidate. A reason that
is a measurement (a basis that turns slowly, a dominant term) is stated as
observed, without the number, with a pointer to the section that reports it,
or a `\todo` if none does. When the source of a choice also records a caveat
(a case the choice leaves uncovered), the scope states it with the reason the
choice still holds here, never as a cited sentence arguing against the
choice with no answer beside it. A wording a
decision rules out or constrains ("not 'prevents negation'") binds every
sentence of the section, roadmap, topic sentences, lead-ins, consequence and
summary sentences included, not only the scope paragraph; its synonyms
("removes", "eliminates") are ruled out with it. A derivation that is part of the
method (this project's adaptation of a cited result) belongs in the prose,
marked as this work's. A claim whose evidence comes from superseded data gets
an open point.

**Open points are `\todo{}`s in the file.** `\todo` prints in red in the draft
build and disappears with `\draftfalse` (`util/format.tex`), so it costs nothing
at submission. Resolve everything you can yourself first; a `\todo` marks only
an open decision for Dirk or a fact you could not verify, never a style remark
or a reminder to yourself. Place it at the end of the paragraph it concerns,
never inside a sentence, one per item (the open items of one display share
one `\todo`, lettered (a), (b)), in this form:
`\todo{Missing: <reason, source, check or notation decision>. Candidate: <answer> (<basis>).}`
Leave out "Candidate" when there is none. `because: ?` is written as such a
`\todo`. Existing `\todo`s are removed only when Dirk decides them or the draft
resolves them (one that records its own resolution is resolved); the drafting
session does not drop them silently, and a kept one is rewritten in this form.
A `\todo`, bullet or display line in another section's file that this draft
resolves or makes stale is named, with its file and line, in the header
comment and the delivery report (search the other files for every label this
section defines and recheck each citing line); that file is not edited. The appendix part this section owns (its proofs and
expanded matrices) is updated where the draft makes it stale. A pointer to
an appendix part or another section names a result stated there; a result
not stated there is stated here or is a `	odo`.

**Draft bullets per (sub)section.** Below each `\section` and `\subsection`
heading, the file keeps a draft-only list of what that part must state:

```latex
\ifdraft
\begin{itemize}
\item <one line: a point this part must state> (<source>)
\item <an open point> (?)
\end{itemize}
\fi
```

The bullets are content, one line each: what the reader must learn, not
planning notes (sources lists, figure plans and page budgets belong in the
header comment). They print in the draft PDF and disappear in the submission
build. Every bullet is covered by the prose and every paragraph maps to a
bullet; when the content changes, change the bullet first. Existing bullets are
kept and updated, not deleted.

**Notation.** Keep the symbols the other sections use, earlier and later ones,
even where the code or a paper names a quantity differently. A change that
affects another section is Dirk's decision: keep the existing symbol and list
the conflict. One symbol has one meaning in the whole thesis, inside this
section too: a new quantity (a normalised state, a horizon, a residual
statistic) gets a new symbol, checked against `util/format.tex` and the
symbols of every section before use. A decoration (bar, tilde, hat, star) is
part of the symbol and keeps one meaning across base letters. One meaning
also binds a letter used as a prefix or index ($\Delta$ names the scheduling
block, so not a displacement $\Delta\xi$ too) and a map's argument list: a
per-sample map and its stack are two symbols (bold stack, as
$\boldsymbol{f}_{\mathrm{aug}}$). When another section's symbol for
this quantity already carries a different meaning, one-meaning wins: use a new
symbol here, leave the other section unchanged and add a `\todo` naming both.
The same holds for a clash between this section's symbol and a later
section's: the `\todo` goes in this section. A clashing symbol that other
sections already use is kept everywhere, this section included, with the
`\todo`; renaming it is Dirk's decision. An operator or symbol quoted from
a cited result (a spectral radius, a norm) is checked like an introduced one.

**Contributions.** The Introduction lists the contributions; each section
makes its own share visible. The section opening states which contribution it
delivers, in the Introduction's wording; when the section's own share is wider
or narrower than that wording, the opening claims what the Introduction states
and a `\todo` proposes the new wording. Every component is one of three kinds, and the prose shows which:
adopted (cite it, "as in [X]"), adapted (cite it and say what differs, "unlike
[X], we estimate $\theta_{\mathrm{base}}$ jointly"), or this work's own (state
it with "we", no citation). A display, or a prose method citation
("following [X]", an optimiser, a schedule), that differs from the cited one
in any index range, signal or convention is adapted, not "as in"; when two
cited sources with different conventions feed one display, the prose says
which one it follows. The gantry-specific realisation (scheduling through
$M(Y)$, the $P$ frames, the free axes, the identifiable combinations) is this
work's and is presented as such, not as a neutral fact. The contribution map in
`reference/sources.md` says which is which.

**Citations and values.** Cite each borrowed component once, where it is used,
with a verb that states the relation ("we adapt", "as in", "following").
Consecutive sentences drawn from the same cited location carry the citation
once, on the first; it is repeated only after an own sentence intervenes or
the location changes.
A method section defines the symbol; its value is a row of the hyperparameter
table in Experiment design, unless the value is itself part of the argument. Report errors in metres in scientific notation. Refer to every figure
in the text where the reader first needs it.

**Length.** The page budget in the header is a target, not a cap. Length
follows from two rules together: everything the section must state is in it,
and nothing in it fails the deletion test. Never drop required content to make
a section shorter; cut only what the reader would not miss. A section more
than 1.5 times its budget (measured by compiling a `\draftfalse` copy in
the scratchpad, figures counted) gets a second
deletion pass aimed at sentences appended after displays, literature
positioning and optimisation mechanics; if it is still over, the delivery
report names the required content that causes it, and, when the budget is
older than the section's ownership row or an agreed sketch, proposes a budget
from the displays they require. An agreed sketch in the
README (for example PS2) fixes the displays and the jobs of the sentences.
More than 1.5 times its sentence count fails the review, and the cut starts
with the content the sketch excludes. Both tests apply: the sentence count to
the sketched part, the page test to the whole section.

For sources and verification, `Thesis-writeup/Writing/README.md` stays
authoritative: its "Paper structure and ownership", "Math standard", "Inspect
sources in this order", "Claim rules", "Derivation policy", "Mandatory
methodological reading" and "Reference verification gate".

## Workflow

Copy this checklist into your reply and keep it updated:

```
- [ ] 1. Read: the README (ownership, Math standard, this section's source-map row), preceding sections in full, the sections that use this one, the header, the style profile, the sources, the final code path
- [ ] 2a. What the section must establish (claims, left-out content, source conflicts); wait for Dirk
- [ ] 2b. Paragraph outline from the agreed claims; wait for Dirk's approval
- [ ] 3. Draft the full section into the .tex file
- [ ] 4. Review loop until every check passes
- [ ] 5. Deliver: length, todos added or resolved, what moved elsewhere
```

**1. Read.** Read every preceding section in full and skim the sections that
use this one. Note what they establish (systems, signals, symbols, claims) and
what they do not yet establish: this section may build only on the first.
Then read the README's "Source map per subsection" row for this section
(the per-part list of papers, decisions, code and documentation) and
[reference/sources.md](reference/sources.md). The latter gives the
configuration of the thesis runs, the code that implements each concept, the
superseded paths to ignore, how to use `docs/decisions.md` and
`docs/references.md`, the contribution map, and known traps. The map is a starting
point, not an authority: it was written by an earlier session. For every
configuration value, mechanism or decision this section states, open the code
or the decision entry yourself and confirm it; when the map and the code
disagree, the code wins and the disagreement becomes a `\todo`. Describe
the method as the code defines it, never from the base configuration or from a
superseded decision. Note which contribution of the Introduction this section
delivers.

Read the literature this section relies on yourself. Use the README's
"Mandatory methodological reading" table and `docs/references.md` to find the
PDFs under `literature/`. For every citation in the section, read the passage
it points to (definition, equation, theorem, remark) and confirm it supports
the exact sentence. Read the method sections of the papers this section adapts
in full, so the difference between their method and ours is stated correctly.
A summary in a decision entry, the source map or a code comment does not
replace the paper.

**2. Outline, in two approvals.** What a section must capture is Dirk's
judgment; the sources only suggest it. So agree on content before structure.

*2a. What the section must establish.* Present in chat, before any outline:

- three to six claims the section must establish, each in one sentence, each
  with what needs it (an Introduction contribution, a later section that uses
  it, a header Caution or anchor item, an examiner question);
- candidate content you propose to leave out or move, with the reason;
- points where the sources disagree (section text, decisions, code, papers),
  with your reading of each.

Wait for Dirk's answer and adopt his corrections. If the section header holds
a "Must establish" block from an earlier session, start from it and show only
what you would change. A block fixes claims and restrictions, not sentences:
a claim already visible in a display or an earlier sentence needs no sentence
of its own, one item may need several paragraphs (its consequences then follow
the README "Prose around the math"), and a restriction ("no new symbols", "no
display", "no proof environment") binds every display and sentence of the
part (an entry of a defined matrix, $M_{0,33}$, is not a new symbol). A block item is checked like any claim: a citation the paper does not
support at that location is corrected to the passage that does, and a
condition or sign the code or a later decision contradicts (a signed
parameter called positive, a superseded initialisation) is corrected; the
claim is kept and the correction named in the header comment. A block item's
position yields to Known before new: an item that needs a later display goes
after it, in whichever subsection holds that display. A part with no
block gets one inferred from the README ownership and source-map rows, marked
PROPOSED in the header comment.

*2b. Paragraph outline.* Built from the agreed claims: the opening in one
line; per subsection its heading, its job and first its list of displayed
equations (README "Math standard"); per paragraph its topic
sentence, the equation it carries, its sources, and whether it is adopted,
adapted (with the difference) or this work's own; and what moves to
Experiment design or the appendix, with a reason. No word counts. Wait for Dirk's approval, which
then covers the whole section.

After 2a is approved, record the agreed claims as a `% MUST ESTABLISH` block
in the section's header comment, so later sessions start from Dirk's decision
instead of inferring it again. After 2b is approved, write the draft bullets of
every (sub)section into the file before any prose; they follow the approved
outline, with open points marked (?).

**3. Draft.** Write the whole section into its `.tex` file under the draft
bullets, following the outline. The bullets stay. Edit content only and keep the build intact (balanced braces and
environments, escaped special characters, no root directives). Do not compile
unless the task asks for it (then follow the task's build instruction); else,
after Dirk saves and the editor rebuilds, check `build/main.log` for lines
starting with `!`. An overfull box reported on this section's lines (often a
display wider than the column) is fixed by breaking the display; one from a
figure the section inputs is fixed by scaling it to `\columnwidth` at the
input. Edit `.tex` files with the Edit tool: shell heredocs can collapse
`\\` and break matrix rows silently.

**4. Review loop.** Re-read the section as an examiner, then check:

- Deletion pass: test every sentence, clause and adjective; delete what the
  reader would not miss. Do this pass in full, once, before the other checks.
- Nothing restates an equation, the previous sentence, an earlier subsection
  or an earlier section; a topic sentence with a pointer and a sentence after
  a display that repeats its lead-in are checked too. A problem display
  repeating its constraint lines (README Math standard item 3) is not a
  restatement.
- Every argument starts from something already established in the thesis.
  No sentence states a specific mechanism (a mode, an effect, a dataset
  property) as fact before the section that introduces it; the argument uses
  the general case instead.
- No standard term is explained; no decoration, announcement or hedge remains.
- Packing pass: list every semicolon outside an equation, every "and" or "so"
  that joins two clauses with their own subjects, and every sentence over 30
  words outside a "where" clause, figure captions included (a lead-in
  sentence is counted without its display and "where" clause; a "where"
  clause defines the display's new symbols only, and a fact or reason in it
  counts as a sentence; naming a symbol as held fixed, refreshed or solved,
  with its pointer, is a definition). For each, first delete what the
  reader would not miss, then split what remains into one fact per sentence,
  each with its own explicit subject (a pronoun only for the previous
  sentence's subject).
  The section passes when no semicolon or "and" joins two clauses with their
  own subjects, no sentence holds three facts, and no listed sentence
  exceeds 30 words, the roadmap included (a formal assumption list is
  exempt); "X, so Y" with Y following from X is two facts and passes.
- The first sentence of each subsection states its problem (Order); reading
  only the first sentence of each paragraph gives the argument, and
  every later sentence of a paragraph serves its first; a closing sentence on
  another point moves to the paragraph it serves or starts its own.
- Every displayed equation has a lead-in, a "where" clause that gives each new
  map its domain and codomain and each new signal, parameter or matrix its
  space (every symbol it defines appears in that display) and gives the
  invertibility of every matrix the display inverts or divides by, stated
  once in the section and cited by number after that, and
  at most one sentence of meaning; further content goes into the lead-in of
  the display it motivates or into its own paragraph (README "Prose around the
  math"). Count the sentences from a display to the next display or heading,
  a fact or reason inside the "where" clause included: more than one after
  the symbol definitions fails, and a following paragraph counts unless its
  topic sentence opens a new step of the argument. No reason for a
  display's inputs arrives after a later display; a choice that fixes an element of a display, or a condition it needs to exist, sits in its lead-in,
  and only choices no display shows share one paragraph after the
  component's last display, whose first sentence names the one property they
  jointly fix; no run of short paragraphs follows a display.
  Every removed display is textbook machinery and is listed in the report.
- Coherence: read as a whole, the section is one argument (short roadmap,
  one job per subsection, problem, construction, consequence), not a list of
  formula and reason blocks, as in Hoekstra's and Drenth's papers.
- Math: the displayed equations show how each part of this section works on
  the gantry, concisely; no displayed equation is a derivation step that is
  not this thesis's own contribution. A standard step applied to a displayed
  map (a Taylor expansion, a normal equation from a pseudoinverse) is not own:
  its result is one sentence, and it is displayed only when a proof cites it.
  A standard numerical scheme or initialisation distribution (a
  finite-difference stencil, an RK4 tableau, a Xavier bound) is named, with
  its order where it has one, not displayed; it carries a citation only when
  a verified one exists, else none and no `\todo`.
  A structural claim (a map is an LFR, two forms are equal) is visible in a
  display or stated by the equivalence sentence (Order, Method).
- Ownership: no training explanation sits outside a method section; no value
  sits in a method section unless it is part of the argument.
- Every reason traces to a source, decision, code or listed documentation;
  every claim is as strong as its display or evidence, `\todo` candidates
  included, and no reason clause states what a `\todo` of the same part asks
  for; every citation supports the exact sentence it is attached to, index
  ranges and conventions included, and a sentence that compares with or summarises a cited work is checked
  against the passage even when it carries no citation.
- Every header Caution and Research-plan anchor item is covered, and every
  `MUST ESTABLISH` claim and restriction is honoured. Every wording that a
  source-map row used, or a decision it cites, rules out is searched for in
  the whole section with its synonyms; no hit remains.
- Every own mathematical claim, in a display or in prose (a count, a
  determinant, a null or lost direction, an iff condition), was checked in
  this session by algebra or against the code (README Derivation policy),
  against the exact object the sentence names (the equation it cites), in a
  form free of unstated conventions (a magnitude when the sign depends on an
  unstated order); a `MUST ESTABLISH` wording copied into a sentence passes
  the same check (a signed parameter called positive is corrected).
- Every sentence that states a fact, value or term taken from a paper carries
  that citation itself, or follows directly a sentence citing the same
  location (Citations and values); "we call" or "we define" appears only for this work's
  own terms. List every cited location: one that supports sentences in two
  separate paragraphs must support two different claims, else one goes.
- From this section alone, an examiner can name which parts are this work's,
  which are adapted and how they differ from the cited method, and which
  contribution of the Introduction the section delivers. The opening claims
  as this work's only what the subsections mark as own or adapted, and no
  more than the Introduction's contribution states (else a `\todo`); no
  opening adjective ("compact", "exact") is contradicted by a limitation
  the section states. Every "as in", "following" and "as [X] do" is checked
  against the cited display for signal, index range and when the quantity
  is computed; a difference makes it adapted.
- Every configuration value or mechanism described was confirmed in the code
  by you in this session, not only taken from `reference/sources.md`.
- Every citation, also one inside a `\todo` candidate, was checked against the
  passage in the PDF, in the version the cite key names (the log records whose
  numbering a location uses); a passage recorded in `citation-log.md` or a
  source-map cell marked verified for that location counts, any other is
  read in this session and recorded.
- No symbol used by another section was renamed, and no symbol carries two
  meanings: list every symbol the section uses, quoted operators and block
  names taken from a paper ($C_y$ beside $c_y$) included, search the earlier
  and later section files for each and compare its definition there, frame
  included (actuator or generalised force, physical or normalised); after a
  normalisation display, every display uses the normalised symbols and all
  its lines, cited unchanged lines included, one set of coordinates; every acronym is expanded at its first use
  in the thesis, every letter as its source expands it; every figure is first referenced at the display or paragraph
  it depicts; `\subsection` headings are in sentence case (`\section`
  headings keep the title case of the other sections).
- Every draft bullet is covered by the prose, and every paragraph maps to a
  bullet.
- Every `\todo` sits at a paragraph end, names what is missing, and gives a
  candidate with its basis where one exists; nothing a `\todo` asks could have
  been resolved in this session: each `\todo` asking for a reason or source
  is rechecked against the part's source-map row, `citation-log.md` and the
  header's corrections, and an item found there or marked resolved in the
  header is stated in the prose with its source instead.

Fix what fails and check again. Proceed only when everything passes.

**5. Deliver.** Report briefly in chat: the length against the budget; the
`\todo`s you added or resolved, each with its location (the file is the list,
the chat only points to it); and what moved to other sections, so Dirk can
check they receive it.

## Relation to the README phases

This skill replaces the README's "Order of work" steps 2 to 4 and the
sentence-level rules of its "Phase 1: Structure": the deliverable is a finished
section, not a claim list converted sentence by sentence. Draft bullets and
`\todo`s stay in the file until Dirk removes them. The README's Phase 2
(mark), Phase 3 (Dirk's rewrite) and Phase 4 (check) still follow when Dirk
names them, and its source, claim, derivation and reference rules apply in
full.
