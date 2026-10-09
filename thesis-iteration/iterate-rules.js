export const meta = {
  name: 'thesis-rules-iterate',
  description: 'Overnight loop improving the thesis README and writing skill, tested by fresh drafts of Sections III, IV and II in a sandbox',
  phases: [
    { title: 'Cycles', detail: 'per cycle: fresh draft from the current rules, two critics, rules editor' },
    { title: 'Report', detail: 'rules diff, score history and best drafts for Dirk' },
  ],
}

const ROOT = 'C:/Users/20203253/OneDrive - TU Eindhoven/Graduation Project/Baseline FP model/Baseline-LPV-Augmentation'
const SB = ROOT + '/thesis-iteration'
const ORIG = ROOT + '/Thesis-writeup/Writing'
const RULES = SB + '/rules'
const MAX_ROTATIONS = 10
const pad = n => (n < 10 ? '0' : '') + n

const SECTIONS = [
  { key: 'III', file: '03_augmentation.tex', rows: 'III-A, III-B, III-C, III-D',
    task: 'Section III (Dynamic LPV-LFR Augmentation), including a new subsection on the added-state initialisation (PS2, see Thesis-writeup/Documentation/PS2-METHOD.md and THESIS-RESULTS.md; PS2 is part of the reported method)' },
  { key: 'IV', file: '04_obc.tex', rows: 'IV-A, IV-B, IV-C',
    task: 'Section IV (Interpretability by Orthogonal Construction)' },
  { key: 'II', file: '02_system_baseline.tex', rows: 'II-A, II-B, II-C and II Frozen-LTI baseline',
    task: 'Section II (Dual-Drive Gantry System and Physics Baseline)' },
]

const CONTEXT = `
You take part in an overnight run whose GOAL is to improve the thesis writing rules, not to polish one text. First read ${SB}/SESSION-CONTEXT.md in full: it is Dirk's brief (his supervisor Maarten's feedback and Dirk's own direction). Everything you do must serve it.
The rules under test are SANDBOX COPIES: ${RULES}/README.md (copy of the thesis README) and ${RULES}/skill/SKILL.md with ${RULES}/skill/reference/ (copy of the thesis-section skill). Never use the originals: do not invoke the Skill tool, do not read ${ORIG}/README.md or ${ROOT}/.claude/skills/thesis-section/ as rules.
HARD RULES
- Write only inside ${SB}. Never write anything under ${ROOT}/Thesis-writeup/ or ${ROOT}/.claude/ (reading sources is fine). Never touch kamtin-fp-model/ or kamtin-data/.
- Nobody answers questions tonight. Never stop to ask. A choice that needs Dirk becomes a \\todo{Missing: ... Candidate: ... (basis).} at a paragraph end.
- No em-dashes anywhere (not the Unicode dash, not ---, not --).
- Compile check (when you changed LaTeX): from ${SB}/Writing run \`latexmk -pdf main.tex\` (timeout about 3 min); done when build/main.log has no line starting with "!".
- Paper verification: before relying on a paper passage, check ${SB}/citation-log.md; if it is not there, read the passage in the PDF under ${ROOT}/literature/ and append it (key, location, what it supports, date 2026-10-09).
`

const RUBRIC = `
RUBRIC (judge the whole section as an examiner in system identification and control would; SESSION-CONTEXT.md is the standard)
A. Coherence, most important: one argument in the style of Hoekstra's papers (primary) and Drenth et al. 2025 (secondary): short roadmap, one job per subsection, problem, construction, consequence, building only on what is defined. Not a list of formula and reason blocks.
B. The math makes clear how each part works on the gantry: what is built, which signals go in and out (spaces defined in "where" clauses), what is trained and fixed, against which cost. Identification problem and initialisation displayed where the section owns them (Hoekstra 2026 Eqs. 22, 28 to 31; Drenth Eqs. 19 to 21). FAIL if a part is only described in words.
C. Not overdone: no intermediate derivation steps or textbook machinery displayed; proofs only for this thesis's own contribution and short. Gantry-specific realisation, not a general framework paper, and not overstating Hoekstra's framework either.
D. Prose: reasoning before a display, a "where" clause, at most one consequence; no paraphrase of equations; choices justified from a source, decision or code, else a \\todo.
E. Concise expert register per the skill (deletion test, no restatement, no packing, no decoration, no em-dashes).
F. Correct against the thesis-path code and current decisions (source map rows), citations support the exact sentence, notation consistent with neighbouring sections, values in Experiment design unless part of the argument, no training explanation pushed to Experiment design.
G. Compiles.
`

const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    score: { type: 'number', description: '1 to 10 overall against the rubric' },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['major', 'minor'] },
          criterion: { type: 'string' },
          where: { type: 'string' },
          problem: { type: 'string' },
          rule_cause: { type: 'string', description: 'which rule in rules/README.md or rules/skill (quote its heading) caused this, failed to prevent it, or is missing' },
        },
        required: ['severity', 'criterion', 'where', 'problem', 'rule_cause'],
      },
    },
  },
  required: ['score', 'issues'],
}

const history = []
let cycle = 0
let prevMean = null
let noGain = 0

phase('Cycles')
for (let rot = 1; rot <= MAX_ROTATIONS; rot++) {
  const rotScores = []
  for (const sec of SECTIONS) {
    cycle += 1
    const C = pad(cycle)
    const dir = `${SB}/rounds/cycle-${C}-${sec.key}`
    const secPath = `${SB}/Writing/sections/${sec.file}`

    await agent(`${CONTEXT}
TASK (writer, cycle ${C}): draft ${sec.task} FROM SCRATCH using only the current rules.
1. Reset the sandbox: copy ${ORIG}/sections/${sec.file}, ${ORIG}/sections/99_appendix.tex and ${ORIG}/refs.bib over their copies in ${SB}/Writing/ (sections/ and the project root). This restores the original header, MUST ESTABLISH block and draft bullets; the existing prose is a starting point you may rewrite entirely.
2. Follow ${RULES}/skill/SKILL.md (its workflow steps 1, 3 and 4; the approvals of steps 2a and 2b are replaced by the MUST ESTABLISH block, do not wait) and ${RULES}/README.md, including its Source map rows ${sec.rows}. Read the sources those rules point to (code, decisions, papers, Documentation). Do exactly what the rules say; where the rules are silent or unclear, use your best judgment and record it.
3. Write the section into ${secPath}; you may move detail to sections/99_appendix.tex and add verified entries to refs.bib. Compile.
4. Copy the result to ${dir}/draft.tex (create the folder) and write ${dir}/writer-notes.md: max 15 lines, listing where the rules were silent, unclear or contradictory and what you did instead. These notes are evidence for the rules editor.
Return one line.`, { label: `writer:${C}-${sec.key}`, phase: 'Cycles' })

    const lenses = [
      { key: 'examiner', text: 'Lens: Maarten Schoukens reading the section once as examiner (criteria A to D first). Would he see exactly what is built, what is trained against what, and why, without overloaded algebra, in one coherent argument?' },
      { key: 'checker', text: 'Lens: meticulous co-author (criteria E to G and B, C): verify claims against the code and decisions in the source map rows, citations against citation-log.md or the PDF, run the deletion and packing passes, compile.' },
    ]
    const crit = (await parallel(lenses.map(l => () => agent(`${CONTEXT}\n${RUBRIC}
TASK (fresh critic, cycle ${C}, section ${sec.key}): ${l.text}
Read ${dir}/draft.tex in full, the neighbouring sections in ${SB}/Writing/sections/ for context, and the reference papers (Hoekstra 2026 Sec. 5 and 6: ${ROOT}/literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf; Drenth: ${ROOT}/literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf). Do not read other rounds. Do not edit anything except your critique file. Score 1 to 10 and list concrete issues; for each issue name the rule that caused it, failed to prevent it, or is missing (rule_cause). Do not invent issues; the same draft should get the same score from any careful critic. Write your critique to ${dir}/critic-${l.key}.md and return it.`,
      { label: `critic-${l.key}:${C}-${sec.key}`, phase: 'Cycles', schema: CRITIC_SCHEMA })))).filter(Boolean)

    const score = crit.length ? crit.reduce((a, c) => a + c.score, 0) / crit.length : 0
    const majors = crit.flatMap(c => c.issues.filter(i => i.severity === 'major')).length
    rotScores.push(score)
    history.push({ cycle: C, section: sec.key, rotation: rot, rulesVersion: `v${pad(cycle - 1)}`, score, majors })
    log(`cycle ${C} ${sec.key}: score ${score.toFixed(1)}, ${majors} major (rules v${pad(cycle - 1)})`)

    await agent(`${CONTEXT}\n${RUBRIC}
TASK (rules editor, cycle ${C}): improve the RULES, never the draft.
Evidence: ${dir}/critic-*.md, ${dir}/writer-notes.md, ${dir}/draft.tex. Earlier evidence: ${SB}/rounds/ (all cycles) and ${RULES}/CHANGELOG.md. Score history so far (rules version used, section, mean critic score): ${JSON.stringify(history)}.
Do:
1. Group the issues by rule_cause. A rule edit is justified only if the failure would recur in other sections (II, III, IV) or comes from a rule that is missing, unclear, contradictory or wrong. Issues that are one-off content errors of this draft need no rule change.
2. Edit ${RULES}/README.md and/or ${RULES}/skill/SKILL.md (and skill/reference/ if a source or trap is wrong) with the smallest general change that would have prevented the failure. Prefer sharpening or replacing a rule over adding one; delete rules that caused wrong behaviour. Keep the files from growing without need. Keep everything SESSION-CONTEXT.md requires and never reintroduce what it lists as rejected. Do not weaken the concise-writing rules of the skill. Removing or weakening a user-authored rule needs a justification in the changelog.
3. If the score history shows that an earlier rule change made the same section score lower on its next visit, revert or repair that change (see CHANGELOG.md).
4. Correct source-map cells you found wrong; mark a paper cell verified (with date) only if citation-log.md holds the verified passage.
5. Append to ${RULES}/CHANGELOG.md: "## v${pad(cycle)} (cycle ${C}, section ${sec.key}, score ${score.toFixed(1)})" and per change: rule, old to new in one line, evidence (critic file and issue), expected effect.
6. Snapshot: copy ${RULES}/README.md and ${RULES}/skill/ to ${RULES}/versions/v${pad(cycle)}/.
Return one line.`, { label: `editor:${C}-${sec.key}`, phase: 'Cycles' })
  }
  const mean = rotScores.reduce((a, b) => a + b, 0) / rotScores.length
  log(`rotation ${rot}: mean score ${mean.toFixed(2)}${prevMean === null ? '' : ` (previous ${prevMean.toFixed(2)})`}`)
  if (prevMean !== null && mean <= prevMean + 0.2) { noGain += 1 } else { noGain = 0 }
  prevMean = mean
  if (noGain >= 2) { log('two rotations without improvement (continuing: Dirk asked for 10 rotations)') }
}

phase('Report')
await agent(`${CONTEXT}
TASK: write ${SB}/REPORT.md for Dirk, short and concrete (no em-dashes):
1. Score history per cycle and per rotation: ${JSON.stringify(history)}. Which rules version is best (highest rotation mean; name the version folder under ${RULES}/versions/) and whether the scores still improved in the last rotations.
2. What changed in the rules: a readable summary of ${RULES}/CHANGELOG.md grouped by theme (math standard, ownership, source map, skill workflow, review checks), and the net diff of the best version against ${RULES}/versions/v00 (the starting rules), in a few lines per file.
3. Rule changes that were tried and reverted, and why.
4. Remaining recurring problems that no rule change fixed.
5. Best draft per section (cycle folder with the highest score for that section) and its open \\todo items, as a by-product.
6. How to adopt: review the diff of the best version's README.md against ${ORIG}/README.md and its skill/ against ${ROOT}/.claude/skills/thesis-section/, then copy (Dirk decides; nothing is applied automatically).
Read the files you need. Return the report text.`, { label: 'report', phase: 'Report' })
return { cycles: cycle, history }
