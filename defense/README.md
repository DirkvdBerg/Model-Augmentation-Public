# Thesis defense preparation

Practice for the defense with Claude: committee-style questions, answered first, then feedback with
checked facts. Run by the `defense-prep` skill (`.claude/skills/defense-prep/SKILL.md`).
Decisions D-288 and D-289. **Design frozen 2026-10-29: only fixes from here.**

## 0. Committee (from `Thesis-writeup/Writing/frontmatter/cover.tex`)
Defense date: 17-11-2026.

| Member | Role | What to expect |
|-|-|-|
| dr. ir. M. Schoukens | Thesis supervisor (TU/e, Control Systems) | Identification practice, validation, model augmentation; results argued against a criterion stated beforehand |
| prof. dr. ir. R. Toth | Second supervisor (TU/e, Control Systems) | LPV, LFR, well-posedness, identifiability, formal properties behind claims |
| ir. J.H. Hoekstra | Committee (TU/e) | His augmentation framework, SUBNET encoder, the extension for the additional states |
| ir. J.W. Gerritsen | Company supervisor (ASMPT) | Relevance for the machine, practical use, real data; asks exact implementation numbers (sampling rates, settings); favours ZOH over Tustin (own revnote, `03_augmentation.tex:885`) |
| dr. ir. M.F. Shakib | Committee, from Mechanical Engineering | Nonlinear and Lur'e-type identification, model updating; link to the thesis through Kessels |

Company supervisor ir. Q. van den Elsen is not on the committee list.

**Shakib and Kessels.** `refs.bib` cites no paper co-authored by Shakib. He is not a promotor of
`kessels2025ai` (van de Wouw, Fey; PDF p. 7) but co-authored the state extension and augmentation
papers behind its Chapter 5 (PDF p. 268). Every Kessels citation was checked on 2026-10-29 (PDF pages):

| Thesis claim | Kessels | Verdict |
|-|-|-|
| Accurate models help meet the requirements (intro) | p. 31, p. 171 (Sec. 5.1) | supported |
| Preserving interpretability in model updating is open (intro) | p. 210 | supported |
| Closed-loop data, physical parameters fixed (intro) | p. 183, Eq. (5.12) minimises over network weights only | supported |
| Closed-loop rollout per window, Eqs. (5.12), (5.13) | pp. 183 to 184 | supported |
| Output first, no loop through the controller, Remark 5.1 | p. 179 ("non-causal relation") | supported |
| Kessels "needs" the controller state, Remark 5.4 | p. 184, p. 198 | wording too strong: he initialises it from measurement or reconstruction; say that |

Defense point: Kessels p. 210 names as future work tuning the physical parameters inside the
optimisation with regularisation, and enforcing orthogonality to prevent negation. The thesis does both.

## 1. What a defense here looks like
Two peers who defended at TU/e (2026-10-29):
- Mostly a **discussion** of the work and how it could be better; almost no defending of claims.
- **Out-of-the-box** questions: methods not in the thesis, how you would evaluate differently, how
  you would continue (architecture).
- **"Did you consider"** questions on choices that look obvious to a reader (one peer tracked five
  sensors and used one).
- **"How does your model work"** when the report is unclear.
- Advice: prepare **backup slides** for the things you think are obvious.

Two reports, possibly from other committees: treat as the best evidence available, not as certain.
Ask both peers which committee members they had.

## 2. Question mix (a setting, in the skill)
Discussion 40 %, did you consider 20 %, explain the model 20 %, claim chains 20 % (HEURISTIC, from
section 1). Discussion includes hindsight: choices I would not make again (my own `\revnote`s are input).
Fixed opener "why this topic", contribution early and late, closer "anything else to add".

**Facts, outside the mix.** Every `practice` opens with a rapid-fire warm-up of 5 to 10 facts I must
know cold, from `fact-sheet.md`: rates and timing (training, validation and test sampling rate, controller
rate, integration step), data (records per split, record length, Y range, multisine records, how the data
were generated), training (window length, encoder past length, network sizes, optimiser and polish,
iterations, seeds, training time, hardware), model and loop (added states, bandwidth, absorber
frequencies, what is identified and what is fixed), evaluation (metrics, normalisation, noise floor
normalised only, black-box size and budget). Each value is checked in the thesis and the final code; a
mismatch or a doubting `\revnote` becomes an open item first.

**Capture.** A question, position or hindsight point that comes up in any conversation is saved at once
as a question with a checked draft answer.

## 3. Where questions come from
- **Committee read** (`committee-read.md`): one fresh-eyes read of the thesis by a subagent without this
  conversation's context: obvious choices, out-of-the-box topics, how to continue, hard-to-follow
  parts, backup-slide candidates. I strike what nobody would ask.
- **Open items** (`open-items.md`): what I could not answer.
- **Thesis claims**, for the claim chains.

## 4. Facts are always checked
- **Three sources of truth for my own work:** the live thesis text, the final code in `Clean/`, and the
  final run results in `Thesis-writeup/Results/24-10-2026/CSV/` (checked per row and cell as
  `csv:<path>`; a rounded number says so). Open items and fact-sheet values are verified against all
  three; a disagreement stays open and is reported, never resolved by picking one.
- Questions may be outside the thesis (`grounding: outside`).
- Every factual statement in feedback or an answer rests on an evidence record that is re-checked in
  the session where it is used (`defense_db.py recheck` / `add-evidence`): verbatim excerpt on its page
  of the source text (`defense/sources/`, extracted from the PDFs) or the thesis `.tex` (comments
  ignored, struck text and draft bullets never cited). Equations are checked on the PDF page itself.
- What another method does is stated only from its paper, opened locally.
- Opinions are marked as mine, separate from what the thesis or a paper shows.
- An excerpt match shows the text is there, not that it supports the claim; that judgement (direct or
  derived) is recorded and is mine to check.

## 5. Steps
| Step | What happens |
|-|-|
| `/defense-prep` | Status: due today, questions per type, one recommended step |
| `read` | Fresh-eyes committee read (once), including effort vs a simpler alternative and choices I might not make again |
| `facts` | Build the fact sheet, each value checked in thesis and code (once) |
| `practice [n]` | n questions (default 5) answered at once, then a live challenge chain on the weakest one or two, then feedback with checked facts; I give each result |
| `study [ids]` | Question with its checked answer and first challenge, for learning; comes back cold 2 days later |
| `open` | Resolve open items from sources, or phrase an honest limitation |
| `mock` | Paste-in pack for voice mode in the Claude app (presentation, then about 12 questions, no feedback until the end); transcript reviewed afterwards |

Schedule (HEURISTIC): wrong again next session, shaky after 2 days, solid after 5, solid twice retired.
One step per session keeps the context small; all state is in files.

## 6. Answer structure (`answers/Q-<nnn>.md`)
Claim or position; mechanism (said out loud); evidence (E-ids); what else was possible; my view and
next step (marked as opinion); a 30-second version and an "if pushed" version.

## 7. Files
```
defense/
  README.md, overview.md     this method; generated overview (open in the file pane)
  committee-read.md          fresh-eyes read with checkboxes
  fact-sheet.md              facts to know cold, with sources (written by `facts`)
  open-items.md              what I cannot answer yet, and its resolution
  mock-pack.md               paste-in pack for voice mocks (written by `mock`)
  sources/<citekey>.txt      page-tagged text of each cited PDF (generated)
  data/evidence.jsonl        checked excerpts (source register)
  data/questions.jsonl       questions with type, voice, challenges, status
  data/drills.jsonl          append-only results and schedule
  answers/Q-<nnn>.md         model answers
  tools/defense_db.py        sources, check, recheck, rows, annots, add-evidence, add, drop,
                             log-drill, due, validate, overview
```
Run: `conda run -n GraduationProject python defense/tools/defense_db.py <command>`.

## 8. Order (19 days)
1. `read` and `facts` once (resolve O-02 first). 2. `practice` daily, `study` for new topics, `open` when items pile up. 3. A first
`mock` within about 5 days, at least 4 in total. 4. Backup slides from the committee read.

## 9. Known weaknesses
- **Two peer reports** set the mix; they may not match this committee.
- **The committee read is an agent, not an examiner**; a friend skimming the thesis is better still.
- **Model answers can be wrong**; the answer I give must be mine.
- **Text extraction garbles equations**; math is checked on the page image.
- **A skill enforces, it does not make an expert.**
- **Unverified:** the TU/e study guide (committee composition, 20 to 25 min presentation, 50 to 60 min
  defense). Check the current regulation.

## 10. Evidence behind the design
| Choice | Source | Checked |
|-|-|-|
| Discussion-heavy mix, "did you consider", explain the model, backup slides | Two peer reports, 2026-10-29 | first-hand |
| Fresh-eyes read, ungrounded questions allowed | D-289 critique of D-288 | n/a |
| Generate from one passage, short prompts | Savaal arXiv 2502.12477 Sec. 1; Scaria et al. AIED 2024 Table 4 | yes |
| Answer before feedback; challenge chains | Roediger and Karpicke 2006; three critiques in `tasks/handoff.md` | critiques yes, paper no |
| Early and repeated mocks | Gunn et al. 2026 (clinical orals) | no |
| Kessels citations | `kessels2025ai.pdf`, pages in section 0 | yes |
