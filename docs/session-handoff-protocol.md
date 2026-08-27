# Session Handoff Protocol

How to hand the current session's work to a fresh session, **when the user asks for it and only
then**. The output is not a diary and not an archive: it is **a prompt for Claude Opus 5**, and it
is written to be read cold by a session that knows nothing except `CLAUDE.md`.

Decision record: D-133. Prompting rules below follow Anthropic's
`platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5`
(fetched 2026-07-30). If that guidance changes, this file changes with it.

## When to write one: only when the user asks

The **only** trigger is an explicit request from the user. "Write a handoff", "write a prompt for
a new session", "hand this off", "I am starting a fresh session on this". Nothing else.

Never decide on your own to hand off, to end the session, or to start a new one. Context
pressure, repeated tool errors, and looping on one fix are **not** triggers: they are things to
report in one sentence so the user can decide. Report and continue working. Do not write the
file, do not prepare it in advance, and do not suggest it twice.

The **Respect session boundaries** standing rule ("work happens in a new or other session") means
stop doing that work here. It does not by itself authorise writing this file. If the user says
work moves elsewhere and it is unclear whether they want a handoff document, ask in one line.

## Where it goes

`tasks/handoffs/YYYY-MM-DD-<slug>.md`, one file per handoff, slug naming the problem
(`2026-07-30-t4-kxy-dataset.md`). Do NOT write to `tasks/handoff.md`: that file means open
blockers plus cross-agent proposals, and reading it triggers its archive-on-read rule.

These files are the one exception to the **No new files unless asked** standing rule. This
protocol is the standing authorisation.

## How to write it as an Opus 5 prompt

Nine rules. Each one exists because of a documented behaviour of the model that will read it.

1. **Complete specification up front.** Opus 5 performs best when given the whole task at once
   and left to run. The successor must be able to start work without asking a single clarifying
   question and without reconstructing state by exploration. Everything it needs to decide the
   first action is in the file or behind an explicit pointer in it.
2. **State the anti-scope explicitly.** The model expands scope and applies its own judgment
   about what the task should be. An **Out of scope** section is not optional; it is the main
   defence. Name the adjacent work it must not start.
3. **No verification instructions.** Do not write "double-check", "verify before responding",
   or "use a subagent to verify". The model already verifies its own work, and these compound
   into wasted tokens with no quality gain. State the **acceptance criterion** instead: what
   number, from what data, means done. A criterion is a target; a verification instruction is a
   ritual. Only the first belongs in a prompt.
4. **Cap delegation.** The model delegates to subagents readily, and delegation multiplies cost
   on small tasks. Say whether the next task warrants an Explore subagent at all, and if so how
   many (the project default is one).
5. **Never hedge the reporting instruction.** "Only flag major issues", "be conservative", "keep
   it short if you can" get obeyed literally and suppress real findings. If you want filtering,
   ask for everything and filter in a separate pass.
6. **Positive examples beat prohibitions.** For anything about style, format, or communication,
   show one short example of the wanted output rather than listing what to avoid. Keep the
   prohibitions for actions (paths not to touch, approaches already dead).
7. **Recommend an effort level.** Say which the next task wants and why: `low`/`medium` for
   mechanical or well-specified work, `high` (default) for normal engineering, `xhigh` for
   demanding derivations or wide agentic work. Keep thinking enabled either way.
8. **Do not restate `CLAUDE.md`.** It is auto-loaded in the successor session. Repeating the
   standing rules, the environment, or the file map wastes the context the handoff exists to
   save. Reference a rule by name (for example "per the run-discipline rule") when it matters.
9. **It is a prompt, not an archive.** Roughly 400 lines maximum. Detail lives behind
   `file:line`, a `D-` number, or a run ID. Paste code only when the exact text is the subject
   of the next action, and then only the lines that are, verified per the code-quote rule.

## Template

Fill every section. If a section is genuinely empty, write "None" rather than deleting it: an
absent section reads as forgotten, and the successor cannot tell the difference.

```markdown
# Handoff: <one-line problem statement>
**From**: session of YYYY-MM-DD | **Branch**: <branch> | **Effort suggested**: <low|medium|high|xhigh>

## 1. Task
One paragraph, imperative, complete. What the successor is to accomplish, not what happened.
State it so that this paragraph alone would be a usable instruction.

## 2. Out of scope
The adjacent work it must not start, and the reason each is excluded (already done, blocked,
user's decision, different session). Name files it must not modify.

## 3. Where things stand
Branch, last commit, whether the tree is dirty and in which directories, and any run currently
in flight (ID, launch command, output path, expected finish).

## 4. Established and verified
Facts the successor may build on without re-deriving, each with its evidence: a `file:line`, a
`D-` number, a run ID, or a measured number with units. Verified means measured or read, not
inferred.

## 5. Assumed but not verified
Everything load-bearing that has NOT been measured, and what would settle each one. Keeping this
separate from section 4 is the point: an assumption promoted to a fact is how a wrong result
survives a handoff.

## 6. Tried and failed
Each dead end, with the mechanism of failure, not just the symptom. Format:
- <what was tried> -> <what happened, with numbers> -> <why, mechanically> -> <evidence pointer>
This is the highest-value section in the file. Its job is to stop the successor spending a
multi-hour run rediscovering a known failure.

## 7. Achieved
What is working and proven, with the artefact paths and the numbers that prove it. Distinguish
"implemented" from "implemented and validated".

## 8. The open question
The single decision or unknown that currently blocks progress, stated as a question with the
candidate answers and what evidence would choose between them. If nothing is blocked, say
"Nothing blocked" and go to 9.

## 9. Next action
Exactly one concrete action, with its rationale. Not a menu. Include the command to run if there
is one, and where its output will land.

## 10. Acceptance criterion
The number that means done, where it comes from, and the threshold. Data-derived and defensible
(noise floor), never oracle-based or model-based, per the Control Engineering Stance.

## 11. Read these first
Ranked, at most five, each with one line on why. Anything beyond five is a pointer inside another
section, not a reading assignment.

## 12. Do not
Dead ends not to retry (cross-reference section 6), paths not to touch, and any approach the user
has already declined. Keep this to actions.

## 13. Operational
Launch command verbatim, expected runtime, where output streams to, which conda env, any cluster
or data dependency, and any artefact the next step consumes.

## 14. Delegation
Whether an Explore subagent is warranted for the next action, and the ceiling. Default: none for
targeted work, one for a genuinely wide search.
```

## Anti-patterns

Each of these has produced a bad handoff:
- **Narrative instead of specification.** "I first tried X, then noticed Y, then wondered
  about Z." The successor needs the current state and the next action, not the path taken. The
  path belongs in section 6 only where it prevents a repeat.
- **Symptom without mechanism.** "The run diverged" is not reusable. "Val sim-RMS degraded 127x
  on epoch 1 because the 400-sample training horizon discounts the K=0 position drift by 120x"
  is.
- **Blurring verified and assumed.** See sections 4 and 5. This is the failure that costs the
  most, because it is invisible downstream.
- **Restating `CLAUDE.md`.** Auto-loaded. Do not spend context on it.
- **A menu instead of a decision.** Section 9 is one action. Alternatives, if any, belong in
  section 8 as candidate answers with the evidence that would choose between them.
- **Pasted output dumps.** Point at the `.output` file and quote the two signal lines that
  matter.
- **Silent scope drift in the writing.** The handoff describes the task the user set, not the
  more interesting task discovered along the way. If a better task exists, say so in one
  sentence in section 8 and leave the decision to the user.

## Done when

The handoff is complete when a session holding only `CLAUDE.md` and this handoff could execute
section 9 without asking a clarifying question. That is the criterion; there is no separate
review pass to run afterwards.

After writing it, tell the user the path and give the one-line summary of section 9 so they can
open the next session with it.
