# Handoff: answer the cold-read questions in the comments of `Clean/` (D-284 amendment 11, step 2)
**From**: session of 2026-10-31 | **Branch**: Augmentation | **Effort suggested**: high (judging and writing
comments over about 5,000 lines; no derivations, no runs)

## 1. Task
Dirk: "I still dont think the comments are up to standard, i dont wnat unnecessary number of comments but also for
some settings or variables/parameters to me its not clear or for some steps". Two independent cold readers (Codex and
Claude, no thesis, no history) wrote `tasks/clean-code/cold-read-codex.md` and `cold-read-claude.md`: list A,
questions they could not answer from the code (`Qnnn`); list B, comment lines they could delete without loss (`Nnnn`).
First merge them into `tasks/clean-code/cold-read-merged.md`: items found by both are agreed; items found by one
reader get your one-line recommendation for Dirk. Stop for Dirk's ruling on the single-reader items. Then answer
every agreed or accepted question and delete every agreed or accepted noise line in the 24 grey-box files of
`Clean/`, by editing comments and docstrings only, with the pattern of section 4, and write
`tasks/clean-code/cold-read-answers.md`: one line per item, `Qnnn | answered at file:line`, `Qnnn | rejected:
<reason>`, `Qnnn | rename proposed: old -> new, <reason>`, `Nnnn | deleted` or `Nnnn | kept: <reason>`. At the top:
counts per verdict and the comment line count per file before and after.

## 2. Out of scope
- Any code change: no renames, no new or removed code lines, no changed strings, error messages or keys. A question
  whose best answer is a rename goes in the answers file as `rename proposed`; Dirk rules on those later.
- Comments neither list names. Exception: a comment the answer makes redundant may go (list it as `answered`).
- A `# THEORY:` or `# HEURISTIC:` label is never deleted, only shortened into the pattern's source tag (CLAUDE.md,
  signal processing rule).
- Noise items that hit a standing ruling are `kept: <ruling>` in the merge, applied by you without asking Dirk:
  `# CHANGED:` markers, `@added`, `__project_origin__` and the header naming Jan's commit (CLAUDE.md "Tracking
  Additions", D-282, BSD attribution); Jan's own comments and docstrings (Jan's code 1:1, amendment 10 point 1; check
  against `Model-Augmentation-Public-main/`); a formula comment above its step (amendment 4), unless it repeats the
  code token for token. A `history` item that is not a marker is real noise. The readers did not know these rulings
  (Codex flagged 29 CHANGED lines, about 20 attribution or Jan lines and 88 formula lines), so filter them first.
- `Clean/blackbox/` (D-285), `Clean/data_generation/` (D-286): later, same method.
- Any run (training, harness, suite, smoke). The proof is static (section 10).
- The step-5 server job itself and its harness files, except the hash updates of section 9 step 5.

## 3. Where things stand
Tree dirty, Dirk commits. `Clean/` equals the `final` snapshot (`simulations/clean-golden/src-snapshots/final/`,
checked 2026-10-31: `sha256sum -c` from `Clean/` prints nothing, manifest file hash `0a4a8636...5bff`). The step-5
server job (`tasks/clean-code/implementation-report.md`, "Server commands (D-284 step 5)") checks the tree against
that manifest and reverts the MM16 line by an exact `sed` on `Clean/gantry_dynamic/data.py:218`. It is not submitted:
Dirk ruled the comments go in **before** it (2026-10-31, "before"), so the server job proves the commented tree.

## 4. Established and verified
- The diagnosis, from `Clean/gantry_dynamic/config.py`: comments state provenance ("HEURISTIC: Adam budget of the
  thesis runs, in epochs") instead of meaning and effect; `n_its`, `truncate_s`, `start_phase` and `up_sample` have no
  comment. D-284 amendment 11.
- The pattern (amendment 11 point 2), on top of amendments 4 and 7 (formula comments, docstrings with `Args:` and
  `Returns:`, numbered phase headers, no history, D-numbers, "we" or filler):
  - a setting or constant: one line, *what it is [unit]: what changing it does* (or what `None` means), the
    THEORY/HEURISTIC label kept as a short source tag after it. Example of the shape (the content must come from the
    code, not from this line):
    ```python
    epochs: int = 200               # Adam passes over the training windows; HEURISTIC: thesis budget
    ```
  - a parameter: an `Args:` line only where the name does not say it;
  - a step: a numbered phase header in thesis terms, `# 2. Encoder start: least squares on the past windows.`
- Labels stay: every number keeps its `# THEORY:` or `# HEURISTIC:` tag (CLAUDE.md, signal processing rule).

## 5. Assumed but not verified
- That every question is answerable from the code and the thesis. Where it is not (the reason for a value is unknown),
  write what the setting does and leave the HEURISTIC tag; do not invent a reason.
- That the thesis section names in `Clean/README.md` match `Thesis-writeup/Writing/sections/`. Use the README names.

## 6. Tried and failed
- Four earlier comment passes (`tasks/clean-code/comment-check*-codex.md`) reviewed correctness and redundancy of
  existing comments against the thesis -> Dirk still found settings and steps unclear -> they judged the comments
  that exist, never asked what a reader misses -> amendment 11.

## 7. Achieved
- Decision logged: D-284 amendment 11. Cold-read prompt (both readers): `tasks/handoffs/2026-10-31-comment-cold-read.md`.

## 8. The open question
Nothing blocked. Decided by Dirk: comments before the step-5 server job.

## 9. Next action
1. Copy the current `final` snapshot folder to `simulations/clean-golden/src-snapshots/final-step5a/` (the baseline
   of the AST check and the archive of `0a4a8636...5bff`, as `final-step4` was archived before).
2. Merge the two reports into `cold-read-merged.md` (section 1) and stop for Dirk's ruling on the single-reader
   items.
3. Go through the merged list file by file: answer each question, delete each noise line, fill the answers file, run
   the AST check (section 10).
4. Dirk runs the cold-read prompt again with both readers (`cold-read2-codex.md`, `cold-read2-claude.md`); answer
   anything on a line marked answered, and bring new items to Dirk.
5. Remake `final` from the commented tree by the method that made it before (`MANIFEST.sha256` of the 24 files,
   relative to `Clean/`, and `snapshot.json` with label, file count, manifest hash and a note naming amendment 11),
   and replace the old hashes everywhere they are pinned: `timing_driver.py:107` (`final`), the "Server commands
   (D-284 step 5)" section of `tasks/clean-code/implementation-report.md` (manifest hash, the data.py SHA256 in the
   final and in the reverted state; compute the reverted one by applying the `sed` to a copy, never to `Clean/`), and
   the D-284 STEP 5 row in `docs/gantry-augmentation-problem-log.md`. The MM16 line text at `data.py:218` stays byte
   for byte, so the `sed` still matches; if a comment edit shifts its line number, update the line number in those
   texts.

## 10. Acceptance criterion
- Every `Qnnn` has a verdict line.
- Per file, the syntax tree with comments and docstrings removed equals that of `final-step5a`: write
  `clean-up-verification/comment_ast_check.py` (parse both, drop every docstring expression, compare `ast.dump`),
  which prints one line per file and `AST_IDENTICAL 24/24`. Exact; no tolerance.
- The second cold read (both readers) raises no question or noise item on a line marked answered or deleted. New questions on untouched lines go to Dirk, not into this
  task.
- Comment lines grow only at answered questions (the counts at the top of the answers file).
- After step 5: `(cd Clean && sha256sum -c --quiet ../simulations/clean-golden/src-snapshots/final/MANIFEST.sha256)`
  prints nothing, the `sed` applied to a copy of `data.py` gives the reverted hash written in the report, and no file
  outside `archive/` still names `0a4a8636` except `final-step5a/snapshot.json`.

## 11. Read these first
1. `tasks/clean-code/cold-read-codex.md` and `cold-read-claude.md`: the questions and noise lines.
2. `docs/decisions.md`, D-284 amendments 4, 7 and 11: the comment standard.
3. `Clean/README.md`: thesis section names per file.
4. `Clean/gantry_dynamic/config.py`: most settings questions land here.

## 12. Do not
- Edit code, strings, keys or names (section 2).
- Answer a question with history ("of the thesis runs", "mainline", "was"), a D-number or a harness name.
- Add a comment where no question was asked, or add thresholds, checks or options.
- Touch `Thesis-writeup/Code/Verification/` or the snapshots beyond section 9 steps 1 and 5.

## 13. Operational
No runs. The AST check is quick: `conda run -n GraduationProject python clean-up-verification/comment_ast_check.py`.

## 14. Delegation
None. The questions are listed; answering them needs one consistent voice.
