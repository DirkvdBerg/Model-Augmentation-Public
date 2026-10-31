# Prompt for the cold read of the comments in `Clean/` (D-284 amendment 11, step 1)

Two independent readers use the section below: Dirk pastes it into Codex (`<reader>` = `codex`) and into a fresh
Claude session (`<reader>` = `claude`). Neither reads the other's report. Step 2 (merge and answers) is
`2026-10-31-comment-answers.md`, run after both reports exist. The same prompt, with "2" after `cold-read` in the
report name, is the step-3 check.

---

You are an engineer at ASMPT who receives the folder `Clean/` of this repository: the code of a grey-box model of a
dual-gantry motion stage (physical baseline plus a learned augmentation, trained in closed loop). You know control
engineering, system identification and PyTorch well. You have not read the thesis and know nothing of the
project's history. Your only sources are the files in `Clean/` (code, comments, docstrings, `Clean/README.md`).
Another reader does the same task independently; do not open any other `cold-read*` file.

**Task.** Read the 24 grey-box files: `Clean/gantry_interconnect_dynamic.py`, every `.py` file in
`Clean/gantry_dynamic/` and in `Clean/model_augmentation/` (with subfolders). Make two lists.

**List A, questions.** Every place where you cannot tell from the code and its comments alone:
- what a setting, constant, variable, field, key or parameter **is** (meaning, unit, allowed values, what `None`
  or a special value means);
- what **changing** a setting does (what gets longer, slower, more accurate, switched off);
- what a **step** in a function is for, or why the steps come in this order;
- what a function **returns** or expects, where the name and shapes do not say it.

Write questions only, not fixes or proposed comments. A question counts when you would have to read other code, the
thesis or the git history to answer it. Ask everything you would ask; do not filter by importance and do not drop
a question because you can guess the answer: a guess you are not sure of is a question. Do not ask about names or
comments you do understand, about style, or about whether the code is correct.

**List B, noise.** Every comment or docstring line you could delete without losing anything you need, with the
reason: `repeats-code` (says what the line below plainly says), `repeats-doc` (says what a nearby docstring or
comment already says), `history` (how it used to be, where it came from, earlier runs), `empty-tag` (a source tag
that tells you nothing about what the value does), `filler`. List every such line; keep labels `# THEORY:` and
`# HEURISTIC:` out of `empty-tag` when the source they name is informative.

**Output.** Write `tasks/clean-code/cold-read-<reader>.md` (step 3: `cold-read2-<reader>.md`), and nothing else. At
the top: the count of questions and of noise lines per file. Then list A, then list B, one line per item, in file
order:

```
Q012 | Clean/gantry_dynamic/config.py:74 | setting | n_its: what is it, how does it relate to epochs, and what does None mean?
Q013 | Clean/gantry_dynamic/config.py:76 | setting | its_per_val = 1300: iterations of what? What changes if I halve it?
N004 | Clean/gantry_dynamic/config.py:98 | empty-tag | "of the thesis runs" says where the values came from, not what they do
```

Kinds in list A: `setting`, `variable`, `parameter`, `return`, `step`. Do not edit any file, do not run training or
tests, do not read `kamtin-data/`, do not run git write commands. Reading the code with grep or Python's `ast` is
fine.
