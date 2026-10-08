# Handoff: how to decimate noisy 20 kHz closed-loop encoder data to 4 kHz for simulation-error training
**From**: session of 2026-09-28 | **Branch**: Augmentation | **Effort suggested**: high (a literature synthesis mapped onto one concrete setup; no derivation-heavy work)

## 1. Task
The user's words (2026-09-28): "I think we should create a prompt for another session to look online for this", where
"this" is the choice of anti-alias filter before the training loader decimates the noisy thesis data from 20 kHz to
4 kHz. Search the literature and the web for how fast-sampled, noisy, CLOSED-LOOP position data should be decimated
before (simulation-error / output-error) system identification, and answer the five questions of section 8 for OUR
setup (sections 4 and 6). The deliverable is a written research result with sources, ending in ONE recommended
decimation rule for the measured output y (and whether the input u must change with it), stated so that it can be
tested with the existing measurement script against the existing criteria (D-232). This is research on paper: no
code changes, no numerical runs, no data generation.

## 2. Out of scope
- Editing `scripts/gantry/gantry_dynamic/data.py` or any loader, generator or training file: the filter is chosen
  after this research, with the user (D-232: "implementation ... after the user's OK").
- Running `scripts/gantry/anti-alias/measure_aliasing.py` or any other script, training or MATLAB job (another session
  and the user decide what runs; the machine has about 2.8 GB of free RAM).
- Regenerating or modifying data in `Thesis-writeup/Data/` (the D-230 set is final and checked).
- Changing the D-232 criteria. If the literature says a criterion is wrong, say so and why, as a finding.
- Item 7b of `tasks/todo.md` (report rows on a model copy) and anything else in that checklist.

## 3. Where things stand
- Branch `Augmentation`, last commit `3e34605`, working tree dirty in many directories (other sessions' work too):
  touch nothing outside your own report file.
- Nothing is running for this topic.
- Data: `Thesis-writeup/Data/Coulomb-tanh-and-MSD` (noisy) and `Coulomb-tanh-and-MSD-noise-free` (the same 48 records
  without noise), 12 s at 20 kHz each (D-230, D-231; `check_records` PASS on all 48).

## 4. Established and verified
- The loader decimates u and y differently: u by the block mean of 5 samples (`scripts/gantry/gantry_dynamic/data.py:142`
  `_resample_u`, D-087: ZOH-consistent, the plant input is held over each 20 kHz step), y by point sampling `y[::5]`
  (`data.py:172` and `:192`). D-099 (`docs/decisions.md:4190`) justified point sampling because the TRUE position is
  band-limited far below 2 kHz; that premise no longer holds for the recorded y (next item).
- Since D-225 (`docs/decisions.md:7037` and its amendment) the recorded y is the MEASURED position y = q + v: encoder noise
  v injected inside the simulated closed loop (the controller sees q + v), one independent generator per axis, the
  measured Telica standstill spectrum (9.8 / 10.4 / 6.3 nm rms, full band to 10 kHz, over 90 % of the variance above
  200 Hz) with its shape mapped through the loop above 50 Hz. The noise-free twin has v = 0 and is otherwise bitwise
  identical. `y_true` (= q) and `v_enc` (= v) are also stored in every record.
- Training objective: a CLOSED-LOOP simulation-error rollout at 4 kHz in the residual form
  u_model = u_total + K1 (y_data - y_model) (`model_augmentation/fit_systems/closed_loop.py:245` `closed_loop_rollout`),
  K1 Tustin-discretised at 4 kHz. It equals f + K1 (r - y_model) exactly only when y_data is the signal K1 saw and K1 is
  the same operator; the data's K1 runs at 20 kHz, so this is already inexact at the rate change. u_total at 20 kHz
  contains K1's response to the full-band encoder noise.
- D-232 measurement (`docs/decisions.md:7297` and its result paragraph; `scripts/gantry/anti-alias/measure_aliasing.py`
  and `measure_aliasing.log`), six training records, per axis, noise part n = y_noisy - y_noisefree:
  - point sampling raises the 4 kHz noise to 1.28 to 1.40 x the true in-band (< 2 kHz) noise (X1 7.7 -> 10.2 nm,
    Y 4.8 -> 6.6 nm); block mean, a zero-phase FIR (101 taps, 1.6 kHz) and `scipy.signal.decimate` (order-8 Chebyshev,
    zero phase) all bring it to 0.91 to 0.99 x;
  - the closed-loop floor (the truth itself, simulated at 4 kHz in the residual form, error against y_data) with point
    sampling is 0.08 to 1.0 um on the multisine records, set by the 4 kHz discretisation, and 7 to 11 nm on the route
    record TR-T1: the aliased noise (about 3 nm extra rms) is far below 1 % of the floor on multisine records and a
    visible share on the route record.

## 5. Assumed but not verified
- That the aliased noise only adds variance and does not bias the estimated parameters or the learned block. In
  closed loop the input u_total is correlated with the output noise through K1, which is exactly the case where
  output-error estimates can be biased (a question for section 8, Q2).
- That filtering y - r instead of y removes the numerical passband problem of section 6 (not run; the reference r is
  jerk-limited and assumed to have negligible content above 2 kHz).
- That u should keep its block mean if y is filtered (filtering u and y with the SAME filter is the textbook prefilter
  rule; whether that holds with the residual form and the controller in the loop is Q1).

## 6. Tried and failed
- Block mean of 5 on y -> signal change 1 to 100 um during moves, closed-loop floor 12 to 44 x -> it takes the sample at
  the block centre, 2 samples (0.1 ms) later than point sampling, so a moving position is off by velocity x 0.1 ms
  -> D-232 result.
- Zero-phase FIR (101-tap Hamming, 1.6 kHz, `filtfilt`) on y -> 55 to 170 nm signal change below 1 kHz, 7 to 35 x the
  in-band noise, floor 1.3 to 2 x on multisine records -> the filter acts on the full position (0.1 to 0.3 m), where a
  1e-6 relative passband error is already 100 nm -> D-232 result.
- `scipy.signal.decimate` (Chebyshev I, order 8, zero phase) on y -> up to mm near the record end, floor up to 40 x -> end
  transients of the zero-phase IIR on the large position offset -> D-232 result.

## 7. Achieved
- The problem is measured and located (section 4, 6); the criteria are fixed (D-232: removes the folded excess to within
  10 % of the true in-band noise; changes the signal by less than 10 % of that noise; does not raise the closed-loop floor
  by more than 10 % against point sampling).
- `measure_aliasing.py` can test any new decimation rule by adding it to its `DECIM` dictionary.

## 8. The open question
What decimation rule for y (and u) is right for closed-loop simulation-error training on these data? Answer each,
with primary sources (book section or equation number, paper section), then map it onto our setup:
- Q1 Prefiltering in identification: must u and y pass the SAME anti-alias filter (Ljung-style prefilter
  L(q) on both), and is a ZOH-consistent block mean for u plus a different rule for y consistent? Does the answer change
  in closed loop and with the residual form (u_total already contains the controller's reaction to the noise)?
- Q2 Effect of aliased output noise on simulation-error / output-error estimates in CLOSED loop: variance only, or bias
  (noise correlated with the input through feedback; direct method versus indirect or joint input-output methods)?
  Is point sampling with the aliasing then acceptable when the loss floor is dominated by other errors (section 4)?
- Q3 How to low-pass a signal with a large offset and large motion at nm precision: decimate the tracking error (y - r)
  or a deviation from a smooth trend; multirate / polyphase / CIC decimators; zero-phase versus causal; edge handling.
  Is "filter the error, not the position" established practice (precision motion control, servo-error logging)?
- Q4 Does the frequency-domain view (periodic multisine records, identification on excited lines: Pintelon and
  Schoukens) make the question moot for the multisine records, and what does it say for the non-periodic routes?
- Q5 Any guidance for neural state-space / SUBNET-type simulation-error training on noisy fast-sampled data
  (Beintema, Toth, Schoukens; Hoekstra et al.; Forgione and Piga) about decimation and anti-aliasing?

## 9. Next action
Invoke the `deep-research` skill (mandatory for literature work, D-121) with step 0 (FRAME) on the five questions of
section 8, then search. Write the result to `scripts/gantry/anti-alias/LITERATURE.md` (the one file this task may create),
with the skill's Research Log at the end.

## 10. Acceptance criterion
`LITERATURE.md` answers Q1 to Q5 each with at least one primary source read in full text (section or equation cited)
and states for each whether it supports, contradicts or does not address our setup; it ends with ONE recommended
decimation rule for y (and the matching rule for u), with the expected effect on each D-232 criterion and the exact
entry to add to `DECIM` in `measure_aliasing.py` so the next session can test it. Example of a good recommendation line:

> Recommended: y_dec = r[::5] + FIR_1.6kHz(y - r)[::5], u unchanged (block mean); expected (i) about 0.95 x Q1,
> (ii) below 1 nm because the filter now acts on um-scale signals, (iii) within 5 % of point sampling. Source: <book,
> section>. Test: a `DECIM` entry `'fir_err'` computing `r[::D] + filtfilt(FIR, [1.0], y - r, axis=0)[::D]`; the
> script's entries take y only today, so the recommendation also says to pass the record's `r_sim` to them.

## 11. Read these first
1. `docs/decisions.md` D-232 (line 7297) and its result paragraph right below it: the criteria and every number above.
2. `scripts/gantry/anti-alias/measure_aliasing.py`: what each quantity means, and where a new rule plugs in.
3. `docs/decisions.md` D-099 (line 4190) and D-087 (line 4520): why u is block-meaned and y point-sampled today.
4. `Thesis-writeup/Documentation/NOISE-INJECTION.md`: what the noise is and how it enters (section 3 to 5, 8).
5. `.claude/skills/deep-research/SKILL.md`: the mandatory search procedure.

## 12. Do not
- Do not retry block mean, full-position FIR or full-position Chebyshev decimation as recommendations without a reason
  that addresses their failure mechanism (section 6).
- Do not edit any code or data, and do not run any script (section 2).
- Do not ask the user whether to start: the task and deliverable are fixed here.

## 13. Operational
No runs. Web search and paper download through the deep-research skill's tools; downloaded PDFs go where the skill says
(`literature/` with an entry in `docs/references.md`). Output: `scripts/gantry/anti-alias/LITERATURE.md`.

## 14. Delegation
The deep-research skill runs in subagents; use at most 2 (standing limit on this account), each with the specific
sub-questions pasted in, not the whole document tree.
