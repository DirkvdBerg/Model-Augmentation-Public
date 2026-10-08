# Training design (settings for every thesis result)

Companion to `DATA-DESIGN.md` and `RESULTS-DESIGN.md`. Evidence: D-220; `scripts/gantry/joint-estimation-horizon/REPORT.md` (window memory); `scripts/gantry/joint-estimation-horizon/LITERATURE.md` section C (literature). Open points are marked (?).

**No regularisation penalty in any arm (user 2026-09-27).** The only constraint on the augmentation is orthogonality by construction (OBC). Not used: the parameter penalty (Hoekstra Eq. 26, `param_rmse_baseline`, the Aug-Lambda arms) and the orthogonal-projection penalty (Gyorok 2025, `orth=False`).

**Added states for the main results (R4, R6): 2 (user 2026-09-27),** one absorber mode; earlier runs had 2 within 3.1 % of 8; R2 confirms it.

**Training data: the noisy records (user 2026-09-27).** The noise-free twins give the floors and serve R5.

## 1. Training window (joint-estimation horizon)
- One window for every arm, every added-state count and both starts (correct and detuned)
- Why: in the residual closed-loop rollout a force error reaches the servo error through `(I + G K1)^-1 G`; that loop forgets in about 50 ms at all five Y (D-220)
- Slowest closed-loop pole 11 ms; the loop damps the absorber mode to 4.7 ms (35 ms open loop)
- Model matrices (A) set the bound; the measured FRF (B) confirms it within 1 ms where it resolves; friction adds under 1 %
- 0.1 s is 9 times the slowest pole; the field rule of 1.5 to 4 times gives 17 to 44 ms (Gyorok 2025; Beintema 2021)
- A longer window only shrinks the blind start of each window: 5 % (absorber-type error) and 11 % (damping-type error) of the error energy at 0.1 s, halving per doubling
- Detuning changes the kind of error (damping-type), not the loop memory; the model's loop stays close to the truth's
- Moot since D-224: the nonlinear stick dwell of the Karnopp law, which A and B did not cover (the FRF records spent 2 to 10 % stuck); the truth's tanh law has no stick state
- Current value: nf = 400 at 4 kHz (0.1 s); code comments updated to D-220 (config.py, entry script; vendored copies left as snapshots)

### 1.1 Confirmation sweep (planned, user 2026-09-27)
- Three full production runs (Adam, then the L-BFGS polish); only the window changes: 0.1, 0.2, 0.4 s
- Detuned start (largest blind share, parameters matter), 8 added states (largest model, so the result covers fewer states)
- Arm: Aug-OBC (user 2026-09-27)
- One seed and one detuning direction shared by all three runs
- Run 1 (0.1 s) is also the pilot: maximum budget inside the 24 h wall (about 200 epochs); its plateau sets the budget
- Runs 2 and 3: the same number of Adam updates as run 1's plateau times 1.5 (HEURISTIC margin), then the same polish
- Converged: best validation free-run error improves by less than the S6 detection threshold over the last quarter of the run; same test on the ten-parameter error norm
- Compared on: validation servo-error RMS per axis against the floor, and all ten parameter errors
- Prediction, stated before the runs: a longer window gains at most the blind share (5 to 11 %); the parameters do not improve beyond it
- Decision rule: keep the shortest window that is not clearly worse; if 0.2 and 0.4 s do not beat 0.1 s by more than the blind share, keep 0.1 s
- If the parameters do not plateau within the budget: the sweep compares their movement per update, stated as such
- Check afterwards on every trained model (R2 included): slowest pole of the linearised model in the loop against the window
- Prerequisites: the production script reads the thesis data (?); the start-transient share measured on the pilot's trained checkpoint (section 3, Encoder) before runs 2 and 3, since a large transient would favour longer windows for the wrong reason
- Run-table rows before launch (D-090)

## 2. Budget method (all results)
- Budget counted in Adam updates (Adam moves each parameter by at most about the learning rate per update)
- One pilot run per result; the budget is its plateau times 1.5; the same budget for every arm within a result
- Checkpoint: best validation free-run error, then the polish
- The convergence test above also defines "GPU-h to convergence" for R8

## 3. Other settings
- Sample rate: 4 kHz kept (user 2026-09-27); 2 and 1 kHz lose 8.1 and 17.1 deg of phase margin against 20 kHz, beyond the 5 deg gate; 4 kHz loses 3.6 deg (D-166)
- Stride: 10 (?)
- Encoder lag: 29 for every state count (user 2026-09-27), the code's 2 nx + 1 at 8 added states; written source: Beintema thesis Sec. 2.5 (upper bound from Aeyels 1981); Hoekstra's papers use 7 to 13 without a stated rule
- Network: 16 x 2, tanh, for every state count (user 2026-09-27), the net of the 2-state runs; only the input and output layers and the added-state maps grow with the states, and the R2 table lists the parameter count per state count
  - Support: 81265 (2 states, 16 x 2) within 3.1 % of 81262 (8 states, 24 x 3) after 350 epochs (RESULTS-PLAN)
  - Caveat: at 18 epochs the larger setup was 2x ahead (D-189), so the small net converges more slowly; the pilot must reach its plateau (?)
  - Fixed across R2 as in Hoekstra (same network for 4 and 6 states); Kessels scales width with inputs by a fixed rule, rejected here for clean attribution
- Batch: 512 (?)
- Adam: lr 1e-5 kept (user 2026-09-27), eps 1e-16 (D-148, D-189); the field's default is 1e-3
- L-BFGS polish: 100 iterations, history 50, 16384 windows (D-171) (?)
- Normalisation: kept as now, zero mean and unit std from the training records (user 2026-09-27); the force noise leaves the positions smooth, so the velocity scale is unaffected
- Precision: float64 for both phases of every thesis run (user 2026-09-27; D-171 polish needs it, dtype coherence across phases)
- Encoder: production reconstructability init (Hoekstra 2026), trained jointly; the start transient (99.6 % of the 0.1 s loss) was measured with the untrained encoder; measure its share on the pilot's trained checkpoint, adopt the per-epoch rebuilt map only if it stays large (?)
- Parameter prior (penalty pulling the ten combinations back to their start): off everywhere (user 2026-09-27); where it was on it outweighed the data term 49 to 1433 times (D-193)
- Starts: a start is a network seed plus a 10 % detuning direction of the ten identifiable combinations (D-191); how many per run: section 4
- Budget: 200 epochs inside the 24 h wall (18 training records; 300 fitted only with 14)
- Production script: must be updated to read the thesis data (user 2026-09-27)
- Physics-only refits (B-refit, B-LTI-refit: baselines with only their parameters fitted, no network): dropped for now (user 2026-09-27); R1 uses the baselines at the correct parameters without training, R4 compares Aug-OBC with Aug-U and the truth (`RESULTS-DESIGN.md` change log 2026-09-27)
- Black-box search space: deferred (user 2026-09-27)

## 4. Run plan (user 2026-09-27)
| Runs | Arms and setting | Starts | Noise sets | Count |
|-|-|-|-|-|
| Window sweep | Aug-OBC, detuned, 8 added states, 0.1 / 0.2 / 0.4 s | 1 | noisy | 3 |
| R4/R6 | Aug-OBC and Aug-U, detuned, 2 added states | 5, the same on both sets | both | 20 |
| R4, correct start | Aug-OBC and Aug-U, 2 added states | 1, the same on both sets | both | 4 |
| R2 | Aug-OBC, 0 and 8 added states (2 comes from R4/R6) | 1, the same on both sets | both | 4 |
| **Now** | | | | **31** |
| R5 (later) | Aug-OBC and Aug-U on T_OA, detuned | 1 | noise-free | 2 |
- Every run except the sweep uses 0.1 s
- Noise comparison: each noisy run has a noise-free twin with the same seed and detuning direction (paired), so the noise is the only difference
- 5 starts only for R4/R6, whose claim is about spread; single-run comparisons count only if the difference exceeds 2 times the R4/R6 start standard deviation on the same noise set (HEURISTIC; assumes a similar spread elsewhere, stated in the thesis)
- R5 waits for the orthogonal addition T_OA and its data (not generated)
