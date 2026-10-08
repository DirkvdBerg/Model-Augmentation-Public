# blackbox-cl-trainable: critic

One reviewer agent, cold read of REPORT, IMPLEMENTATION, DECISIONS, RUNS, LITERATURE and the
code (2026-09-30), with one read-only check on the 6 validation records. Findings and handling.

## Findings
1. Major: the PASS bar is weaker than a zero-parameter predictor. y_hat = r scores 1.1e-5 to
   3.7e-5 m per axis on the validation records; the black box 3e-4 to 1.4e-3 m, i.e. 30 to 100x
   worse than y = r. The mean-model bar sits about 1000x above that; criterion (c) is near automatic.
2. Major: auto_fit_norm scales y by its move std (0.05 to 0.15 m); the loop's own behaviour
   (y - r, ~1e-5 m) is ~1e-4 of the normalised range. Fix: train on the servo error e = r - y
   (same score, RMS(y_hat - y) = RMS(e_hat - e)).
3. Minor: noisy = noise-free partly because the data are cast to float32 (resolution ~3e-8 m near
   0.3 m, above the 7e-9 m noise; 17 to 48 % of noisy samples identical after the cast). Treat as
   3 independent seeds, not 6 runs.
4. Minor: na = nb = 29 comes from Jan's rule (nx_phys + nx_ann) 2 + 1, i.e. contains the physical
   state count; nf = 400 rests on D-220, partly from model matrices. No truth, eigenvalues, test
   or TF records enter.
5. Major: the negative result misses the direct method (Jan's SUBNET on measured u -> y, then
   fine-tune under K1); the drift wording ignores that windows plus encoder bound drift; the
   joint input-output / two-stage methods use the bounded u; the exact integrator split for ICI
   was not tried; the nf = 1 stage trains only h and the encoder, not f.
6. Minor: scoring asymmetries (na_right 0 vs 1; the grey box's per-window controller reset;
   point-sampled r, f vs block-mean u; selection and reporting on the same validation records).
7. Minor: server runner not ready (GPU requested, CPU used; over the 24 h wall at 200 epochs; no
   early stopping or partial results).
8. Note: the BB-013 row uses old data.

## Handling
1. Accepted. REPORT now reports y_hat = r as the data-only reference and the ratio; the verdict is
   restated as "trains stably, has not learned the loop". The FP epoch-0 score on these records is
   still missing (?).
2. Accepted as the proposed next step (train on e = r - y, float64), for the user to approve
   (another ~3 h of local runs); not done.
3. Accepted: REPORT and RUNS state the float32 cause; counted as 3 seeds.
4. Accepted: disclosed in IMPLEMENTATION ("Plant information" section).
5. Partly. The direct-then-closed-loop route was tried on the old data by the previous session:
   all four open-loop checkpoints were unstable under the controller (blackbox-closed-loop
   REPORT G2); now cited in REPORT. The drift wording is corrected (a free run drifts; windowed
   training with an encoder does not). The exact integrator split and a thesis-data recheck of
   the direct route are listed as open (?). The nf = 1 remark is correct and noted in RUNS.
6. Accepted: listed in REPORT as known asymmetries to resolve before accuracies are compared.
7. Accepted: runner marked not ready in REPORT; fix belongs to the search setup.
8. Already labelled "Old data (previous session)"; kept as context.

## Second review: option B on the servo error (2026-10-04)
One reviewer agent, cold read of REPORT (first section), `train_bb_clmap.py`, `ref_scores.py`,
BB2-019 to BB2-027, RUNS and the result files; no runs. Findings and handling.
1. Note: scoring comparable (same samples, k0 29, float64 y; y_hat = r identical to 9 digits).
2. Minor: r, f point-sampled vs the grey box's block-mean u. Accepted: listed in REPORT
   asymmetries; block-averaging f left open.
3. Minor: dr and e = y - r are transforms of exogenous r, dr not ablated at final settings.
   Accepted: stated in REPORT.
4. Note: rules respected (no test loading, no override, no BLA).
5. Note: float64 claim correct (deepSI normalises in float64, casts after).
6. Major: "beats the grey box on Y" overstated; the epoch-0 grey box is worse than y_hat = r on Y.
   Accepted: headline reworded, the Y bar called uninformative.
7. Major: verdict branch asserted before e8. Accepted: marked provisional, settled by e8.
8. Minor: numbers check out; noisy vs noise-free at seed 1 is 1.6 / 1.9 / 4.3 %, not 1 to 2 %.
   Accepted: corrected.
9. Minor: grey-box bar is one detuning draw. Accepted: stated.
10. Minor: `result.json` criteria is the old BB2-018 test. Accepted: stated in REPORT.
11. Note: floor mixes anti-alias decimation with noise; total RMS does not prove per band.
    Accepted: stated.
12. Note: residual form exact only if the 4 kHz Tustin controller equals the 20 kHz one. Accepted:
    one line in REPORT.
