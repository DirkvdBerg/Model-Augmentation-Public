# Overnight 2026-08-23: BLA Phase B, random vs fitted initialisation

Authorised by `tasks/handoffs/2026-08-22-bla-phase-b-random-vs-fitted.md` section 13. Appended
after every unit, never composed at the end. Every unit gets a row whether it succeeds, fails,
aborts or is dropped.

```
BLA e-7 WITHOUT HEURISTICS?  NO. A2 = 2.0385e-06 against < 1.0e-06 (band 3.80-4.89e-07).
                             No heuristic was reintroduced; the constant-free eps rule deleted
                             the 158 Hz mode from the init (5.02 Hz pair kept), the trained
                             block ended a net liability (F = -0.096), and D10's falsifier fired
RANDOM e-7?                  NO. A1 = 2.0152e-06; F = 0.0218 (x_a carries ~2 % of the
                             improvement). A2 within 1.1 % of A1: the BLA line CLOSES on this
                             construction (pre-registered condition)
HEURISTICS OUTSTANDING:      none as of Block W: D5 eps resolved by derivation (D-156, split-half
                             H-infinity disagreement); the 2.0x ablation threshold replaced by the
                             improvement fraction F with a noise-draw significance floor (D-157)
TELICA AUDIT:                complete, written into DESIGN.md D9 before any A2 launch; the one
                             consumed-at-init quantity that is NOT available (x0 as a true state)
                             is substituted by per-record least-squares x0 plus differencing
NOISE ARMS:                  none ran. A2 noisy NOT RUNNABLE (construction refused, row F/N1);
                             A0 noisy dropped first per drop order (N3); A1 noisy dropped for
                             time (N2: arms cost 2.5-2.9 h each against the estimated ~85 min)
PRE-FLIGHT dL/dp AT STEP 1:  W^a 2.58e-11 / 5.19e-12, nu_log 2.70e-11, theta 1.82e-11, B 1.02e-10
                             (all non-zero; gate PASSED)
```

## Reference re-basing, recorded BEFORE any arm ran

The restored harness's untrained closed-loop free-run scalar is `2.534187007593955e-06` m, not the
historical `2.1866011034177349e-06`. The handoff's own drop table anticipated exactly this ("the
restore changes `closed_loop_free_run_rms`, so nothing is comparable to `1.3933793e-06` until
[A0] re-establishes it on the restored harness"). Consequences, fixed now:

* the D-072 pre-flight gate is judged in substance: the ARM-APPLIED untrained scalar must equal
  the NO-ARM untrained scalar bit-identically (baseline equality), on this harness;
* the section 13 stop condition "epoch 1 worse than 2.1866011034177349e-06" is applied as "epoch 1
  worse than the run's own printed untrained base", which is the condition's stated purpose;
* every historical comparison value (plateau `1.3933793e-06`, target band) is quoted with the
  caveat that A0 re-establishes the reference chain on this harness.

Cause not diagnosed tonight (candidates: uncommitted working-tree changes in
`closed-loop-controller/cl_*.py` since the gate was recorded, or regenerated trajectory data - the
Matlab `gtd_*` files are modified in `git status`). Logged rather than chased, per the budget.

## Unit rows

| # | unit | hypothesis | what ran | artefact | number | verdict | eliminates |
|-|-|-|-|-|-|-|-|
| P | pre-flight (hard gate) | D7's amendment: ReZero + random `W^a` unblocks the augmented gradient one update late | restore of `closed_loop.py`; `probe_preflight.py` | `runs/preflight.json` | `dL/dW^a` step 0 = exactly 0; `dL/d alpha` step 0 = `3.69e-06`; step 1: `W^a 2.58e-11 / 5.19e-12`, block `nu_log 2.70e-11`, `theta 1.82e-11`, `B 1.02e-10` - **non-zero** | **PASS** (D-072 baseline equality bit-identical arm-vs-no-arm at `2.534187007593955e-06`; historical constant re-based, see note above) | the "D7 amendment wrong, no arm runs" branch |
| F | A2 construction | the pole-gate construction yields an installable `(A_r, B_u)` under the resolved `eps`, with the refusal condition live | `fit_reduce.py` | `runs/fit_reduce.json`, `runs/a2_spec_clean.json` | clean-ARX: `na 28`, `eps 8.154e-03`, **`nx_aug = 2` derived**, pair `r 0.99927 @ 5.02 Hz` (**absorber mode DELETED by the bound - reported, not repaired**); clean-IV unstable; noisy: both estimators REFUSED; nothing-to-find case: REFUSED both | **A2-clean runnable; A2-noisy NOT RUNNABLE** (oracle-free order selection is VAF-based and floor-dominated under noise - the named missing piece); refusal condition exercised and fires | `nx_aug = 8` assumptions; a silent-default band under noise |
| A0 | A0 noiseless training (control) | re-establish the reference chain on the restored harness; ablation decides whether `x_a` was dead at the plateau | `cl_train.py`, 4 epochs, 1040 updates, `4cdb7c1` config, seed 0 | `runs/cl_train_bla_a0_clean.json`; ckpt `SSE_Interconnect_MultipleShooting_1ht2Zw_best.pth` | untrained `2.5342e-06` -> trained `1.9050e-06` (`+24.8 %`); val series `2.5342, 1.9173, 1.9299, 1.9278, 1.9050 e-06`; 9850 s | plateau-shaped improvement reproduced on the restored harness (epoch 1 does most, then flat); ablation F pending (unit A0abl) | stop rule (epoch 1 better than base) |
| A0abl | A0 ablation | pre-registered: `F ~ 0` confirms D7's amendment (`x_a` dead at the plateau) | `run_ablation.py`, surfaces A and B, noiseless (deterministic) | `BLA-Augmentation/runs/ablation_a0_clean.json` | trained `1.904981e-06`; A blind `1.905385e-06` (`1.0002x`); B zero `1.905437e-06` (`1.0002x`); **`F = 0.0007`** | **`x_a` DEAD at the plateau - D7's amendment CONFIRMED.** The `4cdb7c1` improvement is entirely through the physical rows; A1/A2 are testing the right thing | the "plateau was never the dead zone" branch (answer 3 of the open question) |
| A2 | A2 noiseless training (fitted init, THE DELIVERABLE) | the fitted reduced realisation (5.02 Hz pair, mode deleted by the bound) beats random and reaches e-7 | `cl_train.py` + `BLA_ARM_SPEC=a2_spec_clean.json`, 4 epochs, 1040 updates, seed 0 | `runs/cl_train_bla_a2_clean.json`; ckpt `SSE_Interconnect_MultipleShooting_jB05lQ_best.pth` | untrained `2.5342e-06` -> trained `2.0385e-06` (`+19.6 %`); val series `2.5342, 2.2885, 2.2046, 2.1069, 2.0385 e-06`, still descending at epoch 4 | **e-7 NOT reached** (`2.04e-06` vs bar `< 1.0e-06`); WORSE than A0's `1.9050e-06` at matched budget. Monotone descent means not converged, but the bar is 4 epochs by pre-registration. Ablation F pending; D10 pole falsifier pending | |
| A2abl | A2 ablation + D10 falsifier | pre-registered: F on surface B; D10: poles moving AWAY from truth while RMS improves refutes composition | `run_ablation.py` + pole readout from the best checkpoint | `BLA-Augmentation/runs/ablation_a2_clean.json` | trained `2.038484e-06`; A blind `1.990814e-06` (`0.9766x`); B zero `1.990788e-06` (`0.9766x`); **`F = -0.0962`**; poles: installed `r 0.999266 @ 5.019 Hz` -> trained `r 0.999258 @ 3.926 Hz`; `||B||` grew `2.2e-03 -> 2.2e-02`; `alpha = 2.8e-05` | **NEGATIVE, twice over.** (1) The fitted block is a net LIABILITY: removing `x_a` IMPROVES the free run. (2) **D10's falsifier FIRES**: training moved the installed pole 22 % further from the truth while the RMS improved - the fitted values acted as ballast, not as the missing dynamics. The BLA-init-composes-with-closed-loop-training assumption is refuted on this construction | the "fitted low-frequency residual content helps" reading |
| A1 | A1 noiseless training (random init) | the framework's own random initialisation (full-disk poles, `U(-1,1)` B, Xavier `W^a`, ReZero) with the gradient unblocked reaches e-7 | `cl_train.py` + `BLA_ARM_SPEC=spec_a1.json`, 4 epochs, 1040 updates, seed 0; drawn pole `r 0.734 @ 523.4 Hz` | `runs/cl_train_bla_a1_clean.json`; ckpt `SSE_Interconnect_MultipleShooting_aQlAow_best.pth` | untrained `2.5342e-06` -> trained `2.0152e-06` (`+20.5 %`); val series `2.5342, 2.3316, 2.1938, 2.0938, 2.0152 e-06`; 10609 s | **e-7 NOT reached.** A1 and A2 track within `1.1 %` at the end and within `2 %` at every epoch: **A2 is within noise of A1 - the pre-registered condition that CLOSES the BLA line on this construction.** Both are worse than A0 (`1.9050e-06`) at matched budget. Ablation F pending (unit A1abl) | "random cannot do what fitted does" at this nx and budget |
| N1 | A2 noisy + ablation | n/a | **not runnable**: the construction refused under noise (both estimators, row F) | `runs/fit_reduce.json` | n/a | not runnable, construction refused; the missing oracle-free noise-proof selection criterion is the named next piece | |
| N2 | A1 noisy + ablation | n/a | **dropped for time**: measured arm cost `2.5-2.9 h` each against the handoff's `~85 min`; budget consumed by run 3 + ablations. What ran instead: A1-clean ablation | n/a | n/a | dropped for time | |
| N3 | A0 noisy | n/a | **dropped first**, per the pre-registered drop order | n/a | n/a | dropped by pre-registration | |
| A1abl | A1 ablation | pre-registered: F on surface B decides whether the unblocked `x_a` is load-bearing | `run_ablation.py`, noiseless (deterministic) | `BLA-Augmentation/runs/ablation_a1_clean.json` | trained `2.015236e-06`; A blind `2.026522e-06` (`1.0056x`); B zero `2.026541e-06` (`1.0056x`); **`F = 0.0218`** | `x_a` is USED but MINOR (~2 % of the improvement flows through it, against A0's exact-dead 0.0007). Of the three candidate answers to the open question: **answer 2 - the dead zone was real, is fixed (pre-flight + non-zero F), and fixing it is not sufficient** at `nx_aug = 2`, 4 epochs | answer 1 ("the dead zone was the whole story") and answer 3 ("the plateau was never the dead zone") |
| C | mid-session correction (`tasks/handoffs/2026-08-22-d1-reopened-correction.md`) | n/a | applied: D9 gains a binding provenance paragraph (A2 = simulation result, residual construction contested, no transfer claim); audit re-headed as a checklist; "resolution transfers" withdrawn; D-158 addendum; RESULTS.md provenance note | `DESIGN.md` D9, `docs/decisions.md` D-158, `RESULTS.md` | n/a | arms, bar, budget, drop order unchanged per the correction's own instruction | any Telica-portability reading of tonight's A2 |
| W | Block W (authoring, no GPU) | D9/D10 can be filled and both outstanding constants resolved without a tuned constant | writing only; two deep-research subagents (eps source; residual-fit-init licence); 7 quotes verified MATCH OK (claims 29, 30, 31) | `DESIGN.md` D9/D10, `EVIDENCE.md` claims 29-31, `docs/decisions.md` D-156..D-158 | n/a | **complete.** eps resolved by derivation; ablation threshold replaced by `F`; literature confirms no precedent for fitting the added dynamic block (the step is ours) and no published eps rule | the "refuse and stop" branch for both constants |

## Recommended next action (one, per the standing rule)

**Replace D5's reduction criterion with a mode-preserving, still constant-free one before any
further BLA arm runs**: frequency-weighted balanced reduction with the weight taken from the
split-half-agreed part of the fitted spectrum (the reproducibility idea already named in
RESULTS.md), so that content BOTH record halves agree on - which is exactly the 158 Hz mode, per
the pole gate - cannot be discarded in favour of low-frequency content they disagree on. The
candidate source is already fetched and unread:
`literature/model-reduction/anand2025_frequency-weighted-extended-BT_arXiv2512.02298.pdf` (flagged
in DESIGN.md D5 as the principled route, "left open"). Rationale: tonight's chain shows the ONLY
step that failed without a heuristic is the reduction's choice of WHAT to keep - the fit finds the
mode (pole gate, 0.12 %), the gradient path is fixed (pre-flight), the wiring holds D-072
bit-exactly - so the one missing derivation is a weighting that keeps identified, reproducible
modes, and that is also precisely the criterion that would have refused the noisy arm for the
right reason instead of the VAF-floor reason.

Script time spent: ~10.1 h (pre-flight 0.35 + fit/reduce 0.12 + three arms 2.7-2.9 each + three
ablations 0.6-0.8 overlapped + refusal exercise inside fit/reduce). Nothing is left running.
