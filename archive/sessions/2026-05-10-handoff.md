# Session Handoff

_Previous sessions archived to `archive/sessions/`._

**Last written**: 2026-05-08 by Claude (Sonnet 4.6)

---

## Open Blockers (carried forward)

- **LFR discretization paper**: Still not found. Less critical since RK4 is chosen.
- **M0 choice**: M0 = M(0) vs M(Y_nom=0.3). State explicitly in write-up.
- **Sample rate**: D-012 — 16 kHz (main.m) vs 20 kHz (ETEL spec), unresolved.
- **Float32 acceptability**: Run training in both dtypes, compare param_table().

---

## Parameter Recovery — Open Issues

| # | Issue | Status | Dependency |
|---|-------|--------|------------|
| 2 | Multiple trajectories | **Start here** | none |
| 1 | Channel normalization | Design complete — implement next | none |
| 3 | MSE vs RMSE logging | Done (2026-04-16) | — |
| 4 | Multi-start initialization | Not started | needs Issue 2 first |
| 5 | Local minimum diagnosis | Blocked | needs Issues 2 + 4 |
| 6 | Log constraint | Design pending | resolve Issues 2+4 first |
| 7 | Identifiability | Subsumed by 2 | fix is Issue 2 |

See `archive/sessions/2026-05-08-handoff.md` for full design notes on each issue.

---

## Diagnostics restructuring — completed 2026-05-08

`experiment_diagnostics.py` restructured per `docs/diagnostics-theory-basis.md`:
- `fs_new` now derived from `f_osc_min` (physics) via `_FS_RULE_FACTOR=10`, not from `f_99`
- `segment_len = max(period rule, 10*tau_max, 10*n_params)` per Lecture 9 slide 9
- `f_99` is now warning-only in the report
- `[::D]` strides replaced with `scipy.signal.decimate` (anti-aliasing)

MATLAB multisine `f_high` fix still pending (item 2 from `docs/multisine-diagnostics-interface.md`).
