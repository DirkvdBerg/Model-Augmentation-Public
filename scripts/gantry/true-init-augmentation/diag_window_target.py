"""Task item (i): is the per-window TARGET clean when the IC is exact?

THE QUESTION. Training fits `nf`-sample windows. Each window's target is
`y_truth - y_baseline(x0)`, and every run to date carried three faults in that
target at once: the encoder's initial condition was wrong, the record's velocity
rows are finite differences, and the baseline's mass distribution omits the
absorber's static centre-of-gravity term. This measures the target with all
three removed, BEFORE spending a training run on it.

THREE SEEDING ARMS, one start grid, so the numbers are directly comparable:

  record   x0 = x_logical[s]      today's "True-x0 (oracle)". Its velocity rows
                                  are gradient() finite differences. This should
                                  reproduce the established config-A scatter
                                  (Y 1.045e-04 m) and is the harness gate.
  exact    x0 = truth x6[s]       THIS EXPERIMENT. Velocities are integrator
                                  states of the 8-state truth, not differences.
  freerun  one continuous run from the rest IC, chunked on the SAME windows.
                                  This is the model-plus-discretisation floor:
                                  no per-window re-seeding happens at all, so
                                  nothing but the missing absorber and the
                                  integrator can contribute.

`freerun` is the acceptance threshold, and it is measured HERE rather than taken
from the 20 kHz figures in the coulomb-offset log, because this check runs at the
training rate (4 kHz, block-mean input, up_sample = 1) and a floor from a
different rate is a floor for a different question (trap T6 in that log: "measure
the floor on the dataset in use"). The 20 kHz numbers are printed alongside for
continuity, not used as the criterion.

WHAT IS REPORTED, per state and per output channel:
  scatter   std ACROSS windows of the per-window mean error. This is the
            quantity that matters. The per-window DC is already zero-mean
            across windows (coulomb-offset F4, HAC |t| < 1.3 on six axis/record
            combinations), so its variance, not its mean, is what corrupts the
            training target.
  bias      the grand mean, with a Newey-West t so "zero-mean" is a measurement
            and not an assumption.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \\
      -n GraduationProject python -u \\
      scripts/gantry/true-init-augmentation/diag_window_target.py
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

from plant_cog import make_block, rollout_batch      # noqa: E402
from data_exact import exact_truth                   # noqa: E402

sys.path.insert(0, REPO)
from model_augmentation.systems.gantry_ss import P as _P   # noqa: E402

P_np = _P.numpy().astype(np.float64)
OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')

RECORDS = ('V1_standstill_Yp10', 'V3_ysweep_Yp10')
NF = 400                 # 0.100 s at 4 kHz -- the training window
STRIDE = 100             # window-start grid
K0 = 17                  # max(na, nb) = 2*(6+2)+1, the pipeline's first usable sample
STATES = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']
UNITS = ['m', 'rad', 'm', 'm/s', 'rad/s', 'm/s']
# Measured at 20 kHz on the frictionless free run from the exact IC
# (coulomb-offset diag_continuity_limit.py, config B). Printed for continuity;
# NOT the criterion here -- see the module docstring.
B20K = dict(X=9.1470e-08, Theta=3.7303e-09, Y=2.9787e-08)
A4K = dict(X=6.3023e-07, Theta=1.1584e-06, Y=1.0448e-04)   # config A, 4 kHz


def newey_west_t(v):
    """t statistic of mean(v) with a Newey-West HAC variance, lag = n^(1/3).

    # THEORY: Newey & West (1987) HAC estimator; bandwidth n^(1/3) is the
    # standard rule of thumb. Windows on one record are serially correlated, so
    # an iid standard error would overstate significance.
    """
    v = np.asarray(v, float)
    n = len(v)
    if n < 8:
        return float('nan')
    e = v - v.mean()
    L = max(1, int(round(n ** (1 / 3))))
    s = float((e ** 2).mean())
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        s += 2.0 * w * float((e[l:] * e[:-l]).mean())
    if s <= 0:
        return float('nan')
    return float(v.mean() / np.sqrt(s / n))


def window_stats(err_win):
    """err_win (n_win, nf, k) -> per-window means and their statistics."""
    wm = err_win.mean(axis=1)                     # (n_win, k)
    return dict(
        win_means=wm,
        scatter=wm.std(axis=0),
        bias=wm.mean(axis=0),
        t=np.array([newey_west_t(wm[:, c]) for c in range(wm.shape[1])]),
        inwin_rms=np.sqrt((err_win ** 2).mean(axis=(0, 1))),
    )


def run_record(name, dtype=torch.float64, cog=True, verbose=True):
    tr = exact_truth(name)
    rec, x6 = tr['rec'], tr['x6']
    N = len(rec['u'])
    starts = np.arange(K0, N - NF + 1, STRIDE)
    nb = len(starts)
    u_win = np.stack([rec['u'][s:s + NF] for s in starts])         # (B, NF, 3)
    ref = np.stack([x6[s:s + NF] for s in starts])                 # (B, NF, 6)
    blk = make_block(Y_op=None, cog=cog, ts=rec['ts'], up_sample=1, dtype=dtype)

    out = {}
    # --- per-window re-seeding arms ---------------------------------------
    for tag, x0 in (('record', rec['x_logical'][starts]),
                    ('exact', x6[starts])):
        sim = rollout_batch(blk, x0, u_win, n_out=6)
        out[tag] = window_stats(sim - ref)

    # --- free run from the rest IC, chunked on the SAME windows ------------
    x0_rest = np.array([[0., 0., rec['Y_op'], 0., 0., 0.]])
    free = rollout_batch(blk, x0_rest, rec['u'][None, :N], n_out=6)[0]   # (N, 6)
    efree = free - x6
    out['freerun'] = window_stats(np.stack([efree[s:s + NF] for s in starts]))
    out['_free_err'] = efree
    out['_starts'] = starts
    out['_nwin'] = nb
    if verbose:
        print(f'  {name}: {N} samples, {nb} windows of {NF} '
              f'({NF*rec["ts"]*1e3:.0f} ms), stride {STRIDE}, '
              f'{"float64" if dtype == torch.float64 else "float32"}, '
              f'CoG {"ON" if cog else "off"}')
    return out, tr


def print_table(out, title):
    print(f'\n=== {title} ===')
    print(f'  {"state":<8}{"unit":<7}' + ''.join(f'{k:>15}' for k in
                                                 ('record seed', 'exact seed', 'free run'))
          + f'{"exact/free":>12}{"gain":>9}')
    for c in range(6):
        r = out['record']['scatter'][c]
        e = out['exact']['scatter'][c]
        f = out['freerun']['scatter'][c]
        print(f'  {STATES[c]:<8}{UNITS[c]:<7}{r:>15.4e}{e:>15.4e}{f:>15.4e}'
              f'{e/max(f,1e-300):>12.2f}{r/max(e,1e-300):>8.1f}x')
    print(f'  scatter = std ACROSS windows of the per-window mean error; '
          f'"gain" = record/exact')

    print(f'\n  zero-mean check on the per-window means (Newey-West HAC t)')
    print(f'  {"state":<8}' + ''.join(f'{k:>17}' for k in
                                      ('exact bias', 'exact t', 'free bias', 'free t')))
    for c in range(6):
        print(f'  {STATES[c]:<8}{out["exact"]["bias"][c]:>17.4e}'
              f'{out["exact"]["t"][c]:>17.2f}{out["freerun"]["bias"][c]:>17.4e}'
              f'{out["freerun"]["t"][c]:>17.2f}')


def main():
    print('Per-window target check: is the target clean from an EXACT initial condition?\n')
    allres = {}
    for name in RECORDS:
        print(f'--- {name} ---')
        out64, tr = run_record(name, dtype=torch.float64, cog=True)
        print_table(out64, f'{name}  float64, CoG ON')

        outoff, _ = run_record(name, dtype=torch.float64, cog=False, verbose=False)
        print(f'\n  CoG OFF control (same arms, uncorrected baseline)')
        print(f'  {"state":<8}' + ''.join(f'{k:>16}' for k in
                                          ('exact CoG on', 'exact CoG off', 'free CoG on',
                                           'free CoG off')))
        for c in range(6):
            print(f'  {STATES[c]:<8}{out64["exact"]["scatter"][c]:>16.4e}'
                  f'{outoff["exact"]["scatter"][c]:>16.4e}'
                  f'{out64["freerun"]["scatter"][c]:>16.4e}'
                  f'{outoff["freerun"]["scatter"][c]:>16.4e}')

        out32, _ = run_record(name, dtype=torch.float32, cog=True, verbose=False)
        print(f'\n  float32 arm (the dtype training actually runs in, cfg.use_f64 = False)')
        print(f'  {"state":<8}{"exact f64":>16}{"exact f32":>16}{"free f64":>16}{"free f32":>16}')
        for c in range(6):
            print(f'  {STATES[c]:<8}{out64["exact"]["scatter"][c]:>16.4e}'
                  f'{out32["exact"]["scatter"][c]:>16.4e}'
                  f'{out64["freerun"]["scatter"][c]:>16.4e}'
                  f'{out32["freerun"]["scatter"][c]:>16.4e}')

        # --- output space: what the loss actually sees ---------------------
        print(f'\n  OUTPUT space (stage positions, what the MSE is computed on)')
        print(f'  {"chan":<8}{"record seed":>16}{"exact seed":>16}{"free run":>16}')
        lab = ['X1', 'X2', 'Ystage']
        ystats = {}
        for tag in ('record', 'exact', 'freerun'):
            wm = out64[tag]['win_means'][:, :3]            # logical position means
            ystats[tag] = (P_np.T @ wm.T).T                # logical -> stage
        for c in range(3):
            print(f'  {lab[c]:<8}{ystats["record"][:,c].std():>16.4e}'
                  f'{ystats["exact"][:,c].std():>16.4e}'
                  f'{ystats["freerun"][:,c].std():>16.4e}')

        allres[name] = dict(
            n_win=int(out64['_nwin']), nf=NF, stride=STRIDE, k0=K0,
            f64_cog_on={k: dict(scatter=out64[k]['scatter'].tolist(),
                                bias=out64[k]['bias'].tolist(),
                                t=out64[k]['t'].tolist(),
                                inwin_rms=out64[k]['inwin_rms'].tolist())
                        for k in ('record', 'exact', 'freerun')},
            f64_cog_off={k: dict(scatter=outoff[k]['scatter'].tolist())
                         for k in ('record', 'exact', 'freerun')},
            f32_cog_on={k: dict(scatter=out32[k]['scatter'].tolist())
                        for k in ('record', 'exact', 'freerun')},
            output_stage_scatter={k: ystats[k].std(axis=0).tolist()
                                  for k in ('record', 'exact', 'freerun')},
            replay_gate=[float(v) for v in tr['gate']],
        )
        np.savez_compressed(
            os.path.join(HERE, 'figures', f'_winmeans_{name}.npz'),
            record=out64['record']['win_means'], exact=out64['exact']['win_means'],
            freerun=out64['freerun']['win_means'], starts=out64['_starts'],
            free_err=out64['_free_err'])
        print()

    print('=== ACCEPTANCE (handoff section 10) ===')
    print('  criterion: per-window scatter on EVERY state at or below the '
          'free-run-from-exact-IC level.')
    print(f'  20 kHz reference (coulomb-offset config B), positions only: '
          f'X {B20K["X"]:.4e}  Theta {B20K["Theta"]:.4e}  Y {B20K["Y"]:.4e}')
    print(f'  4 kHz config A (encoder-style seeding), positions only:     '
          f'X {A4K["X"]:.4e}  Theta {A4K["Theta"]:.4e}  Y {A4K["Y"]:.4e}')
    v1 = allres[RECORDS[0]]['f64_cog_on']
    print(f'\n  {RECORDS[0]}, exact vs free-run ratio per state:')
    for c in range(6):
        e, f = v1['exact']['scatter'][c], v1['freerun']['scatter'][c]
        verdict = 'AT FLOOR' if e <= 2 * f else ('near' if e <= 10 * f else 'ABOVE')
        print(f'    {STATES[c]:<8}{e:>14.4e} / {f:>12.4e} = {e/max(f,1e-300):>8.2f}   {verdict}')

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'true_init_window_target.json')
    with open(p, 'w') as fh:
        json.dump(dict(nf=NF, stride=STRIDE, k0=K0, fs=4000,
                       b20k=B20K, a4k=A4K, records=allres), fh, indent=2)
    print(f'\n  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
