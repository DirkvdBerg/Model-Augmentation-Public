"""Why the trained model misses the absorber above its 212 Hz pole (run 42 and siblings).

Per checkpoint, on the four multisine validation records, Y channel, closed-loop free run from k0:
  L = y_model - y_bt(pipeline)      the correction the model learned (learned block + parameter drift)
  T = y_full - y_bt(oracle)         the correction the truth needs (absorber + friction)
  A = y_full - y_noabs(oracle)      the absorber part of T
H = S_LT / S_TT (Welch, 0.5 Hz), coherence; the model error spectrum; the static-only correction.
Also the K1 sensitivity |S_YY| on the rigid plant at Y = 0, to locate the closed-loop peak.
Reading: |H| = 1 at 0 deg is a perfect correction. A phase swing / gain notch near 212 Hz = wrong
learned pole; gain collapsing above 230 Hz = bandwidth limit; a defect centred at the |S| peak = loop.

Usage: python f_above_pole.py [job ...]   (default 86894 = run 42)
"""
import os
import sys

import numpy as np
import torch
from scipy.signal import csd, welch, coherence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e_thesis_checkpoints import JOBS, CK, cfg_for, static_only          # noqa: E402  (sets sys.path)
from b_error_budget import load_set, variant, oracle_cl_batch            # noqa: E402

FREQS = [110, 130, 150, 170, 190, 200, 205, 210, 212, 215, 220, 225, 230, 240, 250, 260, 264, 270, 280, 290, 296]
NPERSEG = 8000                                                            # 2 s at 4 kHz -> 0.5 Hz


def oracle_val(n_ms=4):
    """Oracle full, noabs, bt outputs of the first n_ms val records (from k0 = 29), cached on disk."""
    cache = os.path.join(HERE, 'outputs', 'f_oracle_val_cache.npz')
    if os.path.exists(cache):
        c = np.load(cache)
        return {v: c[v] for v in ('full', 'noabs', 'bt')}, c['ymeas'], int(c['N'])
    cfg, recs, _ = load_set('noisy')
    recs = [r for r in recs if r['split'] == 'val'][:n_ms]
    N = min(len(r['u']) for r in recs)
    out = {}
    for v in ('full', 'noabs', 'bt'):
        out[v] = oracle_cl_batch(np.stack([r['u'][:N] for r in recs]), np.stack([r['y'][:N] for r in recs]),
                                 np.stack([r['x0'] for r in recs]), [variant(r['tp'], v) for r in recs],
                                 [r['ctrl'] for r in recs], cfg.ts_new)
    ymeas = np.stack([r['y'][:N] for r in recs])
    np.savez(cache, ymeas=ymeas, N=N, **out)
    return out, ymeas, N


def sensitivity_Y(freqs, ts):
    """|S_YY| of K1 (design Y = 0) on the rigid stage plant at Y = 0: S = (I + G C)^-1."""
    from scipy.signal import freqz
    from gantry_dynamic.controller import sys_stage_frf, build_cfb_at
    cfb, _ = build_cfb_at(0.0, ts)
    out = []
    for f in freqs:
        G = sys_stage_frf(0.0, 1j * 2 * np.pi * f)
        C = np.diag([freqz(b, a, worN=[f], fs=1 / ts)[1][0] for b, a in cfb])
        out.append(abs(np.linalg.inv(np.eye(3) + G @ C)[2, 2]))
    return np.array(out)


def spec(fn, a, b=None, fs=4000):
    f, p = fn(a, b, fs=fs, nperseg=NPERSEG) if b is not None else fn(a, fs=fs, nperseg=NPERSEG)
    return f, p


def main(jobs):
    from gantry_dynamic.data import load_datasets, compute_normalization
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop, build_controller_bank
    from gantry_dynamic.training import load_checkpoint
    from gantry_dynamic.closed_loop_report import _without_learned, _true_free
    from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch

    orc, ymeas, N = oracle_val()
    print('oracle done', flush=True)
    Sy = sensitivity_Y(FREQS, 1 / 4000)
    data = norm = None
    for job in jobs:
        run, arm, na, seed = JOBS[job]
        cfg = cfg_for(arm, na, seed)
        np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        if data is None:
            data = load_datasets(cfg); norm = compute_normalization(cfg, data)
        np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        fs_ = build_model(cfg.hp, cfg, data, norm)
        fs_.simulator = build_closed_loop(fs_, norm, cfg, train_files=data.train_files,
                                          val_files=data.val_files, val_data=data.val_ckpt_data, verbose=False)
        load_checkpoint(fs_, os.path.join(CK, 'SSE_Interconnect_Composed_%s_best.pth' % job), True, 'lbfgs')
        files, sdl = list(data.val_files)[:4], list(data.val_list)[:4]
        bank, rows, _ = build_controller_bank(files, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u, dtype=cfg.dtype_pt)
        phy = next(b for b in fs_.hfn.connected_blocks if hasattr(b, 'free_params'))
        sys_ = {'model': fs_, 'bt': _without_learned(fs_, _true_free(phy))}
        if na > 0 and os.environ.get('F_NO_STATIC') != '1':
            sys_['static'] = static_only(fs_, 6 + na)
        pred = {}
        for n, s in sys_.items():
            with torch.no_grad():
                _, _, p = closed_loop_free_run_rms_batch(s, sdl, bank, rows, return_pred=True)
            pred[n] = np.stack([np.asarray(x)[:N, 2] for x in p])
        # consistency: pipeline bt against oracle bt
        d_bt = np.sqrt(np.mean((pred['bt'] - orc['bt'][:, :, 2]) ** 2))
        T = orc['full'][:, :, 2] - orc['bt'][:, :, 2]
        A = orc['full'][:, :, 2] - orc['noabs'][:, :, 2]
        L = pred['model'] - pred['bt']
        Ls = pred['static'] - pred['bt'] if 'static' in pred else None
        E = ymeas[:, :, 2] - pred['model']
        acc = {}
        for i in range(T.shape[0]):
            f, Ptt = spec(welch, T[i]); _, Plt = spec(csd, T[i], L[i]); _, Paa = spec(welch, A[i])
            _, Pee = spec(welch, E[i]); _, Pll = spec(welch, L[i])
            _, Pst = spec(csd, T[i], Ls[i]) if Ls is not None else (None, np.zeros_like(Plt))
            for k, v in (('tt', Ptt), ('lt', Plt), ('aa', Paa), ('ee', Pee), ('ll', Pll), ('st', Pst)):
                acc[k] = acc.get(k, 0) + v / T.shape[0]
        H = acc['lt'] / acc['tt']                      # csd(x, y) = conj(X) Y -> Y/X
        Hs = acc['st'] / acc['tt']
        coh = np.abs(acc['lt']) ** 2 / (acc['tt'] * acc['ll'])
        print('\n######## run %d (job %s) arm=%s n_a=%d | pipeline bt vs oracle bt rms %.2e m' % (run, job, arm, na, d_bt))
        print('  f[Hz]  |T| asd   |A| asd   |E| asd   |H|    ph(H)   coh   |H_static| ph   |S_YY|   (asd in m/sqrt(Hz))')
        for fq, s in zip(FREQS, Sy):
            j = np.argmin(np.abs(f - fq))
            print('  %5.0f  %.2e  %.2e  %.2e  %5.2f  %+6.0f  %4.2f   %5.2f %+5.0f   %.2f' % (
                f[j], np.sqrt(acc['tt'][j]), np.sqrt(acc['aa'][j]), np.sqrt(acc['ee'][j]), abs(H[j]),
                np.degrees(np.angle(H[j])), coh[j], abs(Hs[j]), np.degrees(np.angle(Hs[j])), s), flush=True)
        np.savez(os.path.join(HERE, 'outputs', 'f_above_pole_%s.npz' % job), f=f, H=H, Hs=Hs, coh=coh,
                 Ptt=acc['tt'], Paa=acc['aa'], Pee=acc['ee'], Pll=acc['ll'])


if __name__ == '__main__':
    main(sys.argv[1:] or ['86894'])
