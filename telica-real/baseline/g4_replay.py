"""G4 (2)-(3): closed-loop replay of one baseline variant on every readable record (TR-014).

    python telica-real/baseline/g4_replay.py <variant>     variant in rec | ds | nf | 70821

Records are processed in chunks (memory), both forms per chunk. Writes
outputs/g4_replay_<variant>/{metrics.json, psd.npz, example.png}. Summaries only.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
from scipy.signal import welch                                         # noqa: E402

from params import telica_params as tp                                 # noqa: E402
from baseline.records import iter_records                              # noqa: E402
from baseline.cl_sim import make_block, pack, simulate                 # noqa: E402
import rate                                                            # noqa: E402
tr_env.check_no_leak()

VARIANT = sys.argv[1]
OUT = tr_env.out_dir(f'g4_replay_{VARIANT}')
FIVE = VARIANT.endswith('_5k')                  # TR-022: 5 kHz data, controller and model step
FSR = rate.FS5 if FIVE else tp.FS
PRE = 500 if FIVE else 2000                     # 100 ms before motion
SOURCE = 'identified_5k' if FIVE else 'identified'
NPS = 512 if FIVE else 2048                     # the same 0.1 s Welch segment at either rate
HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = int(os.environ.get('G4_CHUNK', '40'))
AX = ('X1', 'X2', 'Y')

# 70821 recovered parameters: simulations/server-output/telica_param_recovery_70821.out,
# "Step 5: Parameter recovery", Tables 1-2 (lines 4141-4167). No Coulomb friction in 70821.
P70821 = dict(kb1=26672.6169, kb2=26672.6169, cg1=840.8646, cg2=783.7107, cy=664.2947,
              cb1=46.6072, cb2=46.6072, mh=12.3702, m1=13.4330, m2=12.9205, mb=26.7589,
              Jb=3.1965, Jh=0.1588, d=0.0623)


def variant_block():
    if VARIANT in ('rec', 'rec_a2', 'rec_a2_5k'):
        tag = '_a2' if VARIANT.startswith('rec_a2') else ''
        R = json.load(open(os.path.join(HERE, f'recovered_params{tag}.json')))
        mode = 'tanh' if VARIANT.startswith('rec_a2') else 'karnopp'        # TR-015 (2)
        return (make_block(raw=R['raw14'], cc=tuple(R['cc']), mode=mode,
                           ts=rate.TS5 if FIVE else tp.TS), R['raw14'], R['cc'])
    if VARIANT == 'ds':
        return make_block(raw=None, cc=tp.CC, mode='karnopp'), None, list(tp.CC)
    if VARIANT == 'nf':
        return make_block(raw=None, cc=(0.0, 0.0, 0.0), mode='none'), None, [0, 0, 0]
    if VARIANT == '70821':
        return make_block(raw=P70821, cc=(0.0, 0.0, 0.0), mode='none'), P70821, [0, 0, 0]
    raise ValueError(VARIANT)


def rms(a, axis=0):
    return np.sqrt(np.mean(a ** 2, axis=axis))


def main():
    t0 = time.time()
    blk, raw, cc = variant_block()
    print(f'[g4] variant {VARIANT}: cc {cc}, raw {raw if raw else "TR-003"}')
    per = []
    psd = dict(floor=0.0, meas=0.0, resid_d=0.0, resid_r=0.0, n_floor=0, n_mot=0)
    example = None
    chunk, meta = [], []

    def run_chunk():
        nonlocal example
        data = pack(chunk, pre=PRE)
        yd, yr, ud, ur = simulate(blk, data, source=SOURCE, log=lambda m: print(m, flush=True))
        for i, (s, op, it) in enumerate(meta):
            n, km = data['n'][i], data['k_motion'][i]
            mot = slice(km, n)
            e_meas = data['e'][i, mot]
            e_sim = data['r'][i, mot] - yd[i, mot]
            res_d = e_sim - e_meas
            res_r = yr[i, mot] - data['y'][i, mot]
            den = rms(e_meas)
            fin = bool(np.isfinite(yd[i, :n]).all() and np.isfinite(yr[i, :n]).all())
            bound = 100 * np.abs(e_meas).max(0)
            stab_d = bool(fin and (np.abs(e_sim).max(0) <= bound).all())
            stab_r = bool(fin and (np.abs(res_r).max(0) <= bound).all())
            u_log = data['u'][i, mot]
            row = dict(split=s, op=op, it=it, finite=fin, stable_direct=stab_d, stable_resid=stab_r,
                       nrmse_direct=(rms(res_d) / den).tolist() if fin else [np.nan] * 3,
                       nrmse_resid=(rms(res_r) / den).tolist() if fin else [np.nan] * 3,
                       e_meas_rms_nm=(den * 1e9).tolist(),
                       force_nrmse_direct=(rms(ud[i, mot] - u_log) / rms(u_log)).tolist()
                       if fin else [np.nan] * 3)
            per.append(row)
            if fin:
                f, pm = welch(e_meas, fs=FSR, nperseg=NPS, axis=0)
                _, pd_ = welch(res_d, fs=FSR, nperseg=NPS, axis=0)
                _, pr_ = welch(res_r, fs=FSR, nperseg=NPS, axis=0)
                psd['meas'] = psd['meas'] + pm; psd['resid_d'] = psd['resid_d'] + pd_
                psd['resid_r'] = psd['resid_r'] + pr_; psd['n_mot'] += 1; psd['f'] = f
            if example is None and s != 'train' and it == 'iter0' and fin:
                t = np.arange(n - km) / FSR
                example = dict(t=t, e_meas=e_meas, e_sim=e_sim, res_r=res_r, name=f'{s}/{op}/{it}')
        print(f'[g4] chunk of {len(meta)} done ({time.time() - t0:.0f} s)', flush=True)

    for s, op, it, d in iter_records():
        if FIVE:
            d = rate.decimate_record(d)
        k0 = d['motion_idx']
        still = d['e'][:max(k0 - (50 if FIVE else 200), 0)]
        if len(still) >= NPS:
            f, pf = welch(still, fs=FSR, nperseg=NPS, axis=0)
            psd['floor'] = psd['floor'] + pf; psd['n_floor'] += 1
        chunk.append({k: d[k] for k in ('r', 'q1', 'e', 'u', 'u_fb', 'motion_idx')})
        meta.append((s, op, it))
        if len(chunk) == CHUNK:
            run_chunk(); chunk, meta = [], []
    if chunk:
        run_chunk()

    # ==== summaries ====
    def med(rows, key):
        v = np.array([r[key] for r in rows], float)
        return np.nanmedian(v), np.nanmedian(v, axis=0)
    ho = [r for r in per if r['split'] != 'train']
    tr = [r for r in per if r['split'] == 'train']
    summ = dict(variant=VARIANT, cc=cc, n=len(per),
                all_stable=bool(all(r['stable_direct'] and r['stable_resid'] for r in per)),
                n_unstable_direct=int(sum(not r['stable_direct'] for r in per)),
                n_unstable_resid=int(sum(not r['stable_resid'] for r in per)))
    for name, rows in (('heldout', ho), ('train', tr)):
        for key in ('nrmse_direct', 'nrmse_resid', 'force_nrmse_direct'):
            m, ma = med(rows, key)
            summ[f'{name}_{key}_median'] = float(m)
            summ[f'{name}_{key}_median_per_axis'] = ma.tolist()
    # iter0-only held-out (feedback-dominated) as a separate line
    ho0 = [r for r in ho if r['it'] == 'iter0']
    summ['heldout_iter0_nrmse_direct_per_axis'] = med(ho0, 'nrmse_direct')[1].tolist()
    f = psd['f']
    floor = psd['floor'] / max(psd['n_floor'], 1)
    meas = psd['meas'] / max(psd['n_mot'], 1)
    rd = psd['resid_d'] / max(psd['n_mot'], 1)
    rr = psd['resid_r'] / max(psd['n_mot'], 1)
    bands = [(1, 20), (20, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10000)]
    bands = [(lo, min(hi, FSR / 2)) for lo, hi in bands if lo < FSR / 2]
    summ['psd_bands'] = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        summ['psd_bands'].append(dict(
            band=[lo, hi],
            resid_direct_over_floor=(rd[m].mean(0) / floor[m].mean(0)).tolist(),
            resid_resid_over_floor=(rr[m].mean(0) / floor[m].mean(0)).tolist(),
            meas_over_floor=(meas[m].mean(0) / floor[m].mean(0)).tolist()))
    json.dump(dict(summary=summ, per_record=per), open(os.path.join(OUT, 'metrics.json'), 'w'),
              indent=1)
    np.savez(os.path.join(OUT, 'psd.npz'), f=f, floor=floor, meas=meas, resid_direct=rd,
             resid_resid=rr)
    print(f'[g4] {VARIANT}: stable all = {summ["all_stable"]} (unstable direct '
          f'{summ["n_unstable_direct"]}, residual {summ["n_unstable_resid"]} of {len(per)})')
    for name in ('heldout', 'train'):
        print(f'[g4] {VARIANT} {name}: NRMSE direct median {summ[f"{name}_nrmse_direct_median"]:.3f} '
              f'per axis {np.round(summ[f"{name}_nrmse_direct_median_per_axis"], 3).tolist()}; '
              f'residual form {summ[f"{name}_nrmse_resid_median"]:.3f} '
              f'{np.round(summ[f"{name}_nrmse_resid_median_per_axis"], 3).tolist()}; force NRMSE '
              f'{np.round(summ[f"{name}_force_nrmse_direct_median_per_axis"], 3).tolist()}')
    print(f'[g4] {VARIANT} held-out iter0 NRMSE direct per axis '
          f'{np.round(summ["heldout_iter0_nrmse_direct_per_axis"], 3).tolist()}')
    for b in summ['psd_bands']:
        print(f'[g4] {VARIANT} PSD {b["band"][0]}-{b["band"][1]} Hz / floor: measured '
              f'{np.round(b["meas_over_floor"], 1).tolist()}, direct residual '
              f'{np.round(b["resid_direct_over_floor"], 1).tolist()}, residual-form '
              f'{np.round(b["resid_resid_over_floor"], 1).tolist()}')
    if example is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for j, ax in enumerate(AX):
            axs[j].plot(example['t'], example['e_meas'][:, j] * 1e6, 'k', lw=0.7, label='measured e')
            axs[j].plot(example['t'], example['e_sim'][:, j] * 1e6, 'C1', lw=0.7,
                        label='sim e (direct form)')
            axs[j].plot(example['t'], example['res_r'][:, j] * 1e6, 'C0', lw=0.7, alpha=0.7,
                        label='y_model - y_data (residual form)')
            axs[j].set_ylabel(f'{ax} [um]')
        axs[0].legend(fontsize=8); axs[-1].set_xlabel('t from motion start [s]')
        fig.suptitle(f'G4 {VARIANT}: {example["name"]}')
        fig.tight_layout(); fig.savefig(os.path.join(OUT, 'example.png'), dpi=110)
    print(f'[g4] done in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
