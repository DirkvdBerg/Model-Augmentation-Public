"""Does the REAL machine show lightly damped modes, and where?  Tier 1: logged signals only.

WHY THIS IS THE FIRST REAL-DATA QUESTION. The augmentation's whole initialisation proposal is built
on `rho(A_aa)` at a resonance. On the SIMULATED data `cl_residual_spectrum.py` recovers the injected
absorber cleanly (158.20 Hz, zeta 0.058, against a truth of 150 Hz / 0.05, 54 of 54 dominant peaks
inside a 14 Hz band). That validates the METHOD and says nothing about Telica: the 150 Hz absorber is
something we injected, and the user's position is that the real machine is not known to have it.
So this is DISCOVERY, and "no lightly damped mode" is an admissible and consequential answer.

WHY IT NEEDS NO BASELINE MODEL, WHICH IS THE POINT OF DOING IT FIRST. The simulation arm computes
`r = y - y_baseline`, which needs the FP model, its recovered parameters, the P-transform frame and
a 20 kHz -> 4 kHz decimation. None of that is required to ask whether the machine has resonances.
Two logged signals answer it directly:

    e    = r - q1   [m]   the TRACKING ERROR (= M2). What the loop failed to reject.
    i_fb            [A]   the FEEDBACK CURRENT (= MF230). What the controller had to inject.

`i_fb` is the important one and it is why this is Tier 1 rather than a compromise. The tracking error
is suppressed below the ~crossover by the sensitivity, so a low-frequency plant feature is INVISIBLE
in `e` -- and on a real machine the likely missing dynamics (friction, stick-slip, cable forces,
drift) are exactly there. The feedback effort is not attenuated the same way: it IS the loop's own
estimate of what the model does not explain. Reading both, and comparing where they peak, is the
un-blinded view that the simulation arm's docstring flags as owed.

WHAT IT CANNOT SAY. This measures the CLOSED-LOOP machine, so every peak here is a property of plant
AND controller together: controller notches and the loop's own resonances appear too. A peak is
therefore a candidate, not a plant mode. Two cheap discriminators are applied:
  * a plant resonance moves the tracking error and the feedback current TOGETHER; a controller
    artefact typically does not,
  * a plant resonance is present at every operating point, a trajectory artefact is not.
Separating them properly needs the controller FRF divided out (`dFeedbackControllersTelica_ba.mat`,
verified bit-exact, D-074) and that is Tier 2 along with the baseline residual.

DATA. `iter0` only, per operating point: feedforward is off there, so `MF30 == MF230` bit-identically
and the log is a pure-feedback run (schema, "Iteration file meaning"). Later iterations have ILC
feedforward that is DESIGNED to cancel the tracking error, so their spectra are shaped by the ILC and
are the wrong thing to average over. 15 operating points across xpos {-60,-135,-210} mm and
ypos {-200,-120,-40,+40,+120} mm, which is also what makes the position-dependence reading possible.

This script READS the Telica logs through `telica_loader.load_telica_log_cl` and reports only
aggregate spectra. No raw file contents are printed.

Usage:  python -u telica_residual_spectrum.py
"""
__project_origin__ = "added"

import glob
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for p in (REPO, HERE, os.path.join(REPO, 'scripts', 'gantry', 'closed-loop-controller'),
          os.path.join(REPO, 'scripts', 'gantry')):
    if p not in sys.path:
        sys.path.insert(0, p)

from telica_loader import load_telica_log_cl                               # noqa: E402
# ONE peak finder for both datasets. It takes fs, nperseg and the band as arguments precisely so the
# 4 kHz simulation arm and this 20 kHz one cannot drift apart.
from cl_residual_spectrum import modes_of                                  # noqa: E402

ROOT = os.path.join(REPO, 'kamtin-data', 'Data Telica', '06 40 mm XL 80 mm YL')
AXES = ['LX1', 'LX2', 'LY']
NPERSEG = 4096                 # 4.88 Hz at 20 kHz; the logs are short (~0.5-1 s after trim)
FMIN, FMAX = 10.0, 8000.0      # 10 Hz to below the 10 kHz Nyquist
N_MODES = 6
OUT = os.path.join(HERE, 'runs', 'telica_residual_spectrum.json')
t0 = time.time()


def op_of(path):
    """(xpos, ypos) in mm from the folder name, and the split it sits in."""
    m = re.search(r'xpos_(-?\d+)_ypos(-?\d+)', path.replace('\\', '/'))
    sp = 'train' if '/train/' in path.replace('\\', '/') else \
         'validation' if '/validation/' in path.replace('\\', '/') else 'test'
    return (int(m.group(1)), int(m.group(2)), sp) if m else (None, None, sp)


def main():
    logs = sorted(glob.glob(os.path.join(ROOT, '*', 'xpos_*_ypos*', 'iter0.log')))
    print('=' * 104)
    print('TELICA TIER 1: do the logged signals show lightly damped modes?')
    print('=' * 104)
    print('root %s' % ROOT)
    print('%d iter0 logs found (pure feedback, MF30 == MF230)' % len(logs))
    if not logs:
        print('NO LOGS FOUND. Check the path against docs/kamtin-telica-schema.md.')
        return
    print('nperseg %d, band [%.0f, %.0f] Hz' % (NPERSEG, FMIN, FMAX))
    print('-' * 104, flush=True)

    res, rows = {}, []
    for path in logs:
        xp, yp, split = op_of(path)
        d = load_telica_log_cl(path)
        fs = float(d['fs'])
        e = (np.asarray(d['r']) - np.asarray(d['q1'])).astype(np.float64)     # tracking error [m]
        i_fb = np.asarray(d['i_fb']).astype(np.float64)                        # feedback current [A]
        rec = dict(xpos=xp, ypos=yp, split=split, n=int(e.shape[0]), fs=fs,
                   rms_e=[], rms_ifb=[], modes_e=[], modes_ifb=[])
        for c in range(3):
            _, _, _, me = modes_of(e[:, c], fs, n=N_MODES, nperseg=NPERSEG, fmin=FMIN, fmax=FMAX)
            _, _, _, mi = modes_of(i_fb[:, c], fs, n=N_MODES, nperseg=NPERSEG, fmin=FMIN, fmax=FMAX)
            rec['rms_e'].append(float(np.sqrt(np.mean(e[:, c] ** 2))))
            rec['rms_ifb'].append(float(np.sqrt(np.mean(i_fb[:, c] ** 2))))
            rec['modes_e'].append(me)
            rec['modes_ifb'].append(mi)
        res[path.replace(REPO, '').replace('\\', '/')] = rec
        print('x %+5d y %+5d  %-10s N %6d   rms(e) [m] %s'
              % (xp, yp, split, rec['n'],
                 '  '.join('%s %.2e' % (AXES[c], rec['rms_e'][c]) for c in range(3))))
        for c in range(3):
            me, mi = rec['modes_e'][c], rec['modes_ifb'][c]
            fe = {round(m['f_hz']) for m in me if m['over_floor_db'] > 10}
            fi = {round(m['f_hz']) for m in mi if m['over_floor_db'] > 10}
            both = sorted(f for f in fe if any(abs(f - g) < 15 for g in fi))
            print('      %-3s  e: %-42s  i_fb: %-42s  BOTH: %s'
                  % (AXES[c],
                     ' '.join('%.0f(%+.0fdB)' % (m['f_hz'], m['over_floor_db'])
                              for m in me if m['over_floor_db'] > 10) or '-',
                     ' '.join('%.0f(%+.0fdB)' % (m['f_hz'], m['over_floor_db'])
                              for m in mi if m['over_floor_db'] > 10) or '-',
                     ' '.join('%d' % f for f in both) or 'none'))
            for f in both:
                rows.append((AXES[c], xp, yp, f))
        sys.stdout.flush()

    print('\n' + '=' * 104)
    print('READING: peaks present in BOTH the tracking error and the feedback current')
    print('=' * 104)
    if not rows:
        print('NONE. On this evidence the real machine shows no lightly damped resonance that moves')
        print('both signals, in [%.0f, %.0f] Hz, at any operating point. Consequences, and they are' % (FMIN, FMAX))
        print('the same ones the simulation script spells out:')
        print('  * nx_aug = 2 has no data-side justification on the real system')
        print('  * A_aa has nothing to be initialised FROM, so the initialisation proposal does not')
        print('    apply to Telica as stated')
        print('  * the missing behaviour is then more likely STATIC nonlinearity or slow drift, i.e.')
        print('    a static augmentation plus a friction model rather than extra dynamic states')
        print('  * before concluding that, note this is Tier 1: the controller FRF has NOT been')
        print('    divided out, so a plant mode sitting under a controller notch could be hidden')
    else:
        arr = {}
        for ax, xp, yp, f in rows:
            arr.setdefault(ax, []).append((xp, yp, f))
        for ax in AXES:
            v = arr.get(ax, [])
            if not v:
                print('  %-3s none' % ax)
                continue
            fv = np.array([q[2] for q in v], dtype=float)
            # cluster crudely: anything within 20 Hz is the same candidate mode
            clusters, used = [], np.zeros(len(fv), bool)
            for i in np.argsort(fv):
                if used[i]:
                    continue
                sel = (np.abs(fv - fv[i]) < 20) & (~used)
                used |= sel
                clusters.append((float(np.median(fv[sel])), int(sel.sum())))
            print('  %-3s %d candidate mode(s): %s   (out of %d operating points)'
                  % (ax, len(clusters),
                     ', '.join('%.0f Hz in %d/%d ops' % (c, n, len(logs)) for c, n in clusters),
                     len(logs)))
            for c, n in clusters:
                sub = [(xp, yp) for xp, yp, f in v if abs(f - c) < 20]
                if len(sub) >= 4:
                    X = np.array([q[0] for q in sub], float)
                    Y = np.array([q[1] for q in sub], float)
                    F = np.array([f for xp, yp, f in v if abs(f - c) < 20], float)
                    if F.std() > 0:
                        sx = np.polyfit(X, F, 1)[0]
                        sy = np.polyfit(Y, F, 1)[0]
                        print('        %.0f Hz: df/dx %+0.3f Hz/mm, df/dy %+0.3f Hz/mm, spread %.1f Hz'
                              % (c, sx, sy, F.max() - F.min()))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), indent=2)
    print('\nwrote %s   [%.0fs]' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
