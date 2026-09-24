"""What dynamics is the baseline missing, read from DATA rather than from the true model?

THE QUESTION THIS EXISTS TO ANSWER. Everything the augmentation's initialisation needs -- how many
augmented states, at what frequency and damping, at what magnitude, and whether any of it depends
on Y -- is a property of the residual

    r_k = y_k - y_baseline,k                (both in metres, same closed loop, same harness)

and of nothing else. `r` uses only the logged `u`, the logged `y` and the baseline model, so it
carries no oracle information and the same script runs on real data. That matters because
`rho(A_aa) = 0.976 at 159 Hz`, the number the initialisation proposal is built around, currently
comes from the PLANTED model, which was regressed onto `x_aug`. It cannot ship. This measures the
data-side replacement.

WHY IT IS RUN ON SIMULATION FIRST, and this is the point of the exercise. The simulated truth has a
known absorber: `plant.FA = 150 Hz`, `plant.ZETA_A = 0.05` (`msd-offset/plant.py:37-38`). So the
method can be VALIDATED before it is trusted: if the residual spectrum does not recover a lightly
damped mode near there, the procedure is wrong and nothing it says about real data means anything.
Expect the recovered frequency to sit somewhat ABOVE 150 Hz: the absorber is coupled to the head
through the mass matrix, so the mode of the coupled system is not the isolated `sqrt(KA/MA)`, and
the planted model's own `A_aa` reads 159.4 Hz.

WHAT COMES OFF IT, and what each decides:
  * number of lightly damped modes above the floor   ->  nx_aug (two states per mode). Today's
                                                          nx_aug = 2 was chosen because the
                                                          SIMULATED truth has one absorber, which
                                                          is not an argument that survives to real
                                                          data (Kessels et al. 2025 needed
                                                          n_ext = 14 on an ASMPT wire bonder).
  * each mode's (f, zeta)                            ->  A_aa init, as rho exp(+-j theta) with
                                                          rho = exp(-zeta wn Ts), theta = 2 pi f Ts
  * residual RMS                                     ->  the magnitude the readout must reach, i.e.
                                                          the lr and epoch budget
  * the high-frequency floor                         ->  a DATA-derived acceptance threshold, which
                                                          the current 2.81e-08 is not (it comes
                                                          from plant.deriv8, a model)
  * the same, per Y operating point                  ->  whether the augmented dynamics must be
                                                          Y-SCHEDULED. This one sits directly on
                                                          the thesis's LPV contribution and is free
                                                          from the same measurement.

THE SIMULATION NUMBER DOES NOT TRANSFER, and this is the whole reason the script is discovery-shaped
rather than a confirmation of 159 Hz. The 150 Hz absorber is something WE injected into the
simulated truth; on the real machine it is not known to exist (user, 2026-08-19: "on the real data
we don't have the 157 resonance as far as we know"). So on simulation this measures a KNOWN answer
and validates the method; on Telica it asks an OPEN question and every outcome below is admissible,
including "no lightly damped mode anywhere", which would say that extra DYNAMIC states are the wrong
augmentation for the real system and that the missing behaviour is static nonlinearity (friction,
hysteresis) or very slow drift instead. Nothing in this script should be read as expecting a peak.

THE LOW-FREQUENCY BLIND SPOT, which matters far more on real data than on simulation. `r` is the
closed-loop tracking error, so below the ~100 Hz crossover the loop has already suppressed it by
`So`. On simulation that is harmless because the thing we are looking for sits at 150 Hz, above
crossover. On the real machine the likeliest missing dynamics -- friction, stick-slip, cable forces,
thermal drift -- are LOW frequency, exactly where this spectrum is blinded. Two ways out, neither
implemented here: divide by the sensitivity `So = (I + P C)^-1`, which is computable since `Cfb` is
verified bit-exact and the baseline `P` is known; or look at the FEEDBACK EFFORT `u_fb` instead of
the tracking error, since what the controller had to inject IS the loop's own estimate of the
mismatch and it is not suppressed the same way. Decide which before running the Telica arm; reading
the raw low-frequency end of this spectrum on real data would be a mistake.

TWO THINGS TO READ CAREFULLY RATHER THAN AT FACE VALUE.
  1. `r` is the CLOSED-LOOP residual: it is exactly the `e` signal inside `closed_loop_run`
     (`cl_headroom.py:80`). Below the ~100 Hz crossover the loop suppresses it, so the low-frequency
     part of this spectrum understates the open-loop model error. Above crossover `So ~ 1` and the
     residual is essentially the open-loop error, which is where the absorber lives, so the mode of
     interest is unaffected. No de-suppression is applied here; the spectrum is reported raw and
     this caveat travels with it.
  2. On SIMULATION the baseline carries the true parameters minus the absorber, so `r` is purely
     missing dynamics. On REAL data the baseline parameters are estimates, so
     `r = missing dynamics + parameter mismatch + noise`. Parameter error shows up as broadband or
     low-frequency mismatch rather than a sharp lightly damped peak, so an isolated resonance is
     still attributable, but the separation is not free.

THE TELICA ARM IS NOT IMPLEMENTED HERE. `kamtin-data/Data Telica/` is blocked by policy and this
session must not read it. `residual_for()` is written so that supplying (u, y, x0, ctrl) from any
loader is the only change needed; what the real arm additionally needs is stated in its docstring.

Usage:
  python -u cl_residual_spectrum.py              # the four validation records
  CL_RS_FILES=all python -u cl_residual_spectrum.py
"""
__project_origin__ = "added"

import dataclasses
import json
import os
import sys
import time
import zlib

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
for p in (REPO, GANTRY, HERE, os.path.join(GANTRY, 'drift-demo'),
          os.path.join(GANTRY, 'msd-offset')):
    if p not in sys.path:
        sys.path.insert(0, p)

from demo_common import CFG                                                # noqa: E402
from gantry_dynamic.data import load_traj, load_mat_aug, TRAIN_FILES, VAL_FILES   # noqa: E402
from cl_controller import y_op_for                                         # noqa: E402
from loss_variants import controller_ss                                    # noqa: E402
from cl_headroom import closed_loop_run                                    # noqa: E402
import plant as PL                                                         # noqa: E402

K0 = 17                       # the D-087 interior seed, as everywhere else in this folder
NPERSEG = 8192                # 0.488 Hz resolution at 4 kHz; the 150 Hz mode's -3 dB width is ~15 Hz
FMIN, FMAX = 5.0, 1900.0      # ignore DC/drift and the last decade before Nyquist
N_MODES = 5
CHANNELS = ['X1', 'X2', 'Y']  # the three measured stage positions
# ---- the noise configuration, resolved ONCE at module level so it can be written into the
# artefact. The 2026-08-21 handoff's step 5 item 1: `cl_residual_spectrum_noisy.json` records NO
# sigma field, so its provenance rested entirely on a sentence in a doc. An artefact that cannot
# say what noise produced it is not evidence. -------------------------------------------------
_NS = os.environ.get('CL_RS_NOISE_SIGMA', '')
NOISE_SIGMA = [float(v) for v in _NS.split(',')] if _NS else None
if NOISE_SIGMA is not None:
    assert len(NOISE_SIGMA) == 3, 'CL_RS_NOISE_SIGMA needs three per-channel values [m]'
# CL_RS_SIGMA_SCALE multiplies the base sigma. It exists for the BREAKDOWN SWEEP (step 5 item 3):
# the 38.8 dB headroom quoted in the handoff is inferred from two points, and the SNR at which the
# band recipe stops working is a stated thesis deliverable that no run has ever produced.
SIGMA_SCALE = float(os.environ.get('CL_RS_SIGMA_SCALE', '1') or 1)
if NOISE_SIGMA is not None:
    NOISE_SIGMA = [s * SIGMA_SCALE for s in NOISE_SIGMA]
# CL_RS_NOISE_CONSISTENT=1 (step 5 item 2): perturb the record the way a real machine would.
# The y-only path is PHYSICALLY WRONG for this script in the same way C5 was wrong for the training
# drive: `closed_loop_run` below is driven by the recorded `u`, and on a real machine the recorded
# `u` ALREADY contains the controller's reaction to the noisy measurement. Adding v to y while
# leaving u untouched therefore models a machine whose controller cannot see its own sensor noise.
# THEORY: Sugie & Maruta 2020 (stabilized PEM / simplified dual Youla), Eq. (8) and Sec. 3 -- under
# a noise realisation v,   y_data -> y_data + v,   u_data -> u_data - C_fb(v),
# and the model's driving signal is unchanged by linearity of the controller. This is the same
# transformation cl_train.py applies under CL_NOISE_CONSISTENT, applied to the same records, so the
# spectrum and the training drive finally agree about what "noisy data" means.
NOISE_CONSISTENT = bool(os.environ.get('CL_RS_NOISE_CONSISTENT'))

OUT = os.path.join(HERE, 'runs', 'cl_residual_spectrum.json')
# A noisy run writes its OWN artefact: the clean one is the SOURCE of the D-150 initialisation band
# and must never be overwritten (never-overwrite-datasets rule). The name now carries the noise
# MODE and the scale, so a sweep cannot silently overwrite its own earlier points and the existing
# `cl_residual_spectrum_noisy.json` (y-only, sigma unrecorded) is left intact as the historical
# record rather than being replaced under the same name.
if NOISE_SIGMA is not None:
    OUT = os.path.join(HERE, 'runs', 'cl_residual_spectrum_noise_%s_x%g.json'
                       % ('consistent' if NOISE_CONSISTENT else 'yonly', SIGMA_SCALE))
# CL_RS_OUT: send this run's artefact somewhere else entirely.
#
# ADDED 2026-08-22 AFTER THIS EXACT MISTAKE WAS MADE. A verification run of the CLEAN path
# (no CL_RS_NOISE_SIGMA) writes to `runs/cl_residual_spectrum.json`, which is the SOURCE OF TRUTH
# for the AUG_LRU initialisation band: `lru_band_from_artifact` reads it at every gated build. The
# default record set is VAL_FILES (4 records), while the artefact in the repo was generated with
# CL_RS_FILES=all (18 records), so an innocent-looking regression run silently replaced an 18-record
# band [149.90234375, 164.06250] Hz with a 4-record band [153.80859, 164.06250] Hz -- and every
# subsequent AUG_LRU arm would have been initialised from a DIFFERENT band than the arms it was
# being compared against, with nothing in any log to say so.
#
# The artefact was restored from a backup taken before the run and re-verified byte-identical. This
# gate exists so the next verification does not need luck. Precedent: PROBE_OUT in
# transient-investigation/probe_d072_matrix.py, added for the same reason.
OUT = os.environ.get('CL_RS_OUT') or OUT
t0 = time.time()


def cfb_response(ctrl, sig):
    """C_fb(sig) in PHYSICAL units [N], from xc = 0, stepped at the record's rate.

    Same state-space and same step order as `cl_headroom.closed_loop_run` (`u_fb = Cc xc + Dc e`,
    then `xc <- Ac xc + Bc e`), so the correction this returns is exactly the feedback contribution
    that harness would have produced for the same input. Written out rather than reusing
    `closed_loop_run` because that function also integrates the plant, and here only the controller
    is wanted.
    """
    Ac, Bc, Cc, Dc = ctrl
    xc = np.zeros(Ac.shape[0])
    out = np.empty_like(np.asarray(sig, dtype=np.float64))
    for k in range(len(out)):
        e = sig[k]
        out[k] = Cc @ xc + Dc @ e
        xc = Ac @ xc + Bc @ e
    return out


def half_power(f, P, i):
    """(f_n, zeta) for the peak at index `i` of the one-sided POWER spectrum `P`.

    THEORY: half-power (-3 dB) bandwidth for a lightly damped SDOF resonance,
    `zeta = (f2 - f1) / (2 f_n)` with `f1, f2` the frequencies at which the POWER is half its peak
    (Ewins, Modal Testing). Valid for isolated peaks and `zeta` below roughly 0.1, which is why the
    returned value is reported next to the peak's prominence rather than on its own: a shoulder on a
    neighbouring mode gives a wide apparent bandwidth and an overestimated zeta.

    Returns (f_n, zeta, ok) with ok False when the half-power points fall outside the band, in which
    case zeta is not trustworthy.
    """
    target = P[i] / 2.0
    lo = i
    while lo > 0 and P[lo] > target:
        lo -= 1
    hi = i
    while hi < len(P) - 1 and P[hi] > target:
        hi += 1
    ok = lo > 0 and hi < len(P) - 1
    if not ok:
        return float(f[i]), float('nan'), False
    # linear interpolation in f on the power axis, on each flank
    f1 = f[lo] + (target - P[lo]) * (f[lo + 1] - f[lo]) / max(P[lo + 1] - P[lo], 1e-300)
    f2 = f[hi - 1] + (target - P[hi - 1]) * (f[hi] - f[hi - 1]) / min(P[hi] - P[hi - 1], -1e-300)
    fn = float(f[i])
    return fn, float(abs(f2 - f1) / (2.0 * fn)), True


def modes_of(r, fs, n=N_MODES, nperseg=None, fmin=None, fmax=None):
    """The n most prominent lightly damped peaks of a residual channel.

    Welch rather than a raw periodogram because the residual is a long stationary-ish record and the
    question is where its POWER sits, not its exact phase; `nperseg` is stated at module level with
    its resolution against the expected bandwidth so the choice is checkable rather than a default.
    """
    # nperseg/fmin/fmax are arguments so the SAME peak finder serves the 4 kHz simulation and the
    # 20 kHz Telica logs; the module-level values are the simulation defaults, not a global.
    nperseg = NPERSEG if nperseg is None else nperseg
    fmin = FMIN if fmin is None else fmin
    fmax = FMAX if fmax is None else fmax
    f, P = signal.welch(r, fs=fs, nperseg=min(nperseg, len(r)), noverlap=None,
                        detrend='constant', scaling='density')
    band = (f >= fmin) & (f <= fmax)
    fb, Pb = f[band], P[band]
    # A floor estimate that a resonance cannot bias: the median of the band, which for a spectrum
    # dominated by a few narrow peaks is the broadband level between them.
    floor = float(np.median(Pb))
    pk, props = signal.find_peaks(np.log10(np.maximum(Pb, 1e-300)), prominence=0.3, distance=8)
    if len(pk) == 0:
        return f, P, floor, []
    order = np.argsort(props['prominences'])[::-1][:n]
    out = []
    for j in sorted(pk[order]):
        fn, zeta, ok = half_power(fb, Pb, j)
        out.append(dict(f_hz=fn, zeta=zeta, zeta_ok=bool(ok),
                        power=float(Pb[j]), over_floor_db=float(10 * np.log10(Pb[j] / max(floor, 1e-300))),
                        prominence_decades=float(props['prominences'][list(pk).index(j)])))
    return f, P, floor, out


def residual_for(fname, cfg):
    """r = y - y_baseline for one SIMULATED record, in metres, plus its Y operating point.

    THE SEAM FOR REAL DATA. Everything below the `x0` line is dataset-independent: given `u` [N],
    `y` [m], an initial physical state and the controller for that record, `closed_loop_run` with
    `PL.deriv6` produces the baseline output and the residual follows. A Telica arm therefore needs
    exactly three things and no changes here:
      * `u`, `y` in the model's own frame and units (P-transform, `docs/kamtin-telica-schema.md`)
      * `x0`, which on the real system is NOT available as a true state: positions are measured and
        velocities come from numerical differentiation, which is what Kessels et al. do. Over a full
        record the closed loop makes the model track the recorded output, so an `x0` error decays
        into the first transient rather than persisting -- but the first ~K0 samples should be
        dropped, as they are here.
      * the baseline PARAMETERS, which on real data are ESTIMATES from the recovery work, so the
        residual then also contains parameter mismatch (see the module docstring).
    """
    sd = load_traj(fname, cfg)
    _, _, x_log, _ = load_mat_aug(fname, cfg)
    ctrl = controller_ss(y_op_for(fname), cfg.ts_new)
    u = np.asarray(sd.u, dtype=np.float64)[K0:]
    y = np.asarray(sd.y, dtype=np.float64)[K0:]
    # CL_RS_NOISE_SIGMA="sx1,sx2,sy" [m]: the D-150 noise gate, step 2 (handoff 2026-08-19
    # section 10 / HYPOTHESES section 4e). White measurement noise on y BEFORE the residual is
    # formed, so it enters both the loop and the spectrum, exactly as it would on hardware. The
    # sigmas are data-derived at launch (Telica error-PSD floor bound); nothing is defaulted here.
    if NOISE_SIGMA is not None:
        # PER-RECORD RNG, and this is a correction. The previous line was
        #     np.random.default_rng(150).normal(...)
        # constructed INSIDE this function, so every record received the byte-identical noise
        # sequence. Across an 18-record set that is not white measurement noise, it is one
        # realisation repeated 18 times, and the band recipe aggregates over records, so a fluke
        # of that single draw would have been reproduced in every one of them rather than averaged
        # out. Seeding from the record NAME keeps the run reproducible while making the draws
        # independent across records.
        # zlib.crc32 rather than the builtin hash(): hash() of a str is SALTED per process
        # (PYTHONHASHSEED), so it would give a different noise realisation on every invocation and
        # on every machine, which is exactly the reproducibility this seeding exists to preserve.
        _seed = 150 + zlib.crc32(fname.encode('utf-8')) % 100000
        _rng = np.random.default_rng(_seed)
        v = _rng.normal(0.0, NOISE_SIGMA, y.shape)
        y = y + v
        if NOISE_CONSISTENT:
            # u_data -> u_data - C_fb(v). See the module-level note: without this the recorded u is
            # that of a machine whose controller could not see its own sensor noise, which is not a
            # machine that exists.
            u = u - cfb_response(ctrl, v)
    x0 = np.asarray(x_log[K0], dtype=np.float64)
    y_b = closed_loop_run(PL.deriv6, x0, u, y, ctrl, 6)
    return y - y_b, y, y_b, y_op_for(fname)


def main():
    cfg = dataclasses.replace(CFG, seed=0)
    fs_hz = 1.0 / cfg.ts_new
    files = (list(TRAIN_FILES) + list(VAL_FILES)
             if (os.environ.get('CL_RS_FILES') or '').lower() == 'all' else list(VAL_FILES))

    print('=' * 104)
    print('RESIDUAL SPECTRUM: what is the baseline missing, from u, y and the baseline alone')
    print('=' * 104)
    print('fs %.0f Hz, nperseg %d (df %.3f Hz), band [%.0f, %.0f] Hz, %d records'
          % (fs_hz, NPERSEG, fs_hz / NPERSEG, FMIN, FMAX, len(files)))
    print('VALIDATION TARGET (simulation only): plant.FA = %.1f Hz, plant.ZETA_A = %.3f; the planted '
          'model reads its own A_aa at 159.4 Hz / 0.098' % (PL.FA, PL.ZETA_A))
    print('-' * 104, flush=True)

    res, summary = {}, {}
    for fname in files:
        t_r = time.time()
        r, y, y_b, Y_op = residual_for(fname, cfg)
        rec = dict(Y_op=float(Y_op), rms_y=[], rms_r=[], floor=[], modes=[])
        for c in range(3):
            f, P, floor, modes = modes_of(r[:, c], fs_hz)
            rec['rms_y'].append(float(np.sqrt(np.mean(y[:, c] ** 2))))
            rec['rms_r'].append(float(np.sqrt(np.mean(r[:, c] ** 2))))
            rec['floor'].append(floor)
            rec['modes'].append(modes)
        res[fname] = rec
        print('%-26s Y_op %+0.3f   rms(r) [m] %s   [%.0fs]'
              % (fname[:-4] if fname.endswith('.mat') else fname, Y_op,
                 '  '.join('%s %.3e' % (CHANNELS[c], rec['rms_r'][c]) for c in range(3)),
                 time.time() - t_r))
        for c in range(3):
            ms = rec['modes'][c]
            if not ms:
                print('      %-3s no peaks above the prominence threshold' % CHANNELS[c])
                continue
            print('      %-3s %s' % (CHANNELS[c], '   '.join(
                '%.1f Hz (zeta %s, %+.1f dB)'
                % (m['f_hz'], ('%.3f' % m['zeta']) if m['zeta_ok'] else 'n/a',
                   m['over_floor_db'])
                for m in ms)))
        sys.stdout.flush()

    # ---- the two readings the design actually needs -------------------------------------------
    print('\n' + '=' * 104)
    print('READING 1: is there a dominant lightly damped mode, and where?  (nx_aug and A_aa init)')
    print('=' * 104)
    # ONE peak per record-channel: the most prominent. Taking every peak over the floor mixes the
    # resonance with excitation content (the APRBS records carry 40-100 Hz peaks that the standstill
    # records do not) and produces a "median mode" spanning a decade, which is not a mode.
    allm = []
    for fname in files:
        rec = res[fname]
        for c in range(3):
            ms = [m for m in rec['modes'][c] if m['zeta_ok']]
            if ms:
                allm.append((fname, CHANNELS[c], max(ms, key=lambda m: m['over_floor_db'])))
    strong = [t for t in allm if t[2]['over_floor_db'] > 10.0]
    print('%d peaks over 10 dB above their band floor, across %d records x 3 channels'
          % (len(strong), len(files)))
    if not strong:
        # A legitimate and important outcome, not a failure of the script. Spelled out because the
        # temptation on a real dataset is to lower the threshold until something appears.
        print('  NO dominant lightly damped mode in the residual.')
        print('  On SIMULATION that would mean the procedure is broken, since a 150 Hz absorber is')
        print('  known to be there. On REAL data it is an admissible answer and a consequential one:')
        print('    * nx_aug has no data-side justification and the 2 augmented states are unmotivated')
        print('    * A_aa has nothing to be initialised FROM, so the whole initialisation proposal')
        print('      does not apply as stated')
        print('    * the missing behaviour is then most likely STATIC nonlinearity (friction,')
        print('      hysteresis) or very slow drift, i.e. a static augmentation plus a friction')
        print('      model, not extra dynamic states')
        print('    * and before concluding any of that, re-read the low-frequency blind spot in the')
        print('      module docstring: below crossover this spectrum cannot see the model error at all')
    if strong:
        fz = np.array([[t[2]['f_hz'], t[2]['zeta']] for t in strong if t[2]['zeta_ok']])
        if len(fz):
            print('  f  [Hz]  median %.2f   range [%.2f, %.2f]' %
                  (np.median(fz[:, 0]), fz[:, 0].min(), fz[:, 0].max()))
            print('  zeta     median %.4f   range [%.4f, %.4f]' %
                  (np.median(fz[:, 1]), fz[:, 1].min(), fz[:, 1].max()))
            ts = cfg.ts_new
            fm, zm = float(np.median(fz[:, 0])), float(np.median(fz[:, 1]))
            wn = 2 * np.pi * fm / max(np.sqrt(max(1 - zm ** 2, 1e-12)), 1e-12)
            rho = float(np.exp(-zm * wn * ts))
            print('  ->  A_aa init from DATA: rho %.4f at %.2f Hz   (planted, oracle: 0.976 at 159.4 Hz)'
                  % (rho, fm))
            summary['A_aa_from_data'] = dict(rho=rho, f_hz=fm, zeta=zm)
            print('  ->  VALIDATION against the known absorber (%.1f Hz, zeta %.3f): %s'
                  % (PL.FA, PL.ZETA_A,
                     'RECOVERED' if abs(fm - PL.FA) / PL.FA < 0.15 else
                     'NOT recovered -- the procedure is wrong and its real-data reading means nothing'))

    print('\n' + '=' * 104)
    print('READING 2: does the mode move with Y?  (must the augmented dynamics be Y-SCHEDULED?)')
    print('=' * 104)
    rows = []
    for fname in files:
        rec = res[fname]
        best = None
        for c in range(3):
            for m in rec['modes'][c]:
                if m['zeta_ok'] and m['over_floor_db'] > 10.0 and (
                        best is None or m['over_floor_db'] > best['over_floor_db']):
                    best = m
        if best:
            rows.append((rec['Y_op'], best['f_hz'], best['zeta']))
    rows.sort()
    for Yv, fv, zv in rows:
        print('  Y_op %+0.3f   f %.2f Hz   zeta %.4f' % (Yv, fv, zv))
    if len(rows) >= 3:
        Y = np.array([q[0] for q in rows])
        F = np.array([q[1] for q in rows])
        sl, ic = np.polyfit(Y, F, 1)
        pred = sl * Y + ic
        r2 = 1 - np.sum((F - pred) ** 2) / max(np.sum((F - F.mean()) ** 2), 1e-30)
        print('\n  linear fit f(Y) = %+.3f Y %+.3f Hz, R2 %.3f, spread %.2f Hz over Y in [%.2f, %.2f]'
              % (sl, ic, r2, F.max() - F.min(), Y.min(), Y.max()))
        print('  ->  %s' % ('the mode MOVES with Y: a fixed A_aa is structurally wrong and the '
                            'augmented dynamics need Y-scheduling'
                            if r2 > 0.5 and (F.max() - F.min()) > 2.0 else
                            'no usable Y dependence at this resolution: an LTI A_aa is defensible '
                            'and the LPV-ness stays in the baseline'))
        summary['Y_dependence'] = dict(slope_hz_per_m=float(sl), intercept_hz=float(ic), r2=float(r2),
                                   spread_hz=float(F.max() - F.min()))

    # ---- READING 3: does the BAND RECIPE still work on this artefact? --------------------------
    # Added 2026-08-22 (handoff step 5). The artefact is the input to
    # `gantry_dynamic.model.lru_band_from_artifact`, and until now nothing in it said whether that
    # function would succeed, so the breakdown point had to be found by running the model builder
    # and catching an exception. This block reproduces the recipe EXACTLY -- dominant peak per
    # record-channel among those with `zeta_ok` and `over_floor_db > 10`, then [min, max] over all
    # of them -- and records the answer, so a sigma sweep can read the breakdown straight off the
    # artefacts. Keep it in step with `lru_band_from_artifact`; it is a deliberate duplication so
    # that this script does not import the model builder.
    print('\n' + '=' * 104)
    print('READING 3: would the AUG_LRU band recipe succeed on this artefact?')
    print('=' * 104)
    # HEURISTIC: over_floor_db > 10 -- the strong-peak threshold, inherited verbatim from READING 1
    # and from `lru_band_from_artifact`, where it is also labelled HEURISTIC. A chosen threshold
    # with no literature source; it is repeated here rather than imported so that this script does
    # not depend on the model builder, and the two must be kept in step.
    _STRONG_DB = 10.0
    _fz = []
    for rec in res.values():
        for ch_modes in rec['modes']:
            _s = [m for m in ch_modes if m.get('zeta_ok') and m['over_floor_db'] > _STRONG_DB]
            if _s:
                _b = max(_s, key=lambda m: m['over_floor_db'])
                _fz.append((_b['f_hz'], _b['zeta'], _b['over_floor_db']))
    band = dict(n_dominant_peaks=len(_fz), would_raise=bool(not _fz))
    if _fz:
        _f = np.array([q[0] for q in _fz])
        _z = np.array([q[1] for q in _fz])
        _d = np.array([q[2] for q in _fz])
        _wn = 2 * np.pi * _f / np.sqrt(np.maximum(1 - _z ** 2, 1e-12))
        _rho = np.exp(-_z * _wn * cfg.ts_new)
        band.update(f_band_hz=[float(_f.min()), float(_f.max())],
                    rho_band=[float(_rho.min()), float(_rho.max())],
                    over_floor_db=dict(min=float(_d.min()), median=float(np.median(_d)),
                                       max=float(_d.max())),
                    margin_db_over_threshold=float(_d.min() - _STRONG_DB))
        print('  %d dominant peaks -> band f [%.5f, %.5f] Hz, rho [%.6f, %.6f]'
              % (len(_fz), band['f_band_hz'][0], band['f_band_hz'][1],
                 band['rho_band'][0], band['rho_band'][1]))
        print('  over_floor_db  min %.1f  median %.1f  max %.1f   -> %.1f dB of margin above the '
              '10 dB threshold' % (_d.min(), np.median(_d), _d.max(),
                                   band['margin_db_over_threshold']))
        print('  -> lru_band_from_artifact WOULD SUCCEED on this artefact')
    else:
        print('  NO peak clears over_floor_db > 10 on any record-channel.')
        print('  -> lru_band_from_artifact WOULD RAISE, and correctly so: the band would then have')
        print('     to be supplied from loop bandwidth and sample rate via AUG_LRU_BAND/AUG_LRU_RHO.')
        print('     THIS IS THE TELICA CASE, reached here by adding noise to simulation data.')
    summary['band_recipe'] = band

    # The noise provenance, written INTO the artefact. Step 5 item 1: the previous noisy artefact
    # carried no sigma at all, so what produced it rested on a sentence in a doc rather than on the
    # file. Every field here is what a reader needs to reproduce this run exactly.
    noise = dict(sigma=NOISE_SIGMA, sigma_scale=SIGMA_SCALE,
                 sigma_base=([s / SIGMA_SCALE for s in NOISE_SIGMA]
                             if NOISE_SIGMA is not None else None),
                 consistent=NOISE_CONSISTENT,
                 rng='per-record, seed = 150 + crc32(record name) % 100000',
                 applied_to=('y and u (u -= C_fb(v), Sugie & Maruta 2020 Eq. 8)'
                             if NOISE_CONSISTENT else
                             'y only (NON-PHYSICAL for a u-driven baseline sim; see the module note)')
                 if NOISE_SIGMA is not None else 'none, clean run')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(noise=noise, records=res, summary=summary,
                   config=dict(fs_hz=fs_hz, nperseg=NPERSEG, fmin=FMIN, fmax=FMAX,
                               k0=K0, n_modes=N_MODES, files=files, ts=cfg.ts_new)),
              open(OUT, 'w'), indent=2)
    print('\nnoise: %s' % noise['applied_to'])
    if NOISE_SIGMA is not None:
        print('       sigma %s m  (base x %g)' % (NOISE_SIGMA, SIGMA_SCALE))
    print('wrote %s   [%.0fs]' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
