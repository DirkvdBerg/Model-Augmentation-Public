"""
telica_controller.py
--------------------
Steppable Telica feedback controller for closed-loop model evaluation (D-074).

Source: kamtin-data/dFeedbackControllersTelica_ba.mat, exported from the
supervisor-provided dFeedbackControllersTelica.mat (6x6 diagonal zpk,
Ts = 5e-5 s). Axis order confirmed by the supervisor:
    row 1 LX1, row 2 LX2, row 3 LY, row 4 RX1, row 5 RX2, row 6 RY.
The BHL data uses rows LX1, LX2, LY.

Convention (verified in diag_controller_fingerprint.py, D-073):
    input  : tracking error [m]      (positive = position behind reference)
    output : feedback current [A]
    rate   : 20 kHz, one update per log sample (log is 20 kHz native, D-073)

Force conversion (current -> stage force) is the caller's job via the motor
force constants Kt (telica_loader._A_TO_N).

Self-test:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/telica_controller.py
verifies (1) the per-sample step() against scipy.signal.lfilter and
(2) replays iter0: controller fed with the MEASURED error must reproduce the
logged feedback current MF230 in shape (corr ~0.96-0.97). Known amplitude
offsets remain: time-domain LS scale X1 1.16, X2 1.33, Y 3.55 (whole-band,
dominated by low frequencies); in the coherent band the offsets are smaller
(0.74/1.08/1.23, D-073). Consistent with an unmodeled decoupling transform
(rail cross-coupling); accepted for closed-loop validation.
"""

__project_origin__ = "added"

import os
import numpy as np
from scipy.io import loadmat

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
_BA_PATH = os.path.join(_ROOT, 'kamtin-data', 'dFeedbackControllersTelica_ba.mat')

_BHL_AXES = ('LX1', 'LX2', 'LY')   # controller rows used for BHL [X1, X2, Y]


def _load_ba(ba_path=_BA_PATH):
    """Return {axis_name: (b, a)} for all six controllers, plus Ts."""
    m = loadmat(ba_path, squeeze_me=True)
    num_mat = np.atleast_2d(np.asarray(m['num_mat'], dtype=np.float64))
    den_mat = np.atleast_2d(np.asarray(m['den_mat'], dtype=np.float64))
    names   = [str(s).strip() for s in m['axis_names_char']]
    out = {}
    for i, name in enumerate(names):
        b = np.trim_zeros(num_mat[i], 'b')
        a = np.trim_zeros(den_mat[i], 'b')
        out[name] = (b, a)
    return out, float(m['Ts'])


class TelicaFeedbackController:
    """
    Three parallel SISO IIR controllers (LX1, LX2, LY) with per-sample step().

    THEORY: direct form II transposed recursion (Oppenheim & Schafer,
    Discrete-Time Signal Processing, sec. 6.5) -- identical arithmetic to
    scipy.signal.lfilter, verified in the self-test below.
    """

    def __init__(self, axes=_BHL_AXES, ba_path=_BA_PATH):
        ba, self.Ts = _load_ba(ba_path)
        self._filt = []
        for name in axes:
            b, a = ba[name]
            L = max(len(b), len(a))
            bp = np.zeros(L); bp[:len(b)] = b
            ap = np.zeros(L); ap[:len(a)] = a
            if abs(ap[0] - 1.0) > 1e-12:      # normalize a0 = 1
                bp, ap = bp / ap[0], ap / ap[0]
            self._filt.append([bp, ap, np.zeros(L - 1)])
        self.axes = tuple(axes)

    def reset(self):
        for f in self._filt:
            f[2][:] = 0.0

    def step(self, e_m):
        """One 20 kHz update. e_m: (3,) error [m] -> (3,) current [A]."""
        out = np.empty(3)
        for i in range(3):
            b, a, z = self._filt[i]
            x = float(e_m[i])
            y = b[0] * x + z[0]
            z[:-1] = z[1:] + b[1:-1] * x - a[1:-1] * y
            z[-1]  = b[-1] * x - a[-1] * y
            out[i] = y
        return out

    def filt(self, e_seq):
        """Vectorized whole-sequence filtering (zero initial state).
        e_seq: (T, 3) error [m] -> (T, 3) current [A]."""
        from scipy.signal import lfilter
        out = np.empty_like(e_seq, dtype=np.float64)
        for i in range(3):
            b, a, _ = self._filt[i]
            out[:, i] = lfilter(b, a, np.asarray(e_seq[:, i], dtype=np.float64))
        return out


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, _HERE)
    from telica_loader import load_telica_log_cl

    print('=' * 70)
    print('TELICA CONTROLLER SELF-TEST')
    print('=' * 70)

    # -- 1. step() vs scipy lfilter on white noise ----------------------------
    ctrl = TelicaFeedbackController()
    rng = np.random.default_rng(0)
    e = rng.standard_normal((2000, 3)) * 1e-6
    y_ref = ctrl.filt(e)
    ctrl.reset()
    y_step = np.stack([ctrl.step(e[k]) for k in range(len(e))])
    err = np.abs(y_step - y_ref).max() / max(np.abs(y_ref).max(), 1e-300)
    print(f'1. step() vs lfilter: max rel err = {err:.3e}  '
          f'{"PASS" if err < 1e-9 else "FAIL"}')

    # -- 2. iter0 replay: measured error -> logged MF230 ----------------------
    log = os.path.join(_ROOT, 'kamtin-data', 'Data Telica',
                       '06 40 mm XL 80 mm YL', 'train', 'xpos_-60_ypos-40',
                       'iter0.log')
    d = load_telica_log_cl(log)
    e_meas = (d['r'] - d['q1']).numpy()          # (T,3) [m] = M2 * 1e-6
    i_log  = d['i_fb'].numpy()                   # (T,3) [A]
    ctrl.reset()
    i_sim = ctrl.filt(e_meas)

    save_dir = os.path.join(_ROOT, 'simulations', 'gantry_subnet',
                            'diagnostics', 'controller_replay')
    os.makedirs(save_dir, exist_ok=True)
    t = np.arange(len(e_meas)) / d['fs']
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    names = ('X1', 'X2', 'Y')
    print('2. iter0 replay (expected: corr ~0.96-0.97; time-domain LS scale '
          '~1.16/1.33/3.55, see module doc):')
    for i, name in enumerate(names):
        c = float(np.corrcoef(i_sim[:, i], i_log[:, i])[0, 1])
        s = float(np.dot(i_sim[:, i], i_log[:, i]) / np.dot(i_sim[:, i], i_sim[:, i]))
        print(f'   {name}: corr = {c:.4f}   LS scale (logged/replayed) = {s:.3f}')
        axs[i].plot(t, i_log[:, i], 'k', lw=0.8, label='logged MF230 [A]')
        axs[i].plot(t, i_sim[:, i], 'C1', lw=0.8, alpha=0.8,
                    label='controller replay of measured error [A]')
        axs[i].set_ylabel(f'{name} [A]')
        axs[i].set_title(f'{name}: corr = {c:.3f}, scale = {s:.2f}', fontsize=9)
        axs[i].grid(alpha=0.3)
        if i == 0:
            axs[i].legend(fontsize=8, loc='upper right')
    axs[-1].set_xlabel('time [s]')
    fig.suptitle('iter0: real controller replay vs logged feedback current')
    fig.tight_layout()
    fp = os.path.join(save_dir, 'controller_replay_iter0.png')
    fig.savefig(fp, dpi=150)
    print(f'   figure -> {os.path.relpath(fp, _ROOT)}')
