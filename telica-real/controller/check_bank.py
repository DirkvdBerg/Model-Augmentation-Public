"""G3 (ii) (TR-009): the residual-form loop with the Telica bank, through the framework's own
`closed_loop_rollout`.

(ii-a) y_model = y_data  ->  u_cl must equal u_data EXACTLY (float32 and float64), whole record.
(ii-b) y_model = y_data + d, d = the logged error (realistic spectrum)  ->  u_cl - u_data must
       equal diag(Kt s_F) K(-d) computed independently with scipy sosfilt (+ lag): rel. error
       <= 1e-6 (float64), <= 1e-3 (float32).
Plus `ControllerBank.check_units()`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
from scipy.signal import sosfilt                                       # noqa: E402

import data_index                                                      # noqa: E402
from params import telica_params as tp                                 # noqa: E402
from controller import telica_ctrl as tc                               # noqa: E402
from controller.telica_bank import build_telica_bank, controller_sos   # noqa: E402
from model_augmentation.fit_systems.closed_loop import closed_loop_rollout  # noqa: E402
from real_data_verification.telica_loader import load_telica_log_full  # noqa: E402
from controller.telica_bank import SOURCE_TS                           # noqa: E402
import rate                                                            # noqa: E402
tr_env.check_no_leak()

SOURCE = sys.argv[1] if len(sys.argv) > 1 else 'identified'
TS = SOURCE_TS[SOURCE]


class LookupModel:
    """A 'model' whose output is a prescribed table: state = time index. Records its inputs."""

    def __init__(self, y_table_n):
        self.y = y_table_n            # (1, T, 3) normalised
        self.u_seen = []

    def output_only(self, x):
        k = int(x[0, 0].item())
        return self.y[:, k, :]

    def __call__(self, x, u):
        self.u_seen.append(u.detach().clone())
        return None, x + 1


def main():
    ok = True
    d = load_telica_log_full(data_index.path('validation', 'xpos_-135_ypos120'))
    if SOURCE == 'identified_5k':                          # TR-022: decimated record
        d = rate.decimate_record(d)
    a2n = np.asarray(tp.a_to_n())
    y = d['q1']; u = d['i_tot'] * a2n; e = d['e']
    T = len(y)
    y0, ystd = y.mean(0), y.std(0)
    u0, ustd = u.mean(0), u.std(0)
    sos, lags = controller_sos(SOURCE)
    print(f'[bank] source={SOURCE}, lags={lags}, record T={T}')
    # independent reference of the feedback for y_model = y_data + d  (e_res = -d)
    ref = np.zeros_like(u)
    for j, ax in enumerate(tc.AXES):
        r = sosfilt(sos[ax], -e[:, j]) * a2n[j]
        L = lags[ax]
        ref[:, j] = np.concatenate([np.zeros(L), r[:T - L]]) if L > 0 else r
    for dt, tol in ((torch.float64, 1e-6), (torch.float32, 1e-3)):
        bank = build_telica_bank(TS, ystd=ystd, std_u=ustd, dtype=dt, source=SOURCE)
        _, expect, rel = bank.check_units()
        print(f'[bank] {str(dt)}: check_units rel err {np.abs(rel).max():.2e}')
        ok &= bool(np.abs(rel).max() < tol)
        T_ = lambda a: torch.as_tensor(a, dtype=dt).unsqueeze(0)         # noqa: E731
        u_n = T_((u - u0) / ustd); y_n = T_((y - y0) / ystd)
        ix = torch.zeros(1, dtype=torch.long)
        # (ii-a) y_model = y_data
        m = LookupModel(y_n)
        with torch.no_grad():
            closed_loop_rollout(m, m.output_only, u_n, y_n, torch.zeros(1, 1, dtype=dt), bank, ix)
        u_cl = torch.cat(m.u_seen, 0)
        dmax = float((u_cl - u_n[0]).abs().max())
        oka = dmax == 0.0
        ok &= oka
        print(f'(ii-a) {str(dt)}: y_model = y_data -> max|u_cl - u_data| = {dmax:.3e} '
              f'(normalised)  {"PASS" if oka else "FAIL"}')
        # (ii-b) y_model = y_data + d, d = logged error
        m = LookupModel(T_((y + e - y0) / ystd))
        with torch.no_grad():
            closed_loop_rollout(m, m.output_only, u_n, y_n, torch.zeros(1, 1, dtype=dt), bank, ix)
        u_cl = torch.cat(m.u_seen, 0).double().numpy()
        fb = (u_cl - u_n[0].double().numpy()) * ustd
        relb = float(np.abs(fb - ref).max() / np.abs(ref).max())
        okb = relb <= tol
        ok &= okb
        print(f'(ii-b) {str(dt)}: feedback vs scipy sosfilt reference: max rel err {relb:.2e} '
              f'(tol {tol:g}), |ref| max {np.abs(ref).max():.1f} N  {"PASS" if okb else "FAIL"}')
        if dt == torch.float32:
            # (ii-c) diagnostic: the float64 reference driven by the error AS float32 represents it
            yd32 = ((y - y0) / ystd).astype(np.float32).astype(np.float64)
            ym32 = ((y + e - y0) / ystd).astype(np.float32).astype(np.float64)
            e32 = (yd32 - ym32) * ystd                       # what the f32 rollout feeds K
            ref32 = np.zeros_like(u)
            for j, ax in enumerate(tc.AXES):
                ref32[:, j] = sosfilt(sos[ax], e32[:, j]) * a2n[j]
            relc = float(np.abs(fb - ref32).max() / np.abs(ref32).max())
            rnd = np.abs(e32 + e).max(0)
            print(f'(ii-c) float32 input rounding: max |e_f32 - e| per axis {rnd} m '
                  f'(ystd {ystd} m); f32 rollout vs f64 reference fed the f32-rounded error: '
                  f'max rel err {relc:.2e}')
    print('G3 (ii)', 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
