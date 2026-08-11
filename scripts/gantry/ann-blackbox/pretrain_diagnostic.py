"""Pre-training diagnostics for the standalone full ANN. No training, no baseline, no augmentation.

Everything here uses the KNOWN 8-state truth, which is what makes an a priori answer possible at
all: Beintema et al. measure the characteristic timescale from a trained model's k-step error
curve, and we can compute the same quantity from the truth before training anything.

  A  float32 free-run accumulation   is the acceptance target reachable in float32 at all?
  B  characteristic timescale        -> nf, via the SUBNET guideline "a few times the largest
                                       characteristic time scale" (Automatica 156:111210, S3.4)
  C  encoder recoverability          can an na-sample past window determine the state?
  D  oracle pipeline bound           best free-run sim-RMS achievable with PERFECT dynamics and
                                       an encoder, i.e. the ceiling the ANN is training towards
  E  state -> window conditioning    the joint (fs, nf) criterion

D is the one that answers "can the ANN learn this": it hands the pipeline exact dynamics and asks
what it still cannot do. Anything the ANN is asked to beat that D cannot reach is unreachable.

Reference for the nf rule:
  Beintema, Schoukens, Toth, "Deep subspace encoders for nonlinear system identification",
  Automatica 156:111210, 2023, DOI 10.1016/j.automatica.2023.111210, Section 3.4.
  NOTE ITS SCOPE: the guideline is stated "for stable data-generating systems" and the paper's
  Condition 1 requires incremental exponential output stability with lambda < 1. Our plant has two
  poles exactly at z = 1, so that condition FAILS here and the rule is applied to the damped
  subspace only. Test B reports the integrator modes separately rather than hiding them.
"""
__project_origin__ = "added"

import os
import sys
import json
import numpy as np
from scipy.linalg import expm, svdvals

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry', 'msd-offset'))
from data import load, FS_NATIVE                   # same anti-alias decimation as the trainer
import plant                                       # the truth: M8, _K4, _C4, _E43, P_np

TRAIN, VAL = 'T10_aprbs_60', 'V2_aprbs_Ylow'
RATES = [4000.0, 2000.0, 1000.0, 800.0]
NF_GRID = [400, 800, 2000, 4000, 8000]
NA = 17                                            # the trainer's encoder lag
TARGET = 1.6e-4                                    # m, the FP baseline's untrained sim-RMS


def truth_ct(Y):
    """Continuous-time truth linearised at frozen Y, absorber at rest. x = [q; qdot], q logical."""
    Minv = np.linalg.inv(plant.M8(Y, 0.0, freeze=True))
    A = np.block([[np.zeros((4, 4)), np.eye(4)],
                  [-Minv @ plant._K4, -Minv @ plant._C4]])
    B = np.vstack([np.zeros((4, 3)), Minv @ plant._E43])
    C = np.hstack([plant.P_np.T @ np.hstack([np.eye(3), np.zeros((3, 1))]), np.zeros((3, 4))])
    return A, B, C                                 # y = C x is STAGE position [x1, x2, Y], metres


def discretise(A, B, ts):
    """Exact ZOH via the block-matrix exponential."""
    n, m = A.shape[0], B.shape[1]
    E = expm(np.block([[A, B], [np.zeros((m, n + m))]]) * ts)
    return E[:n, :n], E[:n, n:]


def free_run(Ad, Bd, C, x0, u, dtype):
    Ad, Bd, C = Ad.astype(dtype), Bd.astype(dtype), C.astype(dtype)
    x = np.asarray(x0, dtype)
    out = np.empty((len(u), C.shape[0]), dtype)
    u = u.astype(dtype)
    for k in range(len(u)):
        out[k] = C @ x
        x = Ad @ x + Bd @ u[k]
    return out.astype(np.float64)


def past_window(u, y, na):
    """Regressor [u[k-na:k], y[k-na:k]] flattened, and the aligned sample index."""
    n = len(u) - na
    idx = np.arange(na, na + n)
    cols = [u[i:i + n] for i in range(na)] + [y[i:i + n] for i in range(na)]
    return np.hstack(cols), idx


def true_states(name, fs):
    """The exact 8-state truth from the record, decimated like the outputs.

    x = [X, Th, Y, da, dX, dTh, dY, vda] (plant.py docstring). The record stores x_logical as
    [q_logical, qdot_logical] plus delta_a and vdelta_a separately, so it is reassembled here.
    """
    from scipy.signal import decimate
    rec = plant.load_record(name, fs_new=int(FS_NATIVE))
    xl, da, vda = rec['x_logical'], rec['delta_a'], rec['vdelta_a']
    x = np.column_stack([xl[:, 0:3], da, xl[:, 3:6], vda])
    q = int(round(FS_NATIVE / fs))
    if q > 1:
        x = decimate(x, q, ftype='fir', zero_phase=True, axis=0)
    return np.ascontiguousarray(x, dtype=float)


rows = []
for fs in RATES:
    ts = 1.0 / fs
    va = load(VAL, fs)
    xt = true_states(VAL, fs)[:len(va.y)]
    Y_op = float(np.mean(va.y[:, 2]))
    A, B, C = truth_ct(Y_op)
    Ad, Bd = discretise(A, B, ts)

    # u in the record is STAGE force; the truth's B expects LOGICAL force (plant.deriv8 uses _E43 @ u)
    u_log = (plant.P_np @ va.u.T).T

    # --- B: characteristic timescales, and the integrator modes named separately -------------
    lam_c = np.linalg.eigvals(A)
    integ = np.abs(lam_c) < 1e-9                                  # the two free integrators
    damped = lam_c[(~integ) & (lam_c.real < -1e-12)]
    taus = np.sort(-1.0 / damped.real)[::-1]
    tau_max = float(taus[0])
    nf_rule = int(np.ceil(3 * tau_max * fs))                      # "a few times" taken as 3

    # transient decay of an output perturbation, restricted to the damped subspace
    Ad_p = Ad.copy()
    w, V = np.linalg.eig(Ad)
    keep = np.abs(np.abs(w) - 1.0) > 1e-9                         # drop the z = 1 directions
    P_damp = (V[:, keep] @ np.linalg.pinv(V[:, keep])).real
    dx = P_damp @ np.random.default_rng(0).standard_normal(8)
    dx /= np.linalg.norm(dx)
    dec, xk = [], dx.copy()
    for _ in range(int(6 * tau_max * fs)):
        dec.append(np.linalg.norm(C @ xk)); xk = Ad @ xk
    dec = np.array(dec) / (dec[0] + 1e-300)
    k99 = int(np.argmax(dec < 0.01)) if np.any(dec < 0.01) else len(dec)

    # --- A: float32 accumulation of the TRUE model against itself in float64 -----------------
    x0 = np.zeros(8); x0[2] = Y_op
    y64 = free_run(Ad, Bd, C, x0, u_log, np.float64)
    y32 = free_run(Ad, Bd, C, x0, u_log, np.float32)
    f32_gap = float(np.sqrt(np.mean((y64 - y32) ** 2)))
    f32_gap_Y = float(np.sqrt(np.mean((y64[:, 2] - y32[:, 2]) ** 2)))

    # --- C: encoder recoverability, closed form ----------------------------------------------
    # ONE-SIDED TEST: the true state being linearly recoverable proves an encoder can do it.
    # The converse does not hold, since a black box may use any state coordinates it likes.
    Phi, idx = past_window(va.u, va.y, NA)
    Xtrue = xt[idx]                                                # the exact 8-state, not a proxy
    Phi1 = np.hstack([Phi, np.ones((len(Phi), 1))])
    coef, *_ = np.linalg.lstsq(Phi1, Xtrue, rcond=None)
    Xhat = Phi1 @ coef
    sc = np.std(Xtrue, axis=0) + 1e-300                            # per-state, the 8 states differ by orders
    enc_rel = float(np.mean(np.std(Xtrue - Xhat, axis=0) / sc))
    enc_per_state = [float(v) for v in np.std(Xtrue - Xhat, axis=0) / sc]

    # --- D: oracle pipeline bound -------------------------------------------------------------
    # PERFECT dynamics, encoder-estimated initial state, free run over the rest of the record.
    # This is the ceiling the ANN trains towards: it cannot beat exact dynamics.
    k0 = int(idx[0])
    y_or_enc = free_run(Ad, Bd, C, Xhat[0], u_log[k0:], np.float64)
    oracle_enc = float(np.sqrt(np.mean((y_or_enc - va.y[k0:]) ** 2)))
    # and the same with the TRUE initial state, isolating how much the encoder alone costs
    y_or_true = free_run(Ad, Bd, C, xt[k0], u_log[k0:], np.float64)
    oracle_rms = float(np.sqrt(np.mean((y_or_true - va.y[k0:]) ** 2)))
    oracle_Y = float(np.sqrt(np.mean((y_or_true[:, 2] - va.y[k0:, 2]) ** 2)))

    # --- E: state -> nf-window conditioning ---------------------------------------------------
    cond = {}
    for nf in NF_GRID:
        O, Ak = [], np.eye(8)
        step = max(1, nf // 200)                                   # subsample rows; rank is unaffected
        for k in range(0, nf, step):
            O.append(C @ Ak)
            for _ in range(step):
                Ak = Ad @ Ak
        s = svdvals(np.vstack(O))
        cond[nf] = dict(sigma_min=float(s[-1]), sigma_max=float(s[0]),
                        cond=float(s[0] / (s[-1] + 1e-300)))

    row = dict(fs=fs, Y_op=Y_op, n_integrator_modes=int(integ.sum()),
               tau_max=tau_max, taus=[float(t) for t in taus[:4]],
               nf_beintema_3tau=nf_rule, k99_damped=k99, k99_seconds=k99 * ts,
               f32_gap=f32_gap, f32_gap_Y=f32_gap_Y,
               encoder_rel_residual=enc_rel, encoder_per_state=enc_per_state,
               oracle_sim_rms=oracle_rms, oracle_sim_rms_Y=oracle_Y,
               oracle_sim_rms_encoder_init=oracle_enc,
               conditioning=cond)
    rows.append(row)

    print(f"fs={fs:6.0f}")
    print(f"  B  integrator modes: {int(integ.sum())}   damped taus: {np.round(taus[:4], 4)} s"
          f"   tau_max={tau_max:.4f} s")
    print(f"     Beintema nf = 3*tau_max*fs = {nf_rule}   (we use 400)"
          f"   damped transient reaches 1% at k={k99} ({k99*ts:.4f} s)")
    print(f"  A  float32 vs float64 free run of the TRUE model: {f32_gap:.4e} m pooled,"
          f" {f32_gap_Y:.4e} m on Y   (target {TARGET:.1e})")
    print(f"  C  encoder LS relative residual over {NA} past samples: {enc_rel:.4e}"
          f"   per state {np.array2string(np.array(enc_per_state), precision=2)}")
    print(f"  D  oracle free run, PERFECT dynamics + true x0:    {oracle_rms:.4e} m pooled,"
          f" {oracle_Y:.4e} m on Y")
    print(f"     oracle free run, PERFECT dynamics + encoder x0: {oracle_enc:.4e} m"
          f"   <- ceiling for the ANN pipeline")
    print(f"  E  cond(state->window):" + "".join(
        f"  nf={nf}: {cond[nf]['cond']:.2e}" for nf in NF_GRID))

out = os.path.join(HERE, 'results')
os.makedirs(out, exist_ok=True)
json.dump(dict(val=VAL, na=NA, target=TARGET, rows=rows),
          open(os.path.join(out, 'pretrain_diagnostic.json'), 'w'), indent=1)
print(f"\nwritten: {os.path.join(out, 'pretrain_diagnostic.json')}")
