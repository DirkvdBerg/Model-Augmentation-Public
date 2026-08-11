"""Data-only oversampling diagnostic. No training, no model.

Tests the mechanism the decimation sweep is built on (session brief section 3 item 2):
at 4 kHz one-step prediction is nearly trivial, so a short-horizon loss can look healthy
having learned no dynamics. Scores trivial predictors that a network could collapse onto,
in the training loss's own normalised units, at each candidate rate.

Predictors scored, all needing zero knowledge of the plant:
  HOLD    yhat[k+j] = y[k-1]                          "output the last sample forever"
  CVEL    yhat[k+j] = y[k-1] + (j+1)*(y[k-1]-y[k-2])  "integrate the last slope"
  MEAN    yhat[k]   = mean(y_train)                   free-run collapse of an untrained net

Everything is computed from the mean squared increment at lag L, so no windows are built.
"""
__project_origin__ = "added"

import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
from data import load                        # same loader and same anti-alias decimation as the trainer

TRAIN, VAL, NF = 'T10_aprbs_60', 'V2_aprbs_Ylow', 400
RATES = [4000.0, 2000.0, 1000.0, 800.0]


def increments(y, nf):
    """d[L] = mean over k and channels of (y[k+L] - y[k])**2, for L = 1..nf. HOLD error at lag L."""
    return np.array([np.mean((y[L:] - y[:-L]) ** 2) for L in range(1, nf + 1)])


def increments_cvel(y, nf):
    """Same but after removing the last one-step slope: (y[k+L] - y[k] - L*(y[k]-y[k-1]))**2."""
    v = np.diff(y, axis=0)                                  # v[k] = y[k+1] - y[k]
    return np.array([np.mean((y[L + 1:] - y[1:-L] - L * v[:-L]) ** 2) for L in range(1, nf + 1)])


rows = []
for fs in RATES:
    tr, va = load(TRAIN, fs), load(VAL, fs)
    y0, ystd = np.mean(tr.y, axis=0), np.std(tr.y, axis=0)  # deepSI's own norm (system_data.py:848-849)

    yn = (va.y - y0) / ystd                                 # validation in the loss's units
    d_hold, d_cvel = increments(yn, NF), increments_cvel(yn, NF)
    ytn = (tr.y - y0) / ystd                                # the TRAINING record, so these compare directly to the printed sqrt train loss
    dt_hold, dt_cvel = increments(ytn, NF), increments_cvel(ytn, NF)

    # MEAN predictor in raw metres, i.e. what an untrained net's free run collapses to
    bias = y0 - np.mean(va.y, axis=0)
    per_ch_mean = np.sqrt(bias ** 2 + np.var(va.y, axis=0))
    rms_mean_train = float(np.sqrt(np.mean(per_ch_mean ** 2)))
    rms_mean_oracle = float(np.sqrt(np.mean(np.var(va.y, axis=0))))   # same predictor, val's own mean

    row = dict(
        fs=fs, n=len(va), window_s=NF / fs,
        samples_per_period_158=fs / 158.114,
        # one step, normalised: 0 means y[k+1] is exactly y[k]
        hold_1step=float(np.sqrt(d_hold[0])),
        cvel_1step=float(np.sqrt(d_cvel[0])),
        # over the nf-step horizon the training loss actually averages over
        hold_nf=float(np.sqrt(np.mean(d_hold))),
        cvel_nf=float(np.sqrt(np.mean(d_cvel))),
        # same three on the TRAINING record: directly comparable to fit()'s printed sqrt train loss
        hold_1step_train=float(np.sqrt(dt_hold[0])), cvel_1step_train=float(np.sqrt(dt_cvel[0])),
        hold_nf_train=float(np.sqrt(np.mean(dt_hold))), cvel_nf_train=float(np.sqrt(np.mean(dt_cvel))),
        # free-run, raw metres, comparable to sim-RMS
        rms_mean_train=rms_mean_train, rms_mean_oracle=rms_mean_oracle,
        per_channel_mean=[float(v) for v in per_ch_mean],
        val_std=[float(v) for v in np.std(va.y, axis=0)], bias=[float(v) for v in bias],
    )
    rows.append(row)
    print(f"fs={fs:6.0f}  window {row['window_s']*1e3:6.1f} ms  {row['samples_per_period_158']:5.2f} samp/period(158Hz)")
    print(f"    1-step  HOLD {row['hold_1step']:.4e}   CVEL {row['cvel_1step']:.4e}   (normalised, 1.0 = as bad as predicting the mean)")
    print(f"    {NF}-step HOLD {row['hold_nf']:.4e}   CVEL {row['cvel_nf']:.4e}   (normalised, this is what the loss averages)")
    print(f"    on TRAIN record: 1-step HOLD {row['hold_1step_train']:.4e} CVEL {row['cvel_1step_train']:.4e}"
          f" | {NF}-step HOLD {row['hold_nf_train']:.4e} CVEL {row['cvel_nf_train']:.4e}   <- compare to fit()'s sqrt train loss")
    print(f"    free-run MEAN predictor: {rms_mean_train:.4e} m using train mean, {rms_mean_oracle:.4e} m using val's own mean")
    print(f"      per channel [x1 {per_ch_mean[0]:.4e}, x2 {per_ch_mean[1]:.4e}, Y {per_ch_mean[2]:.4e}] m"
          f"  bias [{bias[0]:+.4e}, {bias[1]:+.4e}, {bias[2]:+.4e}] m")

out = os.path.join(HERE, 'results')
os.makedirs(out, exist_ok=True)
json.dump(dict(train=TRAIN, val=VAL, nf=NF, rows=rows),
          open(os.path.join(out, 'oversampling_diagnostic.json'), 'w'), indent=1)
print(f"\nwritten: {os.path.join(out, 'oversampling_diagnostic.json')}")
