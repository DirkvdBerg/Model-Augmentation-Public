"""Is the open-loop drift explained by a missing DC HOLD force?
u_total = u_fb + f_ms (closed loop). Open-loop uses only f_ms (zero-mean). If the
feedback u_fb carried a sustained (DC) force, removing it omits a constant force ->
the K=0 axis drifts at v_ss = F_dc/c. Check the means (logical) and predict v_ss."""
import os, sys
import numpy as np
from scipy.io import loadmat
sys.path.insert(0, os.path.dirname(__file__))
import drift_common as dc

REC = 'V1_standstill_Yp10.mat'
rec = dc.load_record(REC, dc.DEFAULT_FS_NEW)
m = loadmat(os.path.join(dc.DATA_DIR, REC), squeeze_me=True)
D = rec.D
def blockmean(a):
    a = np.asarray(a, float);  a = a[:, None] if a.ndim == 1 else a
    n = a.shape[0] // D
    return a[:n*D].reshape(n, D, a.shape[1]).mean(axis=1)
u_total = blockmean(m['u_total']); f_ms = blockmean(m['f_sim']); u_fb = blockmean(m['u_fb'])
# interior window (match s3 START=100)
sl = slice(100, None)
u_total, f_ms, u_fb = u_total[sl], f_ms[sl], u_fb[sl]

P = dc.Pnp
def logmean(x): return P @ x.mean(axis=0)   # logical DC per channel
print('Means over the interior window (STAGE):')
print(f'  u_total mean = {u_total.mean(axis=0)}')
print(f'  u_fb    mean = {u_fb.mean(axis=0)}')
print(f'  f_ms    mean = {f_ms.mean(axis=0)}')
print('\nMeans in LOGICAL coords (P @ stage-mean):')
ut_l, ufb_l, fms_l = logmean(u_total), logmean(u_fb), logmean(f_ms)
print(f'  u_total logical mean = {ut_l}')
print(f'  u_fb    logical mean = {ufb_l}   <- DC force the feedback provided')
print(f'  f_ms    logical mean = {fms_l}')

# Predicted open-loop drift velocity if the missing DC hold force is u_fb's mean.
# For the K=0 axes: X (logical idx0) uses cX=cg1+cg2; Y (idx2) uses cy.
cX = dc.cg1f + dc.cg2f; cY = dc.cyf
print(f'\nK=0 damping: cX(=cg1+cg2)={cX:.3f}  cY={cY:.3f}')
print(f'Predicted v_ss from MISSING u_fb DC (open-loop omits +u_fb): v = -u_fb_dc / c')
print(f'  X logical: -{ufb_l[0]:+.3e}/{cX:.2f} = {-ufb_l[0]/cX:+.3e} m/s   (s3 measured X ~ +1.03e-4 m/s)')
print(f'  Y logical: -{ufb_l[2]:+.3e}/{cY:.2f} = {-ufb_l[2]/cY:+.3e} m/s   (s3 measured Y ~ -1.73e-4 m/s)')
print('\nReading: if -u_fb_dc/c matches the measured open-loop drift velocity, the drift is the')
print('MISSING DC HOLD FORCE (the feedback was providing it), NOT a model defect or multisine rectification.')
