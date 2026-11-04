"""D-295 tree B: swap only the P^-T of D-294 back to the thesis forms, keeping theta^0 in float64 (_D).

    HARNESS_CLEAN_ROOT=<copy of Clean/> python -u pinv_revert.py apply | restore | status [--expect final|reverted]

d294_revert.py also swaps _D to float32, which on today's Clean/ changes more than D-294 did (its tree differs from
mainline by about 1e-8 where the final tree is bitwise). Here only the P^-T forms of the thesis code come back, each
where mainline uses it: the float64 inverse of the float32 P (mainline obc_gantry.py reference set, model.py encoder
start), and in the normalisation the inverse computed in float32 (mainline data.py); and the output map Cd, Dd in
float32, as mainline builds it from its float32 P (2026-11-04, after the common-start PS2 test). Same mechanism as
d294_revert.py; never run it on the repository's Clean/.
"""
import argparse
import sys

import d294_revert as R

R.SWAPS = (
    ('model_augmentation/systems/gantry_ss.py',
     'P_INV_T = np.array([[0.5, 0.5, 0.0], [1 / float(Lb), -1 / float(Lb), 0.0], [0.0, 0.0, 1.0]])',
     'P_INV_T = np.linalg.inv(P.detach().cpu().numpy().astype(np.float32).astype(np.float64).T)'),
    ('gantry_dynamic/data.py',
     'from model_augmentation.systems.gantry_ss import Cd, Dd, P_INV_T',
     'from model_augmentation.systems.gantry_ss import Cd, Dd, P_INV_T, P'),
    ('gantry_dynamic/data.py',
     '        q = record.y @ P_INV_T.T',
     '        q = (np.linalg.inv(P.numpy().astype(np.float32).T).astype(cfg.dtype_np) @ record.y.T).T'),
    # the output map: mainline builds Cd = [P^T 0] and Dd from its float32 P (gantry_ss.py), so in float32
    ('model_augmentation/systems/gantry_ss.py',
     'Cd = np.hstack([P.numpy().T, np.zeros((3, 3), P.numpy().dtype)])',
     'Cd = np.hstack([P.numpy().astype(np.float32).T, np.zeros((3, 3), np.float32)])'),
    ('model_augmentation/systems/gantry_ss.py',
     'Dd = np.zeros((3, 3), P.numpy().dtype)',
     'Dd = np.zeros((3, 3), np.float32)'),
)
R.FILES = sorted({f for f, _, _ in R.SWAPS})

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('apply', 'restore', 'status'))
    ap.add_argument('--expect', choices=('final', 'reverted'))
    args = ap.parse_args()
    if R.CLEAN == R.os.path.abspath(R.os.path.join(R.REPO, 'Clean')) and args.action != 'status':
        sys.exit('refused: set HARNESS_CLEAN_ROOT to a copy of Clean/, never the repository tree')
    if args.action == 'status':
        s = R.report()
        sys.exit(1 if s == 'mixed' or (args.expect and s != args.expect) else 0)
    need, to = ('final', 'reverted') if args.action == 'apply' else ('reverted', 'final')
    files = {f: R._read(f) for f in R.FILES}
    if R.state(files) != need:
        sys.exit('the tree is %s, %s needs %s' % (R.state(files), args.action, need))
    for f, b in R.swapped(files, to).items():
        with open(R.os.path.join(R.CLEAN, *f.split('/')), 'wb') as fh:
            fh.write(b)
    R.report()
