"""Focused checks for ps2.py (black box: two tiny runs, then their logs and checkpoints are compared).

  T1 the model equals the baseline at update 0 in both arms (update-0 loss 1.5133e-08, run 42 settings);
  T2 the thesis step is the control's step while the parameters agree: thesis losses at updates 1 and 2 are identical
     across arms (from update 3 on the auxiliary step has moved the encoder's x_a rows, so the runs may diverge);
  T3 inside one candidate run (P_CHECK=1), every auxiliary step leaves the physical block, the shared hidden layers,
     the physical-row output and the encoder's x_b path bit-identical, and S2 changes only hidden + added-row output;
  T4 S2 changes the added-row output layer (the deployed g_aug was trained);
  T5 the stratified split holds out TR-S5, TR-Y3, TR-P4, TR-T4.
Run: python test_ps2.py   (about 3 minutes, CPU; writes outputs/ps2_test_*.{log,pt})
"""
import os
import re
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs')


def run(tag, **env):
    e = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', P_SEED='0', P_CAL='stratified',
             P_S1_CAP='4', P_S2_CAP='200', P_HEAD='mlp', **env)
    log = subprocess.run([sys.executable, '-u', os.path.join(HERE, 'ps2.py'), '4', tag], env=e, capture_output=True,
                         text=True, encoding='utf-8').stdout
    open(os.path.join(OUT, 'ps2_%s.log' % tag), 'w', encoding='utf-8').write(log)
    return log


def main():
    lc, lk = run('test_cand', P_MODE='cand', P_THR='1.5', P_CHECK='1'), run('test_ctrl', P_MODE='control')
    ok = True

    def check(name, cond, msg=''):
        nonlocal ok
        ok &= bool(cond)
        print('  %s %s %s' % ('PASS' if cond else 'FAIL', name, msg), flush=True)

    l0 = [re.search(r'\[eval    0\] phase \S+ loss (\S+)', l).group(1) for l in (lc, lk)]
    check('T1 update-0 loss = baseline', l0[0] == l0[1] == '1.5133e-08', str(l0))
    oe = [re.findall(r'upd +[123] oe (\S+)', l) for l in (lc, lk)]
    check('T2 thesis losses identical at updates 1 and 2', oe[0][:2] == oe[1][:2] and len(oe[0]) == 3, str(oe))
    sc = torch.load(os.path.join(OUT, 'ps2_test_cand.pt'), weights_only=False)
    sk = torch.load(os.path.join(OUT, 'ps2_test_ctrl.pt'), weights_only=False)
    ran_s2 = 'S2 end' in lc
    check('T3/T4 precondition: S2 ran in the candidate', ran_s2)
    same, diff = [], []
    for k in sc['hfn']:
        (same if torch.equal(sc['hfn'][k], sk['hfn'][k]) else diff).append(k)
    aux = re.findall(r'CHECK aux step \d+: (.*)', lc); s2c = re.findall(r'CHECK S2: (.*)', lc)
    check('T3 auxiliary steps write no protected tensor', len(aux) >= 3 and all(a == 'OK' for a in aux), str(aux))
    check('T3 S2 writes only hidden layers and the added-row output', s2c == ['OK'], str(s2c))
    check('T4 added-row output layer changed by S2', any('out_a' in k for k in diff), str([k for k in diff if 'out_a' in k]))
    cal = re.search(r'calibration records (\[[^\]]*\])', lc).group(1)
    check('T5 stratified calibration records', sorted(eval(cal)) == [4, 7, 11, 17], cal)
    print('ALL PASS' if ok else 'SOME FAILED', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
