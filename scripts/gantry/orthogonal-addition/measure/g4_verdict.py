"""G4 verdict per OA-008, from two measure_delta.py JSONs (null control and addition dataset).

    B = RMS over the ten of b_null (combo scale)
    PASS iff (i)  max over the nine non-m_diff parameters |b_add - b_null| <= B      (combo %)
             (ii) |b_add,m_diff - b_null,m_diff| <= B, both in % of |m1 - m2|      (own scale)
             (iii) ||Delta*_add|| >= 8.49 and >= 5 ||Delta*_null||
All at stride 1, J at theta*, Delta* stencil variant. The J at theta_0 (detuned) numbers are
printed beside it, not deciding.
Usage: python measure/g4_verdict.py <null.json> <addition.json> [<contrast.json> ...]
"""
__project_origin__ = "added"

import json
import sys

import numpy as np

M_DIFF = 7
OWN = 10.45 / 0.5          # combo scale of m_diff (mean actuator mass) over its own |m1 - m2|


def load(path, tag='J @ theta*, Delta* (stencil)'):
    js = json.load(open(path))
    p = next(p for p in js['predictions'] if p['tag'] == tag)
    return js, np.asarray(p['pct'], float), float(p['m_diff_own_pct']), float(p['rho']), \
        float(js['delta_norm']['stencil'])


def main():
    jn, bn, own_n, rho_n, dn = load(sys.argv[1])
    names = jn['combo_names']
    assert jn['stride'] == 1, 'the verdict needs stride 1'
    B = float(np.sqrt((bn ** 2).mean()))
    print('null control %s: ||Delta*|| %.4f, rho* %.4f, B = %.4f %% combo-err, m_diff own %+.4f %%'
          % (jn['mode'], dn, rho_n, B, own_n))
    for path in sys.argv[2:]:
        ja, ba, own_a, rho_a, da = load(path)
        assert ja['stride'] == 1
        d = ba - bn
        nine = [i for i in range(10) if i != M_DIFF]
        worst = max(abs(d[i]) for i in nine)
        dmd = own_a - own_n
        c1 = worst <= B
        c2 = abs(dmd) <= B
        c3 = da >= 8.49 and da >= 5 * dn
        print('\n%s' % ja['mode'])
        print('  %-8s %10s %10s %10s  %s' % ('param', 'b_null %', 'b_add %', 'shift %', 'in band'))
        for i in range(10):
            if i == M_DIFF:
                print('  %-8s %10.4f %10.4f %10.4f  %s   (combo scale)' % (names[i], bn[i], ba[i], d[i], '-'))
                print('  %-8s %10.4f %10.4f %10.4f  %s   (own scale)' % ('m_diff', own_n, own_a, dmd,
                                                                      'yes' if c2 else 'NO'))
            else:
                print('  %-8s %10.4f %10.4f %10.4f  %s' % (names[i], bn[i], ba[i], d[i],
                                                          'yes' if abs(d[i]) <= B else 'NO'))
        print('  (i)  max shift, nine params: %.4f %% vs B %.4f %%  -> %s' % (worst, B, 'ok' if c1 else 'FAIL'))
        print('  (ii) m_diff shift own scale: %+.4f %% vs B %.4f %%  -> %s' % (dmd, B, 'ok' if c2 else 'FAIL'))
        print('  (iii) ||Delta*_add|| %.4f vs max(8.49, 5 x %.4f = %.4f)  -> %s'
              % (da, dn, 5 * dn, 'ok' if c3 else 'FAIL'))
        print('  rho* %.4e; combo-err RMS %.4f %% (null %.4f %%)'
              % (rho_a, float(np.sqrt((ba ** 2).mean())), B))
        try:
            _, b0n, own0n, _, _ = load(sys.argv[1], 'J @ theta_0 (detuned), Delta* (stencil)')
            _, b0a, own0a, _, _ = load(path, 'J @ theta_0 (detuned), Delta* (stencil)')
            d0 = b0a - b0n
            print('  [not deciding] J at theta_0: max shift nine %.4f %%, m_diff own shift %+.4f %%'
                  % (max(abs(d0[i]) for i in nine), own0a - own0n))
        except StopIteration:
            print('  [not deciding] J at theta_0: not measured (OA_LEAN run)')
        print('  G4 VERDICT: %s' % ('PASS' if (c1 and c2 and c3) else 'FAIL'))


if __name__ == '__main__':
    main()
