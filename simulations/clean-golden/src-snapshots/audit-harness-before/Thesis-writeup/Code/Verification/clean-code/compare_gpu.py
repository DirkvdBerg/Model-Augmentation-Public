"""Compare the compiled GPU smokes of design 6.4 (a) and (b) between the two code roots.

    python -u compare_gpu.py OUT        (OUT = the folder server_gpu_smoke.sbatch wrote)

REFUSED, nothing compared, when a clean-side result.json records no clean source manifest or one
that differs from the current Clean/ tree (H2).
(a), per row: each side ran to the end, logged "training rollout COMPILED" and no "skipping
cudagraphs", wrote both pairs and ran the compiled branch (unique_graphs > 0); the initial
validation score and the four training losses agree within 1e-9 relative. (b): both sides report
the same deterministic and used_compiled, and loss_first within 1e-9 relative.
"""
import json
import os
import sys

import hcommon as H

CASES = ('a-221', 'a-226', 'b-226')


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def load(p):
    """result.json without its clean manifest (checked by refusal)."""
    res = json.load(open(p))
    res.pop('clean_tree', None)
    return res


def refusal(out):
    """H2: [] when every clean result present was made from the current Clean/ tree, else the problems."""
    probs = []
    for case in CASES:
        p = os.path.join(out, '%s-clean' % case, 'result.json')
        if os.path.exists(p):
            probs += ['%s-clean: %s' % (case, x) for x in H.clean_tree_problems(json.load(open(p)).get('clean_tree'))]
    return probs


def main(out):
    probs = refusal(out)
    if probs:
        for x in probs:
            print('REFUSED: %s' % x)
        with open(os.path.join(out, 'compare-gpu.json'), 'w') as fh:
            json.dump({'status': 'REFUSED', 'problems': probs}, fh, indent=1)
        print('COMPARE_GPU REFUSED')
        return False
    ok, rep = True, {}
    for row in (221, 226):
        r = {}
        for side in ('mainline', 'clean'):
            p = os.path.join(out, 'a-%d-%s' % (row, side), 'result.json')
            if not os.path.exists(p):
                r[side] = 'missing'
                ok = False
                continue
            res = load(p)
            log = open(os.path.join(out, 'logs', 'a-%d-%s.log' % (row, side)), errors='replace').read()
            res['log_compiled'] = 'training rollout COMPILED' in log
            res['log_no_skipping_cudagraphs'] = 'skipping cudagraphs' not in log
            ok &= bool(res['ok'] and res['log_compiled'] and res['log_no_skipping_cudagraphs'])
            r[side] = res
        if all(isinstance(r.get(s), dict) for s in ('mainline', 'clean')):
            m, c = r['mainline'], r['clean']
            r['initial_validation_rel'] = rel(m['initial_validation'], c['initial_validation'])
            r['adam_losses_rel'] = [rel(a, b) for a, b in zip(m['adam_losses'], c['adam_losses'])]
            r['pass'] = bool(r['initial_validation_rel'] <= 1e-9 and len(r['adam_losses_rel']) == 4
                             and max(r['adam_losses_rel']) <= 1e-9 and m['ok'] and c['ok']
                             and m['log_compiled'] and c['log_compiled']
                             and m['log_no_skipping_cudagraphs'] and c['log_no_skipping_cudagraphs'])
            ok &= r['pass']
        rep['a-%d' % row] = r
    b = {}
    for side in ('mainline', 'clean'):
        p = os.path.join(out, 'b-226-%s' % side, 'result.json')
        b[side] = load(p) if os.path.exists(p) else 'missing'
    if all(isinstance(b[s], dict) for s in b):
        m, c = b['mainline'], b['clean']
        b['pass'] = bool(m['deterministic'] == c['deterministic'] and m['used_compiled'] == c['used_compiled']
                         and rel(m['loss_first'], c['loss_first']) <= 1e-9)
        b['loss_first_rel'] = rel(m['loss_first'], c['loss_first'])
        ok &= b['pass']
    else:
        ok = False
    rep['b-226'] = b
    print(json.dumps(rep, indent=1, default=str))
    with open(os.path.join(out, 'compare-gpu.json'), 'w') as fh:
        json.dump(rep, fh, indent=1, default=str)
    print('COMPARE_GPU %s' % ('PASS' if ok else 'FAIL'))
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(sys.argv[1]) else 1)
