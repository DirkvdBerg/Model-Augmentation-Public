"""AD1: the pipeline's own selection metric on probe checkpoints: closed-loop FREE RUN over the whole validation
records (closed_loop.py validation_error, sim-RMS in metres, about 12 s per record), learned block on and off.
A locally unstable learned pair that the 0.1 s windows hide would show up here as growth / a large RMS.

Usage: python ad_valfree.py CKPT [CKPT ...]   (files in outputs/)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import u_probe as up                                                           # noqa: E402


def main(files):
    os.environ['U_INIT'] = 'zero'
    cfg, data, norm, fs, ann, others = up.setup()
    print('\n######## AD1 free-run validation (sim-RMS, metres), %d val records' % len(fs.simulator.val_records),
          flush=True)
    for fname in files:
        st = torch.load(os.path.join(HERE, 'outputs', fname), weights_only=False)
        fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
        out = []
        for tag, off in (('on', False), ('off', True)):
            g = ann.out_gate.clone()
            if off:
                ann.out_gate = torch.zeros_like(g)
            try:
                with torch.no_grad():
                    v = fs.simulator.validation_error(fs, data.val_ckpt_data, 'sim-RMS')
                per = np.asarray(fs.simulator.last_per_record, dtype=float)
            finally:
                ann.out_gate = g
            out.append('%s %.4e (per record %s)' % (tag, v, ' '.join('%.3e' % p for p in np.ravel(per))))
        print('  [%s, %d updates] %s' % (fname, st['n_upd'], ' | '.join(out)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
