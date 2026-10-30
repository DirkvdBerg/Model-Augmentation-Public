"""Model selection over the R8 grid: selected = argmin over configurations of the validation NRMS_e (mean over
records and axes, best checkpoint, result.json `nrms_e_per_record`). Writes the search table with the parameter
count of each configuration.

Usage: python -u Clean/blackbox/select_bb.py <grid folder> <output csv>
       (grid folder: the run folders <index>_<name>_n<its>_<job> of one run_r8.sh job)
"""
__project_origin__ = "added"

import json
import sys
from pathlib import Path

import numpy as np
from deepSI.fit_systems.encoders import SS_encoder_general_hf

from bb import model_kwargs

# u = (r, u^ms) and e = y - r, three stage axes each
NU, NY = 6, 3


def n_params(nx, fh_width, e_width):
    """Number of trainable parameters of psi, f_bb and h_bb for one configuration.

    Built from the configuration, so the count needs no checkpoint.

    Args:
        nx: state dimension n_x.
        fh_width: hidden width of f_bb and h_bb.
        e_width: hidden width of psi.
    Returns:
        Parameter count (int).
    """
    model = SS_encoder_general_hf(**model_kwargs(nx, fh_width, e_width))
    model.nu, model.ny = NU, NY
    model.init_nets(NU, NY)
    return sum(p.numel() for net in (model.encoder, model.hfn) for p in net.parameters())


def main():
    """Writes the search table of one grid folder."""
    grid, out = Path(sys.argv[1]), Path(sys.argv[2])
    # 1. Read each run: its validation NRMS_e, parameter count, best update and status.
    rows = []
    for d in sorted(grid.glob('*_n*_*'), key=lambda p: int(p.name.split('_')[0])):
        res = json.load(open(d / 'result.json'))
        a = res['args']
        rows.append(dict(index=int(d.name.split('_')[0]), name='_'.join(d.name.split('_')[1:-2]), nx=a['nx'],
                         width=a['fh_width'], lr=a['lr'], score=float(np.mean(res['nrms_e_per_record'])),
                         n_par=n_params(a['nx'], a['fh_width'], a['e_width']), best_update=res['best_update'],
                         status=res['status']))
    # 2. Select: argmin NRMS_e over the runs with a finite score.
    sel = min((r for r in rows if np.isfinite(r['score'])), key=lambda r: r['score'])
    # 3. Write the search table.
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as fh:
        fh.write('index,name,nx,width,lr,best_val_nrms_e,n_params,best_update,status,selected\n')
        for r in rows:
            fh.write('%d,%s,%d,%d,%.0e,%.6f,%d,%d,%s,%d\n' % (r['index'], r['name'], r['nx'], r['width'], r['lr'],
                                                             r['score'], r['n_par'], r['best_update'], r['status'],
                                                             int(r is sel)))
            print('%2d %-16s NRMS_e %.4f | %5d params | update %5d | %s%s'
                  % (r['index'], r['name'], r['score'], r['n_par'], r['best_update'], r['status'],
                     '  <- SELECTED' if r is sel else ''), flush=True)
    print('wrote', out)


if __name__ == '__main__':
    main()
