"""Model selection over the black-box grid: the configuration with the lowest validation NRMS_e of its best checkpoint; writes
the search table with the parameter count of each configuration."""
__project_origin__ = "added"

import json
import sys
from pathlib import Path

import numpy as np
from deepSI.fit_systems.encoders import SS_encoder_general_hf

from blackbox.bb import model_kwargs
from gantry_dynamic.config import RunConfig

# folder of one run_black_box.sh job, with run folders <index>_<name>_n<updates>_<job>; the first argument overrides it
GRID_DIR = None
OUT_CSV = None     # search table; None: <GRID_DIR>/selection.csv; the second argument overrides it
NU = RunConfig.ny + RunConfig.nu   # black-box inputs u = (r, u^ms)
NY = RunConfig.ny                  # black-box outputs e = y - r


def n_params(nx, fh_width, e_width):
    """Trainable parameters of psi, f_bb and h_bb for one configuration, from nets built for it (no checkpoint).

    Args:
        nx: state dimension n_x.
        fh_width: hidden width of f_bb and h_bb.
        e_width: hidden width of psi.
    Returns:
        Parameter count (int).
    """
    model = SS_encoder_general_hf(**model_kwargs(nx, fh_width, e_width))
    # deepSI needs the input and output sizes before it builds the nets
    model.nu, model.ny = NU, NY
    model.init_nets(NU, NY)
    return sum(p.numel() for net in (model.encoder, model.hfn) for p in net.parameters())


def main(grid, out):
    """Writes the search table of one grid folder.

    Args:
        grid: folder of the run folders.
        out: search table (csv).
    """
    # 1. Read each run: its validation NRMS_e, parameter count, best update and status.
    rows = []
    for run_dir in sorted(grid.glob('*_n*_*'), key=lambda p: int(p.name.split('_')[0])):
        with open(run_dir / 'result.json') as fh:
            res = json.load(fh)
        args = res['args']
        rows.append(dict(index=int(run_dir.name.split('_')[0]), name='_'.join(run_dir.name.split('_')[1:-2]),
                         nx=args['nx'], width=args['fh_width'], lr=args['lr'],
                         score=float(np.mean(res['nrms_e_per_record'])),
                         n_par=n_params(args['nx'], args['fh_width'], args['e_width']),
                         best_update=res['best_update'], status=res['status']))
    # 2. Select the lowest NRMS_e; a failed run's score is not finite and is skipped.
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
    grid_dir = Path(sys.argv[1] if len(sys.argv) > 1 else GRID_DIR)
    out_csv = sys.argv[2] if len(sys.argv) > 2 else OUT_CSV
    main(grid_dir, Path(out_csv) if out_csv else grid_dir / 'selection.csv')
