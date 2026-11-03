"""Model selection over the black-box grid: the configuration with the lowest validation NRMS_e of its best
checkpoint; writes the search table with the parameter count of each configuration."""
__project_origin__ = "added"

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from deepSI.fit_systems.encoders import SS_encoder_general_hf

from blackbox.bb import GREY_BOX, data_config, model_kwargs
from gantry_dynamic.config import RunConfig

# one grid job folder, containing <index>_<name>_n<updates>_<job> run folders
GRID_DIR = None
OUT_CSV = None     # search table; None: <GRID_DIR>/selection.csv; the second argument overrides it
NU = RunConfig.ny + RunConfig.nu   # black-box inputs u = (r, u^ms)
NY = RunConfig.ny                  # black-box outputs e = y - r


def n_params(nx, fh_width, e_width, cfg=GREY_BOX):
    """Trainable parameters of psi, f_bb and h_bb for one configuration, from nets built for it (no checkpoint).

    Args:
        nx: state dimension n_x.
        fh_width: hidden width of f_bb and h_bb.
        e_width: hidden width of psi.
    Returns:
        Parameter count (int).
    """
    model = SS_encoder_general_hf(**model_kwargs(nx, fh_width, e_width, cfg))
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
    run_dirs = sorted(grid.glob('*_n*_*'), key=lambda p: (int(p.name.split('_')[0]), p.name))
    if len({path.name.rsplit('_', 1)[-1] for path in run_dirs}) > 1:
        raise ValueError('multiple grid jobs found; select one grid job folder at a time')
    rows = []
    for run_dir in run_dirs:
        if not (run_dir / 'result.json').is_file():
            print('skipping incomplete run:', run_dir.name)
            continue
        with open(run_dir / 'result.json') as fh:
            res = json.load(fh)
        args = res['args']
        # New runs record the count from the trained networks; construct networks only for legacy results.
        count = res['n_params'] if 'n_params' in res else n_params(
            args['nx'], args['fh_width'], args['e_width'], data_config(res))
        rows.append(dict(index=int(run_dir.name.split('_')[0]), name='_'.join(run_dir.name.split('_')[1:-2]),
                         nx=args['nx'], width=args['fh_width'], e_width=args['e_width'], lr=args['lr'],
                         best_val_nrms_e=float(np.mean(res['nrms_e_per_record'])), n_params=count,
                         best_update=res['best_update'], status=res['status']))
    # lowest NRMS_e among the finite scores. A 'failed' run (a non-finite loss at some update) still competes if its
    # best checkpoint scored finite; only a non-finite score is skipped
    sel = min((r for r in rows if np.isfinite(r['best_val_nrms_e'])),
              key=lambda r: r['best_val_nrms_e'], default=None)
    if sel is None:
        raise ValueError('no completed results with finite validation scores in %s' % grid)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=[*rows[0], 'selected'])
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r, lr='%.0e' % r['lr'], best_val_nrms_e='%.6f' % r['best_val_nrms_e'],
                                 selected=int(r is sel)))
            print('%2d %-16s NRMS_e %.4f | %5d params | update %5d | %s%s'
                  % (r['index'], r['name'], r['best_val_nrms_e'], r['n_params'], r['best_update'], r['status'],
                     '  <- SELECTED' if r is sel else ''), flush=True)
    print('wrote', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Select the lowest validation NRMS_e within one grid job.')
    p.add_argument('grid', nargs='?', type=Path, default=GRID_DIR)
    p.add_argument('out', nargs='?', type=Path, default=OUT_CSV)
    args = p.parse_args()
    if args.grid is None:
        p.error('provide a grid job folder or set GRID_DIR at the top of this file')
    main(args.grid, args.out or args.grid / 'selection.csv')
