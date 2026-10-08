"""Phase D on the thesis checkpoints (server copies in meeting-01-10-2026/server/Checkpoints).

Per checkpoint, on the six validation records, closed-loop free run from k0 (the selection chain):
  model          the trained model
  static only    the added states' ANN output rows zeroed (x_a stays 0 after the first step, the
                 static correction on the physical rows is kept): the model without added dynamics
  silenced       the learned block removed (closed_loop_report._without_learned)
  baseline true  silenced, physics at the true combinations
RMS per channel and per band (b_error_budget.BANDS); the learned combinations against the truth.
Evaluation only: no optimizer step, nothing written outside outputs/.

Usage: python e_thesis_checkpoints.py [job_id[_last] ...]   (default: every file in the folder)
"""
import os
import sys
import glob
import importlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GANTRY = os.path.dirname(HERE)
sys.path.insert(0, GANTRY)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(GANTRY, '..', '..')))

CK = os.path.join(GANTRY, 'meeting', 'meeting-01-10-2026', 'server', 'Checkpoints')
# job id -> (run id, arm, n_a, seed), from the log headers (thesis_86892_<run>.out)
JOBS = {'86893': (41, 'obc', 2, 1), '86894': (42, 'u', 2, 1), '86909': (43, 'obc', 2, 2),
        '86918': (44, 'u', 2, 2), '86945': (45, 'obc', 2, 3), '86960': (46, 'u', 2, 3),
        '86961': (47, 'obc', 0, 1), '87018': (48, 'obc', 4, 1), '87055': (49, 'obc', 8, 1)}

from b_error_budget import band_rms, BANDS                                      # noqa: E402


def cfg_for(arm, na, seed):
    os.environ.update(THESIS_ARM=arm, THESIS_NOISE='noisy', THESIS_START='detuned',
                      THESIS_NA=str(na), THESIS_SEED=str(seed), THESIS_DEVICE='cpu')
    for k in ('OBC_ARM', 'THESIS_SMOKE', 'THESIS_NITS'):
        os.environ.pop(k, None)
    mod = importlib.import_module('gantry_interconnect_dynamic')
    mod = importlib.reload(mod)
    return mod.CFG


def static_only(fs, route_len):
    import copy
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    m = copy.deepcopy(fs)
    ann = next(b for b in m.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    last = [l for l in ann.net.net if isinstance(l, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight[6:route_len].zero_()
        if last.bias is not None:
            last.bias[6:route_len].zero_()
    return m


def main(which):
    from gantry_dynamic.data import load_datasets, compute_normalization
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop, build_controller_bank
    from gantry_dynamic.training import load_checkpoint
    from gantry_dynamic.closed_loop_report import _without_learned, _true_free
    from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch
    from model_augmentation.fit_systems.blocks import Reduced_Gantry_State_Block

    data = norm = None
    ob = np.load(os.path.join(HERE, 'outputs', 'b_budget_noisy.npz'))
    vmask = ob['split'] == 'val'
    orc = {k: ob['band_' + k][vmask] for k in ('full', 'noabs', 'bt')}
    orc_rms = {k: ob['rms_' + k][vmask] for k in ('full', 'noabs', 'bt')}
    for tag in which:
        job, kind = (tag.split('_') + ['best'])[:2]
        run, arm, na, seed = JOBS[job]
        cfg = cfg_for(arm, na, seed)
        np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        if data is None:
            data = load_datasets(cfg)
            norm = compute_normalization(cfg, data)
        np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        fs = build_model(cfg.hp, cfg, data, norm)
        fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files,
                                         val_files=data.val_files, val_data=data.val_ckpt_data, verbose=False)
        path = os.path.join(CK, 'SSE_Interconnect_Composed_%s_%s.pth' % (job, kind))
        meta = load_checkpoint(fs, path, joint_estimation=True, start_phase='lbfgs')
        files, sdl = list(data.val_files), list(data.val_list)
        bank, rows, _ = build_controller_bank(files, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u, dtype=cfg.dtype_pt)
        phy = next(b for b in fs.hfn.connected_blocks if hasattr(b, 'free_params'))
        systems = {'model': fs, 'silenced': _without_learned(fs),
                   'baseline_true': _without_learned(fs, _true_free(phy))}
        if na > 0:
            systems['static_only'] = static_only(fs, 6 + na)
        k0 = max(fs.na, fs.nb)
        with torch.no_grad():
            c = phy.combinations_from_free(phy.free_params).double()
            ct = type(phy).combos_of(phy.params_init.double(), float(phy.Lb))
        dev = 100 * (c - ct) / ct.abs()
        print('\n######## run %d (job %s, %s) arm=%s n_a=%d seed=%d | ckpt bestfit %.4e, epoch %s' % (
            run, job, kind, arm, na, seed, float(meta['bestfit']), meta['done_epochs']))
        print('  combos vs true [%%]: %s' % ' '.join('%s %+.1f' % (n, v) for n, v in
                                                   zip(Reduced_Gantry_State_Block.COMBO_NAMES, dev.tolist())))
        out = {'combo_dev': dev.numpy()}
        for name, s in systems.items():
            with torch.no_grad():
                pc, agg, preds = closed_loop_free_run_rms_batch(s, sdl, bank, rows, return_pred=True)
            e = [np.asarray(sd.y)[k0:] - np.asarray(p) for sd, p in zip(sdl, preds)]
            band = np.stack([band_rms(x, cfg.fs_new_hz) for x in e])
            out[name + '_pc'], out[name + '_agg'], out[name + '_band'] = np.asarray(pc), np.asarray(agg), band
            print('  %-14s val %.4e m | Y per record %s' % (name, np.mean(agg), ' '.join('%.2e' % v for v in np.asarray(pc)[:, 2])))
        # captured fraction per band, Y, multisine val records (the first four), against the oracle:
        # (baseline true - model) / (oracle noabs ... ) is not additive; report energies instead
        ms = slice(0, 4)
        print('  Y band RMS, mean over the 4 multisine val records [m] (oracle full / noabs / bt from phase B):')
        for j, (lo, hi) in enumerate(BANDS):
            vals = [out[n + '_band'][ms, j, 2].mean() for n in systems]
            print('    %6.0f-%-6.0f  %s | oracle %.2e / %.2e / %.2e' % (
                lo, hi, '  '.join('%s %.2e' % (n[:6], v) for n, v in zip(systems, vals)),
                orc['full'][ms, j, 2].mean(), orc['noabs'][ms, j, 2].mean(), orc['bt'][ms, j, 2].mean()))
        print('  routes VA-T1/T2 Y: model %s | static %s | oracle full %s noabs %s bt %s' % (
            ' '.join('%.2e' % v for v in out['model_pc'][4:, 2]),
            ' '.join('%.2e' % v for v in out.get('static_only_pc', out['model_pc'])[4:, 2]),
            ' '.join('%.2e' % v for v in orc_rms['full'][4:, 2]), ' '.join('%.2e' % v for v in orc_rms['noabs'][4:, 2]),
            ' '.join('%.2e' % v for v in orc_rms['bt'][4:, 2])), flush=True)
        np.savez(os.path.join(HERE, 'outputs', 'e_ckpt_%s_%s.npz' % (job, kind)), **out)


if __name__ == '__main__':
    tags = sys.argv[1:] or sorted(os.path.basename(p)[len('SSE_Interconnect_Composed_'):-4]
                                  for p in glob.glob(os.path.join(CK, '*.pth')))
    main(tags)
