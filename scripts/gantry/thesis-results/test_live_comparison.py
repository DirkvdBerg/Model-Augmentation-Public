"""Focused CPU preflight: real lowpass data, nf=400, no full validation.

Run with the GraduationProject Python. Checks both architectures, matched
plain/PS2 initialization, baseline equivalence, finite updates, and a shortened
real PS2 opening. Does not submit jobs or run full training/validation.
"""
import importlib
import json
import os
import runpy
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

HELPERS = runpy.run_path(str(Path(__file__).with_name('test_ps2_pipeline.py')))


def main():
    for architecture in ('resnet', 'mlp'):
        os.environ['THESIS_ARCH'] = architecture
        xavier_draws = []
        initialize_xavier = torch.nn.init.xavier_uniform_

        def capture_xavier(tensor, gain=1.0, **kwargs):
            result = initialize_xavier(tensor, gain=gain, **kwargs)
            xavier_draws.append((result.detach().clone(), gain))
            return result

        with patch('torch.nn.init.xavier_uniform_', side_effect=capture_xavier):
            cfg, data, norm, plain = HELPERS['build']('u')
        os.environ['THESIS_PS2'] = '1'
        cfg_ps2 = importlib.reload(importlib.import_module('gantry_interconnect_dynamic')).CFG
        from gantry_dynamic.model import build_model
        from gantry_dynamic.controller import build_closed_loop
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        ps2 = build_model(cfg_ps2.hp, cfg_ps2, data, norm)
        ps2.simulator = build_closed_loop(ps2, norm, cfg_ps2,
                                         train_files=data.train_files, val_files=data.val_files,
                                         val_data=data.val_ckpt_data, verbose=False)
        assert cfg.nf == 400 and cfg.mode == 'thesis_taf_lowpass_noise'
        assert not cfg.obc and not cfg.orth and not cfg.ps2 and cfg_ps2.ps2
        from gantry_dynamic.data import THESIS_DATASETS
        assert THESIS_DATASETS[cfg.mode] == 'Coulomb-tanh-and-MSD-lowpass-noise'
        assert cfg.wa_encoder_init == 'xavier_uniform_gain1'
        joint = torch.cat((plain.encoder.Wa_psi_y, plain.encoder.Wa_psi_u), dim=1)
        draws = [(value, gain) for value, gain in xavier_draws if value.shape == joint.shape]
        assert len(draws) == 1 and draws[0][1] == 1.0
        assert torch.equal(draws[0][0], joint), 'encoder must split one complete Xavier draw'
        legacy_cfg = replace(cfg, wa_encoder_init='legacy_kaiming')
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        legacy = build_model(legacy_cfg.hp, legacy_cfg, data, norm)
        for key in ('Wb_psi_y', 'Wb_psi_u'):
            assert torch.equal(getattr(plain.encoder, key), getattr(legacy.encoder, key))
        del legacy
        a, b = HELPERS['snapshot'](plain), HELPERS['snapshot'](ps2)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        ann = plain.hfn.connected_blocks[plain._ann_ix]
        physical = [j for j, i in enumerate(cfg.ann_route_ix) if i < cfg.nx_phys]
        added = [j for j, i in enumerate(cfg.ann_route_ix) if i >= cfg.nx_phys]
        final = ann.net.net[-1]
        assert not torch.count_nonzero(final.bias)
        assert not torch.count_nonzero(final.weight[physical])
        live = ann.net.W_a if architecture == 'resnet' else final.weight
        assert not torch.count_nonzero(live[physical])
        assert torch.count_nonzero(live[added])
        assert float(live[added].abs().max()) <= live.shape[1] ** -0.5
        if architecture == 'resnet':
            assert not torch.count_nonzero(final.weight)
        else:
            assert not hasattr(ann.net, 'W_a')
        probe = torch.randn(4, ann.net.n_in, dtype=live.dtype)
        assert not torch.count_nonzero(ann.net(probe)[:, physical])
        assert torch.count_nonzero(ann.net(probe)[:, added])
        spec = HELPERS['short_spec'](cfg.seed)
        phase = HELPERS['opening'](cfg_ps2, data, ps2, spec)
        phase._start(ps2)
        opening_horizon = phase._M
        phase._M = cfg.nf
        batch = phase._sample(ps2, phase.fit_ix, 4, np.random.default_rng(4))
        phase._M = opening_horizon

        def loss(fs):
            return fs.loss(batch['uh'], batch['yh'], batch['uf'], batch['yf'],
                           ctrl_ix=batch['ctrl_ix'], nf=cfg.nf)

        with torch.no_grad():
            initial = loss(plain)
            gate = ann.out_gate.clone()
            ann.out_gate.zero_()
            baseline = loss(plain)
            ann.out_gate.copy_(gate)
        assert torch.equal(initial, baseline), (architecture, initial, baseline)
        for name, fs, opening in (('plain', plain, None), ('ps2', ps2, phase)):
            live_parameter = (fs.hfn.connected_blocks[fs._ann_ix].net.W_a
                              if architecture == 'resnet'
                              else fs.hfn.connected_blocks[fs._ann_ix].net.net[-1].weight)
            gradient_maxima = {'g': 0.0, 'Wy': 0.0, 'Wu': 0.0}
            for step in range(4):
                fs.optimizer.zero_grad(set_to_none=True)
                value = loss(fs)
                assert torch.isfinite(value), (architecture, name, step)
                value.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all()
                           for group in fs.optimizer.param_groups for p in group['params'])
                gradients = {'g': live_parameter.grad[added],
                             'Wy': fs.encoder.Wa_psi_y.grad, 'Wu': fs.encoder.Wa_psi_u.grad}
                for key, gradient in gradients.items():
                    magnitude = 0.0 if gradient is None else float(gradient.abs().max())
                    if step == 0:
                        assert magnitude == 0.0, (architecture, name, key, magnitude)
                    else:
                        gradient_maxima[key] = max(gradient_maxima[key], magnitude)
                fs.optimizer.step()
                if opening is not None:
                    opening.after_step(fs)
            assert torch.isfinite(loss(fs))
            assert all(torch.isfinite(v).all() for v in HELPERS['snapshot'](fs).values())
            assert all(value > 0 for value in gradient_maxima.values()), gradient_maxima
            print('PASS gradients', architecture, name, gradient_maxima, flush=True)
        assert phase.status == 'completed', phase.status
        from gantry_dynamic.config import config_json_dict
        from gantry_dynamic.training import load_checkpoint
        import inspect_checkpoint
        with tempfile.TemporaryDirectory(prefix='live-comparison-') as directory:
            plain.checkpoint_save_system(name='_livecomparison', directory=directory)
            paths = list(Path(directory).glob('*.pth'))
            assert paths, 'checkpoint not saved'
            checkpoint_path = paths[0]
            metadata = config_json_dict(cfg)
            Path(directory, 'config.json').write_text(json.dumps(metadata), encoding='utf-8')
            # Conflicting inherited architecture must not influence reconstruction.
            os.environ['THESIS_ARCH'] = 'mlp' if architecture == 'resnet' else 'resnet'
            restored_cfg = inspect_checkpoint.checkpoint_config(checkpoint_path, 'u', 1, 2)
            assert restored_cfg.augmentation_init == cfg.augmentation_init
            assert restored_cfg.mode == cfg.mode
            restored = build_model(restored_cfg.hp, restored_cfg, data, norm)
            load_checkpoint(restored, str(checkpoint_path), joint_estimation=True, start_phase='lbfgs')
            actual = HELPERS['snapshot'](restored)
            expected = HELPERS['snapshot'](plain)
            assert all(torch.equal(actual[k], expected[k]) for k in expected)
            try:
                inspect_checkpoint.checkpoint_config(checkpoint_path, seed=2)
            except ValueError:
                pass
            else:
                raise AssertionError('conflicting seed was accepted')
            Path(directory, 'config.json').unlink()
            try:
                inspect_checkpoint.checkpoint_config(checkpoint_path)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('missing metadata was accepted')
        print('PASS checkpoint reconstruction and exact reload', architecture, flush=True)
        print('PASS', architecture, 'matched initialization, baseline equivalence, '
              'nf=400 finite updates, shortened PS2 completed', flush=True)
    print('ALL PASS (focused checks; no full validation)', flush=True)


if __name__ == '__main__':
    main()
