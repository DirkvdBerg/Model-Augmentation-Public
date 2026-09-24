"""Zero-update check of the Telica arm (ET-012, ET-013): build the telica-real smoke model, swap the
G4 encoder in through et_hook, report the frozen / trainable parameter counts, and run one
closed-loop validation forward (no gradient, no optimizer step). Needs ET_NA_NB and ET_G2_ENCODER.
"""
__project_origin__ = "added"

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'vendor_telica', 'pipeline'))
sys.path.insert(0, os.path.join(HERE, 'vendor_telica'))
os.environ.setdefault('TELICA_SMOKE', '1')
import telica_augment as ta                                      # noqa: E402
import torch                                                     # noqa: E402


def main():
    cfg = ta.make_cfg()
    data = ta.load_datasets_real(cfg, max_records=2, max_len=6000)
    norm = ta.frozen_norm(cfg)
    hp = cfg.hp
    fs = ta.build_model_real(hp, cfg, data, norm, baseline_tag='_a2')
    import et_hook
    fs = et_hook.swap(fs, cfg, norm, hp, os.environ['ET_G2_ENCODER'])
    n_req = sum(p.numel() for p in fs.encoder.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in fs.encoder.parameters())
    print(f'[check_hook] cfg nf {cfg.nf} ({cfg.nf_seconds * 1e3:.1f} ms), na_nb {hp["na_nb"]}; encoder '
          f'parameters {n_all}, requires_grad {n_req}')
    fs.simulator = ta.build_closed_loop_telica(fs, norm, cfg, train_files=data.train_names,
                                               val_files=data.val_names, val_data=data.val_ckpt_data)
    with torch.no_grad():
        v = fs.simulator.validation_error(fs, data.val_ckpt_data)
    print(f'[check_hook] closed-loop validation sim-RMS (no update): {float(v):.4e} m; '
          f'optimizer steps taken: 0')


if __name__ == '__main__':
    main()
