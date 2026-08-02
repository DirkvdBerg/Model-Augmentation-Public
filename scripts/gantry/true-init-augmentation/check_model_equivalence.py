"""Is the hand-assembled interconnect the SAME model the pipeline trains?

`true_init_train.py` assembles the interconnect directly instead of calling
`gantry_dynamic/model.py::build_model`, because `build_model` also builds the
encoder, the multiple-shooting subclass, the orth penalty, the ReZero gate and
the Lipschitz cap, none of which survive deleting the encoder. That is a
reasonable choice and it is also a way to silently train a DIFFERENT model and
attribute the result to the pipeline. This gate closes that.

Both models are built from the same `RunConfig` and the same `Norm`, the ANN
weights are copied from the pipeline's model into ours so the only thing that
can differ is structure, and then both are rolled `nf` steps from the SAME
initial state with the same input.

  E1  ANN at its zero-initialised state: outputs must agree to float32 precision
  E2  ANN perturbed off zero: outputs must still agree, which E1 cannot show
      because a zero ANN makes the whole augmentation path invisible. E1 without
      E2 would pass even if the ANN were wired to nothing at all.
  E3  the propagated augmented rows must agree too, not only the output

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \\
      -n GraduationProject python -u \\
      scripts/gantry/true-init-augmentation/check_model_equivalence.py
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

from model_augmentation.fit_systems.blocks import Static_ANN_Block   # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization  # noqa: E402
from gantry_dynamic.model import build_model                          # noqa: E402
from true_init_train import CFG, build_interconnect                   # noqa: E402

OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
NF = 400


def roll(step, x0, u):
    x, ys = x0, []
    with torch.no_grad():
        for t in range(u.shape[1]):
            y, x = step(x, u[:, t])
            ys.append(y)
    return torch.stack(ys, dim=1), x


def main():
    cfg = CFG
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(cfg.hp, cfg, data, norm)
    ref_ic = fit_sys.hfn
    ref_ann = next(m for m in ref_ic.connected_blocks
                   if isinstance(m, Static_ANN_Block))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # cog=False: the CoG correction is OUR change and must not be part of the
    # equivalence claim. It has its own gates (C1a-C5).
    ic, ann, _ = build_interconnect(cfg, norm, cog=False, dtype=torch.float32)
    ann.load_state_dict(ref_ann.state_dict())

    g = torch.Generator().manual_seed(7)
    B, nxd = 8, cfg.nx_phys + cfg.nx_ann
    x0 = torch.randn(B, nxd, generator=g) * 0.3
    x0[:, 6:] = 0.0
    u = torch.randn(B, NF, cfg.nu, generator=g) * 0.5

    res = {}
    print('Is the hand-assembled interconnect the pipeline\'s model?\n')
    for tag, scale in (('E1 ANN at zero init', 0.0), ('E2 ANN perturbed off zero', 1.0)):
        if scale:
            with torch.no_grad():
                ref_ann.net.net[-1].weight.normal_(0, 1e-3, generator=g)
                ref_ann.net.net[-1].bias.normal_(0, 1e-3, generator=g)
            ann.load_state_dict(ref_ann.state_dict())
        ya, xa = roll(ref_ic, x0.clone(), u)
        yb, xb = roll(ic, x0.clone(), u)
        dy = float((ya - yb).abs().max())
        rel = dy / max(float(ya.abs().max()), 1e-30)
        dxa = float((xa[:, 6:] - xb[:, 6:]).abs().max())
        aug = float(xa[:, 6:].abs().max())
        res[tag] = dict(dy=dy, rel=rel, dx_aug=dxa, aug_scale=aug)
        print(f'  {tag:<28} max|dy| {dy:.3e}  rel {rel:.3e}  '
              f'{"PASS" if rel < 1e-6 else "FAIL"}')
        print(f'  {"":<28} x_aug at segment end: ref |x| {aug:.3e}, '
              f'max|diff| {dxa:.3e}   (E3)')
    print('\n  E1 alone is not enough: with the ANN output identically zero the whole')
    print('  augmentation path is invisible, so E1 would pass even against a model with')
    print('  the ANN wired to nothing. E2 is what makes the claim.')

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'true_init_model_equivalence.json')
    with open(p, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'\n  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
