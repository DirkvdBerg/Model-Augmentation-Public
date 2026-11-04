"""Convert the thesis black-box run (R8) into a run folder Clean/blackbox reads: its checkpoint pickles the nets of
jan_model.py (JanEncoder, JanStateNet), which Clean/ does not have.

    python -u clean-up-verification/convert_bb.py SRC_RUN_DIR DST_RUN_DIR

JanEncoder and JanStateNet subclass deepSI's default_encoder_net and default_state_net and change only forward,
reshape in place of view (same values); their parameters and state are the base classes'. The converter loads the
checkpoint named in SRC's result.json (in SRC, its localappdata/deepSI/checkpoints, or ~/.deepSI/checkpoints) with the
thesis jan_model on sys.path, gives each of those nets its deepSI base class, and saves it into DST next to a copy of
result.json, where eval_bb.py looks first. DST must not exist; SRC is only read.
"""
import json
import os
import shutil
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# the thesis black-box code, whose jan_model.py the checkpoint pickles
THESIS_BB = os.path.join(REPO, 'scripts', 'gantry', 'blackbox-cl-trainable', 'bbcl')


def convert_bb(src, dst):
    """Convert the run folder src into dst (created here); returns the converted checkpoint path."""
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if dst.exists():
        raise FileExistsError('%s exists; the converter never writes into an existing folder' % dst)
    with open(src / 'result.json') as fh:
        name = json.load(fh)['checkpoint']
    candidates = (src / name, src / 'localappdata' / 'deepSI' / 'checkpoints' / name,
                  Path.home() / '.deepSI' / 'checkpoints' / name)
    ckpt = next((p for p in candidates if p.is_file()), None)
    if ckpt is None:
        raise FileNotFoundError('checkpoint %s not found; checked %s' % (name, ', '.join(map(str, candidates))))
    sys.path.insert(0, THESIS_BB)
    import torch
    from deepSI.fit_systems.encoders import default_encoder_net, default_state_net
    import jan_model
    base = {jan_model.JanEncoder: default_encoder_net, jan_model.JanStateNet: default_state_net}
    d = torch.load(ckpt, map_location='cpu', weights_only=False)
    swapped = []

    def swap(key, value):
        """value with every jan_model net instance and class reference (the model also stores its constructor
        arguments, e_net and hf_net_kwargs['f_net']) given its deepSI base class."""
        if isinstance(value, type) and value in base:
            swapped.append('%s: class %s -> %s' % (key, value.__name__, base[value].__name__))
            return base[value]
        if isinstance(value, torch.nn.Module):
            for m in value.modules():
                if type(m) in base:
                    swapped.append('%s: %s -> %s' % (key, type(m).__name__, base[type(m)].__name__))
                    m.__class__ = base[type(m)]
            return value
        if isinstance(value, dict):
            return {k: swap('%s.%s' % (key, k), v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(swap('%s[%d]' % (key, i), v) for i, v in enumerate(value))
        return value
    d = {k: swap(k, v) for k, v in d.items()}
    if not swapped:
        raise RuntimeError('%s holds no jan_model net; nothing to convert' % ckpt)
    dst.mkdir(parents=True)
    shutil.copy2(src / 'result.json', dst / 'result.json')
    torch.save(d, dst / name)
    sys.path.remove(THESIS_BB)
    return dst / name, swapped


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    out, swapped = convert_bb(sys.argv[1], sys.argv[2])
    print('CONVERT_BB DONE %s (%s)' % (out, '; '.join(swapped)))
