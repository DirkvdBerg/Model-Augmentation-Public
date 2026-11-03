"""The state of a thesis deepSI dump as plain tensors, for the clean side of a D-295 Adam resume.

    python -u extract_dump.py SRC_PTH DST_PT

The dump pickles the mainline system, so it is read here on the mainline root; DST_PT holds the hfn and encoder
state_dicts, the Adam state_dict and the progress fields, which clean-up-verification/convert_run.convert_dump turns
into a dump Clean/ resumes from. The Adam state is read as mainline's resolve_checkpoint reads it (`optimizer`, or
`_optimizer` since the optimizer property, D-275).
"""
import sys

import hcommon as H

if len(sys.argv) != 3:
    sys.exit(__doc__)
H.select_side('mainline')
import torch  # noqa: E402

src, dst = sys.argv[1:]
d = torch.load(src, map_location='cpu', weights_only=False)
opt = d.get('optimizer') or d.get('_optimizer')
if opt is None:
    sys.exit('%s holds no Adam state' % src)
out = dict(hfn=d['hfn'].state_dict(), encoder=d['encoder'].state_dict(), optimizer=opt.state_dict())
out.update({k: d[k] for k in ('epochs_before', 'epoch_counter', 'bestfit', 'ps2_opening') if k in d})
# mainline counts the epochs done as round(epoch_counter), Clean/ as round(epochs_before + epoch_counter); the two
# agree only for a dump of a run that was never resumed
if float(out.setdefault('epochs_before', 0)) != 0:
    sys.exit('%s has epochs_before = %s: mainline and Clean/ would count its epochs differently'
             % (src, out['epochs_before']))
torch.save(out, dst)
H.assert_roots()
print('EXTRACT_DUMP DONE %s -> %s fields=%s' % (src, dst, sorted(out)), flush=True)
