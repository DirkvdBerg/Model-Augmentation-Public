"""D-295: merge per-run l4_loader.py outputs of one side into the one folder compare_l4.py reads.

    python -u merge_l4.py DST SRC_R1 SRC_R2 ...

Each SRC is an l4_loader.py --out of one run (--runs R<n>) of the same side, staging and Clean/ tree. DST gets
every output file and one summary.json with the union of runs, checks, eval_check and outputs; the output hashes
are checked again on the way. Refused when the sides, staging, lstsq driver or clean manifests of the SRCs differ.
"""
import json
import os
import shutil
import sys

import hcommon as H

if len(sys.argv) < 3:
    sys.exit(__doc__)
dst, srcs = os.path.abspath(sys.argv[1]), [os.path.abspath(s) for s in sys.argv[2:]]
if os.path.exists(dst):
    sys.exit('%s exists' % dst)
sums = []
for s in srcs:
    with open(os.path.join(s, 'summary.json')) as fh:
        sums.append(json.load(fh))
for key in ('side', 'staging', 'lstsq_driver', 'clean_tree', 'runs'):
    if any(x[key] != sums[0][key] for x in sums):
        sys.exit('REFUSED: the inputs differ in %s' % key)
merged = {k: sums[0][k] for k in ('side', 'staging', 'runs', 'lstsq_driver', 'clean_tree')}
merged.update(checks={}, eval_check={}, outputs={}, merged_from=srcs)
os.makedirs(dst)
for s, x in zip(srcs, sums):
    for f, h in x['outputs'].items():
        if f in merged['outputs']:
            sys.exit('REFUSED: %s in two inputs' % f)
        if H.sha256_file(os.path.join(s, f)) != h:
            sys.exit('REFUSED: %s differs from its summary.json hash' % os.path.join(s, f))
        shutil.copy2(os.path.join(s, f), os.path.join(dst, f))
        merged['outputs'][f] = h
    for k in ('checks', 'eval_check'):
        merged[k].update(x[k])
with open(os.path.join(dst, 'summary.json'), 'w') as fh:
    json.dump(merged, fh, indent=1, default=str)
print('MERGE_L4 DONE %s: %d inputs, runs %s, %d files' % (dst, len(srcs), sorted(merged['checks']),
                                                            len(merged['outputs'])))
