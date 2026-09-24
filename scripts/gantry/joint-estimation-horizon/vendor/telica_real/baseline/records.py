"""Record access for G4-G6: every log of the split, loaded once, unreadable ones reported.

Uses the vendored full loader with the fast parser (checked identical to the reference parser on
the first file, run g4_survey). Returns SI arrays; nothing is cached to disk (no raw-data copies).
"""
import numpy as np

import data_index
from params import telica_params as tp
from real_data_verification.telica_loader import load_telica_log_full


def iter_records(splits=('train', 'validation', 'test'), iters=None, optional=True, log=print):
    """Yield (split, op, it, d) with d = load_telica_log_full(...) + 'u' [N] stage force (reading
    A) and 'u_fb' [N]. Unreadable or degenerate files are reported via `log` and skipped."""
    iters = tuple(data_index.ITERS_COMMON) if iters is None else tuple(iters)
    a2n = np.asarray(tp.a_to_n())
    for (s, op, it, p) in data_index.records(splits=splits, iters=iters, optional=optional):
        try:
            d = load_telica_log_full(p, engine='c')
        except Exception as exc:                                  # empty / corrupt file
            log(f'[records] SKIP unreadable {s}/{op}/{it}: {type(exc).__name__}: {exc}')
            continue
        if len(d['q1']) < 2000:
            log(f'[records] SKIP short {s}/{op}/{it}: T = {len(d["q1"])}')
            continue
        d['u'] = d['i_tot'] * a2n
        d['u_fb'] = d['i_fb'] * a2n
        d['degenerate_axes'] = [j for j in range(3) if d['i_fb'][:, j].std() == 0]
        if d['degenerate_axes']:
            log(f'[records] NOTE {s}/{op}/{it}: MF230 constant on axes {d["degenerate_axes"]}')
        yield s, op, it, d
