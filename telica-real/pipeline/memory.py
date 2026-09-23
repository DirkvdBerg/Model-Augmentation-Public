"""Process memory without psutil (not in the env; handoff sec. 2), and the G6 memory probe.

rss_mb / peak_rss_mb: Windows GetProcessMemoryInfo via ctypes, Linux /proc/self/status.
mem_probe: forward + backward of the closed-loop rollout at two horizons, NO optimizer step, to
measure the autograd memory per (window x step) for the server batch estimate.
"""
import ctypes
import os
import sys

import numpy as np
import torch


def _win_counters():
    class PMC(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    # 64-bit: HANDLE is pointer-sized; without these signatures the call fails silently (g6_smoke
    # read 0 MB)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    ok = psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    if not ok:
        raise OSError('GetProcessMemoryInfo failed')
    return c.WorkingSetSize / 2 ** 20, c.PeakWorkingSetSize / 2 ** 20


def _linux(key):
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith(key):
                return float(line.split()[1]) / 1024
    return float('nan')


def rss_mb():
    return _win_counters()[0] if sys.platform == 'win32' else _linux('VmRSS')


def peak_rss_mb():
    return _win_counters()[1] if sys.platform == 'win32' else _linux('VmHWM')


def mem_probe(fit_sys, cfg, data, batch=8, nfs=(50, 100)):
    """Autograd memory of the closed-loop rollout per window-step (forward+backward, no step)."""
    sd = data.train_list[0]
    n = fit_sys.norm
    un = torch.as_tensor((sd.u - n.u0) / n.ustd, dtype=cfg.dtype_pt)
    yn = torch.as_tensor((sd.y - n.y0) / n.ystd, dtype=cfg.dtype_pt)
    nxd = cfg.nx_phys + cfg.nx_ann
    res = []
    for nf in nfs:
        starts = np.linspace(0, len(un) - nf - 1, batch).astype(int)
        U = torch.stack([un[s:s + nf] for s in starts]); Y = torch.stack([yn[s:s + nf] for s in starts])
        x0 = torch.zeros(batch, nxd, dtype=cfg.dtype_pt)
        ix = torch.zeros(batch, dtype=torch.long)
        fit_sys.hfn.zero_grad(set_to_none=True)
        r0 = rss_mb()
        y_pred, _ = fit_sys.simulator(fit_sys, x0, U, Y, ctrl_ix=ix)
        loss = ((y_pred - Y) ** 2).mean()
        r1 = rss_mb()
        loss.backward()
        r2 = rss_mb()
        res.append(dict(nf=nf, batch=batch, rss_before=r0, rss_after_fwd=r1, rss_after_bwd=r2))
        del y_pred, loss
    d_mb = res[1]['rss_after_fwd'] - res[0]['rss_after_fwd']
    per_ws = d_mb / (batch * (nfs[1] - nfs[0]))          # MB per window-step of retained graph
    out = dict(points=res, mb_per_window_step=per_ws)
    print(f'[mem] forward graph: {per_ws * 1024:.2f} KB per window-step '
          f'(batch {batch}, nf {nfs[0]} -> {nfs[1]}: +{d_mb:.1f} MB)', flush=True)
    fit_sys.hfn.zero_grad(set_to_none=True)
    return out
