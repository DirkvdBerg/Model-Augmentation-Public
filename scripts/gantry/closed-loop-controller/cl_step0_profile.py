"""MIGRATION STEP 0: where does one closed-loop training step actually spend its time?

`PLAN-move-to-model-augmentation.md` section 3.8 states a cost table as PRIORS and says so:
`hfn` dominant, controller step "a few percent", `bank.gather` and the encoder negligible. Two
optimisations are conditional on that table being right (block-diagonal `Cfb` storage, fusing the
four controller einsums), so the table is measured here before anything is optimised. If the
controller step is 1 % of a training step, both are dead and the plan says to skip them.

WHAT IS MEASURED, and why in this order
---------------------------------------
1. **Wall clock, closed loop vs open loop, same batch, forward + backward.** The open-loop arm is
   the production loss with `simulator = None`, i.e. the identical code path with the loop removed.
   READ THIS DIFFERENCE CAREFULLY: it is NOT the controller's cost. The open loop takes `y` from
   `hfn`'s own output while the closed loop calls `Interconnect.output_only` to get `y` BEFORE
   forming `u`, so the difference is controller + output_only + the residual and input adds. At
   batch 256 that difference is 15.2 % of the forward while the controller step ALONE is 2.6 %
   (item 2), and the gap between those two numbers is almost entirely `output_only`. Quoting this
   difference as "the controller costs 15 %" is the mistake this note exists to prevent.
2. **A controller-only microbenchmark**, `nf` calls of `ControllerBank.step` at the same batch
   size, on the same tensors, forward and backward. This bounds what the 3.8 optimisations could
   possibly buy: they cannot make the controller cheaper than zero, so the wall-clock saving from
   fusing the einsums is at most this number, and in practice a fraction of it.
3. **cProfile, for attribution only.** It says WHERE inside the step the time goes (which is what
   the priors table claims), but its per-call overhead is large next to a matmul on a batch of 32,
   so its absolute times are NOT comparable with 1 and 2 and are reported as shares only.

`nf = 400` sequential `hfn` calls per window is inherent to a recurrent rollout and no part of the
migration changes it, so the question is never "how fast is the step" but "what fraction of it is
touchable".

BATCH SIZE, and it matters more than expected. Production is 256 (config.py:70). Measured at both:

    item                          batch 32   batch 256
    A/B difference, forward only     8.1 %     15.2 %
    bank.step alone (the ceiling)    7.6 %      2.6 %
    ControllerBank.step tottime      0.8 %      0.4 %

The two move in OPPOSITE directions, which is the whole finding: the controller's own cost falls
as a share when FLOPs grow 8x and dispatch count does not, while the closed-loop overhead rises
because `output_only` scales with the batch like everything else. Any share quoted from a batch-32
run is an upper bound on the controller and a LOWER bound on output_only.

Usage:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 python -u cl_step0_profile.py
"""
__project_origin__ = "added"

import cProfile
import dataclasses
import io
import json
import os
import pstats
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
for p in (REPO, GANTRY, HERE, os.path.join(GANTRY, 'drift-demo'),
          os.path.join(GANTRY, 'msd-offset')):
    if p not in sys.path:
        sys.path.insert(0, p)

import demo_common as dm                                                  # noqa: E402
from demo_common import CFG                                               # noqa: E402
from gantry_dynamic.data import load_traj, TRAIN_FILES                    # noqa: E402
import cl_validation as CV                                                # noqa: E402
import deepSI                                                             # noqa: E402
from gantry_dynamic.data import VAL_FILES                                 # noqa: E402
from cl_pipeline import build_closed_loop                                 # noqa: E402

BATCH = int(os.environ.get('CL_BATCH', 32))
REPS = int(os.environ.get('CL_REPS', 5))
RECORD = 'T10_aprbs_60'
OUT = os.path.join(HERE, 'runs', 'step0_profile.json')
t0 = time.time()

print('=' * 96)
print('MIGRATION STEP 0: profile of one closed-loop training step')
print('=' * 96)

cfg = dataclasses.replace(CFG, seed=0, ann_route_ix=tuple(range(8)), lr=1e-7)
fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=True)
nx = cfg.nx_phys + cfg.nx_ann
print('nf %d   batch %d   nx %d   threads %d'
      % (cfg.nf, BATCH, nx, torch.get_num_threads()))

train_names = [f[:-4] for f in TRAIN_FILES]
val_data = deepSI.System_data_list([load_traj(f, cfg) for f in VAL_FILES])
fs.simulator = build_closed_loop(fs, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
                                   val_data=val_data, verbose=False)
bank = fs.simulator.bank

# ---- one fixed batch of production-shaped windows, all from one record ------------------------
sd = load_traj(RECORD + '.mat', cfg)
rv = lambda a: np.asarray(a).ravel()                                      # noqa: E731
un = ((sd.u - rv(fs.norm.u0)) / rv(fs.norm.ustd)).astype(cfg.dtype_np)
yn = ((sd.y - rv(fs.norm.y0)) / rv(fs.norm.ystd)).astype(cfg.dtype_np)
starts = np.linspace(K0, len(un) - cfg.nf - 2, BATCH).astype(int)
uh, yh, uf, yf = [], [], [], []
for s in starts:
    a, b = CV.encoder_window(un, yn, int(s), na, nb, na_r, nb_r, cfg.dtype_pt)
    uh.append(a); yh.append(b)
    uf.append(un[s:s + cfg.nf])
    yf.append(yn[s:s + cfg.nf])
T = lambda A: torch.as_tensor(np.stack(A), dtype=cfg.dtype_pt)            # noqa: E731
UH, YH, UF, YF = torch.cat(uh, 0), torch.cat(yh, 0), T(uf), T(yf)
RIX = torch.full((BATCH,), fs.simulator.train_ctrl_rows[train_names.index(RECORD)],
                 dtype=torch.long)
print('batch built: ufuture %s   [%.0fs]' % (tuple(UF.shape), time.time() - t0), flush=True)


PARAMS = list(fs.hfn.parameters()) + list(fs.encoder.parameters())


def timed(fn, reps=REPS):
    """Median wall clock of fn over reps, after one warm-up. Gradients zeroed each call."""
    ts = []
    for i in range(reps + 1):
        for p in PARAMS:
            p.grad = None
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
        if i:
            ts.append(dt)
    return float(np.median(ts)), ts


def timed_pairs(fn_a, fn_b, reps=REPS):
    """Interleaved A/B timing: a, b, a, b, ... so a machine that drifts affects both equally.

    Timing two arms in two separate blocks measures the difference between them PLUS whatever
    the CPU did between the blocks. On this box that second term is large enough to make the
    controller's cost come out negative, i.e. the open loop apparently slower than the closed
    loop, which is impossible: the closed loop runs the identical rollout and then does more.
    Pairing removes the common-mode drift and the per-pair difference is the quantity of
    interest, so the spread of those differences is also the error bar on it.
    """
    da, db = [], []
    for i in range(reps + 1):
        for p in PARAMS:
            p.grad = None
        t = time.perf_counter(); fn_a(); ta = time.perf_counter() - t
        for p in PARAMS:
            p.grad = None
        t = time.perf_counter(); fn_b(); tb = time.perf_counter() - t
        if i:
            da.append(ta); db.append(tb)
    d = np.array(da) - np.array(db)
    return float(np.median(da)), float(np.median(db)), float(np.median(d)), da, db


def closed_fb():
    fs.loss(UH, YH, UF, YF, ctrl_ix=RIX).backward()


def open_fb():
    _open_loop_loss().backward()


def closed_f():
    fs.loss(UH, YH, UF, YF, ctrl_ix=RIX)


def _open_loop_loss():
    """The IDENTICAL loss with the loop removed and nothing else changed.

    MIGRATION step 7: this was `cl_fitsys.loss_open_loop`, a hand-written twin of the closed-loop
    loss that existed only so the two could be compared. It is now the production loss with the
    simulator detached, which is the same statement with no second implementation to keep in step.
    """
    sim, fs.simulator = fs.simulator, None
    try:
        return fs.loss(UH, YH, UF, YF)
    finally:
        fs.simulator = sim


# ---- 1. the decisive measurement -------------------------------------------------------------
print('\n1. WALL CLOCK, no profiler attached, interleaved (median of %d pairs)' % REPS)
t_cl, t_ol, t_diff, all_cl, all_ol = timed_pairs(closed_fb, open_fb)
with torch.no_grad():
    t_clf, _ = timed(closed_f)
print('  closed loop, forward + backward : %8.3f s   %s' % (t_cl, ['%.3f' % v for v in all_cl]))
print('  open loop,   forward + backward : %8.3f s   %s' % (t_ol, ['%.3f' % v for v in all_ol]))
_d = np.array(all_cl) - np.array(all_ol)
print('  per-pair difference             : %8.3f s   %s'
      % (t_diff, ['%+.3f' % v for v in _d]))
print('     spread of that difference    : IQR [%+.3f, %+.3f] s over %d pairs'
      % (np.percentile(_d, 25), np.percentile(_d, 75), len(_d)))
print('  closed loop, forward only (no grad): %5.3f s' % t_clf)

# Forward only, also interleaved. This is the low-noise version of the same comparison: the
# backward pass is more than half the step and is the noisiest part of it, so removing it
# shrinks the error bar on a difference that is a few percent of the total. The controller's
# backward cost is proportional to the graph it built, so the forward share is the right
# quantity to reason about and the full-step number above is the one to quote.
with torch.no_grad():
    def closed_f_ng():
        fs.loss(UH, YH, UF, YF, ctrl_ix=RIX)

    def open_f_ng():
        _open_loop_loss()

    f_cl, f_ol, f_diff, fa, fb = timed_pairs(closed_f_ng, open_f_ng, reps=max(REPS, 9))
_fd = np.array(fa) - np.array(fb)
print('  forward only: closed %.3f s   open %.3f s   diff %+.3f s = %.1f %%   '
      'IQR [%+.3f, %+.3f]'
      % (f_cl, f_ol, f_diff, 100 * f_diff / f_cl,
         np.percentile(_fd, 25), np.percentile(_fd, 75)))
ctrl_share = t_diff / t_cl
print('  -> controller + residual + gather = %.3f s = %.1f %% of a closed-loop step'
      % (t_diff, 100 * ctrl_share))

# ---- 2. what the 3.8 optimisations could possibly buy -----------------------------------------
print('\n2. CONTROLLER MICROBENCHMARK, %d steps at batch %d (the ceiling on 3.8)' % (cfg.nf, BATCH))
ctrl = bank.gather(RIX)
e_dummy = torch.randn(BATCH, cfg.ny, dtype=cfg.dtype_pt, requires_grad=True)


def ctrl_only():
    xc = bank.zero_state(BATCH, dtype=cfg.dtype_pt)
    acc = 0.0
    for _ in range(cfg.nf):
        u_fb, xc = bank.step(xc, e_dummy, ctrl)
        acc = acc + u_fb.sum()
    acc.backward()


t_ctrl, _ = timed(ctrl_only)
print('  %d x bank.step, forward + backward : %.3f s  = %.1f %% of a closed-loop step'
      % (cfg.nf, t_ctrl, 100 * t_ctrl / t_cl))
t_gather, _ = timed(lambda: bank.gather(RIX), reps=20)
print('  bank.gather, once per batch        : %.6f s = %.3f %% of a closed-loop step'
      % (t_gather, 100 * t_gather / t_cl))
t_enc, _ = timed(lambda: fs.encoder(UH, YH))
print('  encoder, once per batch            : %.6f s = %.3f %% of a closed-loop step'
      % (t_enc, 100 * t_enc / t_cl))

# ---- 3. attribution ---------------------------------------------------------------------------
print('\n3. cPROFILE, ONE closed-loop step (shares only; per-call overhead inflates absolutes)')
pr = cProfile.Profile()
for p in list(fs.hfn.parameters()) + list(fs.encoder.parameters()):
    p.grad = None
pr.enable()
closed_fb()
pr.disable()
buf = io.StringIO()
pstats.Stats(pr, stream=buf).sort_stats('tottime').print_stats(22)
prof_text = buf.getvalue()
print(prof_text[prof_text.index('ncalls'):][:3200])

st = pstats.Stats(pr)
total = st.total_tt

# TOTTIME, not cumtime. cumtime attributes a callee's time to every ancestor, so a table built
# from it double-counts: a first attempt had `backward` at 197 % and an adapter's `__call__` at
# 43 %, the latter because the substring matched two different classes in one file, one of which
# CONTAINED the whole hfn forward. tottime is the time in the function itself and sums to the
# profiled total, so the shares below are a partition and can be read as such.


def tot_of(pred):
    """pred(file, func, line) -> bool. Summed tottime and call count over matching entries."""
    tot = 0.0
    calls = 0
    for (fn, ln, func), (cc, nc, tt, ct, _) in st.stats.items():
        if pred(fn, func, ln):
            tot += tt
            calls += nc
    return tot, calls


# MIGRATION step 7: the ModelStep and AffineOutput rows are GONE, not zeroed. Those classes were
# deleted along with the adapters they belonged to, and a row that always prints 0.0 reads as
# "this costs nothing" rather than "this no longer runs", which is worse than no row at all. The
# accessor they provided is `Interconnect.output_only`, which has its own row below and is the
# item this table exists to price: at batch 256 it is about 5 % of a step, against the controller
# step's 0.4 %, so it, not the controller, is where any future optimisation would go.
rows = [
    ('backward (autograd engine)',
     tot_of(lambda f, n, l: 'run_backward' in n)),
    ('model blocks (blocks.py)',
     tot_of(lambda f, n, l: f.endswith('blocks.py'))),
    ('Interconnect.forward',
     tot_of(lambda f, n, l: f.endswith('interconnect.py') and n == 'forward')),
    ('Interconnect.output_only (y = h(x))',
     tot_of(lambda f, n, l: f.endswith('interconnect.py') and n == 'output_only')),
    ('ControllerBank.step',
     tot_of(lambda f, n, l: f.endswith('closed_loop.py') and n == 'step')),
    ('ControllerBank.gather',
     tot_of(lambda f, n, l: f.endswith('closed_loop.py') and n == 'gather')),
    ('closed_loop_rollout loop body',
     tot_of(lambda f, n, l: f.endswith('closed_loop.py') and n == 'closed_loop_rollout')),
    ('torch.baddbmm + bmm (controller)',
     tot_of(lambda f, n, l: n in ('baddbmm', 'bmm'))),
    ('torch.matmul',
     tot_of(lambda f, n, l: n == 'matmul')),
    ('nn.Module.__getattr__',
     tot_of(lambda f, n, l: n == '__getattr__' and 'torch' in f)),
]
print('\n  profiled total %.3f s (vs %.3f s unprofiled, overhead %.1fx). tottime shares.'
      % (total, t_cl, total / max(t_cl, 1e-12)))
print('  %-34s %10s %12s %8s' % ('item', 'tot [s]', 'calls', 'share'))
for label, (ct, nc) in rows:
    print('  %-34s %10.3f %12d %7.1f %%' % (label, ct, nc, 100 * ct / total))

# ---- verdict -----------------------------------------------------------------------------------
print('\n' + '=' * 96)
print('VERDICT on plan section 3.8')
print('=' * 96)
print('priors: hfn dominant / controller "a few percent" / gather negligible / encoder negligible')
print('measured: controller path %.1f %% (interleaved wall clock, forward + backward), '
      'gather %.3f %%, encoder %.3f %%'
      % (100 * ctrl_share, 100 * t_gather / t_cl, 100 * t_enc / t_cl))
print('          microbenchmark ceiling on any controller optimisation: %.1f %%'
      % (100 * t_ctrl / t_cl))
if ctrl_share < 0.02:
    print('-> The controller step is under 2 %% of a training step. The block-diagonal Cfb storage')
    print('   and the einsum fusion (plan 3.8) are NOT worth writing: their combined ceiling is')
    print('   below the run-to-run scatter. Port cl_controller.step unchanged and stop there.')
elif ctrl_share < 0.10:
    print('-> The controller step is a few percent, as the prior said. The einsum fusion is the')
    print('   cheaper of the two 3.8 items and is the only one worth considering; neither is')
    print('   required for correctness and both stay optional.')
else:
    print('-> The controller step is a material share. Both 3.8 items are justified; do them')
    print('   AFTER the seams are in and the reference sets pass, never before.')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(dict(batch=BATCH, nf=cfg.nf, reps=REPS, record=RECORD,
               threads=torch.get_num_threads(),
               t_closed_fb=t_cl, t_open_fb=t_ol, t_closed_f=t_clf,
               t_controller_only=t_ctrl, t_gather=t_gather, t_encoder=t_enc,
               controller_share=ctrl_share,
               pairs_closed=all_cl, pairs_open=all_ol,
               fwd_closed=f_cl, fwd_open=f_ol, fwd_diff=f_diff,
               fwd_share=f_diff / f_cl, fwd_pairs_closed=fa, fwd_pairs_open=fb,
               profile_total=total,
               profile_rows={k: dict(cum=v[0], calls=v[1]) for k, v in rows},
               seconds=time.time() - t0),
          open(OUT, 'w'), indent=2)
print('\nwrote %s   [%.0fs]' % (OUT, time.time() - t0))
