"""instrument_select.py -- Phase B0: choose the Phase B instrument on evidence, not assertion.

The raw autograd gradient in the DC direction is noise-dominated on the z=1 (marginal) dY axis:
its variance explodes ~H^3 while its mean is unresolvable (ground_truth.py: SE ~= mean at every H).
This script tests whether the LOSS-OPTIMUM / CURVATURE instrument is (1) statistically better and
(2) actually measures what our method cares about, with three falsifiable tests. It decides between
Option A (loss-landscape instrument) and the fallback (Adam training-dynamics instrument).

Why c* and kappa should be the stable quantities: the gradient at c=0 is g(H) = -kappa(H)*c*(H).
On the z=1 axis kappa(H) grows (steep valley) while c*(H) -> 0, so the GRADIENT blows up even as the
OPTIMUM converges. c*(H) and kappa(H) are the bounded, resolvable readouts; the derivative is the
worst possible one. The map c -> yhat is affine to first order in c, so the windowed loss L(c) is
essentially exactly quadratic -> a parabola fit gives c* and kappa cleanly.

FAITHFUL injection (training-matched): a constant is added to the ANN OUTPUT on the dY route
(column route_col of ann.forward, cfg.ann_route_ix), exactly where the trained DC lives, by patching
ann.forward (the interconnect calls block.forward directly, interconnect.py:92 -- nn.Module hooks do
NOT fire; lesson instrument-deepSI-at-the-called-method). Encoder init, real with-MSD data.
v11's (true-init + raw dY-state-row add) is kept as an exact cross-check.

TESTS (pre-registered):
  Test 1 (BETTER):  bootstrap relative SE of {raw gradient} vs {landscape slope, c*, kappa} vs H.
                    Predict landscape rel-SE << gradient rel-SE and BOUNDED as H grows. Falsify: it
                    also explodes.
  Test 2 (CONTROL): plant +eps on the route so the optimum sits at c*=-eps; recover it. Sizes
                    {4e-6,1e-6,3e-7} (v8-inj ladder). Predict landscape recovers -eps; gradient can't.
  Test 3 (WHAT WE WANT): does the landscape reflect the v12-proven mechanism? Read kappa(H) (does the
                    horizon pin the DC harder as H grows?) and c*(H) vs the v12 fixed DC (-4.5e-6),
                    the v12 ARTBP DC (~0), and v11's c*(400)=+2.5e-7. Two pre-registered branches:
                      truncation-bias-as-loss-optimum -> c*(400)~-4.5e-6, |c*(H)|->0, kappa grows;
                      Adam-artifact -> c*(400)~0 (flat, kappa~0 at 400) != trained DC -> pivot the
                      Phase B instrument to the Adam training-dynamics view.

Convention: data -> ./data/b0_instrument_select.npz, figures -> ./figures/b0_*.png.
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE   = os.path.dirname(os.path.abspath(__file__))
GANTRY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(GANTRY, 'drift-demo'))

import demo_common as dm
from demo_common import CFG
from gantry_dynamic.data import TRAIN_FILES
from model_augmentation.fit_systems.blocks import Static_ANN_Block

# ── one config surface ─────────────────────────────────────────────────────────
SEED      = int(os.environ.get('IS_SEED', str(CFG.seed)))
B         = int(os.environ.get('IS_B', '64'))
N_BATCHES = int(os.environ.get('IS_NBATCH', '4'))          # 4 x 64 = 256 windows
STRIDE    = int(os.environ.get('IS_STRIDE', '20'))
H_LAND    = [400, 1600, 3200]                              # landscape horizon sweep (real)
H_POS     = 400                                            # training horizon: positive control + null + v11 cross
EPS_LIST  = [4e-6, 1e-6, 3e-7]                             # planted-offset ladder (v8-inj)
CGRID     = np.linspace(-9e-6, 9e-6, 13)                   # constant-on-dY grid (parabola fit -> sub-grid c*)
NBOOT     = 200                                            # window bootstrap for SE
IY        = 5                                              # dY state index; ann output col route_col maps here
V12_FIXED_DC = -4.5e-6                                     # v12 fixed-window trained DC (dY)
V11_CSTAR_400 = 2.5e-7                                     # v11 reported real c*(400)

figDir = os.path.join(HERE, 'figures'); os.makedirs(figDir, exist_ok=True)
datDir = os.path.join(HERE, 'data');    os.makedirs(datDir, exist_ok=True)
HMAX = max(H_LAND)


# ── landscape helpers ──────────────────────────────────────────────────────────
def fit_parabola(cgrid, Lmean):
    """Quadratic fit L ~ a c^2 + b c + d. Returns (c_star, kappa=2a, valid). valid=False if not a
    positive-curvature well or the min falls outside the grid (short-window flat case)."""
    a, b, d = np.polyfit(cgrid, Lmean, 2)
    if a <= 0:
        return np.nan, 2.0 * a, False
    cstar = -b / (2.0 * a)
    valid = cgrid.min() <= cstar <= cgrid.max()
    return cstar, 2.0 * a, valid


def boot_cstar_kappa(Lwin, cgrid, nboot, rng):
    """Bootstrap over windows: Lwin is (Nwin, Ncgrid) per-window losses. Returns dict with mean c*,
    kappa, slope-at-0 and their SE, plus the pinned fraction (bootstraps giving a valid well)."""
    Nwin = Lwin.shape[0]
    cs, ks, gs = [], [], []
    j0 = int(np.argmin(np.abs(cgrid)))                     # c=0 index (grid is symmetric, includes 0)
    for _ in range(nboot):
        idx = rng.integers(0, Nwin, Nwin)
        Lm = Lwin[idx].mean(0)
        cstar, kappa, valid = fit_parabola(cgrid, Lm)
        # slope at c=0 by central finite difference on the averaged curve (same estimand as the grad)
        g0 = (Lm[j0 + 1] - Lm[j0 - 1]) / (cgrid[j0 + 1] - cgrid[j0 - 1])
        gs.append(g0)
        if valid:
            cs.append(cstar); ks.append(kappa)
    cs = np.array(cs); ks = np.array(ks); gs = np.array(gs)
    Lm_full = Lwin.mean(0)
    cstar0, kappa0, _ = fit_parabola(cgrid, Lm_full)
    return dict(cstar=cstar0, cstar_se=(cs.std() if len(cs) else np.nan),
                kappa=kappa0, kappa_se=(ks.std() if len(ks) else np.nan),
                slope=gs.mean(), slope_se=gs.std(),
                pinned_frac=len(cs) / nboot, Lmean=Lm_full)


def main():
    print(f'instrument_select (Phase B0) | seed={SEED} | {N_BATCHES}x{B}={N_BATCHES*B} windows | '
          f'H_LAND={H_LAND} | eps={EPS_LIST} | cgrid[{CGRID.min():.1e},{CGRID.max():.1e}] x{len(CGRID)}')
    fit_sys, norm, K0, na, nb, na_right, nb_right = dm.build_pipeline(cfg=CFG, verbose=True)
    for p in list(fit_sys.hfn.parameters()) + list(fit_sys.encoder.parameters()):
        p.requires_grad_(False)

    ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    nw = ann.nw
    route_col = list(int(i) for i in np.asarray(CFG.ann_route_ix).ravel()).index(IY)
    print(f'[ann] nw={nw}, dY route column={route_col} (ann_route_ix={tuple(CFG.ann_route_ix)})')

    # Patch ann.forward to add a constant on the dY route column (faithful DC injection).
    # route_const: no-grad constant for the landscape. cbox['leaf']: an autograd leaf for the gradient
    # instrument (None when unused). A mutable holder so the closure always reads the current leaf.
    route_const = torch.zeros(1, nw, 1)
    e_col = torch.zeros(1, nw, 1); e_col[0, route_col, 0] = 1.0
    cbox = {'leaf': None}
    orig_forward = ann.forward
    def forward_patched(z):
        out = orig_forward(z) + route_const
        if cbox['leaf'] is not None:
            out = out + cbox['leaf'].view(1, 1, 1) * e_col
        return out
    ann.forward = forward_patched

    # ── window bank: real data (u,y), encoder init, and true init (v11 cross) ──────
    um = norm.u_mean.flatten()[None, :]; us = norm.std_u.flatten()[None, :]
    y0 = np.asarray(norm.y0).flatten()[None, :]; ys = np.asarray(norm.ystd).flatten()[None, :]
    xm = norm.x_mean.flatten(); xs = norm.std_x.flatten()
    recs, starts = [], []
    for ri, f in enumerate(TRAIN_FILES):
        u, y, xl, _ = dm.load_T(f, CFG)
        un = ((u - um) / us).astype(np.float32); yn = ((y - y0) / ys).astype(np.float32)
        xln = ((xl - xm[None, :6]) / xs[None, :6]).astype(np.float32)
        recs.append((un, yn, xln)); N = len(un)
        for p in range(max(K0, na, nb), N - HMAX, STRIDE):
            starts.append((ri, p))
    starts = np.array(starts)
    rng = np.random.default_rng(SEED); rng.shuffle(starts)
    starts = starts[:N_BATCHES * B]
    print(f'[bank] {len(starts)} windows, HMAX={HMAX}')
    batches = []
    for bi in range(N_BATCHES):
        bs = starts[bi * B:(bi + 1) * B]
        U = np.empty((B, HMAX, 3), np.float32); Y = np.empty((B, HMAX, 3), np.float32)
        UH = np.empty((B, nb + 1, 3), np.float32); YH = np.empty((B, na + 1, 3), np.float32)
        Xt = np.zeros((B, 8), np.float32)
        for i, (ri, p) in enumerate(bs):
            un, yn, xln = recs[ri]
            U[i] = un[p:p + HMAX]; Y[i] = yn[p:p + HMAX]
            UH[i] = un[p - nb:p + 1]; YH[i] = yn[p - na:p + 1]
            Xt[i, :6] = xln[p]
        U = torch.from_numpy(U); Y = torch.from_numpy(Y)
        UH = torch.from_numpy(np.ascontiguousarray(UH)); YH = torch.from_numpy(np.ascontiguousarray(YH))
        with torch.no_grad():
            x0e = fit_sys.encoder(UH.contiguous(), YH.contiguous()).detach()
        batches.append(dict(U=U, Y=Y, x0e=x0e, x0t=torch.from_numpy(Xt)))

    # ── rollout primitives ─────────────────────────────────────────────────────────
    def roll_traj(x0, U, H, c, eps, mode='ann_route'):
        """Forward-only trajectory (B,H,3). mode 'ann_route' -> route_const injection; 'state_row' ->
        add c+eps to the dY STATE row after each step (v11 convention)."""
        with torch.no_grad():
            if mode == 'ann_route':
                route_const.zero_(); route_const[0, route_col, 0] = c + eps
            x = x0.clone(); outs = []
            for t in range(H):
                yhat, x = fit_sys.hfn(x, U[:, t, :])
                if mode == 'state_row':
                    x = x.clone(); x[:, IY] = x[:, IY] + (c + eps)
                outs.append(yhat)
            if mode == 'ann_route':
                route_const.zero_()
            return torch.stack(outs, dim=1)

    def perwin_mse(traj, target):
        return ((traj - target) ** 2).mean(dim=(1, 2)).cpu().numpy()      # (B,)

    def grad_batch(x0, U, Y, H):
        """Faithful autograd DC-direction gradient (old instrument) via the route-injection leaf."""
        route_const.zero_()
        c = torch.zeros(1, requires_grad=True)
        cbox['leaf'] = c
        x = x0.clone(); acc = 0.0
        for t in range(H):
            yhat, x = fit_sys.hfn(x, U[:, t, :])
            acc = acc + torch.mean((Y[:, t, :] - yhat) ** 2)
        (acc / H).backward()
        g = float(c.grad.item())
        cbox['leaf'] = None
        return g

    # ── run the landscape sweeps ────────────────────────────────────────────────────
    boot_rng = np.random.default_rng(SEED + 1)
    results = {}   # (target, H) -> boot dict
    for H in H_LAND:
        t0 = time.time()
        # eps=0 rollouts serve BOTH real and null; store per-window loss (Nwin, Ncgrid)
        Lreal = np.zeros((N_BATCHES * B, len(CGRID)), np.float32)
        Lnull = np.zeros((N_BATCHES * B, len(CGRID)), np.float32)
        for bi, bat in enumerate(batches):
            sl = slice(bi * B, (bi + 1) * B)
            self_tgt = roll_traj(bat['x0e'], bat['U'], H, 0.0, 0.0)          # c=0 self rollout = null target
            for ci, c in enumerate(CGRID):
                traj = roll_traj(bat['x0e'], bat['U'], H, float(c), 0.0)
                Lreal[sl, ci] = perwin_mse(traj, bat['Y'][:, :H, :])
                Lnull[sl, ci] = perwin_mse(traj, self_tgt)
        results[('real', H)] = boot_cstar_kappa(Lreal, CGRID, NBOOT, boot_rng)
        results[('null', H)] = boot_cstar_kappa(Lnull, CGRID, NBOOT, boot_rng)
        # old instrument: autograd gradient, one per batch -> mean/SE over batches
        gs = np.array([grad_batch(bat['x0e'], bat['U'], bat['Y'], H) for bat in batches])
        results[('grad', H)] = dict(mean=gs.mean(), se=gs.std(ddof=1) / np.sqrt(len(gs)), samples=gs)
        print(f'[H={H}] real c*={results[("real",H)]["cstar"]:+.2e} '
              f'(pinned {results[("real",H)]["pinned_frac"]*100:.0f}%, SE {results[("real",H)]["cstar_se"]:.1e}) '
              f'kappa={results[("real",H)]["kappa"]:.2e} | grad0 {gs.mean():+.2e}+/-{gs.std(ddof=1)/np.sqrt(len(gs)):.1e} '
              f'| {time.time()-t0:.0f}s')

    # ── Test 2: injection recovery at the training horizon ──────────────────────────
    inj = {}
    for eps in EPS_LIST:
        Linj = np.zeros((N_BATCHES * B, len(CGRID)), np.float32)
        for bi, bat in enumerate(batches):
            sl = slice(bi * B, (bi + 1) * B)
            self_tgt = roll_traj(bat['x0e'], bat['U'], H_POS, 0.0, 0.0)
            for ci, c in enumerate(CGRID):
                traj = roll_traj(bat['x0e'], bat['U'], H_POS, float(c), float(eps))
                Linj[sl, ci] = perwin_mse(traj, self_tgt)
        r = boot_cstar_kappa(Linj, CGRID, NBOOT, boot_rng)
        inj[eps] = r
        print(f'[inject +{eps:.0e}] recovered c*={r["cstar"]:+.3e} (target {-eps:+.1e}, '
              f'SE {r["cstar_se"]:.1e}, pinned {r["pinned_frac"]*100:.0f}%)')

    # ── v11 cross-check: real, TRUE init, STATE-ROW injection, H=H_POS ───────────────
    Lv11 = np.zeros((N_BATCHES * B, len(CGRID)), np.float32)
    for bi, bat in enumerate(batches):
        sl = slice(bi * B, (bi + 1) * B)
        for ci, c in enumerate(CGRID):
            traj = roll_traj(bat['x0t'], bat['U'], H_POS, float(c), 0.0, mode='state_row')
            Lv11[sl, ci] = perwin_mse(traj, bat['Y'][:, :H_POS, :])
    rv11 = boot_cstar_kappa(Lv11, CGRID, NBOOT, boot_rng)
    print(f'[v11 cross] real true-init state-row c*(400)={rv11["cstar"]:+.3e} '
          f'(v11 reported {V11_CSTAR_400:+.1e}, SE {rv11["cstar_se"]:.1e})')

    ann.forward = orig_forward   # restore

    # ── verdicts (pre-registered) ───────────────────────────────────────────────────
    print('\n==== PRE-REGISTERED VERDICTS ====')
    # Test 1: relative SE, landscape (c*, slope) vs raw gradient, per H
    print('Test 1 (better): relative SE  [ |SE/value| ]')
    for H in H_LAND:
        rr = results[('real', H)]; gg = results[('grad', H)]
        rel_grad = abs(gg['se'] / gg['mean']) if gg['mean'] else np.inf
        rel_slope = abs(rr['slope_se'] / rr['slope']) if rr['slope'] else np.inf
        rel_kappa = abs(rr['kappa_se'] / rr['kappa']) if rr['kappa'] else np.inf
        print(f'  H={H:5d}: raw-grad {rel_grad:6.2f} | slope {rel_slope:6.2f} | kappa {rel_kappa:6.3f}')
    t1 = all(abs(results[('real', H)]['kappa_se'] / results[('real', H)]['kappa'])
             < abs(results[('grad', H)]['se'] / results[('grad', H)]['mean']) for H in H_LAND)
    print(f'  -> Test 1 {"PASS" if t1 else "FAIL"} (kappa rel-SE < gradient rel-SE at every H)')
    # Test 2: recovery of at least eps>=1e-6
    t2 = all(abs(inj[e]['cstar'] - (-e)) < 0.5 * e for e in EPS_LIST if e >= 1e-6)
    print(f'Test 2 (control): {"PASS" if t2 else "FAIL"} (recover -eps within 50% for eps>=1e-6)')
    # Test 3: which branch
    c400 = results[('real', H_POS)]['cstar']; pin400 = results[('real', H_POS)]['pinned_frac']
    kappas = [results[('real', H)]['kappa'] for H in H_LAND]
    kappa_grows = kappas[-1] > kappas[0]
    near_trained = (not np.isnan(c400)) and abs(c400 - V12_FIXED_DC) < abs(V12_FIXED_DC)
    near_zero = np.isnan(c400) or abs(c400) < 1e-6
    branch = ('truncation-bias-as-loss-optimum' if (near_trained and kappa_grows)
              else 'Adam-artifact (flat at 400) -> pivot to training-dynamics instrument'
              if near_zero else 'ambiguous')
    print(f'Test 3 (what we want): real c*(400)={c400:+.2e} (pinned {pin400*100:.0f}%), '
          f'kappa grows with H: {kappa_grows} -> BRANCH: {branch}')
    adopt_A = t1 and t2 and branch.startswith('truncation-bias')
    print(f'\n==> ADOPT OPTION A: {"YES" if adopt_A else "NO -> use the training-dynamics instrument"}')

    # ── save ──────────────────────────────────────────────────────────────────────
    np.savez(os.path.join(datDir, 'b0_instrument_select.npz'),
             cgrid=CGRID, H_land=np.array(H_LAND), eps_list=np.array(EPS_LIST),
             real=np.array([results[('real', H)]['Lmean'] for H in H_LAND]),
             null=np.array([results[('null', H)]['Lmean'] for H in H_LAND]),
             inj=np.array([inj[e]['Lmean'] for e in EPS_LIST]),
             v11_Lmean=rv11['Lmean'],
             cstar=np.array([results[('real', H)]['cstar'] for H in H_LAND]),
             cstar_se=np.array([results[('real', H)]['cstar_se'] for H in H_LAND]),
             kappa=np.array([results[('real', H)]['kappa'] for H in H_LAND]),
             kappa_se=np.array([results[('real', H)]['kappa_se'] for H in H_LAND]),
             grad_mean=np.array([results[('grad', H)]['mean'] for H in H_LAND]),
             grad_se=np.array([results[('grad', H)]['se'] for H in H_LAND]),
             inj_cstar=np.array([inj[e]['cstar'] for e in EPS_LIST]),
             inj_cstar_se=np.array([inj[e]['cstar_se'] for e in EPS_LIST]),
             v11_cstar=rv11['cstar'], v11_cstar_se=rv11['cstar_se'],
             v12_fixed_dc=V12_FIXED_DC, v11_ref=V11_CSTAR_400, adopt_A=adopt_A)

    # ── figures ─────────────────────────────────────────────────────────────────────
    cx = CGRID * 1e6
    # Fig 1: landscape curves (bounded quantity we are switching to)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for H in H_LAND:
        Lm = results[('real', H)]['Lmean']
        ax[0].plot(cx, Lm / Lm.min(), 'o-', lw=1.3, ms=3.5, label=f'real  H={H}')
    # null (self target) has min ~0 at c=0 -> not /min-normalizable; its clean-well sanity is in the
    # injected panel and the saved data. Omit from this normalized panel (was a divide-by-zero).
    ax[0].axvline(0, color='k', lw=0.6); ax[0].set_yscale('log')
    ax[0].set_xlabel('constant c on the dY route  [1e-6, normalized]')
    ax[0].set_ylabel('windowed loss / its minimum'); ax[0].grid(True, which='both', alpha=0.3)
    ax[0].legend(fontsize=8); ax[0].set_title('A  Loss landscape vs horizon: does a longer H deepen the well?', fontsize=9)
    for e in EPS_LIST:
        Lm = inj[e]['Lmean']
        ax[1].plot(cx, Lm / Lm.min(), 'o-', lw=1.3, ms=3.5, label=f'inject +{e:.0e} (min at {-e*1e6:+.1f})')
    ax[1].axvline(0, color='k', lw=0.6); ax[1].set_yscale('log')
    ax[1].set_xlabel('constant c on the dY route  [1e-6, normalized]')
    ax[1].set_ylabel('windowed loss / its minimum'); ax[1].grid(True, which='both', alpha=0.3)
    ax[1].legend(fontsize=8); ax[1].set_title('B  Injection recovery (H=400): is the planted -eps found?', fontsize=9)
    fig.suptitle('Phase B0  Loss-landscape instrument (real + null + planted-offset controls)', fontsize=11)
    fig.text(0.005, 0.005, f'b0_instrument_select.npz | seed={SEED} | {N_BATCHES*B} windows | 2026-07-22',
             fontsize=6, color='0.45', va='bottom')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(figDir, 'b0_landscapes.png'), dpi=150); plt.close(fig)

    # Fig 2: the four decisive panels
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    Hs = np.array(H_LAND, float)
    # A: relative SE A/B
    relg = np.array([abs(results[('grad', H)]['se'] / results[('grad', H)]['mean']) for H in H_LAND])
    relk = np.array([abs(results[('real', H)]['kappa_se'] / results[('real', H)]['kappa']) for H in H_LAND])
    rels = np.array([abs(results[('real', H)]['slope_se'] / results[('real', H)]['slope']) for H in H_LAND])
    ax[0, 0].semilogy(Hs, relg, 'o-', label='raw gradient (old)')
    ax[0, 0].semilogy(Hs, rels, 's-', label='landscape slope@0 (new)')
    ax[0, 0].semilogy(Hs, relk, 'D-', label='landscape kappa (new)')
    ax[0, 0].set_xlabel('horizon H'); ax[0, 0].set_ylabel('relative SE  |SE/value|')
    ax[0, 0].grid(True, which='both', alpha=0.3); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title('A  Test 1: is the landscape statistically better?', fontsize=9)
    # B: kappa(H)
    kap = np.array([results[('real', H)]['kappa'] for H in H_LAND])
    kse = np.array([results[('real', H)]['kappa_se'] for H in H_LAND])
    ax[0, 1].errorbar(Hs, kap, yerr=2 * kse, fmt='o-', capsize=3)
    ax[0, 1].set_xlabel('horizon H'); ax[0, 1].set_ylabel('curvature kappa (pinning strength)')
    ax[0, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[0, 1].grid(True, alpha=0.3); ax[0, 1].set_title('B  Test 3: does the horizon pin the DC harder?', fontsize=9)
    # C: c*(H) vs trained DC
    cst = np.array([results[('real', H)]['cstar'] for H in H_LAND])
    cse = np.array([results[('real', H)]['cstar_se'] for H in H_LAND])
    ax[1, 0].errorbar(Hs, cst, yerr=2 * cse, fmt='o-', capsize=3, label='landscape c*(H)')
    ax[1, 0].axhline(V12_FIXED_DC, color='tab:red', ls='--', label=f'v12 fixed DC {V12_FIXED_DC:.1e}')
    ax[1, 0].axhline(0, color='tab:green', ls=':', label='v12 ARTBP DC ~0')
    ax[1, 0].plot([H_POS], [V11_CSTAR_400], 'kP', ms=11, label=f'v11 c*(400) {V11_CSTAR_400:.1e}')
    ax[1, 0].plot([H_POS], [rv11['cstar']], 'mx', ms=10, label=f'this run v11-repro {rv11["cstar"]:.1e}')
    ax[1, 0].set_xlabel('horizon H'); ax[1, 0].set_ylabel('loss-optimal constant c*')
    ax[1, 0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[1, 0].grid(True, alpha=0.3); ax[1, 0].legend(fontsize=7.5)
    ax[1, 0].set_title('C  Test 3: does c*(H) match the trained DC?', fontsize=9)
    # D: injection recovery
    epsx = np.array(EPS_LIST)
    rec = np.array([inj[e]['cstar'] for e in EPS_LIST]); recse = np.array([inj[e]['cstar_se'] for e in EPS_LIST])
    ax[1, 1].errorbar(epsx, rec, yerr=2 * recse, fmt='o', capsize=3, label='recovered c*')
    ax[1, 1].plot(epsx, -epsx, 'k--', label='ideal  c*=-eps')
    ax[1, 1].set_xscale('log'); ax[1, 1].set_xlabel('planted eps'); ax[1, 1].set_ylabel('recovered c*')
    ax[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[1, 1].grid(True, which='both', alpha=0.3); ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title('D  Test 2: positive control (recover the planted offset)', fontsize=9)
    fig.suptitle('Phase B0  Instrument selection: landscape (c*, kappa) vs raw gradient', fontsize=11)
    fig.text(0.005, 0.005, f'b0_instrument_select.npz | adopt Option A: {adopt_A} | 2026-07-22',
             fontsize=6, color='0.45', va='bottom')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(figDir, 'b0_instrument.png'), dpi=150); plt.close(fig)
    print(f'\nsaved figures -> {figDir}\ndone | data -> {os.path.join(datDir, "b0_instrument_select.npz")}')


if __name__ == '__main__':
    main()
