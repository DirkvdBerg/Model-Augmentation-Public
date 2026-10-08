"""PS2: a data-driven initialisation of the added-state part of an S-DP augmentation (D-234, D-236).

Hoekstra et al. (arXiv:2602.17297, Sec. 5.4) initialise the baseline part of the encoder from the baseline's
reconstructability map and leave the added-state part random, with a zero learning function. PS2 gives that part an
initial meaning from data, by the same reconstructability principle, and then hands over to the unchanged training:

  S1  predictive state (Hefny, Downey, Gordon, NeurIPS 2015, S1A regression, "possibly non-linear", p. 4): while the
      normal training runs, a temporary head h and the encoder's added-state outputs are fitted to
          min || h(psi_a(past_k)) - e_k ||^2,
      e_k = the CURRENT model's next M closed-loop output residuals (prediction path, detached, unit batch RMS),
      M = the encoder's past output window.
  S2  state map (Hefny 2015 S2 regression; 2SR initialisation then BPTT, Downey et al., NeurIPS 2017, Sec. 4.2): once,
      the deployed added-state transition is fitted on a FROZEN snapshot of detached encoder pairs,
          min || g_aug(psi(past_k), u_k) - psi_a(past_{k+1}) ||^2.
  S3  nothing: the run continues as the plain method.

Attached through the declared seam `SSE_Interconnect.training_phase`; fit() calls `after_step(fit_sys)` once after every
optimizer update. The opening owns its own Adam and never touches the training optimizer's gradients; it predicts only
through the codebase's prediction path (`_sync_prediction_state_for_validation` + `simulate` without `obc_pass`).
Parameter SELECTION only, no masks: S1's loss reaches the encoder only through its added-state outputs and S2's reaches
the network only through its added-state rows, so every other entry gets an exactly zero gradient and Adam leaves it
exactly unchanged (asserted in scripts/gantry/thesis-results/test_ps2_pipeline.py). If the residual is not predictable
(screen), S2 is skipped and the run is the plain method (fall-back).
"""
__project_origin__ = "added"

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PS2Spec:
    """The opening's rules (D-234 / D-236). No number below is a property of any particular plant."""
    aux_batch: int = 64            # HEURISTIC: the prototype's batch (D-234); the opening's own, not the training batch
    lr: float = 1e-3               # THEORY: Adam's default rate (Kingma and Ba 2015) for both supervised stages
    eval_every: int = 25           # HEURISTIC: calibration check cadence, in optimizer updates
    s1_min: int = 100              # HEURISTIC: no S1 switch before this many updates
    s1_cap: int = 350              # HEURISTIC: S1 ends here at the latest (decided every prototype run, D-234)
    s1_tol: float = 0.02           # HEURISTIC: plateau = best of the last s1_window checks improved < s1_tol (relative)
    s1_window: int = 4
    screen_at: int = 150           # HEURISTIC: screen: calibration unexplained fraction <= screen_max by this update
    screen_max: float = 0.90       #            (and at the S1 end), else fall back to the plain method
    n_cal: int = 128               # calibration windows for the S1 checks
    n_snap: int = 8192             # S2 snapshot pairs from the fit records
    n_snap_cal: int = 2048         # S2 snapshot pairs from the calibration records
    s2_batch: int = 512
    s2_check: int = 100            # HEURISTIC: S2 plateau = best of the last 5 checks improved < s2_tol (relative)
    s2_min: int = 500
    s2_cap: int = 8000
    s2_tol: float = 0.01
    seed: int = 0


def calibration_split(names):
    """Record-level split: hold out the LAST record of every record class with at least 3 records.

    Class = record name without its trailing number, e.g. 'TR-S5' -> 'TR-S'. # HEURISTIC (D-234 P1/D2): a calibration set
    must be class-representative; the predictive state of one excitation class need not transfer to another."""
    cls = [n.rstrip('0123456789') for n in names]
    cal = sorted(max(i for i, c in enumerate(cls) if c == k) for k in set(cls) if cls.count(k) >= 3)
    return [i for i in range(len(names)) if i not in cal], cal


class PS2Opening:
    def __init__(self, spec, records, ctrl_rows, names, n_xb, n_a, width):
        """records: normalised (u, y) arrays of the training records, in the simulator's training order;
        ctrl_rows: their controller rows; names: record names; n_xb, n_a: physical and added states;
        width: the head's hidden width (= the deployed network's)."""
        if len(records) != len(ctrl_rows) or len(records) != len(names):
            raise ValueError('records, ctrl_rows and names must be aligned one per training record')
        self.spec, self.records, self.ctrl_rows, self.names = spec, records, list(ctrl_rows), list(names)
        self.n_xb, self.n_a, self.width = n_xb, n_a, width
        self.fit_ix, self.cal_ix = calibration_split(self.names)
        if not self.cal_ix:
            raise ValueError('no record class has 3 records; the calibration split is empty')
        self.phase, self.it, self.history = 'S1', 0, []
        self.s1_end = self.s2 = self.status = None
        self._rng = np.random.default_rng(spec.seed)
        self._head = self._opt = self._cal = None

    @classmethod
    def for_pipeline(cls, fit_sys, data, n_xb, n_a, width, spec=None):
        """Build from the gantry pipeline's DataBundle and the attached ClosedLoopSimulator."""
        recs = []
        for sd in data.train_list:
            sdn = fit_sys.norm.transform(sd)
            recs.append((np.asarray(sdn.u), np.asarray(sdn.y)))
        names = [os.path.splitext(os.path.basename(str(f)))[0].split('_')[0] for f in data.train_files]
        return cls(spec or PS2Spec(), recs, fit_sys.simulator.train_ctrl_rows, names, n_xb, n_a, width)

    # ------------------------------------------------------------------ the seam
    @property
    def done(self):
        return self.phase == 'S3'

    def after_step(self, fs):
        if self.done:
            return
        if self._head is None:
            self._start(fs)
        self.it += 1
        self._s1_step(fs)
        if self.it % self.spec.eval_every:
            return
        u = self._cal_unexplained(fs)
        self.history.append((self.it, u))
        best_before = min(v for _, v in self.history[:-self.spec.s1_window]) if len(self.history) > self.spec.s1_window else None
        plateau = (self.it >= self.spec.s1_min and best_before is not None
                   and min(v for _, v in self.history[-self.spec.s1_window:]) > (1 - self.spec.s1_tol) * best_before)
        end = plateau or self.it >= self.spec.s1_cap
        print('[ps2] update %d: calibration unexplained %.4f%s' % (self.it, u, ' (S1 ends: %s)' % (
            'plateau' if plateau else 'cap') if end else ''), flush=True)
        if (self.it == self.spec.screen_at and not end and u > self.spec.screen_max) or (end and u > self.spec.screen_max):
            self.phase, self.status = 'S3', 'fallback: residual not predictable (%.4f > %.2f at update %d)' % (
                u, self.spec.screen_max, self.it)
            print('[ps2] %s; the run continues as the plain method' % self.status, flush=True)
        elif end:
            self.s1_end = (self.it, u)
            self.s2 = self._fit_map(fs)
            self.phase, self.status = 'S3', 'completed'
            print('[ps2] opening completed; the run continues as the plain method', flush=True)
        if self.done:
            self._head = self._opt = None          # nothing of the opening survives it

    def summary(self):
        return {'status': self.status or 'running', 'phase': self.phase, 'updates': self.it, 's1_end': self.s1_end,
                's2': self.s2, 'calibration_records': [self.names[i] for i in self.cal_ix], 'history': self.history,
                'spec': asdict(self.spec)}

    def summary_json(self):
        return json.dumps(self.summary())

    # ------------------------------------------------------------------ internals
    def _device(self, fs):
        p = next(fs.hfn.parameters())
        return p.device, p.dtype

    def _start(self, fs):
        dev, dt = self._device(fs)
        enc = fs.encoder
        self._M = int(fs.na + getattr(fs, 'na_right', 0))     # = the encoder's past output window (samples)
        ny = self.records[0][1].shape[1]
        self._head = nn.Sequential(nn.Linear(self.n_a, self.width), nn.Tanh(),
                                   nn.Linear(self.width, self._M * ny)).to(device=dev, dtype=dt)
        nn.init.zeros_(self._head[-1].weight); nn.init.zeros_(self._head[-1].bias)   # starts predicting no error
        aux = [enc.Wa_psi_u, enc.Wa_psi_y]
        if getattr(enc, 'net', None) is not None:                                    # x_a rows of the NL part
            last = [m for m in enc.net.modules() if isinstance(m, nn.Linear)][-1]
            aux += [last.weight, last.bias]
        self._aux = list(self._head.parameters()) + aux
        self._opt = torch.optim.Adam(self._aux, lr=self.spec.lr)
        self._cal = self._sample(fs, self.cal_ix, self.spec.n_cal, np.random.default_rng(self.spec.seed + 321))
        print('[ps2] opening started: M %d, calibration records %s, spec %s' % (
            self._M, [self.names[i] for i in self.cal_ix], json.dumps(asdict(self.spec))), flush=True)

    def _sample(self, fs, rec_ix, B, rng, pair=False):
        """B windows from the given records: encoder history, future (M samples), controller row (and the history one
        sample later). Window convention of n_phase1.windows / deepSI to_hist_future_data."""
        nb, na = fs.nb, fs.na
        na_r, nb_r = getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0)
        k0, M = max(na, nb), self._M
        dev, dt = self._device(fs)
        lens = np.array([len(self.records[i][0]) - k0 - M - 1 for i in rec_ix], dtype=float)
        ri = np.asarray(rec_ix)[rng.choice(len(rec_ix), size=B, p=lens / lens.sum())]
        out = {k: [] for k in ('uh', 'yh', 'uf', 'yf', 'uh1', 'yh1')}
        for r in ri:
            u, y = self.records[r]
            k = int(rng.integers(k0, len(u) - M - 1))
            out['uh'].append(u[k - nb:k + nb_r]); out['yh'].append(y[k - na:k + na_r])
            out['uf'].append(u[k:k + M]); out['yf'].append(y[k:k + M])
            if pair:
                out['uh1'].append(u[k + 1 - nb:k + 1 + nb_r]); out['yh1'].append(y[k + 1 - na:k + 1 + na_r])
        T = lambda a: torch.as_tensor(np.ascontiguousarray(np.stack(a)), device=dev, dtype=dt)   # noqa: E731
        b = {k: T(v) for k, v in out.items() if v}
        b['ctrl_ix'] = torch.as_tensor([self.ctrl_rows[r] for r in ri], device=dev, dtype=torch.long)
        return b

    def _residual(self, fs, b, x):
        """The current model's next M closed-loop output residuals, through the prediction path."""
        with torch.no_grad():
            fs._sync_prediction_state_for_validation(refresh_basis=False)
            y_pred, _ = fs.simulate(x, b['uf'], b['yf'], ctrl_ix=b['ctrl_ix'], nf=self._M)
            e = (b['yf'] - y_pred).reshape(len(x), -1)
            return e / e.pow(2).mean().sqrt()

    def _xa(self, x):
        return x[:, self.n_xb:self.n_xb + self.n_a]

    def _s1_step(self, fs):
        b = self._sample(fs, self.fit_ix, self.spec.aux_batch, self._rng)
        x = fs.encoder(b['uh'], b['yh'])
        e = self._residual(fs, b, x.detach())
        loss = (self._head(self._xa(x)) - e).pow(2).mean()
        grads = torch.autograd.grad(loss, self._aux, allow_unused=True)
        saved = [p.grad for p in self._aux]                  # the training optimizer's gradients, left as found
        for p, g in zip(self._aux, grads):
            p.grad = torch.zeros_like(p) if g is None else g
        self._opt.step()
        for p, g in zip(self._aux, saved):
            p.grad = g

    def _cal_unexplained(self, fs):
        with torch.no_grad():
            x = fs.encoder(self._cal['uh'], self._cal['yh'])
            e = self._residual(fs, self._cal, x)
            return float((self._head(self._xa(x)) - e).pow(2).mean() / e.pow(2).mean())

    def _fit_map(self, fs):
        """S2: one-step regression of the deployed added-state transition on frozen encoder pairs."""
        sp = self.spec
        ann = fs.hfn.connected_blocks[fs._ann_ix]
        params = list(ann.parameters())
        srng = np.random.default_rng(sp.seed + 1000)

        def snap(ix, n):
            b = self._sample(fs, ix, n, srng, pair=True)
            with torch.no_grad():
                return fs.encoder(b['uh'], b['yh']), b['uf'][:, 0], self._xa(fs.encoder(b['uh1'], b['yh1']))
        (Xf, Uf, Tf), (Xc, Uc, Tc) = snap(self.fit_ix, sp.n_snap), snap(self.cal_ix, sp.n_snap_cal)

        def ee(X, U, T):
            return (self._xa(fs.hfn(X, U)[1].reshape(len(X), -1)) - T).pow(2).mean() / T.var(0).mean()
        fs._sync_prediction_state_for_validation(refresh_basis=False)
        opt = torch.optim.Adam(params, lr=sp.lr)
        g = torch.Generator(device='cpu').manual_seed(sp.seed)
        with torch.no_grad():
            r0 = float(ee(Xc, Uc, Tc))
        best, state, checks, s = r0, None, [r0], 0
        for s in range(1, sp.s2_cap + 1):
            j = torch.randint(len(Xf), (sp.s2_batch,), generator=g).to(Xf.device)
            opt.zero_grad(set_to_none=True)
            ee(Xf[j], Uf[j], Tf[j]).backward()
            opt.step()
            if s % sp.s2_check:
                continue
            with torch.no_grad():
                rc = float(ee(Xc, Uc, Tc))
            checks.append(rc)
            if rc < best:
                best, state = rc, [p.detach().clone() for p in params]
            if s >= sp.s2_min and len(checks) > 5 and min(checks[-5:]) > (1 - sp.s2_tol) * min(checks[:-5]):
                break
        with torch.no_grad():
            if state is not None:
                for p, v in zip(params, state):
                    p.copy_(v)
            for p in params:
                p.grad = None
        print('[ps2] S2 state-map fit: calibration EE residual %.4f -> %.4f in %d steps' % (r0, best, s), flush=True)
        return {'residual_start': r0, 'residual_best': best, 'steps': s}
