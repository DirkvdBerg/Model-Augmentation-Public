"""PS2, the predictive-state opening: a data-driven initialisation of the added states during the first Adam updates."""
__project_origin__ = "added"

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PS2Spec:
    """The settings of the opening; none of them is a property of a particular plant."""
    aux_batch: int = 64            # HEURISTIC: the opening's own batch, not the training batch
    lr: float = 1e-3               # THEORY: Adam's default rate (Kingma and Ba) for both supervised stages
    eval_every: int = 25           # HEURISTIC: calibration check cadence, in optimizer updates
    s1_min: int = 100              # HEURISTIC: no S1 switch before this many updates
    s1_cap: int = 350              # HEURISTIC: S1 ends here at the latest
    s1_tol: float = 0.02           # HEURISTIC: plateau = the best of the last s1_window checks improved by less than s1_tol (relative)
    s1_window: int = 4
    # HEURISTIC: screen_at checks the current calibration fraction at this update, which must be on the evaluation grid
    screen_at: int = 150
    screen_max: float = 0.90
    n_cal: int = 128
    n_snap: int = 8192
    n_snap_cal: int = 2048
    s2_batch: int = 512
    s2_check: int = 100            # HEURISTIC: S2 plateau = the best of the last 5 checks improved by less than s2_tol (relative)
    s2_min: int = 500
    s2_cap: int = 8000
    s2_tol: float = 0.01
    seed: int = 0


def calibration_split(names):
    """Hold out the last listed record of each name class containing at least three records."""
    # a class is the record name without its trailing number, 'TR-S5' -> 'TR-S'
    # HEURISTIC: the calibration set covers every such class, since the predictive state of one excitation class
    # need not transfer to another
    cls = [n.rstrip('0123456789') for n in names]
    cal = sorted(max(i for i, c in enumerate(cls) if c == k) for k in set(cls) if cls.count(k) >= 3)
    return [i for i in range(len(names)) if i not in cal], cal


class PS2Opening:
    """The opening after every Adam update until it ends: S1 fits a decoder of the residual on the added states,
    S2 the network's added-state transition."""

    def __init__(self, fit_sys, data, n_xb, n_a, width, spec):
        """Prepare normalised training records and the decoder specification (n_xb physical, n_a added states)."""
        self.records = []
        for sd in data.train_list:
            sdn = fit_sys.norm.transform(sd)
            self.records.append((np.asarray(sdn.u), np.asarray(sdn.y)))
        self.names = [os.path.splitext(os.path.basename(str(f)))[0].split('_')[0] for f in data.train_files]
        self.spec, self.ctrl_rows = spec, list(fit_sys.simulator.train_ctrl_rows)
        self.n_xb, self.n_a, self.width = n_xb, n_a, width
        self.fit_ix, self.cal_ix = calibration_split(self.names)
        if not self.cal_ix:
            raise ValueError('no record class has 3 records, so the calibration split is empty; give at least one '
                             'record class 3 training records')
        self.phase, self.it, self.history = 'S1', 0, []
        self.s1_end = self.s2 = self.status = None
        self._rng = np.random.default_rng(spec.seed)
        self._decoder = self._opt = self._cal = None

    @property
    def done(self):
        return self.phase == 'S3'

    def after_step(self, fs):
        """Update S1 and check calibration on its evaluation grid, running S2 when S1 ends successfully."""
        if self.done:
            return
        if self._decoder is None:
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
        if (end or self.it == self.spec.screen_at) and u > self.spec.screen_max:
            self.phase, self.status = 'S3', 'fallback: residual not predictable (%.4f > %.2f at update %d)' % (
                u, self.spec.screen_max, self.it)
            print('[ps2] %s; the run continues as the plain method' % self.status, flush=True)
        elif end:
            self.s1_end = (self.it, u)
            self.s2 = self._fit_added_state_transition(fs)
            self.phase, self.status = 'S3', 'completed'
            print('[ps2] opening completed; the run continues as the plain method', flush=True)
        if self.done:
            self._decoder = self._opt = None

    def summary_json(self):
        """The opening's record, written as ps2_summary."""
        return json.dumps({'status': self.status or 'running', 'phase': self.phase, 'updates': self.it,
                           's1_end': self.s1_end, 's2': self.s2,
                           'calibration_records': [self.names[i] for i in self.cal_ix],
                           'history': self.history, 'spec': asdict(self.spec)})

    def _device(self, fs):
        p = next(fs.hfn.parameters())
        return p.device, p.dtype

    def _start(self, fs):
        """The decoder (output layer at zero), its optimizer over the decoder and theta^a_enc, and the
        calibration windows."""
        dev, dt = self._device(fs)
        enc = fs.encoder
        self._M = int(fs.na)
        ny = self.records[0][1].shape[1]
        self._decoder = nn.Sequential(nn.Linear(self.n_a, self.width), nn.Tanh(),
                                      nn.Linear(self.width, self._M * ny)).to(device=dev, dtype=dt)
        nn.init.zeros_(self._decoder[-1].weight); nn.init.zeros_(self._decoder[-1].bias)
        # theta^a_enc: W^a and the output layer of psi_tilde, of which only the added-state rows get a gradient
        aux = [enc.Wa_psi_u, enc.Wa_psi_y, enc.net[-1].weight, enc.net[-1].bias]
        self._aux = list(self._decoder.parameters()) + aux
        self._opt = torch.optim.Adam(self._aux, lr=self.spec.lr)
        self._cal = self._sample(fs, self.cal_ix, self.spec.n_cal, np.random.default_rng(self.spec.seed + 321))
        print('[ps2] opening started: M %d, calibration records %s, spec %s' % (
            self._M, [self.names[i] for i in self.cal_ix], json.dumps(asdict(self.spec))), flush=True)

    def _sample(self, fs, rec_ix, B, rng, pair=False):
        """Sample windows using eligible-start counts, reserving one future successor and an extra end sample;
        optionally include shifted histories."""
        nb, na = fs.nb, fs.na
        k0, M = max(na, nb), self._M
        dev, dt = self._device(fs)
        lens = np.array([len(self.records[i][0]) - k0 - M - 1 for i in rec_ix], dtype=float)
        ri = np.asarray(rec_ix)[rng.choice(len(rec_ix), size=B, p=lens / lens.sum())]
        out = {k: [] for k in ('uh', 'yh', 'uf', 'yf', 'uh1', 'yh1')}
        for r in ri:
            u, y = self.records[r]
            k = int(rng.integers(k0, len(u) - M - 1))
            out['uh'].append(u[k - nb:k]); out['yh'].append(y[k - na:k])
            out['uf'].append(u[k:k + M]); out['yf'].append(y[k:k + M])
            if pair:
                out['uh1'].append(u[k + 1 - nb:k + 1]); out['yh1'].append(y[k + 1 - na:k + 1])
        T = lambda a: torch.as_tensor(np.ascontiguousarray(np.stack(a)), device=dev, dtype=dt)
        b = {k: T(v) for k, v in out.items() if v}
        b['ctrl_ix'] = torch.as_tensor([self.ctrl_rows[r] for r in ri], device=dev, dtype=torch.long)
        return b

    def _residual(self, fs, b, x):
        """e_tau / sigma_e: the current model's next M closed-loop output residuals over their RMS on the batch."""
        with torch.no_grad():
            fs._sync_obc_coefficient(refresh_basis=False)
            y_pred, _ = fs.simulate(x, b['uf'], b['yf'], ctrl_ix=b['ctrl_ix'], nf=self._M)
            e = (b['yf'] - y_pred).reshape(len(x), -1)
            return e / e.pow(2).mean().sqrt()

    def _added_states(self, x):
        """x_bar of a model state: the n_a rows after the physical ones."""
        return x[:, self.n_xb:self.n_xb + self.n_a]

    def _s1_step(self, fs):
        """One S1 update: the decoder on x_bar predicts the residual; trains the decoder and theta^a_enc."""
        b = self._sample(fs, self.fit_ix, self.spec.aux_batch, self._rng)
        x = fs.encoder(b['uh'], b['yh'])
        e = self._residual(fs, b, x.detach())
        loss = (self._decoder(self._added_states(x)) - e).pow(2).mean()
        grads = torch.autograd.grad(loss, self._aux, allow_unused=True)
        # S1 updates shared encoder parameters with its own Adam state, then restores the existing gradient slots
        saved = [p.grad for p in self._aux]
        for p, g in zip(self._aux, grads):
            p.grad = torch.zeros_like(p) if g is None else g
        self._opt.step()
        for p, g in zip(self._aux, saved):
            p.grad = g

    def _cal_unexplained(self, fs):
        """nu: the fraction of the residual the decoder leaves unexplained on the calibration windows."""
        with torch.no_grad():
            x = fs.encoder(self._cal['uh'], self._cal['yh'])
            e = self._residual(fs, self._cal, x)
            return float((self._decoder(self._added_states(x)) - e).pow(2).mean() / e.pow(2).mean())

    def _fit_added_state_transition(self, fs):
        """S2: one-step regression of the network's added-state transition on frozen encoder pairs, kept at its
        best calibration cost."""
        sp = self.spec
        ann = fs.hfn.connected_blocks[fs._ann_ix]
        params = list(ann.parameters())
        srng = np.random.default_rng(sp.seed + 1000)

        def snap(ix, n):
            b = self._sample(fs, ix, n, srng, pair=True)
            with torch.no_grad():
                return fs.encoder(b['uh'], b['yh']), b['uf'][:, 0], self._added_states(fs.encoder(b['uh1'], b['yh1']))
        (Xf, Uf, Tf), (Xc, Uc, Tc) = snap(self.fit_ix, sp.n_snap), snap(self.cal_ix, sp.n_snap_cal)

        def state_map_loss(X, U, T):
            # the one-step error of x_bar over the target variance; T.var(0) divides by B - 1, which scales the
            # cost by (B - 1) / B and keeps its minimiser
            return (self._added_states(fs.hfn(X, U)[1].reshape(len(X), -1)) - T).pow(2).mean() / T.var(0).mean()
        fs._sync_obc_coefficient(refresh_basis=False)
        opt = torch.optim.Adam(params, lr=sp.lr)
        g = torch.Generator(device='cpu').manual_seed(sp.seed)
        with torch.no_grad():
            r0 = float(state_map_loss(Xc, Uc, Tc))
        # the start is a candidate too, so S2 never ends worse than it began on the calibration pairs
        best, state, checks, s = r0, [p.detach().clone() for p in params], [r0], 0
        for s in range(1, sp.s2_cap + 1):
            j = torch.randint(len(Xf), (sp.s2_batch,), generator=g).to(Xf.device)
            opt.zero_grad(set_to_none=True)
            state_map_loss(Xf[j], Uf[j], Tf[j]).backward()
            opt.step()
            if s % sp.s2_check:
                continue
            with torch.no_grad():
                rc = float(state_map_loss(Xc, Uc, Tc))
            checks.append(rc)
            if rc < best:
                best, state = rc, [p.detach().clone() for p in params]
            if s >= sp.s2_min and len(checks) > 5 and min(checks[-5:]) > (1 - sp.s2_tol) * min(checks[:-5]):
                break
        with torch.no_grad():
            for p, v in zip(params, state):
                p.copy_(v)
            for p in params:
                p.grad = None
        print('[ps2] S2 state-map fit: calibration cost %.4f -> %.4f in %d steps' % (r0, best, s), flush=True)
        return {'residual_start': r0, 'residual_best': best, 'steps': s}
