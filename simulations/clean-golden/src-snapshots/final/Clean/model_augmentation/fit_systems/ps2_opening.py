"""PS, the predictive-state initialisation of the added states during the first Adam updates: S1 trains a decoder p
and theta^a_enc to predict the residual from xbar, S2 trains g_aug^n on one-step encoder pairs.

Conventions: superscript n = normalised; xbar is not normalised; mean squares average over the batch, the horizon
and the outputs. Code names: M = H = n + 1 (residual horizon), nu = the fraction of the residual left unexplained,
cal = the left-out records, 'S3' = the opening has ended. In S1, sigma_e is one dimensionless RMS of y^n - y_hat^n
over the batch, the horizon and the axes, not the servo-error scales sigma_e,r,j of W_r and validation.
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


def _plateaued(values, window, tol):
    """Plateau: len(values) > window and min(values[-window:]) > (1 - tol) min(values[:-window])."""
    return len(values) > window and min(values[-window:]) > (1 - tol) * min(values[:-window])


def calibration_split(names):
    """Left-out records of PS.

    The last record of each excitation class (name without trailing number) with at least three records; they judge
    S1 and S2 but stay in V.

    Args:
        names: training record names.
    Returns:
        (fit indices, left-out indices).
    """
    # a class is the record name without its trailing number, 'TR-S5' -> 'TR-S'
    # HEURISTIC: the calibration set covers every such class, since the predictive state of one excitation class
    # need not transfer to another
    cls = [n.rstrip('0123456789') for n in names]
    cal = sorted(max(i for i, c in enumerate(cls) if c == k) for k in set(cls) if cls.count(k) >= 3)
    return [i for i in range(len(names)) if i not in cal], cal


class PS2Opening:
    """The opening after every Adam update until it ends: S1 fits a decoder of the residual on the added states,
    S2 the network's added-state transition."""

    def __init__(self, fit_sys, data, n_xb, n_a, width, spec, route_ix):
        """Prepare normalised training records and the decoder specification (n_xb physical, n_a added states;
        route_ix, the model state each network output is routed to)."""
        # the network outputs of the added states, in state order
        routes = list(route_ix)
        self._added_cols = [routes.index(n_xb + k) for k in range(n_a)]
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
        self._decoder = self._opt = self._cal = self._mats = None

    @property
    def done(self):
        """True once the opening has ended."""
        return self.phase == 'S3'

    def after_step(self, fs):
        """PS step after each Adam step on V.

        One S1 step; every eval_every steps nu on the left-out records decides. S1 ends at a plateau or the cap, then S2
        runs if nu <= screen_max (the decoder explains at least 10 %). At the screen_at check, or at the end of S1,
        nu > screen_max ends PS without S2; S1's encoder changes are kept.

        Args:
            fs: the augmented model, just updated by Adam.
        """
        if self.done:
            return
        if self._decoder is None:
            self._start(fs)
        self.it += 1
        # one coefficient sync and one set of matrices per update: S1 leaves the network unchanged
        fs._sync_obc_coefficient(refresh_basis=False)
        with torch.no_grad():
            self._mats = fs.objective_matrices()
        # 1. One S1 step.
        self._s1_step(fs)
        # 2. On the check grid: nu on the left-out records, then end S1 or continue.
        if self.it % self.spec.eval_every:
            return
        u = self._cal_unexplained(fs)
        self.history.append((self.it, u))
        plateau = self.it >= self.spec.s1_min and _plateaued([v for _, v in self.history], self.spec.s1_window,
                                                             self.spec.s1_tol)
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

    def _start(self, fs):
        """Decoder p(xbar) = W_p2 tanh(W_p1 xbar + b_p1) + b_p2 with W_p2 = b_p2 = 0, its own Adam over theta_p and
        theta^a_enc, and the left-out windows."""
        p = next(fs.hfn.parameters())
        self._dev, self._dt = dev, dt = p.device, p.dtype
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
        """Sample windows for S1 or the S2 pairs.

        Start samples drawn in proportion to each record's eligible count.

        Args:
            fs: the model; rec_ix: records to draw from; B: number of windows; rng: generator; pair: S2 pairs.
        Returns:
            dict, normalised: uh (B, n + 1, 3), yh (B, n + 1, 3); for S1 uf, yf (B, H, 3) and ctrl_ix (B,) integer bank
            rows; with pair u0 (B, 3) and the next histories uh1, yh1 (B, n + 1, 3).
        """
        nb, na = fs.nb, fs.na
        k0, M = max(na, nb), self._M
        lens = np.array([len(self.records[i][0]) - k0 - M - 1 for i in rec_ix], dtype=float)
        ri = np.asarray(rec_ix)[rng.choice(len(rec_ix), size=B, p=lens / lens.sum())]
        out = {k: [] for k in (('uh', 'yh', 'u0', 'uh1', 'yh1') if pair else ('uh', 'yh', 'uf', 'yf'))}
        for r in ri:
            u, y = self.records[r]
            k = int(rng.integers(k0, len(u) - M - 1))
            out['uh'].append(u[k - nb:k]); out['yh'].append(y[k - na:k])
            if pair:
                out['u0'].append(u[k])
                out['uh1'].append(u[k + 1 - nb:k + 1]); out['yh1'].append(y[k + 1 - na:k + 1])
            else:
                out['uf'].append(u[k:k + M]); out['yf'].append(y[k:k + M])
        # np.stack keeps the column-major layout of the records; the encoder views need C order
        b = {k: torch.as_tensor(np.ascontiguousarray(np.stack(v)), device=self._dev, dtype=self._dt)
             for k, v in out.items()}
        if not pair:
            b['ctrl_ix'] = torch.as_tensor([self.ctrl_rows[r] for r in ri], device=self._dev, dtype=torch.long)
        return b

    def _residual(self, fs, b, x):
        """S1 residual target e_tau.

        e_tau = (y^n_k - y_hat^n_{k|tau}, k = tau .. tau + H - 1) / sigma_e, sigma_e^2 = mean over the batch, the
        horizon and the axes of (y^n - y_hat^n)^2, from the current model's closed-loop rollout; held fixed during
        the S1 step.

        Args:
            fs: the model; b: sampled windows; x: (B, 6 + n_xbar) encoder states.
        Returns:
            (B, H n_y) scaled residuals.
        """
        with torch.no_grad():
            y_pred, _ = fs.simulate(x, b['uf'], b['yf'], ctrl_ix=b['ctrl_ix'], **self._mats)
            # Residual over H samples of the closed-loop rollout from x_hat^n_{tau|tau}.
            e = (b['yf'] - y_pred).reshape(len(x), -1)
            # Scale: e_tau / sigma_e.
            return e / e.pow(2).mean().sqrt()

    def _added_states(self, x):
        """xbar = psi_a(chi): the n_xbar rows after the physical ones."""
        return x[:, self.n_xb:self.n_xb + self.n_a]

    def _s1_step(self, fs):
        """S1 step.

        min over theta_p, theta^a_enc of mean ||p(psi_a(chi_tau)) - e_tau||^2: the decoder sees only xbar, so the
        encoder is pushed to put the predictable residual into the added states.

        Args:
            fs: the model.
        """
        # 1. Sample windows and encode.
        b = self._sample(fs, self.fit_ix, self.spec.aux_batch, self._rng)
        x = fs.encoder(b['uh'], b['yh'])
        # 2. Frozen residual target.
        e = self._residual(fs, b, x.detach())
        # 3. Decoder cost and gradient.
        # S1 cost: mean ||p(xbar_{tau|tau}) - e_tau||^2 over the batch, H and the axes.
        loss = (self._decoder(self._added_states(x)) - e).pow(2).mean()
        # g = grad_(theta_p, theta^a_enc) V_S1.
        grads = torch.autograd.grad(loss, self._aux, allow_unused=True)
        # 4. S1's own Adam step.
        # S1 updates shared encoder parameters with its own Adam state, then restores the existing gradient slots
        saved = [p.grad for p in self._aux]
        for p, g in zip(self._aux, grads):
            p.grad = torch.zeros_like(p) if g is None else g
        self._opt.step()
        for p, g in zip(self._aux, saved):
            p.grad = g

    def _cal_unexplained(self, fs):
        """S1 check nu.

        nu = sum ||p(xbar) - e||^2 / sum ||e||^2 on the left-out windows; it ends S1 and decides whether S2 runs.

        Args:
            fs: the model.
        Returns:
            nu (dimensionless).
        """
        with torch.no_grad():
            x = fs.encoder(self._cal['uh'], self._cal['yh'])
            e = self._residual(fs, self._cal, x)
            return float((self._decoder(self._added_states(x)) - e).pow(2).mean() / e.pow(2).mean())

    def _fit_added_state_transition(self, fs):
        """S2: one-step fit of the added-state transition.

        min over theta_aug of sum ||g_aug^n(x_hat^n_k, u^n_k) - xbar_{k+1|k+1}||^2 / sum (xbar_{k+1|k+1} - m)^2, times
        (B - 1) / B (sample variance), on fixed encoder pairs; theta_aug kept at the lowest left-out cost.

        Args:
            fs: the model after S1.
        Returns:
            dict with the left-out cost at the start, at the best point, and the steps run.
        """
        sp = self.spec
        ann = fs.hfn.connected_blocks[fs._ann_ix]
        # S2 trains theta_aug through g_aug^n; the physical output rows stay fixed, but shared hidden layers can
        # also change f_aug^n
        params = list(ann.parameters())
        srng = np.random.default_rng(sp.seed + 1000)

        def snap(ix, n):
            """Pairs (psi(chi_k), u^n_k, psi_a(chi_{k+1})) of n sampled windows."""
            b = self._sample(fs, ix, n, srng, pair=True)
            with torch.no_grad():
                return fs.encoder(b['uh'], b['yh']), b['u0'], self._added_states(fs.encoder(b['uh1'], b['yh1']))
        # 1. Fixed pairs from the frozen encoder.
        (Xf, Uf, Tf), (Xc, Uc, Tc) = snap(self.fit_ix, sp.n_snap), snap(self.cal_ix, sp.n_snap_cal)

        def state_map_loss(X, U, T, t_var):
            """Mean squared added-state prediction error divided by t_var, the mean sample variance of the target
            coordinates."""
            return (ann(torch.cat((X, U), dim=1))[:, self._added_cols, 0] - T).pow(2).mean() / t_var
        # 2. Adam on the S2 cost, checked on the left-out pairs.
        opt = torch.optim.Adam(params, lr=sp.lr)
        g = torch.Generator(device='cpu').manual_seed(sp.seed)
        # every minibatch index of the S2 budget in one draw: the same stream as one draw per step
        batches = torch.randint(len(Xf), (sp.s2_cap, sp.s2_batch), generator=g).to(Xf.device)
        # t_var = (1/n_xbar) sum_j var(T_j), var with 1/(B - 1), for the left-out set here and per batch below.
        cal_var = Tc.var(0).mean()
        with torch.no_grad():
            r0 = float(state_map_loss(Xc, Uc, Tc, cal_var))
        # the start is a candidate too, so S2 never ends worse than it began on the calibration pairs
        best, state, checks, s = r0, [p.detach().clone() for p in params], [r0], 0
        for s in range(1, sp.s2_cap + 1):
            j = batches[s - 1]
            opt.zero_grad(set_to_none=True)
            Tj = Tf[j]
            state_map_loss(Xf[j], Uf[j], Tj, Tj.var(0).mean()).backward()
            opt.step()
            if s % sp.s2_check:
                continue
            with torch.no_grad():
                rc = float(state_map_loss(Xc, Uc, Tc, cal_var))
            checks.append(rc)
            if rc < best:
                best, state = rc, [p.detach().clone() for p in params]
            if s >= sp.s2_min and _plateaued(checks, 5, sp.s2_tol):
                break
        with torch.no_grad():
            # 3. Restore the best parameters.
            for p, v in zip(params, state):
                p.copy_(v)
            for p in params:
                p.grad = None
        print('[ps2] S2 state-map fit: calibration cost %.4f -> %.4f in %d steps' % (r0, best, s), flush=True)
        return {'residual_start': r0, 'residual_best': best, 'steps': s}
