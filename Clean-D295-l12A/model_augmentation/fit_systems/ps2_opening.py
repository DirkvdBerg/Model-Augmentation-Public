"""Predictive-state (PS) initialisation of the added states.

S1 trains the encoder's added states to predict the model's output residual through a temporary decoder,
alongside Adam training on the closed-loop loss V. S2 pauses V to fit the added-state update to consecutive
encoder estimates. The decoder is then removed and training on V continues, keeping S1 changes even if S2 is skipped.
"""
__project_origin__ = "added"

import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from gantry_dynamic.data import record_name


@dataclass(frozen=True)
class PS2Spec:
    """Batch sizes, learning rate and chosen stopping thresholds for S1 and S2."""
    aux_batch: int       # windows per S1 step, separate from the training batch
    lr: float            # learning rate for the separate S1 and S2 optimizers
    # S1: fit the decoder and added-state encoder parameters
    eval_every: int      # updates between calibration checks
    s1_min: int          # earliest update at which a plateau can end S1
    s1_cap: int          # stop at the first calibration check at or after this update
    s1_tol: float        # required relative reduction in nu; see _plateaued
    s1_window: int       # number of recent nu values compared with the best preceding value
    screen_at: int       # skip S2 if nu > screen_max here; must be a multiple of eval_every
    screen_max: float    # largest unexplained-residual ratio nu that allows S2
    n_cal: int           # fixed calibration windows for S1
    # S2: added-state transition fit
    n_snap: int          # number of fixed pairs of consecutive encoder estimates used to fit S2
    n_snap_cal: int      # number of fixed pairs from left-out records used to evaluate S2
    s2_batch: int        # pairs per step
    s2_check: int        # steps between calibration costs
    s2_window: int       # number of recent calibration costs compared with the best preceding cost
    s2_min: int          # earliest step at which a plateau can end S2
    s2_cap: int          # maximum S2 steps
    s2_tol: float        # required relative reduction in calibration cost; see _plateaued
    seed: int            # seed for sampling windows and S2 minibatches


def _plateaued(values, window, tol):
    """Return True if none of the last window costs is at least tol lower than the best preceding cost.

    tol is relative (0.02 means 2%). Return False until window + 1 costs are available.
    """
    return len(values) > window and min(values[-window:]) > (1 - tol) * min(values[:-window])


def calibration_split(names):
    """Return (fit indices, calibration indices) into names.

    For each excitation class with >= 3 records, reserve the last listed record for calibration of S1 and S2.
    These records remain in the main loss V.
    """
    # Names without their trailing record number identify excitation classes: 'TR-S5' -> 'TR-S'.
    cls = [n.rstrip('0123456789') for n in names]
    cal = sorted(max(i for i, c in enumerate(cls) if c == k) for k in set(cls) if cls.count(k) >= 3)
    return [i for i in range(len(names)) if i not in cal], cal


class PS2Opening:
    """Run S1 after each main Adam update, then optionally S2 once."""

    def __init__(self, fit_sys, data, n_xb, n_a, width, spec):
        """Normalise the records and split them for fitting and calibration.

        n_xb and n_a count physical and added states; width is the decoder's hidden-layer size.
        """
        self._added_cols = list(range(n_xb, n_xb + n_a))
        self.records = []
        for sd in data.train_list:
            sdn = fit_sys.norm.transform(sd)
            self.records.append((np.asarray(sdn.u), np.asarray(sdn.y)))
        self.names = [record_name(f) for f in data.train_files]
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
        """True when the opening ends (stored as 'S3')."""
        return self.phase == 'S3'

    def after_step(self, fs):
        """Take one S1 step after the main model's Adam update.

        Check the unexplained-residual ratio nu every eval_every updates. End S1 at the first check
        with a plateau at update >= s1_min or with update >= s1_cap. Run S2 only if nu <= screen_max.
        At screen_at, nu > screen_max also ends S1 without S2. The plateau test is _plateaued.
        """
        if self.done:
            return
        if self._decoder is None:
            self._start(fs)
        self.it += 1
        # The residual must include the OBC correction for the just-updated model; its basis stays fixed.
        fs.sync_obc_coefficient(refresh_basis=False)
        with torch.no_grad():
            self._mats = fs.objective_matrices()
        self._s1_step(fs)
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
        """Return settings, calibration record names, S1 (update, nu) history and S2 results as JSON."""
        return json.dumps({'status': self.status or 'running', 'phase': self.phase, 'updates': self.it,
                           's1_end': self.s1_end, 's2': self.s2,
                           'calibration_records': [self.names[i] for i in self.cal_ix],
                           'history': self.history, 'spec': asdict(self.spec)})

    def _start(self, fs):
        """Build a one-hidden-layer tanh decoder, its Adam optimiser and fixed calibration windows.

        Map n_a added states to H * ny residual entries: H = fs.na future samples and ny output channels.
        """
        p = next(fs.hfn.parameters())
        self._dev, self._dt = dev, dt = p.device, p.dtype
        enc = fs.encoder
        self._M = int(fs.na)
        ny = self.records[0][1].shape[1]
        # Zero the final layer so the decoder initially predicts zero residual.
        self._decoder = nn.Sequential(nn.Linear(self.n_a, self.width), nn.Tanh(),
                                      nn.Linear(self.width, self._M * ny)).to(device=dev, dtype=dt)
        nn.init.zeros_(self._decoder[-1].weight); nn.init.zeros_(self._decoder[-1].bias)
        # Adam receives the full output-layer weights and bias; only their added-state rows have S1 gradients.
        aux = [enc.Wa_psi_u, enc.Wa_psi_y, enc.net[-1].weight, enc.net[-1].bias]
        self._aux = list(self._decoder.parameters()) + aux
        self._opt = torch.optim.Adam(self._aux, lr=self.spec.lr)
        # Fixed windows avoid changes in nu caused by drawing a different calibration sample.
        self._cal = self._sample(fs, self.cal_ix, self.spec.n_cal, np.random.default_rng(self.spec.seed + 321))
        print('[ps2] opening started: M %d, calibration records %s, spec %s' % (
            self._M, [self.names[i] for i in self.cal_ix], json.dumps(asdict(self.spec))), flush=True)

    def _sample(self, fs, rec_ix, B, rng, pair=False):
        """Draw B windows with replacement, uniformly over allowed starts in records rec_ix.

        Both stages use max(fs.na, fs.nb) <= k < N - self._M - 1 for a record of N samples;
        each record must allow at least one start. The bound leaves >= 2 samples beyond the S1 future.

        Return normalised uh (B, fs.nb, inputs) and yh (B, fs.na, outputs), ending at k - 1.
        S1 also returns uf and yf from k through k + self._M - 1, and each record's ctrl_ix.
        With pair=True, return u0 at k and shifted histories uh1, yh1 ending at k instead.
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
        b = {k: torch.as_tensor(np.ascontiguousarray(np.stack(v)), device=self._dev, dtype=self._dt)
             for k, v in out.items()}
        if not pair:
            b['ctrl_ix'] = torch.as_tensor([self.ctrl_rows[r] for r in ri], device=self._dev, dtype=torch.long)
        return b

    def _residual(self, fs, b, x):
        """Form the fixed S1 target from the current model's closed-loop residual.

        From encoder state x at k, simulate H = self._M = fs.na samples, matching the output-history length.
        Use the model after the latest main Adam update.
        Start the controller state at zero; add its feedback on measured minus predicted output to recorded u.

        r stacks measured minus predicted normalised outputs over k .. k + H - 1, time then channel:
            e = r / sqrt(mean(r^2)).
        The RMS covers all windows, times and channels and must be nonzero. Return (batch, H * outputs).
        The rollout and RMS are fixed when calculating S1 gradients.
        """
        with torch.no_grad():
            y_pred, _ = fs.simulate(x, b['uf'], b['yf'], ctrl_ix=b['ctrl_ix'], **self._mats)
            e = (b['yf'] - y_pred).reshape(len(x), -1)
            return e / e.pow(2).mean().sqrt()

    def _added_states(self, x):
        """Return the added-state columns of a batch of full states x."""
        return x[:, self.n_xb:self.n_xb + self.n_a]

    def _s1_step(self, fs):
        """Fit decoder p and the encoder's added-state parameters using encoded states xbar and target e from _residual:
            L_S1 = mean((p(xbar) - e)^2),
        averaging over windows, times and output channels.

        A separate Adam updates p, linear maps Wa_psi_u and Wa_psi_y, and the added-state rows of the encoder's
        output weights and bias. Encoder hidden layers, physical-state parameters and model dynamics stay fixed.
        Differentiate through xbar, holding e fixed, to encourage the added states to predict the residual.
        """
        b = self._sample(fs, self.fit_ix, self.spec.aux_batch, self._rng)
        x = fs.encoder(b['uh'], b['yh'])
        e = self._residual(fs, b, x.detach())
        loss = (self._decoder(self._added_states(x)) - e).pow(2).mean()
        grads = torch.autograd.grad(loss, self._aux, allow_unused=True)
        # Use S1 gradients for this step, then restore the gradients left by V. Each Adam keeps its own moments.
        saved = [p.grad for p in self._aux]
        for p, g in zip(self._aux, grads):
            p.grad = torch.zeros_like(p) if g is None else g
        self._opt.step()
        for p, g in zip(self._aux, saved):
            p.grad = g

    def _cal_unexplained(self, fs):
        """Evaluate decoder p on fixed calibration windows using added states xbar and residual target e:
            nu = sum((p(xbar) - e)^2) / sum(e^2).

        Recompute xbar and e with the current encoder and model, as in _s1_step.
        Sum over windows, times and output channels. nu = 1 matches predicting zero; nu <= 0.90 explains
        at least 10% of the residual energy. nu can exceed one.
        """
        with torch.no_grad():
            x = fs.encoder(self._cal['uh'], self._cal['yh'])
            e = self._residual(fs, self._cal, x)
            return float((self._decoder(self._added_states(x)) - e).pow(2).mean() / e.pow(2).mean())

    def _fit_added_state_transition(self, fs):
        """Fit next-added-state network g_aug using fixed encoder pairs after S1, with V paused.

        X is the full state encoded from histories ending at k - 1, U the recorded input at k, and T the
        next added state encoded from histories ending at k. Keep these estimates and the encoder fixed.

        A separate Adam fits the network parameters to minimise
            L_S2 = mean((g_aug(X, U) - T)^2) / mean(var(T, dim=0))
        Average error over pairs and added states. Calculate each target state's variance across B pairs,
        then average across states. With divisor B - 1, this cost is (B - 1)/B times thesis eq:ps2_s2.
        Use each training minibatch's variance and all fixed calibration pairs for evaluation.
        Require B > 1 and positive mean target variance.

        Check every s2_check steps; stop on _plateaued at step >= s2_min or at s2_cap, even between checks.
        Keep parameters with the lowest calibration cost, including the start; ties keep the earlier set.
        The plateau history includes the starting cost. Return starting/best costs and steps taken.
        """
        sp = self.spec
        ann = fs.hfn.connected_blocks[fs._ann_ix]
        # The loss gives the physical output rows zero gradients, but shared hidden layers also change
        # the network's correction to the physical-state dynamics.
        params = list(ann.parameters())
        srng = np.random.default_rng(sp.seed + 1000)

        def snap(ix, n):
            """Encode n fixed (X, U, T) samples from records ix."""
            b = self._sample(fs, ix, n, srng, pair=True)
            with torch.no_grad():
                return fs.encoder(b['uh'], b['yh']), b['u0'], self._added_states(fs.encoder(b['uh1'], b['yh1']))
        # At a window's first sample the controller state is zero, so recorded and closed-loop inputs agree
        # if the encoder reproduces that sample's output. S2 uses the recorded input without this correction.
        (Xf, Uf, Tf), (Xc, Uc, Tc) = snap(self.fit_ix, sp.n_snap), snap(self.cal_ix, sp.n_snap_cal)

        def state_map_loss(X, U, T, t_var):
            """Evaluate L_S2 with supplied mean target variance t_var."""
            return (ann(torch.cat((X, U), dim=1))[:, self._added_cols, 0] - T).pow(2).mean() / t_var
        opt = torch.optim.Adam(params, lr=sp.lr)
        g = torch.Generator(device='cpu').manual_seed(sp.seed)
        # Draw minibatches with replacement using a separate seed, so other torch draws do not change them.
        batches = torch.randint(len(Xf), (sp.s2_cap, sp.s2_batch), generator=g).to(Xf.device)
        cal_var = Tc.var(0).mean()
        with torch.no_grad():
            r0 = float(state_map_loss(Xc, Uc, Tc, cal_var))
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
            if s >= sp.s2_min and _plateaued(checks, sp.s2_window, sp.s2_tol):
                break
        with torch.no_grad():
            for p, v in zip(params, state):
                p.copy_(v)
            for p in params:
                p.grad = None
        print('[ps2] S2 state-map fit: calibration cost %.4f -> %.4f in %d steps' % (r0, best, s), flush=True)
        return {'residual_start': r0, 'residual_best': best, 'steps': s}
