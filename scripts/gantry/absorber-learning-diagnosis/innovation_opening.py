"""Innovation opening phase for the thesis augmentation model (absorber-learning diagnosis, DECISIONS.md E3).

Two proper classes, imported by `model_innovation.py` and `gantry_innovation_opening.py` (copies of the
pipeline files, changes marked "INNOVATION OPENING"); nothing in the pipeline is overridden at runtime.

`InnovationOpening` is `SSE_Interconnect_Composed` with a temporary opening phase. For the first `n_open`
training evaluations of the loss the objective is
    L_thesis + w(t) * (lam_ee * EE + lam_inn * INN),   w(t) = 1 - t / n_open,
  EE  = mean over the batch of the squared one-step mismatch on the ADDED rows between the encoder's state at
        window start k+1 and one model step from the encoder's state at k (both encoder states detached, so
        the term trains only the model's state map); the k+1 history is the window's own history shifted by
        one sample (uhist, yhist plus the first future sample), so no extra data are needed;
  INN = unexplained fraction of the current model's first `m_inn` output errors (y - y_pred of THIS rollout,
        detached, divided by their batch RMS), predicted from the encoder's added states x_a(k) by a
        training-only linear head `inn_head` (zero-initialised; the encoder receives this gradient).
  lam_ee, lam_inn are set once, at the first evaluation, so each term equals L_thesis there.
After `n_open` evaluations the loss is exactly `SSE_Interconnect_Composed.loss` and the optimizer's row
factor is switched to 1. The model (physics block, Jan's Static_ANN_Block with its zero-init MLP, routing,
encoder) is untouched and equals the baseline at start; the head is never used by the model.
# THEORY: equation error is quadratic in the pole parameters of a linear model, output error is not
# (Shynk 1989, IEEE ASSP Magazine, printed pp.6-9, Eqs. (3)-(5)); start in equation error, finish in output error.
# HEURISTIC: the innovation target (an observer-like signal; the head predicting the model's own next errors),
# m_inn = 40 (10 ms), the matched weights and the linear decay have no literature value for this setting
# (local probe E3, DECISIONS.md section 4).

`RowScaledAdam` is torch's Adam whose update on selected rows of selected tensors is multiplied by `factor`.
Adam's update is lr * m_hat / (sqrt(v_hat) + eps), linear in lr, so this is exactly a per-row learning rate
lr * factor, with standard Adam state (state_dict compatible). Used for the added-state rows of the ANN's last
layer, the encoder's added-state map W^a and the head during the opening phase (factor 100: 1e-3 at the
thesis lr 1e-5, the E3 setting), then factor 1.
"""
__project_origin__ = "added"

import torch
import torch.nn as nn

from model_augmentation.fit_systems.interconnect import SSE_Interconnect_Composed


class RowScaledAdam(torch.optim.Adam):
    """Adam with a per-row learning-rate factor on selected tensors (see the module docstring)."""

    def __init__(self, params, scaled=(), factor=1.0, **kwargs):
        super().__init__(params, **kwargs)
        self.scaled = list(scaled)            # [(parameter, row slice or None for the whole tensor)]
        self.factor = float(factor)

    def set_factor(self, factor):
        self.factor = float(factor)

    @torch.no_grad()
    def step(self, closure=None):
        if self.factor == 1.0 or not self.scaled:
            return super().step(closure)
        before = [p.detach().clone() for p, _ in self.scaled]
        loss = super().step(closure)
        for (p, rows), b in zip(self.scaled, before):
            if rows is None:
                p.copy_(b + self.factor * (p - b))
            else:
                p[rows] = b[rows] + self.factor * (p[rows] - b[rows])
        return loss


class InnovationOpening(SSE_Interconnect_Composed):
    """SSE_Interconnect_Composed with the temporary innovation opening phase (module docstring)."""

    opening = None          # None = exactly SSE_Interconnect_Composed

    def configure_opening(self, n_open, m_inn, n_phys, n_aug, ny, dtype):
        self.opening = dict(n_open=int(n_open), m_inn=int(m_inn), n_phys=int(n_phys), n_aug=int(n_aug), ny=int(ny))
        self.inn_head = nn.Linear(n_aug, m_inn * ny).to(dtype)
        with torch.no_grad():
            self.inn_head.weight.zero_(); self.inn_head.bias.zero_()
        self._open_it = 0
        self._lam = None
        self._last_y = None

    def __getstate__(self):
        state = super().__getstate__()
        state.pop('_last_y', None)            # a batch tensor, not model state
        return state

    def simulate(self, x, ufuture, yfuture=None, **Loss_kwargs):
        y_pred, x_final = super().simulate(x, ufuture, yfuture, **Loss_kwargs)
        if self.opening is not None:
            self._last_y = y_pred.detach().clone()
        return y_pred, x_final

    def _shifted_history(self, uhist, yhist, ufuture, yfuture):
        """Encoder history of the window starting one sample later, from the window's own data."""
        nb_r, na_r = int(getattr(self, 'nb_right', 0)), int(getattr(self, 'na_right', 0))
        u1 = torch.cat([uhist[:, 1:], ufuture[:, nb_r:nb_r + 1]], 1)
        y1 = torch.cat([yhist[:, 1:], yfuture[:, na_r:na_r + 1]], 1)
        return u1, y1

    def opening_terms(self, uhist, yhist, ufuture, yfuture):
        """(EE, INN) of the module docstring for this batch; requires a preceding simulate() call."""
        o = self.opening
        p0, p1 = o['n_phys'], o['n_phys'] + o['n_aug']
        with torch.no_grad():
            xk = self.encoder(uhist, yhist)
            xk1 = self.encoder(*self._shifted_history(uhist, yhist, ufuture, yfuture))
        xp = self.hfn(xk, ufuture[:, 0])[1].reshape(xk.shape[0], -1)
        ee = (xk1.reshape(xk.shape[0], -1) - xp)[:, p0:p1].pow(2).mean()
        if self.inn_head.weight.device != xk.device:
            self.inn_head.to(xk.device)
        e = (yfuture[:, :o['m_inn']] - self._last_y[:, :o['m_inn']]).reshape(xk.shape[0], -1)
        e = e / e.pow(2).mean().sqrt().clamp_min(torch.finfo(e.dtype).tiny)
        xa = self.encoder(uhist, yhist).reshape(xk.shape[0], -1)[:, p0:p1]
        inn = (self.inn_head(xa) - e).pow(2).mean()
        return ee, inn

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        L = super().loss(uhist, yhist, ufuture, yfuture, **Loss_kwargs)
        o = self.opening
        if o is None or not torch.is_grad_enabled():
            return L
        if self._open_it >= o['n_open']:
            return L
        self._open_it += 1
        ee, inn = self.opening_terms(uhist, yhist, ufuture, yfuture)
        if self._lam is None:
            self._lam = (float(L) / max(float(ee), 1e-300), float(L) / max(float(inn), 1e-300))
        w = 1.0 - self._open_it / o['n_open']
        if self._open_it == 1 or self._open_it % 100 == 0 or self._open_it == o['n_open']:
            print('[innovation opening] eval %d/%d: thesis loss %.4e, EE %.4e, INN unexplained %.4f, weight %.3f'
                  % (self._open_it, o['n_open'], float(L), float(ee), float(inn), w), flush=True)
        if self._open_it == o['n_open']:
            if hasattr(self.optimizer, 'set_factor'):
                self.optimizer.set_factor(1.0)
            print('[innovation opening] phase ended: loss = thesis loss from now on, row lr factor 1', flush=True)
        return L + w * (self._lam[0] * ee + self._lam[1] * inn)
