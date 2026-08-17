"""Loss variants for A, B and C, as mixins over the production fit system.

Nothing outside this folder is modified. The production `build_model` constructs an
`SSE_Interconnect_MultipleShooting`; these mixins are grafted onto the instance with

    fit_sys.__class__ = type('X', (Mixin, type(fit_sys)), {})

so optimiser, encoder, normalisation, shooting structure and every default remain the
production ones and only `loss()` differs. That is the point: a bespoke trainer would answer
"can my trainer learn this", which is not the question.

  A  no mixin. The production loss, open loop.
  C  SoWeightedLossMixin: the residual is filtered by So before the norm.
  B  ClosedLoopLossMixin: the controller is closed around the model during the rollout.

B needs neither the reference nor the injected multisine. Cfb is linear, so

    u_model = Cfb(r - y_model) + f_ms
            = Cfb(r - y_data) + Cfb(y_data - y_model) + f_ms
            = u_data + Cfb(y_data - y_model)

i.e. the closed-loop input is the recorded input plus the controller applied to the output
residual. Everything needed is already in the training window.

UNITS. The model works in normalised coordinates; Cfb and So are physical, m -> N. Both mixins
denormalise the residual, filter, and renormalise the result, using the same norm object the
pipeline built. Getting this wrong silently rescales the loss.

SAMPLE RATE. The pipeline runs at cfg.fs_new (4 kHz by default) while Cfb was designed at
20 kHz. p2_rate_compare.py measured the 4 kHz loop's sensitivity peak 15.3 % higher in the
absorber band. The controller used here is re-discretised at the pipeline rate, so B and C are
run against a loop that is close to, but not identical to, the one that made the data.
"""
__project_origin__ = "added"

import numpy as np
import torch

from p2_rate_compare import build_cfb_at
import so_filter as SOF
from scipy.signal import cont2discrete, tf2ss


def _tf_to_ss_batch(cfb):
    """Per-channel (b, a) -> block-diagonal (A, B, C, D) for the 3-channel diagonal Cfb."""
    As, Bs, Cs, Ds = [], [], [], []
    for b, a in cfb:
        A, B, C, D = tf2ss(b, a)
        As.append(A); Bs.append(B); Cs.append(C); Ds.append(D)
    n = sum(A.shape[0] for A in As)
    A = np.zeros((n, n)); B = np.zeros((n, 3)); C = np.zeros((3, n)); D = np.zeros((3, 3))
    i = 0
    for j, (Aj, Bj, Cj, Dj) in enumerate(zip(As, Bs, Cs, Ds)):
        m = Aj.shape[0]
        A[i:i + m, i:i + m] = Aj
        B[i:i + m, j] = Bj.ravel()
        C[j, i:i + m] = Cj.ravel()
        D[j, j] = Dj.ravel()[0]
        i += m
    return A, B, C, D


def controller_ss(Y_op, ts):
    """Cfb at the given rate, as one 3-in 3-out state space."""
    cfb, _ = build_cfb_at(Y_op, ts)
    return _tf_to_ss_batch(cfb)


def sensitivity_ss(Y_op, ts):
    """So = (I + Gop Cfb)^-1 at the given rate."""
    ctrl = controller_ss(Y_op, ts)
    return SOF.so_ss(Y_op, ctrl, ts=ts)


class _NormMixin:
    """Shared denormalise / renormalise helpers. Set by attach()."""

    def _y_phys_resid(self, y_true_n, y_pred_n):
        """Normalised residual -> physical residual [m]."""
        return (y_true_n - y_pred_n) * self._ystd_t

    def _u_norm_from_phys(self, u_phys):
        """Physical force [N] -> normalised input increment (mean cancels for an increment)."""
        return u_phys / self._stdu_t


class SoWeightedLossMixin(_NormMixin):
    """Option C. L = || So (y_model - y_data) ||, open loop otherwise."""

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist)
        nf = ufuture.shape[1]
        A, B, C, D = self._so_t
        xs = torch.zeros(yfuture.shape[0], A.shape[0], dtype=yfuture.dtype,
                         device=yfuture.device)
        errs = []
        for t in range(nf):
            yhat, x = self.hfn(x, ufuture[:, t])
            e_phys = self._y_phys_resid(yfuture[:, t], yhat)          # [m]
            e_w = xs @ C.T + e_phys @ D.T                              # So applied
            xs = xs @ A.T + e_phys @ B.T
            errs.append(torch.mean((e_w / self._ystd_t) ** 2))         # back to normalised scale
        L = torch.mean(torch.stack(errs))
        self.last_mse = float(L.detach())
        for m in self.hfn.connected_blocks:
            if hasattr(m, 'param_loss'):
                L = L + m.param_loss()
        if self.orth_penalty is not None and self.orth_penalty.beta != 0.0:
            from model_augmentation.fit_systems.blocks import Static_ANN_Block
            ann = next(m for m in self.hfn.connected_blocks
                       if isinstance(m, Static_ANN_Block))
            L = L + self.orth_penalty(ann)
        return L


class ClosedLoopLossMixin(_NormMixin):
    """Option B. The controller is closed around the model during the rollout.

    u_model = u_data + Cfb(y_data - y_model), exact because Cfb is linear.

    The plant output has no feedthrough (D_d = 0), so yhat depends on x only and the two hfn
    calls per step differ only in the state they advance. The first call reads yhat, the second
    advances the state with the corrected input.
    """

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist)
        nf = ufuture.shape[1]
        Ac, Bc, Cc, Dc = self._cfb_t
        xc = torch.zeros(yfuture.shape[0], Ac.shape[0], dtype=yfuture.dtype,
                         device=yfuture.device)
        errs = []
        for t in range(nf):
            yhat, _ = self.hfn(x, ufuture[:, t])                       # D = 0, so yhat = h(x)
            e_phys = self._y_phys_resid(yfuture[:, t], yhat)           # [m]
            u_fb = xc @ Cc.T + e_phys @ Dc.T                           # [N]
            xc = xc @ Ac.T + e_phys @ Bc.T
            u_cl = ufuture[:, t] + self._u_norm_from_phys(u_fb)
            _, x = self.hfn(x, u_cl)                                   # advance with the loop input
            errs.append(torch.nn.functional.mse_loss(yfuture[:, t], yhat))
        L = torch.mean(torch.stack(errs))
        self.last_mse = float(L.detach())
        for m in self.hfn.connected_blocks:
            if hasattr(m, 'param_loss'):
                L = L + m.param_loss()
        if self.orth_penalty is not None and self.orth_penalty.beta != 0.0:
            from model_augmentation.fit_systems.blocks import Static_ANN_Block
            ann = next(m for m in self.hfn.connected_blocks
                       if isinstance(m, Static_ANN_Block))
            L = L + self.orth_penalty(ann)
        return L


def attach(fit_sys, variant, norm, cfg, Y_op=0.10):
    """Graft the variant's loss onto the instance. variant in {'A', 'B', 'C'}."""
    if variant == 'A':
        return fit_sys
    ts = 1.0 / (cfg.fs_new or cfg.fs_orig)
    dt_t = torch.float64 if cfg.use_f64 else torch.float32

    def T(M):
        return torch.tensor(np.asarray(M, float), dtype=dt_t)

    mixin = {'B': ClosedLoopLossMixin, 'C': SoWeightedLossMixin}[variant]
    # The grafted class must be findable by `pickle`, or `evaluate_and_save`'s checkpoint save
    # dies with "Can't pickle <class 'loss_variants.FitSys_B'>: attribute lookup FitSys_B on
    # loss_variants failed" AFTER training has completed, losing the entire verdict block.
    # That is exactly what happened to the first variant B run (4 epochs, 3.2 h wall).
    # `type()` sets __module__ to this module but does NOT bind the name here, so pickle's
    # lookup fails. Bind it, and reuse the same object on repeat calls so a second attach()
    # does not create a second, non-identical class under the same name.
    cls_name = 'FitSys_' + variant
    new_cls = globals().get(cls_name)
    if new_cls is None or new_cls.__bases__ != (mixin, type(fit_sys)):
        new_cls = type(cls_name, (mixin, type(fit_sys)), {})
        new_cls.__module__ = __name__
        globals()[cls_name] = new_cls
    fit_sys.__class__ = new_cls
    fit_sys._ystd_t = T(np.asarray(norm.ystd).ravel())
    fit_sys._stdu_t = T(np.asarray(norm.std_u).ravel())
    if variant == 'C':
        fit_sys._so_t = tuple(T(M) for M in sensitivity_ss(Y_op, ts))
    else:
        fit_sys._cfb_t = tuple(T(M) for M in controller_ss(Y_op, ts))
    return fit_sys
