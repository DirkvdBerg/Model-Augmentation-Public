"""Multiple shooting: the continuity (defect) term the SUBNET loss leaves out (D-127).

Background
----------
`docs/multiple-shooting-sweep-2026-07-25.md` established that this pipeline already IS
a multiple-shooting method: the `nf`-step windows are the shooting segments and the
encoder is the node-elimination step (Beintema, Toth, Schoukens, L4DC 2021, whose own
keyword list ends with "Multiple Shooting"). What is missing is the continuity
constraint between consecutive segments.

Ribeiro, Tiels, Umenberger, Schon, Aguirre, "On the smoothness of nonlinear system
identification", Automatica 121:109158, 2020 (arXiv:1905.00820):

  * Theorem 2 / Corollary 3: the multiple-shooting objective `V_M` equals the
    single-shooting objective `V` over the WHOLE record **iff the defects vanish**,
    and this holds regardless of the segment length. With fully decoupled windows the
    defects are never formed, so today's objective is NOT equivalent to the long-horizon
    problem the model is selected on. That is the formal statement of the 120x
    train/select horizon gap (`drift-conclusions-2026-07-25.md` section 3 item 1).
  * Theorem 1 at `L_h = 1` (our marginal, |lambda| = 1 case): Lipschitz constant of `V`
    is `O(N)` and beta-smoothness is `O(N^3)` in the WITHIN-SEGMENT length `N`. So the
    defect buys the long-horizon objective over SHORT gradient paths, which is exactly
    why it is categorically different from simply raising `nf` (refuted on this rig,
    SLURM 71013, and divergent at NF=900 precisely as `O(N^3)` predicts).

Contents
--------
  SSE_Interconnect_MultipleShooting   OrthLoss subclass adding the defect penalty when
                                      configured; exact no-op otherwise.

Usage
-----
Build the fit system with `n_seg > 1` and set `defect_weight`. The training data must
carry `nf = n_seg * nf_seg` so each sample spans `n_seg` contiguous segments; the
gradient path stays `nf_seg` because each segment is re-anchored to the encoder.
"""
__project_origin__ = "added"

import torch

from model_augmentation.fit_systems.blocks import Static_ANN_Block
from model_augmentation.fit_systems.orth_projection import SSE_Interconnect_OrthLoss


class SSE_Interconnect_MultipleShooting(SSE_Interconnect_OrthLoss):
    """OrthLoss + optional inter-segment continuity (defect) penalty.

    A training sample of `nf = n_seg * nf_seg` steps is split into `n_seg` contiguous
    segments. Each segment is independently encoder-initialised (exactly as a single
    window is today), and at every internal boundary the loss adds

        defect_weight * mean_j mean_batch || W * (x_j0_encoder - x_{j-1}[end]) ||

    over the NORMALISED state, with a **non-squared** norm.

    Why non-squared: Turan and Jaschke (IEEE L-CSS 6:1897-1902, 2022) note that the
    `l1`, `l2`-not-squared and `l_inf` norms are EXACT penalty functions, i.e. under
    standard assumptions a single minimisation at some finite weight yields the
    constrained solution. The quadratic default has no such finite-weight guarantee;
    it only reaches the constrained solution as the weight goes to infinity.

    Why the weight is not a swept hyperparameter: Fisher, Tremolet, Auvinen, Tan, Poli
    ("Weak-constraint and long-window 4D-Var", ECMWF, 2011) give the continuity penalty
    a statistical meaning as `Q^-1`, the inverse model-error covariance. `defect_scale`
    is where that enters: pass the per-state residual std and the penalty becomes
    `Q^{-1/2}`-weighted rather than arbitrary. Leave it None for an unweighted norm.

    Exact no-op contract (mirrors D7.2 for the orth penalty): with `n_seg <= 1` or
    `defect_weight == 0` this class returns the parent value untouched, so the feature
    is bit-identical to the current pipeline when off.

    Attributes
    ----------
    n_seg : int
        Number of contiguous shooting segments per training sample. 1 = off.
    defect_weight : float
        Penalty weight. 0 = off.
    defect_scale : Tensor (nx,) or None
        Per-state weights `W` applied inside the norm (e.g. 1/sigma_residual).
        None = unweighted.
    defect_norm : {'l2', 'l1', 'linf'}
        Which exact-penalty norm to use. Default 'l2' (NOT squared).
    """

    n_seg = 1              # class defaults; instance attributes set by the pipeline
    defect_weight = 0.0
    defect_scale = None
    defect_norm = 'l2'

    # ---- diagnostics: populated every loss() call so the trainer can log them ----
    last_defect_rms = None     # float, RMS of the raw (unweighted) defect
    last_mse = None            # float, the fit term alone

    def _defect_penalty(self, defects):
        """Reduce a list of (batch, nx) defects to one scalar penalty."""
        d = torch.stack(defects, dim=0)                       # (n_seg-1, batch, nx)
        self.last_defect_rms = float(torch.sqrt((d.detach() ** 2).mean()))
        if self.defect_scale is not None:
            d = d * self.defect_scale.to(d.device, d.dtype)
        if self.defect_norm == 'l1':
            per = d.abs().sum(dim=-1)
        elif self.defect_norm == 'linf':
            per = d.abs().amax(dim=-1)
        else:                                                  # 'l2', NOT squared
            per = torch.linalg.vector_norm(d, dim=-1)
        return self.defect_weight * per.mean()

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        if self.n_seg <= 1 or self.defect_weight == 0.0:
            return super().loss(uhist, yhist, ufuture, yfuture, **Loss_kwargs)

        na_right = getattr(self, 'na_right', 0)
        nb_right = getattr(self, 'nb_right', 0)
        nf_tot = ufuture.shape[1]
        if nf_tot % self.n_seg:
            raise ValueError(f'nf ({nf_tot}) must be divisible by n_seg ({self.n_seg})')
        nf_seg = nf_tot // self.n_seg
        if nf_seg < max(self.na, self.nb):
            raise ValueError(
                f'segment length {nf_seg} < max(na,nb)={max(self.na, self.nb)}: the encoder '
                f'window for segment j>0 would reach outside the sample')

        errors, defects = [], []
        x = self.encoder(uhist, yhist)                         # segment 0 node
        for j in range(self.n_seg):
            s = j * nf_seg
            if j > 0:
                # deepSI window convention (System_data.to_hist_future_data):
                #   uhist = [u[k-nb] ... u[k-1+nb_right]], ufuture = [u[k] ... u[k+nf-1]]
                # so the encoder window for the node at future offset s is entirely
                # inside this sample whenever s >= max(na, nb).
                x_node = self.encoder(ufuture[:, s - self.nb: s + nb_right],
                                      yfuture[:, s - self.na: s + na_right])
                defects.append(x_node - x)      # gradient flows BOTH into the encoder
                x = x_node                      # and back through segment j-1's rollout
            for t in range(s, s + nf_seg):
                yhat, x = self.hfn(x, ufuture[:, t])           # type: ignore
                errors.append(torch.nn.functional.mse_loss(yfuture[:, t], yhat))

        # Length-weighted sum over segments (Ribeiro et al. V_M); with equal segment
        # lengths that is the plain mean over all steps, matching the parent's scale.
        loss_MSE = torch.mean(torch.stack(errors))
        self.last_mse = float(loss_MSE.detach())
        L = loss_MSE + self._defect_penalty(defects)

        # Same theta sweep as SSE_Interconnect_ParamLoss.loss (interconnect.py:742-747);
        # duplicated because the segmented rollout replaces the parent's rollout.
        for m in self.hfn.connected_blocks:                    # type: ignore
            if hasattr(m, 'param_loss'):
                L = L + m.param_loss()

        # Same orth pickup as SSE_Interconnect_OrthLoss.loss (orth_projection.py:87-93).
        if self.orth_penalty is not None and self.orth_penalty.beta != 0.0:
            ann = next(m for m in self.hfn.connected_blocks    # type: ignore
                       if isinstance(m, Static_ANN_Block))
            L = L + self.orth_penalty(ann)

        return L
