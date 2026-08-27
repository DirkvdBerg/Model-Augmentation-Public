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
    defect_acc_weight = 0.0
    defect_scale = None
    defect_norm = 'l2'

    # ---- diagnostics: populated every loss() call so the trainer can log them ----
    last_defect_rms = None     # float, RMS of the raw (unweighted) defect
    last_defect_acc = None     # float, norm of the ACCUMULATED defect (see below)
    last_mse = None            # float, the fit term alone

    def _norm(self, d):
        """The exact-penalty norm over the state axis. Never squared (Turan & Jaschke)."""
        if self.defect_norm == 'l1':
            return d.abs().sum(dim=-1)
        if self.defect_norm == 'linf':
            return d.abs().amax(dim=-1)
        return torch.linalg.vector_norm(d, dim=-1)             # 'l2', NOT squared

    def _defect_penalty(self, defects):
        """Reduce a list of (batch, nx) defects to one scalar penalty.

        TWO reductions of the SAME defects, and they measure different things:

          defect_weight      mean_j || d_j ||        a POWER detector (sign-blind)
          defect_acc_weight  || sum_j d_j ||         a COHERENCE detector

        By the triangle inequality ||sum d|| <= sum ||d||, with equality only when
        every d_j points the same way. So alternating (ripple) defects cancel in the
        sum while same-sign (drift) defects survive at full size. A norm cannot tell
        those apart because it discards sign.

        WHY THE SECOND TERM EXISTS (measured, not assumed):
          * MS6: the norm is BLIND to the failure. INIT-vs-DEGRADED defect ratio
            1.01x (encoder nodes) / 1.78x (true nodes) against a 65.04x free-run
            ratio, and an n_seg sweep of 4/12/30 flat at 1.08/1.02/1.04.
          * MS7 (`scripts/gantry/coulomb-offset/diag_mean_defect.py`): the summed
            defect RESPONDS and improves with coupling -- ratio 1.16/0.98/1.37/1.60/
            2.88 over n_seg 4/12/30/60/119, monotone from n_seg=12 and still rising
            at 47,600 coupled steps, while the norm stays 1.07/0.98/1.01/1.03/1.01.
            Mechanism visible in the raw values: the INIT arm's summed defect decays
            as ~1/sqrt(n_seg-1) (incoherent averaging) while the DEGRADED arm's does
            not, i.e. it has a coherent component underneath.
          * Why coherence is the right target: the failure is the ACCUMULATION of
            per-step-unbiased increments on the K=0 axes. The per-window DC is
            zero-mean across the dataset (HAC |t| < 1.3 on six axis/record
            combinations, frictionless AND Coulomb), so no per-window statistic can
            see it. To price accumulation you have to accumulate.

        SUM, not mean, deliberately: MS7 measured the discrimination GROWING with
        n_seg, and a mean would normalise exactly that growth away.

        WHAT THE SECOND TERM IS NOT: it is not a continuity constraint and does NOT
        give Ribeiro's Thm 2 equivalence, which needs each d_j to vanish, not their
        sum. Keep defect_weight non-zero for that; the two are complementary.

        AUGMENTED ROWS: exclude them via defect_scale (set those entries to 0), as
        MS3 does. With the ANN's final layer zero-initialised the propagated x_aug is
        identically zero (measured 0.000000e+00, verify_ms_gradient.py G6), because
        xp is a SUM of block contributions with no identity path from x and only the
        ANN writes those rows. So an augmented-row defect reduces to enc_aug(node_j)
        and its only minimiser is enc_aug = 0: a live gradient with a degenerate
        target. Revisit only once the ANN is demonstrably off zero.
        """
        d = torch.stack(defects, dim=0)                       # (n_seg-1, batch, nx)
        self.last_defect_rms = float(torch.sqrt((d.detach() ** 2).mean()))
        if self.defect_scale is not None:
            d = d * self.defect_scale.to(d.device, d.dtype)

        L = d.new_zeros(())
        if self.defect_weight != 0.0:
            L = L + self.defect_weight * self._norm(d).mean()

        if self.defect_acc_weight != 0.0:
            d_acc = d.sum(dim=0)                               # (batch, nx)
            self.last_defect_acc = float(self._norm(d_acc.detach()).mean())
            L = L + self.defect_acc_weight * self._norm(d_acc).mean()
        else:
            self.last_defect_acc = None
        return L

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        # CHANGED (accumulated defect): the guard must cover BOTH weights, or setting
        # defect_acc_weight alone would silently fall through to the parent and the
        # new term would never run. The exact-no-op contract is unchanged: with both
        # weights zero this still returns the parent value untouched.
        if self.n_seg <= 1 or (self.defect_weight == 0.0
                               and self.defect_acc_weight == 0.0):
            return super().loss(uhist, yhist, ufuture, yfuture, **Loss_kwargs)

        # CHANGED (closed-loop seam): refuse the combination rather than guess it. Routing the
        # segment rollout through self.simulate() below makes "multiple shooting with a driven
        # rollout" REACHABLE for the first time, and nothing decides whether the driver's own
        # state (for the closed loop, the controller state xc) resets at a segment boundary the
        # way x is re-anchored to the encoder. Both readings are defensible and they are
        # different objectives. The xc = 0 argument in closed_loop.py is a statement about a
        # WINDOW start and does not settle a SEGMENT start for free. A raise is not a second
        # rollout path; a silently wrong combination is worse than an unsupported one.
        if getattr(self, 'simulator', None) is not None:
            raise NotImplementedError(
                'n_seg = %d with an attached simulator (%s) is not defined: whether the '
                'simulator resets its own state at a segment boundary is an open modelling '
                'question, not an implementation gap. Run multiple shooting with '
                'fit_sys.simulator = None, or decide the semantics first.'
                % (self.n_seg, type(self.simulator).__name__))

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

        preds, defects = [], []
        x = self.encoder(uhist, yhist)                         # segment 0 node
        for j in range(self.n_seg):
            s = j * nf_seg
            if j > 0:
                # deepSI window convention (System_data.to_hist_future_data):
                #   uhist = [u[k-nb] ... u[k-1+nb_right]], ufuture = [u[k] ... u[k+nf-1]]
                # so the encoder window for the node at future offset s is entirely
                # inside this sample whenever s >= max(na, nb).
                # CHANGED (contiguity): the encoders reshape their inputs with .view, which
                # requires contiguity, and a time-axis slice of a (batch, nf, nu) tensor is not
                # contiguous (dim-0 stride stays nf*nu). deepSI's own to_hist_future_data hands
                # the encoder contiguous windows, so the n_seg = 1 path never sees this, but
                # EVERY n_seg > 1 call raised RuntimeError: at pre_encoder.py:450 for
                # encoder_init='linear_map' and at interconnect.py:384 for 'default', i.e. the
                # segmented path could not run with either encoder. Fixed at the caller because
                # it is the caller violating the encoder's input contract; the copy is one small
                # window per interior node and does not touch the n_seg = 1 path at all.
                x_node = self.encoder(
                    ufuture[:, s - self.nb: s + nb_right].contiguous(),
                    yfuture[:, s - self.na: s + na_right].contiguous())
                defects.append(x_node - x)      # gradient flows BOTH into the encoder
                x = x_node                      # and back through segment j-1's rollout
            # CHANGED (closed-loop seam): the segment rollout goes through self.simulate() rather
            # than calling self.hfn directly, so EXACTLY ONE rollout implementation exists in the
            # codebase. Bypassing the seam here would reproduce the bug this migration removes in
            # mirror image: before, the closed-loop loss silently dropped the defect terms; after,
            # multiple shooting would silently ignore the driver.
            y_seg, x = self.simulate(x, ufuture[:, s:s + nf_seg],
                                     yfuture[:, s:s + nf_seg], **Loss_kwargs)
            preds.append(y_seg)

        # Length-weighted sum over segments (Ribeiro et al. V_M); with equal segment
        # lengths that is the plain mean over all steps, matching the parent's scale.
        # One mse_loss over the concatenated prediction, matching the parent's reduction after
        # migration step 2b: identical in value to a mean of per-timestep values because every
        # timestep carries the same element count, and one autograd node instead of nf.
        loss_MSE = torch.nn.functional.mse_loss(yfuture, torch.cat(preds, dim=1))
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
