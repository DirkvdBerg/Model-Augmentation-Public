"""Fixed-reference trajectory orthogonality: geometry, projection, and exact chunking.

The GENERIC mathematical component of the plan in
`scripts/gantry/orthogonality/documentation/plan/fixed-reference-rollout-orthogonality-implementation.md`
(Sect. 9.1). It knows nothing about the gantry, the testbed, deepSI, controllers, or how a
rollout is produced. It is handed a stacked, weighted contribution vector and does linear
algebra on it. Both the synthetic testbed and the gantry adapter import THIS module, so a
bug found in the small system is fixed for both.

WHAT IT IS FOR. The pointwise penalty of `orth_projection.py` evaluates the network on the
slice where the augmented states are zero, where the latent-to-physical map is multiplied by
zero and receives no gradient. This module instead projects the augmentation's TOTAL
on/off rollout effect, which contains the direct correction, the latent route, and the
propagation of both.

THE THREE THINGS IT PROVIDES

  1. Geometry.  `Q_S = orth(M_E L S)` with `M_E = I - Q_E Q_E^T`, `Q_E = orth(L E)`.
     `S` and `E` are FULL AUGMENTED-ROLLOUT sensitivities of the scored output stack to the
     physical and encoder parameters, not one-step Jacobians. Encoder profiling comes first
     so that nuisance directions the encoder can absorb are not charged to the augmentation.
  2. Penalty.  `V = beta ||Q_S^T L d||^2`. Note `Q_S^T M_E = Q_S^T` because `Q_S` lies in
     `range(M_E)` and `M_E` is a symmetric projector, so profiling is embedded in the basis
     and the explicit `M_E` is redundant in the penalty (but NOT in the diagnostics, which
     take norms of `M_E L d` itself).
  3. Exact chunking.  `||sum_b P_b d_b||^2 != sum_b ||P_b d_b||^2`. A per-chunk penalty is a
     DIFFERENT objective, not an approximation of this one. `ProjectedResidualAccumulator`
     implements the two-pass form that is exact at bounded memory:
         pass 1, no graph : r = sum_b P_b d_b
         pass 2, r frozen : backward 2 beta <r, P_b d_b(q)> per chunk
     whose chunk gradients sum to the exact full-stack gradient, since
         grad beta ||r||^2 = 2 beta sum_b (d d_b / d q)^T P_b^T r.

WHAT IT DOES NOT DO, and no caller may imply otherwise. `V = 0` says the CURRENT contribution
is orthogonal to the REFERENCE sensitivity directions on the stack it was built for. It says
nothing about sensitivities at the current parameters if they have moved, nothing about the
network's attainable tangent directions, nothing about parameter recovery, and nothing about
prediction.
"""
__project_origin__ = "added"

import numpy as np
import torch
from torch import nn, Tensor


def _orth(M, rtol=1e-12, atol=0.0):
    """Orthonormal basis of the numerically supported column space, with its spectrum.

    Returns (Q, s, rank). The rank tolerance is RECORDED rather than hidden: the singular
    spectrum of the gantry regressor spans some twenty orders of magnitude, so where the cut
    falls is a reported choice, not a detail.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0 or M.shape[1] == 0:
        return np.zeros((M.shape[0], 0)), np.zeros(0), 0
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0 or s[0] == 0.0:
        return np.zeros((M.shape[0], 0)), s, 0
    keep = int(np.sum(s > max(rtol * s[0], atol)))
    return U[:, :keep], s, keep


def _np64(x):
    """Any array-like to a float64 numpy array, INCLUDING a CUDA tensor.

    `np.asarray` raises on a CUDA tensor ("can't convert cuda:0 device type tensor to numpy"),
    which is what broke the first GPU run of the geometry build (job 82430 would have hit it
    right after the `.to()` fix). The geometry is deliberately assembled in float64 on the host,
    because the SVDs and the rank decision must not be done in float32, and it is moved to the
    model dtype and device afterwards by `to_model`. So coming off the GPU here is correct, not
    a workaround. Mirrors the `detach().cpu().numpy()` pattern already used by `save_state`.
    """
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


class TrajectoryProjectionGeometry(nn.Module):
    """Frozen reference geometry. Buffers only; never a parent of a live model.

    Built once at a documented reference checkpoint and then held fixed. It must NOT hold a
    registered reference to the live `hfn` or encoder: `Interconnect` warns that registering
    the same module twice can duplicate optimizer parameter groups.

    Parameters
    ----------
    S : array (N, r)      d(scored output stack) / d(physical parameters), full rollout
    E : array (N, n_xi)   d(scored output stack) / d(encoder parameters), or None if the
                          encoder is fixed, in which case M_E = I
    L : array (N, N) or (N,) or None
        Loss weight factor with `L^T L = W`. A 1-D array is treated as a diagonal. `None`
        means identity. Output normalisation and any 1/N averaging belong HERE, so that the
        penalty lives in the same metric as the prediction loss.
    """

    def __init__(self, S, E=None, L=None, rank_rtol=1e-12, rank_atol=0.0,
                 dtype=torch.float64, metadata=None, Q_E=None, strict=True):
        """`Q_E` : an ALREADY ORTHONORMAL nuisance basis, used instead of orthonormalising `E`.

        CHANGED 2026-09-08. The gantry builds `range(E)` by the compressed factorisation of
        plan Eq. (15) (`compressed_nuisance_basis` below) because `E` itself has one row per
        scored output and never needs to exist. Passing the basis in is therefore the normal
        gantry path, not an optimisation: `E` and `Q_E` are mutually exclusive.

        `strict=False` turns the zero-profiled-rank case from an exception into a REPORTED
        result. The specification's rank policy (Sect. 6.5) says `r_bar = 0` stops the
        protection pilot and must be diagnosed; a probe whose job is to measure the rank cannot
        do that if constructing the object raises.
        """
        super().__init__()
        S = _np64(S)
        N = S.shape[0]
        self.N, self.n_phys = N, S.shape[1]
        self.rank_rtol, self.rank_atol = float(rank_rtol), float(rank_atol)
        self.metadata = dict(metadata or {})

        # ---- weight factor, stored in the SMALLEST form that represents it ---------
        # OOM FIX 2026-09-07. This previously materialised `np.eye(N)` whenever the weight was
        # the identity, and expanded a diagonal weight to a dense NxN matrix, then registered
        # that as a buffer. At the gantry's N = 7200 that is 415 MB per geometry object in
        # float64, plus an equal numpy temporary while building, and re-anchoring (D-183) keeps
        # two alive at once. It crashed the machine. The identity is now never built, and a
        # diagonal weight stays a vector; only a genuinely dense L is stored dense.
        self._L_kind = 'identity'
        Lvec = Ldense = None
        if L is not None:
            L = _np64(L)
            if L.ndim == 1:
                assert L.shape == (N,), f'diagonal L must have {N} entries, got {L.shape}'
                self._L_kind, Lvec = 'diagonal', L
            else:
                assert L.shape == (N, N), f'dense L must be {N}x{N}, got {L.shape}'
                self._L_kind, Ldense = 'dense', L
        self._L_is_identity = self._L_kind == 'identity'

        def apply_L_np(M):
            if self._L_kind == 'identity':
                return M
            if self._L_kind == 'diagonal':
                return Lvec[:, None] * M if M.ndim == 2 else Lvec * M
            return Ldense @ M

        # ---- encoder profiling ---------------------------------------------------
        if E is not None and Q_E is not None:
            raise ValueError('pass E or Q_E, not both: they are two spellings of one space and '
                             'supplying both hides which one was actually used.')
        if Q_E is not None:
            QE = _np64(Q_E)
            assert QE.shape[0] == N, f'Q_E has {QE.shape[0]} rows, the stack has {N}'
            if QE.shape[1]:
                off = float(np.abs(QE.T @ QE - np.eye(QE.shape[1])).max())
                assert off < 1e-8, (
                    f'Q_E is not orthonormal (max |Q_E^T Q_E - I| = {off:.3e}). A non-orthonormal '
                    '"basis" makes M_E = I - Q_E Q_E^T not a projector, and every identity below '
                    'silently stops holding.')
            sE, rE = np.zeros(0), QE.shape[1]
        elif E is None:
            QE, sE, rE = np.zeros((N, 0)), np.zeros(0), 0
        else:
            QE, sE, rE = _orth(apply_L_np(np.asarray(E, dtype=np.float64)),
                               rank_rtol, rank_atol)
        self.rank_E = rE
        self.sv_E = sE

        LS = apply_L_np(S)
        Sbar = LS - QE @ (QE.T @ LS) if rE else LS
        # BUG FIXED 2026-09-07 (caught by V3). The rank of `Sbar` must be judged against the
        # PRE-PROFILING scale, not against `Sbar`'s own largest singular value. When profiling
        # removes all physical information, `Sbar` is numerical noise at ~1e-16 whose own top
        # singular value is also ~1e-16, so a purely relative test reports FULL rank and the
        # "no physical information survives" guard below never fires. Judging against
        # `||L S||_2` makes a numerically-zero `Sbar` come out as rank 0, which is the truth.
        s_LS = float(np.linalg.svd(LS, compute_uv=False)[0]) if LS.size else 0.0
        floor = max(rank_atol, rank_rtol * s_LS)
        QS, sS, rS = _orth(Sbar, rtol=rank_rtol, atol=floor)
        self.rank_S = rS
        self.sv_S = sS
        self.scale_LS = s_LS
        self.rank_S_before_profiling = _orth(LS, rank_rtol, rank_atol)[2]

        if rS == 0 and not strict:
            # REPORTED, not raised: see the `strict` note in __init__.
            self.zero_rank = True
        elif rS == 0:
            raise ValueError(
                'Encoder profiling removed ALL physical sensitivity directions '
                f'(rank {self.rank_S_before_profiling} -> 0). The encoder can reproduce '
                'every physical parameter effect on this stack, so no penalty can protect '
                'them here. Fix the experiment, do not lower the tolerance.')
        else:
            self.zero_rank = False

        if self._L_kind == 'diagonal':
            self.register_buffer('L_diag', torch.as_tensor(Lvec, dtype=dtype))
        elif self._L_kind == 'dense':
            self.register_buffer('L', torch.as_tensor(Ldense, dtype=dtype))
        self.register_buffer('Q_S', torch.as_tensor(QS, dtype=dtype))
        self.register_buffer('Q_E', torch.as_tensor(QE, dtype=dtype))

    # ------------------------------------------------------- serialisation (ADDED 2026-09-08)
    #
    # WHY A DEDICATED PAIR AND NOT A PLACEHOLDER CONSTRUCTOR. The first attempt at reloading a
    # frozen geometry built the object from a `zeros((N, 1))` stand-in for `S` and then
    # overwrote `Q_S`, `Q_E` and the rank counts. The projection still multiplied correctly, but
    # the reconstructed object was internally INCONSISTENT: `zero_rank` stayed True, `scale_LS`
    # stayed 0, `rank_atol` was silently the constructor default rather than the saved value, and
    # `sv_E` was empty. Every one of those feeds a decision somewhere (the rank policy, the
    # diagnostics, a later re-anchoring comparison), so an object that projects correctly while
    # describing itself wrongly is worse than one that fails.
    _STATE_SCALARS = ('N', 'n_phys', 'rank_rtol', 'rank_atol', 'rank_E', 'rank_S',
                      'scale_LS', 'rank_S_before_profiling', 'zero_rank')

    def save_state(self) -> dict:
        """Everything needed to rebuild an identical object. No live tensors, no graph."""
        st = {k: getattr(self, k) for k in self._STATE_SCALARS}
        st.update(
            sv_S=np.asarray(self.sv_S), sv_E=np.asarray(self.sv_E),
            Q_S=self.Q_S.detach().cpu().numpy(), Q_E=self.Q_E.detach().cpu().numpy(),
            L_kind=self._L_kind, metadata=dict(self.metadata))
        if self._L_kind == 'diagonal':
            st['L_diag'] = self.L_diag.detach().cpu().numpy()
        elif self._L_kind == 'dense':
            st['L'] = self.L.detach().cpu().numpy()
        return st

    @classmethod
    def from_state(cls, st, dtype=torch.float64, device=None):
        """Rebuild from `save_state`, restoring EVERY field rather than the ones projection uses.

        `dtype` and `device` must match the model the penalty will be applied to. They are
        arguments rather than inferred, because a float64 geometry silently promoting a float32
        rollout (or living on the wrong device) is the kind of mismatch that shows up as a
        confusing error deep inside a matmul, or worse as a silent upcast.
        """
        g = cls.__new__(cls)
        nn.Module.__init__(g)
        for k in cls._STATE_SCALARS:
            v = st[k]
            setattr(g, k, bool(v) if k == 'zero_rank' else
                    (int(v) if k in ('N', 'n_phys', 'rank_E', 'rank_S',
                                     'rank_S_before_profiling') else float(v)))
        g.sv_S = np.asarray(st['sv_S'])
        g.sv_E = np.asarray(st['sv_E'])
        g.metadata = dict(st.get('metadata') or {})
        g._L_kind = str(st.get('L_kind', 'identity'))
        g._L_is_identity = g._L_kind == 'identity'
        if g._L_kind == 'diagonal':
            g.register_buffer('L_diag', torch.as_tensor(st['L_diag'], dtype=dtype, device=device))
        elif g._L_kind == 'dense':
            g.register_buffer('L', torch.as_tensor(st['L'], dtype=dtype, device=device))
        g.register_buffer('Q_S', torch.as_tensor(st['Q_S'], dtype=dtype, device=device))
        g.register_buffer('Q_E', torch.as_tensor(st['Q_E'], dtype=dtype, device=device))
        assert g.Q_S.shape[0] == g.N, (
            f'the saved basis has {g.Q_S.shape[0]} rows but the record says N = {g.N}')
        assert g.Q_S.shape[1] == g.rank_S and g.Q_E.shape[1] == g.rank_E, (
            'the saved bases and the saved ranks disagree, so one of them is from another build')
        g.validate()
        return g

    def to_model(self, dtype=None, device=None):
        """Align the frozen buffers with the model they will be applied to. Returns self."""
        if dtype is not None or device is not None:
            for name in ('Q_S', 'Q_E', 'L_diag', 'L'):
                t = getattr(self, name, None)
                if t is not None:
                    setattr(self, name, t.to(dtype=dtype or t.dtype,
                                             device=device if device is not None else t.device))
        return self

    # ------------------------------------------------------------------ core ops
    def project(self, d: Tensor) -> Tensor:
        """r = Q_S^T L d. `M_E` is embedded in `Q_S`; see the module docstring."""
        Ld = self._apply_L(d.reshape(-1))
        return self.Q_S.T @ Ld

    def penalty(self, d: Tensor, beta: float) -> Tensor:
        """SUM convention: `beta ||Q^T L d||^2`, no 1/N. Kept for the existing callers."""
        r = self.project(d)
        return beta * (r * r).sum()

    def penalty_mse(self, d: Tensor, beta: float) -> Tensor:
        """PLAIN-MSE convention: `V = (beta/N) ||Q^T d||^2`, plan Eq. (11).

        ADDED 2026-09-08. The production loss is `nn.functional.mse_loss` over the scored slice,
        i.e. a MEAN over `N = n_W T n_y` entries, so a penalty that sums puts itself in a
        different metric from the loss it competes with, by exactly the factor `N`. The whitening
        `L = I/sqrt(N)` is a scalar and leaves every column span unchanged, which is why it is
        never instantiated -- and why it is also easy to drop by accident. It is NOT dropped
        here.

        A beta measured under the sum convention does not transfer: convert it by `N`, and say
        which convention a reported beta was measured in.
        """
        r = self.project(d)
        return beta * (r * r).sum() / self.N

    def _apply_L(self, v: Tensor) -> Tensor:
        if self._L_kind == 'identity':
            return v
        if self._L_kind == 'diagonal':
            return self.L_diag * v
        return self.L @ v

    def _profile(self, v: Tensor) -> Tensor:
        """M_E L v. Used by the DIAGNOSTICS, where the projector is not redundant."""
        Lv = self._apply_L(v)
        return Lv if self.Q_E.shape[1] == 0 else Lv - self.Q_E @ (self.Q_E.T @ Lv)

    @torch.no_grad()
    def contribution_metrics(self, d: Tensor, tol=1e-12) -> dict:
        """Separate ORTHOGONALITY from SHRINKAGE.

        A single ratio cannot do this. Uniform shrinking lowers both amplitudes and leaves
        `ell` flat; selective removal of the in-span part lowers `ell`. Reporting only a ratio
        also lets a penalty look successful by switching the augmentation off, which is why
        the absolute amplitudes are returned too and must be quoted with it.

        A projected numerator over an UNprofiled denominator would conflate overlap with
        removal of encoder directions; `norm` below is `||M_E L d||`, matching the numerator.
        """
        dbar = self._profile(d.reshape(-1))
        nb = float(dbar.norm())
        a_par = float((self.Q_S.T @ dbar).norm())
        a_perp = float((dbar - self.Q_S @ (self.Q_S.T @ dbar)).norm())
        ell = a_par / nb if nb > tol else float('nan')
        return dict(ell=ell, a_par=a_par, a_perp=a_perp, norm_profiled=nb,
                    norm_raw=float(d.reshape(-1).norm()))

    def summary(self) -> str:
        s = (f'TrajectoryProjectionGeometry  N={self.N}  n_phys={self.n_phys}\n'
             f'  rank(S) before profiling : {self.rank_S_before_profiling}\n'
             f'  rank(Q_E)                : {self.rank_E}\n'
             f'  rank(Q_S) after profiling: {self.rank_S} of {self.n_phys}\n'
             f'  rank tolerance           : rtol {self.rank_rtol:.1e} '
             f'atol {self.rank_atol:.1e}')
        if self.rank_S < self.n_phys:
            s += ('\n  WARNING: physical directions were lost. The penalty cannot protect '
                  'information the geometry does not contain.')
        return s

    def validate(self, atol=None):
        """Cheap invariants. Cheaper to assert than to debug in a training run.

        THE TOLERANCE FOLLOWS THE DTYPE. Fixed 2026-09-08: this asserted `1e-10` unconditionally,
        which float64 meets and float32 cannot -- float32 eps is about `1.2e-7`, so an orthonormal
        basis cast to float32 has residuals near `1e-6` and is not defective for having them. The
        geometry is now cast to the model's dtype (a float32 checkpoint is the normal gantry
        case), so a fixed float64 tolerance rejected every correct float32 geometry. Scaled by
        the basis width because the residual accumulates over the inner product's length.
        """
        k = self.Q_S.shape[1]
        if atol is None:
            eps = torch.finfo(self.Q_S.dtype).eps
            atol = max(1e-10, 20.0 * eps * max(k, 1) ** 0.5)
        I = torch.eye(k, dtype=self.Q_S.dtype, device=self.Q_S.device)
        off = (self.Q_S.T @ self.Q_S - I).abs().max().item() if k else 0.0
        assert off <= atol, (
            f'Q_S is not orthonormal: max |Q_S^T Q_S - I| = {off:.3e} against {atol:.3e} for '
            f'{self.Q_S.dtype}')
        if self.Q_E.shape[1] and k:
            cross = (self.Q_E.T @ self.Q_S).abs().max().item()
            assert cross <= atol, (
                f'Q_E is not orthogonal to Q_S (max {cross:.2e} against {atol:.3e})')
        return True


class ProjectedResidualAccumulator:
    """Exact two-pass chunked penalty. See the module docstring for why chunking is subtle.

    Usage, inside ONE optimizer step, with parameters unchanged between the passes:

        acc = ProjectedResidualAccumulator(geom, beta)
        with torch.no_grad():                       # pass 1: form r, no graph
            for d_b in chunks: acc.accumulate_value(d_b)
        acc.freeze()
        for d_b in chunks:                          # pass 2: r frozen, graph live
            acc.backward_chunk(d_b)

    `d_b` must be the SAME chunk contributions in both passes. The chunks partition ONE
    mathematical stack, so `r` from pass 1 is a property of the whole set and is a constant
    covector in pass 2.
    """

    def __init__(self, geometry: TrajectoryProjectionGeometry, beta: float):
        self.g = geometry
        self.beta = float(beta)
        self._r = None
        self._frozen = None

    def accumulate_value(self, d_chunk: Tensor, rows: slice = None):
        """Pass 1. `rows` selects this chunk's rows of the stack when chunking by window."""
        assert self._frozen is None, 'accumulate_value after freeze()'
        r = self._chunk_project(d_chunk, rows)
        self._r = r if self._r is None else self._r + r
        return self

    def freeze(self):
        assert self._r is not None, 'freeze() before any accumulate_value'
        self._frozen = self._r.detach().clone()
        return self._frozen

    @property
    def value(self) -> float:
        assert self._frozen is not None, 'value before freeze()'
        return self.beta * float((self._frozen * self._frozen).sum())

    def backward_chunk(self, d_chunk: Tensor, rows: slice = None, retain_graph=False):
        """Pass 2. Backward `2 beta <r, P_b d_b>`; chunk gradients sum to the exact total."""
        assert self._frozen is not None, 'backward_chunk before freeze()'
        surrogate = 2.0 * self.beta * (self._frozen * self._chunk_project(d_chunk, rows)).sum()
        surrogate.backward(retain_graph=retain_graph)
        return float(surrogate.detach())

    def _chunk_project(self, d_chunk: Tensor, rows):
        d_chunk = d_chunk.reshape(-1)
        if rows is None:
            return self.g.project(d_chunk)
        # Row-restricted: r_b = Q_S[rows]^T (L d)[rows]. Requires L block-diagonal over the
        # chunking (true for a diagonal L, i.e. per-sample weights). A dense L that mixes
        # rows across chunks cannot be chunked this way.
        assert self.g._L_kind in ('identity', 'diagonal'), (
            'row-chunked projection needs an identity or diagonal L; a dense L mixes rows '
            'across chunks and cannot be split this way')
        Ld = d_chunk if self.g._L_kind == 'identity' else self.g.L_diag[rows] * d_chunk
        return self.g.Q_S[rows].T @ Ld


def compressed_nuisance_basis(gammas, jacs, rtol=1e-12, atol=0.0, row_slices=None, N=None):
    """`range(E)` from per-window factors, WITHOUT ever forming `E`. Plan Eq. (13)-(15).

    ADDED 2026-09-08.

    Parameters
    ----------
    gammas : list of (T n_y, n_x) arrays
        `Gamma_w = D_{x_w,0} p_w`, the sensitivity of window `w`'s scored outputs to its own
        initial PHYSICAL state. Windows do not interact, so this is block diagonal globally and
        is never assembled as such.
    jacs : list of (n_x, n_xi) arrays
        `J_w = D_xi_p e_p(H_w)`, the Jacobian of the encoder alone. No rollout is differentiated
        with respect to an encoder weight anywhere in this construction.

    Returns (Q_E, info) with `Q_E` orthonormal and `range(Q_E) = range(E)` BEFORE numerical
    truncation. The truncation threshold is an approximation with a recorded value, not part of
    the identity; both spectra are returned so the choice is visible.

    WHY THIS SHAPE. `E = blockdiag_w(Gamma_w) stack_w(J_w)` has `N` rows, and `N` is the whole
    scored trajectory. Writing `Gamma_w = U_w C_w` with `U_w` orthonormal gives `E = U_G Z` with
    `Z = stack_w(C_w J_w)` of size `r_G x n_xi`, `r_G = sum_w rank(Gamma_w) <= n_x n_W`. The
    saving is against `E`, whose row count carries the trajectory; it is NOT automatically a
    saving against the stacked `J_w`, and `Z` can still be too large. Measure before scaling up.

    `Z Z^T` is deliberately not used: it squares the conditioning of the thing whose small
    singular values decide the rank.
    """
    n_w = len(gammas)
    assert len(jacs) == n_w, 'one Jacobian per window'
    if row_slices is None:
        offs, acc = [], 0
        for G in gammas:
            offs.append(slice(acc, acc + G.shape[0]))
            acc += G.shape[0]
        row_slices, N = offs, (N if N is not None else acc)
    U_blocks, Z_blocks, ranks, sv_G = [], [], [], []
    for G, J in zip(gammas, jacs):
        G = _np64(G)
        U, s, _ = np.linalg.svd(G, full_matrices=False)
        r = int(np.sum(s > max(rtol * (s[0] if s.size else 0.0), atol))) if s.size else 0
        U = U[:, :r]
        U_blocks.append(U)
        # C_w = U_w^T Gamma_w, THEN Z_w = C_w J_w. Both factors are required.
        #
        # BUG FIXED 2026-09-08, caught in review, not by this suite. The first version wrote
        # `U.T @ J` and omitted `Gamma_w` entirely. That is not a small numerical error: it is
        # dimensionally invalid the moment `T n_y != n_x` (on the gantry, `U.T` is `r x 1200`
        # against a `6 x 6774` Jacobian), so it raises rather than returning a wrong basis. The
        # reason no test caught it is the lesson, and it is recorded in the verification report:
        # the small reference case implemented this construction a SECOND time, correctly, and
        # compared that private copy against explicit `E`. It never called this function.
        # `test_trajectory_projection.v3b_compressed_nuisance` and the `A5b` check in
        # `trajectory_ownership_geometry.py` now both call THIS function.
        C = U.T @ G
        Z_blocks.append(C @ _np64(J))
        ranks.append(r)
        sv_G.append(s.tolist())
    r_G = int(sum(ranks))
    Z = np.vstack(Z_blocks) if r_G else np.zeros((0, jacs[0].shape[1]))
    if r_G == 0:
        return np.zeros((N, 0)), dict(r_G=0, rank_E=0, sv_Z=[], sv_Gamma=sv_G, ranks_w=ranks)
    UZ, sZ, _ = np.linalg.svd(Z, full_matrices=False)
    rE = int(np.sum(sZ > max(rtol * sZ[0], atol))) if sZ.size and sZ[0] > 0 else 0
    # Q_E = U_G U_Z[:, :rE], expanded block by block. U_G is never materialised densely.
    Q_E = np.zeros((N, rE))
    c0 = 0
    for sl, U in zip(row_slices, U_blocks):
        k = U.shape[1]
        if k:
            Q_E[sl, :] = U @ UZ[c0:c0 + k, :rE]
        c0 += k
    return Q_E, dict(r_G=r_G, rank_E=rE, sv_Z=sZ.tolist(), sv_Gamma=sv_G, ranks_w=ranks,
                     rtol=float(rtol), atol=float(atol), Z_shape=list(Z.shape))


def penalty_full_stack(geometry, d, beta):
    """Reference single-stack penalty. The two-pass path must reproduce this exactly."""
    return geometry.penalty(d, beta)


def subspace_drift(Q_prev, Q_new) -> dict:
    """Principal angles between two projection bases, in degrees. D-183 drift logging.

    The frozen-basis scheme's validity has only ever been assumed. This turns "is the basis
    stale" into a number, and distinguishes the two outcomes that matter:
        angles decaying toward zero  -> theta is settling; freezing was defensible
        angles persisting or growing -> the network is chasing a moving constraint set, which
                                        is the instability re-anchoring risks
    A RANK CHANGE is itself a drift signal and is reported separately: it means a physical
    direction entered or left the numerically supported span, which no angle can express.
    """
    A = Q_prev.detach().cpu().numpy() if hasattr(Q_prev, 'detach') else np.asarray(Q_prev)
    B = Q_new.detach().cpu().numpy() if hasattr(Q_new, 'detach') else np.asarray(Q_new)
    if A.shape[1] == 0 or B.shape[1] == 0:
        return dict(max_deg=float('nan'), mean_deg=float('nan'),
                    rank_prev=A.shape[1], rank_new=B.shape[1], n_angles=0)
    c = np.clip(np.linalg.svd(A.T @ B, compute_uv=False), -1.0, 1.0)
    ang = np.degrees(np.arccos(c))
    return dict(max_deg=float(ang.max()), mean_deg=float(ang.mean()),
                rank_prev=int(A.shape[1]), rank_new=int(B.shape[1]), n_angles=int(ang.size),
                angles_deg=[float(a) for a in ang])


def assert_geometry_detached(geometry) -> bool:
    """The geometry buffers carry no autograd graph. NOT the gradient policy.

    RENAMED 2026-09-07 after review. The old name `assert_no_gradient_path` claimed far more
    than this function checks: it inspects the STORED BASIS only, and is blind to whether the
    caller detached `theta` when building `d`. A probe passed the old assertion while the
    penalty's gradient into the physical parameters was 36. Use `assert_gradient_policy` for
    the policy itself; this remains useful as the narrower check that a rebuilt basis was not
    constructed inside the graph.
    """
    for name in ('Q_S', 'Q_E', 'L', 'L_diag'):
        t = getattr(geometry, name, None)
        if t is None:
            continue
        assert not t.requires_grad and t.grad_fn is None, (
            f'{name} carries a gradient path (D-182). The basis must be rebuilt under '
            'torch.no_grad() and installed as a buffer, or theta can reduce the penalty by '
            'rotating the subspace instead of the network being corrected.')
    return True


# Backwards-compatible alias for existing callers; the name overstates what it does.
assert_no_gradient_path = assert_geometry_detached


def assert_gradient_policy(penalty_scalar, forbidden: dict, allowed=None,
                           tol=0.0, retain_graph=False) -> dict:
    """THE ACTUAL D-182 CHECK: backward the penalty ALONE and require forbidden groups empty.

    This is what `assert_geometry_detached` was mistakenly assumed to do. The failure it catches
    is silent: if `theta` has a gradient path into the penalty, the optimiser can reduce the
    penalty by MOVING THE SUBSPACE rather than by correcting the network, which corrupts the
    quantity the method exists to protect while the penalty value falls and the run looks fine.

    Parameters
    ----------
    penalty_scalar : the penalty term, and ONLY that term, not the total loss.
    forbidden : {'physical': [tensors], 'encoder': [...]} groups that must receive nothing.
    allowed : optional {'augmentation': [...]} groups that must receive SOMETHING, which
        catches the opposite failure of a penalty that is silently inert.

    Gradients are saved and restored, so this can run inside a training loop.
    """
    import torch as _t
    groups = {**forbidden, **(allowed or {})}
    saved = {id(p): (None if p.grad is None else p.grad.detach().clone())
             for ts in groups.values() for p in ts}
    for ts in groups.values():
        for p in ts:
            p.grad = None
    penalty_scalar.backward(retain_graph=retain_graph)

    report = {}
    for gname, ts in forbidden.items():
        worst = 0.0
        for p in ts:
            if p.grad is not None:
                worst = max(worst, float(p.grad.abs().max()))
        report[gname] = worst
        assert worst <= tol, (
            f'D-182 VIOLATED: the penalty put gradient {worst:.3e} into the "{gname}" group '
            f'(tolerance {tol:g}). The optimiser can now reduce the penalty by moving the '
            'subspace instead of correcting the network. Pass these parameters into the '
            'penalty rollout as DETACHED constants.')
    for gname, ts in (allowed or {}).items():
        got = max((float(p.grad.abs().max()) for p in ts if p.grad is not None), default=0.0)
        report[gname] = got
        assert got > 0.0, (
            f'the penalty produced NO gradient into "{gname}"; it is inert and cannot be '
            'doing what it is reported to do')

    for ts in groups.values():
        for p in ts:
            p.grad = saved[id(p)]
    return report
