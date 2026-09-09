"""The gantry adapter: the ONE seam between the shared projection algebra and production code.

Specification Sect. 9.1. Everything mathematical is imported from
`model_augmentation/fit_systems/trajectory_orth_projection.py`; everything about how a rollout
is produced is delegated to `fit_sys.encoder`, `fit_sys.simulate` and the attached
`ClosedLoopSimulator`. There is deliberately NO private rollout in this file: a second
implementation of the physics is how a diagnostic and a training run stop being the same
experiment, and the previous adapter's parity was only ever against another private loop.

OWNERSHIP, spec Eqs. (7)-(8), and it is asserted rather than described:

    physical     lambda = the parameterised physical block's `log_params`
    augmentation eta    = the ANN weights (eta_d) AND the latent-initialisation branch of the
                          split encoder (eta_a), plus any trainable augmentation gate
    encoder      xi_p   = the PHYSICAL-initialisation branch of the split encoder only

`check_ownership` in `trajectory_adapter.py` proves disjointness by tensor identity; the
perturbation checks in `checks.py` prove that the tensors actually behave that way, because
identity alone does not. Two disjoint tensor SETS can still both feed one shared subnetwork,
which is precisely the state the encoder split exists to leave.
"""
__project_origin__ = "added"

import contextlib

import numpy as np
import torch

from model_augmentation.fit_systems.trajectory_adapter import TrajectoryAdapter
from .encoder_split import SplitEncoderInitAug

MODES = ('full', 'off', 'clamped')


# --------------------------------------------------------------------------- intervention
@contextlib.contextmanager
def ann_gate(ann_block, mask):
    """Scope an ANN output-gate mask, restoring the previous one even on an exception.

    The gate is a NON-PERSISTENT BUFFER on the existing ANN output path (blocks.py), not a
    wrapper and not a weight mutation, so module registration, the state dict and the
    optimiser's parameter list are all unchanged inside the scope.

    ONE CAVEAT, and it is real. `torch.utils.checkpoint` re-runs the forward during the BACKWARD
    pass, by which time this context manager has exited and the buffer holds whatever the global
    mode then is. A graph-bearing mode must therefore complete its backward INSIDE its own scope,
    or `cfg.checkpoint_chunk` must be 0. `checks.check_gate_checkpoint_replay` is the executable
    form of that statement, not a comment about it.
    """
    prev = ann_block.out_gate.detach().clone()
    ann_block.out_gate = torch.as_tensor(mask, dtype=prev.dtype, device=prev.device)
    try:
        yield
    finally:
        ann_block.out_gate = prev


def gate_masks(route_ix, n_phys):
    """full / off / clamped gate vectors over ANN OUTPUT COLUMNS.

    `route_ix[j]` is the STATE ROW that ANN output `j` is written into, so the classification is
    on `route_ix[j] < n_phys` and the mask is indexed by `j`. Confusing the two silently gates
    the wrong channels whenever the routing is not the identity.
    """
    route_ix = np.asarray(route_ix)
    n_out = len(route_ix)
    return dict(full=np.ones(n_out), off=np.zeros(n_out),
                clamped=(route_ix < n_phys).astype(float))


# ------------------------------------------------------------------------------- adapter
class GantryTrajectoryAdapter(TrajectoryAdapter):
    """Scored stacks, interventions and per-window geometry factors from production code."""

    def __init__(self, env, windows, burn_in=None):
        self.env = env
        self.fit_sys = env.fit_sys
        self.cfg = env.cfg
        self.w = windows
        self.enc = self.fit_sys.encoder
        self.ann = env.ann_block
        self.phys = env.phys_block
        self.n_win = int(self.w['ufuture'].shape[0])
        self.burn_in = env.cfg.burn_in if burn_in is None else int(burn_in)
        self.T = int(self.w['ufuture'].shape[1]) - self.burn_in
        self.ny = int(self.w['yfuture'].shape[2])
        self._masks = gate_masks(self.cfg.ann_route_ix, self.cfg.nx_phys)
        assert self.T > 0, 'burn_in leaves nothing to score'

    # ------------------------------------------------------------------ rebinding
    def rebind(self, fit_sys=None):
        """Re-resolve the cached module references from the CURRENT fit system.

        WHY THIS EXISTS, and it is a defect this class had until 2026-09-08. `__init__` caches
        `enc`, `ann` and `phys` as snapshots, while `self.fit_sys` is only a reference to the
        system OBJECT. deepSI's `checkpoint_load_system` does `self.__dict__ = torch.load(file)`,
        which keeps the system object's identity but replaces every module inside it. After such
        a load the adapter is spliced across two model generations:

            self.enc / self.ann / self.phys   the OLD modules
            self.fit_sys.simulate(...)        the NEWLY LOADED dynamics

        The consequences are silent and severe rather than merely inaccurate:

          - `ann_gate` gates an ANN block that is no longer in the rollout, so the gating half
            of the `off` and `clamped` interventions is applied to nothing and the intervention
            becomes INCORRECT. It does not become empty: `scored_stack` zeros the latent
            initialisation for every non-full mode independently of the gate, so `d` generally
            stays nonzero and the penalty keeps reporting a plausible value. What is being
            penalised is then a different, unspecified intervention
          - `parameter_groups` hands the optimizer and the ownership checks tensors that the
            live model no longer contains, so gradients land on orphaned parameters
          - `physical_jacobian` substitutes `log_params` into the new `hfn` by NAME, but reads
            the old block's dtype and device

        None of that raises. Rebinding after any operation that replaces the model is therefore
        mandatory, not hygienic, and it is called from `checkpoint_load_system`.
        """
        from model_augmentation.fit_systems.blocks import (
            Static_ANN_Block, Gantry_State_Block)
        if fit_sys is not None:
            self.fit_sys = fit_sys
        fs = self.fit_sys
        self.enc = fs.encoder
        blocks = list(fs.hfn.connected_blocks)
        self.ann = next(m for m in blocks if isinstance(m, Static_ANN_Block))
        self.phys = next(m for m in blocks if isinstance(m, Gantry_State_Block))
        if getattr(self, 'env', None) is not None:
            self.env.fit_sys = fs
        return self

    def assert_bound(self):
        """Every cached reference is reachable from the live fit system. Cheap, and it fires."""
        from model_augmentation.fit_systems.blocks import (
            Static_ANN_Block, Gantry_State_Block)
        fs = self.fit_sys
        blocks = list(fs.hfn.connected_blocks)
        problems = []
        if self.enc is not fs.encoder:
            problems.append('encoder')
        if self.ann is not next((m for m in blocks if isinstance(m, Static_ANN_Block)), None):
            problems.append('ann')
        if self.phys is not next((m for m in blocks if isinstance(m, Gantry_State_Block)), None):
            problems.append('physical block')
        if problems:
            raise RuntimeError(
                'the adapter is bound to stale module(s): %s. The live model was replaced (a '
                'checkpoint load does this) without rebinding, so the interventions would gate '
                'modules outside the rollout and the parameter groups would be orphaned. Call '
                'adapter.rebind().' % ', '.join(problems))
        return True

    # ---------------------------------------------------------------- description
    def n_rows(self):
        return self.n_win * self.T * self.ny

    def metadata(self):
        m = self.env.manifest()
        m.update(n_windows=self.n_win, T_scored=self.T, N=self.n_rows(),
                 stack_order='window-major, then time, then channel',
                 window_provenance=self.w.get('provenance'))
        return m

    def loss_weight(self):
        """None = identity. The 1/N of the plain-MSE convention lives in `penalty_mse`.

        Deliberately NOT expressed as a diagonal `L = I/sqrt(N)`: a scalar leaves every column
        span unchanged, so instantiating it would buy nothing and would cost an N-vector buffer
        per geometry object.
        """
        return None

    # ---------------------------------------------------------------- parameter groups
    def parameter_groups(self):
        enc = self.enc
        if not isinstance(enc, SplitEncoderInitAug):
            raise RuntimeError(
                'the attached encoder is %s, not a SplitEncoderInitAug. Before the split the '
                'physical and latent initialisations share one nonlinear correction net, so '
                '"disjoint parameter groups" is not a property this configuration has, and the '
                'ownership contract cannot be honoured by slicing outputs.' % type(enc).__name__)
        return {
            'physical': [self.phys.log_params],
            'augmentation': list(self.ann.parameters()) + enc.latent_parameters(),
            'encoder': enc.physical_parameters(),
        }

    def group_named(self):
        enc = self.enc
        return {
            'physical': [('phys.log_params', self.phys.log_params)],
            'augmentation': ([(f'ann.{n}', p) for n, p in self.ann.named_parameters()]
                             + [('enc.Wa_psi_u', enc.Wa_psi_u), ('enc.Wa_psi_y', enc.Wa_psi_y)]
                             + [(f'enc.net_a.{n}', p) for n, p in enc.net_a.named_parameters()]),
            'encoder': ([('enc.Wb_psi_u', enc.Wb_psi_u), ('enc.Wb_psi_y', enc.Wb_psi_y)]
                        + [(f'enc.net_p.{n}', p) for n, p in enc.net_p.named_parameters()]),
        }

    # ---------------------------------------------------------------- encoding
    def encode(self, ix=None):
        """(x_phys_0, x_lat_0) from the PRODUCTION encoder, once per evaluation.

        The specification is explicit that a window's history is encoded ONCE per evaluation and
        the physical initial value reused across the three interventions, so that full, off and
        clamped differ by the intervention and by nothing else.
        """
        uh = self.w['uhist'] if ix is None else self.w['uhist'][ix]
        yh = self.w['yhist'] if ix is None else self.w['yhist'][ix]
        return self.enc.encode_split(uh, yh)

    @staticmethod
    def x0_from(xp, xa):
        return torch.cat([xp, xa], dim=1)

    # ---------------------------------------------------------------- rollout
    def _signals(self, ix):
        uf = self.w['ufuture'] if ix is None else self.w['ufuture'][ix]
        yf = self.w['yfuture'] if ix is None else self.w['yfuture'][ix]
        cix = None
        if 'ctrl_ix' in self.w:
            cix = self.w['ctrl_ix'] if ix is None else self.w['ctrl_ix'][ix]
        return uf, yf, cix

    def simulate(self, x0, ix=None):
        """The production driven rollout. `fit_sys.simulate` routes to the closed-loop seam."""
        uf, yf, cix = self._signals(ix)
        kw = {} if cix is None else dict(ctrl_ix=cix)
        y_pred, _ = self.fit_sys.simulate(x0, uf, yf, **kw)
        return y_pred

    def score(self, y_pred):
        """The scored slice, flattened window-major/time/channel. Mirrors the production loss."""
        return y_pred[:, self.burn_in:].reshape(-1)

    def scored_stack(self, params=None, mode='full', rows=None, ix=None, x0=None):
        """p_mode for the selected windows, from ONE code path differing only by the gate."""
        if x0 is None:
            xp, xa = self.encode(ix)
            if mode != 'full':
                xa = torch.zeros_like(xa)
            x0 = self.x0_from(xp, xa)
        with ann_gate(self.ann, self._masks[mode]):
            p = self.score(self.simulate(x0, ix))
        return p if rows is None else p[rows]

    def contributions(self, ix=None):
        """d, d_lat, d_dir sharing ONE encode and ONE rollout path, as the contract requires."""
        xp, xa = self.encode(ix)
        z = torch.zeros_like(xa)
        p_full = self.scored_stack(mode='full', ix=ix, x0=self.x0_from(xp, xa))
        p_off = self.scored_stack(mode='off', ix=ix, x0=self.x0_from(xp, z))
        p_cl = self.scored_stack(mode='clamped', ix=ix, x0=self.x0_from(xp, z))
        return dict(full=p_full, off=p_off, clamped=p_cl,
                    total=p_full - p_off, lat=p_full - p_cl, direct=p_cl - p_off)

    # ---------------------------------------------------------------- production parity
    def production_loss(self, ix=None):
        """fit_sys's OWN loss on these windows. The primary parity target of Sect. 9.2."""
        sel = slice(None) if ix is None else ix
        kw = {}
        if 'ctrl_ix' in self.w:
            kw['ctrl_ix'] = self.w['ctrl_ix'][sel]
        return self.fit_sys.loss(self.w['uhist'][sel], self.w['yhist'][sel],
                                 self.w['ufuture'][sel], self.w['yfuture'][sel], **kw)

    # ---------------------------------------------------------------- geometry factors
    def _log_param_name(self):
        for n, p in self.fit_sys.hfn.named_parameters():
            if p is self.phys.log_params:
                return n
        raise RuntimeError('log_params is not a registered parameter of hfn')

    def _stack_with_log_params(self, v, ix=None, mode='full'):
        """Scored stack with `log_params` functionally replaced by `v` for the WHOLE rollout.

        `torch.func.functional_call` substitutes for ONE module call; the closed-loop rollout
        calls `hfn` `nf` times, so the substitution is scoped around the whole rollout with
        `torch.nn.utils.stateless._reparametrize_module`, which is what `functional_call` itself
        uses internally. NOTHING IS DELETED FROM `_parameters`: the specification forbids
        removing parameters from the registry to accommodate forward-mode duals, and doing so
        would drop the tensor from the optimiser as a silent side effect.
        """
        from torch.nn.utils.stateless import _reparametrize_module
        from model_augmentation.fit_systems.closed_loop import closed_loop_rollout

        name = self._log_param_name()
        root = self.fit_sys.hfn
        xp, xa = self.encode(ix)
        if mode != 'full':
            xa = torch.zeros_like(xa)
        x0 = self.x0_from(xp, xa)
        uf, yf, cix = self._signals(ix)

        with ann_gate(self.ann, self._masks[mode]):
            with _reparametrize_module(root, {name: v}, tie_weights=False):
                sim = self.fit_sys.simulator
                if sim is None:
                    ys, x = [], x0
                    for u in uf.unbind(1):
                        yhat, x = root(x, u)
                        ys.append(yhat)
                    y_pred = torch.stack(ys, dim=1)
                else:
                    y_pred, _, _ = closed_loop_rollout(
                        root, root.output_only, uf, yf, x0, sim.bank, cix.long(),
                        chunk=sim.checkpoint_chunk)
        return self.score(y_pred)

    def physical_directional(self, direction, ix=None, mode='full'):
        """One forward-mode JVP of the scored stack along a physical log-coordinate direction."""
        lp = self.phys.log_params.detach()
        v = torch.as_tensor(direction, dtype=lp.dtype, device=lp.device).reshape(lp.shape)

        def f(w):
            return self._stack_with_log_params(w, ix=ix, mode=mode)

        _, jv = torch.func.jvp(f, (lp,), (v,))
        return jv.detach()

    def _clear_phys_cache(self):
        """Drop the physical block's per-forward matrix cache.

        Called after every `torch.func` transform. Inside a JVP or a jacrev the block caches
        functorch dual tensors in `_cur`, and leaving them there hands a `TensorWrapper` to
        whatever runs next -- `torch.save` raises "Cannot access storage of TensorWrapper", which
        is how this was found. `blocks.py` also drops `_cur` from its serialised state, so this
        is belt and braces; it costs one attribute assignment and removes a class of confusing
        downstream failures.
        """
        self.phys._cur = None

    def physical_jacobian(self, ix=None, mode='full'):
        """S = D_lambda p as an (N, 14) array, one forward JVP per log coordinate.

        FORWARD, not reverse: the stack has `N` rows (thousands) against 14 columns, so reverse
        mode costs one pass per output row. Batching the 14 tangents together is a benchmark
        question, not a correctness one, and is measured separately rather than assumed.
        """
        n = self.phys.log_params.numel()
        cols = []
        for j in range(n):
            e = torch.zeros(n, dtype=self.phys.log_params.dtype,
                            device=self.phys.log_params.device)
            e[j] = 1.0
            cols.append(self.physical_directional(e, ix=ix, mode=mode).cpu().numpy())
        self._clear_phys_cache()
        return np.stack(cols, axis=1).astype(np.float64)

    def initial_state_jacobians(self, ix=None):
        """Gamma_w = D_{x_phys,0} p_w, per window, batched over the six direction labels.

        Windows do not interact, so the six labels are vectorised ACROSS windows: six rollouts of
        the whole batch, not six per window. Globally `Gamma` has `6 n_W` columns and is block
        diagonal; the blocks are returned and the block-diagonal is never assembled. Six columns
        PER WINDOW is not a six-dimensional global nuisance space.
        """
        n_sel = self.n_win if ix is None else len(ix)
        xp, xa = self.encode(ix)
        xp, xa = xp.detach(), xa.detach()
        cols = []
        for j in range(xp.shape[1]):
            tang = torch.zeros_like(xp)
            tang[:, j] = 1.0

            def f(v):
                return self.scored_stack(mode='full', ix=ix, x0=self.x0_from(v, xa))

            _, jv = torch.func.jvp(f, (xp,), (tang,))
            cols.append(jv.detach().reshape(n_sel, self.T * self.ny))
        G = torch.stack(cols, dim=-1)
        self._clear_phys_cache()
        return [G[w].cpu().numpy().astype(np.float64) for w in range(n_sel)]

    def latent_init_jacobians(self, ix=None):
        """D_{a_0} p_full per window. For VERIFYING the latent gradient; NOT part of E."""
        n_sel = self.n_win if ix is None else len(ix)
        xp, xa = self.encode(ix)
        xp, xa = xp.detach(), xa.detach()
        cols = []
        for j in range(xa.shape[1]):
            tang = torch.zeros_like(xa)
            tang[:, j] = 1.0

            def f(v):
                return self.scored_stack(mode='full', ix=ix, x0=self.x0_from(xp, v))

            _, jv = torch.func.jvp(f, (xa,), (tang,))
            cols.append(jv.detach().reshape(n_sel, self.T * self.ny))
        G = torch.stack(cols, dim=-1)
        self._clear_phys_cache()
        return [G[w].cpu().numpy().astype(np.float64) for w in range(n_sel)]

    def _named_xip(self):
        enc = self.enc
        out = [('Wb_psi_u', enc.Wb_psi_u), ('Wb_psi_y', enc.Wb_psi_y)]
        if not enc.flag_linear_only:
            out += [(f'net_p.{n}', p) for n, p in enc.net_p.named_parameters()]
        return out

    def encoder_jacobians(self, ix=None):
        """J_w = D_xi_p e_p(H_w), the ENCODER ALONE. No rollout is differentiated here.

        This is the whole point of the D-184 factorisation: differentiating the rollout with
        respect to the encoder weights directly costs one rollout per encoder parameter, which
        for this configuration is thousands.
        """
        enc = self.enc
        named = self._named_xip()
        params = {n: p.detach() for n, p in named}
        uh = self.w['uhist'] if ix is None else self.w['uhist'][ix]
        yh = self.w['yhist'] if ix is None else self.w['yhist'][ix]

        def f(pdict):
            return torch.func.functional_call(enc, pdict, (uh, yh))[:, :self.cfg.nx_phys]

        jac = torch.func.jacrev(f)(params)
        n_sel = uh.shape[0]
        flat = [jac[n].reshape(n_sel, self.cfg.nx_phys, -1) for n, _ in named]
        Jfull = torch.cat(flat, dim=2)
        return ([Jfull[w].detach().cpu().numpy().astype(np.float64) for w in range(n_sel)],
                [n for n, _ in named], [int(p.numel()) for _, p in named])

    def free_initial_state_nuisance(self, ix=None):
        """The FREE per-window initial-state nuisance space, an explicitly labelled BOUND.

        Profiling a free `x_w,0` per window removes AT LEAST as much as profiling the shared
        encoder, because `range(E) subseteq range(blockdiag Gamma_w)` (plan Eq. (14)). Full
        physical rank surviving THIS larger space is reassuring; collapse here does NOT prove
        collapse for the shared encoder, and it must never be reported as the proposed geometry.
        """
        blocks = self.initial_state_jacobians(ix)
        n = self.n_rows() if ix is None else len(ix) * self.T * self.ny
        cols, off = [], 0
        for B in blocks:
            padded = np.zeros((n, B.shape[1]))
            padded[off:off + B.shape[0]] = B
            cols.append(padded)
            off += B.shape[0]
        return np.hstack(cols) if cols else np.zeros((n, 0))
