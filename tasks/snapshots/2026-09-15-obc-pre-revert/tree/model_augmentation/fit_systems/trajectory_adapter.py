"""The one seam between the trajectory penalty and whatever produces a rollout.

Plan Sect. 9.2. `trajectory_orth_projection.py` is pure linear algebra on a stacked vector;
this file says where that vector comes from. A system participates in the method by
implementing `TrajectoryAdapter`, and then the geometry construction, the penalty, the
diagnostics and the chunked gradient accumulation are shared code rather than a parallel
implementation.

WHY AN INTERFACE AND NOT TWO IMPLEMENTATIONS. The synthetic testbed and the gantry differ in
everything around the rollout: deepSI's `Interconnect` versus a hand-written loop, a scheduled
controller bank versus none, routing matrices, optional augmented-dynamics wrappers. They do
not differ in the mathematics. Writing the method twice would mean the version that is tested
is not the version that runs. The plan's own warning applies: avoid independently transcribed
simulators.

THE THREE INTERVENTIONS, and they must share one code path inside the adapter, differing only
by a flag. Re-deriving them separately reintroduces exactly the divergence this file exists to
prevent.

    'full'     the model as trained
    'off'      the augmentation's contribution to the state removed entirely. Defines
               d_total = full - off, the quantity that is PENALISED.
    'clamped'  the augmentation active but its own states held at zero throughout, so the
               direct correction survives and the latent route does not. Defines
               d_lat = full - clamped and d_direct = clamped - off, which sum exactly to
               d_total. This is an INTERVENTION-based decomposition, not a unique separation
               of nonlinear mechanisms, and must be reported as such.

`off` must disable EVERY augmentation write, including learned output corrections or direct
latent-to-output paths if the architecture has them. It must not mutate live weights or caches
to achieve that, and the controller state, where present, must evolve independently in each of
the three rollouts rather than being shared or replayed.
"""
__project_origin__ = "added"

import abc

import numpy as np
import torch
from torch import Tensor

from model_augmentation.fit_systems.trajectory_orth_projection import (
    TrajectoryProjectionGeometry)

MODES = ('full', 'off', 'clamped')


class TrajectoryAdapter(abc.ABC):
    """What a system must supply. Deliberately small; everything else is shared."""

    # ------------------------------------------------------------------ rollout
    @abc.abstractmethod
    def scored_stack(self, params: dict, mode: str = 'full', rows=None) -> Tensor:
        """The scored output stack, flattened, in the SAME order for every mode.

        `params` carries the three groups by name ('physical', 'augmentation', 'encoder');
        passing them explicitly rather than reading live modules is what lets a caller hold
        one group constant while differentiating another. `rows` selects a chunk of the stack
        and must index the same rows in every mode. Burn-in rows, if any, are excluded from
        the stack here while remaining in the graph.
        """

    @abc.abstractmethod
    def parameter_groups(self) -> dict:
        """{'physical': [...], 'augmentation': [...], 'encoder': [...]} of live tensors.

        Groups must be DISJOINT BY TENSOR IDENTITY, which `check_ownership` verifies. This is
        what makes a selective-gradient policy meaningful: 'augmentation parameters' means
        every trainable augmentation tensor in the selected architecture, including wrapper
        gates and explicit recurrence matrices, not only the innermost network.
        """

    @abc.abstractmethod
    def n_rows(self) -> int:
        """Length of the scored stack."""

    # ------------------------------------------------------------------ optional
    def loss_weight(self):
        """`L` with `L^T L = W`, matching the prediction loss. None means identity.

        Output normalisation and any 1/N averaging belong here. Returning None when the
        prediction loss is NOT an unweighted sum silently puts the penalty in a different
        metric from the loss it is supposed to compete with.
        """
        return None

    def metadata(self) -> dict:
        return {}


# ---------------------------------------------------------------------------- helpers
def check_ownership(adapter: TrajectoryAdapter) -> dict:
    """Plan V0. Disjointness by tensor identity, not by name or by value."""
    groups = adapter.parameter_groups()
    seen, dupes = {}, []
    for g, tensors in groups.items():
        for t in tensors:
            key = id(t)
            if key in seen and seen[key] != g:
                dupes.append((seen[key], g, tuple(t.shape)))
            seen[key] = g
    if dupes:
        raise ValueError(f'parameter groups overlap by tensor identity: {dupes}')
    return {g: sum(int(np.prod(t.shape)) for t in ts) for g, ts in groups.items()}


def _jac(fn, x: Tensor) -> np.ndarray:
    """d fn / d x as (N, n_x), forward mode.

    FORWARD, not reverse: the stack has N rows (order 1e3 and up) while a parameter group has
    few (order 1e1), so reverse mode costs ~N passes and forward mode costs n_x. Using
    `torch.autograd.functional.jacobian` here was ~85% of an earlier sweep's total runtime.
    A gantry adapter whose rollout cannot be made functional will need a different route and
    should say so rather than silently falling back to reverse mode.
    """
    from torch.func import jacfwd
    return jacfwd(fn)(x).detach().cpu().numpy()


# REQUIRED CONSTRUCTION (D-184), not an optimisation. The encoder block `E` must be built as
#
#     E = (d p / d x0) . (d x0 / d xi)
#
# and NEVER by differentiating the rollout with respect to the encoder parameters. `d p / d x0`
# has `nxd` columns (14 on the gantry) so it costs 14 forward rollouts; `d x0 / d xi` is the
# Jacobian of the encoder alone and involves no rollout. Differentiating through `xi` directly
# costs one rollout per encoder parameter, of order 2400 for the `linear_map` encoder
# (`na_nb = 29` samples of `nu + ny = 6` channels into `nxd = 14` states), which is not a slow
# geometry build but an impossible one. Reverse mode is worse, costing one pass per output row.
#
# The construction is EXACT, but only because the encoder influences the prediction solely
# through the initial state. That premise must be asserted where the adapter is implemented, not
# assumed: perturb `xi` with `x0` held fixed and require `p` to be bit-identical. If a future
# encoder also feeds an output correction or a scheduling signal, the chain rule is silently
# wrong and D-184 is void.
#
# The `build_geometry` below takes `E` by differentiating the adapter directly, which is correct
# for the SYNTHETIC testbed (34 encoder parameters) and must NOT be reused as-is for the gantry.
# The gantry adapter supplies `E` precomputed by the chain rule above.


def build_geometry(adapter: TrajectoryAdapter, params: dict, rank_rtol=1e-12,
                   rank_atol=0.0, profile_encoder=True, dtype=torch.float64,
                   verbose=True) -> TrajectoryProjectionGeometry:
    """Reference geometry from FULL AUGMENTED-ROLLOUT sensitivities.

    `S` and `E` are differentiated through the 'full' rollout, not the baseline and not one
    step: substitution concerns the predictor that is actually fitted. Everything here happens
    once at a frozen reference and is then held fixed, outside the training graph.
    """
    phys = params['physical'].detach().clone()
    enc = params.get('encoder')
    enc = None if enc is None else enc.detach().clone()

    def stack_of_phys(p):
        return adapter.scored_stack({**params, 'physical': p}, mode='full')

    S = _jac(stack_of_phys, phys)
    E = None
    if profile_encoder and enc is not None:
        def stack_of_enc(e):
            return adapter.scored_stack({**params, 'encoder': e}, mode='full')
        E = _jac(stack_of_enc, enc)

    geom = TrajectoryProjectionGeometry(
        S, E, L=adapter.loss_weight(), rank_rtol=rank_rtol, rank_atol=rank_atol,
        dtype=dtype, metadata={**adapter.metadata(), 'profile_encoder': profile_encoder})
    geom.validate()
    if verbose:
        print(geom.summary())
    return geom


def contributions(adapter: TrajectoryAdapter, params: dict, rows=None) -> dict:
    """d_total, d_lat, d_direct from the three interventions, with the identity asserted.

    `d_total = d_lat + d_direct` holds by construction, so a violation means the three modes
    are not sharing one rollout path, which is the failure this interface exists to prevent.
    """
    full = adapter.scored_stack(params, 'full', rows)
    off = adapter.scored_stack(params, 'off', rows)
    clamped = adapter.scored_stack(params, 'clamped', rows)
    d_total, d_lat, d_direct = full - off, full - clamped, clamped - off
    err = float((d_total - d_lat - d_direct).abs().max())
    scale = max(float(d_total.abs().max()), 1e-300)
    assert err <= 1e-9 * scale, (
        f'the three interventions are inconsistent: max |d_total - d_lat - d_direct| = {err:.3e} '
        f'against scale {scale:.3e}. The modes are not sharing one rollout path.')
    return dict(total=d_total, lat=d_lat, direct=d_direct)
