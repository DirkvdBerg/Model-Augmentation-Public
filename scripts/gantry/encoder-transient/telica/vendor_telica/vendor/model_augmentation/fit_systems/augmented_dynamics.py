"""Augmented-state recurrence: a live linear bypass on the augmented rows, behind a zero gate.

WHAT THIS IS FOR. In the augmentation setting the augmented rows are written by the correction
network alone, and that network's output is exactly zero at initialisation (baseline equality).
So `x_a = 0` for every `k`, the network's read weights on the `x_a` columns see an all-zero input
and receive zero gradient, and its write path changes nothing downstream because nothing reads
`x_a`. The augmented states are dead and stay dead. Measured on the gantry: ablation `1.0002x`,
improvement fraction `F = 0.0007`, i.e. the entire gain came through the physical rows
(`scripts/gantry/BLA-Augmentation/RESULTS.md:298`).

This module installs, on the augmented rows only,

    x_a[k+1] = A_aa x_a[k] + Gamma * B z[k] + alpha * F(z[k])[aug rows]

so `x_a` is driven from the first sample whatever the network does.

WHY A WRAPPER ON THE NETWORK RATHER THAN A NEW BLOCK. The interconnect already routes the
correction network's output additively into the rows named by the routing, and the baseline block
writes only the physical rows. For the augmented rows nothing else contributes, so replacing the
network with this module realises the equation above exactly, without changing the interconnect
graph or any shared build path. A configuration that wants no recurrence simply does not install it.

STRUCTURE AND CITATIONS, one per element.

* `A_aa` is block-diagonal, one 2x2 rotation-scaling per conjugate pair, in the stable exponential
  parameterisation `lambda = exp(-exp(nu_log) + i theta)`.
  # THEORY: Orvieto et al., ICML 2023, Sec. 3.3. Verified quote (MATCH OK, 2026-08-23):
  # "to learn long-range dependencies and avoid quickly vanishing gradients, eigenvalues in the
  # recurrence [need to have magnitude close to 1]". rho = exp(-exp(nu)) lies in (0, 1) for every
  # real nu, so the block is stable by construction at every point of the optimisation path.
* `B` is a random input map whose columns acting on `x_a` are structurally zero, enforced by a
  registered 0/1 mask so those entries receive exactly zero gradient. `A_aa` owns the recurrence;
  a non-zero `x_a` column would be a second, unconstrained recurrence outside the stable
  parameterisation.
  # THEORY: hoekstra2026lfrfp p10 -- matrices not pinned by the baseline-equality requirement are
  # initialised U(-1, 1). Recorded as EVIDENCE.md claim 9; not re-verified in this pass.
* `B` is then SCALED so that `x_a` has unit standard deviation ON THE TRAINING DATA.
  # THEORY: Schoukens, ECC 2021, Sec. IV. Verified quote (MATCH OK, 2026-08-23): "The state-space
  # matrices of the linear approximate model are normalized such that each of the states has a
  # standard deviation equal to 1". This is a measured scaling, not a formula. See
  # `empirical_input_scale`.
* the correction network keeps a ZERO OUTPUT PROJECTION over RANDOM inner layers, which is the
  harness default (`zero_init_feed_forward_nn`) and requires no change here.
  # THEORY: Schoukens, ECC 2021, Sec. IV.2. Verified quote (MATCH OK, 2026-08-23): "This paper
  # proposes to do this the other way around, random weights in the nonlinear layer and zero
  # weights in the linear layers, works better in the benchmark examples". Sec. IV.3 gives the
  # reason for the gR-SS-NN structure: random inner layers generate "a pool of nonlinearly
  # transformed outputs which the estimator can pick from using the linear weights during
  # optimization".

WHAT IS LIVE AT STEP ZERO. The output projection `W_out` of the correction network. Its gradient is
`dL/dW_out = <dL/dw, sigma(z)>`, and `sigma(z)` contains `x_a`, which is non-zero from the first
sample because `B` is live. So the readout starts learning to USE the augmented states immediately,
and it does so only because they are driven. The inner layers are frozen at step one, because
`dL/d(sigma) ∝ W_out = 0`; they start once `W_out` has moved.

WHAT WE DID EARLIER, AND WHY IT CHANGED (2026-08-23). The first version of this module carried two
extra elements, both now removed as redundant against the citation above:

* a ReZero scalar gate `alpha`, zero at init, over a RE-INITIALISED (non-zero) final layer, per
  Bachlechner et al. 2020 Eq. (6). It was introduced for the D-130 `W^a` dead zone. It is strictly
  weaker here than what Schoukens specifies: it makes ONE scalar live at step one where the zero
  output projection makes the whole of `W_out` live, and it does not fix `W^a` either, which is
  gradient-free at step one under both schemes. Bachlechner's argument concerns deep networks; this
  correction net has two hidden layers and no benefit was ever measured. `OUTPUT_ZERO` is now the
  default and `GATE_ZERO` is retained only as a comparison arm.
* `Gamma = sqrt(1 - rho^2)` on the random input map, per Orvieto Sec. 3.4. Exact for a WHITE
  unit-variance input: `P = A P A' + b b'` gives `tr(P) = |b|^2 / (1 - rho^2)`. Our `z` is not
  white, being dominated by low-frequency setpoint motion plus a narrow excitation band, so the
  formula normalises for the wrong spectrum. It is also nearly vacuous once the poles come from an
  identification, since every pair then carries the same damping and `Gamma` collapses to a single
  scalar the readout absorbs. The empirical scaling above replaces it, uses the real signal
  statistics, and additionally closes a confound recorded as open in
  `scripts/gantry/augmented-states/README.md` section 7: driven-state RMS at initialisation
  differed by up to `10x` across arms, so "right pole" and "louder pole" were not separated.
  Unit-variance-by-measurement makes that gauge identical across arms by construction.

THE POLES ARE NOT LEARNED, so they must be placed. Measured on the gantry: with the true mode
planted, `dL/d(nu_log) < 0` on 7 of 8 disjoint batches and monotone over 150 steps, no non-negative
residual weighting can flip that sign, and both trained arms moved their poles under `0.15 Hz` over
520 updates. The block therefore behaves as a fixed basis whose SPAN matters, not as an adaptive
resonator. `rho` and `theta` are inputs to this class and it deliberately supplies no default:
placing them is the caller's decision and must be justified, never defaulted.
"""
__project_origin__ = "added"

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

GATE_ZERO = 'GATE_ZERO'
OUTPUT_ZERO = 'OUTPUT_ZERO'


class AugmentedDynamics(nn.Module):
    """Wraps a correction network and adds the linear augmented-state recurrence.

    Parameters
    ----------
    mlp : the correction network the augmentation block was built with.
    aug_out_pos : positions inside the network OUTPUT vector that write the augmented state rows.
    x_aug_in_pos : positions inside the network INPUT `z = [x, u]` that hold `x_a`.
    rho, theta : per-pair pole magnitude and angle AT THE MODEL RATE. No default: see the module
        docstring on why placement is the caller's decision.
    B : `(nx_aug, nz)` input map; its `x_aug` columns must be exactly zero.
    gamma_on_B : apply Orvieto's `sqrt(1 - rho^2)` white-noise normalisation. DEFAULT False: the
        supported route is to pass a `B` already scaled by `empirical_input_scale`, which is what
        Schoukens specifies and which uses the real signal statistics. Retained as a comparison arm.
    gate_mode : `OUTPUT_ZERO` (zero output projection, gate fixed at 1) is the default and is what
        Schoukens Sec. IV.2 specifies. `GATE_ZERO` (live branch, zero scalar) is the ReZero variant,
        retained only as a comparison arm; see the module docstring on why it is weaker here.
    train_A, train_B : whether the recurrence and the input map are optimised.
    """

    def __init__(self, mlp: nn.Module, aug_out_pos: Sequence[int], x_aug_in_pos: Sequence[int],
                 rho, theta, B, gamma_on_B: bool = False, gate_mode: str = OUTPUT_ZERO,
                 train_A: bool = True, train_B: bool = True, dtype=torch.float32):
        super().__init__()
        aug_out_pos = tuple(int(i) for i in aug_out_pos)
        x_aug_in_pos = tuple(int(i) for i in x_aug_in_pos)
        if len(aug_out_pos) != len(x_aug_in_pos):
            raise ValueError('augmented output and input positions must have equal length')
        if len(aug_out_pos) % 2:
            raise ValueError('augmented states come in conjugate pairs; length must be even')
        rho = np.asarray(rho, dtype=np.float64).reshape(-1)
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        n_pairs = len(aug_out_pos) // 2
        if rho.shape != (n_pairs,) or theta.shape != (n_pairs,):
            raise ValueError('expected %d (rho, theta) pairs, got %d/%d'
                             % (n_pairs, rho.size, theta.size))
        if not np.all((rho > 0.0) & (rho < 1.0)):
            raise ValueError('every pole radius must lie strictly inside the unit disc')
        B = np.asarray(B, dtype=np.float64)
        if B.shape[0] != len(aug_out_pos):
            raise ValueError('B must have one row per augmented state')
        if np.abs(B[:, list(x_aug_in_pos)]).max(initial=0.0) != 0.0:
            raise ValueError('B columns acting on x_a must start exactly zero')

        self.mlp = mlp
        self.aug_out_pos = aug_out_pos
        self.x_aug_in_pos = x_aug_in_pos
        self.gamma_on_B = bool(gamma_on_B)
        self.gate_mode = str(gate_mode)
        self.n_pairs = n_pairs
        # runtime-only ablation flags; never checkpointed as anything but their default
        self.ablate_xa_to_f = False
        self.ablate_xa_update = False

        self.nu_log = nn.Parameter(torch.tensor(np.log(-np.log(rho)), dtype=dtype),
                                   requires_grad=bool(train_A))
        self.theta = nn.Parameter(torch.tensor(theta, dtype=dtype), requires_grad=bool(train_A))
        self.B = nn.Parameter(torch.tensor(B, dtype=dtype), requires_grad=bool(train_B))
        mask = torch.ones_like(self.B)
        mask[:, list(x_aug_in_pos)] = 0.0
        self.register_buffer('B_mask', mask)

        if self.gate_mode == GATE_ZERO:
            self.alpha = nn.Parameter(torch.zeros((), dtype=dtype))
        elif self.gate_mode == OUTPUT_ZERO:
            # alpha is fixed at one and is not a parameter: the zero final layer supplies the
            # baseline-preserving property, and a second redundant scalar would receive exactly
            # zero gradient at step zero anyway.
            self.register_buffer('alpha', torch.ones((), dtype=dtype))
        else:
            raise ValueError('unknown gate_mode %r' % gate_mode)

    # -- read-only views ---------------------------------------------------------------------
    @property
    def net(self):
        """Compatibility with consumers that reach for the inner `nn.Sequential`."""
        return self.mlp.net if hasattr(self.mlp, 'net') else self.mlp

    def rho(self) -> torch.Tensor:
        """`exp(-exp(nu_log))`, clamped strictly below 1 in the ACTUAL float type.

        # CHANGED (D-160, found by tests/test_augmented_dynamics.py): the exponential
        # parameterisation is stable by construction in EXACT arithmetic, but not in floating
        # point. `exp(-exp(nu_log))` rounds to exactly 1.0 once `nu_log` is below about -45 in
        # float64 (-16 in float32), because `exp(-1.9e-22) == 1.0` at that precision. At `rho = 1`
        # the pair is a pure oscillator that never decays over a free run, and `gamma =
        # sqrt(1 - rho^2)` collapses to 0, silently killing the input map that is the whole point
        # of the block. The clamp keeps the documented invariant true as implemented. It is not
        # expected to bind: initialisation sits near `nu_log = -4.4` for a `zeta = 0.05` mode, and
        # C6 measured the objective driving `nu_log` UP (radius down), away from this edge.
        # The low end underflows too: `exp(-exp(+50)) == 0.0` exactly, which is stable but has no
        # memory at all, i.e. the one-sample-delay degeneracy the recurrence exists to avoid, and
        # it would make `pole_table`'s `log(lambda)` undefined. Both ends are clamped.
        """
        eps = torch.finfo(self.nu_log.dtype).eps
        return torch.clamp(torch.exp(-torch.exp(self.nu_log)), min=eps, max=1.0 - eps)

    def gamma(self) -> torch.Tensor:
        r = self.rho()
        return torch.sqrt(torch.clamp(1.0 - r * r, min=0.0))

    def effective_B(self) -> torch.Tensor:
        """`B` after the structural mask and, for a random map, after `Gamma`."""
        b = self.B * self.B_mask
        if self.gamma_on_B:
            b = b * torch.repeat_interleave(self.gamma(), 2).unsqueeze(1)
        return b

    def trainable_parameters(self):
        out = [p for p in (self.nu_log, self.theta, self.B) if p.requires_grad]
        if isinstance(self.alpha, nn.Parameter):
            out.append(self.alpha)
        return out

    def pole_table(self, ts: float):
        """Per-pair (rho, theta, f_d [Hz], f_n [Hz], zeta) at the model rate, for diagnostics."""
        with torch.no_grad():
            r = self.rho().detach().cpu().numpy().astype(np.float64)
            th = self.theta.detach().cpu().numpy().astype(np.float64)
        rows = []
        for i in range(self.n_pairs):
            lam = complex(r[i] * math.cos(th[i]), r[i] * math.sin(th[i]))
            s = np.log(lam) / ts
            wn = abs(s)
            rows.append({'pair': i, 'rho': float(r[i]), 'theta': float(th[i]),
                         'f_d_hz': float(abs(np.angle(lam)) / (2 * math.pi * ts)),
                         'f_n_hz': float(wn / (2 * math.pi)),
                         'zeta': float(-s.real / wn) if wn > 0 else float('nan')})
        return rows

    # -- forward -----------------------------------------------------------------------------
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        aug_out = list(self.aug_out_pos)
        x_aug_in = list(self.x_aug_in_pos)

        x_for_net = X
        if self.ablate_xa_to_f:
            x_for_net = X.clone()
            x_for_net[..., x_aug_in] = 0.0
        w = self.alpha * self.mlp(x_for_net)

        xa = X[..., x_aug_in]
        x1, x2 = xa[..., 0::2], xa[..., 1::2]
        r = self.rho()
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        drive = X @ self.effective_B().T
        upd = drive + w[..., aug_out]

        out = w.clone()
        out[..., aug_out[0::2]] = r * (c * x1 - s * x2) + upd[..., 0::2]
        out[..., aug_out[1::2]] = r * (s * x1 + c * x2) + upd[..., 1::2]
        if self.ablate_xa_update:
            out = out.clone()
            out[..., aug_out] = 0.0
        return out


# ── helpers for locating the installed block ─────────────────────────────────────────────────

def find_ann_block(fit_sys):
    """The `Static_ANN_Block` in an interconnect, which is what this module wraps."""
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    return next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))


def find_wrapper(fit_sys) -> Optional[AugmentedDynamics]:
    """The installed recurrence, or None when the configuration has none."""
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    ann = next((m for m in fit_sys.hfn.connected_blocks
                if isinstance(m, Static_ANN_Block)), None)
    if ann is None:
        return None
    return ann.net if isinstance(ann.net, AugmentedDynamics) else None


def final_linear(module: nn.Module) -> nn.Linear:
    """The last `nn.Linear` of an MLP, which both gate variants act on."""
    seq = module.net if hasattr(module, 'net') else module
    for m in reversed(list(seq)):
        if isinstance(m, nn.Linear):
            return m
    raise ValueError('no nn.Linear found in the correction network')


def reset_final_layer(layer: nn.Linear, seed: int) -> None:
    """`nn.Linear.reset_parameters()` reproduced from a DEDICATED generator.

    # THEORY: torch's default is kaiming_uniform_(a=sqrt(5)) on the weight and a matching fan-in
    # uniform on the bias; both reduce to U(-1/sqrt(fan_in), +1/sqrt(fan_in)). Reproducing it from a
    # dedicated generator keeps the global stream, and therefore the batch order, untouched, so a
    # run that re-initialises the final layer stays paired with one that does not.

    Needed by `GATE_ZERO`: the branch must be LIVE behind the zero scalar, otherwise the pair
    (zero branch, zero gate) is an exact saddle rather than a delayed start.
    """
    gen = torch.Generator().manual_seed(int(seed))
    fan_in = layer.weight.shape[1]
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=gen)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=gen)


def empirical_input_scale(B, rho, theta, Z, target_std: float = 1.0, eps: float = 1e-30):
    """Scale each pair's rows of `B` so `x_a` has unit std under the REAL driving signal `Z`.

    # THEORY: Schoukens, ECC 2021, Sec. IV. Verified quote (MATCH OK, 2026-08-23): "The state-space
    # matrices of the linear approximate model are normalized such that each of the states has a
    # standard deviation equal to 1".

    WHY THIS AND NOT `Gamma = sqrt(1 - rho^2)`. Orvieto's normalisation is exact for a WHITE
    unit-variance input, and `Z` here is not white: it is dominated by low-frequency setpoint
    motion plus a narrow excitation band. The formula would normalise for the wrong spectrum. This
    measures the actual thing instead, and because the recurrence is LINEAR in `B` the measurement
    is exact rather than iterative: running it once with `B` and dividing by the observed std lands
    the state at `target_std` in one shot.

    It also makes the input GAUGE identical across configurations. `augmented-states/README.md`
    section 7 records driven-state RMS at initialisation differing by up to `10x` across arms, so
    "right pole" and "louder pole" were confounded. Unit variance by measurement removes that.

    Parameters
    ----------
    B : `(nx_aug, nz)`, the drawn map, with its `x_a` columns already zero.
    rho, theta : per-pair pole magnitude and angle at the model rate.
    Z : `(N, nz)` the ACTUAL correction-network inputs `z = [x, u]` over a data slice, normalised
        exactly as training normalises them. The `x_a` columns are ignored, because `B` masks them.
    Returns
    -------
    `(nx_aug, nz)` scaled `B`, plus the per-pair std that was measured.
    """
    B = np.asarray(B, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    Z = np.asarray(Z, dtype=np.float64)
    n_pairs = len(rho)
    drive = Z @ B.T                                   # (N, nx_aug)
    xa = np.zeros((len(Z) + 1, B.shape[0]), dtype=np.float64)
    c, s = np.cos(theta), np.sin(theta)
    for k in range(len(Z)):
        x1, x2 = xa[k, 0::2], xa[k, 1::2]
        xa[k + 1, 0::2] = rho * (c * x1 - s * x2) + drive[k, 0::2]
        xa[k + 1, 1::2] = rho * (s * x1 + c * x2) + drive[k, 1::2]
    burn = min(len(Z) // 10, 2000)                    # drop the x_a(0) = 0 transient
    std = np.array([xa[burn:, 2 * i:2 * i + 2].std() for i in range(n_pairs)])
    scale = target_std / np.maximum(std, eps)
    out = B * np.repeat(scale, 2)[:, None]
    return out, std


def draw_input_map(n_aug: int, nz: int, x_aug_ix: Sequence[int], seed: int) -> np.ndarray:
    """`B` rows from a dedicated stream, with the `x_a` columns structurally zeroed.

    # THEORY: hoekstra2026lfrfp p10 -- matrices not pinned by the baseline-equality requirement are
    # initialised U(-1, 1). Recorded as EVIDENCE.md claim 9; not re-verified in this pass.
    """
    gen = torch.Generator().manual_seed(int(seed))
    B = (2.0 * torch.rand((n_aug, nz), generator=gen, dtype=torch.float64) - 1.0).numpy()
    B[:, list(x_aug_ix)] = 0.0      # STRUCTURAL_INVARIANT: A_aa owns the x_a recurrence
    return B
