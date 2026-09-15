"""Kessels-derived nonlinear augmented-state writer and split history encoder.

The state owned by :class:`KesselsExtensionBlock` is ordered ``[q_a, v_a]`` and
is updated by

    q_a[k+1] = q_a[k] + gain * v_a[k]
    v_a[k+1] = L_psi(x_b[k], q_a[k], v_a[k], u[k]).

``L_psi`` assigns the next velocity.  There is deliberately no velocity
increment, legacy pole bank, or output/readout path in this module.
"""

from __future__ import annotations

__project_origin__ = "added"

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from model_augmentation.fit_systems.blocks import Block, Static_ANN_Block


COORDINATE_MODES = ("kessels_raw", "normalized_similarity")
WRITER_SCALING_MODES = ("project_mean_std", "kessels_fp_range")
WRITER_INITIALIZATIONS = ("matched_project", "kessels_small")


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def _init_small(module: nn.Module, std: float) -> None:
    if not (std > 0.0):
        raise ValueError("small initialization std must be positive")
    with torch.no_grad():
        for layer in _linear_layers(module):
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.normal_(layer.bias, mean=0.0, std=std)


class KesselsWriter(nn.Module):
    """Two-hidden-layer tanh writer, optionally with a trainable linear bypass."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        *,
        linear_bypass: bool = False,
        hidden_widths: Optional[Sequence[int]] = None,
        initialization: str = "matched_project",
        small_init_std: float = 1e-5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if n_in <= 0 or n_out <= 0:
            raise ValueError("writer input/output dimensions must be positive")
        if initialization not in WRITER_INITIALIZATIONS:
            raise ValueError(f"unknown writer initialization {initialization!r}")
        widths = tuple(hidden_widths or (3 * n_in, 3 * n_out))
        if len(widths) != 2 or min(widths) <= 0:
            raise ValueError("writer_hidden_widths must contain two positive integers")
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.hidden_widths = widths
        self.linear_bypass_enabled = bool(linear_bypass)
        self.initialization = initialization
        self.small_init_std = float(small_init_std)
        self.mlp = nn.Sequential(
            nn.Linear(n_in, widths[0]), nn.Tanh(),
            nn.Linear(widths[0], widths[1]), nn.Tanh(),
            nn.Linear(widths[1], n_out),
        )
        # Appendix D describes weighted direct connections, not a separate bias.
        self.linear_bypass = nn.Linear(n_in, n_out, bias=False) if linear_bypass else None
        if initialization == "kessels_small":
            _init_small(self.mlp, self.small_init_std)
            if self.linear_bypass is not None:
                _init_small(self.linear_bypass, self.small_init_std)
        self.to(dtype=dtype)

    def branches(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        nonlinear = self.mlp(z)
        linear = torch.zeros_like(nonlinear)
        if self.linear_bypass is not None:
            linear = self.linear_bypass(z)
        return linear, nonlinear

    def forward(self, z: Tensor) -> Tensor:
        linear, nonlinear = self.branches(z)
        return linear + nonlinear


class KesselsExtensionBlock(Block):
    """Sole writer of ``x_a[k+1]`` for the production interconnect."""

    qv_ordering = "q_then_v"

    def __init__(
        self,
        *,
        nx_phys: int,
        n_aug: int,
        nu: int,
        sample_time: float,
        coordinate_mode: str = "kessels_raw",
        writer_io_scaling: str = "project_mean_std",
        writer_input_multiplier: Optional[Sequence[float]] = None,
        writer_input_offset: Optional[Sequence[float]] = None,
        writer_linear_bypass: bool = False,
        writer_hidden_widths: Optional[Sequence[int]] = None,
        writer_initialization: str = "matched_project",
        small_init_std: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        name: str = "kessels_extension",
    ) -> None:
        if n_aug <= 0 or n_aug % 2:
            raise ValueError("Kessels n_a must be positive and even")
        if nx_phys <= 0 or nu <= 0 or not (sample_time > 0.0):
            raise ValueError("nx_phys, nu and sample_time must be positive")
        if coordinate_mode not in COORDINATE_MODES:
            raise ValueError(f"unknown xa_coordinate_mode {coordinate_mode!r}")
        if writer_io_scaling not in WRITER_SCALING_MODES:
            raise ValueError(f"unknown writer_io_scaling {writer_io_scaling!r}")
        self.nx_phys = int(nx_phys)
        self.n_aug = int(n_aug)
        self.nu = int(nu)
        self.m = self.n_aug // 2
        self.sample_time = float(sample_time)
        self.coordinate_mode = coordinate_mode
        self.writer_io_scaling = writer_io_scaling
        self.kinematic_gain = self.sample_time if coordinate_mode == "kessels_raw" else 1.0
        n_in = self.nx_phys + self.n_aug + self.nu
        super().__init__(nz=n_in, nw=self.n_aug, name=name)
        mult = torch.ones(n_in, dtype=dtype)
        offs = torch.zeros(n_in, dtype=dtype)
        if writer_input_multiplier is not None:
            mult = torch.as_tensor(writer_input_multiplier, dtype=dtype).reshape(-1)
        if writer_input_offset is not None:
            offs = torch.as_tensor(writer_input_offset, dtype=dtype).reshape(-1)
        if mult.numel() != n_in or offs.numel() != n_in:
            raise ValueError(f"writer scaling vectors must have length {n_in}")
        if not torch.isfinite(mult).all() or not torch.isfinite(offs).all():
            raise ValueError("writer scaling contains nonfinite values")
        if torch.any(mult == 0):
            raise ValueError("writer scaling multiplier contains a zero")
        self.register_buffer("writer_input_multiplier", mult)
        self.register_buffer("writer_input_offset", offs)
        # q'=q/Ts, v'=v is the declared deterministic similarity coordinate.
        sq = 1.0 if coordinate_mode == "kessels_raw" else 1.0 / self.sample_time
        self.register_buffer("S_q", torch.full((self.m,), sq, dtype=dtype))
        self.register_buffer("S_v", torch.ones(self.m, dtype=dtype))
        self.writer = KesselsWriter(
            n_in, self.m, linear_bypass=writer_linear_bypass,
            hidden_widths=writer_hidden_widths,
            initialization=writer_initialization, small_init_std=small_init_std,
            dtype=dtype,
        )
        self.ablate_linear_branch = False
        self.ablate_nonlinear_branch = False
        self.force_zero_state = False
        self.hold_state = False
        self.last_linear_branch = None
        self.last_nonlinear_branch = None

    def scale_writer_input(self, z_flat: Tensor) -> Tensor:
        return z_flat * self.writer_input_multiplier + self.writer_input_offset

    def forward(self, z: Tensor) -> Tensor:
        if z.ndim != 3 or z.shape[1:] != (self.nz, 1):
            raise ValueError(f"expected (batch,{self.nz},1), got {tuple(z.shape)}")
        xa = z[:, self.nx_phys:self.nx_phys + self.n_aug, 0]
        if self.force_zero_state:
            return torch.zeros_like(xa).unsqueeze(-1)
        if self.hold_state:
            return xa.unsqueeze(-1)
        q, v = xa[:, :self.m], xa[:, self.m:]
        writer_z = self.scale_writer_input(z[:, :, 0])
        linear, nonlinear = self.writer.branches(writer_z)
        self.last_linear_branch = linear.detach()
        self.last_nonlinear_branch = nonlinear.detach()
        if self.ablate_linear_branch:
            linear = torch.zeros_like(linear)
        if self.ablate_nonlinear_branch:
            nonlinear = torch.zeros_like(nonlinear)
        q_next = q + self.kinematic_gain * v
        # Assignment: no v term is added here.
        v_next = linear + nonlinear
        return torch.cat((q_next, v_next), dim=1).unsqueeze(-1)

    def manifest(self) -> Dict[str, object]:
        return {
            "method": "kessels_extension",
            "state_count": self.n_aug,
            "state_ordering": self.qv_ordering,
            "sample_time": self.sample_time,
            "xa_coordinate_mode": self.coordinate_mode,
            "S_q": self.S_q.detach().cpu().tolist(),
            "S_v": self.S_v.detach().cpu().tolist(),
            "kinematic_gain": self.kinematic_gain,
            "writer_io_scaling": self.writer_io_scaling,
            "writer_input_multiplier": self.writer_input_multiplier.detach().cpu().tolist(),
            "writer_input_offset": self.writer_input_offset.detach().cpu().tolist(),
            "writer_linear_bypass": self.writer.linear_bypass_enabled,
            "writer_hidden_widths": list(self.writer.hidden_widths),
            "writer_activation": "tanh",
            "writer_output_activation": "linear",
            "writer_initialization": self.writer.initialization,
            "small_initialization_interpretation": "N(0, 1e-5) interpreted as std=1e-5",
        }


class KesselsAugmentedEncoder(nn.Module):
    """Trainable augmented-only encoder consuming exactly ``n_history`` past u/y samples."""

    def __init__(self, nu: int, ny: int, n_history: int, n_aug: int, *,
                 initialization: str = "matched_project", small_init_std: float = 1e-5,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if n_history <= 0 or n_aug <= 0 or n_aug % 2:
            raise ValueError("augmented encoder requires positive history and positive even n_a")
        if initialization not in WRITER_INITIALIZATIONS:
            raise ValueError(f"unknown encoder initialization {initialization!r}")
        self.nu, self.ny = int(nu), int(ny)
        self.n_history, self.n_aug = int(n_history), int(n_aug)
        n_in = n_history * (nu + ny)
        self.net = nn.Sequential(
            nn.Linear(n_in, 3 * n_in), nn.Tanh(),
            nn.Linear(3 * n_in, 3 * n_aug), nn.Tanh(),
            nn.Linear(3 * n_aug, n_aug),
        )
        if initialization == "kessels_small":
            _init_small(self.net, small_init_std)
        self.initialization = initialization
        self.small_init_std = float(small_init_std)
        self.to(dtype=dtype)

    def forward(self, upast: Tensor, ypast: Tensor) -> Tensor:
        if upast.shape[1:] != (self.n_history, self.nu):
            raise ValueError(f"strict-past u history must have shape (*,{self.n_history},{self.nu})")
        if ypast.shape[1:] != (self.n_history, self.ny):
            raise ValueError(f"strict-past y history must have shape (*,{self.n_history},{self.ny})")
        return self.net(torch.cat((upast.reshape(upast.shape[0], -1),
                                   ypast.reshape(ypast.shape[0], -1)), dim=1))


class HoekstraPhysicalEncoder(nn.Module):
    """Physical rows copied from a pre-split ``linear_encoder_init_aug`` instance."""

    def __init__(self, legacy: nn.Module) -> None:
        super().__init__()
        self.nu, self.ny, self.nx = legacy.nu, legacy.ny, legacy.nx
        self.na, self.nb = legacy.na, legacy.nb
        self.flag_linear_only = legacy.flag_linear_only
        self.Wb_psi_y = nn.Parameter(legacy.Wb_psi_y.detach().clone())
        self.Wb_psi_u = nn.Parameter(legacy.Wb_psi_u.detach().clone())
        self.fix_enabled = bool(legacy.fix_enabled)
        if self.fix_enabled:
            for name in ("u_off", "y_off", "x_off"):
                self.register_buffer(name, getattr(legacy, name).detach().clone())
        if not self.flag_linear_only:
            old_layers = _linear_layers(legacy.net)
            modules: list[nn.Module] = []
            for layer in old_layers[:-1]:
                new = nn.Linear(layer.in_features, layer.out_features,
                                bias=layer.bias is not None).to(layer.weight)
                new.load_state_dict(copy.deepcopy(layer.state_dict()))
                modules.extend((new, nn.Tanh()))
            old_final = old_layers[-1]
            new_final = nn.Linear(old_final.in_features, self.nx,
                                  bias=old_final.bias is not None).to(old_final.weight)
            with torch.no_grad():
                new_final.weight.copy_(old_final.weight[:self.nx])
                if old_final.bias is not None:
                    new_final.bias.copy_(old_final.bias[:self.nx])
            modules.append(new_final)
            self.net = nn.Sequential(*modules)

    def forward(self, uhist: Tensor, yhist: Tensor) -> Tensor:
        u = uhist.reshape(uhist.shape[0], self.nu * (self.nb + 1), 1)
        y = yhist.reshape(yhist.shape[0], self.ny * (self.na + 1), 1)
        if self.fix_enabled:
            u, y = u + self.u_off, y + self.y_off
        xb = self.Wb_psi_u @ u + self.Wb_psi_y @ y
        if self.fix_enabled:
            xb = xb - self.x_off
        xb = xb.reshape(-1, self.nx)
        if self.flag_linear_only:
            return xb
        inp = torch.cat((uhist.reshape(uhist.shape[0], -1),
                         yhist.reshape(yhist.shape[0], -1)), dim=1)
        return xb + self.net(inp)


class KesselsCompositeEncoder(nn.Module):
    """Boundary-inclusive Hoekstra E_b plus a separately sliced strict-past E_a."""

    physical_history_end = "k0"
    augmented_history_end = "k0-1"
    augmented_history_signals = ("u", "y")

    def __init__(self, physical_encoder: HoekstraPhysicalEncoder,
                 augmented_encoder: KesselsAugmentedEncoder) -> None:
        super().__init__()
        self.physical_encoder = physical_encoder
        self.augmented_encoder = augmented_encoder
        self.force_zero_augmented = False
        if augmented_encoder.n_history > physical_encoder.na:
            raise ValueError("strict-past augmented history exceeds available pre-boundary history")

    def strict_past_views(self, uhist: Tensor, yhist: Tensor) -> Tuple[Tensor, Tensor]:
        expected_u = (self.physical_encoder.nb + 1, self.physical_encoder.nu)
        expected_y = (self.physical_encoder.na + 1, self.physical_encoder.ny)
        if uhist.shape[1:] != expected_u or yhist.shape[1:] != expected_y:
            raise ValueError(
                f"boundary-inclusive physical histories must have shapes (*,{expected_u[0]},"
                f"{expected_u[1]}) and (*,{expected_y[0]},{expected_y[1]})")
        n = self.augmented_encoder.n_history
        # deepSI supplies [k0-na, ..., k0] for the Hoekstra boundary-inclusive encoder.
        return uhist[:, -(n + 1):-1, :], yhist[:, -(n + 1):-1, :]

    def forward(self, uhist: Tensor, yhist: Tensor) -> Tensor:
        ua, ya = self.strict_past_views(uhist, yhist)
        xb = self.physical_encoder(uhist, yhist)
        xa = self.augmented_encoder(ua, ya)
        if self.force_zero_augmented:
            xa = torch.zeros_like(xa)
        return torch.cat((xb, xa), dim=1)


def split_physical_ann(legacy: Static_ANN_Block, output_positions: Iterable[int]) \
        -> Tuple[Static_ANN_Block, Dict[str, object]]:
    """Copy a mixed legacy correction ANN into a physical-only output head exactly."""
    positions = tuple(int(i) for i in output_positions)
    if not positions or min(positions) < 0 or max(positions) >= legacy.nw:
        raise ValueError("invalid physical head output positions")
    layers = _linear_layers(legacy.net)
    if not layers:
        raise ValueError("legacy ANN has no linear layers")
    hidden = layers[0].out_features
    activation = type(next(m for m in legacy.net.modules()
                           if isinstance(m, (nn.Tanh, nn.Identity))))
    split = Static_ANN_Block(nz=legacy.nz, nw=len(positions),
                             n_nodes_per_layer=hidden,
                             n_hidden_layers=len(layers) - 1,
                             net=type(legacy.net), activation=activation)
    split.to(device=layers[0].weight.device, dtype=layers[0].weight.dtype)
    new_layers = _linear_layers(split.net)
    with torch.no_grad():
        for old, new in zip(layers[:-1], new_layers[:-1]):
            new.load_state_dict(copy.deepcopy(old.state_dict()))
        new_layers[-1].weight.copy_(layers[-1].weight[list(positions)])
        if layers[-1].bias is not None:
            new_layers[-1].bias.copy_(layers[-1].bias[list(positions)])
    probe = torch.linspace(-1.0, 1.0, 13 * legacy.nz,
                           device=layers[0].weight.device,
                           dtype=layers[0].weight.dtype).reshape(13, legacy.nz, 1)
    with torch.no_grad():
        old_y = legacy(probe)[:, list(positions), :]
        new_y = split(probe)
    diff = (old_y - new_y).abs()
    evidence = {
        "legacy_output_positions": list(positions),
        "tensor_map": [
            {"old": f"net.linear[{i}]", "new": f"net.linear[{i}]",
             "rows": list(positions) if i == len(layers) - 1 else "all"}
            for i in range(len(layers))
        ],
        "max_abs_difference": float(diff.max()),
        "bitwise_equal": bool(torch.equal(old_y, new_y)),
    }
    if not evidence["bitwise_equal"]:
        raise RuntimeError("physical ANN split-copy equivalence failed")
    return split, evidence


def bla_initialize_writer_(block: KesselsExtensionBlock, rho, theta) -> Dict[str, object]:
    """Seed the writer's linear bypass so the pair STARTS at an identified pole. In place.

    Kessels' extension row is `q+ = q + g v`, `v+ = L_psi(...)`, so the local pair Jacobian is the
    companion matrix `[[1, g], [a, b]]` with `g` the kinematic gain. Matching its characteristic
    polynomial to `z^2 - 2 rho cos(theta) z + rho^2` gives

        b = 2 rho cos(theta) - 1,      a = (b - rho^2) / g

    which is an EXACT algebraic construction, not a fit: it is the unique companion row realising
    that pole under this fixed kinematic row. Only the `x_a` columns of the bypass are written; the
    `x_b`/`u` columns and the whole nonlinear branch keep the initialization they were built with,
    and every weight stays trainable. This is therefore an INITIALIZATION, not a placed or frozen
    `A_aa`: nothing here is registered as a buffer, excluded from the optimizer, or
    stability-parameterised.

    PROVENANCE. The construction is Kessels' Eq. (5.4) extension row; the numerical `(rho, theta)`
    must come from an identification supplied by the caller. Neither is a Kessels-published value,
    so an arm using this must be labelled a Kessels/identified-initialization synthesis.

    The writer sees `scale_writer_input(z) = z * mult + offset`, so the stored weight is divided by
    the corresponding multiplier to make the EFFECTIVE Jacobian equal `a` and `b`.
    """
    if block.writer.linear_bypass is None:
        raise ValueError('BLA writer initialization needs the linear bypass; build the block with '
                         'writer_linear_bypass=True')
    rho = torch.as_tensor(rho, dtype=torch.float64).reshape(-1)
    theta = torch.as_tensor(theta, dtype=torch.float64).reshape(-1)
    if rho.numel() != block.m or theta.numel() != block.m:
        raise ValueError('expected %d (rho, theta) pairs, got %d/%d'
                         % (block.m, rho.numel(), theta.numel()))
    if not bool(((rho > 0) & (rho < 1)).all()):
        raise ValueError('every identified radius must lie strictly inside the unit disc')
    g = float(block.kinematic_gain)
    b = 2.0 * rho * torch.cos(theta) - 1.0
    a = (b - rho * rho) / g
    w = block.writer.linear_bypass.weight
    mult = block.writer_input_multiplier
    rows = []
    with torch.no_grad():
        for i in range(block.m):
            q_col = block.nx_phys + i
            v_col = block.nx_phys + block.m + i
            w[i, q_col] = float(a[i]) / float(mult[q_col])
            w[i, v_col] = float(b[i]) / float(mult[v_col])
            rows.append({'pair': i, 'rho': float(rho[i]), 'theta': float(theta[i]),
                         'a': float(a[i]), 'b': float(b[i]), 'kinematic_gain': g,
                         'q_col': q_col, 'v_col': v_col})
    return {'construction': 'companion row from (rho, theta); b = 2 rho cos(theta) - 1, '
                            'a = (b - rho^2) / kinematic_gain',
            'trainable': True, 'pairs': rows}


def augmented_state_jacobian(block: KesselsExtensionBlock, z: Tensor) -> Tensor:
    """`d x_a[k+1] / d x_a[k]` at `z`, exactly. Block-local: `y_k` does not read `x_a`."""
    nxp, n_aug = block.nx_phys, block.n_aug

    def step(xa_flat: Tensor) -> Tensor:
        zz = z.clone()
        zz[0, nxp:nxp + n_aug, 0] = xa_flat
        return block(zz)[0, :, 0]

    return torch.autograd.functional.jacobian(step, z[0, nxp:nxp + n_aug, 0], vectorize=True)


def find_kessels_block(fit_sys) -> Optional[KesselsExtensionBlock]:
    return next((m for m in fit_sys.hfn.connected_blocks
                 if isinstance(m, KesselsExtensionBlock)), None)


def find_physical_ann(fit_sys) -> Static_ANN_Block:
    anns = [m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block)]
    if len(anns) != 1:
        raise RuntimeError(f"expected exactly one physical ANN, found {len(anns)}")
    return anns[0]


def parameter_counts(fit_sys) -> Dict[str, Dict[str, int]]:
    block = find_kessels_block(fit_sys)
    ann = find_physical_ann(fit_sys)
    enc = fit_sys.encoder
    groups = {
        "physical_ann": ann,
        "kessels_writer": block.writer if block is not None else None,
        "physical_encoder": getattr(enc, "physical_encoder", enc),
        "augmented_encoder": getattr(enc, "augmented_encoder", None),
    }
    out = {}
    for name, module in groups.items():
        if module is None:
            out[name] = {"total": 0, "trainable": 0, "frozen": 0}
            continue
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        out[name] = {"total": total, "trainable": trainable, "frozen": total - trainable}
    return out


@dataclass(frozen=True)
class SimilarityMap:
    """Declared raw -> normalized latent coordinate map."""

    sample_time: float
    n_aug: int

    @property
    def diagonal(self) -> Tensor:
        m = self.n_aug // 2
        return torch.tensor([1.0 / self.sample_time] * m + [1.0] * m)


def similarity_map_writer_(raw: KesselsExtensionBlock,
                           normalized: KesselsExtensionBlock) -> None:
    """Map a raw writer into the declared normalized-similarity coordinates in place."""
    if (raw.coordinate_mode != "kessels_raw"
            or normalized.coordinate_mode != "normalized_similarity"):
        raise ValueError("expected raw source and normalized-similarity destination")
    if (raw.n_aug, raw.nx_phys, raw.nu) != (normalized.n_aug,
                                            normalized.nx_phys, normalized.nu):
        raise ValueError("similarity map dimensions differ")
    if raw.writer.linear_bypass_enabled != normalized.writer.linear_bypass_enabled:
        raise ValueError("similarity map requires matched bypass modes")
    normalized.writer.load_state_dict(copy.deepcopy(raw.writer.state_dict()))
    t = normalized.S_q.new_ones(raw.n_aug)
    t[:raw.m] = normalized.S_q / raw.S_q
    t[raw.m:] = normalized.S_v / raw.S_v
    d = normalized.writer_input_multiplier.new_ones(raw.nz)
    d[raw.nx_phys:raw.nx_phys + raw.n_aug] = 1.0 / t
    src_layers = _linear_layers(raw.writer.mlp)
    dst_layers = _linear_layers(normalized.writer.mlp)
    with torch.no_grad():
        dst_layers[0].weight.mul_(d.unsqueeze(0))
        # v' = S_v v.  The deterministic map uses S_v=I, but keep the general row map.
        sv = normalized.S_v / raw.S_v
        dst_layers[-1].weight.mul_(sv.unsqueeze(1))
        dst_layers[-1].bias.mul_(sv)
        if normalized.writer.linear_bypass is not None:
            normalized.writer.linear_bypass.weight.mul_(d.unsqueeze(0))
            normalized.writer.linear_bypass.weight.mul_(sv.unsqueeze(1))


def similarity_map_reader_(raw_reader: nn.Module, normalized_reader: nn.Module,
                           *, nx_phys: int, n_aug: int, transform: Tensor) -> None:
    """Copy a physical reader and map its latent input columns by ``T^-1``."""
    normalized_reader.load_state_dict(copy.deepcopy(raw_reader.state_dict()))
    first = _linear_layers(normalized_reader)[0]
    with torch.no_grad():
        first.weight[:, nx_phys:nx_phys + n_aug].mul_(
            (1.0 / transform.to(first.weight)).unsqueeze(0))


def similarity_map_augmented_encoder_(raw_encoder: KesselsAugmentedEncoder,
                                      normalized_encoder: KesselsAugmentedEncoder,
                                      transform: Tensor) -> None:
    """Copy an augmented encoder and map its output coordinates by ``T``."""
    normalized_encoder.load_state_dict(copy.deepcopy(raw_encoder.state_dict()))
    final = _linear_layers(normalized_encoder.net)[-1]
    t = transform.to(final.weight)
    with torch.no_grad():
        final.weight.mul_(t.unsqueeze(1))
        final.bias.mul_(t)
