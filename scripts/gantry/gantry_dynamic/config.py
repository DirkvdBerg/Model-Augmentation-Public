"""Run configuration for the gantry augmentation training pipeline.

Single source of truth: `RunConfig` holds EVERY user-tunable parameter (both the
experiment knobs and the model/training hyperparameters). The entry file
constructs one object with all fields visible. Derived quantities (d, ts_new,
nf, na_nb, dtype, and the `hp` dict) are read-only properties.

`cfg.hp` is a derived dict view with the exact legacy keys/order. It exists only
because the downstream functions and the checkpoint/npz JSON round-trip consume a
dict; it is NOT a second place to edit parameters. Edit the RunConfig fields.
"""
__project_origin__ = "added"

import os
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch

# Repo root, resolved independently of where this module lives on disk so that
# every data/simulation path matches the pre-refactor absolute paths exactly.
_PKG_DIR  = os.path.dirname(os.path.abspath(__file__))          # scripts/gantry/gantry_dynamic
REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, '..', '..', '..'))


@dataclass(frozen=True)
class RunConfig:
    # ═══ Experiment identity ══════════════════════════════════════════════════
    # --- Track: 'joint' (broadband [1,200] Hz) or 'augmentation' (narrowband [130,180] Hz) ---
    mode: str = 'augmentation'
    # encoder_init: 'linear_map' = Hoekstra 2026 reconstructability init (trainable);
    #               'default' = standard deepSI learned encoder
    encoder_init: str = 'linear_map'
    # ann_activation: 'linear' = Identity activation (Jan's ECC setup, D-071); 'tanh' = nonlinear ANN
    ann_activation: str = 'tanh'
    joint_estimation: bool = False  # D-076: True = trainable damping/stiffness scalars
    param_rmse_baseline: float = 0.01  # HEURISTIC: measured initial sqrt-loss, jobs 68675/68676 (D-076 Lambda scale)
    # D-076 run design: None = start at true values (run T: measures absorber-induced bias).
    # A 14-vector aligned to PARAM_NAMES = detuned start (run D: recovery test).
    # NOTE: param_loss anchors to the (possibly detuned) INIT values -- Jan's prior semantics.
    param_init_detune: Optional[List[float]] = field(default_factory=lambda: [
        1.10, 1.10, 1.10, 0.90, 1.10, 0.90, 0.90, 0.90, 1.10, 0.90, 1.10, 0.90, 0.90, 1.10])
    # --- Output noise (Jan's ECC noise-floor convention, D-078) ---
    # sigma_n = rms(y) * 10^(-SNR/20); reaching sigma_n on val sim-RMS = acceptance floor.
    snr: Optional[int] = None   # dB: 50/55/60; None = noiseless (supervisor 07-07: make it work without noise first)
    seed: int = 42
    # --- Training-loss rollout: closed loop (known controller wrapped around the model)
    #     or open loop (plant input replayed from the record). This selects the OBJECTIVE,
    #     not just a diagnostic: closed loop routes loss() through interconnect.py's
    #     closed-loop rollout and `bestfit` becomes the V1-V4 closed-loop free-run RMS.
    #     False leaves interconnect.py's `simulator = None` default in force (open loop).
    closed_loop: bool = True

    # ═══ Sampling / data conditioning ═════════════════════════════════════════
    fs_orig: int = 20000
    fs_new: Optional[int] = 4000   # None = no downsampling (use fs_orig)
    stride: int = 10               # keep every STRIDE-th BPTT window (STRIDE=1 = every window)
    use_f64: bool = False
    save_flag: bool = True
    nf_probe_print: bool = True    # print per-epoch train/val nf-window RMS (D-095 probe); runtime-only, not in hp

    # ═══ Model + training hyperparameters (were the default_hp dict) ══════════
    nx_ann: int = 2                # augmented (ANN) latent states
    # ANN correction routing: rows the ANN writes into. State layout (logical):
    #   [X, Theta, Y, dX, dTheta, dY, delta_a, vdelta_a] = idx 0..7.
    #   (1,4,6,7)=Theta+absorber (D-068 default, K>0 only); (0..7)=X+Theta+Y+absorber.
    #   NOTE: routing to K=0 rows (X/Y: 0,2,3,5) needs a much smaller lr (~1e-7) -- D-101/D-102.
    ann_route_ix: tuple = (0, 1, 2, 3, 4, 5, 6, 7)
    n_nodes_per_layer: int = 16
    n_hidden_layers: int = 2
    up_sample: int = 2             # model discretization sub-steps per Ts
    batch_size: int = 256
    lr: float = 1e-4
    # Explicit because the augmented writer can have 1e-11..1e-14 gradients; Adam's
    # 1e-8 default then suppresses its effective step. Kept out of `hp` so the legacy
    # checkpoint hyperparameter schema/order remains unchanged.
    adam_eps: float = 1e-8
    epochs: int = 10
    nf_seconds: float = 0.100      # [s] SEGMENT length (5*tau_msd, tau=1/(zeta*wn)=20ms, 5tau=100ms)
    # Optional direct overrides (None = derive). Set a number to bypass the formula.
    nf_override: Optional[int] = None      # None -> nf_seg = nf_seconds / ts_new
    na_nb_override: Optional[int] = None   # None -> na_nb = (nx_phys + nx_ann)*2 + 1 (Jan's rule)

    # ═══ Multiple shooting: the continuity (defect) term (D-127) ══════════════
    # The nf-windows ARE the shooting segments and the encoder IS the node elimination
    # (Beintema et al. L4DC 2021); what was missing is the defect between segments.
    # n_seg=1 + defect_weight=0 is an EXACT no-op (bit-identical to the old loss).
    # The objective then spans n_seg * nf_seconds while the GRADIENT PATH stays
    # nf_seconds, which is the whole point: beta-smoothness is O(N^3) in the
    # WITHIN-segment length at |lambda|=1 (Ribeiro et al. 2020 Thm 1), so raising n_seg
    # is cheap where raising nf_seconds is not (NF=900 diverged, problem-log line 319).
    n_seg: int = 1                  # contiguous shooting segments per training sample
    defect_weight: float = 0.0      # 0 = off; see defect_scale for the principled magnitude
    # Accumulated (summed) defect: a COHERENCE detector, where defect_weight is a
    # POWER detector. MS6 measured the norm BLIND to the failure (1.01x against a
    # 65x free-run ratio, flat in n_seg); MS7 measured the sum RESPONDING and
    # improving with coupling (1.16/0.98/1.37/1.60/2.88 over n_seg 4/12/30/60/119).
    # 0 = off; with defect_weight also 0 the loss stays bit-identical to the old one.
    defect_acc_weight: float = 0.0
    defect_norm: str = 'l2'         # 'l2' (NOT squared) / 'l1' / 'linf' = exact penalties
                                    # (Turan & Jaschke 2022); squared has no finite-weight guarantee
    # Per-state weights W inside the norm. None = unweighted. The principled choice is
    # Q^{-1/2} from the MEASURED residual covariance (Fisher et al. 2011 weak-constraint
    # 4D-Var), never a swept rho.
    defect_scale: Optional[List[float]] = None

    # ═══ Orthogonal-projection regularization (docs/orthogonal-projection-plan.md D7) ═══
    orth_beta: float = 0.0          # 0 = off (exact no-op, D7.2); >0 adds beta*||Q^T f_ANN||^2
    orth_point_stride: int = 100    # penalty point-set decimation (Step 5 coverage verified at 100)
    orth_rank_tol: float = 1e-12    # numerical-rank truncation of Q (plan Sect. 2.2 step 5)
    orth_observe: bool = False      # attach the penalty object with beta possibly 0 so the
                                    # [joint-probe] orth-frac meter observes WITHOUT penalizing
                                    # (loss path skips at beta==0 -- beta=0 control stays a no-op)

    # ═══ Fixed model dimensions ═══════════════════════════════════════════════
    nx_phys: int = 6   # physical states: q1, q2, q3, dq1, dq2, dq3
    nu: int = 3
    ny: int = 3

    # ───────────────────────── Derived quantities ────────────────────────────
    @property
    def fs_new_hz(self) -> int:
        return self.fs_orig if self.fs_new is None else self.fs_new

    @property
    def d(self) -> int:
        return self.fs_orig // self.fs_new_hz

    @property
    def ts_new(self) -> float:
        return 1.0 / self.fs_new_hz

    @property
    def dtype_np(self):
        return np.float64 if self.use_f64 else np.float32

    @property
    def dtype_pt(self):
        return torch.float64 if self.use_f64 else torch.float32

    @property
    def nf_seg(self) -> int:
        """Length of ONE shooting segment [samples]; this is the gradient-path length."""
        if self.nf_override is not None:
            return self.nf_override
        return max(1, int(self.nf_seconds / self.ts_new))

    @property
    def nf(self) -> int:
        """Total steps per training sample = n_seg segments. Equals nf_seg when n_seg=1."""
        return self.n_seg * self.nf_seg

    @property
    def na_nb(self) -> int:
        if self.na_nb_override is not None:
            return self.na_nb_override
        # THEORY: na=nb=nxd*2+1 (Jan's standard; nxd=NX_PHYS+NX_ANN encoder history)
        return (self.nx_phys + self.nx_ann) * 2 + 1

    @property
    def hp(self) -> dict:
        """Derived hyperparameter dict with the legacy keys/order (checkpoint + npz contract)."""
        return dict(
            NX_ANN=self.nx_ann,
            n_nodes_per_layer=self.n_nodes_per_layer,
            n_hidden_layers=self.n_hidden_layers,
            up_sample=self.up_sample,
            nf=self.nf,
            na_nb=self.na_nb,
            batch_size=self.batch_size,
            lr=self.lr,
            epochs=self.epochs,
        )


def default_hp(cfg: RunConfig) -> dict:
    """Backward-compat accessor for the derived hp dict; edit RunConfig fields, not this."""
    return cfg.hp


def save_dir(cfg: RunConfig) -> str:
    """Output directory for this run's simulations (created by the caller)."""
    return os.path.join(REPO_ROOT, 'simulations', 'gantry_subnet',
                        f'{cfg.mode}_{cfg.encoder_init}')


def config_json_dict(cfg: RunConfig) -> dict:
    """Config metadata for the results npz -- exact keys and order of the pre-refactor dump."""
    return dict(
        MODE=cfg.mode, ENCODER_INIT=cfg.encoder_init,
        ANN_ACTIVATION=cfg.ann_activation, FS_NEW=cfg.fs_new_hz, D=cfg.d,
        FS_ORIG=cfg.fs_orig, SEED=cfg.seed, SNR=cfg.snr,
        JOINT_ESTIMATION=cfg.joint_estimation,
        PARAM_RMSE_BASELINE=cfg.param_rmse_baseline,
        PARAM_INIT_DETUNE=cfg.param_init_detune,
        ORTH_BETA=cfg.orth_beta,
        ORTH_POINT_STRIDE=cfg.orth_point_stride,
        ORTH_RANK_TOL=cfg.orth_rank_tol,
        ORTH_OBSERVE=cfg.orth_observe,
        # Appended, not inserted: the pre-refactor key order above is load-bearing
        # for old npz readers. CLOSED_LOOP records which objective the run used --
        # without it an open-loop and a closed-loop run are indistinguishable in the
        # saved metadata, and their `bestfit` numbers are not comparable.
        CLOSED_LOOP=cfg.closed_loop,
        # Appended for backward compatibility with readers that rely on the old order.
        ADAM_EPS=cfg.adam_eps,
    )
