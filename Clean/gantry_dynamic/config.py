"""Run configuration: the five axes of a thesis run, the run environment and the fixed settings."""
__project_origin__ = "added"

import dataclasses
import functools
import json
import os
import subprocess
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT)
DATA_ROOT = os.path.join(REPO_ROOT, 'Thesis-writeup', 'Data')
OUT_ROOT = os.path.join(REPO_ROOT, 'simulations', 'gantry_clean')

# HEURISTIC: order 8, fc = 0.8 x the 2 kHz Nyquist of the 4 kHz training rate (TR-022's ratio)
# THEORY: Butterworth forward-backward magnitude 1/(1 + x^16), x = tan(pi f/fs) / tan(pi fc/fs):
#   DC gain exactly 1, error 1.7e-12 at 300 Hz and 4.4e-4 at 1 kHz (anti-alias/LITERATURE.md Q3)
AA_ORDER, AA_FC_HZ = 8, 1600.0
# HEURISTIC: the filter's edge transient lasts at most 4.6 ms (above 0.01 x the in-band noise, all
# 36 records, anti-alias/end_transient.py), rounded up to whole ms
AA_TRIM_S = 0.005

PRESETS = {
    'R1': dict(arm='obc', start='detuned', n_a=2, ps2=True),     # runs.tsv 221
    'R2': dict(arm='obc', start='correct', n_a=2, ps2=True),     # 222
    'R3': dict(arm='u', start='correct', n_a=2, ps2=False),      # 223
    'R4': dict(arm='u', start='detuned', n_a=2, ps2=True),       # 224
    'R5': dict(arm='u', start='correct', n_a=2, ps2=True),       # 225
    'R6': dict(arm='u', start='correct', n_a=0, ps2=False),      # 226
    'R7': dict(arm='u', start='correct', n_a=8, ps2=True),       # 227
    'R13': dict(arm='obc', start='detuned', n_a=0, ps2=False),   # 229
    'R14': dict(arm='u', start='detuned', n_a=0, ps2=False),     # 230
}

# config.json keys of the run-environment fields, read back by from_config_json
_ENV_KEYS = dict(device='DEVICE', compile_mode='COMPILE_MODE', truncate_s='TRUNCATE_S',
                 epochs='EPOCHS', n_its='N_ITS', its_per_val='ITS_PER_VAL',
                 lbfgs_max_iter='LBFGS_MAX_ITER', lbfgs_inner_iter='LBFGS_INNER_ITER',
                 lbfgs_windows='LBFGS_WINDOWS', ps2_smoke='PS2_SMOKE')
_CLEAN_KEYS = ('ARM', 'START', 'N_A', 'NOISE', 'PS2_SMOKE', 'DATA_ROOT', 'OUT_ROOT', 'LAST_SAVE', 'CODE')


def y_decimation(cfg) -> str:
    """The D-232 rule that brings y to FS_NEW, as recorded in config.json."""
    if cfg.noise == 'lowpass':
        return 'r + zero-phase Butterworth %d at %.0f Hz on y - r, then [::%d]' % (AA_ORDER, AA_FC_HZ, cfg.d)
    return 'point sampling [::%d]' % cfg.d


def record_trim(cfg) -> int:
    """Samples dropped at EACH record end at FS_NEW (D-232): loader sample i is the recorded
    FS_ORIG sample (i + record_trim) * D."""
    return int(round(AA_TRIM_S * cfg.fs_new_hz))


@dataclass(frozen=True)
class RunConfig:
    # the five axes
    arm: str
    start: str
    n_a: int
    ps2: bool
    noise: str
    # run environment: production defaults; only the CLI flags, main (start_phase) and the harness change them
    device: str = 'cuda'
    compile_mode: Optional[str] = 'reduce-overhead'
    truncate_s: Optional[float] = None
    epochs: int = 200
    n_its: Optional[int] = None
    its_per_val: int = 1300
    lbfgs_max_iter: int = 100
    lbfgs_inner_iter: int = 20
    lbfgs_windows: int = 16384
    ps2_smoke: bool = False
    start_phase: str = 'adam'
    # roots: recorded in config.json, never read back from it
    data_root: str = DATA_ROOT
    out_root: str = OUT_ROOT

    # fixed settings of the thesis runs (runs.tsv 221 to 230), under their mainline names
    seed: ClassVar[int] = 1
    nx_phys: ClassVar[int] = 6
    nu: ClassVar[int] = 3
    ny: ClassVar[int] = 3
    fs_orig: ClassVar[int] = 20000
    fs_new: ClassVar[int] = 4000
    stride: ClassVar[int] = 10
    nf_seconds: ClassVar[float] = 0.100
    na_nb: ClassVar[int] = 30
    n_nodes_per_layer: ClassVar[int] = 16
    n_hidden_layers: ClassVar[int] = 2
    up_sample: ClassVar[int] = 1
    batch_size: ClassVar[int] = 512
    lr: ClassVar[float] = 1e-5
    adam_eps: ClassVar[float] = 1e-16
    param_prior: ClassVar[bool] = False
    param_rmse_baseline: ClassVar[float] = 0.01  # HEURISTIC: measured initial sqrt-loss, jobs 68675/68676 (D-076 Lambda scale)
    use_f64: ClassVar[bool] = True
    joint_estimation: ClassVar[bool] = True
    lbfgs: ClassVar[bool] = True
    encoder_init: ClassVar[str] = 'data_lsq'
    validation_measure: ClassVar[str] = 'sim-NRMSe'
    controller_gain: ClassVar[bool] = False
    lbfgs_history: ClassVar[int] = 50
    # HEURISTIC (D-263, FINAL-RUN-FIXES item 16): the production polish tolerances (1e-12, 1e-14, set for the
    # unweighted loss) kept at the same ratio to the loss. The sigma_e weighting scales the loss by 2.58e7 and the
    # largest gradient entry by 2.60e7 (local pilot 2026-10-16: 128 windows of the full training records, same
    # model, weighted vs unweighted).
    lbfgs_tol_grad: ClassVar[float] = 2.6e-5
    lbfgs_tol_change: ClassVar[float] = 2.6e-7
    obc_space: ClassVar[str] = 'affine'
    obc_refresh: ClassVar[str] = 'epoch'
    obc_rank_rtol: ClassVar[Optional[float]] = None
    obc_ref_successor: ClassVar[bool] = True
    obc_ref_chunk: ClassVar[int] = 16384

    def __post_init__(self):
        """The axis and environment values arrive from the CLI, a config.json or the harness."""
        allowed = dict(arm=('obc', 'u'), start=('detuned', 'correct'), n_a=(0, 2, 8), ps2=(False, True),
                       noise=('lowpass', 'noisefree'), device=('cpu', 'cuda'),
                       compile_mode=(None, 'default', 'reduce-overhead', 'max-autotune'),
                       start_phase=('adam', 'lbfgs'))
        for name, values in allowed.items():
            if getattr(self, name) not in values:
                raise ValueError('%s must be one of %s, got %r' % (name, values, getattr(self, name)))
        if self.compile_mode is not None and self.device != 'cuda':
            raise ValueError('compile_mode=%r needs device="cuda", got %r' % (self.compile_mode, self.device))

    @classmethod
    def from_config_json(cls, meta, **overrides):
        """D-275 D2: a run's config.json (clean or mainline) as a clean config; any setting outside
        the spec is rejected. Roots and start_phase come from `overrides` or the defaults."""
        if meta.get('CODE') == 'clean':
            axes = dict(arm=meta['ARM'], start=meta['START'], n_a=int(meta['N_A']), ps2=bool(meta['PS2']),
                        noise=meta['NOISE'])
        else:
            axes = dict(arm='obc' if meta['OBC'] else 'u',
                        start='correct' if meta['COMBO_INIT_DETUNE'] is None else 'detuned',
                        n_a=int(meta['NX_ANN']), ps2=bool(meta.get('PS2', False)),
                        noise={'thesis_taf_lowpass_noise': 'lowpass',
                               'thesis_taf_noisefree': 'noisefree'}[meta['MODE']])
        cfg = cls(**axes, **{f: meta[k] for f, k in _ENV_KEYS.items() if k in meta})
        want = json.loads(json.dumps(config_json_dict(cfg)))
        skip = set(_ENV_KEYS.values()) | set(_CLEAN_KEYS) | {'GIT', 'NF_PROBE_PRINT', 'START_PHASE'}
        bad = {k: (want[k], meta.get(k, '<missing>')) for k in want
               if k not in skip and meta.get(k, '<missing>') != want[k]}
        if bad:
            raise ValueError('config.json is outside the clean spec (key: (spec, file)): %s' % bad)
        return dataclasses.replace(cfg, **overrides)

    # axis properties
    @property
    def obc(self) -> bool:
        return self.arm == 'obc'

    @property
    def nx_ann(self) -> int:
        return self.n_a

    @property
    def ann_route_ix(self) -> tuple:
        return tuple(range(6 + self.n_a))

    @property
    def mode(self) -> str:
        return {'lowpass': 'thesis_taf_lowpass_noise', 'noisefree': 'thesis_taf_noisefree'}[self.noise]

    @property
    def combo_init_detune(self):
        if self.start == 'correct':
            return None
        # HEURISTIC (D-222): D-191's 10 % magnitude, one sign per combination drawn from the seed, so
        # a start fixes its direction and the noisy and noise-free twins of a seed share it.
        _signs = np.random.default_rng(self.seed).choice([-1.0, 1.0], size=10)
        return [float(1.0 + 0.10 * s) for s in _signs]

    # derived quantities
    @property
    def fs_new_hz(self) -> int:
        return self.fs_new

    @property
    def d(self) -> int:
        return self.fs_orig // self.fs_new_hz

    @property
    def ts_new(self) -> float:
        return 1.0 / self.fs_new_hz

    @property
    def dtype_np(self):
        return np.float64

    @property
    def dtype_pt(self):
        return torch.float64

    @property
    def nf(self) -> int:
        """Steps per training sample; this is also the gradient-path length."""
        return max(1, int(self.nf_seconds / self.ts_new))

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


def save_dir(cfg: RunConfig) -> str:
    """Output directory for this run's family (the run folder is created below it by the caller)."""
    return os.path.join(cfg.out_root, f'{cfg.mode}_{cfg.encoder_init}')


@functools.lru_cache(maxsize=1)
def git_provenance() -> str:
    """Which code produced this run: '0bded3d Augmentation dirty:302 diff:3b437e0c'."""
    def _git(*args):
        return subprocess.run(('git',) + args, cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        sha = _git('rev-parse', '--short', 'HEAD')
        branch = _git('rev-parse', '--abbrev-ref', 'HEAD')
        n_dirty = len([ln for ln in _git('status', '--porcelain').splitlines() if ln.strip()])
        if not n_dirty:
            return '%s %s clean' % (sha, branch)
        diff = subprocess.run(('git', 'diff', 'HEAD'), cwd=REPO_ROOT,
                              capture_output=True, check=True)
        h = subprocess.run(('git', 'hash-object', '--stdin'), cwd=REPO_ROOT,
                           input=diff.stdout, capture_output=True, check=True)
        return '%s %s dirty:%d diff:%s' % (sha, branch, n_dirty,
                                           h.stdout.decode().strip()[:8])
    except Exception as e:
        env = os.environ.get('GIT_SHA') or os.environ.get('GIT_COMMIT')
        if env:
            return '%s (from the environment; git not usable here: %s)' % (
                env.strip(), type(e).__name__)
        return ('unavailable (%s) and no GIT_SHA in the environment: this run has NO code '
                'provenance' % type(e).__name__)


def config_json_dict(cfg: RunConfig, git: Optional[str] = None) -> dict:
    """Every mainline config.json key in mainline order with the value of this run (removed options
    as the literals the final runs wrote), then the clean-only keys and GIT."""
    return dict(
        MODE=cfg.mode, ENCODER_INIT=cfg.encoder_init,
        ANN_ACTIVATION='tanh',
        ANN_ARCH='mlp',
        G_INIT='live',
        LIVE_G_WEIGHT_INIT='kaiming_uniform_a_sqrt5',
        WA_ENCODER_INIT='xavier_uniform_gain1',
        FS_NEW=cfg.fs_new_hz, D=cfg.d,
        FS_ORIG=cfg.fs_orig, SEED=cfg.seed, SNR=None,
        JOINT_ESTIMATION=cfg.joint_estimation,
        PHYSICS_PARAMETERIZATION='reduced',
        PARAM_RMSE_BASELINE=cfg.param_rmse_baseline,
        PARAM_INIT_DETUNE=None,
        COMBO_INIT_DETUNE=cfg.combo_init_detune,
        ORTH_BETA=4.66e-4,
        ORTH_POINT_STRIDE=100,
        ORTH_RANK_TOL=1e-12,
        CLOSED_LOOP=True,
        ADAM_EPS=cfg.adam_eps,
        ORTH=False,
        ANN_ROUTE_IX=list(cfg.ann_route_ix),
        STRIDE=cfg.stride,
        N_ITS=cfg.n_its,
        ITS_PER_VAL=cfg.its_per_val,
        NF_SECONDS=cfg.nf_seconds,
        USE_F64=cfg.use_f64,
        SAVE_FLAG=True,
        NF_PROBE_PRINT=False,
        NX_ANN=cfg.nx_ann,
        N_NODES_PER_LAYER=cfg.n_nodes_per_layer,
        N_HIDDEN_LAYERS=cfg.n_hidden_layers,
        UP_SAMPLE=cfg.up_sample,
        BATCH_SIZE=cfg.batch_size,
        LR=cfg.lr,
        EPOCHS=cfg.epochs,
        NF=cfg.nf,
        NA_NB=cfg.na_nb,
        DEVICE=cfg.device,
        CHECKPOINT_CHUNK=0,
        COMPILE_MODE=cfg.compile_mode,
        LBFGS=cfg.lbfgs,
        LBFGS_MAX_ITER=cfg.lbfgs_max_iter,
        LBFGS_INNER_ITER=cfg.lbfgs_inner_iter,
        LBFGS_HISTORY=cfg.lbfgs_history,
        LBFGS_WINDOWS=cfg.lbfgs_windows,
        LBFGS_CHUNK=None,
        LBFGS_TOL_GRAD=cfg.lbfgs_tol_grad,
        LBFGS_TOL_CHANGE=cfg.lbfgs_tol_change,
        LBFGS_FREEZE=[],
        LBFGS_TIME_BUDGET_S=None,
        LBFGS_METER_REL_TOL=0.001,
        START_PHASE=cfg.start_phase,
        PARAM_PRIOR=cfg.param_prior,
        STATIC_PASS_BUFFERS=False,
        OBC=cfg.obc,
        OBC_SPACE=cfg.obc_space,
        OBC_REFRESH=cfg.obc_refresh,
        OBC_REF_CHUNK=cfg.obc_ref_chunk,
        OBC_RANK_RTOL=cfg.obc_rank_rtol,
        TRUNCATE_S=cfg.truncate_s,
        CONTROLLER_GAIN=cfg.controller_gain,
        PS2=cfg.ps2,
        Y_DECIMATION=y_decimation(cfg),
        RECORD_TRIM=record_trim(cfg),
        LOSS_WEIGHTING='servo',
        VALIDATION_MEASURE=cfg.validation_measure,
        LBFGS_METERS_GATE=False,
        OBC_REF_SUCCESSOR=cfg.obc_ref_successor,
        ARM=cfg.arm,
        START=cfg.start,
        N_A=cfg.n_a,
        NOISE=cfg.noise,
        PS2_SMOKE=cfg.ps2_smoke,
        DATA_ROOT=cfg.data_root,
        OUT_ROOT=cfg.out_root,
        LAST_SAVE='every validation',
        CODE='clean',
        GIT=git,
    )
