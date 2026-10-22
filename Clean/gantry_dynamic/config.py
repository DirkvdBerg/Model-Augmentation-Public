"""Run configuration: the five axes of a thesis run, the run environment and the fixed settings."""
__project_origin__ = "added"

import functools
import json
import os
import subprocess
from dataclasses import dataclass, replace
from typing import ClassVar, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT)
DATA_ROOT = os.path.join(REPO_ROOT, 'Thesis-writeup', 'Data')
OUT_ROOT = os.path.join(REPO_ROOT, 'simulations', 'gantry_clean')

# HEURISTIC: order 8, fc = 0.8 of the 2 kHz Nyquist frequency of the 4 kHz training rate
# THEORY: zero-phase Butterworth magnitude 1 / (1 + x^16), x = tan(pi f / fs) / tan(pi fc / fs) (the order-8
#   response squared by the forward-backward pass): DC gain 1, error 1.7e-12 at 300 Hz and 4.4e-4 at 1 kHz
AA_ORDER, AA_FC_HZ = 8, 1600.0
# HEURISTIC: the filter's edge transient stays above 0.01 of the in-band noise for at most 4.6 ms on the 36
# records; rounded up to whole ms
AA_TRIM_S = 0.005

# the thesis runs: OBC or unconstrained arm, correct or detuned start, n_a = n_{bar x} added states, PS2 opening
# or plain
PRESETS = {
    'R1': dict(arm='obc', start='detuned', n_a=2, ps2=True),
    'R2': dict(arm='obc', start='correct', n_a=2, ps2=True),
    'R3': dict(arm='u', start='correct', n_a=2, ps2=False),
    'R4': dict(arm='u', start='detuned', n_a=2, ps2=True),
    'R5': dict(arm='u', start='correct', n_a=2, ps2=True),
    'R6': dict(arm='u', start='correct', n_a=0, ps2=False),
    'R7': dict(arm='u', start='correct', n_a=8, ps2=True),
    'R13': dict(arm='obc', start='detuned', n_a=0, ps2=False),
    'R14': dict(arm='u', start='detuned', n_a=0, ps2=False),
}

# config.json keys of the run environment fields, read back by from_config_json
_ENV_KEYS = dict(device='DEVICE', compile_mode='COMPILE_MODE', truncate_s='TRUNCATE_S',
                 epochs='EPOCHS', n_its='N_ITS', its_per_val='ITS_PER_VAL',
                 lbfgs_max_iter='LBFGS_MAX_ITER', lbfgs_inner_iter='LBFGS_INNER_ITER',
                 lbfgs_windows='LBFGS_WINDOWS', ps2_smoke='PS2_SMOKE')
# config.json keys only the clean code writes
_CLEAN_KEYS = ('ARM', 'START', 'N_A', 'NOISE', 'PS2_SMOKE', 'DATA_ROOT', 'OUT_ROOT', 'LAST_SAVE', 'CODE')


def y_decimation(cfg) -> str:
    """The rule that brings y to FS_NEW, as recorded in config.json."""
    if cfg.noise == 'lowpass':
        return 'r + zero-phase Butterworth %d at %.0f Hz on y - r, then [::%d]' % (AA_ORDER, AA_FC_HZ, cfg.d)
    return 'point sampling [::%d]' % cfg.d


def record_trim(cfg) -> int:
    """Return samples trimmed at each FS_NEW record end; sample i maps to FS_ORIG index (i + trim) D."""
    return int(round(AA_TRIM_S * cfg.fs_new))


@dataclass(frozen=True)
class RunConfig:
    """One thesis run: the five axes, the run environment and the fixed settings."""
    arm: str
    start: str
    n_a: int
    ps2: bool
    noise: str
    # run environment: production defaults, changed only by the CLI flags, main (start_phase) and the harness
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

    # fixed settings of the thesis runs under the mainline names; config_json_dict writes them and the
    # evaluation reads several
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
    # HEURISTIC: the measured initial square root loss; only the Lambda_combo buffer uses it, and no loss reads it
    param_rmse_baseline: ClassVar[float] = 0.01
    use_f64: ClassVar[bool] = True
    joint_estimation: ClassVar[bool] = True
    lbfgs: ClassVar[bool] = True
    encoder_init: ClassVar[str] = 'data_lsq'
    validation_measure: ClassVar[str] = 'sim-NRMSe'
    controller_gain: ClassVar[bool] = False
    lbfgs_history: ClassVar[int] = 50
    # HEURISTIC: fixed gradient and loss-change tolerances for the servo-weighted objective
    lbfgs_tol_grad: ClassVar[float] = 2.6e-5
    lbfgs_tol_change: ClassVar[float] = 2.6e-7
    obc_space: ClassVar[str] = 'affine'
    obc_refresh: ClassVar[str] = 'epoch'
    obc_rank_rtol: ClassVar[Optional[float]] = None
    obc_ref_successor: ClassVar[bool] = True
    obc_ref_chunk: ClassVar[int] = 16384
    # derived; nf = steps per training window (n_f), also the length of the gradient path
    d: ClassVar[int] = fs_orig // fs_new
    ts_new: ClassVar[float] = 1.0 / fs_new
    nf: ClassVar[int] = max(1, int(nf_seconds / ts_new))
    dtype_np: ClassVar[type] = np.float64
    dtype_pt: ClassVar[torch.dtype] = torch.float64

    def __post_init__(self):
        """Check the axis and environment values, which arrive from the CLI, a config.json or the harness."""
        allowed = dict(arm=('obc', 'u'), start=('detuned', 'correct'), n_a=(0, 2, 8), ps2=(False, True),
                       noise=('lowpass', 'noisefree'), device=('cpu', 'cuda'),
                       compile_mode=(None, 'default', 'reduce-overhead', 'max-autotune'),
                       start_phase=('adam', 'lbfgs'))
        for name, values in allowed.items():
            if getattr(self, name) not in values:
                raise ValueError('%s must be one of %s, got %r' % (name, values, getattr(self, name)))
        if self.compile_mode is not None and self.device != 'cuda':
            raise ValueError('compile_mode=%r needs device="cuda", got %r; set compile_mode=None on the CPU '
                             '(the --cpu flag does)' % (self.compile_mode, self.device))

    @classmethod
    def from_config_json(cls, meta, **overrides):
        """Read a supported run configuration, taking roots and start_phase from defaults or overrides."""
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
            raise ValueError('config.json has settings outside the thesis runs (key: (expected, file)): %s; '
                             'run this folder with the mainline code' % bad)
        return replace(cfg, **overrides)

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
        # HEURISTIC: 10 % per combination, the sign of each drawn from the seed, so a seed fixes the direction
        # and its noisy and noise-free runs share it
        signs = np.random.default_rng(self.seed).choice([-1.0, 1.0], size=10)
        return [float(1.0 + 0.10 * s) for s in signs]

    @property
    def hp(self) -> dict:
        """The hyperparameter dict; its keys and order are those of the checkpoint .npz field hp."""
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
    """Output folder of the run family; the caller creates the run folder below it."""
    return os.path.join(cfg.out_root, f'{cfg.mode}_{cfg.encoder_init}')


@functools.lru_cache(maxsize=1)
def git_provenance() -> str:
    """The code that produced this run, e.g. '0bded3d Augmentation dirty:302 diff:3b437e0c'."""
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
    """Return mainline configuration keys in their original order, then Clean keys and GIT."""
    return dict(
        MODE=cfg.mode, ENCODER_INIT=cfg.encoder_init,
        ANN_ACTIVATION='tanh',
        ANN_ARCH='mlp',
        G_INIT='live',
        LIVE_G_WEIGHT_INIT='kaiming_uniform_a_sqrt5',
        WA_ENCODER_INIT='xavier_uniform_gain1',
        FS_NEW=cfg.fs_new, D=cfg.d,
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
