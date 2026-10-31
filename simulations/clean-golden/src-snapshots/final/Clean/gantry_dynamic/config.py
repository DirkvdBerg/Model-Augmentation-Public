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
                 lbfgs_windows='LBFGS_WINDOWS', baseline_form='BASELINE_FORM')


def y_decimation(cfg) -> str:
    """The rule that brings y to FS_NEW, as recorded in config.json."""
    if cfg.noise == 'lowpass':
        return 'r + zero-phase Butterworth %d at %.0f Hz on y - r, then [::%d]' % (AA_ORDER, AA_FC_HZ, cfg.d)
    return 'point sampling [::%d]' % cfg.d


def record_trim(cfg) -> int:
    """Return samples trimmed at each FS_NEW record end; sample i maps to FS_ORIG index (i + trim) D."""
    return round(AA_TRIM_S * cfg.fs_new)


@dataclass(frozen=True)
class RunConfig:
    """One thesis run: the five axes, the run environment and the fixed settings."""
    arm: str
    start: str
    n_a: int
    ps2: bool
    noise: str
    # run environment: production defaults, changed by the CLI flags and by main (start_phase)
    device: str = 'cuda'
    compile_mode: Optional[str] = 'reduce-overhead'
    truncate_s: Optional[float] = None
    # HEURISTIC: Adam budget of the thesis runs, in epochs
    epochs: int = 200
    n_its: Optional[int] = None
    # HEURISTIC: Adam updates between validations (checkpoint selection) of the thesis runs
    its_per_val: int = 1300
    # HEURISTIC: L-BFGS polish budget: iterations, iterations per step() call, and windows in its one fixed batch
    # (memory: lower it after an out-of-memory error)
    lbfgs_max_iter: int = 100
    lbfgs_inner_iter: int = 20
    lbfgs_windows: int = 16384
    start_phase: str = 'adam'
    # 'lfr': the derivative through the LFR output equation, as the thesis runs; 'closed': (qdot, a) directly
    baseline_form: str = 'lfr'
    # roots: set by the CLI or the caller, not read from config.json (main records the data folder as data_dir)
    data_root: str = DATA_ROOT
    out_root: str = OUT_ROOT

    # fixed settings of the method; config_json_dict records those the code computes with
    seed: ClassVar[int] = 1
    nx_phys: ClassVar[int] = 6
    nu: ClassVar[int] = 3
    ny: ClassVar[int] = 3
    # THEORY: simulation rate of the data sets (Clean/data_generation/tdg_config.m, cfg.fs = 20e3)
    fs_orig: ClassVar[int] = 20000
    # HEURISTIC: training rate of the thesis runs, an integer decimation D = fs_orig / fs_new of the data
    fs_new: ClassVar[int] = 4000
    # HEURISTIC: window stride, window length n_f T_s and encoder window n + 1 = na_nb samples of the thesis runs
    stride: ClassVar[int] = 10
    nf_seconds: ClassVar[float] = 0.100
    na_nb: ClassVar[int] = 30
    # HEURISTIC: width n_h and depth of phi_aug and psi_tilde of the thesis runs
    n_nodes_per_layer: ClassVar[int] = 16
    n_hidden_layers: ClassVar[int] = 2
    up_sample: ClassVar[int] = 1
    # HEURISTIC: Adam minibatch, rate and eps of the thesis runs (eps 1e-16 instead of the 1e-8 default, chosen
    # for the float64 runs)
    batch_size: ClassVar[int] = 512
    lr: ClassVar[float] = 1e-5
    adam_eps: ClassVar[float] = 1e-16
    # HEURISTIC: L-BFGS memory, the curvature pairs kept
    lbfgs_history: ClassVar[int] = 50
    # HEURISTIC: fixed gradient and loss-change tolerances for the servo-weighted objective
    lbfgs_tol_grad: ClassVar[float] = 2.6e-5
    lbfgs_tol_change: ClassVar[float] = 2.6e-7
    # HEURISTIC: OBC reference tuples per jacfwd call in the basis build: memory only, the basis is the same at any
    # chunk
    obc_ref_chunk: ClassVar[int] = 16384
    # derived; nf = steps per training window (n_f), also the length of the gradient path
    d: ClassVar[int] = fs_orig // fs_new
    ts_new: ClassVar[float] = 1.0 / fs_new
    nf: ClassVar[int] = max(1, int(nf_seconds / ts_new))
    dtype_np: ClassVar[type] = np.float64
    dtype_pt: ClassVar[torch.dtype] = torch.float64

    def __post_init__(self):
        """Check the axis and environment values, which arrive from the CLI or a config.json."""
        allowed = dict(arm=('obc', 'u'), start=('detuned', 'correct'), n_a=(0, 2, 8), ps2=(False, True),
                       noise=('lowpass', 'noisefree'), device=('cpu', 'cuda'),
                       compile_mode=(None, 'reduce-overhead'),
                       start_phase=('adam', 'lbfgs'), baseline_form=('lfr', 'closed'))
        for name, values in allowed.items():
            if getattr(self, name) not in values:
                raise ValueError('%s must be one of %s, got %r' % (name, values, getattr(self, name)))
        if self.compile_mode is not None and self.device != 'cuda':
            raise ValueError('compile_mode=%r needs device="cuda", got %r; set compile_mode=None on the CPU '
                             '(the --cpu flag does)' % (self.compile_mode, self.device))

    @classmethod
    def from_config_json(cls, meta, **overrides):
        """Read the configuration of a run folder written by this code; overrides set roots, device and
        start_phase. Every recorded setting must equal this code's, so a folder written with other settings is
        refused."""
        if meta.get('CODE') != 'clean':
            raise ValueError('config.json was not written by this code (CODE %r); convert the run folder with '
                             'clean-up-verification/convert_run.py' % meta.get('CODE'))
        cfg = cls(arm=meta['ARM'], start=meta['START'], n_a=int(meta['N_A']), ps2=bool(meta['PS2']),
                  noise=meta['NOISE'], **{f: meta[k] for f, k in _ENV_KEYS.items() if k in meta})
        want = json.loads(json.dumps(config_json_dict(cfg)))
        bad = {k: (v, meta.get(k, '<missing>')) for k, v in want.items()
               if k != 'GIT' and meta.get(k, '<missing>') != v}
        if bad:
            raise ValueError('config.json was written with other settings (key: (this code, file)): %s; use the '
                             'code of its GIT entry' % bad)
        return replace(cfg, **overrides)

    @property
    def obc(self) -> bool:
        """True on the OBC arm."""
        return self.arm == 'obc'

    @property
    def combo_init_detune(self):
        """Detuned start of the baseline parameters.

        theta^0_base,j = theta_base,j(theta^0) (1 + 0.1 s_j), s_j = +-1 drawn from the seed; build_model applies the
        multipliers, so a seed fixes the direction and its noisy and noise-free runs share it.

        Returns:
            list of 10 dimensionless multipliers 1 + 0.1 s_j, or None on the correct arm.
        """
        if self.start == 'correct':
            return None
        # HEURISTIC: s_j in {-1, +1}, multiplier_j = 1 + 0.1 s_j.
        signs = np.random.default_rng(self.seed).choice([-1.0, 1.0], size=10)
        return [float(1.0 + 0.10 * s) for s in signs]

    @property
    def hp(self) -> dict:
        """The hyperparameter dict that build_model and train read."""
        return dict(
            n_a=self.n_a,
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
    """Output folder of the run family, one per data set; the caller creates the run folder below it."""
    return os.path.join(cfg.out_root, cfg.noise)


@functools.lru_cache(maxsize=1)
def git_provenance() -> str:
    """The code that produced this run, e.g. '0bded3d Augmentation dirty:302 diff:3b437e0c'."""
    def _git(*args):
        """stdout of one git command in the repository."""
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
    """The run's settings as config.json records them, read back by RunConfig.from_config_json: the five axes, the
    run environment, the fixed settings the code computes with, CODE (this code wrote the folder) and GIT."""
    return dict(
        ARM=cfg.arm, START=cfg.start, N_A=cfg.n_a, PS2=cfg.ps2, NOISE=cfg.noise,
        **{key: getattr(cfg, field) for field, key in _ENV_KEYS.items()},
        FS_ORIG=cfg.fs_orig,
        FS_NEW=cfg.fs_new,
        Y_DECIMATION=y_decimation(cfg),
        RECORD_TRIM=record_trim(cfg),
        SEED=cfg.seed,
        STRIDE=cfg.stride,
        NF_SECONDS=cfg.nf_seconds,
        NA_NB=cfg.na_nb,
        N_NODES_PER_LAYER=cfg.n_nodes_per_layer,
        N_HIDDEN_LAYERS=cfg.n_hidden_layers,
        UP_SAMPLE=cfg.up_sample,
        BATCH_SIZE=cfg.batch_size,
        LR=cfg.lr,
        ADAM_EPS=cfg.adam_eps,
        LBFGS_HISTORY=cfg.lbfgs_history,
        LBFGS_TOL_GRAD=cfg.lbfgs_tol_grad,
        LBFGS_TOL_CHANGE=cfg.lbfgs_tol_change,
        CODE='clean',
        GIT=git,
    )
