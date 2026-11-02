"""Run configuration: the record of one run's settings, their checks, and config.json; the values are set at the top
of gantry_interconnect_dynamic.py."""
__project_origin__ = "added"

import functools
import json
import os
import subprocess
from dataclasses import dataclass, fields, replace
from typing import ClassVar, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT)
# default folders; --data-root and --out-root override them
DATA_ROOT = os.path.join(REPO_ROOT, 'Thesis-writeup', 'Data')      # the data sets made by Clean/data_generation/
OUT_ROOT = os.path.join(REPO_ROOT, 'simulations', 'gantry_clean')  # run folders <out_root>/<noise>/<run_id>

# fields that config.json does not record: the folders belong to the machine, start_phase to the run mode
_NOT_RECORDED = ('data_root', 'out_root', 'start_phase')


def y_decimation(cfg) -> str:
    """The rule that brings y to fs_new, as recorded in config.json."""
    if cfg.noise == 'lowpass':
        return 'r + zero-phase Butterworth %d at %.0f Hz on y - r, then [::%d]' % (cfg.aa_order, cfg.aa_fc_hz, cfg.d)
    return 'point sampling [::%d]' % cfg.d


def record_trim(cfg) -> int:
    """Return samples trimmed at each fs_new record end; sample i maps to fs_orig index (i + trim) D."""
    return round(cfg.aa_trim_s * cfg.fs_new)


@dataclass(frozen=True)
class RunConfig:
    """One run's settings; each is defined at the top of gantry_interconnect_dynamic.py."""
    # model: the five axes
    arm: str            # 'obc' or 'u' (unconstrained)
    start: str          # baseline parameters at the start: 'correct' or 'detuned'
    n_a: int            # number of added states
    ps2: bool           # PS opening of the added states first
    noise: str          # data set: 'lowpass' or 'noisefree'
    # sampling and windows
    fs_new: int         # [Hz] training rate
    aa_order: int       # order of the anti-aliasing Butterworth (lowpass data set)
    aa_fc_hz: float     # [Hz] its cutoff
    aa_trim_s: float    # [s] cut from both ends of every record
    nf_seconds: float   # [s] simulated length of each training window
    stride: int         # samples between window starts
    na_nb: int          # past samples of y and u the encoder sees
    seed: int           # initial weights, detune signs, polish batch, PS opening
    detune: float       # relative detuning of each baseline parameter on the detuned start
    # networks
    n_nodes_per_layer: int
    n_hidden_layers: int
    up_sample: int      # RK4 steps per sample
    # Adam
    batch_size: int
    lr: float
    adam_eps: float
    epochs: int
    n_its: Optional[int]
    its_per_val: int
    # L-BFGS polish; its optimizer constants are in lbfgs_polish.py
    lbfgs_max_iter: int
    lbfgs_windows: int
    # PS opening: the fields of PS2Spec except its seed, which is seed
    ps2_opening: dict
    # environment
    device: str
    compile_mode: Optional[str]
    truncate_s: Optional[float]
    baseline_form: str
    obc_ref_chunk: int
    data_root: str
    out_root: str
    start_phase: str = 'adam'   # 'adam': Adam, then the polish; 'lbfgs': the polish only (set by the polish-only mode)

    # facts of the system and the data sets, not settings
    nx_phys: ClassVar[int] = 6           # physical states: positions X, Theta, Y [m, rad, m] and their velocities
    nu: ClassVar[int] = 3                # inputs: actuator forces F_X1, F_X2, F_Y [N]
    ny: ClassVar[int] = 3                # outputs: measured positions X1, X2, Y [m]
    fs_orig: ClassVar[int] = 20000       # [Hz] sample rate of the data sets (THEORY: data_generation/tdg_config.m)
    dtype_np: ClassVar[type] = np.float64
    dtype_pt: ClassVar[torch.dtype] = torch.float64

    def __post_init__(self):
        """Check the values, which arrive from the settings, the CLI or a config.json."""
        allowed = dict(arm=('obc', 'u'), start=('detuned', 'correct'), ps2=(False, True),
                       noise=('lowpass', 'noisefree'), device=('cpu', 'cuda'),
                       compile_mode=(None, 'reduce-overhead'),
                       start_phase=('adam', 'lbfgs'), baseline_form=('lfr', 'closed'))
        for name, values in allowed.items():
            if getattr(self, name) not in values:
                raise ValueError('%s must be one of %s, got %r' % (name, values, getattr(self, name)))
        if not isinstance(self.n_a, int) or self.n_a < 0:
            raise ValueError('n_a must be an integer >= 0, got %r' % (self.n_a,))
        if self.compile_mode is not None and self.device != 'cuda':
            raise ValueError('compile_mode=%r needs device="cuda", got %r; set compile_mode=None on the CPU '
                             '(the --cpu flag does)' % (self.compile_mode, self.device))
        if self.fs_orig % self.fs_new:
            raise ValueError('fs_new must divide fs_orig = %d Hz, got %r' % (self.fs_orig, self.fs_new))

    @classmethod
    def from_config_json(cls, meta, **overrides):
        """Read the configuration of a run folder written by this code; overrides set roots, device, budgets and
        start_phase. The settings come from the file; a file whose derived entries this code computes differently
        is refused."""
        # CODE = 'clean' marks a config.json written by config_json_dict; other folders need converting first
        if meta.get('CODE') != 'clean':
            raise ValueError('config.json was not written by this code (CODE %r); convert the run folder with '
                             'clean-up-verification/convert_run.py' % meta.get('CODE'))
        recorded = [f.name for f in fields(cls) if f.name not in _NOT_RECORDED]
        missing = [f.upper() for f in recorded if f.upper() not in meta]
        if missing:
            raise ValueError('config.json lacks %s; convert the run folder with clean-up-verification/convert_run.py'
                             % missing)
        cfg = cls(**{f: meta[f.upper()] for f in recorded}, data_root=DATA_ROOT, out_root=OUT_ROOT)
        want = json.loads(json.dumps(config_json_dict(cfg)))
        bad = {k: (v, meta.get(k, '<missing>')) for k, v in want.items()
               if k != 'GIT' and meta.get(k, '<missing>') != v}
        if bad:
            raise ValueError('config.json differs from what this code derives from its settings (key: (this code, '
                             'file)): %s; use the code of its GIT entry' % bad)
        return replace(cfg, **overrides)

    @property
    def d(self) -> int:
        """Decimation factor fs_orig / fs_new."""
        return self.fs_orig // self.fs_new

    @property
    def ts_new(self) -> float:
        """[s] training sample time."""
        return 1.0 / self.fs_new

    @property
    def nf(self) -> int:
        """Simulation steps per training window."""
        return max(1, int(self.nf_seconds / self.ts_new))

    @property
    def obc(self) -> bool:
        """True on the OBC arm."""
        return self.arm == 'obc'

    @property
    def combo_init_detune(self):
        """Detuned start of the baseline parameters.

        theta^0_base,j = theta_base,j(theta^0) (1 + detune s_j), s_j = +-1 drawn from the seed; build_model applies
        the multipliers, so a seed fixes the direction and its noisy and noise-free runs share it.

        Returns:
            list of 10 dimensionless multipliers 1 + detune s_j, in the order of
            IdentifiableGantryStateBlock.COMBO_NAMES (k_b,Sigma, c_g1, c_g2, c_y, c_b,Sigma, m_h, m_Sigma, m_Delta,
            J_eff, d), or None on the correct arm.
        """
        if self.start == 'correct':
            return None
        signs = np.random.default_rng(self.seed).choice([-1.0, 1.0], size=10)
        return [float(1.0 + self.detune * s) for s in signs]


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
    """The run's settings as config.json records them, read back by RunConfig.from_config_json: every setting under
    its name in capitals, FS_ORIG, the derived Y_DECIMATION and RECORD_TRIM, CODE (this code wrote the folder) and
    GIT."""
    return dict(
        **{f.name.upper(): getattr(cfg, f.name) for f in fields(cfg) if f.name not in _NOT_RECORDED},
        FS_ORIG=cfg.fs_orig,
        Y_DECIMATION=y_decimation(cfg),
        RECORD_TRIM=record_trim(cfg),
        CODE='clean',
        GIT=git,
    )
