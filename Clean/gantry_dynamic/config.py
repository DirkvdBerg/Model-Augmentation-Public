"""Run configuration: one run's settings, their checks, and the run folder's config.json that stores them. The values
are set at the top of gantry_interconnect_dynamic.py."""
__project_origin__ = "added"

import os
from dataclasses import dataclass, fields, replace
from typing import ClassVar, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ROOT)
# default folders; --data-root and --out-root override them
DATA_ROOT = os.path.join(REPO_ROOT, 'Thesis-writeup', 'Data')      # the data sets made by data_generation/
OUT_ROOT = os.path.join(REPO_ROOT, 'simulations', 'gantry_clean')  # run folders <out_root>/<noise>/<run_id>

# paths on the machine that runs the code; config.json leaves them out, so a run folder copied to another machine
# (e.g. the cluster) uses that machine's paths
_PATH_FIELDS = ('data_root', 'out_root')


@dataclass(frozen=True)
class RunConfig:
    """The settings of one run. Their values, and what each one means, are at the top of
    gantry_interconnect_dynamic.py."""
    # model
    arm: str
    start: str
    n_a: int
    ps2: bool
    noise: str
    # sampling and windows
    fs_new: int
    aa_order: int
    aa_fc_hz: float
    aa_trim_s: float
    nf_seconds: float
    stride: int
    na_nb: int
    seed: int
    detune: float
    # networks
    n_nodes_per_layer: int
    n_hidden_layers: int
    up_sample: int
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
    # PS opening: the PS2Spec settings; its seed is the run's seed
    ps2_opening: dict
    # environment
    device: str
    compile_mode: Optional[str]
    truncate_s: Optional[float]
    baseline_form: str
    obc_ref_chunk: int
    data_root: str
    out_root: str

    # fixed values, not settings: sizes of the gantry model, sample rate of the data sets, number type
    nx_phys: ClassVar[int] = 6           # physical states: positions X, Theta, Y [m, rad, m] and their velocities
    nu: ClassVar[int] = 3                # inputs: actuator forces F_X1, F_X2, F_Y [N]
    ny: ClassVar[int] = 3                # outputs: measured positions X1, X2, Y [m]
    fs_orig: ClassVar[int] = 20000       # [Hz] sample rate of the data sets, set in data_generation/tdg_config.m
    # float64: on the server GPU a compiled training step is not slower than in float32
    dtype_np: ClassVar[type] = np.float64
    dtype_pt: ClassVar[torch.dtype] = torch.float64

    def __post_init__(self):
        """Raise a ValueError on a value this code does not handle; without it a typo such as arm='OBC' would silently
        run the U arm."""
        allowed = dict(arm=('obc', 'u'), start=('detuned', 'correct'), ps2=(False, True),
                       noise=('lowpass', 'noisefree'), device=('cpu', 'cuda'),
                       compile_mode=(None, 'reduce-overhead'),
                       baseline_form=('lfr', 'closed'))
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
        """RunConfig of a run folder, from its config.json; overrides, e.g. the folders or the device, replace
        fields."""
        recorded = [f.name for f in fields(cls) if f.name not in _PATH_FIELDS]
        missing = [f.upper() for f in recorded if f.upper() not in meta]
        if missing:
            raise ValueError('config.json lacks %s; convert the run folder with clean-up-verification/convert_run.py'
                             % missing)
        cfg = cls(**{f: meta[f.upper()] for f in recorded}, data_root=DATA_ROOT, out_root=OUT_ROOT)
        return replace(cfg, **overrides)

    @property
    def decimation(self) -> int:
        """Decimation factor fs_orig / fs_new."""
        return self.fs_orig // self.fs_new

    @property
    def ts_new(self) -> float:
        """[s] training sample time."""
        return 1.0 / self.fs_new

    @property
    def nf(self) -> int:
        """Simulation steps per training window."""
        return round(self.nf_seconds * self.fs_new)

    @property
    def record_trim(self) -> int:
        """Samples cut from each end of a record: aa_trim_s at fs_new."""
        return round(self.aa_trim_s * self.fs_new)

    @property
    def obc(self) -> bool:
        """True on the OBC arm."""
        return self.arm == 'obc'

    @property
    def combo_init_detune(self):
        """Multipliers 1 +- detune on the 10 baseline parameters, signs drawn from the seed; None with the correct
        start."""
        if self.start == 'correct':
            return None
        signs = np.random.default_rng(self.seed).choice([-1.0, 1.0], size=10)
        return [float(1.0 + self.detune * s) for s in signs]


def save_dir(cfg: RunConfig) -> str:
    """Folder of all runs on one data set, <out_root>/<noise>; main in gantry_interconnect_dynamic.py creates each
    run folder in it."""
    return os.path.join(cfg.out_root, cfg.noise)


def config_json_dict(cfg: RunConfig) -> dict:
    """The run's settings as config.json stores them: each field in capitals except the machine paths, plus
    FS_ORIG."""
    return dict(
        **{f.name.upper(): getattr(cfg, f.name) for f in fields(cfg) if f.name not in _PATH_FIELDS},
        FS_ORIG=cfg.fs_orig,
    )
