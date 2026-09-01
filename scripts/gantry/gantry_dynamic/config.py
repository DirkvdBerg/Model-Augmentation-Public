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

import functools
import os
import subprocess
from dataclasses import dataclass, field, fields
from typing import Optional, List, Union

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

    # ═══ Execution (D-169) ════════════════════════════════════════════════════
    # WHERE the rollout runs. A device STRING, not a bool: 'gpu=True' cannot say which device,
    # and every framework seam downstream already speaks torch device strings
    # (init_model(device=...), Interconnect.forward's device re-homing).
    #
    # Only 'cpu' and 'cuda' are accepted, NOT 'cuda:N'. Card selection on the cluster is SLURM's
    # job via CUDA_VISIBLE_DEVICES, which makes the allocated card 'cuda'; honouring 'cuda:N'
    # here would additionally require editing deepSI's fit(), whose batch move is a bare
    # `.cuda()` (interconnect.py:823).
    #
    # NOT A SECOND IMPLEMENTATION. The rollout, loss, encoder and controller are pure batched
    # tensor ops and run unchanged on both devices; this field only decides where tensors are
    # created. There is no `if device == 'cuda'` branch in any hot path, deliberately: two
    # implementations of the physics is how a CPU diagnostic and a GPU training run stop being
    # the same experiment (docs/pytorch-optimization-guidelines.md, "no conditional compilation").
    #
    # Recorded in config.json because CPU and GPU runs are NOT bit-identical: float32 reduction
    # order differs, so two runs differing only in this field track each other but do not agree
    # to the last digit.
    device: str = 'cpu'
    # Rollout steps per gradient-checkpoint segment; 0 = off (store the whole graph).
    #
    # Deliberately NOT derived from `device`. It is a function of nf, not of hardware: at
    # nf = 12000 the un-checkpointed BPTT graph is ~25 GB at batch 512 and does not fit on CPU
    # either. Deriving it from the device would make the CPU run silently attempt the 25 GB.
    #
    # EXACT, not an approximation: the segment boundaries are NOT detached, so the gradient
    # still spans all nf steps (contrast truncated BPTT/ARTBP, which detaches and changes the
    # objective). Cost is one extra forward per segment, about +33% compute, for roughly
    # nf/chunk times less activation memory.
    checkpoint_chunk: int = 0
    # torch.compile mode for the TRAINING rollout. None = eager (the default; nothing compiles).
    #
    # MEASURED on blade1 (RTX 2080 Ti), batch 512, chunk 0, nf 200 -- jobs 80610/80634/80652:
    #   eager                     14.6 - 15.1 ms/step
    #   inductor                   5.4 -  6.0 ms   ~2.6x   (fusion only)
    #   inductor reduce-overhead   2.1 -  2.4 ms   ~6.5x   (fusion + CUDA graphs)  <- use this
    #   inductor max-autotune      2.2 -  2.3 ms   ~6.8x   but WORSE in float64 (1.09x vs 1.04x)
    #                                                      and ~10x slower to compile
    # At nf = 12000 that is ~1300 +/- 100 batch updates in a 10 h wall, against the nf = 400
    # pipeline's 1300, i.e. the same training budget at a 30x longer horizon.
    #
    # NOT A FEATURE FLAG. It is a recorded run parameter like `device`: a compiled run and an
    # eager one are not bit-identical (max|dg| = 7.5e-10, job 80610), so config.json has to say
    # which ran. There is no `if compiled:` branch in any hot path; the simulator holds a
    # compiled callable or it does not.
    #
    # Only the TRAINING rollout is affected. `fit_sys.hfn` is never replaced, so the validation
    # free run and every diagnostic keep using the eager module -- which matters because deepSI's
    # fit() moves the model to the CPU around each validation (interconnect.py:716,734) and a
    # compiled callable would recompile per device, twice an epoch.
    compile_mode: Optional[str] = None

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
    # Hard cap on BATCH UPDATES, overriding epochs. None = epochs decide (fit computes
    # n_its = N_batch_updates_per_epoch * epochs, interconnect.py:778). Set an int for a
    # smoke test: a float64 A/B needs tens of updates to show whether the loss moves, not
    # 5 epochs at 260 updates each. Runtime-only, deliberately NOT in `hp`, so it never
    # enters the checkpoint hyperparameter schema and a capped run stays resumable as a
    # normal one.
    n_its: Optional[int] = None
    # Batch updates between validations, handed straight to fit(). None (the default) means
    # 'epoch', i.e. the exact behaviour this pipeline has always had. An int is a number of
    # UPDATES, deepSI's own unit for this argument, so no epoch-length arithmetic is duplicated
    # here and there is nothing to drift out of sync. At batch 512 an epoch is 130 updates, so
    # 650 validates every 5 epochs. NOTE the meaning therefore moves with batch_size.
    # WHY it is worth exposing: one validation is a closed-loop free run over V1-V4 (4 x 48000
    # samples, batch one, eager) costing ~162 s, against 65 s of training per epoch on the GPU
    # (80713). Validating every epoch spends 70% of the wall on validation. That was invisible on
    # the CPU, where an epoch cost ~1300 s and the same validation was 11% of it. n_its still
    # takes precedence in model.py, so a capped smoke test keeps validating at its cap.
    its_per_val: Optional[Union[int, str]] = None
    nf_seconds: float = 0.100      # [s] SEGMENT length (5*tau_msd, tau=1/(zeta*wn)=20ms, 5tau=100ms)
    # Optional direct overrides (None = derive). Set a number to bypass the formula.
    nf_override: Optional[int] = None      # None -> nf = nf_seconds / ts_new
    na_nb_override: Optional[int] = None   # None -> na_nb = (nx_phys + nx_ann)*2 + 1 (Jan's rule)

    # ═══ Multiple shooting: REMOVED 2026-08-28 (D-127 retired) ════════════════
    # `n_seg`, `defect_weight`, `defect_acc_weight`, `defect_norm` and `defect_scale` are gone
    # with SSE_Interconnect_MultipleShooting, which this pipeline no longer builds. They were
    # never anything but their no-op defaults in a production run (n_seg=1, both weights 0), and
    # the closed loop made the feature unreachable anyway: MultipleShooting.loss raises on an
    # attached simulator, because whether a driven rollout resets the driver's own state at a
    # segment boundary is an open modelling question.
    # They are REMOVED rather than left at their defaults for the reason `orth_observe` was: a
    # field that cannot change the run still gets recorded in config.json, where it reads as a
    # setting the run honoured. The class itself is untouched in
    # model_augmentation/fit_systems/multiple_shooting.py and still carries its own class-level
    # defaults, so the ~20 diagnostics that construct it directly keep working.
    # Consequence: `nf` is no longer `n_seg * nf_seg`, so the two properties are now one.

    # ═══ Orthogonal-projection regularization (docs/orthogonal-projection-plan.md D7) ═══
    # CHANGED (2026-08-28): `orth` is THE switch and `orth_beta` is the strength. Previously
    # beta doubled as both, so "is orth on" was a magnitude to interpret rather than a flag to
    # read, and `orth_observe` sat next to it looking like the enable it never was. A boolean
    # cannot be misread. The contradictory state (switch on, strength zero) is rejected in
    # __post_init__ rather than silently running as off, so the config can never claim an
    # objective the run does not optimize.
    # `orth_observe` (attach the basis WITHOUT penalizing, so the [joint-probe] orth-frac meter
    # could watch a free ANN) is REMOVED: it was a third state whose only purpose was
    # measurement, and it cost the full basis build to get. Reinstate it as a diagnostic if the
    # inert-penalty measurement is needed again.
    orth: bool = False              # is the penalty in the loss? False = no object, no basis build
    orth_beta: float = 4.66e-4      # STRENGTH only, must be > 0. Ignored when orth=False.
                                    # beta_center = V_MSE/E_drift (D7.9, measured 07-12)
    orth_point_stride: int = 100    # penalty point-set decimation (Step 5 coverage verified at 100)
    orth_rank_tol: float = 1e-12    # numerical-rank truncation of Q (plan Sect. 2.2 step 5)

    # ═══ Fixed model dimensions ═══════════════════════════════════════════════
    nx_phys: int = 6   # physical states: q1, q2, q3, dq1, dq2, dq3
    nu: int = 3
    ny: int = 3

    # ───────────────────────── Validation ────────────────────────────────────
    def __post_init__(self):
        """Reject configs whose flags and behaviour would disagree.

        Runs on construction AND on dataclasses.replace, so a config derived from this one
        cannot reach a contradictory state either. Assigns nothing, so frozen=True holds.
        """
        if self.orth and self.orth_beta <= 0:
            raise ValueError(
                'orth=True with orth_beta=%r is not a state. `orth` is the switch and '
                '`orth_beta` is the strength: set orth=False to disable the penalty, or give '
                'orth_beta a positive value. (Before 2026-08-28 beta doubled as the switch, so '
                'orth_beta=0.0 meant off; that spelling is gone precisely because it let a '
                'config look enabled while running as off.)' % (self.orth_beta,))
        # D-169. Rejected here rather than at first use: a typo ('gpu', 'CUDA', 'cuda:0') would
        # otherwise surface deep in init_model, or worse run on the CPU while the log says
        # otherwise. Availability is NOT checked here, only spelling: a RunConfig is constructed
        # for inspection on machines without a GPU, and the real check belongs where the model
        # is built (model.py).
        if self.device not in ('cpu', 'cuda'):
            raise ValueError(
                "device=%r is not accepted. Use 'cpu' or 'cuda'. Per-card selection ('cuda:1') "
                "is deliberately not supported: on the cluster SLURM sets CUDA_VISIBLE_DEVICES "
                "so the allocated card IS 'cuda', and honouring an index here would also "
                "require changing deepSI's fit(), whose batch move is a bare .cuda()."
                % (self.device,))
        if self.checkpoint_chunk < 0:
            raise ValueError(
                'checkpoint_chunk=%r is not a length. Use 0 to store the whole rollout graph, '
                'or a positive number of steps per checkpointed segment.'
                % (self.checkpoint_chunk,))
        if self.compile_mode not in (None, 'default', 'reduce-overhead', 'max-autotune'):
            raise ValueError(
                "compile_mode=%r is not a torch.compile mode. Use None (eager), or one of "
                "'default', 'reduce-overhead', 'max-autotune'. Measured best on this pipeline: "
                "'reduce-overhead' (~6.5x, jobs 80610/80634/80652)." % (self.compile_mode,))
        if self.compile_mode is not None and self.device != 'cuda':
            raise ValueError(
                "compile_mode=%r with device=%r. Compilation is only wired for CUDA: the "
                "measured backend is inductor+CUDA graphs, and on CPU it would compile a "
                "different (C++/OpenMP) path that has never been benchmarked here. Set "
                "device='cuda' or compile_mode=None." % (self.compile_mode, self.device))
        # Caught here rather than in fit(): deepSI compares its_per_val against an update counter,
        # so a stray string ('epochs', '5') would never match and validation would simply never
        # run. The failure is a run with no checkpoint selection at all, which looks like a
        # training result until you notice bestfit never moved.
        if not (self.its_per_val is None or self.its_per_val == 'epoch'
                or (isinstance(self.its_per_val, int) and not isinstance(self.its_per_val, bool)
                    and self.its_per_val >= 1)):
            raise ValueError(
                "its_per_val=%r is neither a cadence nor None. Use None (= 'epoch', the default), "
                "the string 'epoch', or a positive int number of BATCH UPDATES between "
                "validations. At batch_size=%r one epoch is N_training_samples // batch_size "
                "updates, so the int moves with the batch size."
                % (self.its_per_val, self.batch_size))

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
    def nf(self) -> int:
        """Steps per training sample; this is also the gradient-path length.

        CHANGED (2026-08-28): was `n_seg * nf_seg`, two properties, when multiple shooting could
        split a sample into segments. With n_seg gone the two collapse into this one. Verified
        numerically unchanged for the production config (400 samples = 0.100 s at 4 kHz), which
        matters because hp['nf'] is the training horizon and a silent change here would move
        every run while looking like a refactor.
        """
        if self.nf_override is not None:
            return self.nf_override
        return max(1, int(self.nf_seconds / self.ts_new))

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


@functools.lru_cache(maxsize=1)
def git_provenance() -> str:
    """Which code produced this run: '0bded3d Augmentation dirty:302 diff:3b437e0c'.

    The config says what was asked for; this says what was running. Without it a log six
    months old cannot be tied to a version of the code.

    The DIFF HASH is the part that earns its keep. A bare 'dirty' flag is the same string on
    every run of a working tree that stays dirty for a month, i.e. no information. Hashing
    `git diff HEAD` makes two runs from one commit with different uncommitted work
    distinguishable: same SHA and same diff hash means the tracked code was identical.
    Untracked files do NOT affect it; hashing the whole worktree would, and is not worth the
    runtime.

    Cached because the code cannot change mid-run, so the four subprocesses (one of them
    `git diff` over the whole tree) are paid once however often this is called. NOT called at
    import: `config_json_dict` takes the string as an argument so importing this module never
    shells out.
    """
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
        # A run outside a checkout (cluster tarball, extracted archive) must still run.
        return 'unavailable (%s)' % type(e).__name__


# RunConfig fields deliberately NOT in config.json, each with the reason it is safe to omit.
# Everything else must appear, enforced at import by _assert_json_coverage below.
_JSON_EXEMPT = (
    'nx_phys', 'nu', 'ny',      # fixed model dimensions; cannot vary between runs
    'fs_new',                   # recorded RESOLVED as FS_NEW (None means fs_orig)
    'nf_override',              # effect is visible in the recorded NF
    'na_nb_override',           # effect is visible in the recorded NA_NB
)


def config_json_dict(cfg: RunConfig, git: Optional[str] = None) -> dict:
    """Config metadata for the results npz -- exact keys and order of the pre-refactor dump.

    PURE function of its arguments: `git` is passed IN rather than queried here, so importing
    this module never shells out and the import-time coverage check below is free. The entry
    point passes `git=git_provenance()`.

    Keys are hand-written, not derived from field names, because they are a contract with
    readers of old npz files: deriving them would turn every future field rename into a silent
    schema change. Completeness is enforced instead by _assert_json_coverage.
    """
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
        # CHANGED (2026-08-28): ORTH_OBSERVE removed with the field. ORTH is appended below
        # rather than replacing it in place, because the key order above is load-bearing for
        # old npz readers. ORTH_BETA alone no longer says whether the penalty was applied:
        # a run with orth=False still carries a positive beta that did nothing.
        # Appended, not inserted: the pre-refactor key order above is load-bearing
        # for old npz readers. CLOSED_LOOP records which objective the run used --
        # without it an open-loop and a closed-loop run are indistinguishable in the
        # saved metadata, and their `bestfit` numbers are not comparable.
        CLOSED_LOOP=cfg.closed_loop,
        # Appended for backward compatibility with readers that rely on the old order.
        ADAM_EPS=cfg.adam_eps,
        ORTH=cfg.orth,
        # Added 2026-08-28. These all change what a run IS and none of them was recorded,
        # so runs that differed in any of them were indistinguishable in the saved metadata:
        #   ANN_ROUTE_IX  which state rows the ANN writes (D-103: X and Y, never Theta-only)
        #   N_ITS         batch-update cap; a capped smoke run otherwise reads as a full one
        #   STRIDE        window decimation, i.e. how many training samples exist
        #   STATE_LAYOUT  the model/training hyperparameters not already in `hp`
        ANN_ROUTE_IX=list(cfg.ann_route_ix),
        STRIDE=cfg.stride,
        N_ITS=cfg.n_its,
        ITS_PER_VAL=cfg.its_per_val,
        NF_SECONDS=cfg.nf_seconds,
        USE_F64=cfg.use_f64,
        SAVE_FLAG=cfg.save_flag,
        NF_PROBE_PRINT=cfg.nf_probe_print,
        NX_ANN=cfg.nx_ann,
        N_NODES_PER_LAYER=cfg.n_nodes_per_layer,
        N_HIDDEN_LAYERS=cfg.n_hidden_layers,
        UP_SAMPLE=cfg.up_sample,
        BATCH_SIZE=cfg.batch_size,
        LR=cfg.lr,
        EPOCHS=cfg.epochs,
        NF=cfg.nf,
        NA_NB=cfg.na_nb,
        # Appended 2026-08-31 (D-169). DEVICE is recorded because CPU and GPU runs are not
        # bit-identical (float32 reduction order), so it is part of what a run IS, not a
        # scheduling detail. CHECKPOINT_CHUNK does not change the gradient, but it does change
        # peak memory and runtime, and a run that OOMed at 0 versus one that survived at 200 is
        # otherwise indistinguishable in the saved metadata.
        DEVICE=cfg.device,
        CHECKPOINT_CHUNK=cfg.checkpoint_chunk,
        # A compiled run is ~6.5x faster and NOT bit-identical to eager (max|dg| = 7.5e-10,
        # job 80610). Two runs differing only here are the same experiment numerically but not
        # byte-for-byte, so the metadata has to record which one ran.
        COMPILE_MODE=cfg.compile_mode,
        # N_SEG / DEFECT_* removed 2026-08-28 with the fields themselves (D-127 retired).
        # Old npz files still carry these keys; new ones do not, which is the honest record:
        # the run had no such knob to honour.
        # Which CODE ran, as opposed to what it was asked to do. None when the caller did not
        # look it up; the entry point passes git_provenance().
        GIT=git,
    )


def _assert_json_coverage():
    """Every RunConfig field is recorded in config.json, or exempted with a reason.

    Runs at import, so adding a field to RunConfig without deciding how it is logged fails
    immediately and names the field, instead of producing runs whose metadata quietly omits
    it. This is why config_json_dict can stay hand-written: the keys remain a stable contract
    while completeness is still guaranteed.

    Free: config_json_dict is pure and is called here without `git`, so no subprocess runs.
    """
    recorded = {k.lower() for k in config_json_dict(RunConfig())}
    missing = {f.name for f in fields(RunConfig)} - recorded - set(_JSON_EXEMPT)
    if missing:
        raise ImportError(
            'RunConfig fields missing from config.json: %s. Add each to config_json_dict, '
            'or to _JSON_EXEMPT with the reason it is safe to omit.' % sorted(missing))


_assert_json_coverage()
