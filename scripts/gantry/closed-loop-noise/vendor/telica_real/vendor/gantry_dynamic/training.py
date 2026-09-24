"""Training orchestration with per-run diagnostics and checkpoint I/O.

The `.pt` component-state_dict format and the D-076 pre-JE resume guard are unchanged from
pre-refactor. The `.npz` meta is APPEND-ONLY: every pre-existing key keeps its name, dtype and
position, and D-171 added `lbfgs_outcome`, `optimizer_state_dropped`, `start_phase`,
`resumed_from` and `resumed_at_epoch` after them, so old readers are unaffected. The `.pt`
gained one conditional: the `optimizer` key is omitted when the Adam state would not match the
weights beside it (see `save_checkpoint_weights`), which both loading routes already tolerate.
"""
__project_origin__ = "added"

import os
import json
import time

import numpy as np
import torch

from .config import RunConfig
from .model import train_model
from .diagnostics import aug_state_r2
from model_augmentation.fit_systems.interconnect import WindowErrorStats
from model_augmentation.fit_systems.lbfgs_polish import PolishSpec, lbfgs_polish


def _json_safe(obj):
    """numpy scalars out of the outcome dict, so json.dumps cannot fail on the meta write."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_checkpoint_weights(fit_sys, base_path, include_optimizer=True):
    """Save the torch components (SSE_Interconnect is a deepSI System, no state_dict). D-070.

    `include_optimizer=False` omits the Adam state. Two cases call for it (D-171):

    * an ACCEPTED polish. The phase never touches `fit_sys.optimizer`, so writing it next to
      polished weights would pair `exp_avg` and `exp_avg_sq` describing the PRE-polish point with
      parameters that have since moved. Nothing in this pipeline reads them today (a resume into
      the polish ignores the optimizer state by design), but a later `start_phase='adam'` resume
      would silently restart Adam with moments belonging to a point it is no longer at.
    * a `start_phase='lbfgs'` run, where no Adam phase ran at all and the optimizer is the one
      `init_model` built, with `state` still empty. An empty state is inert but reads as a real
      one; the state that matches those weights lives in the checkpoint they came from.

    Omitting is safe on both loading routes: `resolve_checkpoint` uses `ckpt.get('optimizer')`
    and `load_checkpoint` skips a None. A resumed Adam then starts with a fresh optimizer at the
    polished point, which is the honest state, and matches the decision not to persist the
    L-BFGS state either.
    """
    payload = {
        'hfn':     fit_sys.hfn.state_dict(),
        'encoder': fit_sys.encoder.state_dict(),
    }
    if include_optimizer:
        payload['optimizer'] = fit_sys.optimizer.state_dict()
    torch.save(payload, base_path + '.pt')


def resolve_checkpoint(base_path):
    """Normalise EITHER checkpoint format to (hfn_sd, encoder_sd, optim_sd_or_None, meta).

    Two formats exist in this project and both are legitimate resume sources (D-171):

    | on disk                              | written by                                    |
    |--------------------------------------|-----------------------------------------------|
    | `<base>.pt` + `<base>.npz`           | `save_checkpoint_weights` below, per run      |
    | `<...>_best.pth`                     | deepSI's `checkpoint_save_system`, per        |
    |                                      | validation, from inside `fit()`               |

    The second is `torch.save(self.__dict__)` of the whole system, so its `hfn` and `encoder`
    entries are live MODULES rather than state dicts. It is the only format available for a run
    that was killed mid-training or whose result folder was not copied back, which is exactly
    when a resume-into-polish is wanted.

    `map_location='cpu'` on both: with D-169's validation flip suppressed, deepSI writes CUDA
    tensors into `_best.pth`, and those must still load on a CPU-only machine.
    """
    pair_pt, pair_npz = base_path + '.pt', base_path + '.npz'
    if os.path.exists(pair_pt) and os.path.exists(pair_npz):
        ckpt = torch.load(pair_pt, map_location='cpu')
        meta = np.load(pair_npz, allow_pickle=True)
        return ckpt['hfn'], ckpt['encoder'], ckpt.get('optimizer'), meta, 'pair'

    # Three spellings of the same file, all natural, and job 81500 died because only one was
    # accepted. deepSI writes `<class>_<code><suffix>.pth` with suffix in {_best, _last}, so the
    # "base" a user copies off disk may be the full path, the path minus `.pth`, or the deepSI
    # name without the suffix. Appending `_best.pth` unconditionally turned
    # `..._p2LPDA_best` into `..._p2LPDA_best_best.pth`.
    # `_last` is NOT tried automatically: silently falling back to the last iterate when the best
    # is missing would swap the experiment without saying so. Name it explicitly to use it.
    candidates = [base_path] if base_path.endswith('.pth') else [
        base_path + '.pth',            # base is the file minus its extension
        base_path + '_best.pth',       # base is the deepSI name, suffix omitted
    ]
    whole = next((c for c in candidates if os.path.exists(c)), None)
    if whole is not None:
        d = torch.load(whole, map_location='cpu', weights_only=False)
        meta = {
            'done_epochs': np.array(int(round(float(d.get('epoch_counter', 0))))),
            'bestfit': np.array(float(d.get('bestfit', float('inf')))),
            'diag_epochs': np.array([]), 'diag_r2_raw': np.array([]),
            'diag_r2_linmap': np.array([]),
        }
        # The pickled system carries a LIVE Adam object, so its moments are recoverable. An
        # earlier version returned None here and the docstring claimed this format "has no state
        # dict at all", which is simply wrong: `d['optimizer']` is an optimizer, and asking it
        # for a state dict is all that was ever needed. The consequence of that mistake was that
        # resuming Adam from a `_best.pth` silently restarted with zeroed moments (D-173).
        _opt = d.get('optimizer')
        _opt_sd = _opt.state_dict() if _opt is not None and hasattr(_opt, 'state_dict') else None
        return d['hfn'].state_dict(), d['encoder'].state_dict(), _opt_sd, meta, 'whole'

    raise FileNotFoundError(
        'no checkpoint at %r. Tried the pair %r + %r, and the deepSI whole-system dump(s): %s'
        % (base_path, pair_pt, pair_npz, ', '.join(repr(c) for c in candidates)))


def load_optimizer_moments(optimizer, saved):
    """Restore an optimizer's MOMENT ESTIMATES while keeping this run's hyperparameters (D-173).

    A torch optimizer state dict stores two unrelated things together:

      `state`        the per-parameter moments (`step`, `exp_avg`, `exp_avg_sq`). This is what
                     "continue Adam" means, and it is what a resume wants.
      `param_groups` the hyperparameters (`lr`, `betas`, `eps`, `weight_decay`) alongside the
                     parameter index lists that map `state` onto real parameters.

    `Optimizer.load_state_dict` restores BOTH, so a plain resume silently adopts the
    checkpoint's learning rate and epsilon and ignores `cfg.lr` / `cfg.adam_eps`. Two runs
    launched from the same config then train at different rates, and `config.json` records the
    one that did not happen. That has been true of every resume this project has ever done, not
    only of the L-BFGS work.

    This takes the settings from the LIVE optimizer and the mapping plus moments from the saved
    one. It still goes through `load_state_dict`, deliberately: that is what resolves saved
    indices onto live `Parameter` objects and casts the moment tensors onto each parameter's
    device and dtype. Assigning `optimizer.state` directly would have to reimplement both.
    """
    live = optimizer.state_dict()
    if len(live['param_groups']) != len(saved['param_groups']):
        raise RuntimeError(
            'optimizer parameter groups do not match: this run has %d, the checkpoint has %d. '
            'The moments cannot be mapped onto these parameters.'
            % (len(live['param_groups']), len(saved['param_groups'])))
    merged = {**saved, 'param_groups': [
        {**live_g, 'params': saved_g['params']}
        for live_g, saved_g in zip(live['param_groups'], saved['param_groups'])
    ]}
    optimizer.load_state_dict(merged)


def _assert_same_architecture(target_sd, ckpt_sd, what):
    """Raise NAMING the mismatch, instead of letting load_state_dict raise a wall of keys.

    Resuming a checkpoint into a config with a different `nx_ann`, width or depth is the mistake
    a resume-into-polish workflow invites (the meeting fixture alone holds an nx_ann=8 and an
    nx_ann=2 run), and torch's own message is a page long and does not say which knob differs.
    """
    missing = sorted(set(target_sd) - set(ckpt_sd))
    extra = sorted(set(ckpt_sd) - set(target_sd))
    shape = [(k, tuple(target_sd[k].shape), tuple(ckpt_sd[k].shape))
             for k in sorted(set(target_sd) & set(ckpt_sd))
             if tuple(target_sd[k].shape) != tuple(ckpt_sd[k].shape)]
    if not (missing or extra or shape):
        return
    parts = []
    if shape:
        parts.append('%d shape mismatch(es), first 3: %s'
                     % (len(shape), '; '.join('%s config%s != ckpt%s' % s for s in shape[:3])))
    if missing:
        parts.append('%d key(s) the config expects and the checkpoint lacks, first 3: %s'
                     % (len(missing), missing[:3]))
    if extra:
        parts.append('%d key(s) the checkpoint has and the config does not, first 3: %s'
                     % (len(extra), extra[:3]))
    raise RuntimeError(
        'checkpoint does not match this config for %s: %s. The usual cause is a different '
        'nx_ann / n_nodes_per_layer / n_hidden_layers than the run that wrote it.'
        % (what, '; '.join(parts)))


def infer_arch_from_hfn(hfn_sd, nx_phys=6, nu=3):
    """Read (nx_ann, n_nodes_per_layer, n_hidden_layers) OFF a checkpoint's ANN block.

    A checkpoint fixes the architecture that produced it. Anything that loads one therefore has
    to take the architecture FROM it, not from whatever `CFG` happens to be checked out: job
    81461 failed exactly that way, building a 16x2 nx_ann=2 model from the cluster's entry file
    and then trying to load the 24x3 nx_ann=8 weights of job 81262 into it.

    The ANN is `Static_ANN_Block.net.net`, an `nn.Sequential` of Linear/activation pairs, so the
    Linear weights sit at even indices and carry everything needed:
        first weight shape (n_nodes, n_in)  ->  n_nodes, and n_in = nx_phys + nx_ann + nu
        number of Linear layers             ->  n_hidden_layers + 1 (the output layer)

    Verified on both meeting checkpoints: p2LPDA gives (24, 17) over four Linears, i.e.
    nx_ann = 17 - 6 - 3 = 8 with 3 hidden layers; tINZQQ/hcfOLA give (16, 11) over three, i.e.
    nx_ann = 2 with 2 hidden layers.
    """
    import re
    pat = re.compile(r'\.net\.net\.(\d+)\.weight$')
    layers = {}
    for k, v in hfn_sd.items():
        m = pat.search(k)
        if m and getattr(v, 'ndim', 0) == 2:
            layers[int(m.group(1))] = tuple(v.shape)
    if not layers:
        raise RuntimeError(
            'no ANN Linear weights (`*.net.net.<i>.weight`) in this checkpoint, so its '
            'architecture cannot be inferred. Keys seen: %s' % (sorted(hfn_sd)[:6],))
    idx = sorted(layers)
    n_nodes, n_in = layers[idx[0]]
    nx_ann = int(n_in) - nx_phys - nu
    if nx_ann < 0:
        raise RuntimeError(
            'inferred nx_ann=%d from an ANN input width of %d with nx_phys=%d, nu=%d; the '
            'checkpoint does not match this model family.' % (nx_ann, n_in, nx_phys, nu))
    return nx_ann, int(n_nodes), len(idx) - 1


def config_for_checkpoint(cfg, hfn_sd):
    """`cfg` with the architecture replaced by the checkpoint's own, and nothing else touched.

    `ann_route_ix` follows, because it must cover the interconnected state (nx_phys + nx_ann);
    `na_nb` is derived by the same rule and needs no help.
    """
    from dataclasses import replace
    nx_ann, n_nodes, n_hidden = infer_arch_from_hfn(hfn_sd, cfg.nx_phys, cfg.nu)
    return replace(cfg, nx_ann=nx_ann, n_nodes_per_layer=n_nodes, n_hidden_layers=n_hidden,
                   ann_route_ix=tuple(range(cfg.nx_phys + nx_ann)))


def _assert_same_dtype(fit_sys, hfn_sd):
    """Refuse to load a checkpoint whose dtype differs from the run's.

    PRECISION MUST STAY COHERENT ACROSS THE PHASES AND ACROSS A RESUME. Silently casting a
    float32 checkpoint into a float64 model works (load_state_dict converts) and produces a run
    whose two halves are not comparable: the loss series changes scale below the 7th digit, the
    stored `bestfit` was computed in the other precision, and the polish's accept/reject decision
    is then made across a dtype change rather than across the optimiser.

    Measured spread for such a change on this project: sim-RMS 5.774376e-06 against 5.774347e-06,
    i.e. 0.005%. Small next to the ~0.3% acceptance bar, and still a systematic bias in exactly
    the comparison the phase exists to make.

    float64 is essentially free on this workload (job 81463: ~4.1 s per closure in both dtypes,
    2x memory), so the right way to get a float64 polish is a float64 TRAINING run, not a cast at
    the resume boundary.
    """
    want = next(fit_sys.hfn.parameters()).dtype
    got = next((v.dtype for v in hfn_sd.values()
                if torch.is_tensor(v) and v.is_floating_point()), None)
    if got is not None and got != want:
        raise RuntimeError(
            'checkpoint dtype %s does not match this run\'s %s. Precision must stay coherent: '
            'set use_f64=%s to match the checkpoint, or retrain at %s from scratch. Casting here '
            'would make the pre- and post-phase numbers incomparable, which is the one comparison '
            'the polish phase exists to make.'
            % (got, want, got == torch.float64, want))


def load_checkpoint(fit_sys, base_path, joint_estimation, start_phase='adam'):
    """Restore weights (+ Adam state on an 'adam' start) from either format; return the meta."""
    hfn_sd, enc_sd, optim_sd, meta, fmt = resolve_checkpoint(base_path)
    _assert_same_dtype(fit_sys, hfn_sd)
    # CHANGED (D-191): raw JE checkpoints carry log_params; reduced JE checkpoints carry
    # free_params. The exact architecture check below rejects cross-parameterization resumes.
    if joint_estimation and not any(
            k.endswith(('log_params', 'free_params')) for k in hfn_sd):
        raise RuntimeError(
            'RESUME_CHECKPOINT points at a pre-JE checkpoint (no physical parameter vector); '
            'JOINT_ESTIMATION runs must start from a matching JE checkpoint')
    _assert_same_architecture(fit_sys.hfn.state_dict(), hfn_sd, 'hfn')
    _assert_same_architecture(fit_sys.encoder.state_dict(), enc_sd, 'encoder')
    fit_sys.hfn.load_state_dict(hfn_sd)
    fit_sys.encoder.load_state_dict(enc_sd)
    # On a 'lbfgs' start Adam never runs, so its moments are irrelevant and are not restored.
    # On an 'adam' start they ARE restored, moments only: `load_optimizer_moments` keeps this
    # run's lr/eps/betas rather than adopting the checkpoint's (D-173).
    if optim_sd is not None and start_phase == 'adam':
        load_optimizer_moments(fit_sys.optimizer, optim_sd)
        print("  Adam moment estimates restored; lr/eps/betas come from this run's config.")
    print(f'  Checkpoint format: {fmt}')
    # STATE the fresh-optimizer case instead of proceeding silently. A checkpoint written after an
    # accepted polish carries no Adam moments on purpose (save_checkpoint_weights), and resuming
    # Adam from it is legitimate but is NOT a continuation of the previous Adam trajectory:
    # `exp_avg_sq` sets the per-parameter step size and starts from zero here, so the first
    # updates are sized by the bias correction rather than by history. Whoever wanted the exact
    # continuation wants the pre-polish file instead, and would otherwise never know it existed.
    if start_phase == 'adam' and optim_sd is None:
        _dropped = bool(meta['optimizer_state_dropped']) \
            if 'optimizer_state_dropped' in getattr(meta, 'files', []) else None
        print('  NOTE: this checkpoint carries NO Adam optimizer state'
              + (' (recorded as deliberate: optimizer_state_dropped=True)' if _dropped
                 else ' (whole-system format: deepSI stores a live optimizer, not a state dict)'
                 if fmt == 'whole' else '')
              + '.\n        Adam will start from fresh moment estimates at these weights. That is'
                ' correct for\n        polished weights, but it is NOT a continuation of the'
                ' earlier Adam trajectory.\n        For that, resume'
                ' `gantry_ckpt_<run_id>_pre_lbfgs`, which keeps the matching state.')
    return meta


class _NfProbe:
    """Per-epoch nf-window error statistics, train and val, recorded alongside validation (D-095).

    CHANGED (2026-08-28): both numbers came from deepSI's `System.n_step_error`, the last deepSI
    call on the training path, and both were wrong for the comparison they invited:
      - it drives the model through `measure_act_multi`, i.e. OPEN LOOP always, never
        `self.simulate`. In a closed-loop run it measured a different system than the one being
        optimised, printed next to a closed-loop `bestfit`, unlabelled;
      - the train side was ONE trajectory, `train_list[0]` = T1_standstill_Ym30, the least
        excited record, standing in for 14;
      - it returned `np.mean` over the n = 1..nf growth curve, not the RMS over an nf window;
      - and each side cost a full extra pass per epoch.

    Now: the TRAIN statistics come from `fit_sys.loss_stats`, accumulated inside loss() over
    every training window of all 14 trajectories, through whatever rollout the run uses, at no
    extra compute. The VAL statistics come from ONE fixed 256-window batch through
    `fit_sys.simulate`, so both sides are in the run's own loop mode by construction.

    That val number is worth its ~1/260-of-an-epoch because it DECOMPOSES the reading:
    train-nf vs val-nf is generalisation at fixed horizon, val-nf vs val-sim is horizon at fixed
    generalisation, and train-nf vs val-sim (the only comparison available before) is both at once.

    The train row is a RUNNING MEAN over weights that moved during the epoch, not a snapshot. For
    the mismatch reading the bias is conservative: it reads slightly high, which makes the gap
    look smaller, so it cannot manufacture a mismatch that is not there.

    # CHANGED (closed-loop seam): this was a MONKEY PATCH on `fit_sys.cal_validation_error`,
    # installed on every production run, wrapping the original and returning its value untouched.
    # It is now an entry in `fit_sys.validation_probes`, called for its side effects only.
    # Three things go away with the patch:
    #   - the ordering hazard. Two different patches wanted the same attribute (this one and the
    #     closed-loop validator), and whichever was installed LAST decided checkpoint selection.
    #     A probe cannot decide selection any more: the seam ignores its return value.
    #   - the `__reduce__` no-op. A patched bound method cannot be pickled, and
    #     `checkpoint_save_system` does `torch.save(self.__dict__)` at every validation, so the
    #     class had to serialise itself back into a placeholder. A declared attribute needs none
    #     of that.
    #   - the try/finally restore, and the silent failure when a checkpoint reload replaced
    #     `__dict__` and dropped the patched method without anyone noticing.
    """

    N_VAL_WINDOWS = 256      # a tight RMS estimate; all windows costs ~7x for no more precision

    def __init__(self, fit_sys, nf, cfg, val_sd, do_print=True):
        self.fit_sys = fit_sys
        self.nf = nf
        self.do_print = do_print
        fit_sys.Loss_train_nf = []
        fit_sys.Loss_val_nf = []

        # Train side: statistics accumulate inside loss() over every training window, so there is
        # nothing to build here and no extra rollout. Attaching the object IS the switch.
        fit_sys.loss_stats = WindowErrorStats(fit_sys.norm.ystd, dtype=cfg.dtype_pt)

        # Val side: ONE fixed batch, drawn once. Fixed and not resampled per epoch, because a
        # changing window set puts selection noise on top of the learning signal and the two
        # cannot be separated afterwards.
        self._val_stats_obj = WindowErrorStats(fit_sys.norm.ystd, dtype=cfg.dtype_pt)
        self._val_batch, self._val_kwargs = None, {}
        # CHANGED (2026-08-31, run 80468): `make_training_data` routes through the attached
        # simulator's `augment_training_data`, and `WindowControllerIndex` holds the TRAINING
        # record rows. Handing it the VALIDATION records tripped its own guard ("the simulator
        # was given 14 per-record controller rows but the training data has 4 records") and the
        # whole val probe was silently lost on every closed-loop run. The guard is right; it was
        # being asked the wrong question. Swap the rows for the duration of this one call.
        # `val_records` is (label, sys_data, ctrl_row) in the order `val_sd` was built, which is
        # exactly the order this call needs, and `verify_by_content` still runs inside `build`,
        # so a genuine misalignment would still raise rather than pass quietly.
        _sim = getattr(fit_sys, 'simulator', None)
        _saved_rows = None
        if _sim is not None and hasattr(_sim, 'indexer') and getattr(_sim, 'val_records', None):
            _saved_rows = _sim.indexer.record_rows
            _sim.indexer.record_rows = [int(r) for _, _, r in _sim.val_records]
        try:
            d = fit_sys.make_training_data(fit_sys.norm.transform(val_sd), nf=nf, stride=nf)
            n = min(self.N_VAL_WINDOWS, len(d[0]))
            ix = np.random.default_rng(cfg.seed).choice(len(d[0]), size=n, replace=False)
            cols = [torch.as_tensor(np.asarray(a)[ix], dtype=cfg.dtype_pt) for a in d]
            self._val_batch = tuple(cols[:4])
            # Arrays a simulator asked for travel BY NAME, the same convention fit() uses.
            names = getattr(fit_sys.simulator, 'extra_array_names', ())
            self._val_kwargs = dict(zip(names, cols[4:]))
        except Exception as e:
            print(f'    [nf val  ] window batch unavailable (non-fatal): {e}')
        finally:
            if _saved_rows is not None:
                _sim.indexer.record_rows = _saved_rows
        # Joint/orth probe state (user 07-12: live recovery + negation meters).
        # Nominal combos computed ONCE; sim-study meter -- on real data the
        # reference must become params_init (no nominal truth exists there).
        # Augmented-state magnitude probe. Two ints rather than the whole cfg: this adds no
        # coupling beyond the state layout it needs. `x_aug` is a LEARNED latent with an
        # arbitrary scale, so this is reported in the normalized frame only, which is the frame
        # the orth penalty's `x_aug = 0` slice lives in. The comparison against the true
        # absorber is the aug-state R2 diagnostic, not this.
        self._nx_phys, self._nx_ann = cfg.nx_phys, cfg.nx_ann
        self._xa_stats = None

        self._pblock = next((m for m in fit_sys.hfn.connected_blocks
                             if hasattr(m, 'identifiable_combinations')), None)
        self._combo_nom = None
        _ref = getattr(self._pblock, 'combo_ref', None) if self._pblock is not None else None
        if _ref is not None:
            # TELICA-REAL: (TR-025) real data has no nominal truth; drift is measured against the
            # start point the block carries (the G4 values), as the comment above asks for.
            self._combo_nom = dict(_ref)
            _raw0 = {n: self._pblock.params_init[i].item()
                     for i, n in enumerate(self._pblock.PARAM_NAMES)}
            self._combo_scale = {n: abs(t) for n, t in self._combo_nom.items() if abs(t) > 1e-12}
            self._combo_scale['m_diff'] = 0.5 * (_raw0['m1'] + _raw0['m2'])
        elif self._pblock is not None:
            from model_augmentation.systems import gantry_ss as _gss
            _true_raw = {n: getattr(_gss, n).item() for n in self._pblock.PARAM_NAMES}
            self._combo_nom = self._pblock._combos_from_raw(_true_raw,
                                                            self._pblock.Lb.item())
            # Per-combo error scale: |nominal|, EXCEPT m_diff.
            # HEURISTIC: m_diff = m1-m2 has near-zero nominal (-0.5 kg vs ~10 kg
            # masses); relative-to-itself error dominates the RMS meaninglessly
            # (run 70783: -418% -> combo-err 132%). Scale by the mean actuator
            # mass instead, so a 10% detune reads ~20% on m_diff, not 418%.
            self._combo_scale = {n: abs(t) for n, t in self._combo_nom.items()
                                 if abs(t) > 1e-12}
            self._combo_scale['m_diff'] = 0.5 * (_true_raw['m1'] + _true_raw['m2'])
        if self._pblock is not None or getattr(fit_sys, 'orth_penalty', None) is not None:
            fit_sys.Probe_combo_err = []
            fit_sys.Probe_orth_frac = []
            fit_sys.Probe_V_orth = []
            fit_sys.Probe_param_loss = []

    def _val_stats(self):
        """The fixed val window batch, through the SAME rollout the loss uses.

        `fit_sys.simulate` and not `closed_loop_free_run_rms`: the mode then follows the attached
        simulator exactly as the loss does, so a closed-loop run gives closed-loop numbers and an
        open-loop run open-loop ones, with no branch here and no second rollout to drift.
        """
        if self._val_batch is None:
            return None
        # CHANGED (D-169, job 80706): the batch is built once on the CPU. It used to match the
        # model by accident, because fit() flipped the model to CPU before every validation; with
        # that flip gone the model stays on its training device and the probe died with
        # "cuda:0 and cpu". Home it once, then it stays resident: the window set is fixed by
        # design and re-used at every validation, so this is one transfer for the whole run.
        # `fit_sys.parameters` is deepSI's LIST of optimizer parameters, not nn.Module.parameters();
        # ask the module that actually holds the weights.
        _dev = next(self.fit_sys.hfn.parameters()).device
        if self._val_batch[0].device != _dev:
            self._val_batch = tuple(t.to(_dev) for t in self._val_batch)
            self._val_kwargs = {k: v.to(_dev) for k, v in self._val_kwargs.items()}
        uh, yh, uf, yf = self._val_batch
        self._xa_stats = None
        try:
            with torch.no_grad():
                x = self.fit_sys.encoder(uh, yh)
                # The final state was already computed and discarded here. It is the augmented
                # state at the END of the rollout, i.e. where it is largest, having accumulated
                # from the encoder's initial value. That is the point to read off the T4
                # leakage curve, which measured alignment 0.0011/0.0107/0.0256/0.0406 at
                # ||x_aug|| = 0.01/0.1/0.3/1.0. Never logged before; severity of the
                # x_aug = 0 slice hole is unmeasured without it.
                y_pred, x_end = self.fit_sys.simulate(x, uf, yf, **self._val_kwargs)
            if self._nx_ann:
                xa = x_end[:, self._nx_phys:self._nx_phys + self._nx_ann]
                n = xa.norm(dim=1)
                self._xa_stats = (float(n.mean()), float(n.std()), float(n.max()))
            self._val_stats_obj.reset()
            self._val_stats_obj.update(y_pred, yf)
            return self._val_stats_obj.summary()
        except Exception as e:
            print(f'    [nf val  ] failed (non-fatal): {e}')
            return None

    @staticmethod
    def _fmt(tag, s):
        return ('    [nf %-5s] rms %.3e  chan %s  grow %s  bias %s  (%d win)'
                % (tag, s['rms'],
                   '/'.join('%.1e' % v for v in s['chan']),
                   ('%.2fx' % s['grow']) if np.isfinite(s['grow']) else 'n/a',
                   '/'.join('%+.1e' % v for v in s['bias']),
                   s['n_win']))

    def _joint_probe(self):
        """One line: recovery + negation meters. Cost: scalars + one batched ANN
        forward over the fixed penalty points (measured 0.03 s, plan Step 3).
        combo part requires a parameterized block (joint runs); orth part only
        an attached penalty, i.e. cfg.orth, independently of joint estimation)."""
        fs = self.fit_sys
        combo_err, worst, rels = float('nan'), None, {}
        if self._pblock is not None:
            # combo-err: RMS scaled error of the 10 identifiable combos vs nominal
            # (scale = |nominal| per combo; m_diff uses the mass scale -- __init__).
            combos = self._pblock.identifiable_combinations()
            rels = {n: (combos[n] - self._combo_nom[n]) / s
                    for n, s in self._combo_scale.items()}
            combo_err = float(np.sqrt(np.mean([r ** 2 for r in rels.values()])))
            worst = max(rels, key=lambda n: abs(rels[n]))
        # orth meters: available when a penalty object is attached (beta>0 or observe).
        pen = getattr(fs, 'orth_penalty', None)
        frac, v_orth = float('nan'), float('nan')
        if pen is not None:
            from model_augmentation.fit_systems.blocks import Static_ANN_Block
            ann = next(m for m in fs.hfn.connected_blocks
                       if isinstance(m, Static_ANN_Block))
            with torch.no_grad():
                w = ann(pen.Z_pts)
                f = w[:, pen.route_cols, 0].reshape(-1)
                f2 = float(torch.linalg.vector_norm(f) ** 2)
                q2 = float(torch.linalg.vector_norm(pen.Q.T @ f) ** 2)
            frac = q2 / f2 if f2 > 1e-30 else float('nan')   # n/a while ANN ~ 0
            v_orth = pen.beta * q2
        pl = sum(float(m.param_loss()) for m in fs.hfn.connected_blocks
                 if hasattr(m, 'param_loss'))
        fs.Probe_combo_err.append(combo_err)
        fs.Probe_orth_frac.append(frac)
        fs.Probe_V_orth.append(v_orth)
        fs.Probe_param_loss.append(pl)
        if not self.do_print:
            return
        combo_s = (f'combo-err {100*combo_err:.2f}% (worst {worst} '
                   f'{100*rels[worst]:+.1f}%)' if worst is not None
                   else 'combo-err n/a (theta frozen)')
        # CHANGED (D-201): the soft-penalty meters are printed only when a soft penalty is
        # ATTACHED. They used to be printed unconditionally, so an OBC or a plain joint run --
        # neither of which has an `orth_penalty` -- reported `orth-frac n/a (ANN~0)` and
        # `V_orth nan`, naming a cause (a zero ANN field) that had not been measured and could
        # not have been: nothing had evaluated the field. The by-construction meters below are
        # a different object and carry their own labels.
        tail = ''
        if pen is not None:
            # `nan` HERE does mean a zero field: `f` was evaluated just above and `f2` is its
            # squared norm, so the branch is the measurement, not the absence of one.
            frac_s = f'{frac:.3f}' if np.isfinite(frac) else 'n/a (ANN field norm 0)'
            tail = f' | orth-frac {frac_s} | V_orth {v_orth:.3e}'
        print(f'    [joint-probe] {combo_s}{tail} | param_loss {pl:.3e}')
        if rels:
            # Point 1 and 2 of the OBC monitoring spec: the ten current combinations and their
            # signed normalised errors, in the block's own reporting order.
            order = [n for n in getattr(self._pblock, 'COMBO_NAMES', sorted(rels)) if n in rels]
            items = [f'{n}={combos[n]:.4e} ({100*rels[n]:+.2f}%)' for n in order]
            for s in range(0, len(items), 5):
                print('      [combos] ' + ' | '.join(items[s:s + 5]))
        if hasattr(self._pblock, 'cc_values'):
            # TELICA-REAL: (TR-025) the trainable Coulomb levels, against their start values
            cc_now, cc0 = self._pblock.cc_values(), getattr(self._pblock, 'cc_ref', None)
            print('      [cc    ] ' + ' | '.join(
                f'{n}={v:.2f} N' + (f' ({100 * (v / cc0[i] - 1):+.2f}%)' if cc0 else '')
                for i, (n, v) in enumerate(cc_now.items())))
        self._obc_probe()

    def _obc_probe(self):
        """The by-construction meters for THIS validation (D-201). Reads only cached scalars.

        Everything printed here was computed by `_sync_prediction_state_for_validation`, from the
        one reference-field pass that D-198 already makes to re-solve the coefficient after the
        optimizer step. This method evaluates nothing: no ANN pass, no basis refresh, no
        float32 rollout-path orthogonality test, no tensor with a graph attached.

        It deliberately does NOT read `fit_sys.obc`. Under `concurrent_val` deepSI validates a
        deepcopy of the system inside a worker process, and `__getstate__` strips the lifecycle
        from that copy; the cached dict travels with the weights it describes instead.
        """
        fs = self.fit_sys
        d = getattr(fs, 'obc_validation_diagnostics', None)
        if d is None:
            return                                   # no OBC on this run: print nothing
        # The state these numbers describe must be the state being validated. In the concurrent
        # path both arrive inside the same deepcopy, so a mismatch means the sync and the probe
        # saw different models and the numbers must NOT be printed beside this validation.
        stamp, now = d.get('batch_counter', -1), int(getattr(fs, 'batch_counter', -1))
        if stamp != now:
            print('      [obc] STALE: diagnostics were synchronised at batch %s but this '
                  'validation is at batch %s; not reporting them.' % (stamp, now))
            return
        alpha_s = '' if d.get('alpha') is None else ' | alpha %+.4e' % d['alpha']
        print('      [obc] space %s | basis rank %d/%d (epoch %s%s) | overlap rho %.4f | '
              'r_perp %.3e'
              % (d['space'], d['rank'], d['n_cols'], d['basis_epoch'],
                 ', rebuilt here' if d.get('basis_rebuilt_here') else '',
                 d['rho_overlap'], d['r_perp']))
        print('      [obc] ||theta_aux|| %.4e (%d coefficients)%s | ||F_ann|| %.4e'
              % (d['theta_norm'], len(d['theta']), alpha_s, d['field_norm']))

    def __call__(self, fit_sys, val_sys_data, value):
        """validation_probes entry: side effects only, `value` is read-only here."""
        # Train: read and RESET, so each line covers the windows trained since the last
        # validation rather than the whole run.
        tr_s = fit_sys.loss_stats.summary() if fit_sys.loss_stats is not None else None
        if fit_sys.loss_stats is not None:
            fit_sys.loss_stats.reset()
        va_s = self._val_stats()
        self.fit_sys.Loss_train_nf.append(tr_s['rms'] if tr_s else float('nan'))
        self.fit_sys.Loss_val_nf.append(va_s['rms'] if va_s else float('nan'))
        if self.do_print:
            if tr_s:
                print(self._fmt('train', tr_s))
            if va_s:
                print(self._fmt('val', va_s))
            # The comparison, stated rather than left as arithmetic. `value` is the 12 s free-run
            # sim-RMS that selects the checkpoint, so both ratios are against the same scalar and
            # each is labelled with what it confounds.
            if np.isfinite(value):
                parts = []
                if va_s and va_s['rms'] > 0:
                    parts.append('%.2fx val_nf (horizon)' % (value / va_s['rms']))
                if tr_s and tr_s['rms'] > 0:
                    parts.append('%.2fx train_nf (horizon+gen)' % (value / tr_s['rms']))
                if parts:
                    print('    [nf gap  ] val_sim %.3e = %s' % (value, ' = '.join(parts)))
            if self._xa_stats is not None:
                m, s, mx = self._xa_stats
                print('    [x_aug   ] ||x_aug|| end-of-rollout  mean %.3e  std %.3e  max %.3e'
                      % (m, s, mx))
        if (self._pblock is not None
                or getattr(self.fit_sys, 'orth_penalty', None) is not None):
            try:
                self._joint_probe()
            except Exception as e:
                print(f'    [joint-probe] failed (non-fatal): {e}')


def _install_nf_val_probe(fit_sys, hp, cfg, val_sd):
    """Append an `_NfProbe` to `fit_sys.validation_probes`; return the previous tuple to restore."""
    prev = fit_sys.validation_probes
    probe = _NfProbe(fit_sys, hp['nf'], cfg, val_sd, do_print=cfg.nf_probe_print)
    fit_sys.validation_probes = tuple(prev) + (probe,)
    if cfg.nf_probe_print:
        mode = 'closed loop' if getattr(fit_sys, 'simulator', None) is not None else 'open loop'
        # The horizon in seconds is derived from the nf actually in force, not from
        # `cfg.nf_seconds`. The two part company whenever `nf_override` is set (config.py:257):
        # nf_seconds is the REQUESTED horizon and the override wins, so a 3 s run at nf=12000 was
        # labelling every line below it "0.100 s".
        print(f"\n[nf] rms = RMS over nf={hp['nf']} ({hp['nf'] * cfg.ts_new:.3f} s) windows, "
              f"{mode}, metres")
        print('     chan = per-channel X/Theta/Y | grow = RMS(last step)/RMS(first) | '
              'bias = mean residual')
        # Report what the probe ACTUALLY has. The header used to claim the fixed val batch
        # unconditionally, so a run whose batch failed to build still printed "256 fixed
        # windows" and then never emitted an [nf val] line.
        _val_state = ('%d fixed windows, snapshot' % probe.N_VAL_WINDOWS
                      if probe._val_batch is not None else
                      'UNAVAILABLE (batch failed to build; no [nf val] line will appear)')
        print(f'     train = all windows since the last validation, running mean | '
              f'val = {_val_state}')
        print('     metre weighting is NOT the objective\'s, which weights channels by 1/ystd^2;')
        print('     numbers are not comparable ACROSS loop modes, the [nf gap] ratios are')
    return prev


def _gantry_meters(fit_sys):
    """The gantry's meters for the polish acceptance rule: `combo_err`, `orth_frac`, `V_orth`.

    Everything here needs to know what a gantry is, so it reaches the phase through the framework
    function's `meters_fn` hook rather than as an import inside `model_augmentation/`.
    `lbfgs_polish.generic_meters` keeps only `param_loss`, which is duck-typed on the block API.

    CHANGED (D-176): `orth_frac` and `V_orth` moved here from `generic_meters`, which computed
    them by reaching into `fit_sys.orth_penalty` and isinstance-testing for `Static_ANN_Block`.
    That put a second definition of an ACCEPTANCE-GATING metric inside the framework, beside the
    one in `_joint_probe` above. The arithmetic is unchanged, so the acceptance rule is unchanged.

    Mirrors `_NfProbe.__init__` and `_joint_probe` above, including the `m_diff` scale
    heuristic. Deliberately a separate function rather than a refactor of the probe: the probe
    is on the training path and this is not.
    """
    m = {}
    blocks = getattr(getattr(fit_sys, 'hfn', None), 'connected_blocks', ())

    # `orth_frac` is the fraction of the learned correction lying in the baseline's parameter
    # span, i.e. the negating part; `V_orth` is the penalty that fraction would carry.
    pen = getattr(fit_sys, 'orth_penalty', None)
    if pen is not None:
        from model_augmentation.fit_systems.blocks import Static_ANN_Block
        ann = next((b for b in blocks if isinstance(b, Static_ANN_Block)), None)
        if ann is not None:
            with torch.no_grad():
                w = ann(pen.Z_pts)
                f = w[:, pen.route_cols, 0].reshape(-1)
                f2 = float(torch.linalg.vector_norm(f) ** 2)
                q2 = float(torch.linalg.vector_norm(pen.Q.T @ f) ** 2)
            m['orth_frac'] = q2 / f2 if f2 > 1e-30 else float('nan')
            m['V_orth'] = float(pen.beta) * q2

    pblock = next((b for b in blocks if hasattr(b, 'identifiable_combinations')), None)
    if pblock is None:
        return m
    from model_augmentation.systems import gantry_ss as _gss
    true_raw = {n: getattr(_gss, n).item() for n in pblock.PARAM_NAMES}
    nom = pblock._combos_from_raw(true_raw, pblock.Lb.item())
    scale = {n: abs(v) for n, v in nom.items() if abs(v) > 1e-12}
    # HEURISTIC (as in _NfProbe): m_diff = m1-m2 has a near-zero nominal, so a relative error
    # against itself is meaningless; scale it by the mean actuator mass instead.
    scale['m_diff'] = 0.5 * (true_raw['m1'] + true_raw['m2'])
    combos = pblock.identifiable_combinations()
    rels = [(combos[n] - nom[n]) / s for n, s in scale.items()]
    m['combo_err'] = float(np.sqrt(np.mean([r ** 2 for r in rels])))
    return m


def run_lbfgs_polish(fit_sys, hp, cfg: RunConfig, data, bestfit=float('nan')):
    """The D-171 phase, with the device round-trip and the gantry meter wired in.

    Returns the outcome dict, or None when `cfg.lbfgs` is off (an exact no-op).

    The device flip lives here rather than in `model.py::train_model` so that BOTH entry paths
    reach the phase identically: a normal run arrives with the model back on the CPU
    (`model.py` line ~317), and a `start_phase='lbfgs'` resume never called `fit()` at all, so
    the model is still on the CPU it was built on (`model.py`, `init_model(device='cpu')`).
    """
    if not cfg.lbfgs:
        return None
    print('\n[lbfgs] polish phase (D-171): Adam is done, quasi-Newton from here')
    # ONE place where the flat RunConfig fields become the framework's spec object. Adding a knob
    # touches RunConfig, config_json_dict and this mapping, and nothing else.
    spec = PolishSpec(
        n_windows=cfg.lbfgs_windows, max_iter=cfg.lbfgs_max_iter,
        inner_iter=cfg.lbfgs_inner_iter, history_size=cfg.lbfgs_history,
        tol_grad=cfg.lbfgs_tol_grad, tol_change=cfg.lbfgs_tol_change,
        freeze=tuple(cfg.lbfgs_freeze), meter_rel_tol=cfg.lbfgs_meter_rel_tol,
        time_budget_s=cfg.lbfgs_time_budget_s, chunk=cfg.lbfgs_chunk, seed=cfg.seed)
    if cfg.device == 'cuda':
        fit_sys.cuda()
        # The caching allocator is holding a pool shaped by Adam's batch-512 graphs, and the
        # phase is about to ask for a very differently shaped one (16384 windows in a single
        # graph). Releasing the cached blocks first costs a few milliseconds and removes a
        # fragmentation-driven OOM that would otherwise land at the END of a training run.
        torch.cuda.empty_cache()
    try:
        return lbfgs_polish(
            fit_sys, data.train_data,
            loss_kwargs={'nf': hp['nf'], 'stride': cfg.stride}, spec=spec,
            val_sys_data=data.val_ckpt_data,
            # `bestfit` IS the pre-phase sim-RMS and fit() has already paid for it, so reusing it
            # saves a ~162 s closed-loop free run. ONLY in the same-run case, though.
            #
            # On a `start_phase='lbfgs'` resume the number comes from the CHECKPOINT, and it was
            # produced by a different run: possibly another device, another dtype (a float32
            # checkpoint polished at use_f64=True is the intended workflow) and the compiled
            # rollout rather than this one. Measured spread across such a change: 5.774376e-06
            # against 5.774347e-06, i.e. 0.005%. That is 20x below the ~0.3% acceptance bar set by
            # run 81262's own validation noise, so it would not flip a clear result, but it is a
            # systematic bias in exactly the comparison the phase exists to make. Pay the 162 s
            # and measure both sides with one code path.
            val_before=(bestfit if (np.isfinite(bestfit) and cfg.start_phase != 'lbfgs')
                        else None),
            validation_measure='sim-RMS',
            meters_fn=_gantry_meters, verbose=True)
    finally:
        if cfg.device == 'cuda':
            fit_sys.cpu()


def train_model_with_diagnostics(fit_sys, hp, cfg: RunConfig, data, norm,
                                 resume_ckpt=None, checkpoint_dir=None, run_id=None):
    """Train for hp['epochs'] epochs with sim-RMS validation; record aug-state R2 after.

    resume_ckpt : base path (no extension, or a `.pth`) of a prior checkpoint to resume from.
    checkpoint_dir : directory for checkpoint; None = no saving.

    With `cfg.lbfgs` set, an L-BFGS polish phase runs after Adam (D-171), and
    `cfg.start_phase='lbfgs'` skips Adam entirely to polish a resumed checkpoint.
    """
    diag_epochs, diag_r2_raw, diag_r2_lin = [], [], []
    done_epochs = 0
    resumed_at_epoch = -1                  # -1 = not a resume
    lbfgs_outcome = None

    # deepSI names its own checkpoints `<class>_<unique_code>_{best,last}.pth`, where `name` is a
    # property over `unique_code` (`systems/system.py:79`) and `unique_code` defaults to a random
    # `token_urlsafe(4)` (`system.py:54`). Those are the files `fit()` writes at every improving
    # validation and at the end, and until now they carried NO run identity: D-169 had to
    # fingerprint `SSE_Interconnect_Composed_DjS6ow` back to job 81088 by content.
    # `run_id` is the SLURM job id, or a timestamp when there is none, and it is unique, so it is
    # simply the better code. One assignment, no deepSI edit: the directory
    # (`~/.deepSI/checkpoints`) is unchanged, only the file name.
    if run_id:
        fit_sys.unique_code = str(run_id)
    bestfit = float('inf')

    # --- Resume ---
    if resume_ckpt is not None:
        meta = load_checkpoint(fit_sys, resume_ckpt, cfg.joint_estimation,
                               start_phase=cfg.start_phase)
        done_epochs = int(meta['done_epochs'])
        resumed_at_epoch = done_epochs      # the switch point actually used (D-171)
        bestfit     = float(meta['bestfit'])
        diag_epochs = list(meta['diag_epochs'])
        diag_r2_raw = [meta['diag_r2_raw'][i]  for i in range(len(meta['diag_r2_raw']))]
        diag_r2_lin = [meta['diag_r2_linmap'][i] for i in range(len(meta['diag_r2_linmap']))]
        print(f'Resumed from {resume_ckpt}  ({done_epochs} epochs done)')

    epochs_remaining = hp['epochs'] - done_epochs
    print(f'\nTraining: nf={hp["nf"]}  epochs={hp["epochs"]}  val=sim-RMS  NX_ANN={hp["NX_ANN"]}')
    if done_epochs > 0:
        print(f'  Resuming: {done_epochs} done, {epochs_remaining} remaining')

    fit_sys.bestfit = float('inf')
    t0 = time.time()
    # D-095: piggyback the nf-window RMS diagnostic (train + val); selection stays full-traj sim-RMS.
    _prev_probes = _install_nf_val_probe(fit_sys, hp, cfg, data.val_ckpt_data)
    try:
        if cfg.start_phase == 'lbfgs':
            # D-171: resume straight into the polish. The nf probe above is still installed and
            # `bestfit` keeps the resumed value, so the phase's own before/after validation is
            # measured against the checkpoint that produced it.
            print('  start_phase=lbfgs: skipping Adam, polishing the resumed weights directly')
        else:
            bestfit = train_model(fit_sys, hp, cfg, data,
                                  epochs=epochs_remaining,
                                  nf=hp['nf'],
                                  validation_measure='sim-RMS')
    finally:
        fit_sys.validation_probes = _prev_probes   # restore always
    loss_val_nf   = np.array(getattr(fit_sys, 'Loss_val_nf', []), dtype=float)
    loss_train_nf = np.array(getattr(fit_sys, 'Loss_train_nf', []), dtype=float)
    elapsed = time.time() - t0
    # D-171: on a polish-only resume no epoch was trained, so the count must stay where the
    # checkpoint left it. Claiming hp['epochs'] here would make a 30-minute polish look like a
    # full training run to every reader of the meta, including a later resume.
    if cfg.start_phase != 'lbfgs':
        done_epochs = hp['epochs']

    _fin = loss_val_nf[np.isfinite(loss_val_nf)] if loss_val_nf.size else loss_val_nf
    if _fin.size:
        print(f'  val nf-window RMS ({hp["nf"]}-step): first={loss_val_nf[0]:.4e}  '
              f'best={_fin.min():.4e}  last={loss_val_nf[-1]:.4e} [m]  (sim-RMS selector unchanged)')
    _fintr = loss_train_nf[np.isfinite(loss_train_nf)] if loss_train_nf.size else loss_train_nf
    if _fintr.size:
        # first/last come from the FINITE entries. The train row is a running mean over the
        # windows seen since the previous validation, so the entry written at the pre-training
        # validation has no windows behind it and is NaN by construction; printing the raw [0]
        # made every log open with `first=nan` and invited a hunt for a numerical problem that
        # was never there. `n` is stated when leading entries were dropped, so the row cannot
        # silently claim more measurements than it has.
        _skipped = int(loss_train_nf.size - _fintr.size)
        print(f'  train nf-window RMS ({hp["nf"]}-step): first={_fintr[0]:.4e}  '
              f'best={_fintr.min():.4e}  last={_fintr[-1]:.4e} [m]'
              + (f'  ({_fintr.size} finite of {loss_train_nf.size}; the pre-training entry has '
                 f'no windows behind it)' if _skipped else ''))

    # --- L-BFGS polish (D-171), between Adam and the checkpoint ---------------------------
    # The PRE-polish weights are written first, for the same reason D-070 writes weights before
    # the diagnostics: a crash inside the phase must not be able to lose the Adam result. It also
    # leaves the unpolished/polished pair on disk, which is the A/B this phase has to justify.
    if cfg.lbfgs and checkpoint_dir is not None:
        pre_base = os.path.join(checkpoint_dir, f'gantry_ckpt_{run_id}_pre_lbfgs')
        save_checkpoint_weights(fit_sys, pre_base)
        np.savez(pre_base + '.npz', done_epochs=np.array(done_epochs),
                 bestfit=np.array(bestfit), hp=np.array(json.dumps(hp)),
                 orig_run_id=np.array(run_id))
        print(f'  Pre-polish checkpoint: {pre_base}.pt')
    lbfgs_outcome = run_lbfgs_polish(fit_sys, hp, cfg, data, bestfit=bestfit)
    if lbfgs_outcome is not None and lbfgs_outcome.get('accepted'):
        # The phase only changes the weights when it improves the same selector fit() used, so
        # bestfit follows it. On a rollback nothing moved and bestfit is untouched.
        # The ATTRIBUTE follows too: `fit_sys.bestfit` is what deepSI's own reporting and any
        # later reader of the system reach for, and leaving it at the pre-polish value would make
        # the object disagree with the checkpoint written from it two lines below.
        bestfit = lbfgs_outcome['val_after']
        fit_sys.bestfit = bestfit

    # --- Checkpoint weights first: diagnostics below must not be able to lose them (D-070)
    ckpt_base = None
    # Two reasons to omit the Adam state, both "the file should say nothing rather than something
    # empty or wrong":
    #   polished    -> the weights moved past the moments (see save_checkpoint_weights)
    #   lbfgs start -> no Adam ran in THIS job. load_checkpoint deliberately does not restore the
    #                  optimizer on a polish-only resume, so `fit_sys.optimizer` is the one
    #                  init_model built, with `state` still empty because torch allocates moments
    #                  lazily on the first step. Writing that would look like a real Adam state
    #                  for these weights; the one that actually matches them is in the SOURCE
    #                  checkpoint.
    _polished = bool(lbfgs_outcome is not None and lbfgs_outcome.get('accepted'))
    _no_adam_ran = (cfg.start_phase == 'lbfgs')
    _drop_optimizer = _polished or _no_adam_ran
    if checkpoint_dir is not None:
        ckpt_base = os.path.join(checkpoint_dir, f'gantry_ckpt_{run_id}')
        save_checkpoint_weights(fit_sys, ckpt_base, include_optimizer=not _drop_optimizer)
        _why = ('weights were polished' if _polished else
                'no Adam phase ran in this job' if _no_adam_ran else '')
        print(f'  Checkpoint weights: {ckpt_base}.pt'
              + (f'  (Adam state omitted: {_why})' if _drop_optimizer else ''))

    fit_sys.eval()
    try:
        # TELICA-REAL: (G6 item 5) real data carries no absorber ground truth; skip the
        # latent-state diagnostic instead of letting it fail into a misleading NaN line.
        if getattr(data, 'val_x_aug', None) is None:
            raise RuntimeError('no ground-truth augmented states (real data): diagnostic skipped')
        # CHANGED (2026-09-01): three values, each of width 2 (the ABSORBER dimension), not
        # NX_ANN. `diag_r2_raw` keeps its name for checkpoint-resume compatibility (:307) but now
        # holds R2_best1; see best_single_channel_r2 for why the old statistic was dropped.
        r2_raw, r2_lin, _r2_ho = aug_state_r2(fit_sys, hp, cfg, data, norm)
    except Exception as e:
        print(f'Warning: aug_state_r2 failed ({e}); recording NaN')
        # Width 2, matching a successful call. The old np.full(NX_ANN, nan) is why plot 4 never
        # crashed at NX_ANN=8: the failure path happened to produce the width the plot expected,
        # so the mismatch only surfaced once the diagnostic worked.
        r2_raw = np.full(2, np.nan)
        r2_lin = np.full(2, np.nan)
    diag_epochs.append(done_epochs)
    diag_r2_raw.append(r2_raw.copy())
    diag_r2_lin.append(r2_lin.copy())

    r2_str = '  '.join([f'{n}={r2_lin[i]:+.4f}'
                        for i, n in enumerate(['delta_a', 'vdelta_a'][:len(r2_lin)])])
    print(f'Training done | {done_epochs} ep | {elapsed:.0f}s | '
          f'bestfit={bestfit:.5f}  R2_linmap: {r2_str}')

    # --- Checkpoint metadata (weights already saved above) ---
    if ckpt_base is not None:
        np.savez(ckpt_base + '.npz',
                 done_epochs    = np.array(done_epochs),
                 bestfit        = np.array(bestfit),
                 elapsed        = np.array(elapsed),
                 diag_epochs    = np.array(diag_epochs),
                 diag_r2_raw    = np.array(diag_r2_raw),
                 diag_r2_linmap = np.array(diag_r2_lin),
                 hp             = np.array(json.dumps(hp)),
                 orig_run_id    = np.array(run_id),
                 # D-171 provenance. Without these a polished checkpoint is indistinguishable
                 # from an Adam-only one, and the switch point that produced it is unrecorded.
                 lbfgs_outcome  = np.array(json.dumps(_json_safe(lbfgs_outcome))),
                 # Says the omission was DELIBERATE, so a reader does not diagnose a truncated
                 # file. Records the ACTUAL omission, not just the polish case, or the flag and
                 # the file would disagree on a rolled-back polish-only resume.
                 optimizer_state_dropped = np.array(_drop_optimizer),
                 start_phase    = np.array(cfg.start_phase),
                 resumed_from   = np.array('' if resume_ckpt is None else str(resume_ckpt)),
                 resumed_at_epoch = np.array(resumed_at_epoch),
        )
        print(f'  Checkpoint meta: {ckpt_base}.npz')

    return bestfit, dict(
        epochs      = np.array(diag_epochs),
        r2_raw      = np.array(diag_r2_raw),
        r2_linmap   = np.array(diag_r2_lin),
        loss_val_nf   = loss_val_nf,    # D-095: per-epoch val nf-window RMS (aligns with Loss_val tail)
        loss_train_nf = loss_train_nf,  # D-095: per-epoch train nf-window RMS (same horizon, meters)
    )


def resolve_resume_hp(cfg: RunConfig, default_hp: dict, resume_ckpt) -> dict:
    """@added (2026-09-15). The hp a resumed run actually trains with (D-179).

    Moved out of the entry point: this is POLICY, not a run choice. THE CHECKPOINT OWNS THE
    ARCHITECTURE, THIS RUN OWNS EVERYTHING ELSE. Taking `epochs` from the checkpoint made job
    81262 skip Adam entirely (`epochs: 350`, `done_epochs: 350` -> zero remaining); taking `lr`
    from it trained at the checkpoint rate while config.json recorded this run's.

    `_assert_same_architecture` below remains the real enforcement; taking the five structural
    keys from the checkpoint is what makes the run WORK rather than fail on key mismatches.
    """
    # D-171: 'lbfgs' means "polish a restored checkpoint", so without one there is nothing to
    # polish and nothing to train either. Caught here rather than in RunConfig.__post_init__,
    # because the checkpoint is an environment variable and a config must stay constructible
    # for inspection without it.
    if cfg.start_phase == 'lbfgs' and not resume_ckpt:
        raise RuntimeError(
            "start_phase='lbfgs' needs RESUME_CHECKPOINT: it skips Adam and polishes restored "
            "weights, so with no checkpoint the run would polish a freshly initialised model.")
    if resume_ckpt:
        # D-179: THE CHECKPOINT OWNS THE ARCHITECTURE, THIS RUN OWNS EVERYTHING ELSE.
        #
        # `hp` used to be taken wholesale from the checkpoint's `.npz`, and two of its keys are
        # not properties of the weights at all. `epochs` is the worse one:
        # `train_model_with_diagnostics` computes `epochs_remaining = hp['epochs'] - done_epochs`,
        # so resuming job 81262 (`epochs: 350`, `done_epochs: 350`) gave ZERO remaining epochs and
        # skipped Adam silently, whatever `cfg.epochs` said. `lr` is the same class of defect one
        # level up from D-173: `init_model` builds Adam with `hp['lr']`, so a resumed run trained
        # at the CHECKPOINT's rate while config.json recorded this run's.
        #
        # The five keys below are STRUCTURAL: any other value makes `load_state_dict` fail, so the
        # checkpoint must own them. Everything else (`nf`, `batch_size`, `lr`, `epochs`) is this
        # run's, which also makes "resume these weights and train at a longer horizon" simply
        # work. `_assert_same_architecture` remains the real enforcement; taking these from the
        # checkpoint is what makes the run WORK rather than fail with a wall of key mismatches.
        _STRUCTURAL = ('NX_ANN', 'n_nodes_per_layer', 'n_hidden_layers', 'na_nb', 'up_sample')
        print(f'\nResuming checkpoint: {resume_ckpt}')
        if os.path.exists(resume_ckpt + '.npz'):
            _meta = np.load(resume_ckpt + '.npz', allow_pickle=True)
            _saved_hp = json.loads(str(_meta['hp']))
            hp = {**default_hp,
                  **{k: _saved_hp[k] for k in _STRUCTURAL if k in _saved_hp}}
            # The diff, not a mode flag, is the safety mechanism: it names every field where the
            # two disagree, so a changed OBJECTIVE is visible rather than hidden behind a boolean.
            print('  hp: architecture from the checkpoint, everything else from this config')
            _diff = [(k, _saved_hp[k], hp[k]) for k in hp
                     if k in _saved_hp and _saved_hp[k] != hp[k]]
            for _k, _was, _now in _diff:
                print(f'      {_k:<20s} checkpoint {_was!r:>10s}  ->  this run {_now!r}'
                      + ('   [OBJECTIVE CHANGE]' if _k in ('nf', 'batch_size') else ''))
            if not _diff:
                print('      (identical on every field)')
        else:
            hp = default_hp
            print('  No .npz alongside it (deepSI whole-system format); using this config\'s hp')
            # Nothing in that format records the horizon it was trained at, so there is no diff to
            # print and no guard on the objective; the architecture is still caught by
            # training.py::_assert_same_architecture.
            print(f"  hp from THIS config: nf={hp['nf']} stride={cfg.stride} "
                  f"na_nb={hp['na_nb']}. If the checkpoint was trained at another nf, the "
                  f"resumed run optimises a different objective than the one that made it.")
        print(f'  start_phase: {cfg.start_phase}')
    else:
        hp = default_hp
    return hp
