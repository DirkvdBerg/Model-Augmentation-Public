"""Build (or load) the frozen trajectory-orthogonality geometry and attach the penalty hook.

The gantry side of the method. Everything mathematical is in
`model_augmentation/fit_systems/trajectory_orth_projection.py`; the gradient contribution is in
`trajectory_penalty_hook.py`; the rollout comes from the production adapter under
`scripts/gantry/orthogonality/gantry/`. This file only decides WHICH windows, WHICH checkpoint
and WHERE the artifact lives, and it mirrors `orth_penalty.py`'s cache conventions so there is
one way to do that in this package rather than two.

THE ARTIFACT IS THE REFERENCE. The specification's whole scheme rests on the geometry being
frozen at a documented reference, so the artifact records what it was built from and is REFUSED
rather than silently rebuilt when the configuration no longer matches. Rebuilding at current
parameters is a different algorithm (D-183 re-anchoring), and silently sliding from one to the
other is exactly the failure the frozen-basis design exists to avoid.

WHAT IS NOT DECIDED HERE. `traj_beta` is a chosen constant with no justification until an L-curve
is run (RULES.md, 2026-09-07). This file records it; it does not select it.
"""
__project_origin__ = "added"

import hashlib
import json
import os

import numpy as np
import torch

from gantry_dynamic.config import RunConfig, save_dir

_ORTH_PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'orthogonality')


def _import_adapter():
    """Import the verification package's adapter without duplicating it here.

    The adapter is the ONE seam onto the production rollout and it is already written, tested and
    exercised by the verification ladder. Re-implementing a second one for training would
    reintroduce precisely the divergence the adapter exists to prevent.
    """
    import sys
    if _ORTH_PKG not in sys.path:
        sys.path.insert(0, _ORTH_PKG)
    from gantry.adapter import GantryTrajectoryAdapter          # noqa: E402
    from gantry.encoder_split import split_encoder              # noqa: E402
    return GantryTrajectoryAdapter, split_encoder


def _file_sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _tensor_sha1(named):
    """Content hash of an ordered (name, tensor) sequence, dtype and shape included."""
    h = hashlib.sha1()
    for name, t in named:
        h.update(name.encode())
        a = t.detach().cpu().numpy()
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _cache_key(cfg: RunConfig, ckpt_base, window_ids, env=None, encoder=None):
    """Hash everything that changes the geometry. Anything omitted is a silent stale load.

    CONTENT, NOT NAMES. Corrected 2026-09-08 after review: this keyed the checkpoint by its
    BASENAME, so two different checkpoints sharing a filename and a configuration would reuse
    each other's geometry. That directly contradicts the "the artifact IS the reference"
    guarantee this module exists to provide, and it would be invisible: the wrong reference
    projects perfectly well.

    Hashed now: the checkpoint FILE, the normalisation constants, the controller bank matrices,
    and the encoder-conversion state. Each of those changes the scored stack or the geometry
    built on it, and none of them is implied by the RunConfig fields.
    """
    ck_path = str(ckpt_base) + '.pt'
    ckpt_hash = _file_sha1(ck_path) if os.path.exists(ck_path) else 'MISSING:' + str(ckpt_base)

    norm_hash = ctrl_hash = enc_hash = 'unavailable'
    if env is not None:
        n = env.norm
        norm_hash = _tensor_sha1([(k, torch.as_tensor(np.asarray(getattr(n, k))))
                                  for k in ('x_mean', 'std_x', 'std_u', 'u_mean', 'ystd', 'y0',
                                            'Cd_norm', 'Dd_np')])
        sim = getattr(env.fit_sys, 'simulator', None)
        bank = getattr(sim, 'bank', None)
        if bank is not None:
            # HASH THE BANK'S ACTUAL BUFFERS, and refuse if the ones that matter are absent.
            #
            # Corrected 2026-09-08 after review. This listed `M_state, M_in, M_out, M_ff` and
            # filtered on `isinstance(..., Tensor)`. Three of those names DO NOT EXIST on
            # `ControllerBank` -- its buffers are `M_state`, `M_error`, `ystd`, `stdu` -- and the
            # filter skipped them in silence, so the hash covered `M_state` alone. `M_error`
            # carries the whole error-feedback path (Dn and Bn), so changing the controller could
            # change every closed-loop prediction while the cache key stayed identical. Taking
            # `state_dict()` means a buffer added later is covered without an edit here, and the
            # completeness assertion means a buffer RENAMED later fails loudly instead of
            # quietly shrinking the hash.
            sd = bank.state_dict()
            required = {'M_state', 'M_error'}
            missing = required - set(sd)
            if missing:
                raise RuntimeError(
                    'the controller bank is missing buffer(s) %s, so the geometry cache key '
                    'cannot cover the controller. Either the bank changed shape or this list is '
                    'stale; do not silently hash a subset -- a controller change would then '
                    'reuse a geometry built against a different closed loop.' % sorted(missing))
            ctrl_hash = _tensor_sha1(sorted(sd.items(), key=lambda kv: kv[0]))
            # WHICH controller each record uses is part of the reference too: the same bank with
            # a permuted row assignment is a different predictor on the same windows.
            rows = getattr(sim, 'train_ctrl_rows', None) or getattr(
                getattr(sim, 'indexer', None), 'record_rows', None)
            if rows is not None:
                ctrl_hash = hashlib.sha1(
                    (ctrl_hash + '|rows:' + json.dumps(list(rows))).encode()).hexdigest()
    enc = encoder if encoder is not None else (env.fit_sys.encoder if env is not None else None)
    if enc is not None:
        enc_hash = _tensor_sha1(sorted(((k, v) for k, v in enc.state_dict().items()),
                                       key=lambda kv: kv[0]))

    payload = dict(
        algorithm='trajectory_orth_v2_content_hashed',
        mode=cfg.mode, fs=cfg.fs_new_hz, nf=cfg.nf, burn_in=cfg.burn_in, stride=cfg.stride,
        nx_ann=cfg.nx_ann, n_nodes=cfg.n_nodes_per_layer, n_hidden=cfg.n_hidden_layers,
        route_ix=list(cfg.ann_route_ix), na_nb=cfg.hp['na_nb'], up_sample=cfg.up_sample,
        encoder_init=cfg.encoder_init, closed_loop=cfg.closed_loop,
        use_f64=cfg.use_f64, rank_rtol=cfg.traj_rank_rtol,
        checkpoint=os.path.basename(str(ckpt_base)), checkpoint_sha1=ckpt_hash,
        normalisation_sha1=norm_hash, controller_sha1=ctrl_hash,
        encoder_state_sha1=enc_hash, encoder_class=type(enc).__name__ if enc else None,
        windows=list(window_ids),
    )
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return key, payload


def geometry_path(cfg: RunConfig, key):
    d = os.path.join(save_dir(cfg), 'traj_orth_cache')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'traj_geom_{key}.npz')


def save_geometry(path, geom, payload, extra):
    """Persist the specification's Sect. 9.3 list, so a resume can VERIFY rather than rebuild.

    Uses the geometry's own `save_state`, so the saved fields and the reconstructed object cannot
    drift apart: adding a field to the class adds it here without an edit.
    """
    st = geom.save_state()
    arrays = {k: v for k, v in st.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in st.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path, payload=json.dumps(payload),
                        extra=json.dumps(extra, default=str),
                        state_scalars=json.dumps(scalars, default=str), **arrays)
    return path


def load_geometry(path, payload, dtype=torch.float64, device=None):
    """Reconstruct the frozen geometry, REFUSING a mismatched provenance.

    Corrected 2026-09-08 after review. The previous version built the object from a
    `zeros((N, 1))` placeholder `S` and then patched in the bases and ranks, which left
    `zero_rank=True`, `scale_LS=0`, an unrestored `rank_atol` and an empty `sv_E` on an object
    that nonetheless projected correctly. `from_state` restores every field or refuses.
    """
    from model_augmentation.fit_systems.trajectory_orth_projection import (
        TrajectoryProjectionGeometry)
    z = np.load(path, allow_pickle=True)
    saved = json.loads(str(z['payload']))
    if saved != payload:
        diff = {k: (saved.get(k), payload.get(k))
                for k in set(saved) | set(payload) if saved.get(k) != payload.get(k)}
        raise RuntimeError(
            'the saved geometry was built from a different reference and will NOT be silently '
            'reused. Differing keys: %s. Rebuilding at the current parameters is a DIFFERENT '
            'algorithm (re-anchoring, D-183); delete the artifact deliberately if that is what '
            'you want.' % diff)
    st = json.loads(str(z['state_scalars']))
    for k in z.files:
        if k not in ('payload', 'extra', 'state_scalars'):
            st[k] = z[k]
    st['metadata'] = dict(loaded_from=path, **json.loads(str(z['extra'])))
    return TrajectoryProjectionGeometry.from_state(st, dtype=dtype, device=device)


# --------------------------------------------------------------------------- entry point
def build_traj_penalty(fit_sys, cfg: RunConfig, data, norm, env=None, verbose=True):
    """Attach the trajectory penalty to `fit_sys`, returning the hook.

    `env` is an already-built `GantryEnv` when the caller has one (the verification driver does);
    the training entry point does not, so the windows are selected here from the same production
    constructor the ladder uses.
    """
    from model_augmentation.fit_systems.trajectory_penalty_hook import TrajectoryPenaltyHook
    from model_augmentation.fit_systems.trajectory_orth_projection import (
        TrajectoryProjectionGeometry, compressed_nuisance_basis)
    GantryTrajectoryAdapter, split_encoder = _import_adapter()
    from gantry.env import production_windows, to_tensors                 # noqa: E402

    if env is None:
        raise RuntimeError(
            'build_traj_penalty needs a GantryEnv: the geometry is defined at a FIXED reference '
            'checkpoint on a FIXED window set, and constructing one implicitly from whatever '
            'model happens to be in hand is how a frozen reference stops being frozen.')

    # ---- the encoder must be split, or the ownership contract does not hold ----------
    from gantry.encoder_split import SplitEncoderInitAug                  # noqa: E402
    if not isinstance(fit_sys.encoder, SplitEncoderInitAug):
        # THE CONVERTED ENCODER MUST LAND WHERE THE MODEL LIVES, dtype AND device. Passing only
        # the dtype left any newly constructed layer on the CPU default; the split's new final
        # Linear was exactly that case. Stated here as a contract as well as fixed at the source.
        _ref = next(fit_sys.hfn.parameters())
        fit_sys.encoder = split_encoder(fit_sys.encoder).to(dtype=_ref.dtype, device=_ref.device)
        if verbose:
            print('[traj-orth] encoder converted to separate physical/latent correction '
                  'branches (output-preserving; optimizer state must be reset identically in '
                  'every arm of a comparison)')

    n_win = int(cfg.traj_windows)
    records = list(range(len(data.train_list)))[:n_win] or [0]
    win = production_windows(env, records, max(1, n_win // len(records)),
                             seed=cfg.seed, verbose=verbose)
    keep = np.arange(min(n_win, win['n_windows']))
    win_t = to_tensors(win, next(fit_sys.hfn.parameters()).dtype, device=cfg.device)
    win_t = {k: (v[keep] if hasattr(v, 'shape') and v.shape[0] == win['n_windows'] else v)
             for k, v in win_t.items()}
    win_t['provenance'] = win['provenance']
    window_ids = [f"{p['record']}@{p['slice_start']}" for p in win['provenance']]

    ad = GantryTrajectoryAdapter(env, win_t)
    key, payload = _cache_key(cfg, env.ckpt_base, window_ids, env=env, encoder=fit_sys.encoder)
    path = cfg.traj_geometry or geometry_path(cfg, key)

    # DTYPE AND DEVICE COME FROM THE MODEL, not from the geometry's float64 default. The build
    # path constructs in float64 (the specification's reference precision) and the model here may
    # be float32 on a GPU; a mismatch surfaces as a confusing error inside a matmul, or worse as
    # a silent upcast of the whole penalty.
    _p = next(fit_sys.hfn.parameters())
    m_dtype, m_device = _p.dtype, _p.device

    # THE RANK FLOOR MUST REFLECT THE PRECISION S WAS COMPUTED IN.
    #
    # Found 2026-09-08 by running this path for the first time. The verification ladder builds
    # geometry on a float64 copy and measures rank(S) = 10, with the four structurally-null
    # directions (kb1-kb2, cb1-cb2, Jb-Jh and the fourth) at 1e-18 and below. This path runs at
    # the CHECKPOINT dtype, float32, and two of those directions surface at 4.4e-09 and 2.4e-10
    # -- float32 round-off, not physics. Against `scale_LS = 1.6e-2` a `rtol = 1e-12` floor is
    # `1.6e-14`, so both survived the cut and the geometry came out RANK 12.
    #
    # A rank-12 basis is not a harmless over-count: `Q_S` then spans two directions in which the
    # "physical sensitivity" is numerical noise, and the penalty charges the augmentation for
    # overlapping with them. That is the specification's own `tau_S` warning (Sect. 6.4) arriving
    # in practice.
    #
    # So the floor is derived from the working precision rather than taken as a constant. At
    # float32 this gives 1e-6 relative, which puts the cut between 3.6e-06 (the tenth, real) and
    # 4.4e-09 (the eleventh, round-off) and recovers rank 10. The effective value is RECORDED in
    # the payload, so it is part of the cache key and cannot change a geometry silently.
    # HEURISTIC: floor the relative rank tolerance at 10 * eps(dtype). There is no literature
    # source for the factor 10 and it is an engineering choice, flagged as CLAUDE.md requires.
    # The REASONING, and it is checkable rather than arbitrary: a singular value of a matrix
    # computed in a precision with machine epsilon `eps` carries an absolute error of order
    # `eps * ||S||_2`, so anything below that is not distinguishable from zero. The factor 10 is
    # a margin, not a derivation. On this checkpoint at float32 it puts the cut between the
    # tenth singular value (3.6e-06, real) and the eleventh (4.4e-09, round-off in a
    # structurally null direction), a gap of nearly three decades -- so the verdict is
    # insensitive to the factor over at least 1 to 100 here. THAT MARGIN IS NOT GUARANTEED on
    # another checkpoint or another window set, and the spectra are saved so the cut can be
    # inspected rather than trusted. Prefer building the geometry in float64 (`--f64`), where
    # the same directions sit at 1e-18 and the choice does not arise.
    eps = float(torch.finfo(m_dtype).eps)
    eff_rtol = max(float(cfg.traj_rank_rtol), 10.0 * eps)
    if eff_rtol > float(cfg.traj_rank_rtol) and verbose:
        print('[traj-orth] rank tolerance raised from %.1e to %.1e: S is computed in %s, whose '
              'eps is %.1e, so a tighter floor would admit round-off in the structurally null '
              'physical directions as real rank.'
              % (cfg.traj_rank_rtol, eff_rtol, m_dtype, eps))
    payload['effective_rank_rtol'] = eff_rtol
    payload['geometry_dtype'] = str(m_dtype)
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    path = cfg.traj_geometry or geometry_path(cfg, key)

    if os.path.exists(path):
        geom = load_geometry(path, payload, dtype=m_dtype, device=m_device)
        if verbose:
            print(f'[traj-orth] loaded frozen geometry {path}  '
                  f'rank(S) {geom.rank_S_before_profiling} -> {geom.rank_S}, '
                  f'rank(Q_E) {geom.rank_E}, N {geom.N}')
    else:
        if verbose:
            print(f'[traj-orth] building geometry on {ad.n_win} windows, N = {ad.n_rows()} ...')
        S = ad.physical_jacobian()
        G = ad.initial_state_jacobians()
        Jb, _names, sizes = ad.encoder_jacobians()
        Q_E, info = compressed_nuisance_basis(G, Jb, rtol=eff_rtol, N=S.shape[0])
        geom = TrajectoryProjectionGeometry(
            S, Q_E=Q_E, L=ad.loss_weight(), rank_rtol=eff_rtol, strict=False,
            metadata=ad.metadata())
        save_geometry(path, geom, payload,
                      dict(windows=window_ids, n_xi_p=int(sum(sizes)),
                           compressed={k: v for k, v in info.items() if k != 'sv_Gamma'}))
        geom.to_model(dtype=m_dtype, device=m_device)
        if verbose:
            print(f'[traj-orth] geometry saved to {path}\n' + geom.summary())

    # THE RANK PREREQUISITE, applied rather than described. Sect. 6.5: zero profiled rank stops
    # the protection experiment, and reduced rank narrows its claim. Refusing here is the point:
    # a pilot on a collapsed geometry would look like a run and measure nothing.
    if geom.rank_S == 0:
        raise RuntimeError(
            'the profiled physical rank is ZERO on this reference: the encoder can reproduce '
            'every physical parameter effect on this stack, so no penalty can protect them here '
            '(spec Sect. 6.5). Diagnose the excitation, the window set and the encoder overlap. '
            'Do NOT lower the rank tolerance to make this pass.')
    if geom.rank_S < geom.rank_S_before_profiling and verbose:
        print(f'[traj-orth] WARNING: encoder profiling removed '
              f'{geom.rank_S_before_profiling - geom.rank_S} supported physical direction(s). '
              f'Any result from this run is a REDUCED-SPAN diagnostic, not the '
              f'all-supported-directions comparison (spec Sect. 6.5).')

    hook = TrajectoryPenaltyHook(ad, geom, beta=cfg.traj_beta, chunk=cfg.traj_chunk,
                                 verbose_every=cfg.traj_log_every)
    fit_sys.traj_penalty = hook
    if verbose:
        # `check_ownership` no longer treats a zero augmentation gradient as a defect (it is the
        # correct value at augmentation-off, at zero projected residual, and at a stationary
        # point). Reachability is asked separately, where a nonzero gradient IS expected.
        rep = hook.check_ownership()
        reach = hook.check_reachability()
        print('[traj-orth] attached. beta = %g (A CHOSEN CONSTANT WITH NO JUSTIFICATION until an L-curve is run), N = %d, chunk = %d'
              % (cfg.traj_beta, geom.N, cfg.traj_chunk))
        print('[traj-orth] ownership on the live model: g_lambda %.1e, g_xi_p %.1e, g_eta %.3e, V %.3e'
              % (rep['physical'], rep['encoder'], rep['augmentation'], rep['V']))
        if rep['augmentation_gradient_is_zero']:
            # NOT a defect. D_eta V = 0 is the correct value at augmentation-off, at zero
            # projected residual, and at a stationary point of the penalty. An earlier
            # version raised here, which would have aborted a run for succeeding.
            print('[traj-orth] NOTE: g_eta is zero at this reference. That is a legitimate '
                  'value, not a defect; see check_ownership.')
        print('[traj-orth] reachability under a %g relative perturbation of eta: g_eta %.3e, V %.3e'
              % (reach['perturbation_scale'], reach['augmentation'], reach['V']))
    return hook
