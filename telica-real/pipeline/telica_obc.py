"""OBC on real Telica data (TR-025): the real-data reference set and the attachment.

Reuses the vendored gantry OBC machinery (`gantry_dynamic/obc_gantry.py`: correction module,
float64 basis builder, reference field; `model_augmentation/fit_systems/obc.py`: lifecycle) and
replaces only what reads the simulation: the reference set comes from the TRAIN records through
the loader, at the pipeline rate, with the frozen normalisation. The protected parameter vector is
the 13-vector of `ReducedGantryFrictionBlockCC` (ten combinations + log cc).
"""
import time

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from model_augmentation.fit_systems.blocks import Static_ANN_Block
from model_augmentation.fit_systems.obc import OBCLifecycle
from gantry_dynamic.obc_gantry import (GantryReferenceSet, GantryOBCCorrection,
                                       make_float64_block, make_basis_builder, make_reference_field)

from params import telica_params as tp
from baseline.records import iter_records
from friction.karnopp import ReducedGantryFrictionBlockCC
import rate

PRE_S = 0.1          # reference tuples from 100 ms before motion start, as the training records
P = np.array([[1.0, 1.0, 0.0], [tp.Lb / 2, -tp.Lb / 2, 0.0], [0.0, 0.0, 1.0]])


def build_reference_set_real(cfg, norm, nx_ann, stride=1, verbose=True, max_records=None):
    """Every train sample at the pipeline rate: q = P^-T y, qdot from the 200 Hz zero-phase
    low-passed positions (TR-016 rule; HEURISTIC), u the block-mean force, x_a = 0."""
    ts = cfg.ts_new
    five = cfg.fs_new is not None
    lp = butter(4, 200.0, fs=1.0 / ts, output='sos')
    PinvT = np.linalg.inv(P.T)
    x_mean = np.asarray(norm.x_mean, np.float64).reshape(1, 6)
    std_x = np.asarray(norm.std_x, np.float64).reshape(1, 6)
    u_mean = np.asarray(norm.u_mean, np.float64).reshape(1, 3)
    std_u = np.asarray(norm.std_u, np.float64).reshape(1, 3)
    xs, us, xn, rec, samp, counts, names = [], [], [], [], [], [], []
    t0 = time.perf_counter()
    for r, (s, op, it, d) in enumerate(iter_records(splits=('train',), log=lambda m: None)):
        if max_records is not None and r >= max_records:
            break
        if five:
            d = rate.decimate_record(d)
        ks = max(0, d['motion_idx'] - int(round(PRE_S / ts)))
        y, u = d['q1'][ks:], d['u'][ks:]
        q = y @ PinvT.T
        qd = np.gradient(sosfiltfilt(lp, q, axis=0), ts, axis=0)
        k = np.arange(2, len(u) - 2)[::stride]
        xs.append(np.hstack([q[k], qd[k]])); us.append(u[k])
        xn.append(np.hstack([q[k + 1], qd[k + 1]]))
        rec.append(np.full(len(k), r)); samp.append(k); counts.append(len(k))
        names.append(f'{s}/{op}/{it}')
    x_phys = np.concatenate(xs); u_stage = np.concatenate(us); x_next = np.concatenate(xn)
    N = len(x_phys)
    x_n = (x_phys - x_mean) / std_x
    u_n = (u_stage - u_mean) / std_u
    dt = cfg.dtype_pt
    ref = GantryReferenceSet(
        Z_step=torch.from_numpy(np.hstack([x_n, u_n])).to(dt).unsqueeze(-1),
        Z_field=torch.from_numpy(np.hstack([x_n, np.zeros((N, nx_ann)), u_n])).to(dt).unsqueeze(-1),
        record_ix=torch.from_numpy(np.concatenate(rec)).long(),
        sample_ix=torch.from_numpy(np.concatenate(samp)).long(),
        x_phys=torch.from_numpy(x_phys), u_stage=torch.from_numpy(u_stage),
        x_phys_next=torch.from_numpy(x_next), x_abs=torch.zeros(N, 2, dtype=torch.float64),
        n_per_record=counts, record_names=names + ([] if stride == 1 else [f'(stride {stride})']),
        ts=ts)
    if verbose:
        v = x_phys[:, 3:] @ P                                  # stage velocities
        slide = (np.abs(v) > 3 * tp.V0_TANH).mean(0)
        print(f'[obc] real reference set: {N} tuples from {len(counts)} train records, stride '
              f'{stride}, {time.perf_counter() - t0:.1f} s; sliding share per rail '
              f'{np.round(slide, 3).tolist()}; hot tensors {ref.memory_bytes() / 1e6:.1f} MB',
              flush=True)
    return ref


def attach_obc_real(fit_sys, cfg, data, norm, ref=None, ref_stride=1, expected_rank=None,
                    verbose=True):
    """Reference set, correction module and lifecycle on a built real-data `fit_sys`."""
    blocks = list(fit_sys.hfn.connected_blocks)
    phy = next((b for b in blocks if isinstance(b, ReducedGantryFrictionBlockCC)), None)
    ann = next((b for b in blocks if isinstance(b, Static_ANN_Block)), None)
    if phy is None or ann is None:
        raise RuntimeError('OBC on real data needs ReducedGantryFrictionBlockCC and a '
                           'Static_ANN_Block (joint_estimation=True, TR-025)')
    nx = fit_sys.hfn.nx
    if ref is None:
        ref = build_reference_set_real(cfg, norm, nx_ann=nx - cfg.nx_phys, stride=ref_stride,
                                       verbose=verbose)
    corr = GantryOBCCorrection(phy, nx=nx, nx_phys=cfg.nx_phys, nu=cfg.nu,
                               route_ix=cfg.ann_route_ix, dtype=cfg.dtype_pt,
                               space=cfg.obc_space).to(next(phy.parameters()).device)
    p = int(phy.free_params.numel())
    names = list(phy.COMBO_NAMES) + list(phy.CC_NAMES)
    affine = cfg.obc_space == 'affine'
    life = OBCLifecycle(
        reference_field=make_reference_field(ref, corr.field_cols),
        basis_builder=make_basis_builder(make_float64_block(phy), ref, corr.row_ix,
                                         chunk=cfg.obc_ref_chunk, space=cfg.obc_space),
        refresh=cfg.obc_refresh, rtol=cfg.obc_rank_rtol, out_dtype=cfg.dtype_pt,
        expected_rank=(expected_rank if expected_rank is not None else p) + int(affine),
        verbose=verbose, name_columns=names + (['f_bar'] if affine else []),
        scale_columns=affine)
    fit_sys.hfn.obc_correction = corr
    fit_sys.obc = life
    fit_sys.obc_reference = ref
    if verbose:
        print(f'[obc] attached (real data): {p} protected directions {names}, rows {corr.row_ix}, '
              f'{ref.n} reference tuples, space={cfg.obc_space}, refresh={cfg.obc_refresh}, '
              f'expected rank {life.expected_rank}', flush=True)
    return life, corr, ref
