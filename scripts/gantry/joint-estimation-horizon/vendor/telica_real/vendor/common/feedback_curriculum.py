"""Experiment support for target-aware residual-feedback training.

This module is intentionally separate from the ordinary training path. The only production change
needed by the experiment is ``feedback_strength`` in the shared closed-loop rollout. Everything
that decides arms, stages, rollback, archives, and the target selector lives here.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import eig


@dataclass(frozen=True)
class FixedWindowBatch:
    uhist: torch.Tensor
    yhist: torch.Tensor
    ufuture: torch.Tensor
    yfuture: torch.Tensor
    ctrl_ix: torch.Tensor
    record_counts: tuple


def build_fixed_window_batch(fit_sys, val_records, nf=400, windows_per_record=64):
    """Build deterministic, non-overlapping validation windows without training-controller rows.

    ``fit_sys.make_training_data`` normally lets the attached simulator append TRAIN controller
    rows. We detach it only while constructing each registered validation record, then append the
    already-verified validation row ourselves. No model or dataset value is modified.
    """
    old_simulator = fit_sys.simulator
    arrays = [[] for _ in range(4)]
    ctrl_rows = []
    counts = []
    try:
        fit_sys.simulator = None
        for _, sd, ctrl_row in val_records:
            d = fit_sys.make_training_data(fit_sys.norm.transform(sd), nf=nf, stride=nf)
            n = min(int(windows_per_record), len(d[0]))
            if n < 1:
                raise RuntimeError('validation record has no complete nf=%d window' % nf)
            # Equally spaced and deterministic. This covers the whole 12 s record rather than
            # making the selector depend on a random draw or on the experiment seed.
            ix = np.linspace(0, len(d[0]) - 1, n, dtype=int)
            for dst, src in zip(arrays, d[:4]):
                dst.append(np.asarray(src)[ix])
            ctrl_rows.append(np.full(n, int(ctrl_row), dtype=np.int64))
            counts.append(n)
    finally:
        fit_sys.simulator = old_simulator

    dtype = next(fit_sys.hfn.parameters()).dtype
    tensors = [torch.as_tensor(np.ascontiguousarray(np.concatenate(a)), dtype=dtype)
               for a in arrays]
    return FixedWindowBatch(*tensors,
                            torch.as_tensor(np.concatenate(ctrl_rows), dtype=torch.long),
                            tuple(counts))


def save_training_snapshot(fit_sys, path, metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'hfn': fit_sys.hfn.state_dict(),
        'encoder': fit_sys.encoder.state_dict(),
        'optimizer': fit_sys.optimizer.state_dict(),
        'metadata': {} if metadata is None else dict(metadata),
    }, path)


def load_training_snapshot(fit_sys, path):
    obj = torch.load(path, map_location='cpu')
    fit_sys.hfn.load_state_dict(obj['hfn'])
    fit_sys.encoder.load_state_dict(obj['encoder'])
    fit_sys.optimizer.load_state_dict(obj['optimizer'])
    return obj.get('metadata', {})


class FeedbackTargetSelector:
    """Validation selector, checkpoint archive, and protected-curriculum state machine.

    The scalar returned to deepSI is always the autonomous ``alpha=0`` fixed-window RMS. Training
    still uses ``fit_sys.simulator.feedback_strength``. Thus selection cannot silently revert to
    the controller-assisted objective.
    """

    SCHEDULE = (1.0, 0.5, 0.2, 0.0)

    def __init__(self, batch, output_dir, arm, total_epochs=20,
                 plateau_rel=0.01, damage_rel=0.05, min_stage_epochs=2,
                 max_stage_epochs=3, chunk_size=64):
        if arm not in ('closed', 'open', 'curriculum'):
            raise ValueError('arm must be closed, open, or curriculum, got %r' % arm)
        self.batch = batch
        self.output_dir = os.path.abspath(output_dir)
        self.arm = arm
        self.total_epochs = int(total_epochs)
        self.plateau_rel = float(plateau_rel)
        self.damage_rel = float(damage_rel)
        self.min_stage_epochs = int(min_stage_epochs)
        self.max_stage_epochs = int(max_stage_epochs)
        self.chunk_size = int(chunk_size)
        self.archive_dir = os.path.join(self.output_dir, 'epoch_checkpoints')
        os.makedirs(self.archive_dir, exist_ok=True)

        self.calls = 0
        self.stage_index = 0
        self.stage_epochs = 0
        self.stage_entry_target = None
        self.stage_best_loss = None
        self.stage_best_target = float('inf')
        self.stage_best_path = None
        self.best_global_target = float('inf')
        self.best_global_path = None
        self.metrics = []
        self.transitions = []
        self.failed_transition = None

    @property
    def initial_strength(self):
        return {'closed': 1.0, 'open': 0.0, 'curriculum': self.SCHEDULE[0]}[self.arm]

    def _evaluate(self, fit_sys, strength):
        batch = self.batch
        ystd = torch.as_tensor(np.asarray(fit_sys.norm.ystd).ravel(),
                               dtype=batch.yfuture.dtype)
        sum_sq = torch.zeros((), dtype=torch.float64)
        chan_sq = torch.zeros(batch.yfuture.shape[-1], dtype=torch.float64)
        n_scalar = 0
        n_per_chan = 0
        max_abs = 0.0
        finite = True
        with torch.no_grad():
            for lo in range(0, len(batch.uhist), self.chunk_size):
                sl = slice(lo, min(lo + self.chunk_size, len(batch.uhist)))
                x0 = fit_sys.encoder(batch.uhist[sl], batch.yhist[sl])
                yp, _ = fit_sys.simulator.rollout_at(
                    fit_sys, x0, batch.ufuture[sl], batch.yfuture[sl],
                    batch.ctrl_ix[sl], strength)
                err = (yp - batch.yfuture[sl]) * ystd
                if not bool(torch.isfinite(err).all()):
                    finite = False
                    break
                e64 = err.double()
                sum_sq += torch.sum(e64.square())
                chan_sq += torch.sum(e64.square(), dim=(0, 1))
                n_scalar += e64.numel()
                n_per_chan += e64.shape[0] * e64.shape[1]
                max_abs = max(max_abs, float(torch.max(torch.abs(e64))))
        if not finite or n_scalar == 0:
            return dict(rms=float('inf'), channel_rms=[float('inf')] * len(chan_sq),
                        max_abs=float('inf'), finite=False)
        return dict(
            rms=math.sqrt(float(sum_sq) / n_scalar),
            channel_rms=torch.sqrt(chan_sq / n_per_chan).cpu().numpy().tolist(),
            max_abs=max_abs,
            finite=True,
        )

    def _checkpoint_path(self, epoch):
        return os.path.join(self.archive_dir, 'epoch_%03d.pt' % epoch)

    def _write_state(self):
        path = os.path.join(self.output_dir, 'feedback_training_trace.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'arm': self.arm,
                'schedule': list(self.SCHEDULE) if self.arm == 'curriculum'
                            else [self.initial_strength],
                'record_counts': list(self.batch.record_counts),
                'plateau_rel': self.plateau_rel,
                'damage_rel': self.damage_rel,
                'min_stage_epochs': self.min_stage_epochs,
                'max_stage_epochs': self.max_stage_epochs,
                'metrics': self.metrics,
                'transitions': self.transitions,
                'failed_transition': self.failed_transition,
                'best_global_target': self.best_global_target,
                'best_global_path': self.best_global_path,
            }, f, indent=2)

    def __call__(self, fit_sys, val_sys_data, validation_measure):
        del val_sys_data, validation_measure  # the fixed, stored validation batch is authoritative
        epoch = self.calls
        strength = float(fit_sys.simulator.feedback_strength)
        target = self._evaluate(fit_sys, 0.0)
        stage = target if strength == 0.0 else self._evaluate(fit_sys, strength)
        path = self._checkpoint_path(epoch)
        save_training_snapshot(fit_sys, path, metadata={
            'epoch': epoch, 'arm': self.arm, 'feedback_strength': strength,
            'target_rms': target['rms'], 'stage_rms': stage['rms'],
        })

        if target['rms'] < self.best_global_target:
            self.best_global_target = target['rms']
            self.best_global_path = path

        transition_reason = None
        if epoch == 0:
            self.stage_entry_target = target['rms']
            self.stage_best_loss = stage['rms']
            self.stage_best_target = target['rms']
            self.stage_best_path = path
        else:
            self.stage_epochs += 1
            prior_best_loss = self.stage_best_loss
            self.stage_best_loss = min(self.stage_best_loss, stage['rms'])
            if target['rms'] < self.stage_best_target:
                self.stage_best_target = target['rms']
                self.stage_best_path = path

            if self.arm == 'curriculum' and strength > 0.0:
                rel_improvement = ((prior_best_loss - self.stage_best_loss) / prior_best_loss
                                   if np.isfinite(prior_best_loss) and prior_best_loss > 0 else 0.0)
                plateau = (self.stage_epochs >= self.min_stage_epochs
                           and rel_improvement < self.plateau_rel)
                damage = target['rms'] > (1.0 + self.damage_rel) * self.stage_entry_target
                capped = self.stage_epochs >= self.max_stage_epochs
                if damage:
                    transition_reason = 'target_damage'
                elif plateau:
                    transition_reason = 'stage_plateau'
                elif capped:
                    transition_reason = 'stage_cap'

        row = dict(epoch=epoch, feedback_strength=strength,
                   stage_index=self.stage_index, stage_epoch=self.stage_epochs,
                   stage_rms=stage['rms'], target_rms=target['rms'],
                   stage_channel_rms=stage['channel_rms'],
                   target_channel_rms=target['channel_rms'],
                   stage_max_abs=stage['max_abs'], target_max_abs=target['max_abs'],
                   checkpoint=path, transition_reason=transition_reason)
        self.metrics.append(row)

        returned_target = target['rms']
        if transition_reason is not None:
            restored = self.stage_best_path
            load_training_snapshot(fit_sys, restored)
            self.stage_index += 1
            next_strength = self.SCHEDULE[self.stage_index]
            fit_sys.simulator.set_feedback_strength(next_strength)
            next_stage = self._evaluate(fit_sys, next_strength)
            restored_target = self._evaluate(fit_sys, 0.0)
            self.transitions.append(dict(
                after_epoch=epoch, reason=transition_reason, from_strength=strength,
                to_strength=next_strength, restored_checkpoint=restored,
                restored_target_rms=restored_target['rms'],
                next_stage_rms=next_stage['rms'], next_stage_finite=next_stage['finite']))
            if not next_stage['finite']:
                self.failed_transition = self.transitions[-1]
                self._write_state()
                raise RuntimeError('feedback curriculum next stage alpha=%g is non-finite'
                                   % next_strength)
            self.stage_epochs = 0
            self.stage_entry_target = restored_target['rms']
            self.stage_best_loss = next_stage['rms']
            self.stage_best_target = restored_target['rms']
            self.stage_best_path = restored
            returned_target = restored_target['rms']

        self.calls += 1
        self._write_state()
        print('    [feedback] epoch %d alpha %.1f stage %.3e target(alpha=0) %.3e%s'
              % (epoch, strength, stage['rms'], target['rms'],
                 '' if transition_reason is None else ' release=' + transition_reason))
        return float(returned_target)


def local_dynamics_diagnostic(fit_sys, batch, dt, n_points=9,
                              absorber_band=(120.0, 180.0), max_abs_z_limit=1.00001):
    """Linearize the learned autonomous one-step map on validation-window states.

    Points are stratified across all four validation records, rather than repeated samples from
    the nearly fixed V1 standstill record. Reported modal residues screen out a pole pair that
    exists algebraically but is disconnected from input/output.
    """
    with torch.no_grad():
        x_all = fit_sys.encoder(batch.uhist, batch.yhist)
    if int(n_points) < len(batch.record_counts):
        raise ValueError('n_points must cover every validation record')
    offsets = np.cumsum((0,) + tuple(int(n) for n in batch.record_counts))
    # Allocate checks as evenly as possible, with any remainder assigned to the records with the
    # largest encoded-Y range. This guarantees record coverage while preferentially sampling the
    # records that actually exercise the LPV scheduling coordinate.
    allocation = np.full(len(batch.record_counts), int(n_points) // len(batch.record_counts), int)
    remainder = int(n_points) - int(allocation.sum())
    y_ranges = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        y_record = x_all[int(start):int(stop), 2]
        y_ranges.append(float(torch.max(y_record) - torch.min(y_record)))
    for record_ix in np.argsort(y_ranges)[::-1][:remainder]:
        allocation[int(record_ix)] += 1
    row_record = {}
    for record_ix, (start, stop, count) in enumerate(
            zip(offsets[:-1], offsets[1:], allocation)):
        local = torch.argsort(x_all[int(start):int(stop), 2])
        pick = torch.linspace(0, len(local) - 1, int(count)).round().long()
        for row in (local[pick] + int(start)).tolist():
            row_record[int(row)] = int(record_ix)
    rows = sorted(row_record)

    points = []
    max_abs_all = 0.0
    any_absorber = False
    for row in rows:
        x = x_all[row].detach().clone().requires_grad_(True)
        u = batch.ufuture[row, 0].detach().clone().requires_grad_(True)

        def state_from_x(xv):
            return fit_sys.hfn(xv[None], u.detach()[None])[1][0]

        def state_from_u(uv):
            return fit_sys.hfn(x.detach()[None], uv[None])[1][0]

        def output_from_x(xv):
            return fit_sys.hfn.output_only(xv[None])[0]

        A = torch.autograd.functional.jacobian(state_from_x, x).detach().cpu().double().numpy()
        B = torch.autograd.functional.jacobian(state_from_u, u).detach().cpu().double().numpy()
        C = torch.autograd.functional.jacobian(output_from_x, x).detach().cpu().double().numpy()
        vals, left, right = eig(A, left=True, right=True)
        max_abs = float(np.max(np.abs(vals)))
        max_abs_all = max(max_abs_all, max_abs)
        modes = []
        for i, pole in enumerate(vals):
            angle = float(np.angle(pole))
            if angle <= 1e-10:
                continue
            radius = float(abs(pole))
            freq = angle / (2 * np.pi * dt)
            log_r = math.log(max(radius, 1e-300))
            zeta = -log_r / math.sqrt(log_r * log_r + angle * angle)
            denom = np.vdot(left[:, i], right[:, i])
            if abs(denom) < 1e-30:
                residue = float('inf')
            else:
                residue_matrix = ((C @ right[:, i:i + 1])
                                  @ (np.conjugate(left[:, i:i + 1]).T @ B)) / denom
                residue = float(np.linalg.norm(residue_matrix, ord='fro'))
            in_band = absorber_band[0] <= freq <= absorber_band[1]
            any_absorber = any_absorber or in_band
            modes.append(dict(frequency_hz=float(freq), damping_ratio=float(zeta),
                              radius=radius, residue_fro=residue,
                              in_absorber_band=bool(in_band)))
        points.append(dict(
            batch_row=int(row), validation_record=row_record[int(row)],
            encoded_y=float(x_all[row, 2]), max_abs_z=max_abs,
            stable=bool(max_abs <= max_abs_z_limit), modes=modes))

    absorber_modes = [m for p in points for m in p['modes'] if m['in_absorber_band']]
    return dict(
        max_abs_z=max_abs_all,
        max_abs_z_limit=float(max_abs_z_limit),
        stable_all_points=bool(max_abs_all <= max_abs_z_limit),
        absorber_mode_present=bool(any_absorber),
        absorber_mode_count=len(absorber_modes),
        absorber_modes=absorber_modes,
        record_point_counts=[int(n) for n in allocation],
        encoded_y_range_by_record=y_ranges,
        points=points,
    )
