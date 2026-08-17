"""STEP 4: the closed-loop training loss, on the SAME rollout validation and selection use.

WHAT THIS REPLACES
------------------
`loss_variants.ClosedLoopLossMixin` (variant B). That run was invalid, not a negative result: it
optimised a closed-loop objective while selection and scoring ran an open-loop free run. It also
had two defects this does not: it called `hfn` TWICE per step to work around the `y = h(x)`
ordering, and it built ONE controller from a single `Y_op` and applied it to every record, which is
wrong for every record whose operating point differs (D-140 measured nine distinct controllers,
with `kappa` varying 1.5x on X1/X2 across the range).

THE THREE THINGS THAT MAKE THIS DIFFERENT
-----------------------------------------
1. **One rollout.** `cl_controller.rollout` is called here, by `cl_validation.ClosedLoopValidator`,
   and by the baselines. Training, validation and checkpoint selection cannot disagree about what
   the loop is, because there is only one implementation of it.
2. **Per-record `Cfb`.** `make_training_data` returns a fifth array, `rec_ix`. deepSI's
   `fit_system.py:393` calls `self.loss(*train_batch)` and `My_Simple_DataLoader` slices every
   array in the list by the same shuffled ids, so the index arrives in `loss` correctly shuffled
   and batched. The window-count-per-record derivation is ASSERTED against the real call rather
   than trusted (see `_record_index`): a silent misalignment would attach the wrong controller to
   most windows and would look like a training problem rather than a bookkeeping one.
3. **`xc = 0` at each window start, which is Kessels' Remark 5.4** (D-142, verified verbatim
   against thesis p157). Not a shortcut: `u_data` already carries the machine's controller history,
   so `xc_A = 0` makes the model's effective controller state exactly the reconstructed `x̄^FB(τ)`.
   Measured cost 8.5-13.4 % of the model error, against 77x available headroom.

WHAT IS DELIBERATELY UNCHANGED
------------------------------
`param_loss` and the orthogonality penalty are carried through untouched. They are easy to drop by
accident when replacing a loss, and dropping `orth_penalty` would quietly delete the thesis
contribution from the objective. G11 checks the whole loss reduces to the production one when the
loop is disabled, which catches that and anything else that changed by accident.
"""
__project_origin__ = "added"

import numpy as np
import torch

from cl_controller import rollout


def _record_index(sys_data, na, nb, nf, na_right, nb_right, stride):
    """rec_ix per window, aligned to System_data_list.to_hist_future_data's concatenation order.

    `System_data_list.to_hist_future_data` (system_data.py:679-681) maps each record through
    `to_hist_future_data` and concatenates the results in `sdl` order, so record identity is
    recoverable from per-record window counts. The count is derived here and then ASSERTED against
    the real call by the caller, because deriving it independently is exactly the kind of thing
    that breaks silently on a change to stride, na, nb or nf. That assert has already earned its
    keep once: the first version of this function subtracted the right-hand encoder extension and
    came out one window short per record, which would have shifted the controller assignment by one
    record for most of the training set.
    The count is NOT guessed. deepSI has two branches (system_data.py:305-329) and they agree:

      stride == 1  uses sliding_window_view over u[npast:], npast = max(na, nb), giving
                   `len(u) - npast - nf + 1` windows. Note this branch IGNORES na_right and
                   nb_right entirely.
      stride != 1  loops `for k in range(k0 + k0_right, len(u) + 1, stride)` with k0 = max(na, nb)
                   and k0_right = max(nf, na_right, nb_right).

    The second expression reproduces the first at stride 1 (k0_right = nf when nf >= na_right,
    nb_right, which holds for every configuration this pipeline uses), so one formula covers both.
    It is written as an actual `range` so it cannot drift from deepSI's own loop.
    """
    sdl = sys_data.sdl if hasattr(sys_data, 'sdl') else [sys_data]
    k0 = max(na, nb)
    k0_right = max(nf, na_right, nb_right)
    counts = [len(range(k0 + k0_right, len(sd.u) + 1, stride)) for sd in sdl]
    return np.concatenate([np.full(c, i, dtype=np.int64) for i, c in enumerate(counts)]), counts


class ClosedLoopLoss:
    """Mixin: closed-loop rollout as the training objective, per-record Cfb, xc = 0 per window.

    Grafted onto the fit-system INSTANCE by `attach`, exactly as `loss_variants.attach` does, so
    optimiser, encoder, normalisation, shooting structure and every default stay the production
    ones and only `loss` and `make_training_data` differ.
    """

    def make_training_data(self, sys_data, **kw):
        data = super().make_training_data(sys_data, **kw)
        nf = kw.get('nf', 25)
        stride = kw.get('stride', 1)
        na_r = getattr(self, 'na_right', 0)
        nb_r = getattr(self, 'nb_right', 0)
        rec_ix, counts = _record_index(sys_data, self.na, self.nb, nf, na_r, nb_r, stride)
        n_real = len(data[0])
        if len(rec_ix) != n_real:
            raise RuntimeError(
                'rec_ix misalignment: derived %d windows from per-record counts %s but '
                'to_hist_future_data produced %d. The controller would be attached to the wrong '
                'records. Fix _record_index rather than truncating.'
                % (len(rec_ix), counts, n_real))
        self._cl_counts = counts
        return list(data) + [rec_ix]

    def loss(self, uhist, yhist, ufuture, yfuture, rec_ix=None, **kw):
        if rec_ix is None:
            raise RuntimeError('closed-loop loss called without rec_ix; make_training_data must '
                               'supply it (attach() installs both together)')
        x = self.encoder(uhist, yhist)
        ctrl = self._cl_bank.gather(rec_ix.long())
        y_pred, _, _ = rollout(self._cl_step_fn, self._cl_out_fn, ufuture, yfuture, x,
                               self._cl_bank, ctrl)          # xc = 0: Remark 5.4, D-142
        L = torch.nn.functional.mse_loss(y_pred, yfuture)
        self.last_mse = float(L.detach())
        for m in self.hfn.connected_blocks:
            if hasattr(m, 'param_loss'):
                L = L + m.param_loss()
        if self.orth_penalty is not None and self.orth_penalty.beta != 0.0:
            from model_augmentation.fit_systems.blocks import Static_ANN_Block
            ann = next(m for m in self.hfn.connected_blocks
                       if isinstance(m, Static_ANN_Block))
            L = L + self.orth_penalty(ann)
        return L

    def loss_open_loop(self, uhist, yhist, ufuture, yfuture, rec_ix=None, **kw):
        """The SAME loss with the loop disabled. G11 requires this to equal the production loss.

        If this and `super().loss` differ, the closed-loop path changed something other than the
        loop (normalisation, reduction, window handling) and any comparison between them is
        meaningless.
        """
        from cl_controller import open_loop_rollout
        x = self.encoder(uhist, yhist)
        y_pred, _ = open_loop_rollout(self._cl_step_fn, self._cl_out_fn, ufuture, x)
        L = torch.nn.functional.mse_loss(y_pred, yfuture)
        for m in self.hfn.connected_blocks:
            if hasattr(m, 'param_loss'):
                L = L + m.param_loss()
        if self.orth_penalty is not None and self.orth_penalty.beta != 0.0:
            from model_augmentation.fit_systems.blocks import Static_ANN_Block
            ann = next(m for m in self.hfn.connected_blocks
                       if isinstance(m, Static_ANN_Block))
            L = L + self.orth_penalty(ann)
        return L


def attach(fit_sys, bank, step_fn, out_fn):
    """Graft the closed-loop loss onto the instance. Returns fit_sys.

    Class binding follows loss_variants.attach's fix: `type()` sets __module__ but does NOT bind
    the name, so pickle's lookup fails and `evaluate_and_save`'s checkpoint save dies AFTER
    training completes. Bind it, and reuse the same object on repeat calls.
    """
    cls_name = 'FitSys_ClosedLoop'
    new_cls = globals().get(cls_name)
    if new_cls is None or new_cls.__bases__ != (ClosedLoopLoss, type(fit_sys)):
        new_cls = type(cls_name, (ClosedLoopLoss, type(fit_sys)), {})
        new_cls.__module__ = __name__
        globals()[cls_name] = new_cls
    fit_sys.__class__ = new_cls
    fit_sys._cl_bank = bank
    fit_sys._cl_step_fn = step_fn
    fit_sys._cl_out_fn = out_fn
    return fit_sys
