"""Jan's interconnect of blocks and his encoder fit system, with the augmented model of the thesis."""
# CHANGED: restricted to the thesis training path; full implementation in
# model_augmentation/fit_systems/interconnect.py at the repository root
from deepSI.fit_systems.encoders import SS_encoder_general

from torch import nn, Tensor
import torch
import numpy as np
# CHANGED: os, time, math, Dataset and the deepSI imports below serve the added fit loop, the epoch index of the
# OBC refresh and the checkpoint saving
import os
import time
import math
from torch.utils.data import Dataset

from deepSI.datasets import get_work_dirs
from deepSI.fit_systems.fit_system import print_array_byte_size, My_Simple_DataLoader, loop_dataset

# CHANGED: explicit names instead of the star imports: the @added marker, the blocks the added code looks for,
# and the normaliser of the training windows
from model_augmentation.utils.utils import added
from model_augmentation.fit_systems.blocks import Block, Static_ANN_Block
from model_augmentation.utils.deepSI_corrections import SystemDataNormalizer

class Interconnect(nn.Module):
    """Blocks joined by signal connection matrices into one state-space step (x, u) -> (y, x_next)."""
    def __init__(self, nx, nu, ny) -> None:
        super().__init__()

        self.nu = nu
        self.ny = ny
        self.nx = nx

        self.nr_blocks = 0
        self.connected_blocks = nn.ModuleList([])

        self.input_signal_sizes = [nx, nu]
        self.output_signal_sizes = [nx, ny]

        self.signal_connections = []

        self.initialized_forward_function = False
        # CHANGED: the OBC correction f_obc^n, set by attach_obc on the OBC arm, else None
        self.obc_correction = None

    @added
    def _apply(self, fn, *args, **kwargs):
        """nn.Module._apply that also moves array_connection_matrices, a plain list, with the module."""
        out = super()._apply(fn, *args, **kwargs)
        acm = getattr(self, 'array_connection_matrices', None)
        if acm is not None:
            self.array_connection_matrices = [[fn(t) for t in row] for row in acm]
        return out

    def forward(self, x: Tensor, u: Tensor, obc_pass=None, pass_mats=None):
        # CHANGED: obc_pass and pass_mats, built once per objective, are arguments, so torch.compile sees them
        # as graph inputs
        x = x.view(x.size(0), self.nx, 1)
        u = u.view(u.size(0), self.nu, 1)

        assert x.size(0) == u.size(0)
        nb = x.size(0)

        if not self.initialized_forward_function:
            self.init_forward()
            self.initialized_forward_function = True

        # CHANGED: the signal loop is _propagate, shared with output_only, on the device and dtype of x
        output_signals = self._propagate(x, u, self.order_output_signal_computation, pass_mats)

        y = output_signals[1]
        xp = output_signals[0]

        # CHANGED: subtract f_obc^n, the part of the network along the baseline sensitivity and offset
        # directions, so the network cannot take over what the baseline parameters explain
        if self.obc_correction is not None:
            xp = xp - self.obc_correction(x, u, obc_pass)

        xp = xp.view(nb, self.nx)
        y = y.view(nb, self.ny)

        return y, xp

    @added
    def _propagate(self, x, u, order, pass_mats):
        """The output signals after computing those in order (forward: all of them, output_only: the output's
        dependency cone); each block input is written before any block reads it."""
        nb = x.size(0)
        input_signals = [x, u] + [None] * (self.n_input_signals - 2)
        output_signals = []
        for ix in range(0, self.n_output_signals):
            output_signals.append(torch.zeros((nb, self.output_signal_sizes[ix], 1), device=x.device, dtype=x.dtype))

        for output_signal_ix in order:
            for input_signal_ix in self.output_ix_sorted_input_ix_dependencies[output_signal_ix]:
                output_signals[output_signal_ix] += torch.matmul(self.array_connection_matrices[output_signal_ix][input_signal_ix], input_signals[input_signal_ix])

            if output_signal_ix >= 2:
                # the physics block takes the matrices built once per objective
                block = self.connected_blocks[output_signal_ix - 2]
                if output_signal_ix - 2 == self._pass_mats_ix:
                    input_signals[output_signal_ix] = block.forward(output_signals[output_signal_ix], pass_mats)
                else:
                    input_signals[output_signal_ix] = block.forward(output_signals[output_signal_ix])
        return output_signals

    def init_forward(self):
        self.n_output_signals = self.nr_blocks+2
        self.n_input_signals = self.nr_blocks+2
        n_nodes = 4 + self.nr_blocks

        self.array_connection_matrices = [[torch.empty((0,0)) for i in range(self.n_output_signals)] for j in range(self.n_input_signals)]
        self.output_ix_sorted_input_ix_dependencies = []

        directional_signal_connection_matrix = np.zeros((n_nodes, n_nodes))
        output_ix_sorted_signal_connections = [[] for i in range(self.n_output_signals)]

        for signal_connection in self.signal_connections:
            input_signal_ix = signal_connection.input_signal_ix
            output_signal_ix = signal_connection.output_signal_ix

            shifted_output_signal_ix = output_signal_ix
            if shifted_output_signal_ix <= 1: shifted_output_signal_ix += n_nodes - 2
            directional_signal_connection_matrix[shifted_output_signal_ix, input_signal_ix] = 1

            output_ix_sorted_signal_connections[output_signal_ix].append(signal_connection)

        connection_interconnect_matrix = np.roll(directional_signal_connection_matrix[2:,:-2], 2, axis=0)

        # order of the block computations; x and u are available from the start
        self.order_output_signal_computation = []
        connection_interconnect_matrix[:,:2] = 0

        while len(self.order_output_signal_computation) < self.n_output_signals:
            computable_elements = np.argwhere(np.sum(connection_interconnect_matrix, axis=1)==0).flatten()
            for element in computable_elements:
                if element not in self.order_output_signal_computation:
                    # CHANGED: Python ints, so torch.compile branches on them without a graph break
                    self.order_output_signal_computation.append(int(element))
                    connection_interconnect_matrix[:,element] = 0

        for output_signal_ix in range(self.n_output_signals):
            input_ix_dependencies = self.init_connection_matrices(output_signal_ix, output_ix_sorted_signal_connections[output_signal_ix])
            self.output_ix_sorted_input_ix_dependencies.append(input_ix_dependencies)

        # CHANGED: the output's dependency cone, resolved once for output_only
        self.output_cone = self._signal_cone(1)
        # CHANGED: the block that takes the per-objective matrices pass_mats, resolved once
        self._pass_mats_ix = next(
            (i for i, b in enumerate(self.connected_blocks) if hasattr(b, 'pass_mats')), None)

        # CHANGED: the connection matrices on the module's device and dtype
        p = next(self.parameters())
        self.array_connection_matrices = [[t.to(device=p.device, dtype=p.dtype) for t in row]
                                          for row in self.array_connection_matrices]

    @added
    def _signal_cone(self, target_output_ix):
        """Output-signal indices that must be computed to obtain output signal target_output_ix."""
        needed, stack = set(), [target_output_ix]
        while stack:
            k = stack.pop()
            if k in needed:
                continue
            needed.add(k)
            for j in self.output_ix_sorted_input_ix_dependencies[k]:
                if j >= 2 and j not in needed:
                    stack.append(j)
        return tuple(k for k in self.order_output_signal_computation if k in needed)

    @added
    def output_only(self, x):
        """y from the output signal's dependency cone alone, without advancing the state."""
        if len(x.size()) <= 2:
            x = x.view(x.size(0), self.nx, 1)
        nb = x.size(0)
        u = torch.zeros((nb, self.nu, 1), device=x.device, dtype=x.dtype)

        if not self.initialized_forward_function:
            self.init_forward()
            self.initialized_forward_function = True

        y = self._propagate(x, u, self.output_cone, None)[1]
        return y.view(nb, self.ny)

    def init_connection_matrices(self, output_signal_ix, signal_connections):
        n_out_total = 0

        concat_signal_connections = []
        additive_signal_connections = []

        input_signal_ixs = []

        for signal_connection in signal_connections:
            if signal_connection.connection_function_method == "concatenation":
                concat_signal_connections.append(signal_connection)
            if signal_connection.connection_function_method == "additive":
                additive_signal_connections.append(signal_connection)

        if len(concat_signal_connections) == 0 and len(additive_signal_connections) != 0:
            concat_signal_connections.append(additive_signal_connections.pop(0))

        for signal_connection in concat_signal_connections:
            n_in = self.input_signal_sizes[signal_connection.input_signal_ix]

            connection_matrix = signal_connection.connection_matrix
            if not connection_matrix.numel(): connection_matrix = torch.eye(n_in)
            n_out_contribution = connection_matrix.size(0)
            if n_out_total > 0: connection_matrix = torch.vstack((torch.zeros((n_out_total, n_in)), connection_matrix))
            self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix] = connection_matrix

            n_out_total += n_out_contribution
            input_signal_ixs.append(signal_connection.input_signal_ix)

        for signal_connection in concat_signal_connections:
            connection_matrix = self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix]
            n_out_current = connection_matrix.size(0)
            n_in = self.input_signal_sizes[signal_connection.input_signal_ix]
            self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix] = torch.vstack((connection_matrix, torch.zeros((n_out_total - n_out_current, n_in))))

        for signal_connection in additive_signal_connections:
            self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix] = signal_connection.connection_matrix
            input_signal_ixs.append(signal_connection.input_signal_ix)

        return input_signal_ixs

    def add_block(self, new_block: Block):
        """Add a block; its output and input get the next signal index."""
        self.nr_blocks += 1
        new_block.block_ix = self.nr_blocks + 1
        self.connected_blocks.append(new_block)

        self.input_signal_sizes.append(new_block.nw)
        self.output_signal_sizes.append(new_block.nz)

    def connect_signals(self, input_signal, output_signal, connection_function_method=None, connection_matrix=torch.empty(0,0)):
        """Connect an input signal to an output signal, by concatenation into a block or by addition into
        x_next and y."""
        input_signal_ix = self.determine_signal_ix(input_signal)
        output_signal_ix = self.determine_signal_ix(output_signal)

        if not isinstance(connection_function_method, str) and output_signal_ix >= 2: connection_function_method = "concatenation"
        if not isinstance(connection_function_method, str) and output_signal_ix <= 1: connection_function_method = "additive"
        connection_function_method = self.parse_connection_function_method(connection_function_method)

        self.signal_connections.append(Signal_Connection(input_signal_ix, output_signal_ix, connection_function_method=connection_function_method, \
                                        connection_matrix=connection_matrix))

    def connect_block_signals(self, block, input_signal_list: list, output_signal_list: list):
        """Connect every signal of input_signal_list into block, and block into every signal of output_signal_list."""
        for input_signal in input_signal_list:
            self.connect_signals(input_signal, block)

        for output_signal in output_signal_list:
            self.connect_signals(block, output_signal)

    def determine_signal_ix(self, signal):
        if isinstance(signal, Block):
            return signal.block_ix
        if isinstance(signal, str):
            return self.convert_signal_name_to_ix(signal)

    def convert_signal_name_to_ix(self, signal_name: str):
        if signal_name in ["x", "xp"]:
            return 0
        if signal_name in ["u", "y"]:
            return 1

    def parse_connection_function_method(self, connection_function_method: str):
        return {"concatenation": "concatenation", "concat": "concatenation",
                "additive": "additive"}[connection_function_method]

class Signal_Connection():
    '''Object to hold information regarding connection between two signals in the interconnect'''
    def __init__(self, input_signal_ix, output_signal_ix, connection_function_method, connection_matrix) -> None:
        self.input_signal_ix = input_signal_ix
        self.output_signal_ix = output_signal_ix

        self.connection_function_method = connection_function_method
        self.connection_matrix = connection_matrix


@added
def _replace_with_retry(src, dst, tries=20):
    """os.replace, retried while another process (a sync client, a virus scanner) briefly holds dst open, which
    Windows reports as PermissionError."""
    for i in range(tries):
        try:
            return os.replace(src, dst)
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.25)


@added
class DtypeDataLoader(My_Simple_DataLoader):
    """My_Simple_DataLoader that keeps the training windows at the model's dtype (the original casts to float32)."""

    def __init__(self, data, batch_size, dtype):
        def _conv(d):
            t = torch.as_tensor(d)
            return t.to(dtype) if t.is_floating_point() else t
        self.data = [_conv(d) for d in data]
        self.ids = np.arange(len(data[0]), dtype=int)
        # the batch order of My_Simple_DataLoader: a generator fixed at seed 0
        self.rng = np.random.default_rng(0)
        self.batch_size = batch_size


class SSE_Interconnect(SS_encoder_general):
    """Jan's fit system: an encoder and an interconnect, trained on deepSI's simulation windows."""
    # CHANGED: the rollout of the loss and the validation, set by build_closed_loop (the closed loop)
    simulator = None

    # CHANGED: callables run after every validation (the validation history)
    validation_probes = ()

    # CHANGED: a callable run after every Adam update (the PS2 opening), or None
    training_phase = None

    def __init__(self, interconnect, na=5, nb=5, na_right=0, nb_right=0):
        super(SSE_Interconnect, self).__init__(nx=interconnect.nx, nb=nb, na=na, na_right=na_right, nb_right=nb_right)

        self.hfn = interconnect

        # CHANGED: a normaliser whose windows keep the data's dtype; build_model sets its values
        self.norm = SystemDataNormalizer()

    def init_model(self, sys_data=None, optimizer_kwargs={}):
        '''Set nu and ny, build the optimizer over every parameter and reset bestfit.'''
        self.nu, self.ny = sys_data.nu, sys_data.ny
        self.optimizer = self.init_optimizer(list(self.parameters_with_names.values()), **optimizer_kwargs)
        self.bestfit = float('inf')
        self.init_model_done = True

    @added
    def simulate(self, x, ufuture, yfuture=None, **Loss_kwargs):
        """Roll the model forward from x through the attached simulator; returns (y_pred, x_final), y_pred
        (batch, nf, ny)."""
        return self.simulator(self, x, ufuture, yfuture, **Loss_kwargs)

    @added
    def make_training_data(self, sys_data, **Loss_kwargs):
        """deepSI's training windows plus the arrays the attached simulator appends per window."""
        data = super().make_training_data(sys_data, **Loss_kwargs)
        return self.simulator.augment_training_data(data, sys_data, self, **Loss_kwargs)

    @added
    def cal_validation_error(self, val_sys_data, validation_measure='sim-NRMS'):
        """The simulator's validation score, then the validation probes (side effects only)."""
        value = self.simulator.validation_error(self, val_sys_data)
        for probe in self.validation_probes:
            probe(self, val_sys_data, value)
        return value

    @added
    def fit(self, train_sys_data, val_sys_data, epochs=30, n_its=None, batch_size=256, loss_kwargs={}, \
            validation_measure='sim-NRMS', its_per_val='epoch', cuda=False):
        """deepSI's batch loop: Adam updates with a validation every its_per_val updates, the best model dumped as
        _best and the last as _last, then the best reloaded."""

        def validation(train_loss=None, time_elapsed_total=None):
            self.eval()
            self._sync_obc_coefficient(refresh_basis=False)
            Loss_val = self.cal_validation_error(val_sys_data, validation_measure=validation_measure)
            self.Loss_val.append(Loss_val)

            self.Loss_train.append(train_loss)
            self.time.append(time_elapsed_total)
            self.batch_id.append(self.batch_counter)
            self.epoch_id.append(self.epoch_counter)
            if self.bestfit>=Loss_val:
                self.bestfit = Loss_val
                # the dump holds the tensors on the training device; resolve_checkpoint maps them to the CPU
                self.checkpoint_save_system()
            # _last.pth at every validation, so a stopped run can resume from its last weights and Adam moments
            # (not from its batch stream or an unfinished PS2 opening)
            self.checkpoint_save_system(name='_last')
            self.train()
            return Loss_val

        # training windows, data loader and the initial validation
        self.train()

        self.epoch_counter = 0
        self.batch_counter = 0

        data_train = self.make_training_data(self.norm.transform(train_sys_data), **loss_kwargs)
        if not isinstance(data_train, Dataset): print_array_byte_size(sum([d.nbytes for d in data_train]))

        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = [], [], [], [], []

        Loss_acc_val_loop, best_it = 0, 0
        N_training_samples = len(data_train) if isinstance(data_train, Dataset) else len(data_train[0])
        batch_size = min(batch_size, N_training_samples)
        N_batch_updates_per_epoch = N_training_samples//batch_size
        n_its = int(N_batch_updates_per_epoch*epochs) if n_its is None else n_its
        its_per_val = N_batch_updates_per_epoch if its_per_val=='epoch' else its_per_val
        print(f'N_training_samples = {N_training_samples}, batch_size = {batch_size}, N_batch_updates_per_epoch = {N_batch_updates_per_epoch}')

        # DtypeDataLoader, since My_Simple_DataLoader casts the windows to float32
        model_dtype = next(self.hfn.parameters()).dtype
        data_train_loader = DtypeDataLoader(data_train, batch_size=batch_size,
                                            dtype=model_dtype)

        validation(train_loss=float('nan'), time_elapsed_total=0)
        print(f'Initial Validation {validation_measure}=', self.Loss_val[-1])

        # Adam
        try:
            start_t = time.time()

            bestfit_old = self.bestfit
            for it_count, train_batch in zip(range(n_its), loop_dataset(data_train_loader)):
                if cuda:
                    train_batch = [b.cuda() for b in train_batch]
                # the simulator's extra arrays reach loss() by name
                batch_kwargs = dict(loss_kwargs, **dict(zip(self.simulator.extra_array_names, train_batch[4:])))

                def closure():
                    Loss = self.loss(*train_batch[:4], **batch_kwargs)
                    self.optimizer.zero_grad()
                    Loss.backward()
                    return Loss

                training_loss = self.optimizer.step(closure).item()

                if np.isnan(training_loss):
                    print(f'&&&&&&&&&&&&& Encountered a NaN value in the training loss at it {it_count}, breaking from loop &&&&&&&&&&')
                    break
                # the PS2 opening after every update, and how far it got at these weights, for the dumps
                if self.training_phase is not None:
                    self.training_phase.after_step(self)
                    self.ps2_opening = (self.training_phase.phase, self.training_phase.it)

                Loss_acc_val_loop += training_loss
                self.batch_counter += 1
                self.epoch_counter += 1/N_batch_updates_per_epoch

                if (it_count+1)%its_per_val==0:
                    train_loss = Loss_acc_val_loop/its_per_val
                    validation(train_loss=train_loss, time_elapsed_total=time.time()-start_t)
                    Loss_acc_val_loop = 0.
                    if bestfit_old > self.bestfit:
                        print(f'########## New lowest validation loss achieved ########### {validation_measure} = {self.bestfit}')
                        best_it = it_count+1
                        bestfit_old = self.bestfit
                    trainstr = f'sqrt loss {train_loss**0.5:7.4}' if train_loss>=0 else f'loss {train_loss:7.4}'
                    Loss_str = f'It {it_count+1:4}, {trainstr}, Val {validation_measure} {self.Loss_val[-1]:6.4}'
                    print(Loss_str)
        except KeyboardInterrupt:
            print('Stopping early due to a KeyboardInterrupt')

        # last dump, then the best reloaded; on the OBC arm the basis is rebuilt at the reloaded parameters and
        # the selected dump rewritten with it
        self.train(); self.cpu()
        del data_train_loader

        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array(self.Loss_val), np.array(self.Loss_train), np.array(self.batch_id), np.array(self.time), np.array(self.epoch_id)
        self.checkpoint_save_system(name='_last')
        selected_checkpoint = '_best'
        try:
            self.checkpoint_load_system(name='_best')
        except FileNotFoundError:
            print('no best checkpoint found keeping last')
            selected_checkpoint = '_last'
        if self._sync_obc_coefficient(refresh_basis=True):
            self.checkpoint_save_system(name=selected_checkpoint)
        print(f'Loaded model with best known validation {validation_measure} of {self.bestfit:6.4} which happened on epoch {best_it} (epoch_id={self.epoch_id[-1]:.2f})')


@added
class AugmentedModel(SSE_Interconnect):
    """The augmented model of the thesis: the W_r-weighted closed-loop objective, the OBC coefficient per
    objective, and dumps without the training-time state."""

    _optimizer = None
    obc = None
    # training-time state: left out of every dump, kept across a dump load
    _UNPICKLABLE = ('obc', 'obc_reference', 'training_phase')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        blocks = tuple(self.hfn.connected_blocks)

        # the network block and the block that takes pass_mats
        self._ann_ix = next((i for i, m in enumerate(blocks)
                             if isinstance(m, Static_ANN_Block)), None)
        self._mats_ix = next((i for i, m in enumerate(blocks) if hasattr(m, 'pass_mats')), None)

    def _epoch_index(self):
        """The integer epoch the OBC refresh counts in."""
        # + 1e-6: the epoch counter sums 1 / N per update and can end just below an integer
        return int(math.floor(float(getattr(self, 'epoch_counter', 0.0)) + 1e-6))

    def _sync_obc_coefficient(self, refresh_basis=False):
        """Pair theta_frozen with the current network on the basis; refresh_basis, or a missing basis, rebuilds
        the basis first. Returns False without OBC."""
        if self.obc is None:
            return False
        correction = self.hfn.obc_correction
        epoch = self._epoch_index()
        with torch.no_grad():
            if refresh_basis or self.obc.basis is None:
                self.obc.refresh(correction.expansion_point(), epoch)
            theta = self.obc.coefficient(self.hfn.connected_blocks[self._ann_ix])
            correction.set_frozen(theta, self.obc.basis.vbar)
        return True

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        """The objective of a batch of windows: the mean of (W_r (y^n - y_hat^n))^2 over windows, steps and axes."""
        x = self.encoder(uhist, yhist)
        # the matrices of this objective: pass_mats, and on the OBC arm the tangent pair of theta_aux
        Loss_kwargs = dict(
            Loss_kwargs,
            pass_mats=self.hfn.connected_blocks[self._mats_ix].pass_mats())
        if self.obc is not None:
            correction = self.hfn.obc_correction
            self.obc.maybe_refresh(self._epoch_index(), correction.expansion_point)
            # theta_aux of this objective, differentiable in the network; the tangent pair after set_frozen,
            # which updates vbar
            theta_aux = self.obc.coefficient(self.hfn.connected_blocks[self._ann_ix])
            correction.set_frozen(theta_aux, self.obc.basis.vbar)
            Loss_kwargs = dict(Loss_kwargs, obc_pass=correction.tangent_pair(theta_aux))
        # loss_w (windows, ny): W_r of each window's record, used by the objective, not by the rollout
        w = Loss_kwargs.pop('loss_w')
        y_pred, _ = self.simulate(x, ufuture, yfuture, **Loss_kwargs)
        # THEORY: w = sigma_y / sigma_e turns the normalised error into the error over sigma_e, per record and axis
        return ((y_pred - yfuture) * w[:, None, :]).square().mean()

    def checkpoint_save_system(self, name='_best'):
        """deepSI's dump of the model, written atomically."""
        # deepSI's save into <file>.tmp, renamed into place once complete, so a process killed while saving keeps
        # the old dump; the dump is __getstate__, without the training-time state
        file = os.path.join(get_work_dirs()['checkpoints'], self.name + name + '.pth')
        torch.save(self.__getstate__(), file + '.tmp')
        _replace_with_retry(file + '.tmp', file)

    def checkpoint_load_system(self, *args, **kwargs):
        """deepSI's dump load, keeping the training-time state of this process."""
        stash = {k: self.__dict__.get(k) for k in self._UNPICKLABLE}
        out = super().checkpoint_load_system(*args, **kwargs)
        for k, v in stash.items():
            if v is not None:
                self.__dict__[k] = v
        return out

    def __getstate__(self):
        """The state without the training-time keys, as every dump holds it."""
        return {k: v for k, v in self.__dict__.items() if k not in self._UNPICKLABLE}

    # deepSI stores the optimizer as _optimizer in the dumps; the property keeps that attribute
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
