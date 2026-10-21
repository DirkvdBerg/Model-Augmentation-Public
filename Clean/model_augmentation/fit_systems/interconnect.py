from deepSI.fit_systems.encoders import SS_encoder_general

from torch import nn, Tensor
import torch
import numpy as np
import os
import time
import math                     # CHANGED (D-192): epoch index for the OBC refresh
from torch.utils.data import Dataset

from deepSI.datasets import get_work_dirs
from deepSI.fit_systems.fit_system import print_array_byte_size, My_Simple_DataLoader, loop_dataset

from model_augmentation.utils.utils import added  # CHANGED: marker for project-added classes (D-076)
from model_augmentation.fit_systems.blocks import Block, Static_ANN_Block  # CHANGED: Static_ANN_Block for SSE_Interconnect_Composed's orth term
from model_augmentation.utils.deepSI_corrections import fixed_System_data_norm

class Interconnect(nn.Module):
    # CHANGED (clean): trimmed to the spec path
    def __init__(self, nx, nu, ny) -> None:
        # CHANGED (clean): trimmed to the spec path
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
        # CHANGED (D-192): the one-step OBC correction, an nn.Module the pipeline attaches
        # (gantry_dynamic/obc_gantry.py::GantryOBCCorrection) or None.
        self.obc_correction = None

    # CHANGED (D-169): make `array_connection_matrices` follow the module's device and dtype.
    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        acm = getattr(self, 'array_connection_matrices', None)
        if acm is not None:
            self.array_connection_matrices = [[fn(t) for t in row] for row in acm]
        return out

    def forward(self, x: Tensor, u: Tensor, obc_pass=None, pass_mats=None):
        # CHANGED (clean): trimmed to the spec path
        # CHANGED (D-194/D-195): `obc_pass` is the once-per-objective `(mats, dmats)` pair,
        # passed as a TENSOR-TUPLE ARGUMENT so its graph contract under
        # torch.compile is explicit (an input, not a mutated attribute).
        x = x.view(x.size(0), self.nx, 1)
        u = u.view(u.size(0), self.nu, 1)

        assert x.size(0) == u.size(0)
        nb = x.size(0)

        if not self.initialized_forward_function:
            self.init_forward()
            self.initialized_forward_function = True

        output_signals = self._propagate(x, u, self.order_output_signal_computation, pass_mats)

        y = output_signals[1]
        xp = output_signals[0]

        # CHANGED (D-192): one-step orthogonal-by-construction subtraction.
        if self.obc_correction is not None:
            xp = xp - self.obc_correction(x, u, obc_pass)

        xp = xp.view(nb, self.nx)
        y = y.view(nb, self.ny)

        return y, xp

    @added
    def _propagate(self, x, u, order, pass_mats):
        '''The output signals after computing those in `order` (forward: all, output_only: the
        output's dependency cone). Each block input is written before any block reads it.'''
        nb = x.size(0)
        input_signals = [x, u] + [None] * (self.n_input_signals - 2)
        output_signals = []
        for ix in range(0, self.n_output_signals):
            output_signals.append(torch.zeros((nb, self.output_signal_sizes[ix], 1), device=x.device, dtype=x.dtype))

        for output_signal_ix in order:
            for input_signal_ix in self.output_ix_sorted_input_ix_dependencies[output_signal_ix]:
                output_signals[output_signal_ix] += torch.matmul(self.array_connection_matrices[output_signal_ix][input_signal_ix], input_signals[input_signal_ix])

            if output_signal_ix >= 2:
                # CHANGED (D-195): hand the per-pass matrix structure to the ONE block that
                # declared it wants it.
                block = self.connected_blocks[output_signal_ix - 2]
                if output_signal_ix - 2 == self._pass_mats_ix:
                    input_signals[output_signal_ix] = block.forward(output_signals[output_signal_ix], pass_mats)
                else:
                    input_signals[output_signal_ix] = block.forward(output_signals[output_signal_ix])
        return output_signals

    def init_forward(self):
        # CHANGED (clean): trimmed to the spec path
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

        self.order_output_signal_computation = []
        connection_interconnect_matrix[:,:2] = 0

        while len(self.order_output_signal_computation) < self.n_output_signals:
            computable_elements = np.argwhere(np.sum(connection_interconnect_matrix, axis=1)==0).flatten()
            for element in computable_elements:
                if element not in self.order_output_signal_computation:
                    # CHANGED (D-169): int(), not the raw numpy scalar.
                    self.order_output_signal_computation.append(int(element))
                    connection_interconnect_matrix[:,element] = 0

        for output_signal_ix in range(self.n_output_signals):
            input_ix_dependencies = self.init_connection_matrices(output_signal_ix, output_ix_sorted_signal_connections[output_signal_ix])
            self.output_ix_sorted_input_ix_dependencies.append(input_ix_dependencies)

        # CHANGED (closed-loop seam): resolve the OUTPUT's dependency cone once, here, so
        # output_only() is a fixed short list rather than a graph traversal per call.
        self.output_cone = self._signal_cone(1)
        # CHANGED (D-195): which block accepts a hoisted per-pass matrix structure, resolved once.
        self._pass_mats_ix = next(
            (i for i, b in enumerate(self.connected_blocks) if hasattr(b, 'pass_mats')), None)

        # CHANGED (D-169): place the matrices on the module's device/dtype NOW.
        p = next(self.parameters())
        self.array_connection_matrices = [[t.to(device=p.device, dtype=p.dtype) for t in row]
                                          for row in self.array_connection_matrices]

    # CHANGED (closed-loop seam): added, see output_only().
    def _signal_cone(self, target_output_ix):
        '''Output-signal indices that must be computed to obtain output signal `target_output_ix`.'''
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

    # CHANGED (closed-loop seam): added.
    def output_only(self, x):
        # CHANGED (clean): trimmed to the spec path
        '''y from the output signal's dependency cone alone, without advancing the state.'''
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
        # CHANGED (clean): trimmed to the spec path
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
        # CHANGED (clean): trimmed to the spec path
        self.nr_blocks += 1
        new_block.block_ix = self.nr_blocks + 1
        self.connected_blocks.append(new_block)

        self.input_signal_sizes.append(new_block.nw)
        self.output_signal_sizes.append(new_block.nz)

    def connect_signals(self, input_signal, output_signal, connection_function_method=None, connection_matrix=torch.empty(0,0)):
        # CHANGED (clean): trimmed to the spec path
        input_signal_ix = self.determine_signal_ix(input_signal)
        output_signal_ix = self.determine_signal_ix(output_signal)

        if not isinstance(connection_function_method, str) and output_signal_ix >= 2: connection_function_method = "concatenation"
        if not isinstance(connection_function_method, str) and output_signal_ix <= 1: connection_function_method = "additive"
        connection_function_method = self.parse_connection_function_method(connection_function_method)

        self.signal_connections.append(Signal_Connection(input_signal_ix, output_signal_ix, connection_function_method=connection_function_method, \
                                        connection_matrix=connection_matrix))

    def connect_block_signals(self, block, input_signal_list: list, output_signal_list: list):
        for input_signal in input_signal_list:
            self.connect_signals(input_signal, block)

        for output_signal in output_signal_list:
            self.connect_signals(block, output_signal)

    def determine_signal_ix(self, signal):
        # CHANGED (clean): trimmed to the spec path
        if isinstance(signal, Block):
            return signal.block_ix
        if isinstance(signal, str):
            return self.convert_signal_name_to_ix(signal)

    def convert_signal_name_to_ix(self, signal_name: str):
        # CHANGED (clean): trimmed to the spec path
        if signal_name in ["x", "xp"]:
            return 0
        if signal_name in ["u", "y"]:
            return 1

    def parse_connection_function_method(self, connection_function_method: str):
        # CHANGED (clean): trimmed to the spec path
        return {"concatenation": "concatenation", "concat": "concatenation",
                "additive": "additive"}[connection_function_method]

class Signal_Connection():
    # CHANGED (clean): trimmed to the spec path
    '''Object to hold information regarding connection between two signals in the interconnect'''
    def __init__(self, input_signal_ix, output_signal_ix, connection_function_method, connection_matrix) -> None:
        self.input_signal_ix = input_signal_ix
        self.output_signal_ix = output_signal_ix

        self.connection_function_method = connection_function_method
        self.connection_matrix = connection_matrix


@added
def _replace(src, dst, tries=20):
    """os.replace, retried while another process (a sync client, a virus scanner) briefly holds dst
    open, which Windows reports as PermissionError."""
    for i in range(tries):
        try:
            return os.replace(src, dst)
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.25)


@added
class Dtype_DataLoader(My_Simple_DataLoader):
    """My_Simple_DataLoader that keeps the training arrays at the MODEL's dtype."""

    def __init__(self, data, batch_size, dtype):
        def _conv(d):
            t = torch.as_tensor(d)
            return t.to(dtype) if t.is_floating_point() else t
        self.data = [_conv(d) for d in data]
        self.ids = np.arange(len(data[0]), dtype=int)
        self.rng = np.random.default_rng(0)
        self.batch_size = batch_size


class SSE_Interconnect(SS_encoder_general):
    # CHANGED (clean): trimmed to the spec path
    # CHANGED (closed-loop seam): how the model is DRIVEN during a loss rollout is a value on the
    # instance, not a position in the ParamLoss -> OrthLoss -> MultipleShooting chain.
    simulator = None

    # CHANGED (closed-loop seam): declared extension point for DIAGNOSTICS during validation.
    validation_probes = ()

    training_phase = None

    def __init__(self, interconnect, na=5, nb=5, na_right=0, nb_right=0):
        # CHANGED (clean): trimmed to the spec path

        super(SSE_Interconnect, self).__init__(nx=interconnect.nx, nb=nb, na=na, na_right=na_right, nb_right=nb_right)

        self.hfn = interconnect

        self.norm = fixed_System_data_norm()

    def init_model(self, sys_data=None, optimizer_kwargs={}):
        # CHANGED (clean): trimmed to the spec path
        '''This function set the nu and ny, inits the network, moves parameters to device, initilizes optimizer and initilizes logging parameters'''
        self.nu, self.ny = sys_data.nu, sys_data.ny
        self.optimizer = self.init_optimizer(list(self.parameters_with_names.values()), **optimizer_kwargs)
        self.bestfit = float('inf')
        self.init_model_done = True

    # CHANGED (closed-loop seam): the rollout is extracted from loss() into an overridable method.
    def simulate(self, x, ufuture, yfuture=None, **Loss_kwargs):
        # CHANGED (clean): trimmed to the spec path
        '''Roll the model forward from x. Returns (y_pred, x_final), y_pred (batch, nf, ny).'''
        return self.simulator(self, x, ufuture, yfuture, **Loss_kwargs)

    # CHANGED (closed-loop seam): make_training_data and cal_validation_error become seams.
    def make_training_data(self, sys_data, **Loss_kwargs):
        # CHANGED (clean): trimmed to the spec path
        '''deepSI's training arrays, plus whatever the attached simulator needs per window.'''
        data = super().make_training_data(sys_data, **Loss_kwargs)
        return self.simulator.augment_training_data(data, sys_data, self, **Loss_kwargs)

    def cal_validation_error(self, val_sys_data, validation_measure='sim-NRMS'):
        # CHANGED (clean): trimmed to the spec path
        '''The scalar fit() minimises over epochs, plus side-effect-only diagnostic probes.'''
        value = self.simulator.validation_error(self, val_sys_data)
        for probe in self.validation_probes:
            probe(self, val_sys_data, value)
        return value

    # CHANGED: added fit() method replacing deepSI's original: supports multi-trajectory, sqrt loss, concurrent validation
    def fit(self, train_sys_data, val_sys_data, epochs=30, n_its=None, batch_size=256, loss_kwargs={}, \
            validation_measure='sim-NRMS', its_per_val='epoch', cuda=False):
        # CHANGED (clean): trimmed to the spec path
        '''The batch optimization method with parallel validation,'''

        def validation(train_loss=None, time_elapsed_total=None):
            # CHANGED (clean): trimmed to the spec path
            self.eval()
            self._sync_prediction_state_for_validation(refresh_basis=False)
            Loss_val = self.cal_validation_error(val_sys_data, validation_measure=validation_measure)
            self.Loss_val.append(Loss_val)

            self.Loss_train.append(train_loss)
            self.time.append(time_elapsed_total)
            self.batch_id.append(self.batch_counter)
            self.epoch_id.append(self.epoch_counter)
            if self.bestfit>=Loss_val:
                self.bestfit = Loss_val
                # CHANGED (D-169): the model is still on its training device here, so this writes
                # CUDA tensors.
                self.checkpoint_save_system()
            # CHANGED (clean): D-275, _last.pth at every validation, so a stopped run can be resumed from
            # its last weights and Adam moments (not its batch stream or an unfinished PS2 opening).
            self.checkpoint_save_system(name='_last')
            self.train()
            return Loss_val

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

        # CHANGED: Dtype_DataLoader instead of My_Simple_DataLoader, which hard-casts the
        # training arrays to float32 and makes use_f64=True raise on the first batch.
        _model_dtype = next(self.hfn.parameters()).dtype
        data_train_loader = Dtype_DataLoader(data_train, batch_size=batch_size,
                                             dtype=_model_dtype)

        validation(train_loss=float('nan'), time_elapsed_total=0)
        print(f'Initial Validation {validation_measure}=', self.Loss_val[-1])

        try:
            start_t = time.time()

            bestfit_old = self.bestfit
            for it_count, train_batch in zip(range(n_its), loop_dataset(data_train_loader)):
                if cuda:
                    train_batch = [b.cuda() for b in train_batch]
                # CHANGED (closed-loop seam): arrays a simulator asked make_training_data to
                # append are passed to loss() BY NAME, not positionally.
                batch_kwargs = dict(loss_kwargs, **dict(zip(self.simulator.extra_array_names, train_batch[4:])))

                def closure():
                    # CHANGED (clean): trimmed to the spec path
                    Loss = self.loss(*train_batch[:4], **batch_kwargs)
                    self.optimizer.zero_grad()
                    Loss.backward()
                    return Loss

                training_loss = self.optimizer.step(closure).item()

                if np.isnan(training_loss):
                    print(f'&&&&&&&&&&&&& Encountered a NaN value in the training loss at it {it_count}, breaking from loop &&&&&&&&&&')
                    break
                # CHANGED (D-236): the declared training-phase seam; None = the unchanged loop.
                if self.training_phase is not None:
                    self.training_phase.after_step(self)
                    # CHANGED (clean): D-279 (F4), how far the opening got at these weights, for the dumps
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

        self.train(); self.cpu()
        del data_train_loader

        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array(self.Loss_val), np.array(self.Loss_train), np.array(self.batch_id), np.array(self.time), np.array(self.epoch_id)
        self.checkpoint_save_system(name='_last')
        _selected_checkpoint = '_best'
        try:
            self.checkpoint_load_system(name='_best')
        except FileNotFoundError:
            print('no best checkpoint found keeping last')
            _selected_checkpoint = '_last'
        if self._sync_prediction_state_for_validation(refresh_basis=True):
            self.checkpoint_save_system(name=_selected_checkpoint)
        print(f'Loaded model with best known validation {validation_measure} of {self.bestfit:6.4} which happened on epoch {best_it} (epoch_id={self.epoch_id[-1]:.2f})')


@added
class SSE_Interconnect_Composed(SSE_Interconnect):
    """MSE plus composed penalty terms, in ONE class instead of a subclass per feature."""

    _optimizer = None
    obc = None
    # CHANGED (D-236): the training-time state (as `obc`): left out of every dump, kept across a
    # dump load
    _UNPICKLABLE = ('obc', 'obc_reference', 'training_phase')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        blocks = tuple(self.hfn.connected_blocks)

        self._ann_ix = next((i for i, m in enumerate(blocks)
                             if isinstance(m, Static_ANN_Block)), None)
        self._mats_ix = next((i for i, m in enumerate(blocks) if hasattr(m, 'pass_mats')), None)

    def _epoch_index(self):
        """The integer epoch the OBC refresh policy counts in."""
        return int(math.floor(float(getattr(self, 'epoch_counter', 0.0)) + 1e-6))

    def _sync_prediction_state_for_validation(self, refresh_basis=False):
        """Pair the frozen prediction coefficient with the current ANN parameters."""
        if self.obc is None:
            return False
        corr = self.hfn.obc_correction
        epoch = self._epoch_index()
        with torch.no_grad():
            if refresh_basis or self.obc.basis is None:
                self.obc.refresh(corr.expansion_point(), epoch)
            theta = self.obc.coefficient(self.hfn.connected_blocks[self._ann_ix])
            corr.set_frozen(theta, self.obc.basis.vbar)
        return True

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist)
        Loss_kwargs = dict(
            Loss_kwargs,
            pass_mats=self.hfn.connected_blocks[self._mats_ix].pass_mats())
        if self.obc is not None:
            _corr = self.hfn.obc_correction
            self.obc.maybe_refresh(self._epoch_index(), _corr.expansion_point)
            _theta = self.obc.coefficient(self.hfn.connected_blocks[self._ann_ix])
            _corr.set_frozen(_theta, self.obc.basis.vbar)
            # CHANGED (D-194): build the once-per-objective, state-independent tangent AFTER
            # set_frozen has updated `vbar`, and THREAD it.
            Loss_kwargs = dict(Loss_kwargs, obc_pass=_corr.tangent_pair(_theta))
        # CHANGED (D-256): `loss_w` (windows, ny), the fixed record-and-axis weight of each window
        # sent by the simulator, is the objective's, not the rollout's.
        w = Loss_kwargs.pop('loss_w')
        y_pred, _ = self.simulate(x, ufuture, yfuture, **Loss_kwargs)
        # THEORY: D-256, w = sigma_y / sigma_e turns the normalised error into error / sigma_e,
        # so the objective is the squared error relative to the servo error per record and axis.
        return ((y_pred - yfuture) * w[:, None, :]).square().mean()

    def checkpoint_save_system(self, name='_best'):
        # D-279 (F16): deepSI's save (fit_system.py:493-496) into a temporary file beside the
        # dump, renamed into place after the complete write, so a process killed while saving
        # keeps the old dump; the dump is __getstate__, the state without the training-time keys
        file = os.path.join(get_work_dirs()['checkpoints'], self.name + name + '.pth')
        torch.save(self.__getstate__(), file + '.tmp')
        _replace(file + '.tmp', file)

    def checkpoint_load_system(self, *args, **kwargs):
        stash = {k: self.__dict__.get(k) for k in self._UNPICKLABLE}
        out = super().checkpoint_load_system(*args, **kwargs)
        for k, v in stash.items():
            if v is not None:
                self.__dict__[k] = v
        return out

    def __getstate__(self):
        """Keep the training-time state OUT of the checkpoint."""
        return {k: v for k, v in self.__dict__.items() if k not in self._UNPICKLABLE}

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
