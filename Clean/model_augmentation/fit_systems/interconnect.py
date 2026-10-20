from deepSI.fit_systems.encoders import SS_encoder_general

from torch import nn, Tensor
import torch
import numpy as np
import time
import itertools
import math                     # CHANGED (D-192): epoch index for the OBC refresh
from torch.utils.data import Dataset

from deepSI.fit_systems.fit_system import print_array_byte_size, My_Simple_DataLoader, loop_dataset

from model_augmentation.utils.utils import detect_algebraic_loop
from model_augmentation.utils.utils import added  # CHANGED: marker for project-added classes (D-076)
from model_augmentation.fit_systems.blocks import Block, Static_ANN_Block  # CHANGED: Static_ANN_Block for SSE_Interconnect_Composed's orth term
from model_augmentation.utils.deepSI_corrections import fixed_System_data_norm

class Interconnect(nn.Module):
    # CHANGED (clean): trimmed to the spec path
    def __init__(self, nx, nu, ny, *args, **kwargs) -> None:
        # CHANGED (clean): trimmed to the spec path
        super().__init__(*args, **kwargs)

        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.nb = None

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

    # CHANGED (D-169): the device/dtype the connection matrices should live at.
    def _module_device_dtype(self):
        for t in itertools.chain(self.parameters(), self.buffers()):
            return t.device, (t.dtype if t.is_floating_point() else None)
        return None, None

    def init_model(self, sys_data):
        for block in self.connected_blocks:
            block.init_block(torch.empty((0)))

    def forward(self, x: Tensor, u: Tensor, obc_pass=None, pass_mats=None):
        # CHANGED (clean): trimmed to the spec path
        # CHANGED (D-194/D-195): `obc_pass` is the once-per-objective `(mats, dmats)` pair,
        # passed as a TENSOR-TUPLE ARGUMENT so its graph contract under
        # torch.compile is explicit (an input, not a mutated attribute).
        if len(x.size()) <= 2:
            x = x.view(x.size(0), self.nx, 1)
            state_has_correct_dimension = False

        if len(u.size()) <= 2:
            u = u.view(u.size(0), self.nu, 1)

        assert x.size(0) == u.size(0)
        self.nb = x.size(0)

        if not self.initialized_forward_function:
            self.init_forward()
            self.initialized_forward_function = True

        input_signals = [x, u]
        output_signals = []
        for ix in range(2, self.n_input_signals):
            input_signals.append(torch.zeros((self.nb, self.input_signal_sizes[ix], 1), device=x.device, dtype=x.dtype))
        for ix in range(0, self.n_output_signals):
            output_signals.append(torch.zeros((self.nb, self.output_signal_sizes[ix], 1), device=x.device, dtype=x.dtype))

        for output_signal_ix in self.order_output_signal_computation:
            for input_signal_ix in self.output_ix_sorted_input_ix_dependencies[output_signal_ix]:
                output_signals[output_signal_ix] += torch.matmul(self.array_connection_matrices[output_signal_ix][input_signal_ix], input_signals[input_signal_ix])

            if output_signal_ix >= 2:
                # CHANGED (D-195): hand the per-pass matrix structure to the ONE block that
                # declared it wants it.
                _bi = output_signal_ix - 2
                if pass_mats is not None and _bi == self._pass_mats_ix:
                    input_signals[output_signal_ix] = self.connected_blocks[_bi].forward(
                        output_signals[output_signal_ix], pass_mats)
                else:
                    input_signals[output_signal_ix] = self.connected_blocks[_bi].forward(output_signals[output_signal_ix])

        y = output_signals[1]
        xp = output_signals[0]

        # CHANGED (D-192): one-step orthogonal-by-construction subtraction.
        _obc = getattr(self, 'obc_correction', None)
        if _obc is not None:
            xp = xp - _obc(x, u, obc_pass)

        if not state_has_correct_dimension: xp = xp.view(self.nb, self.nx)
        if self.ny == 1: y = y.view(self.nb)
        if self.ny >= 2: y = y.view(self.nb, self.ny)

        return y, xp

    def init_forward(self):
        # CHANGED (clean): trimmed to the spec path
        self.n_output_signals = self.nr_blocks+2
        self.n_input_signals = self.nr_blocks+2
        self.n_nodes = 4 + self.nr_blocks

        self.array_connection_matrices = [[torch.empty((0,0)) for i in range(self.n_output_signals)] for j in range(self.n_input_signals)]
        self.output_ix_sorted_input_ix_dependencies = []

        directional_signal_connection_matrix = np.zeros((self.n_nodes, self.n_nodes))
        output_ix_sorted_signal_connections = [[] for i in range(self.n_output_signals)]

        for signal_connection in self.signal_connections:
            input_signal_ix = signal_connection.input_signal_ix
            output_signal_ix = signal_connection.output_signal_ix

            shifted_output_signal_ix = output_signal_ix
            if shifted_output_signal_ix <= 1: shifted_output_signal_ix += self.n_nodes - 2
            directional_signal_connection_matrix[shifted_output_signal_ix, input_signal_ix] = 1

            output_ix_sorted_signal_connections[output_signal_ix].append(signal_connection)

        connection_interconnect_matrix = np.roll(directional_signal_connection_matrix[2:,:-2], 2, axis=0)

        assert not detect_algebraic_loop(directional_signal_connection_matrix)

        self.order_output_signal_computation = []
        connection_interconnect_matrix[:,0] = 0
        connection_interconnect_matrix[:,1] = 0

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
        _dev, _dt = self._module_device_dtype()
        if _dev is not None:
            self.array_connection_matrices = [
                [t.to(device=_dev, dtype=_dt) if (_dt is not None and t.is_floating_point())
                 else t.to(device=_dev) for t in row]
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
    def output_only(self, x, u=None):
        # CHANGED (clean): trimmed to the spec path
        '''y from the output signal's dependency cone alone, without advancing the state.'''
        if len(x.size()) <= 2:
            x = x.view(x.size(0), self.nx, 1)
        nb = x.size(0)
        if u is None:
            u = torch.zeros((nb, self.nu, 1), device=x.device, dtype=x.dtype)
        elif len(u.size()) <= 2:
            u = u.view(nb, self.nu, 1)

        if not self.initialized_forward_function:
            self.init_forward()
            self.initialized_forward_function = True
        self.nb = nb

        input_signals = [x, u]
        output_signals = []
        for ix in range(2, self.n_input_signals):
            input_signals.append(torch.zeros((nb, self.input_signal_sizes[ix], 1), device=x.device, dtype=x.dtype))
        for ix in range(0, self.n_output_signals):
            output_signals.append(torch.zeros((nb, self.output_signal_sizes[ix], 1), device=x.device, dtype=x.dtype))

        for output_signal_ix in self.output_cone:
            for input_signal_ix in self.output_ix_sorted_input_ix_dependencies[output_signal_ix]:
                output_signals[output_signal_ix] += torch.matmul(self.array_connection_matrices[output_signal_ix][input_signal_ix], input_signals[input_signal_ix])
            if output_signal_ix >= 2:
                input_signals[output_signal_ix] = self.connected_blocks[output_signal_ix-2].forward(output_signals[output_signal_ix])

        y = output_signals[1]
        if self.ny == 1: y = y.view(nb)
        if self.ny >= 2: y = y.view(nb, self.ny)
        return y

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
            n_in = self.input_signal_sizes[signal_connection.input_signal_ix]
            connection_matrix = signal_connection.connection_matrix

            if signal_connection.input_signal_ix in input_signal_ixs:
                self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix] += connection_matrix
            else:
                self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix] = connection_matrix
                input_signal_ixs.append(signal_connection.input_signal_ix)

        return input_signal_ixs

    def add_block(self, new_block: Block, name=None):
        # CHANGED (clean): trimmed to the spec path
        self.nr_blocks += 1
        new_block.block_ix = self.nr_blocks + 1
        new_block.name = name if isinstance(name, str) else "Block_" + str(self.nr_blocks+1)
        self.connected_blocks.append(new_block)

        self.input_signal_sizes.append(new_block.nw)
        self.output_signal_sizes.append(new_block.nz)

    def connect_signals(self, input_signal, output_signal, connection_function_method=None, connection_matrix=torch.empty(0,0), add_to_input_signal_ix=None):
        # CHANGED (clean): trimmed to the spec path
        input_signal_ix = self.determine_signal_ix(input_signal)
        output_signal_ix = self.determine_signal_ix(output_signal)
        if add_to_input_signal_ix != None: add_to_input_signal_ix = self.determine_signal_ix(add_to_input_signal_ix)

        if not isinstance(connection_function_method, str) and output_signal_ix >= 2: connection_function_method = "concatenation"
        if not isinstance(connection_function_method, str) and output_signal_ix <= 1: connection_function_method = "additive"
        connection_function_method = self.parse_connection_function_method(connection_function_method)

        self.signal_connections.append(Signal_Connection(input_signal_ix, output_signal_ix, connection_function_method=connection_function_method, \
                                        connection_matrix=connection_matrix, add_to_input_signal_ix=add_to_input_signal_ix))

    def connect_block_signals(self, block, input_signal_list: list, output_signal_list: list):
        for input_signal in input_signal_list:
            self.connect_signals(input_signal, block)

        for output_signal in output_signal_list:
            self.connect_signals(block, output_signal)

    def determine_signal_ix(self, signal):
        # CHANGED (clean): trimmed to the spec path
        if isinstance(signal, Block):
            return signal.block_ix
        if isinstance(signal, int):
            return signal
        if isinstance(signal, str):
            return self.convert_signal_name_to_ix(signal)

    def convert_signal_name_to_ix(self, signal_name: str):
        # CHANGED (clean): trimmed to the spec path
        if signal_name in ["x", "xp"]:
            return 0
        if signal_name in ["u", "y"]:
            return 1
        if len(signal_name) >= 1:
            signal_ix = int(list(signal_name)[1])
            return signal_ix

    def parse_connection_function_method(self, connection_function_method: str):
        connection_function_method = connection_function_method.lower()

        if connection_function_method in ["concatenation", "concat", "con", "cat", "c"]:
            return "concatenation"
        if connection_function_method in ["additive", "add", "a", "additional"]:
            return "additive"
        if connection_function_method in ["add_to", "to", "at", "add to"]:
            return "add_to"

class Signal_Connection():
    # CHANGED (clean): trimmed to the spec path
    '''Object to hold information regarding connection between two signals in the interconnect'''
    def __init__(self, input_signal_ix, output_signal_ix, connection_function_method, connection_matrix = torch.empty((0,0)), add_to_input_signal_ix=None) -> None:
        self.input_signal_ix = input_signal_ix
        self.output_signal_ix = output_signal_ix

        self.connection_function_method = connection_function_method
        self.connection_matrix = connection_matrix
        self.add_to_input_signal_ix = add_to_input_signal_ix


@added
class Dtype_DataLoader(My_Simple_DataLoader):
    """My_Simple_DataLoader that keeps the training arrays at the MODEL's dtype."""

    def __init__(self, data, batch_size=32, seed=0, dtype=torch.float32):
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

    def __init__(self, na=5, nb=5, \
                 interconnect=Interconnect, na_right=0, nb_right=0):
        # CHANGED (clean): trimmed to the spec path

        super(SSE_Interconnect, self).__init__(nx=interconnect.nx, nb=nb, na=na, na_right=na_right, nb_right=nb_right)

        self.hfn = interconnect
        self.encoder = None

        self.norm = fixed_System_data_norm()

    def init_model(self, sys_data=None, nu=-1, ny=-1, device='cpu', auto_fit_norm=True, optimizer_kwargs={}, parameters_optimizer_kwargs={}, scheduler_kwargs={}):
        # CHANGED (clean): trimmed to the spec path
        '''This function set the nu and ny, inits the network, moves parameters to device, initilizes optimizer and initilizes logging parameters'''
        self.nu, self.ny = sys_data.nu, sys_data.ny
        self.to_device(device=device)
        parameters_and_optim = [{**item,**parameters_optimizer_kwargs.get(name,{})} for name,item in self.parameters_with_names.items()]
        self.optimizer = self.init_optimizer(parameters_and_optim, **optimizer_kwargs)
        self.scheduler = self.init_scheduler(**scheduler_kwargs)
        self.bestfit = float('inf')
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        self.init_model_done = True

        self.hfn.init_model(sys_data)

    # CHANGED (closed-loop seam): the rollout is extracted from loss() into an overridable method.
    def simulate(self, x, ufuture, yfuture=None, **Loss_kwargs):
        # CHANGED (clean): trimmed to the spec path
        '''Roll the model forward from x. Returns (y_pred, x_final), y_pred (batch, nf, ny).'''
        if self.simulator is not None:
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
        if self.simulator is not None and hasattr(self.simulator, 'validation_error'):
            value = self.simulator.validation_error(self, val_sys_data, validation_measure)
        for probe in self.validation_probes:
            probe(self, val_sys_data, value)
        return value

    # CHANGED: added fit() method replacing deepSI's original: supports multi-trajectory, sqrt loss, concurrent validation
    def fit(self, train_sys_data, val_sys_data, epochs=30, n_its=None, batch_size=256, loss_kwargs={}, \
            auto_fit_norm=True, validation_measure='sim-NRMS', optimizer_kwargs={}, its_per_val='epoch', cuda=False, \
            verbose=2, sqrt_train=True):
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
            # CHANGED (clean): D-275, _last.pth at every validation so a stopped run can continue
            self.checkpoint_save_system(name='_last')
            self.train()
            return Loss_val

        if verbose: print('Model already initilized (init_model_done=True), skipping initilizing of the model, the norm and the creation of the optimizer')
        self._check_and_refresh_optimizer_if_needed()

        self.dt = train_sys_data.dt
        if cuda:
            self.cuda()
        self.train()

        self.epoch_counter = 0 if len(self.epoch_id)==0 else self.epoch_id[-1]
        self.batch_counter = 0 if len(self.batch_id)==0 else self.batch_id[-1]
        extra_t            = 0 if len(self.time)    ==0 else self.time[-1]

        data_train = self.make_training_data(self.norm.transform(train_sys_data), **loss_kwargs)
        if not isinstance(data_train, Dataset) and verbose: print_array_byte_size(sum([d.nbytes for d in data_train]))

        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = list(self.Loss_val), list(self.Loss_train), list(self.batch_id), list(self.time), list(self.epoch_id)

        Loss_acc_val_loop, it_counter_per_val_loop, val_counter, best_it = 0, 0, 0, 0
        N_training_samples = len(data_train) if isinstance(data_train, Dataset) else len(data_train[0])
        batch_size = min(batch_size, N_training_samples)
        N_batch_updates_per_epoch = N_training_samples//batch_size
        n_its = int(N_batch_updates_per_epoch*epochs) if n_its is None else n_its
        Loss_acc_print_loop = 0.
        its_per_val = N_batch_updates_per_epoch if its_per_val=='epoch' else its_per_val
        if verbose>0:
            print(f'N_training_samples = {N_training_samples}, batch_size = {batch_size}, N_batch_updates_per_epoch = {N_batch_updates_per_epoch}')

        # CHANGED: Dtype_DataLoader instead of My_Simple_DataLoader, which hard-casts the
        # training arrays to float32 and makes use_f64=True raise on the first batch.
        _model_dtype = next(self.hfn.parameters()).dtype
        data_train_loader = Dtype_DataLoader(data_train, batch_size=batch_size,
                                             dtype=_model_dtype)

        validation(train_loss=float('nan'), time_elapsed_total=extra_t)
        if verbose:
            print(f'Initial Validation {validation_measure}=', self.Loss_val[-1])

        try:
            start_t = time.time()
            rang = range(n_its)

            bestfit_old = self.bestfit
            for it_count, train_batch in zip(rang, loop_dataset(data_train_loader)):
                if cuda:
                    train_batch = [b.cuda() for b in train_batch]
                # CHANGED (closed-loop seam): arrays a simulator asked make_training_data to
                # append are passed to loss() BY NAME, not positionally.
                batch_kwargs = dict(loss_kwargs)
                if len(train_batch) > 4:
                    names = getattr(self.simulator, 'extra_array_names', ())
                    batch_kwargs.update(zip(names, train_batch[4:]))

                def closure(backward=True):
                    # CHANGED (clean): trimmed to the spec path
                    Loss = self.loss(*train_batch[:4], **batch_kwargs)
                    if backward:
                        self.optimizer.zero_grad()
                        Loss.backward()
                    return Loss

                training_loss = self.optimizer.step(closure).item()

                if np.isnan(training_loss):
                    if verbose>0: print(f'&&&&&&&&&&&&& Encountered a NaN value in the training loss at it {it_count}, breaking from loop &&&&&&&&&&')
                    break
                # CHANGED (D-236): the declared training-phase seam; None = the unchanged loop.
                if self.training_phase is not None:
                    self.training_phase.after_step(self)

                Loss_acc_val_loop += training_loss
                Loss_acc_print_loop += training_loss
                it_counter_per_val_loop += 1
                self.batch_counter += 1
                self.epoch_counter += 1/N_batch_updates_per_epoch

                if (it_count+1)%its_per_val==0:
                    validation(train_loss=Loss_acc_val_loop/it_counter_per_val_loop, \
                           time_elapsed_total=time.time()-start_t+extra_t)
                    Loss_acc_val_loop, it_counter_per_val_loop, val_counter = 0., 0, val_counter + 1

                if verbose>0 and (it_count+1)%its_per_val==0:
                    if bestfit_old > self.bestfit:
                        print(f'########## New lowest validation loss achieved ########### {validation_measure} = {self.bestfit}')
                        best_it = it_count+1
                        bestfit_old = self.bestfit
                    train_loss_epoch, Loss_acc_print_loop = Loss_acc_print_loop/its_per_val, 0
                    trainstr = f'sqrt loss {train_loss_epoch**0.5:7.4}' if sqrt_train and train_loss_epoch>=0 else f'loss {train_loss_epoch:7.4}'
                    Loss_val_now = self.Loss_val[-1] if len(self.Loss_val)!=0 else float('nan')
                    Loss_str = f'It {it_count+1:4}, {trainstr}, Val {validation_measure} {Loss_val_now:6.4}'
                    print(Loss_str)
        except KeyboardInterrupt:
            print('Stopping early due to a KeyboardInterrupt')

        self.train(); self.cpu()
        del data_train_loader

        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array(self.Loss_val), np.array(self.Loss_train), np.array(self.batch_id), np.array(self.time), np.array(self.epoch_id)
        self.checkpoint_save_system(name='_last')
        # CHANGED (D-172): keep this run's OWN history across the best-checkpoint restore.
        _history = (self.epoch_id.copy(), self.Loss_val.copy(), self.Loss_train.copy())
        _selected_checkpoint = '_best'
        try:
            self.checkpoint_load_system(name='_best')
        except FileNotFoundError:
            print('no best checkpoint found keeping last')
            _selected_checkpoint = '_last'
        self.epoch_id, self.Loss_val, self.Loss_train = _history
        if self._sync_prediction_state_for_validation(refresh_basis=True):
            self.checkpoint_save_system(name=_selected_checkpoint)
        if verbose:
            print(f'Loaded model with best known validation {validation_measure} of {self.bestfit:6.4} which happened on epoch {best_it} (epoch_id={self.epoch_id[-1] if len(self.epoch_id)>0 else 0:.2f})')


@added
class SSE_Interconnect_Composed(SSE_Interconnect):
    """MSE plus composed penalty terms, in ONE class instead of a subclass per feature."""

    _optimizer = None
    obc = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        blocks = tuple(self.hfn.connected_blocks)

        self._ann_ix = next((i for i, m in enumerate(blocks)
                             if isinstance(m, Static_ANN_Block)), None)
        self._mats_ix = next((i for i, m in enumerate(blocks) if hasattr(m, 'pass_mats')), None)

    def _sync_prediction_state_for_validation(self, refresh_basis=False):
        """Pair the frozen prediction coefficient with the current ANN parameters."""
        if self.obc is None:
            return False
        corr = getattr(self.hfn, 'obc_correction', None)
        epoch = int(math.floor(float(getattr(self, 'epoch_counter', 0.0)) + 1e-6))
        with torch.no_grad():
            if refresh_basis:
                self.obc.refresh(corr.expansion_point(), epoch)
            elif self.obc.basis is None:
                self.obc.refresh(corr.expansion_point(), epoch)
            theta = self.obc.prediction_coefficient(self.hfn.connected_blocks[self._ann_ix])
            corr.set_frozen(theta, self.obc.basis.vbar)
        return True

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist)
        if self._mats_ix is not None:
            Loss_kwargs = dict(
                Loss_kwargs,
                pass_mats=self.hfn.connected_blocks[self._mats_ix].pass_mats())
        if self.obc is not None:
            _corr = getattr(self.hfn, 'obc_correction', None)
            _epoch = int(math.floor(float(getattr(self, 'epoch_counter', 0.0)) + 1e-6))
            self.obc.maybe_refresh(_epoch, _corr.expansion_point)
            _theta = self.obc.coefficient(self.hfn.connected_blocks[self._ann_ix])
            _corr.set_frozen(_theta, self.obc.basis.vbar)
            # CHANGED (D-194): build the once-per-objective, state-independent tangent AFTER
            # set_frozen has updated `vbar`, and THREAD it.
            _tan = _corr.tangent_pair(_theta) if hasattr(_corr, 'tangent_pair') else _theta
            Loss_kwargs = dict(Loss_kwargs, obc_pass=_tan)
        # CHANGED (D-256): `loss_w` (windows, ny), the fixed record-and-axis weight of each window
        # sent by the simulator, is the objective's, not the rollout's.
        w = Loss_kwargs.pop('loss_w', None)
        y_pred, _ = self.simulate(x, ufuture, yfuture, **Loss_kwargs)
        # THEORY: D-256, w = sigma_y / sigma_e turns the normalised error into error / sigma_e,
        # so the objective is the squared error relative to the servo error per record and axis.
        L = ((y_pred - yfuture) * w[:, None, :]).square().mean()
        return L

    _OBC_UNPICKLABLE = ('obc', 'obc_reference')
    _PHASE_UNPICKLABLE = ('training_phase',)

    def checkpoint_save_system(self, *args, **kwargs):
        stash = {k: self.__dict__.pop(k) for k in self._OBC_UNPICKLABLE
                 + self._PHASE_UNPICKLABLE  # CHANGED (D-236)
                 if k in self.__dict__}
        try:
            return super().checkpoint_save_system(*args, **kwargs)
        finally:
            self.__dict__.update(stash)

    def checkpoint_load_system(self, *args, **kwargs):
        stash = {k: self.__dict__.get(k)
                 for k in self._OBC_UNPICKLABLE
                 + self._PHASE_UNPICKLABLE}  # CHANGED (D-236)
        out = super().checkpoint_load_system(*args, **kwargs)
        for k, v in stash.items():
            if v is not None:
                self.__dict__[k] = v
        return out

    def __getstate__(self):
        """Keep the training-time state OUT of the checkpoint."""
        state = dict(self.__dict__)
        state.pop('obc', None)
        state.pop('obc_reference', None)
        state.pop('training_phase', None)   # CHANGED (D-236): training-time state, as `obc`
        return state

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
