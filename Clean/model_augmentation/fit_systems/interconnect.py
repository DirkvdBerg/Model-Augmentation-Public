"""Jan's interconnect, which joins blocks into one state-space model, and his encoder fit system, plus the augmented
model that is trained here."""
# CHANGED: restricted to the thesis model; original: Model-Augmentation-Public-main
from deepSI.fit_systems.encoders import SS_encoder_general

from torch import nn, Tensor
import torch
import numpy as np
import os
import time
import math
import warnings

from deepSI.datasets import get_work_dirs
from deepSI.fit_systems.fit_system import print_array_byte_size, My_Simple_DataLoader, loop_dataset

from model_augmentation.utils.utils import added
from model_augmentation.fit_systems.blocks import Block, Static_ANN_Block
from model_augmentation.utils.deepSI_corrections import SystemDataNormalizer

class Interconnect(nn.Module):
    """Blocks joined by signal connection matrices into one state-space step (x, u) -> (y, xp).

    Signals: x = state x_hat_k, u = input, xp = next state x_hat_{k+1}, y = output y_hat_k."""
    # CHANGED: removed init_model (its init_block calls did nothing), reset_saved_signals, convert_signal_ix_to_name
    def __init__(self, nx, nu, ny) -> None:
        """An empty interconnect with state size nx, input size nu and output size ny."""
        # CHANGED: removed unused debugging and recording fields; nb is local to forward
        super().__init__()

        self.nu = nu
        self.ny = ny
        self.nx = nx

        # blocks added so far, in signal-index order (block i has signal index i + 2)
        self.nr_blocks = 0
        self.connected_blocks = nn.ModuleList([])

        # sizes of the signals read (x, u, block outputs) and written (xp, y, block inputs)
        self.input_signal_sizes = [nx, nu]
        self.output_signal_sizes = [nx, ny]

        # every connection added by connect_signals, resolved into matrices by init_forward
        self.signal_connections = []

        # init_forward runs once, at the first forward call
        self.initialized_forward_function = False
        # CHANGED: optional OBC correction, subtracted from the next state in forward
        self.obc_correction = None

    @added
    def _apply(self, fn, *args, **kwargs):
        """Move and cast the connection matrices with the model; they are not registered buffers."""
        out = super()._apply(fn, *args, **kwargs)
        acm = getattr(self, 'array_connection_matrices', None)
        if acm is not None:
            self.array_connection_matrices = [[fn(t) for t in row] for row in acm]
        return out

    def forward(self, x: Tensor, u: Tensor, obc_pass, pass_mats):
        """One step of the augmented model: output now and next state.

        x_tilde^n_{k+1} = f_base^n + f_aug^n (- f_obc^n on the OBC arm), xbar_{k+1} = g_aug^n,
        y_hat^n_k = C^n x_tilde^n_k.

        Args:
            x: (B, nx) state x_hat^n_k.
            u: (B, nu) controller input u_hat^n_k.
            obc_pass: (mats, dmats, c) of the OBC correction, None on the unconstrained arm.
            pass_mats: the baseline matrices of this objective, built once per objective.
        Returns:
            y_hat^n_k (B, ny) and x_hat^n_{k+1} (B, nx).
        """
        # CHANGED: pass matrices explicitly so torch.compile receives them as graph inputs
        # CHANGED: always (B, n) in and out, as the rollout passes; Jan's shape restoring and shape check removed
        # 1. Signals as column vectors (B, n, 1).
        x = x.view(x.size(0), self.nx, 1)
        u = u.view(u.size(0), self.nu, 1)

        assert x.size(0) == u.size(0)
        nb = x.size(0)

        # 2. Computation order and connection matrices, resolved at the first call.
        if not self.initialized_forward_function:
            self.init_forward()
            self.initialized_forward_function = True

        # 3. Evaluate the blocks in order and sum their contributions into xp and y.
        # CHANGED: loop moved to _propagate, so output_only can compute y_k before the controller sets u_k
        output_signals = self._propagate(x, u, self.order_output_signal_computation, pass_mats)

        # signal 1 is y_hat^n_k; signal 0 is x_hat^n_{k+1} before the OBC correction
        y = output_signals[1]
        xp = output_signals[0]

        # CHANGED: subtract f_obc^n to make the network correction orthogonal to J on the reference tuples
        if self.obc_correction is not None:
            xp = xp - self.obc_correction(x, u, obc_pass)

        # CHANGED: return (B, nx) and (B, ny) as required by the rollout; no single-output squeeze
        xp = xp.view(nb, self.nx)
        y = y.view(nb, self.ny)

        return y, xp

    @added
    def _propagate(self, x, u, order, pass_mats):
        """Run the wires and blocks for the requested signals, in the computation order (Jan's loop from forward).

        Each destination needs at least one source. The baseline block receives pass_mats.

        Args:
            x, u: (B, nx, 1) state and (B, nu, 1) input.
            order: destination indices; all for forward, only those needed for y for output_only.
            pass_mats: the baseline matrices of this objective, passed to the physics block.
        Returns:
            [xp, y, block inputs...]; signals not listed in order remain None.
        """
        # input signals: x, u, then each block's output once computed
        input_signals = [x, u] + [None] * (self.n_input_signals - 2)
        output_signals = [None] * self.n_output_signals

        # CHANGED: sums start from the first source, not from Jan's CPU float32 zeros, so device and dtype follow x
        for output_signal_ix in order:
            # destination signal = sum over its sources j of (connection matrix j) @ (source j)
            matrices = self.array_connection_matrices[output_signal_ix]
            first, *rest = self.output_ix_sorted_input_ix_dependencies[output_signal_ix]
            signal = torch.matmul(matrices[first], input_signals[first])
            for input_signal_ix in rest:
                signal = signal + torch.matmul(matrices[input_signal_ix], input_signals[input_signal_ix])
            output_signals[output_signal_ix] = signal

            if output_signal_ix >= 2:
                # indices 2 onward are block inputs; each block's output becomes a source at the same index
                block = self.connected_blocks[output_signal_ix - 2]
                if output_signal_ix - 2 == self._pass_mats_ix:
                    input_signals[output_signal_ix] = block.forward(output_signals[output_signal_ix], pass_mats)
                else:
                    input_signals[output_signal_ix] = block.forward(output_signals[output_signal_ix])
        return output_signals

    def init_forward(self):
        """Resolve the computation order and the connection matrices once, at the first forward call."""
        # CHANGED: removed Jan's debug prints; n_nodes is local; columns 0 and 1 zeroed in one line, same result
        # 1. Directed graph of the connections.
        self.n_output_signals = self.nr_blocks+2
        self.n_input_signals = self.nr_blocks+2
        # graph nodes: x, u, the blocks, then xp and y
        n_nodes = 4 + self.nr_blocks

        self.array_connection_matrices = [[torch.empty((0,0)) for i in range(self.n_output_signals)] for j in range(self.n_input_signals)]
        self.output_ix_sorted_input_ix_dependencies = []

        directional_signal_connection_matrix = np.zeros((n_nodes, n_nodes))
        output_ix_sorted_signal_connections = [[] for i in range(self.n_output_signals)]

        for signal_connection in self.signal_connections:
            input_signal_ix = signal_connection.input_signal_ix
            output_signal_ix = signal_connection.output_signal_ix

            # destinations xp (0) and y (1) are the last two nodes
            shifted_output_signal_ix = output_signal_ix
            if shifted_output_signal_ix <= 1: shifted_output_signal_ix += n_nodes - 2
            directional_signal_connection_matrix[shifted_output_signal_ix, input_signal_ix] = 1

            output_ix_sorted_signal_connections[output_signal_ix].append(signal_connection)

        # row i is output signal i (xp, y, blocks), column j input signal j (x, u, blocks)
        connection_interconnect_matrix = np.roll(directional_signal_connection_matrix[2:,:-2], 2, axis=0)

        # 2. Order signals by their dependencies; x and u are available from the start.
        self.order_output_signal_computation = []
        connection_interconnect_matrix[:,:2] = 0

        while len(self.order_output_signal_computation) < self.n_output_signals:
            n_computed = len(self.order_output_signal_computation)
            computable_elements = np.argwhere(np.sum(connection_interconnect_matrix, axis=1)==0).flatten()
            for element in computable_elements:
                if element not in self.order_output_signal_computation:
                    # CHANGED: use Python indices to avoid NumPy scalar handling inside the compiled rollout
                    self.order_output_signal_computation.append(int(element))
                    connection_interconnect_matrix[:,element] = 0
            # CHANGED: Jan's separate algebraic-loop assertion replaced by this check in the loop, with a clear error
            if len(self.order_output_signal_computation) == n_computed:
                raise ValueError('Algebraic loop in the signal connections.')

        # 3. Connection matrices of every destination signal.
        for output_signal_ix in range(self.n_output_signals):
            input_ix_dependencies = self.init_connection_matrices(output_signal_ix, output_ix_sorted_signal_connections[output_signal_ix])
            self.output_ix_sorted_input_ix_dependencies.append(input_ix_dependencies)

        # CHANGED: output_only needs y before the controller calculates u; skip blocks used only for the state update
        self.output_cone = self._signal_cone(1)
        # CHANGED: identify which block needs pass_mats; the other blocks take only their input signal
        self._pass_mats_ix = next(
            (i for i, b in enumerate(self.connected_blocks) if hasattr(b, 'pass_mats')), None)

        # CHANGED: match the model's device and dtype; Jan's connection matrices are created on the CPU
        p = next(self.parameters())
        self.array_connection_matrices = [[t.to(device=p.device, dtype=p.dtype) for t in row]
                                          for row in self.array_connection_matrices]

    @added
    def _signal_cone(self, target_output_ix):
        """Return the target's required signal indices in computation order."""
        needed, stack = set(), [target_output_ix]
        while stack:
            k = stack.pop()
            needed.add(k)
            for j in self.output_ix_sorted_input_ix_dependencies[k]:
                if j >= 2 and j not in needed:
                    stack.append(j)
        return tuple(k for k in self.order_output_signal_computation if k in needed)

    @added
    def output_only(self, x):
        """Compute y without updating the state; the controller needs y before calculating u.

        Assumes no input feedthrough (D = 0).

        Args:
            x: (B, nx) x_hat^n_k.
        Returns:
            (B, ny) y_hat^n_k.
        """
        x = x.view(x.size(0), self.nx, 1)
        nb = x.size(0)
        # y_hat^n_k does not depend on u (no input feedthrough), so u is set to zero
        u = torch.zeros((nb, self.nu, 1), device=x.device, dtype=x.dtype)

        if not self.initialized_forward_function:
            self.init_forward()
            self.initialized_forward_function = True

        y = self._propagate(x, u, self.output_cone, None)[1]
        return y.view(nb, self.ny)

    def init_connection_matrices(self, output_signal_ix, signal_connections):
        """Connection matrices of one destination signal; returns the indices of its sources.

        Concatenated sources each fill their own row block of the destination (zero-padded matrices); additive
        sources are added onto it.

        Args:
            output_signal_ix: the destination signal.
            signal_connections: the connections into it.
        """
        # CHANGED: removed unused add_to handling and debug output; matrix construction is otherwise Jan's
        # 1. Split the connections by method.
        n_out = self.output_signal_sizes[output_signal_ix]
        n_out_total = 0

        concat_signal_connections = []
        additive_signal_connections = []

        input_signal_ixs = []

        for signal_connection in signal_connections:
            if signal_connection.connection_function_method == "concatenation":
                concat_signal_connections.append(signal_connection)
            if signal_connection.connection_function_method == "additive":
                additive_signal_connections.append(signal_connection)

        # the first concatenation, else the first additive connection, sets the destination's layout
        if len(concat_signal_connections) == 0 and len(additive_signal_connections) != 0:
            concat_signal_connections.append(additive_signal_connections.pop(0))

        # 2. Concatenation: each source below the previous ones (identity if no matrix was given).
        for signal_connection in concat_signal_connections:
            n_in = self.input_signal_sizes[signal_connection.input_signal_ix]

            connection_matrix = signal_connection.connection_matrix
            if not connection_matrix.numel(): connection_matrix = torch.eye(n_in)
            n_out_contribution = connection_matrix.size(0)
            if n_out_total > 0: connection_matrix = torch.vstack((torch.zeros((n_out_total, n_in)), connection_matrix))
            self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix] = connection_matrix

            n_out_total += n_out_contribution
            input_signal_ixs.append(signal_connection.input_signal_ix)

        # 3. Pad every concatenation matrix with zero rows to the full destination size.
        for signal_connection in concat_signal_connections:
            connection_matrix = self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix]
            n_out_current = connection_matrix.size(0)
            n_in = self.input_signal_sizes[signal_connection.input_signal_ix]
            self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix] = torch.vstack((connection_matrix, torch.zeros((n_out_total - n_out_current, n_in))))

        assert n_out_total == n_out, "total: {0}, required: {1}".format(n_out_total, n_out)

        # 4. Additive: the given matrix maps the source onto the whole destination.
        for signal_connection in additive_signal_connections:
            n_in = self.input_signal_sizes[signal_connection.input_signal_ix]
            connection_matrix = signal_connection.connection_matrix
            if not connection_matrix.numel():
                connection_matrix = torch.eye(n_out, n_in)
                warnings.warn("The additive method was not given a connection matrix and thus filled in a identity matrix that might not be square. This could give unintended behaviour")
            else:
                assert connection_matrix.size(0) == n_out
                assert connection_matrix.size(1) == n_in

            if signal_connection.input_signal_ix in input_signal_ixs:
                self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix] += connection_matrix
            else:
                self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix] = connection_matrix
                input_signal_ixs.append(signal_connection.input_signal_ix)

        for signal_connection in signal_connections:
            connection_matrix = self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix]
            if not connection_matrix.numel(): raise ValueError("Connection matrix should not be empty.")
            assert connection_matrix.size(0) == n_out

        return input_signal_ixs

    def add_block(self, new_block: Block):
        """Add a block; its output and input get the next signal index."""
        # CHANGED: removed unused block names and debug output
        assert new_block not in self.connected_blocks
        self.nr_blocks += 1
        new_block.block_ix = self.nr_blocks + 1
        self.connected_blocks.append(new_block)

        self.input_signal_sizes.append(new_block.nw)
        self.output_signal_sizes.append(new_block.nz)

    def connect_signals(self, input_signal, output_signal, connection_function_method=None, connection_matrix=torch.empty(0,0)):
        """Connect two signals.

        Args:
            input_signal, output_signal: source and destination, a block or a name ('x', 'u', 'xp', 'y').
            connection_function_method: 'concat' stacks the source into the destination, 'additive' adds it; None
                takes 'concatenation' into a block and 'additive' into xp or y.
            connection_matrix: maps the source (selection or expansion matrix); empty means identity.
        """
        # CHANGED: removed unused add_to and debug output; method lookup replaces the method assertion
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
        """Signal index of a block or a signal name."""
        if isinstance(signal, Block):
            return signal.block_ix
        if isinstance(signal, str):
            return self.convert_signal_name_to_ix(signal)
        # CHANGED: removed integer references unused by the model builder
        raise TypeError("Input could not be converted to signal ix.")

    def convert_signal_name_to_ix(self, signal_name: str):
        """0 for x and xp, 1 for u and y: x and u index the signals read, xp and y the signals written."""
        if signal_name in ["x", "xp"]:
            return 0
        if signal_name in ["u", "y"]:
            return 1
        # CHANGED: removed numbered names such as 'w2', unused by the model builder
        raise TypeError("Input could not be converted to signal ix.")

    def parse_connection_function_method(self, connection_function_method: str):
        """The canonical name of a connection method."""
        # CHANGED: removed unused aliases and case conversion; accept concatenation, concat and additive
        return {"concatenation": "concatenation", "concat": "concatenation",
                "additive": "additive"}[connection_function_method]

class Signal_Connection():
    '''Connection source, destination, method and matrix.'''
    # CHANGED: removed unused add_to, __str__ and __repr__; connect_signals supplies the matrix argument
    def __init__(self, input_signal_ix, output_signal_ix, connection_function_method, connection_matrix) -> None:
        """Store the two signal indices, the method and the connection matrix."""
        self.input_signal_ix = input_signal_ix
        self.output_signal_ix = output_signal_ix

        self.connection_function_method = connection_function_method
        self.connection_matrix = connection_matrix


@added
def _replace_with_retry(src, dst, tries=20):
    """Retry checkpoint replacement after PermissionError, which can occur when Windows holds the file open."""
    # HEURISTIC: 20 tries 0.25 s apart, about 5 s for the file to be released
    for i in range(tries):
        try:
            return os.replace(src, dst)
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.25)


@added
class DtypeDataLoader(My_Simple_DataLoader):
    """Batch windows at the model dtype; deepSI's loader casts floating arrays to float32."""

    def __init__(self, data, batch_size, dtype):
        """Cast floating arrays to dtype; retain deepSI's shuffle and skip incomplete batches."""
        def _conv(d):
            """Convert to a tensor, casting only floating arrays to dtype."""
            t = torch.as_tensor(d)
            return t.to(dtype) if t.is_floating_point() else t
        self.data = [_conv(d) for d in data]
        self.ids = np.arange(len(data[0]), dtype=int)
        self.rng = np.random.default_rng(0)
        self.batch_size = batch_size


class SSE_Interconnect(SS_encoder_general):
    """Jan's fit system: an encoder and an interconnect, trained on deepSI's simulation windows."""
    # CHANGED: removed measure_act_multi; predictions use the attached simulator
    # CHANGED: AugmentedModel.loss replaces Jan's loss (open-loop error plus his parameter penalty param_loss):
    # the data are closed-loop; param_loss is dropped, as it limits the drift of beta but does not stop the network
    # from taking over what beta models; the OBC arm uses OBC for that
    # CHANGED: use the closed-loop simulator for training rollouts and validation, as the data are closed-loop
    simulator = None

    # CHANGED: callbacks record per-record validation results for training.py
    validation_probes = ()

    # CHANGED: PS2 needs an auxiliary training step after each Adam update
    training_phase = None

    def __init__(self, interconnect, na=5, nb=5, na_right=0, nb_right=0):
        """deepSI's encoder system with the interconnect as the model hfn.

        Args:
            interconnect: the Interconnect, one model step.
            na, nb: past y and u samples in the encoder window.
            na_right, nb_right: future samples in the window (0 here).
        """
        # CHANGED: interconnect is the first, required argument (Jan's default was the Interconnect class)
        super(SSE_Interconnect, self).__init__(nx=interconnect.nx, nb=nb, na=na, na_right=na_right, nb_right=nb_right)

        # CHANGED: model.py sets the augmented encoder, replacing init_nets, e_net options and modified_encoder_net;
        # multi_loss_val dropped with the extra validation scores it held
        self.hfn = interconnect

        # CHANGED: use the renamed SystemDataNormalizer from deepSI_corrections
        self.norm = SystemDataNormalizer()

    def init_model(self, sys_data=None, optimizer_kwargs={}):
        '''Set nu and ny, build Adam over every parameter (encoder, then hfn) and reset bestfit.'''
        # CHANGED: model.py already sets the encoder and normalisation; training.py handles the device
        # CHANGED: removed Jan's optional scheduler (off by default; the thesis uses a fixed learning rate) and his
        # init_block call on each block, which did nothing; build_model sets each block's values when it creates it
        # CHANGED: removed per-parameter optimizer settings (one setting for all parameters here) and the nu/ny
        # option without data; fit resets the loss history
        self.nu, self.ny = sys_data.nu, sys_data.ny
        self.optimizer = self.init_optimizer(list(self.parameters_with_names.values()), **optimizer_kwargs)
        self.bestfit = float('inf')
        self.init_model_done = True

    @added
    def simulate(self, x, ufuture, yfuture=None, **Loss_kwargs):
        """Run the attached simulator; return predictions (B, nf, ny) and final states (B, nx)."""
        return self.simulator(self, x, ufuture, yfuture, **Loss_kwargs)

    @added
    def make_training_data(self, sys_data, **Loss_kwargs):
        """Add controller indices and record weights to deepSI's training windows.

        Returns uhist, yhist (N, 30, 3) (na = nb = 30 samples), ufuture, yfuture (N, nf, 3), ctrl_ix and loss_w."""
        data = super().make_training_data(sys_data, **Loss_kwargs)
        return self.simulator.augment_training_data(data, sys_data, self, **Loss_kwargs)

    @added
    def cal_validation_error(self, val_sys_data):
        """Compute the simulator's validation score and pass it to the history callbacks."""
        value = self.simulator.validation_error(self, val_sys_data)
        for probe in self.validation_probes:
            probe(self, val_sys_data, value)
        return value

    # CHANGED: init_model must be called first (build_model does); Jan's lazy init_model and optimizer refresh removed
    # CHANGED: removed Dataset and parallel-validation options unused by training.py
    # CHANGED: removed timeout, verbosity, profiling, norm fitting, optimizer and scheduler settings, extra validation
    # scores and sqrt_train, all unused by training.py
    def fit(self, train_sys_data, val_sys_data, epochs=30, n_its=None, batch_size=256, loss_kwargs={}, \
            validation_measure='sim-NRMS', its_per_val='epoch', cuda=False):
        """Run Adam and the optional PS2 opening; finish with the weights selected by validation.

        Args:
            train_sys_data, val_sys_data: training and validation records.
            epochs: Adam budget in epochs.
            n_its: Adam budget in updates instead, or None.
            batch_size: windows per update.
            loss_kwargs: nf and stride of the windows.
            validation_measure: score label for logging; the simulator computes the score.
            its_per_val: updates between validations, or 'epoch'.
            cuda: run the updates on the GPU.
        """

        def validation(train_loss=None, time_elapsed_total=None):
            """Return the validation score; save _best if it improves or ties, and always save _last."""
            # CHANGED: validate on the training device (a CPU round trip recompiles the training step); update theta_aux
            self.eval()
            self.sync_obc_coefficient(refresh_basis=False)
            Loss_val = self.cal_validation_error(val_sys_data)
            self.Loss_val.append(Loss_val)

            self.Loss_train.append(train_loss)
            self.time.append(time_elapsed_total)
            self.batch_id.append(self.batch_counter)
            self.epoch_id.append(self.epoch_counter)
            # >=: on equal errors the later weights become the best
            if self.bestfit>=Loss_val:
                self.bestfit = Loss_val
                # saved on the training device; training.py's resolve_checkpoint maps GPU checkpoints to the CPU on load
                self.checkpoint_save_system()
            # CHANGED: save _last at every validation so interrupted runs can resume
            self.checkpoint_save_system(name='_last')
            self.train()
            return Loss_val

        # 1. Windows, data loader and the initial validation.
        # CHANGED: reset counters and history for this job; training.py tracks earlier jobs
        # CHANGED: removed self.dt (unused) and the move to the GPU (training.py does it with on_device)
        self.train()

        self.epoch_counter = 0
        self.batch_counter = 0

        data_train = self.make_training_data(self.norm.transform(train_sys_data), **loss_kwargs)
        print_array_byte_size(sum([d.nbytes for d in data_train]))

        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = [], [], [], [], []

        # budget: n_its updates, epochs x updates per epoch unless given; one epoch = one pass over all windows
        Loss_acc_val_loop, best_it = 0, 0
        N_training_samples = len(data_train[0])
        batch_size = min(batch_size, N_training_samples)
        N_batch_updates_per_epoch = N_training_samples//batch_size
        n_its = int(N_batch_updates_per_epoch*epochs) if n_its is None else n_its
        its_per_val = N_batch_updates_per_epoch if its_per_val=='epoch' else its_per_val
        print(f'N_training_samples = {N_training_samples}, batch_size = {batch_size}, N_batch_updates_per_epoch = {N_batch_updates_per_epoch}')

        # CHANGED: use DtypeDataLoader to prevent float64 windows being cast to float32
        model_dtype = next(self.hfn.parameters()).dtype
        data_train_loader = DtypeDataLoader(data_train, batch_size=batch_size,
                                            dtype=model_dtype)

        # validate before the first update so the starting weights can be selected, including on resume
        validation(train_loss=float('nan'), time_elapsed_total=0)
        print(f'Initial Validation {validation_measure}=', self.Loss_val[-1])

        try:
            # 2. Adam updates, each followed by the PS2 step; validation every its_per_val.
            start_t = time.time()

            bestfit_old = self.bestfit
            for it_count, train_batch in zip(range(n_its), loop_dataset(data_train_loader)):
                if cuda:
                    train_batch = [b.cuda() for b in train_batch]
                # CHANGED: pass each window's controller index and error weight to loss alongside its signals
                batch_kwargs = dict(loss_kwargs, **dict(zip(self.simulator.extra_array_names, train_batch[4:])))

                # one Adam update on V of this minibatch
                def closure():
                    """V and its gradient on the current minibatch."""
                    Loss = self.loss(*train_batch[:4], **batch_kwargs)
                    self.optimizer.zero_grad()
                    Loss.backward()
                    return Loss

                training_loss = self.optimizer.step(closure).item()

                if np.isnan(training_loss):
                    print(f'&&&&&&&&&&&&& Encountered a NaN value in the training loss at it {it_count}, breaking from loop &&&&&&&&&&')
                    break
                # CHANGED: run PS2 after each Adam update; record its progress so loading can reject unfinished openings
                if self.training_phase is not None:
                    self.training_phase.after_step(self)
                    self.ps2_opening = (self.training_phase.phase, self.training_phase.it)

                # training loss summed over the validation interval, for its mean in the log
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

        # 3. Select: reload the best checkpoint.
        # CHANGED: rebuild OBC at the selected beta and resave it so the checkpoint includes the updated correction
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
        if self.sync_obc_coefficient(refresh_basis=True):
            self.checkpoint_save_system(name=selected_checkpoint)
        # CHANGED: log says "update" instead of Jan's "epoch", since best_it counts updates
        print(f'Loaded model with best known validation {validation_measure} of {self.bestfit:6.4}, selected at update {best_it}; last validation epoch_id={self.epoch_id[-1]:.2f}')


@added
class AugmentedModel(SSE_Interconnect):
    """Weighted closed-loop training with OBC and checkpoint handling."""

    # OBCProjection on the OBC arm, else None
    obc = None
    # exclude these objects from checkpoints; preserve their current instances when loading
    _UNPICKLABLE = ('obc', 'training_phase', 'fit_history')
    # save the current job's loss and counters separately before loading the selected checkpoint
    _FIT_HISTORY = ('batch_id', 'epoch_id', 'time', 'Loss_train')

    def __init__(self, *args, **kwargs):
        """Locate the network and baseline blocks used by the loss and OBC projection."""
        super().__init__(*args, **kwargs)
        blocks = tuple(self.hfn.connected_blocks)

        # indices in connected_blocks: _ann_ix the network phi_aug, _mats_ix the physics block
        self._ann_ix = next((i for i, m in enumerate(blocks)
                             if isinstance(m, Static_ANN_Block)), None)
        self._mats_ix = next((i for i, m in enumerate(blocks) if hasattr(m, 'pass_mats')), None)

    def _epoch_index(self):
        """Current epoch index for the OBC basis refresh."""
        # HEURISTIC: + 1e-6, since the summed epoch counter can end just below an integer
        return math.floor(float(getattr(self, 'epoch_counter', 0.0)) + 1e-6)

    def sync_obc_coefficient(self, refresh_basis=False):
        """Update the stored OBC coefficient for prediction from the current network.

        Rebuild the basis first if requested or missing; no gradients are recorded.

        Args:
            refresh_basis: rebuild the basis at the current beta first.
        Returns:
            True if the OBC coefficient was updated; False without OBC.
        """
        if self.obc is None:
            return False
        correction = self.hfn.obc_correction
        epoch = self._epoch_index()
        with torch.no_grad():
            if refresh_basis or self.obc.basis is None:
                # J: rebuild the OBC basis at the current beta
                self.obc.refresh(self.hfn.connected_blocks[self._mats_ix].beta, epoch)
            # theta_aux: least-squares fit of the current network's correction F on J
            theta = self.obc.coefficient(self.hfn.connected_blocks[self._ann_ix])
            correction.set_theta_aux(theta, self.obc.basis.beta_lin)
        return True

    def objective_matrices(self, theta_aux=None):
        """Build baseline and OBC matrices once for reuse throughout the rollout.

        Args:
            theta_aux: (11,) coefficient, or None for the stored correction.theta_aux.
        Returns:
            dict with pass_mats and, on the OBC arm, obc_pass.
        """
        mats = {'pass_mats': self.hfn.connected_blocks[self._mats_ix].pass_mats()}
        correction = self.hfn.obc_correction
        if correction is not None:
            mats['obc_pass'] = correction.correction_matrices(
                correction.theta_aux if theta_aux is None else theta_aux)
        return mats

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        """Training objective V: the weighted output error of closed-loop rollouts from encoder states.

        V = mean over windows tau, steps k and axes of (W_r (y^n_k - y_hat^n_{k|tau}))^2.

        Args:
            uhist, yhist: (B, 30, 3) past u^n, y^n (na = nb = 30 samples).
            ufuture, yfuture: (B, n_f, 3) recorded u^n, y^n of the window.
            Loss_kwargs: ctrl_ix (B,), loss_w (B, 3) = diag of W_r.
        Returns:
            scalar V.
        """
        # 1. Initial states from the past window: x_hat^n_{tau|tau} = psi(theta_enc, chi_tau).
        x = self.encoder(uhist, yhist)
        # 2. Recompute theta_aux from the current network on the fixed OBC tuples.
        theta_aux = None
        if self.obc is not None:
            correction = self.hfn.obc_correction
            # beta_lin = beta at the first objective of each epoch; the basis stays fixed within the epoch
            self.obc.maybe_refresh(self._epoch_index(), self.hfn.connected_blocks[self._mats_ix].beta)
            # theta_aux = (J^T J)^-1 J^T F^n_aug at every objective, differentiable in theta_aug
            theta_aux = self.obc.coefficient(self.hfn.connected_blocks[self._ann_ix])
            # store a detached copy for prediction; use theta_aux below to retain the loss gradient
            correction.set_theta_aux(theta_aux, self.obc.basis.beta_lin)
        # 3. Closed-loop rollout over the window and the weighted squared output error.
        Loss_kwargs = dict(Loss_kwargs, **self.objective_matrices(theta_aux))
        # loss_w (windows, ny): W_r of each window's record
        w = Loss_kwargs.pop('loss_w')
        y_pred, _ = self.simulate(x, ufuture, yfuture, **Loss_kwargs)
        # W_r = diag(sigma_y / sigma_e,r) turns the normalised error into the error over the record's servo-error
        # scale sigma_e,r, the same scaling as the validation score NRMS_e
        return ((y_pred - yfuture) * w[:, None, :]).square().mean()

    def checkpoint_save_system(self, name='_best'):
        """Write the checkpoint to a temporary file, then replace the previous checkpoint."""
        # an interrupted write leaves the previous checkpoint intact
        file = os.path.join(get_work_dirs()['checkpoints'], self.name + name + '.pth')
        torch.save(self.__getstate__(), file + '.tmp')
        _replace_with_retry(file + '.tmp', file)

    def checkpoint_load_system(self, *args, **kwargs):
        """Load weights and checkpoint state; keep the current OBC projection, PS2 opening and job loss history."""
        self.fit_history = {k: self.__dict__[k] for k in self._FIT_HISTORY if k in self.__dict__}
        stash = {k: self.__dict__.get(k) for k in self._UNPICKLABLE}
        out = super().checkpoint_load_system(*args, **kwargs)
        for k, v in stash.items():
            if v is not None:
                self.__dict__[k] = v
        # the kept OBC projection must build J and F^n_aug from the loaded model's correction, not the replaced one
        if self.obc is not None:
            correction = self.hfn.obc_correction
            self.obc.reference_field, self.obc.basis_builder = correction.network_field, correction.basis
        return out

    def __getstate__(self):
        """Checkpoint state without the OBC projection, PS2 opening or separately kept job history."""
        return {k: v for k, v in self.__dict__.items() if k not in self._UNPICKLABLE}
