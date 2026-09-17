from deepSI.fit_systems.encoders import SS_encoder_general

from torch import nn, Tensor
import torch
import numpy as np
import warnings
import time
import itertools
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy

from deepSI.fit_systems.fit_system import print_array_byte_size, My_Simple_DataLoader, Tictoctimer, loop_dataset

from model_augmentation.utils.utils import detect_algebraic_loop
from model_augmentation.utils.utils import added  # CHANGED: marker for project-added classes (D-076)
from model_augmentation.fit_systems.blocks import Block, Parameterized_Linear_Output_Block, Parameterized_MSD_State_Block, Parameterized_Linear_State_Block, Static_ANN_Block  # CHANGED: Static_ANN_Block for SSE_Interconnect_Composed's orth term
from model_augmentation.utils.deepSI_corrections import fixed_System_data_norm

class Interconnect(nn.Module):
    def __init__(self, nx, nu, ny, debugging=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.nb = None # batch dimension

        self.nr_blocks = 0
        self.connected_blocks = nn.ModuleList([])

        self.input_signal_sizes = [nx, nu] # nx, nu, nw2, ...
        self.output_signal_sizes = [nx, ny] # nxp, ny, nz2 , ...

        self.signal_connections = []

        self.initialized_forward_function = False
        self.debugging = debugging

        self.first_step_eval = True
        self.save_signals = False

    # CHANGED (D-169): make `array_connection_matrices` follow the module's device and dtype.
    #
    # They are a plain nested LIST of tensors, not registered buffers, so `.cuda()`, `.cpu()` and
    # `.to(dtype)` never touched them. Until now that was papered over by the per-call
    # `.to(device=x.device, dtype=x.dtype)` checks in forward() and output_only(), which re-home
    # them lazily on the next call.
    #
    # WHY THAT IS NOT GOOD ENOUGH. deepSI's fit() moves the model to the CPU around every
    # validation (interconnect.py:716,734), and the FIRST validation runs before training even
    # starts (:802). So on a CUDA run the matrices are sitting on the CPU when the first training
    # step is traced, `.to(device=...)` becomes a real CPU->CUDA copy inside the graph, and the
    # CPU tensor enters it as an input. inductor then disqualifies CUDA graphs for the whole
    # rollout:
    #     "skipping cudagraphs due to cpu device (primals_2) ... in output_only"
    # `mode='reduce-overhead'` silently degrades to plain `inductor`, ~2.6x instead of ~6.5x, with
    # no error and no recompile. Measured on blade1, job 80688.
    #
    # `_apply` is what .to()/.cuda()/.cpu()/.float()/.double() all funnel through, so overriding
    # it once covers every one of them. NOT gated on CUDA: this is "keep these tensors in sync
    # with the module", which is correct on every device. It also makes `use_f64` work by the
    # normal `.to(dtype)` route instead of depending on the lazy per-call heal.
    #
    # The per-call checks in forward() and output_only() are deliberately LEFT IN PLACE. With this
    # override they never have work to do (`.to()` on a tensor already on the target device
    # returns that same tensor, so nothing enters the graph), and keeping them preserves the
    # self-healing behaviour that the ~20 diagnostics building an Interconnect by hand may rely on.
    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        acm = getattr(self, 'array_connection_matrices', None)
        if acm is not None:
            self.array_connection_matrices = [[fn(t) for t in row] for row in acm]
        return out

    # CHANGED (D-169): the device/dtype the connection matrices should live at.
    # Read off the module's own tensors rather than stored, so it cannot go stale. Falls back to
    # the CPU default when the model holds no tensors yet, which is the state `init_forward` runs
    # in for a bare Interconnect built by a diagnostic.
    def _module_device_dtype(self):
        for t in itertools.chain(self.parameters(), self.buffers()):
            return t.device, (t.dtype if t.is_floating_point() else None)
        return None, None

    def init_model(self, sys_data):
        # x, u = make_tensors_from_sys_data(sys_data)
        # return

        for block in self.connected_blocks:
            block.init_block(torch.empty((0)))

    def forward(self, x: Tensor, u: Tensor):
        # reshape state and input tensor dimensions for use in interconnect
        x_size = x.size()
        if len(x.size()) <= 2:
            x = x.view(x.size(0), self.nx, 1)
            state_has_correct_dimension = False
        else:
            state_has_correct_dimension = True

        u_size = u.size()
        if len(u.size()) <= 2:
            u = u.view(u.size(0), self.nu, 1)
            input_has_correct_dimension = False
        else:
            input_has_correct_dimension = True

        assert x.size(0) == u.size(0) # batch dimension is currently required
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

        # CHANGED: propagate device/dtype so pipeline works on GPU and with float16/float32
        if self.array_connection_matrices[0][0].device != x.device or self.array_connection_matrices[0][0].dtype != x.dtype:
            for i in range(len(self.array_connection_matrices)):
                for j in range(len(self.array_connection_matrices[i])):
                    self.array_connection_matrices[i][j] = \
                        self.array_connection_matrices[i][j].to(device=x.device, dtype=x.dtype)

        for output_signal_ix in self.order_output_signal_computation:
            for input_signal_ix in self.output_ix_sorted_input_ix_dependencies[output_signal_ix]:
                output_signals[output_signal_ix] += torch.matmul(self.array_connection_matrices[output_signal_ix][input_signal_ix], input_signals[input_signal_ix])

            if output_signal_ix >= 2:
                input_signals[output_signal_ix] = self.connected_blocks[output_signal_ix-2].forward(output_signals[output_signal_ix]) # offset by two for connected blocks since the progressed state and output are not registered as blocks

        y = output_signals[1]
        xp = output_signals[0]

        # save input signals for referencing purpose
        if self.nb == 1 and self.save_signals == True:
            concat_input_signals = np.concatenate(input_signals, axis=1)[0,:,:]
            concat_output_signals = np.concatenate(output_signals, axis=1)[0,:,:]
            if self.first_step_eval:
                self.saved_input_signals = concat_input_signals
                self.saved_output_signals = concat_output_signals
                self.first_step_eval = False
            else:
                self.saved_input_signals = np.append(self.saved_input_signals, concat_input_signals, axis=1) # type: ignore
                self.saved_output_signals = np.append(self.saved_output_signals, concat_output_signals, axis=1) # type: ignore


        if not state_has_correct_dimension: xp = xp.view(self.nb, self.nx)
        if self.ny == 1: y = y.view(self.nb)
        if self.ny >= 2: y = y.view(self.nb, self.ny)

        assert x_size == xp.size()
        
        return y, xp
    
    def reset_saved_signals(self):
        self.save_signals = True
        self.saved_input_signals = None
        self.saved_output_signals = None
        self.first_step_eval = True

    def init_forward(self):
        self.n_output_signals = self.nr_blocks+2
        self.n_input_signals = self.nr_blocks+2
        self.n_nodes = 4 + self.nr_blocks # each block connected block + input, output, state and progressed state signal are a node

        # forward function required variables
        self.array_connection_matrices = [[torch.empty((0,0)) for i in range(self.n_output_signals)] for j in range(self.n_input_signals)]
        self.output_ix_sorted_input_ix_dependencies = []

        directional_signal_connection_matrix = np.zeros((self.n_nodes, self.n_nodes))
        output_ix_sorted_signal_connections = [[] for i in range(self.n_output_signals)]

        # for all signals add them to the adjacency matrix
        for signal_connection in self.signal_connections:
            input_signal_ix = signal_connection.input_signal_ix
            output_signal_ix = signal_connection.output_signal_ix

            shifted_output_signal_ix = output_signal_ix
            if shifted_output_signal_ix <= 1: shifted_output_signal_ix += self.n_nodes - 2 # for nodes structure the output signals xp, y counts as seperate node from x, u
            directional_signal_connection_matrix[shifted_output_signal_ix, input_signal_ix] = 1

            output_ix_sorted_signal_connections[output_signal_ix].append(signal_connection)

        connection_interconnect_matrix = np.roll(directional_signal_connection_matrix[2:,:-2], 2, axis=0)

        if self.debugging: print(connection_interconnect_matrix)

        assert not detect_algebraic_loop(directional_signal_connection_matrix)

        # determine order of block computation in forward function
        self.order_output_signal_computation = []
        connection_interconnect_matrix[:,0] = 0 # state signal is already available
        connection_interconnect_matrix[:,1] = 0 # input signal is already available
        
        while len(self.order_output_signal_computation) < self.n_output_signals:
            computable_elements = np.argwhere(np.sum(connection_interconnect_matrix, axis=1)==0).flatten()
            for element in computable_elements:
                if element not in self.order_output_signal_computation:
                    # CHANGED (D-169): int(), not the raw numpy scalar. `computable_elements` comes
                    # from np.argwhere(...).flatten(), so these were numpy.int64. This list is the
                    # loop variable of forward()'s main loop, where `if output_signal_ix >= 2`
                    # then branches on a numpy scalar -- which torch.compile cannot trace, one
                    # `generic_jump NumpyNdarrayVariable` graph break PER OUTPUT SIGNAL
                    # (measured: 11 breaks / 12 graphs in Interconnect.forward before this).
                    # A numpy.int64 and a Python int index a list and compare identically, so this
                    # cannot change any result; it only lets Dynamo specialise the branch away.
                    self.order_output_signal_computation.append(int(element))
                    connection_interconnect_matrix[:,element] = 0

        if self.debugging: print("Order of computation: " + str(self.order_output_signal_computation))

        # initialize the connection matrices and determine the input signal dependencies for all output signals
        for output_signal_ix in range(self.n_output_signals):
            input_ix_dependencies = self.init_connection_matrices(output_signal_ix, output_ix_sorted_signal_connections[output_signal_ix])
            self.output_ix_sorted_input_ix_dependencies.append(input_ix_dependencies)

        if self.debugging: print("Output signal dependencies: " + str(self.output_ix_sorted_input_ix_dependencies))

        # CHANGED (closed-loop seam): resolve the OUTPUT's dependency cone once, here, so
        # output_only() is a fixed short list rather than a graph traversal per call. Doing this
        # lazily per timestep would cost more than the forward it replaces.
        self.output_cone = self._signal_cone(1)
        if self.debugging: print("Output cone: " + str(self.output_cone))

        # CHANGED (D-169): place the matrices on the module's device/dtype NOW.
        # `init_forward` is lazy (it runs on the first forward, :69-71) and builds these on the
        # CPU, so on a CUDA model they would start life on the wrong device and only be corrected
        # by the per-call heal -- which is exactly the state that costs CUDA graphs their fast
        # path. `_apply` keeps them in sync from here on; this puts them in sync to begin with.
        _dev, _dt = self._module_device_dtype()
        if _dev is not None:
            self.array_connection_matrices = [
                [t.to(device=_dev, dtype=_dt) if (_dt is not None and t.is_floating_point())
                 else t.to(device=_dev) for t in row]
                for row in self.array_connection_matrices]

    # CHANGED (closed-loop seam): added, see output_only().
    def _signal_cone(self, target_output_ix):
        '''Output-signal indices that must be computed to obtain output signal `target_output_ix`.

        Walks the dependency graph backwards: output signal k needs input signals
        `output_ix_sorted_input_ix_dependencies[k]`, and an input signal j >= 2 is produced by
        block j-2 from output signal j (signals 0 and 1 are x and u, which are given). Returned in
        `order_output_signal_computation` order so evaluation is a straight loop.
        '''
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

    # CHANGED (closed-loop seam): added. A closed-loop rollout needs y = h(x) BEFORE it can form
    # u, while forward() returns (y, x_next) together. Without this a rollout must call the model
    # twice per timestep, which doubles the FP-plus-ANN forward cost of every step; that is what
    # the predecessor implementation did.
    def output_only(self, x, u=None):
        '''y from the output signal's dependency cone alone, without advancing the state.

        On the gantry model the cone is one block: y <- signal 3 <- (x, u), so this costs one
        Linear_Output_Block forward and a matmul and touches neither the ANN nor the state block.
        That is the real saving, not the avoidance of a second call.

        `u` defaults to zeros. The wiring DOES route u into the output block
        (connect_block_signals(out_phys, ["u"], ["y"])), so a feedthrough is permitted by the
        graph and D_d = 0 holds only because the block's own coefficients are zero on those
        columns. That is why the numerical no-feedthrough gate is not redundant with a structural
        argument and must be kept: this default is an assumption the gate checks, not a fact the
        graph guarantees.
        '''
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

        if self.array_connection_matrices[0][0].device != x.device or self.array_connection_matrices[0][0].dtype != x.dtype:
            for i in range(len(self.array_connection_matrices)):
                for j in range(len(self.array_connection_matrices[i])):
                    self.array_connection_matrices[i][j] = \
                        self.array_connection_matrices[i][j].to(device=x.device, dtype=x.dtype)

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
        if self.debugging: print("Connection matrices for " + self.convert_signal_ix_to_name(output_signal_ix, "output"))

        n_out = self.output_signal_sizes[output_signal_ix]
        n_out_total = 0

        concat_signal_connections = []
        additive_signal_connections = []
        add_to_signal_connections = []

        input_signal_ixs = []

        # split signal connection into lists based on connection method to be applied
        for signal_connection in signal_connections:
            if signal_connection.connection_function_method == "concatenation":
                concat_signal_connections.append(signal_connection)
            if signal_connection.connection_function_method == "additive":
                additive_signal_connections.append(signal_connection)
            if signal_connection.connection_function_method == "add_to":
                add_to_signal_connections.append(signal_connection)

        if self.debugging:
            print("concat signals: " + str(concat_signal_connections))
            print("additive signals: " + str(additive_signal_connections))
            print("add_to signals: " + str(add_to_signal_connections))


        # ensure that at least on concat method is present or change one additive into concat
        if len(concat_signal_connections) == 0 and len(additive_signal_connections) != 0:
            if self.debugging: print("Additive signal changed to concatenation signal.")
            concat_signal_connections.append(additive_signal_connections.pop(0))
        if len(concat_signal_connections) == 0 and len(additive_signal_connections) == 0 and len(add_to_signal_connections) != 0:
            raise ValueError("Add_to signal cannot be only signal connection method.")

        # determine connection matrices for concatenation based connection method
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

        # existing connection matrices should now have dimension nz x ...
        assert n_out_total == n_out, "total: {0}, required: {1}".format(n_out_total, n_out)

        # determine connection matrices for additive based connection method
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

        # determine connection matrices for add_to based connection method
        for signal_connection in add_to_signal_connections:
            raise NotImplementedError

        # check whether all connection matrices have the correct dimensions
        if self.debugging: print("input signal ixs: " + str(input_signal_ixs))

        for signal_connection in signal_connections:
            connection_matrix = self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix]
            if not connection_matrix.numel(): raise ValueError("Connection matrix should not be empty.")
            assert connection_matrix.size(0) == n_out

            if self.debugging: print(self.convert_signal_ix_to_name(signal_connection.input_signal_ix, "input") + ": " + \
                                      str(self.array_connection_matrices[output_signal_ix][signal_connection.input_signal_ix]))
                
        return input_signal_ixs
        
    def add_block(self, new_block: Block, name=None):
        assert new_block not in self.connected_blocks

        self.nr_blocks += 1
        new_block.block_ix = self.nr_blocks + 1 # index is offset by 1 from number of blocks because of external signals # type: ignore
        new_block.name = name if isinstance(name, str) else "Block_" + str(self.nr_blocks+1)
        self.connected_blocks.append(new_block)

        self.input_signal_sizes.append(new_block.nw)
        self.output_signal_sizes.append(new_block.nz)

        if self.debugging: print("Added block to interconnect with name: " + new_block.name + " and signals nw=" + str(new_block.nw) + ", nz=" + str(new_block.nz))
    
    def connect_signals(self, input_signal, output_signal, connection_function_method=None, connection_matrix=torch.empty(0,0), add_to_input_signal_ix=None):     
        input_signal_ix = self.determine_signal_ix(input_signal)
        output_signal_ix = self.determine_signal_ix(output_signal)
        if add_to_input_signal_ix != None: add_to_input_signal_ix = self.determine_signal_ix(add_to_input_signal_ix)

        if not isinstance(connection_function_method, str) and output_signal_ix >= 2: connection_function_method = "concatenation" # default for internal signals is concatenation # type: ignore
        if not isinstance(connection_function_method, str) and output_signal_ix <= 1: connection_function_method = "additive" # default for progressed state and output is additive method # type: ignore
        connection_function_method = self.parse_connection_function_method(connection_function_method)
        assert connection_function_method in ["concatenation", "additive", "add_to"]
        
        self.signal_connections.append(Signal_Connection(input_signal_ix, output_signal_ix, connection_function_method=connection_function_method, \
                                        connection_matrix=connection_matrix, add_to_input_signal_ix=add_to_input_signal_ix))
        
        if self.debugging: print("Connecting input " + self.convert_signal_ix_to_name(input_signal_ix, "input") + ": n=" + str(self.input_signal_sizes[input_signal_ix]) + " ," + \
                                 " with output " + self.convert_signal_ix_to_name(output_signal_ix, "output") + ": n=" + str(self.output_signal_sizes[output_signal_ix]) \
                                    + " with type: " + connection_function_method)

    def connect_block_signals(self, block, input_signal_list: list, output_signal_list: list):
        # if not isinstance(input_signal_list, list): input_signal_list = (input_signal_list)
        for input_signal in input_signal_list:
            self.connect_signals(input_signal, block)

        # if not isinstance(output_signal_list, list): output_signal_list = (output_signal_list)
        for output_signal in output_signal_list:
            self.connect_signals(block, output_signal)

    def determine_signal_ix(self, signal):
        if isinstance(signal, Block):
            return signal.block_ix
        if isinstance(signal, int):
            return signal
        if isinstance(signal, str):
            return self.convert_signal_name_to_ix(signal)
            
        raise TypeError("Input could not be converted to signal ix.")
    
    def convert_signal_name_to_ix(self, signal_name: str):
        if signal_name in ["x", "xp"]:
            return 0
        if signal_name in ["u", "y"]:
            return 1
        if len(signal_name) >= 1:
            signal_ix = int(list(signal_name)[1])
            assert signal_ix >= 2
            return signal_ix

        raise TypeError("Input could not be converted to signal ix.")
    
    def convert_signal_ix_to_name(self, signal_ix: int, signal_type: str):
        assert signal_type in ["input", "output"]

        if signal_type == "input":
            if signal_ix == 0:
                return "x"
            if signal_ix == 1:
                return "u"
            else:
                return "w" + str(signal_ix)
        if signal_type == "output":
            if signal_ix == 0:
                return "xp"
            if signal_ix == 1:
                return "y"
            else:
                return "z" + str(signal_ix)
        
        raise ValueError("Signal ix could not be converted to name.")

    def parse_connection_function_method(self, connection_function_method: str):
        connection_function_method = connection_function_method.lower()

        if connection_function_method in ["concatenation", "concat", "con", "cat", "c"]:
            return "concatenation"
        if connection_function_method in ["additive", "add", "a", "additional"]:
            return "additive"
        if connection_function_method in ["add_to", "to", "at", "add to"]:
            return "add_to"

class Signal_Connection():
    '''Object to hold information regarding connection between two signals in the interconnect'''
    def __init__(self, input_signal_ix, output_signal_ix, connection_function_method, connection_matrix = torch.empty((0,0)), add_to_input_signal_ix=None) -> None:
        self.input_signal_ix = input_signal_ix
        self.output_signal_ix = output_signal_ix

        self.connection_function_method = connection_function_method
        self.connection_matrix = connection_matrix
        self.add_to_input_signal_ix = add_to_input_signal_ix

    def __str__(self):
        str = '(in={0}, out={1}: method={2}, matrix={3})'.format(self.input_signal_ix, self.output_signal_ix, self.connection_function_method, bool(self.connection_matrix.numel()))
        return str

    def __repr__(self):
        return str(self)
    
class modified_encoder_net(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(modified_encoder_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), \
            n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, upast, ypast):
        # ypast = ypast[:,:,0] # <---------- To be disabled after training encoder: This prevents selects a single value from the state to be the output

        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1) # type: ignore
        return self.net(net_in)

@added
class Dtype_DataLoader(My_Simple_DataLoader):
    """My_Simple_DataLoader that keeps the training arrays at the MODEL's dtype.

    deepSI's `My_Simple_DataLoader.__init__` (fit_system.py:684) does

        self.data = [torch.as_tensor(d, dtype=torch.float32) for d in data]

    i.e. it hard-casts every training array to float32. With `use_f64=True` the model is
    float64 and the first batch then raises "expected scalar type Double but found Float".
    This is the ONLY thing that blocks the float64 toggle in the training path: the whole
    pipeline (data, norm, encoder, controller bank, rollout) already builds and runs in
    float64, which cl_direct_vs_residual.py:181-183 exercises via build_pipeline(use_f64=True).

    WHY THE PARENT CONSTRUCTOR IS BYPASSED rather than called-then-recast. `super().__init__`
    truncates to float32 first, so a later `.to(float64)` would widen float32 VALUES back into
    a float64 container: the extra digits are already gone and the run would carry float32
    precision while reporting float64. The float32 round-trip has to not happen at all, so the
    tensors are built at the target dtype directly. The rest of the parent's state (ids, rng,
    batch_size) is reproduced verbatim, including its seed=0 rng (the `seed` argument is unused
    upstream too -- kept in the signature so this stays a drop-in replacement).

    Integer arrays are left alone: the closed-loop seam appends `ctrl_ix`, a per-window integer
    row index into the controller bank (closed_loop.py:407), and casting that to a float dtype
    would break the gather it feeds.
    """

    def __init__(self, data, batch_size=32, seed=0, dtype=torch.float32):
        def _conv(d):
            t = torch.as_tensor(d)                 # always a tensor: the parent returns tensors
            return t.to(dtype) if t.is_floating_point() else t   # ints keep their own dtype
        self.data = [_conv(d) for d in data]
        self.ids = np.arange(len(data[0]), dtype=int)
        self.rng = np.random.default_rng(0)
        self.batch_size = batch_size


class SSE_Interconnect(SS_encoder_general):
    # CHANGED (closed-loop seam): how the model is DRIVEN during a loss rollout is a value on the
    # instance, not a position in the ParamLoss -> OrthLoss -> MultipleShooting chain. None is the
    # open-loop default and an exact no-op; a simulator object is attached after construction,
    # exactly as `orth_penalty` already is (D7.1/D7.8). Declared as a CLASS attribute on purpose:
    # a checkpoint pickled before this existed still resolves it through the class instead of
    # raising, and `checkpoint_load_system`'s `self.__dict__ = torch.load(...)` cannot silently
    # drop it the way it drops a patched bound method.
    simulator = None

    # CHANGED (closed-loop seam): declared extension point for DIAGNOSTICS during validation.
    # Two different concerns were riding on `cal_validation_error`: SELECTION, the scalar that
    # decides the best checkpoint, and diagnostics whose return value is discarded. Only the first
    # belongs on the simulator. Probes are called for their side effects and CANNOT replace the
    # value, so the ordering hazard disappears by construction: previously whichever of two
    # monkey patches was installed last decided selection.
    validation_probes = ()

    def __init__(self, na=5, nb=5, \
                 interconnect=Interconnect, e_net=modified_encoder_net,   e_net_kwargs={}, na_right=0, nb_right=0):

        super(SSE_Interconnect, self).__init__(nx=interconnect.nx, nb=nb, na=na, na_right=na_right, nb_right=nb_right) # type: ignore
        
        self.e_net = e_net
        self.e_net_kwargs = e_net_kwargs
        self.hfn = interconnect
        self.encoder = None
        # hf_net_kwargs['feedthrough'] = feedthrough
        # self.hf_net_kwargs = hf_net_kwargs
        
        self.norm = fixed_System_data_norm()
        
        self.multi_loss_val = np.array([])

    def init_nets(self, nu, ny): # a bit weird
        na_right = self.na_right if hasattr(self,'na_right') else 0
        nb_right = self.nb_right if hasattr(self,'nb_right') else 0
        if self.encoder is None:
            print('Initializing encoder network...')
            self.encoder = self.e_net(nb=self.nb+nb_right, nu=nu, na=self.na+na_right, ny=ny, nx=self.nx,**self.e_net_kwargs)

    def init_model(self, sys_data=None, nu=-1, ny=-1, device='cpu', auto_fit_norm=True, optimizer_kwargs={}, parameters_optimizer_kwargs={}, scheduler_kwargs={}):
        '''This function set the nu and ny, inits the network, moves parameters to device, initilizes optimizer and initilizes logging parameters'''
        if sys_data==None:
            assert nu!=-1 and ny!=-1, 'either sys_data or (nu and ny) should be provided'
            self.nu, self.ny = nu, ny
        else:
            self.nu, self.ny = sys_data.nu, sys_data.ny
            if auto_fit_norm:
                self.norm.fit(sys_data)
                
                
        self.init_nets(self.nu, self.ny)
        self.to_device(device=device)
        parameters_and_optim = [{**item,**parameters_optimizer_kwargs.get(name,{})} for name,item in self.parameters_with_names.items()]
        self.optimizer = self.init_optimizer(parameters_and_optim, **optimizer_kwargs)
        self.scheduler = self.init_scheduler(**scheduler_kwargs)
        self.bestfit = float('inf')
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        self.init_model_done = True

        self.hfn.init_model(sys_data) # type: ignore

    # CHANGED (closed-loop seam): the rollout is extracted from loss() into an overridable method.
    # It was the one thing buried in the middle of loss() with no seam, which is why the closed
    # loop previously had to override loss() wholesale and re-add param_loss and the orthogonality
    # penalty by hand in three files. A fourth hand-written copy is how the thesis contribution
    # gets silently dropped from the objective.
    def simulate(self, x, ufuture, yfuture=None, **Loss_kwargs):
        '''Roll the model forward from x. Returns (y_pred, x_final), y_pred (batch, nf, ny).

        Overridable seam: a subclass or an attached simulator that changes HOW the model is
        driven (closed loop, teacher forcing, a different integrator) replaces this and inherits
        every loss term unchanged. yfuture is passed so a driven rollout can use it; the open-loop
        default ignores it.

        The final state is returned because a rollout produces it and a caller may need it:
        multiple shooting forms its defect from exactly this state at each segment boundary, and
        without it that class would need its own rollout, i.e. a second implementation of the one
        thing this seam exists to keep singular. Callers that only want the prediction drop it.
        '''
        if self.simulator is not None:
            return self.simulator(self, x, ufuture, yfuture, **Loss_kwargs)
        hfn = self.hfn                       # bound once: nf Module.__getattr__ calls otherwise
        ys = []
        for u in ufuture.unbind(1):          # ONE dispatch for all nf views, not nf selects
            yhat, x = hfn(x, u) # type: ignore
            ys.append(yhat)
        return torch.stack(ys, dim=1), x

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states # type: ignore
        y_pred, _ = self.simulate(x, ufuture, yfuture, **Loss_kwargs)
        # CHANGED (closed-loop seam, step 2b): ONE mse_loss over the stacked prediction, replacing
        # a mean over nf per-timestep mse_loss values. Identical in value because every timestep
        # has the same element count, and MEASURED to be so: the difference is exactly 0.000e+00
        # on both reference arms. The gradient differs by one float32 ulp on the largest entries
        # (1 - cos = 6.1e-15) because the reduction order changes, which is 15 orders inside the
        # batch-to-batch scatter SGD already works with (1 - cos ~ 1.2).
        #
        # This is one dispatch and one autograd node instead of nf plus a stack and a mean, and
        # the backward pass is over half of a training step, so the nf forward nodes deleted here
        # take nf backward nodes with them. Step 2a kept the old reduction precisely so that this
        # change could be made on its own, with its own evidence, rather than hidden inside the
        # seam extraction.
        loss_MSE = nn.functional.mse_loss(yfuture, y_pred)

        has_theta_loss = False
        loss_theta = 0
        for m in self.hfn.connected_blocks: # type: ignore
            if isinstance(m, Parameterized_Linear_State_Block):
                loss_theta = loss_theta + m.param_loss()
                has_theta_loss = True
            elif isinstance(m, Parameterized_Linear_Output_Block):
                loss_theta = loss_theta + m.param_loss()
                has_theta_loss = True
            elif isinstance(m, Parameterized_MSD_State_Block):
                loss_theta = loss_theta + nn.functional.mse_loss(m.Lambda * m.params, m.Lambda * m.init_params, reduction="sum")
                has_theta_loss = True
        if has_theta_loss:
            # print(loss_theta)
            return loss_MSE + loss_theta
        else:
            return loss_MSE
    
    # CHANGED (closed-loop seam): make_training_data and cal_validation_error become seams. Both
    # are inherited from deepSI (SS_encoder_general and System_fittable respectively); these are
    # plain overrides in our file that call super() and then delegate, so the installed deepSI
    # package is NOT edited. With no simulator attached both are exact no-ops.
    def make_training_data(self, sys_data, **Loss_kwargs):
        '''deepSI's training arrays, plus whatever the attached simulator needs per window.'''
        data = super().make_training_data(sys_data, **Loss_kwargs)
        if self.simulator is not None and hasattr(self.simulator, 'augment_training_data'):
            return self.simulator.augment_training_data(data, sys_data, self, **Loss_kwargs)
        return data

    def cal_validation_error(self, val_sys_data, validation_measure='sim-NRMS'):
        '''The scalar fit() minimises over epochs, plus side-effect-only diagnostic probes.

        A driven rollout cannot be expressed as an `apply_experiment`: that interface drives the
        model with u in and y out, one step at a time, and cannot carry `y_data`, which a
        closed-loop free run needs to form the residual. deepSI's own docstring anticipates the
        override ("User given callback. (overwrite this function?)"). Hence a seam here rather
        than a patched attribute.
        '''
        if self.simulator is not None and hasattr(self.simulator, 'validation_error'):
            value = self.simulator.validation_error(self, val_sys_data, validation_measure)
        else:
            value = super().cal_validation_error(val_sys_data, validation_measure)
        for probe in self.validation_probes:
            probe(self, val_sys_data, value)      # side effects only, value never replaced
        return value

    def measure_act_multi(self,actions):
        actions = torch.tensor(np.array(actions), dtype=torch.float32) #(N,...)
        with torch.no_grad():
            y_predict, self.state = self.hfn(self.state, actions) # type: ignore
        return y_predict.numpy()
    
    # CHANGED: added fit() method replacing deepSI's original — supports multi-trajectory, sqrt loss, concurrent validation
    def fit(self, train_sys_data, val_sys_data, epochs=30, n_its=None, batch_size=256, loss_kwargs={}, \
            auto_fit_norm=True, validation_measure='sim-NRMS', optimizer_kwargs={}, its_per_val='epoch', concurrent_val=False, cuda=False, \
            timeout=None, verbose=2, sqrt_train=True, num_workers_data_loader=0, print_full_time_profile=False, scheduler_kwargs={}, list_val_measures=[]):
        '''The batch optimization method with parallel validation, 

        Parameters
        ----------
        train_sys_data : System_data or System_data_list
            The system data to be fitted
        val_sys_data : System_data or System_data_list
            The validation system data after each used after each epoch for early stopping. Use the keyword argument validation_measure to specify which measure should be used. 
        epochs : int
        batch_size : int
        loss_kwargs : dict
            The Keyword Arguments to be passed to the self.make_training_data and self.loss of the current fit_system.
        auto_fit_norm : boole
            If true will use self.norm.fit(train_sys_data) which will fit it element wise. 
        validation_measure : str
            Specify which measure should be used for validation, e.g. 'sim-RMS', '10-step-last-RMS', 'sim-NRMS_sys_norm', ect. See self.cal_validation_error for details.
        optimizer_kwargs : dict
            The Keyword Arguments to be passed on to init_optimizer. notes; init_optimizer['optimizer'] is the optimization function used (default torch.Adam)
            and optimizer_kwargs['parameters_optimizer_kwargs'] the learning rates and such for the different elements of the models. see https://pytorch.org/docs/stable/optim.html
        concurrent_val : boole
            If set to true a subprocess will be started which concurrently evaluates the validation method selected.
            Warning: if concurrent_val is set than "if __name__=='__main__'" or import from a file if using self defined method or networks.
        cuda : bool
            if cuda will be used (often slower than not using it, be aware)
        timeout : None or number
            Alternative to epochs to run until a set amount of time has past. 
        verbose : int
            Set to 0 for a silent run, 1 only print and 2 adds a progress bar.
        sqrt_train : boole
            will sqrt the loss while printing
        num_workers_data_loader : int
            see https://pytorch.org/docs/stable/data.html
        print_full_time_profile : boole
            will print the full time profile, useful for debugging and basic process optimization. 
        scheduler_kwargs : dict
            learning rate scheduals are a work in progress.
        
        Notes
        -----
        This method implements a batch optimization method in the following way; each epoch the training data is scrambled and batched where each batch
        is passed to the self.loss method and utilized to optimize the parameters. After each epoch the systems is validated using the evaluation of a 
        simulation or a validation split and a checkpoint will be crated if a new lowest validation loss has been achieved. (or concurrently if concurrent_val=True)
        After training (which can be stopped at any moment using a KeyboardInterrupt) the system is loaded with the lowest validation loss. 

        The default checkpoint location is "C:/Users/USER/AppData/Local/deepSI/checkpoints" for windows and ~/.deepSI/checkpoints/ for unix like.
        These can be loaded manually using sys.load_checkpoint("_best") or "_last". (For this to work the sys.unique_code needs to be set to the correct string)
        '''
        # CHANGED (D-169): the model no longer round-trips to the CPU for validation when a
        # simulator is attached.
        #
        # MEASURED COST OF THE ROUND TRIP (job 80695, blade1, nf=200, compiled with
        # mode='reduce-overhead'): the first training update AFTER a flip took 599.56 s and
        # 795.69 s, against a 3.30 s / 3.07 s eager baseline. Every flip forces Dynamo to
        # recompile, and 599.56/200 = 3.00 s and 795.69/200 = 3.98 s per timestep suggests it
        # recompiles on EVERY invocation until the guards restabilise, not once -- in which case
        # the cost scales with nf and would be hours per flip at nf = 12000. `.cpu()` and
        # `.cuda()` REPLACE the parameter tensors, which is what invalidates the guards.
        #
        # The flip was never needed on this path: ClosedLoopSimulator.validation_error builds its
        # tensors on the model's own device (closed_loop.py, closed_loop_free_run_rms_batch), so
        # validation runs correctly wherever the model already is. deepSI's OWN validators go
        # through apply_experiment and do assume the CPU, hence the guard rather than a deletion.
        # It keys on WHICH VALIDATOR is in use, not on device, and there is no device branch in
        # any hot path.
        #
        # Numerically this changes nothing: the same job measured the compiled arm's loss
        # trajectory and all three validation scores as identical to eager across the flips. The
        # flip was purely a performance fault, and an expensive one.
        _flip_for_validation = getattr(self, 'simulator', None) is None

        def validation(train_loss=None, time_elapsed_total=None):
            self.eval()
            if _flip_for_validation:
                self.cpu()
            Loss_val = self.cal_validation_error(val_sys_data, validation_measure=validation_measure)
            self.Loss_val.append(Loss_val)
            temp_loss_stack = np.array([])
            for val_measure in list_val_measures:
                loss_val = self.cal_validation_error(val_sys_data, validation_measure=val_measure)
                temp_loss_stack = np.hstack((temp_loss_stack, loss_val))
                # print(loss_val)
            self.multi_loss_val = np.append(self.multi_loss_val, temp_loss_stack.T)
            
            self.Loss_train.append(train_loss)
            self.time.append(time_elapsed_total)
            self.batch_id.append(self.batch_counter)
            self.epoch_id.append(self.epoch_counter)
            if self.bestfit>=Loss_val:
                self.bestfit = Loss_val
                # CHANGED (D-169): with `_flip_for_validation` False the model is still on its
                # training device here, so this writes CUDA tensors. Reading them back on a
                # CPU-only machine needs map_location; the project's own loader passes it
                # (gantry_dynamic/training.py::load_checkpoint).
                self.checkpoint_save_system()
            if cuda and _flip_for_validation:
                self.cuda()
            self.train()
            return Loss_val
        
        ########## Initialization ##########
        if self.init_model_done==False:
            if verbose: print('Initilizing the model and optimizer')
            device = 'cuda' if cuda else 'cpu'
            optimizer_kwargs = deepcopy(optimizer_kwargs)
            parameters_optimizer_kwargs = optimizer_kwargs.get('parameters_optimizer_kwargs',{})
            if parameters_optimizer_kwargs:
                del optimizer_kwargs['parameters_optimizer_kwargs']
            self.init_model(sys_data=train_sys_data, device=device, auto_fit_norm=auto_fit_norm, optimizer_kwargs=optimizer_kwargs,\
                    parameters_optimizer_kwargs=parameters_optimizer_kwargs, scheduler_kwargs=scheduler_kwargs)
        else:
            if verbose: print('Model already initilized (init_model_done=True), skipping initilizing of the model, the norm and the creation of the optimizer')
            self._check_and_refresh_optimizer_if_needed() 


        if self.scheduler==False and verbose:
            print('!!!! Your might be continuing from a save which had scheduler but which was removed during saving... check this !!!!!!')
        
        self.dt = train_sys_data.dt
        if cuda: 
            self.cuda()
        self.train()

        self.epoch_counter = 0 if len(self.epoch_id)==0 else self.epoch_id[-1]
        self.batch_counter = 0 if len(self.batch_id)==0 else self.batch_id[-1]
        extra_t            = 0 if len(self.time)    ==0 else self.time[-1] #correct timer after restart

        ########## Getting the data ##########
        data_train = self.make_training_data(self.norm.transform(train_sys_data), **loss_kwargs)
        if not isinstance(data_train, Dataset) and verbose: print_array_byte_size(sum([d.nbytes for d in data_train]))

        #### transforming it back to a list to be able to append. ########
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = list(self.Loss_val), list(self.Loss_train), list(self.batch_id), list(self.time), list(self.epoch_id)

        #### init monitoring values ########
        Loss_acc_val_loop, it_counter_per_val_loop, val_counter, best_it, batch_id_start = 0, 0, 0, 0, self.batch_counter #to print the frequency of the validation step.
        N_training_samples = len(data_train) if isinstance(data_train, Dataset) else len(data_train[0])
        batch_size = min(batch_size, N_training_samples)
        N_batch_updates_per_epoch = N_training_samples//batch_size
        n_its = int(N_batch_updates_per_epoch*epochs) if n_its is None else n_its
        Loss_acc_print_loop = 0.
        its_per_val = N_batch_updates_per_epoch if its_per_val=='epoch' else its_per_val
        if verbose>0: 
            print(f'N_training_samples = {N_training_samples}, batch_size = {batch_size}, N_batch_updates_per_epoch = {N_batch_updates_per_epoch}')
        
        ### convert to dataset ###
        if isinstance(data_train, Dataset):
            persistent_workers = False if num_workers_data_loader==0 else True
            data_train_loader = DataLoader(data_train, batch_size=batch_size, drop_last=True, shuffle=True, \
                                   num_workers=num_workers_data_loader, persistent_workers=persistent_workers)
        else: #add my basic DataLoader
            # CHANGED: Dtype_DataLoader instead of My_Simple_DataLoader, which hard-casts the
            # training arrays to float32 and makes use_f64=True raise on the first batch. The
            # dtype is read off the model rather than passed in, so this stays correct if the
            # system is cast after construction (model.py casts hfn/encoder with .to(DTYPE_PT)).
            _model_dtype = next(self.hfn.parameters()).dtype
            data_train_loader = Dtype_DataLoader(data_train, batch_size=batch_size,
                                                 dtype=_model_dtype) #is quite a bit faster for low data situations

        if concurrent_val:
            self.remote_start(val_sys_data, validation_measure)
            self.remote_send(float('nan'), extra_t)
        else: #start with the initial validation 
            validation(train_loss=float('nan'), time_elapsed_total=extra_t) #also sets current model to cuda
            if verbose: 
                print(f'Initial Validation {validation_measure}=', self.Loss_val[-1])

        try:
            t = Tictoctimer()
            start_t = time.time() #time keeping
            rang = range(n_its) if timeout is None else itertools.count(start=0)
            if verbose>1:
                rang = tqdm(rang)

            if timeout is not None and verbose>0: 
                print(f'Starting indefinite training until {timeout} seconds have passed due to provided timeout')


            bestfit_old = self.bestfit
            t.start()
            t.tic('data get')
            for it_count, train_batch in zip(rang, loop_dataset(data_train_loader)):
                #Loss_acc_print_loop=0
                if cuda:
                    train_batch = [b.cuda() for b in train_batch]
                t.toc('data get')
                # CHANGED (closed-loop seam): arrays a simulator asked make_training_data to
                # append are passed to loss() BY NAME, not positionally. deepSI's convention is
                # `self.loss(*train_batch)`, which forced every loss() in the chain to carry a
                # *sim_args it did not use and only forwarded, and which produced a TypeError on
                # the first optimizer step that no batch-level gate could see. This is the only
                # call site in the path we run, so the convention is ours to fix here.
                batch_kwargs = dict(loss_kwargs)
                if len(train_batch) > 4:
                    names = getattr(self.simulator, 'extra_array_names', ())
                    if len(names) != len(train_batch) - 4:
                        raise RuntimeError(
                            'the training data carries %d array(s) beyond deepSI\'s four but the '
                            'simulator names %d of them (%s). make_training_data and '
                            'extra_array_names must agree, or an array reaches loss() unnamed.'
                            % (len(train_batch) - 4, len(names), names))
                    clash = set(names) & set(loss_kwargs)
                    if clash:
                        raise RuntimeError(
                            'simulator array name(s) %s collide with loss_kwargs of the same name; '
                            'one would silently overwrite the other' % sorted(clash))
                    batch_kwargs.update(zip(names, train_batch[4:]))

                def closure(backward=True):
                    t.toc('optimizer start')
                    t.tic('loss')
                    Loss = self.loss(*train_batch[:4], **batch_kwargs)
                    t.toc('loss')
                    if backward:
                        t.tic('zero_grad')
                        self.optimizer.zero_grad()
                        t.toc('zero_grad')
                        t.tic('backward')
                        Loss.backward()
                        t.toc('backward')
                    t.tic('stepping')
                    return Loss

                t.tic('optimizer start')
                training_loss = self.optimizer.step(closure).item()

                if np.isnan(training_loss):
                    if verbose>0: print(f'&&&&&&&&&&&&& Encountered a NaN value in the training loss at it {it_count}, breaking from loop &&&&&&&&&&')
                    break

                t.toc('stepping')
                if self.scheduler:
                    t.tic('scheduler')
                    self.scheduler.step()
                    t.tic('scheduler')
                
                Loss_acc_val_loop += training_loss
                Loss_acc_print_loop += training_loss
                it_counter_per_val_loop += 1
                self.batch_counter += 1
                self.epoch_counter += 1/N_batch_updates_per_epoch

                t.tic('val')
                if (it_count+1)%its_per_val==0:
                    if concurrent_val:
                        if self.remote_recv(): #only when it is idle
                            self.remote_send(Loss_acc_val_loop/it_counter_per_val_loop, time.time()-start_t+extra_t)
                            Loss_acc_val_loop, it_counter_per_val_loop, val_counter = 0., 0, val_counter + 1
                    else:
                        validation(train_loss=Loss_acc_val_loop/it_counter_per_val_loop, \
                               time_elapsed_total=time.time()-start_t+extra_t) #updates bestfit and goes back to cpu and back
                        Loss_acc_val_loop, it_counter_per_val_loop, val_counter = 0., 0, val_counter + 1
                t.toc('val')
                # t.pause()

                ######### Printing Routine ##########
                if verbose>0 and (it_count+1)%its_per_val==0:
                    if bestfit_old > self.bestfit:
                        print(f'########## New lowest validation loss achieved ########### {validation_measure} = {self.bestfit}')
                        best_it = it_count+1
                        bestfit_old = self.bestfit
                    if concurrent_val: #if concurrent val than print validation freq
                        val_feq = val_counter/(it_count+1)
                        valfeqstr = f', {val_feq:4.3} vals/it' if (val_feq>1 or val_feq==0) else f', {1/val_feq:4.3} its/val'
                    else: #else print validation time use
                        valfeqstr = f''
                    train_loss_epoch, Loss_acc_print_loop = Loss_acc_print_loop/its_per_val, 0
                    trainstr = f'sqrt loss {train_loss_epoch**0.5:7.4}' if sqrt_train and train_loss_epoch>=0 else f'loss {train_loss_epoch:7.4}'
                    Loss_val_now = self.Loss_val[-1] if len(self.Loss_val)!=0 else float('nan')
                    Loss_str = f'It {it_count+1:4}, {trainstr}, Val {validation_measure} {Loss_val_now:6.4}'
                    loss_time = (t.acc_times['loss'] + t.acc_times['optimizer start'] + t.acc_times['zero_grad'] + t.acc_times['backward'] + t.acc_times['stepping'])  /t.time_elapsed
                    time_str = f'Time Loss: {loss_time:.1%}, data: {t.acc_times["data get"]/t.time_elapsed:.1%}, val: {t.acc_times["val"]/t.time_elapsed:.1%}{valfeqstr}'
                    self.batch_feq = (self.batch_counter - batch_id_start)/(time.time() - start_t)
                    batch_str = (f'{self.batch_feq:4.1f} batches/sec' if (self.batch_feq>1 or self.batch_feq==0) else f'{1/self.batch_feq:4.1f} sec/batch')
                    print(f'{Loss_str}, {time_str}, {batch_str}')
                    if print_full_time_profile:
                        print('Time profile:',t.percent())
                t.tic('data get')

                ####### Timeout Breaking ##########
                if timeout is not None:
                    if time.time() >= start_t+timeout:
                        break
        except KeyboardInterrupt:
            print('Stopping early due to a KeyboardInterrupt')

        self.train(); self.cpu()
        del data_train_loader

        ####### end of training concurrent things #####
        if concurrent_val:
            if verbose: print(f'Waiting for started validation process to finish and one last validation... (receiving = {self.remote.receiving})',end='')
            if self.remote_recv(wait=True):
                if verbose: print('Recv done... ',end='')
                if it_counter_per_val_loop>0:
                    self.remote_send(Loss_acc_val_loop/it_counter_per_val_loop, time.time()-start_t+extra_t)
                    self.remote_recv(wait=True)
            self.remote_close()
            if verbose: print('Done!')

        
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array(self.Loss_val), np.array(self.Loss_train), np.array(self.batch_id), np.array(self.time), np.array(self.epoch_id)
        self.checkpoint_save_system(name='_last')
        # CHANGED (D-172): keep this run's OWN history across the best-checkpoint restore.
        # `checkpoint_load_system` is `self.__dict__ = torch.load(file)` (fit_system.py:501), a
        # wholesale replacement, so the line below does not merely restore the best WEIGHTS: it
        # also substitutes the loss history as it stood at the best epoch, discarding everything
        # measured after it. The system this returns then silently disagrees with itself, holding
        # a full run's weights-selection outcome and a truncated record of how it got there.
        # Two bugs came out of that, both in code written to work around it rather than fix it:
        # `evaluation.py::capture_loss_history` re-read `_last.pth` from disk to recover state
        # this process held moments earlier, and that reload ALSO put stale weights back onto the
        # system, which after an accepted L-BFGS polish (D-171) meant every post-run diagnostic
        # silently reported the unpolished model.
        # Restoring the three arrays afterwards is all that is needed: the returned system then
        # has the best weights AND the complete history, which is what every reader assumes.
        _history = (self.epoch_id.copy(), self.Loss_val.copy(), self.Loss_train.copy())
        try:
            self.checkpoint_load_system(name='_best')
        except FileNotFoundError:
            print('no best checkpoint found keeping last')
        self.epoch_id, self.Loss_val, self.Loss_train = _history
        if verbose:
            print(f'Loaded model with best known validation {validation_measure} of {self.bestfit:6.4} which happened on epoch {best_it} (epoch_id={self.epoch_id[-1] if len(self.epoch_id)>0 else 0:.2f})')


@added
class SSE_Interconnect_ParamLoss(SSE_Interconnect):
    """
    SSE_Interconnect with generic parameter-regularization pickup (D-076).

    Drop-in replacement: loss() = SSE_Interconnect.loss() + sum of param_loss()
    over all connected blocks that expose it (hasattr sweep). Exact no-op when
    no block exposes param_loss, so it can be used unconditionally.

    Reimplementation of the D-032 idea (lpv_lfr_baseline lfr_fit_system);
    deliberately not imported from there to avoid a cross-pipeline dependency.

    Caveat: the parent loss() already adds regularization for Jan's
    Parameterized_Linear_State_Block / Parameterized_Linear_Output_Block /
    Parameterized_MSD_State_Block via isinstance checks. Combining one of
    those blocks with this class would count its penalty twice. This pipeline
    never uses them; do not mix.
    """

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        loss_theta = 0
        for m in self.hfn.connected_blocks:  # type: ignore
            if hasattr(m, "param_loss"):
                loss_theta = loss_theta + m.param_loss()
        return super().loss(uhist, yhist, ufuture, yfuture, **Loss_kwargs) + loss_theta


@added
class WindowErrorStats:
    """Residual statistics over the loss rollout's nf-step windows, in physical units.

    The rollout that loss() performs IS an nf-step prediction over every training window, so the
    error statistics below are a by-product of work already done: four detached reductions on a
    tensor that already exists. The alternative, and what this replaces, was a second open-loop
    pass per epoch that recomputed a worse version of the same thing.

    Because it accumulates SQUARES and takes the root once, `rms` is the exact RMS over every
    window it saw, not an average of per-window RMS values. Those differ (Jensen), and the
    per-window average is systematically smaller.

    DELIBERATELY NOT AN nn.Module. deepSI builds its optimizer parameter groups by scanning the
    fit-system for modules, so a module reachable both through `hfn` and through an attribute
    here puts every one of its parameters in two groups and Adam refuses to construct. Holding
    only tensors and floats keeps this object invisible to that scan.

    Units: `ystd` converts the normalised residual to metres, which is what makes these numbers
    comparable with a sim-RMS. Note that metres is NOT the objective's weighting, which weights
    each channel by 1/ystd**2; the two coincide only if ystd is equal across channels.
    """

    def __init__(self, ystd, dtype=None, device=None):
        y = torch.as_tensor(np.asarray(ystd, dtype=float).ravel())
        self.ystd = y.to(dtype=dtype, device=device) if dtype is not None else y
        self.reset()

    def reset(self):
        self._sq = 0.0           # sum e^2, all elements
        self._sq_ch = None       # sum e^2 per channel
        self._sum_ch = None      # sum e   per channel (the bias the RMS hides)
        self._sq_first = 0.0     # sum e^2 at the first step of the window
        self._sq_last = 0.0      # sum e^2 at the last step
        self._n = 0              # elements contributing to _sq
        self._n_win = 0          # windows seen
        self._n_step = 0         # elements contributing to _sq_first / _sq_last

    @torch.no_grad()
    def update(self, y_pred, y_true):
        """Accumulate one batch. y_pred, y_true: (batch, nf, ny), NORMALISED."""
        # CHANGED (D-169, job 80708): follow y_pred's DEVICE as well as its dtype. This object is
        # deliberately not an nn.Module (see the class docstring), so `.cuda()` never moves `ystd`
        # and it stays on the CPU for the life of the run. That was invisible while fit() flipped
        # the model to the CPU before every validation; with the flip gone, both instances of this
        # class -- fit_sys.loss_stats and the val probe's own -- meet CUDA tensors here.
        # The accumulators below are all derived from `e`, so they follow automatically.
        e = (y_pred - y_true).detach() * self.ystd.to(device=y_pred.device, dtype=y_pred.dtype)
        sq = e * e
        sq_ch = sq.sum(dim=(0, 1))
        self._sq += float(sq.sum())
        self._sq_ch = sq_ch if self._sq_ch is None else self._sq_ch + sq_ch
        s_ch = e.sum(dim=(0, 1))
        self._sum_ch = s_ch if self._sum_ch is None else self._sum_ch + s_ch
        self._sq_first += float(sq[:, 0, :].sum())
        self._sq_last += float(sq[:, -1, :].sum())
        self._n += e.numel()
        self._n_win += e.shape[0]
        self._n_step += e.shape[0] * e.shape[2]

    def summary(self):
        """rms and per-channel rms [m], the within-window growth ratio, and the bias [m]."""
        if not self._n:
            return None
        rms = (self._sq / self._n) ** 0.5
        chan = ((self._sq_ch / (self._n // self._sq_ch.numel())) ** 0.5).tolist()
        bias = (self._sum_ch / (self._n // self._sum_ch.numel())).tolist()
        first = (self._sq_first / self._n_step) ** 0.5
        last = (self._sq_last / self._n_step) ** 0.5
        # Growth ACROSS the window: >1 means the error is still opening up at the horizon the
        # loss stops at, which is the window-mismatch signal that needs no validation data.
        grow = (last / first) if first > 0 else float('nan')
        return dict(rms=rms, chan=chan, bias=bias, grow=grow, n_win=self._n_win)


@added
class SSE_Interconnect_Composed(SSE_Interconnect):
    """MSE plus composed penalty terms, in ONE class instead of a subclass per feature.

    Replaces the ParamLoss -> OrthLoss -> MultipleShooting chain, in which every optional loss
    term was a subclass, so the deepest link was instantiated whether or not its feature was on,
    the chain order became load-bearing, and removing a middle link was a breaking change. How
    the model is penalised is a VALUE here, not a position in a chain: the same rule `simulator`
    and `orth_penalty` already follow, and the reason closing the loop needed no fourth subclass.
    Any combination of the terms works, and a feature that is off contributes nothing.

    Deliberately does NOT call super().loss(). The parent picks up param_loss for three block
    types by isinstance and the sweep below finds them again by hasattr, so delegating would
    count them twice: the documented caveat on SSE_Interconnect_ParamLoss. Forming the objective
    in one place removes the caveat rather than restating it. The price is that the parent's
    inline handling of Parameterized_MSD_State_Block is not inherited either, which __init__
    turns into an error rather than a silently missing term.

    The rollout is NOT duplicated: it goes through the `simulate` seam, so an attached simulator
    (the closed loop) drives this class exactly as it drives the others.
    """

    orth_penalty = None     # class default; instance attribute set by the pipeline (D7.1/D7.8)
    loss_stats = None       # ditto: a WindowErrorStats, or None for no statistics at all
    # @added. The TRAJECTORY orthogonality penalty, which is NOT a loss term and therefore is not
    # in the sweep below. It is a gradient contribution applied between the base backward and the
    # optimizer update; see `fit()` and `trajectory_penalty_hook.py`. None = the feature is off
    # and `fit()` delegates straight to the parent, so a run without it is bit-identical.
    traj_penalty = None
    # Backing field for the `optimizer` property below. A class-level default so a checkpoint
    # pickled before the property existed still resolves it, exactly as `orth_penalty` and
    # `burn_in` do.
    _optimizer = None
    _traj_wrapping = False
    _traj_hook_calls = 0
    _traj_steps = 0
    # Samples excluded from the SCORE at the start of every window (D-178). 0 = score the whole
    # window, bit-identically to the pre-D-178 objective. Same class-attribute reasoning as the
    # two above: a checkpoint pickled before this existed still resolves it through the class.
    burn_in = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        blocks = tuple(self.hfn.connected_blocks)                         # type: ignore

        unsupported = sorted({type(m).__name__ for m in blocks
                              if isinstance(m, Parameterized_MSD_State_Block)})
        if unsupported:
            raise TypeError(
                '%s does not expose a param_loss() method: SSE_Interconnect regularises it INLINE '
                'inside its own loss(), and this class deliberately does not call that loss. Its '
                'theta term would therefore vanish from the objective without any error, which is '
                'the one failure this class must not have. Use SSE_Interconnect for models '
                'containing it, or give the block a param_loss() method.' % unsupported)

        # Resolved ONCE: the block set is fixed at construction, so scanning it inside loss()
        # would repeat the same two searches on every batch for an answer that cannot change.
        # INDICES, not module references: deepSI builds its optimizer parameter groups by
        # scanning this object for nn.Modules, so holding a block here as well as in hfn makes
        # every one of its parameters appear in two groups and Adam refuses to construct.
        self._param_ix = tuple(i for i, m in enumerate(blocks) if hasattr(m, 'param_loss'))
        self._ann_ix = next((i for i, m in enumerate(blocks)
                             if isinstance(m, Static_ANN_Block)), None)

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist)                                    # type: ignore
        y_pred, _ = self.simulate(x, ufuture, yfuture, **Loss_kwargs)
        # BURN-IN (D-178): score from sample `burn_in` onward. The window is transient-dominated
        # at nf=400 -- every run's `[nf]` line reports grow = RMS(last)/RMS(first) between 0.46
        # and 0.81, i.e. the error DECAYS across the window -- so scoring the start pays the
        # optimiser to sharpen x_0, which a 12 s free run pays once and forgets. Measured by
        # rescoring (cl_burnin_sweep.py): discrimination 1.249x -> 3.400x at K=100, against
        # 2.54x for an 8x longer window. Time is dim 1; `burn_in = 0` skips the branch entirely
        # and is bit-identical to the objective every run before this used.
        if self.burn_in:
            y_pred_scored, yfuture_scored = y_pred[:, self.burn_in:], yfuture[:, self.burn_in:]
            if yfuture_scored.shape[1] == 0:
                raise ValueError(
                    'burn_in=%d leaves nothing to score in an nf=%d window. It is a number of '
                    'SAMPLES discarded at the window start and must be smaller than nf.'
                    % (self.burn_in, yfuture.shape[1]))
        else:
            y_pred_scored, yfuture_scored = y_pred, yfuture
        L = nn.functional.mse_loss(yfuture_scored, y_pred_scored)
        # Window error statistics, from the rollout that just happened. Accumulated HERE, before
        # the penalties, so the reported error is the fit error and does not move when orth or
        # joint estimation is toggled. None = no statistics, an exact no-op.
        # DELIBERATELY the FULL window, not the scored slice (D-178): `grow` is the evidence that
        # motivates burn-in, and a diagnostic that moved with the knob it justifies is worthless.
        if self.loss_stats is not None:
            self.loss_stats.update(y_pred, yfuture)
        blocks = self.hfn.connected_blocks                                # type: ignore
        # Parameter regularisation, anchored to each block's init values (Jan's prior semantics).
        # Empty when no block is parameterised, i.e. the joint_estimation=False case.
        for i in self._param_ix:
            L = L + blocks[i].param_loss()
        # Orthogonal projection (D7.1/D7.2). Attachment IS the switch: RunConfig rejects
        # orth=True with beta<=0, so an attached penalty always has a non-zero weight and there
        # is no second condition to test here.
        if self.orth_penalty is not None:
            if self._ann_ix is None:
                raise RuntimeError(
                    'an orth penalty is attached but this interconnect has no Static_ANN_Block: '
                    'the penalty is a function of the ANN output and has nothing to act on.')
            L = L + self.orth_penalty(blocks[self._ann_ix])
        return L

    # @added. The TRAJECTORY penalty seam.
    #
    # WHY IT IS HERE AND NOT IN loss(). The parent's training closure is
    #
    #     Loss = self.loss(...)  ->  self.optimizer.zero_grad()  ->  Loss.backward()
    #
    # so anything backwarded inside `loss()` is wiped by the closure's own `zero_grad` a moment
    # later. That is measured, not feared: `checks_v6.check_accumulation_hook_survives_zero_grad`
    # runs both orders and the wrong one loses the penalty entirely. The specification's ordering
    # (Sect. 8.1) is zero_grad; backward J_base; accumulate eta-only gradients; clip; step.
    #
    # WHY IT IS A CLOSURE WRAPPER AND NOT A COPY OF fit(). The closure is defined inside the
    # parent's `fit()`, so a subclass cannot reach it without duplicating that ~100-line loop,
    # and a duplicate would drift from Jan's. Instead this override wraps the optimizer's `step`
    # so that `fit()` still passes ITS closure to what it believes is the optimizer, and the
    # wrapper interposes the accumulation between that closure returning and the update being
    # applied. Jan's `fit()` is not modified.
    #
    # THE REFRESH HAZARD, fixed 2026-09-08 after review. The first version wrapped
    # `self.optimizer` BEFORE calling `super().fit()`. The parent may then REPLACE the optimizer:
    # `init_model` constructs one when `init_model_done` is False, and
    # `_check_and_refresh_optimizer_if_needed` calls `_refresh_optimizer`, which does
    # `self.optimizer = optimizer_new` (deepSI fit_system.py:527-531) on a system loaded from a
    # file. Either path drops the wrapper, and the penalty then silently never fires -- on a
    # RESUME, which is exactly when it would be least noticed. The wrapper is now installed
    # through a property, so any assignment to `self.optimizer` re-wraps, and the step count is
    # reconciled against the hook's call count at the end of fit rather than assumed.
    # THE CHECKPOINT SEAM deepSI ACTUALLY USES, and it is not `__getstate__`.
    #
    # Found 2026-09-08 on the second real training run. `checkpoint_save_system` does
    # `torch.save(self.__dict__, file)` -- it pickles a PLAIN DICT, so the object's
    # `__getstate__` is never called and the exclusion added for it was inert. The run died with
    # "Can't pickle local object 'run_arm.<locals>.observer'", one layer deeper than the first
    # time and for the same underlying reason: the penalty hook must not be in a checkpoint.
    #
    # `checkpoint_load_system` is the mirror hazard and is worse, because it fails SILENTLY:
    # `self.__dict__ = torch.load(file)` REPLACES the dictionary wholesale, so a system that
    # loads its best checkpoint -- which deepSI does at the END of every fit -- would come back
    # without `traj_penalty` and without `_optimizer`. The first is a training-time object that
    # the caller still holds a reference to; the second would make the very next fit refuse to
    # start. Both are carried across the replacement here.
    _TRAJ_UNPICKLABLE = ('traj_penalty',)
    # Counters that must SURVIVE deepSI's end-of-fit `self.__dict__ = torch.load(...)`.
    # They live in __dict__ like everything else, so the load silently reset them to whatever
    # was in the checkpoint -- which was written at the FIRST validation, before any step. The
    # end-of-fit reconciliation then reported "no optimizer step" on a run that had taken three.
    # A bookkeeping bug, but one that made a correct run look broken, which is the same
    # category as the false alarm it replaced.
    _TRAJ_COUNTERS = ('_traj_wrapping', '_traj_hook_calls', '_traj_steps')

    def checkpoint_save_system(self, *args, **kwargs):
        stash = {k: self.__dict__.pop(k) for k in self._TRAJ_UNPICKLABLE
                 if k in self.__dict__}
        try:
            return super().checkpoint_save_system(*args, **kwargs)
        finally:
            self.__dict__.update(stash)

    def checkpoint_load_system(self, *args, **kwargs):
        stash = {k: self.__dict__.get(k)
                 for k in self._TRAJ_UNPICKLABLE + self._TRAJ_COUNTERS}
        out = super().checkpoint_load_system(*args, **kwargs)
        # The load replaced __dict__; put back what was never saved, and re-run the legacy
        # optimizer migration, because a file written before the property existed carries
        # `optimizer` rather than `_optimizer`.
        opt = self.__dict__.pop('optimizer', None)
        if opt is not None and self.__dict__.get('_optimizer') is None:
            self.__dict__['_optimizer'] = opt
        for k, v in stash.items():
            if v is not None:
                self.__dict__[k] = v
        # RE-WRAP AFTER A LOAD, if one happens while a fit is in progress.
        #
        # deepSI loads the best checkpoint at the END of every fit, and `self.__dict__ = ...`
        # installs a freshly unpickled optimizer that carries no wrapper. Two consequences, and
        # the second is the one that matters:
        #
        #   - the end-of-fit reconciliation saw an unwrapped optimizer and reported the run
        #     INVALID even though every step had had the penalty applied (3 of 3). A false alarm
        #     on a correct run is not harmless: it is the kind of thing that gets a guard deleted
        #   - a load DURING a fit -- which deepSI also does -- would have silently unwrapped the
        #     optimizer and trained the remainder without the penalty. That one is real, and it
        #     is the failure the property exists to prevent, arriving by the one route the
        #     property does not see.
        #
        # Re-wrapping here closes both. It is gated on `_traj_wrapping` so a load outside a fit
        # leaves a plain optimizer, which is what a caller inspecting a restored system expects.
        # REBIND THE ADAPTER BEFORE ANYTHING ELSE TOUCHES IT.
        #
        # Restoring the hook object is not enough, and this was a real defect: the hook's adapter
        # caches the encoder, the ANN block and the physical block as SNAPSHOTS taken when it was
        # built. This load has just replaced every one of them inside a system object whose
        # identity is unchanged, so without rebinding the adapter drives the NEW dynamics through
        # `fit_sys.simulate` while gating the OLD ANN block and owning the OLD parameters. The
        # gating half of the interventions then applies to a module outside the rollout, so the
        # penalty is computed for a DIFFERENT, unspecified intervention, and gradients land on
        # orphaned tensors. Note it does not simply vanish: `scored_stack` zeros the latent
        # initialisation for non-full modes regardless of the gate, so `d` generally stays
        # nonzero. Nothing raises, and the reported value still looks reasonable.
        hook = self.traj_penalty
        ad = getattr(hook, 'adapter', None) if hook is not None else None
        if ad is not None and hasattr(ad, 'rebind'):
            ad.rebind(self)
            if hasattr(ad, 'assert_bound'):
                ad.assert_bound()
            # The hook's cached group list, if it holds one, is stale for the same reason.
            if getattr(hook, '_groups', None) is not None:
                hook._groups = ad.parameter_groups()
        if (getattr(self, '_traj_wrapping', False) and self.traj_penalty is not None
                and self.__dict__.get('_optimizer') is not None):
            self._wrap_optimizer_step(self.__dict__['_optimizer'])
        return out

    def __getstate__(self):
        """Keep the trajectory penalty OUT of the checkpoint.

        Found 2026-09-08 by the first real training run, which died at the first validation:

            AttributeError: Can't pickle local object 'run_arm.<locals>.logged'

        deepSI's `checkpoint_save_system` pickles `self.__dict__` wholesale, and `traj_penalty`
        is in it. That is wrong on three counts, of which the crash is the least important:

          - the hook holds an ADAPTER, which holds the window tensors, the environment and the
            model itself, so the checkpoint would balloon and contain the training data
          - it holds the frozen GEOMETRY, which is already persisted as its own artifact with
            its own provenance. Two copies is one too many, and the checkpoint's copy carries no
            provenance at all
          - a checkpoint carrying a live penalty object cannot be loaded anywhere the adapter
            cannot be imported

        The penalty is training-time state, not model state: it is reattached by
        `train_orth.py` from the geometry artifact. Dropping it here also removes the pickling
        constraint from anything a caller attaches to the hook, such as live meters.
        """
        state = dict(self.__dict__)
        state.pop('traj_penalty', None)
        return state

    def __setstate__(self, state):
        """Migrate a checkpoint pickled BEFORE `optimizer` became a property.

        Added 2026-09-08 after review. `optimizer` is now a data descriptor on the class, and a
        data descriptor SHADOWS an instance `__dict__` entry of the same name. A system pickled
        before this change carries `__dict__['optimizer']`, so after unpickling `self.optimizer`
        would resolve through the property to `self._optimizer`, which is None: the restored
        optimizer would be silently LOST and `fit` would refuse to start (or, worse, a caller
        that tolerates None would train without the moments it thought it had). The stale entry
        is moved into the backing field and removed.
        """
        opt = state.pop('optimizer', None)
        if opt is not None and state.get('_optimizer') is None:
            state['_optimizer'] = opt
        self.__dict__.update(state)
        # `traj_penalty` was deliberately not saved (see __getstate__); fall back to the class
        # default so a restored system is a plain one until a penalty is reattached.
        self.__dict__.pop('traj_penalty', None)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt
        if opt is not None and getattr(self, 'traj_penalty', None) is not None \
                and getattr(self, '_traj_wrapping', False):
            self._wrap_optimizer_step(opt)

    _TRAJ_ALLOWED_OPTIMIZERS = (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD,
                                torch.optim.RMSprop, torch.optim.Adagrad)

    def _assert_optimizer_supported(self, opt):
        """Validate EVERY optimizer this seam accepts, not just the one `fit` started with.

        Added 2026-09-08 after review. The rejection used to live at the top of `fit()`, so a
        REPLACEMENT optimizer -- which is the whole reason the property exists -- was wrapped
        without being checked. An L-BFGS refresh would then have been accepted mid-run by the
        very mechanism added to make refreshes safe.

        Allow-list rather than deny-list: an optimizer nobody has reasoned about is refused, and
        the failure names it. A deny-list silently accepts whatever is added to torch next.
        """
        if isinstance(opt, torch.optim.LBFGS):
            raise RuntimeError(
                'L-BFGS is REFUSED with the trajectory penalty. Its line search evaluates the '
                'closure several times per step and picks a step length from the returned '
                'BASE-LOSS values, while the gradient it descends has an eta-only penalty term '
                'added. Those are two different objectives, so the search is inconsistent '
                'regardless of whether accumulation is skipped on the no-backward evaluations: '
                'suppressing it there hides the inconsistency rather than repairing it.')
        if not isinstance(opt, self._TRAJ_ALLOWED_OPTIMIZERS):
            raise RuntimeError(
                '%s has not been analysed against the trajectory penalty and is refused. The '
                'seam assumes one closure evaluation per step whose gradients the update then '
                'consumes directly; an optimizer that re-evaluates, or that rescales gradients, '
                'breaks the ordering contract of spec Sect. 8.1. Allowed: %s.'
                % (type(opt).__name__,
                   ', '.join(c.__name__ for c in self._TRAJ_ALLOWED_OPTIMIZERS)))
        return opt

    def _wrap_optimizer_step(self, opt):
        if getattr(opt, '_traj_wrapped', False):
            return opt
        self._assert_optimizer_supported(opt)
        hook = self.traj_penalty
        real_step = opt.step
        outer = self

        def step_with_penalty(closure=None, **step_kwargs):
            if closure is None:
                # No closure means no base backward in this call, so there is nothing to
                # accumulate ON TOP OF and the ordering contract does not apply. Refused rather
                # than silently skipped: a caller in that shape is not running this method.
                raise RuntimeError(
                    'the trajectory penalty is attached but the optimizer was stepped without a '
                    'closure. The penalty gradient must be added after the base backward and '
                    'before the update, and there is no base backward to follow here.')

            def wrapped(*c_args, **c_kwargs):
                loss = closure(*c_args, **c_kwargs)
                backward = (c_args[0] if c_args else c_kwargs.get('backward', True))
                if backward:
                    hook(outer)
                    outer._traj_hook_calls += 1
                return loss

            outer._traj_steps += 1
            return real_step(wrapped, **step_kwargs)

        opt.step = step_with_penalty
        opt._traj_wrapped = True
        opt._traj_real_step = real_step
        return opt

    @staticmethod
    def _unwrap_optimizer_step(opt):
        if opt is not None and getattr(opt, '_traj_wrapped', False):
            opt.step = opt._traj_real_step
            del opt._traj_wrapped, opt._traj_real_step

    def fit(self, *args, **kwargs):
        if self.traj_penalty is None:
            return super().fit(*args, **kwargs)

        # UNSUPPORTED MODES FAIL HERE, LOUDLY. The specification's warning is that a mode which
        # merely LOOKS supported is worse than one that is refused: the run completes and its
        # objective is not the declared one.
        opt = getattr(self, '_optimizer', None)
        if opt is None:
            raise RuntimeError(
                'the trajectory penalty is attached but no optimizer exists yet. Call '
                'init_model() before fit(): the parent would otherwise construct the optimizer '
                'itself midway through fit, and the geometry was built against a specific '
                'parameter set (the converted encoder), so an optimizer created later may not '
                'own the same parameters.')
        self._assert_optimizer_supported(opt)
        if kwargs.get('concurrent_val'):
            raise RuntimeError(
                'concurrent_val is REFUSED with the trajectory penalty: the penalty is evaluated '
                'on a fixed window set through the same model, and a concurrent validation '
                'thread mutating or moving that model mid-accumulation has not been analysed.')
        if getattr(self, 'scaler', None) is not None:
            raise RuntimeError(
                'automatic mixed precision is REFUSED with the trajectory penalty: the base '
                'gradients are scaled and the penalty gradient added here would not be, so the '
                'two would enter the update at different scales. Start from the verified '
                'non-AMP path (spec Sect. 8.1).')

        self._traj_wrapping = True
        self._traj_hook_calls = 0
        self._traj_steps = 0
        calls0 = self.traj_penalty.calls
        self._wrap_optimizer_step(opt)
        try:
            return super().fit(*args, **kwargs)
        finally:
            self._traj_wrapping = False
            cur = getattr(self, '_optimizer', None)
            # RECONCILIATION, and it has to be done THIS way round. Counting hook calls against
            # `_traj_steps` cannot detect a lost wrapper, because `_traj_steps` is incremented
            # inside the wrapper: if the wrapper goes, BOTH counters stay at zero and the
            # mismatch test passes vacuously. That was the first version of this guard and the
            # regression test in `test_fit_override.py` caught it being useless.
            #
            # What actually detects it is the state of the optimizer the fit ENDED with: if it
            # is not the wrapped one, the property failed to re-wrap a replacement and some
            # portion of the run trained without the penalty it reports.
            lost = cur is not None and not getattr(cur, '_traj_wrapped', False)
            self._unwrap_optimizer_step(cur)
            done = self.traj_penalty.calls - calls0
            if lost:
                raise RuntimeError(
                    'the optimizer this fit ended with was NOT the wrapped one, so the '
                    'trajectory penalty stopped being applied at some point during the run '
                    '(%d applications recorded). The parent replaces `self.optimizer` on a '
                    'resumed system (deepSI _refresh_optimizer), and the property that re-wraps '
                    'it did not fire. The run is INVALID: part of it optimised a different '
                    'objective than the one it reports.' % done)
            if self._traj_steps and done != self._traj_steps:
                raise RuntimeError(
                    'the trajectory penalty fired %d time(s) across %d optimizer step(s). They '
                    'must match; a shortfall means some steps were taken without it.'
                    % (done, self._traj_steps))
            if self._traj_steps == 0:
                print('[traj-orth] WARNING: fit() completed without a single optimizer step, so '
                      'the penalty was never applied.')