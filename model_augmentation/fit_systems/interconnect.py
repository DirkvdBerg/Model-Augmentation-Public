from deepSI.fit_systems.encoders import SS_encoder_general, default_encoder_net

from torch import nn, Tensor
import torch
import numpy as np
import warnings

from model_augmentation.utils.utils import *
from model_augmentation.fit_systems.blocks import *

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
            input_signals.append(torch.zeros((self.nb, self.input_signal_sizes[ix], 1)))
        for ix in range(0, self.n_output_signals):
            output_signals.append(torch.zeros((self.nb, self.output_signal_sizes[ix], 1)))

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
                self.saved_input_signals = np.append(self.saved_input_signals, concat_input_signals, axis=1)
                self.saved_output_signals = np.append(self.saved_output_signals, concat_output_signals, axis=1)


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
                    self.order_output_signal_computation.append(element)
                    connection_interconnect_matrix[:,element] = 0

        if self.debugging: print("Order of computation: " + str(self.order_output_signal_computation))

        # initialize the connection matrices and determine the input signal dependencies for all output signals
        for output_signal_ix in range(self.n_output_signals):
            input_ix_dependencies = self.init_connection_matrices(output_signal_ix, output_ix_sorted_signal_connections[output_signal_ix])
            self.output_ix_sorted_input_ix_dependencies.append(input_ix_dependencies)

        if self.debugging: print("Output signal dependencies: " + str(self.output_ix_sorted_input_ix_dependencies))

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
        new_block.block_ix = self.nr_blocks + 1 # index is offset by 1 from number of blocks because of external signals
        new_block.name = name if isinstance(name, str) else "Block_" + str(self.nr_blocks+1)
        self.connected_blocks.append(new_block)

        self.input_signal_sizes.append(new_block.nw)
        self.output_signal_sizes.append(new_block.nz)

        if self.debugging: print("Added block to interconnect with name: " + new_block.name + " and signals nw=" + str(new_block.nw) + ", nz=" + str(new_block.nz))
    
    def connect_signals(self, input_signal, output_signal, connection_function_method=None, connection_matrix=torch.empty(0,0), add_to_input_signal_ix=None):     
        input_signal_ix = self.determine_signal_ix(input_signal)
        output_signal_ix = self.determine_signal_ix(output_signal)
        if add_to_input_signal_ix != None: add_to_input_signal_ix = self.determine_signal_ix(add_to_input_signal_ix)

        if not isinstance(connection_function_method, str) and output_signal_ix >= 2: connection_function_method = "concatenation" # default for internal signals is concatenation
        if not isinstance(connection_function_method, str) and output_signal_ix <= 1: connection_function_method = "additive" # default for progressed state and output is additive method
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

        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1)
        return self.net(net_in)

class SSE_Interconnect(SS_encoder_general):
    def __init__(self, na=5, nb=5, \
                 interconnect=Interconnect, e_net=modified_encoder_net,   e_net_kwargs={}, na_right=0, nb_right=0):

        super(SSE_Interconnect, self).__init__(nx=interconnect.nx, nb=nb, na=na, na_right=na_right, nb_right=nb_right)
        
        self.e_net = e_net
        self.e_net_kwargs = e_net_kwargs
        self.hfn = interconnect
        # hf_net_kwargs['feedthrough'] = feedthrough
        # self.hf_net_kwargs = hf_net_kwargs

    def init_nets(self, nu, ny): # a bit weird
        na_right = self.na_right if hasattr(self,'na_right') else 0
        nb_right = self.nb_right if hasattr(self,'nb_right') else 0
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
                # self.norm.ustd = 0.9995115994824683
                # self.norm.ystd = 2.165135419802158
                # self.norm.ystd  = np.array([2.165135419802158, 2.165135419802158])
                # self.norm.u0 = 2.8
                # self.norm.y0 = 5.582729231040664
                # self.norm.y0 = np.array([5.582729231040664, 5.582729231040664])
                
                
        self.init_nets(self.nu, self.ny)
        self.to_device(device=device)
        parameters_and_optim = [{**item,**parameters_optimizer_kwargs.get(name,{})} for name,item in self.parameters_with_names.items()]
        self.optimizer = self.init_optimizer(parameters_and_optim, **optimizer_kwargs)
        self.scheduler = self.init_scheduler(**scheduler_kwargs)
        self.bestfit = float('inf')
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        self.init_model_done = True

        self.hfn.init_model(sys_data)

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states
        errors = []
        for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
            yhat, x = self.hfn(x, u)
            errors.append(nn.functional.mse_loss(y, yhat)) #calculate error after taking n-steps
        loss_MSE = torch.mean(torch.stack(errors))
        
        has_theta_loss = False
        loss_theta = 0
        for m in self.hfn.connected_blocks:
            if isinstance(m, Parameterized_Linear_State_Block):
                loss_theta = loss_theta + m.param_loss()
                has_theta_loss = True
            elif isinstance(m, Parameterized_Linear_Output_Block):
                loss_theta = loss_theta + m.param_loss()
                has_theta_loss = True
            elif isinstance(m, Parameterized_LPV_Affine_Linear_State_Block):
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
    
    def measure_act_multi(self,actions):
        actions = torch.tensor(np.array(actions), dtype=torch.float32) #(N,...)
        with torch.no_grad():
            y_predict, self.state = self.hfn(self.state, actions)
        return y_predict.numpy()

