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
from model_augmentation.fit_systems.blocks import Block, Parameterized_Linear_Output_Block, Parameterized_MSD_State_Block, Parameterized_Linear_State_Block
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
        self.ny = tuple()# if ny is None else ((ny,) if isinstance(ny,int) else ny) # <---------- This prevents the output target from ever being larger than 1 dimension, regardless of data size
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), \
            n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, upast, ypast):
        # ypast = ypast[:,:,0] # <---------- To be disabled after training encoder: This prevents selects a single value from the state to be the output

        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1) # type: ignore
        return self.net(net_in)

class SSE_Interconnect(SS_encoder_general):
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

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states # type: ignore
        errors = []
        for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
            yhat, x = self.hfn(x, u) # type: ignore
            errors.append(nn.functional.mse_loss(y, yhat)) #calculate error after taking n-steps
        loss_MSE = torch.mean(torch.stack(errors))
        
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
    
    def measure_act_multi(self,actions):
        actions = torch.tensor(np.array(actions), dtype=torch.float32) #(N,...)
        with torch.no_grad():
            y_predict, self.state = self.hfn(self.state, actions) # type: ignore
        return y_predict.numpy()
    
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
        def validation(train_loss=None, time_elapsed_total=None):
            self.eval(); self.cpu()
            Loss_val = self.cal_validation_error(val_sys_data, validation_measure=validation_measure)
            self.Loss_val.append(Loss_val)
            temp_loss_stack = np.array([])
            for val_measure in list_val_measures:
                loss_val = self.cal_validation_error(val_sys_data, validation_measure=val_measure)
                temp_loss_stack = np.hstack((temp_loss_stack, loss_val))
                # print(loss_val)
            print( temp_loss_stack.T.shape)
            self.multi_loss_val = np.append(self.multi_loss_val, temp_loss_stack.T)
            
            self.Loss_train.append(train_loss)
            self.time.append(time_elapsed_total)
            self.batch_id.append(self.batch_counter)
            self.epoch_id.append(self.epoch_counter)
            if self.bestfit>=Loss_val:
                self.bestfit = Loss_val
                self.checkpoint_save_system()
            if cuda: 
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
            data_train_loader = My_Simple_DataLoader(data_train, batch_size=batch_size) #is quite a bit faster for low data situations

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
                def closure(backward=True):
                    t.toc('optimizer start')
                    t.tic('loss')
                    Loss = self.loss(*train_batch, **loss_kwargs)
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
        try:
            self.checkpoint_load_system(name='_best')
        except FileNotFoundError:
            print('no best checkpoint found keeping last')
        if verbose: 
            print(f'Loaded model with best known validation {validation_measure} of {self.bestfit:6.4} which happened on epoch {best_it} (epoch_id={self.epoch_id[-1] if len(self.epoch_id)>0 else 0:.2f})')