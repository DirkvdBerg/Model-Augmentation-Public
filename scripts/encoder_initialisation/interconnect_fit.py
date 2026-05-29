import argparse
import os
import sys
import time
import warnings

from deepSI import load_system_data
from deepSI.fit_systems.encoders import default_encoder_net
import numpy as np
import torch
from scipy.io import loadmat

from model_augmentation.fit_systems.blocks import (
    Linear_Output_Block,
    Linear_State_Block,
    Parameterized_MSD_State_Block,
    Static_ANN_Block,
    Nonlinear_MSD_State_Block,
)
from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.fit_systems.pre_encoder import (
    SS_pre_encoder,
    System_data_with_x,
    linear_encoder_init,
    System_data_norm_with_x,
)
from model_augmentation.utils.torch_nets import zero_init_resnet, linear_encoder_net
from model_augmentation.utils.utils import (
    expansion_matrix,
    normalize_linear_ss_matrices,
    selection_matrix,
)

from jax_sysid.models import StaticModel
import jax
import flax.linen as nn


# disable some of the warnings from "incorrect" use of functions and modules
warnings.filterwarnings("ignore")

## ------------- Parse input arguments -----------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--encoder_type", nargs="?", default="linear_map", type=str, help="Encoder type"
)
parser.add_argument("--lr", nargs="?", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--epochs", nargs="?", type=int, default=5, help="Epoch count")
parser.add_argument(
    "--snr", nargs="?", type=int, default=20, help="Signa-to-noise ratio"
)
parser.add_argument(
    "--verbose", nargs="?", type=bool, default=True, help="Print code output"
)

args, unknown = parser.parse_known_args()
# args = parser.parse_args()

# changes the output window for print commands such that it does not print in the terminal
if not args.verbose:
    save_stdout = sys.stdout
    sys.stdout = open("trash", "w")

## ------------- Determine root directory -----------------
root_dir = os.getcwd()
expected_folder = "Model Augmentation"
while True:
    if os.path.basename(root_dir) == expected_folder:
        break  # found it!
    parent = os.path.dirname(root_dir)
    if parent == root_dir:
        raise FileNotFoundError(
            f"Folder '{expected_folder}' not found above {os.getcwd()}"
        )
    root_dir = parent

## ------------- Hyper params -----------------
# system parameters
system_dof = 2  # 2 or 3 (number of dof in system)
flag_linear_msd = False  # True or False (use linear mass spring damper system)
snr = args.snr
# TODO: automatically create snr dataset if it does not exist yet

# FP (baseline) model parameters
FP_dof = 2  # 2 or 3 (number of dof in FP model)
FP_linear = True  # True or False (use linear FP model)
FP_type = "ideal"  # "ideal" or "approximate"

# encoder intialisation method
encoder_initialisation_type = args.encoder_type  # "pre_trained", "linear_map" or "none"
# TODO: add "nonlinear_map"
# TODO: add "noisy_map"

# model augmentation parameters
state_aug_type = "parallel"  # "parallel", "baseline"

# training parameters
nb_na_multiply_with_nx = 4
nf = 100
epochs = args.epochs
batch_size = 3000
# utility parameters
save_flag = True  # True or False (dont save model if False, e.g. for debugging)
wait_minutes = (
    0  # minutes to wait before starting the training (to not annoy colleagues)
)

optimizer_kwargs = {"lr": args.lr, "eps": 1e-8}

## ------------- Load data -----------------
# These should depend on the data generated in msd_ndof_data_generation.py, but other values are not supported yet
ny = 1
nu = 1

system_descriptors = ["msd", "{0}dof".format(system_dof)]
system_descriptors.append("linear" if flag_linear_msd else "nonlinear")
system_descriptors.append("multisine")
system_folder_name = "_".join(system_descriptors)

data_file_path = os.path.join(
    root_dir,
    "data",
    "encoder_initialisation",
    system_folder_name,
    "SNR_levels",
    "SNR{0}".format(snr),
)
train_data = load_system_data(os.path.join(data_file_path, "train.npz"))
val_data = load_system_data(os.path.join(data_file_path, "val.npz"))

# ------------ Load baseline simulation data -----------------
baseline_model_descriptors = ["FP", "{0}dof".format(FP_dof)]
baseline_model_descriptors.append("linear" if FP_linear else "nonlinear")
baseline_model_descriptors.append("ideal" if FP_type == "ideal" else "approximate")
baseline_model_folder_name = os.path.join(
    root_dir,
    "data",
    "encoder_initialisation",
    system_folder_name,
    "baseline_simulations",
    "_".join(baseline_model_descriptors),
)

baseline_x_train = np.load(
    os.path.join(
        baseline_model_folder_name,
        "x_train.npy",
    )
)
baseline_x_val = np.load(
    os.path.join(
        baseline_model_folder_name,
        "x_val.npy",
    )
)

train_data = System_data_with_x(x=baseline_x_train, u=train_data.u, y=train_data.y)
val_data = System_data_with_x(x=baseline_x_val, u=val_data.u, y=val_data.y)

## ------------- Load FP model -----------------
nx_FP = 2 * FP_dof  # number of states in FP model (2dof: 4; 3dof: 6)
FP_state_indices = np.arange(
    0, nx_FP
)  # state indices for the FP model (2dof: [0,1,2,3]; 3dof: [0,1,2,3,4,5])


# these matrices are required for both the linear FP model as well as the linear_map initialisation
if FP_type == "ideal":
    data_file_path = os.path.join(
        root_dir, "data", "mass_spring_damper", "msd_{0}dof_Ts_01.mat".format(FP_dof)
    )
elif FP_type == "approximate":
    data_file_path = os.path.join(
        root_dir,
        "data",
        "mass_spring_damper",
        "msd_{0}dof_non_ideal.mat".format(FP_dof),
    )
else:
    raise ValueError("FP_type must be either 'ideal' or 'approximate'")

mat_contents = loadmat(data_file_path, squeeze_me=False)
A_bla = mat_contents["Ad"]
B_bla = mat_contents["Bd"]
C_bla = mat_contents["Cd"]
D_bla = mat_contents["Dd"]
A_bar_bla, B_bar_bla, C_bar_bla, D_bar_bla = normalize_linear_ss_matrices(
    A_bla,
    B_bla,
    C_bla,
    D_bla,
    train_data,  # type: ignore
    state_ix=FP_state_indices,
)

if FP_linear:
    if FP_type == "ideal":
        physical_state_model_block = Linear_State_Block(A=A_bar_bla, B=B_bar_bla)
        physical_output_model_block = Linear_Output_Block(C=C_bar_bla, D=D_bar_bla)
    elif FP_type == "approximate":
        physical_state_model_block = Parameterized_MSD_State_Block(
            nz=nx_FP, nw=nx_FP, FP_type=FP_type
        )
        physical_output_model_block = Linear_Output_Block(C=C_bar_bla, D=D_bar_bla)
    else:
        raise ValueError("FP_type must be either 'ideal' or 'approximate'")
else:
    # TODO: pass physical paremeters into nonlinear msd block
    std_x = np.std(baseline_x_train, axis=0)  # type: ignore
    physical_state_model_block = Nonlinear_MSD_State_Block(
        nz=nx_FP + nu,
        nw=nx_FP,
        Ts=0.1,
        std_x=std_x,
        std_u=float(np.std(train_data.u)),  # type: ignore
    )  # type: ignore
    physical_output_model_block = Linear_Output_Block(
        C = C_bar_bla,
        # C=np.array([[0, 0, 1, 0]]),  # type: ignore
        D=np.array([[0]]),  # type: ignore
    )  # type: ignore

## ------------- Define augmentations structure -----------------
# determine number of model states
nx_model = nx_aug_model = nx_FP
state_aug_state_indices = np.arange(
    0, nx_aug_model
)  # state indices for the state augmentation component model (2dof: [0,1,2,3]; 3dof: [0,1,2,3,4,5])
nx_aug_model_out = nx_aug_model
state_aug_state_indices_output = state_aug_state_indices

# define interconnection and baseline model blocks
interconnect = Interconnect(nx_model, nu, ny, debugging=False)
interconnect.add_block(physical_state_model_block)
interconnect.add_block(physical_output_model_block)

# ----- (dynamic) parallel -------
if state_aug_type == "parallel":  # works for both static and dynamic augmentation
    ANN_state_block = Static_ANN_Block(
        nz=nx_aug_model + nu,
        nw=nx_aug_model_out,
        n_nodes_per_layer=8,
        net=zero_init_resnet,
        activation=torch.nn.Tanh,
    )
    interconnect.add_block(ANN_state_block)
    interconnect.connect_signals(
        "x",
        ANN_state_block,
        "concat",
        selection_matrix(state_aug_state_indices, nx_model),
    )
    interconnect.connect_block_signals(ANN_state_block, ["u"], [])
    interconnect.connect_signals(
        ANN_state_block,
        "xp",
        "additive",
        expansion_matrix(state_aug_state_indices_output, nx_model),
    )

    interconnect.connect_signals(
        "x",
        physical_state_model_block,
        "concat",
        selection_matrix(FP_state_indices, nx_model),
    )
    interconnect.connect_signals(
        physical_state_model_block,
        "xp",
        "additive",
        expansion_matrix(FP_state_indices, nx_model),
    )

    interconnect.connect_signals(
        "x",
        physical_output_model_block,
        "concat",
        selection_matrix(FP_state_indices, nx_model),
    )

# ----- No augmentation -------
elif state_aug_type == "baseline":
    interconnect.connect_signals(
        "x",
        physical_state_model_block,
        "concat",
        selection_matrix(FP_state_indices, nx_model),
    )
    interconnect.connect_signals(
        physical_state_model_block,
        "xp",
        "additive",
        expansion_matrix(FP_state_indices, nx_model),
    )

    interconnect.connect_signals(
        "x",
        physical_output_model_block,
        "concat",
        selection_matrix(FP_state_indices, nx_model),
    )
else:
    raise ValueError("type_aug must be either 'linear_map', 'parallel' or 'none'")

# ----- input -------
# ----- No input augmentation -------
interconnect.connect_block_signals(physical_state_model_block, ["u"], [])

# ----- output -------
# ----- No output augmentation -------
interconnect.connect_block_signals(physical_output_model_block, ["u"], ["y"])

## ------------- Wait Time To Not Annoy Colleagues -----------------
for t in range(wait_minutes):
    time.sleep(60)
    print(f"Time passed: {t + 1} minutes")

## ------------- Encoder Initialisation -----------------
na = nx_model * nb_na_multiply_with_nx + 1
nb = nx_model * nb_na_multiply_with_nx + 1
n_neurons_encoder = 16
if encoder_initialisation_type == "none":
    pass
elif encoder_initialisation_type == "pre_trained":
    ## Normalise baseline states for pre trained encoder
    std_x = baseline_x_train.std(axis=0)  # type: ignore
    train_data = System_data_with_x(
        x=baseline_x_train / std_x, u=train_data.u, y=train_data.y
    )
    val_data = System_data_with_x(x=baseline_x_val / std_x, u=val_data.u, y=val_data.y)

    ## Pre-train encoder
    if FP_linear and flag_linear_msd:
        # this initialises using the linear reconstruction map
        encoder_sys = SS_pre_encoder(
            nx=nx_model,
            na=na,
            nb=nb,
            e_net_kwargs={"n_nodes_per_layer": n_neurons_encoder},
            na_right=1,
            nb_right=1,
            e_net=linear_encoder_init # type: ignore
        )
        
        temp_net = linear_encoder_init(
            A=A_bar_bla,
            B=B_bar_bla,
            C=C_bar_bla,
            D=D_bar_bla,
            nx=nx_model,
            nu=nu,
            ny=ny,
            na=na,
            nb=nb,
            n_nodes_per_layer=n_neurons_encoder,
            flag_linear_only = True

        )

        # create normed system data for encoder fitting
        norm = System_data_norm_with_x()
        norm.fit(train_data)
        normed_data = norm.transform(train_data)  # type: ignore
        up, yp, uf, yf, xf = encoder_sys.make_training_data(normed_data, **{"nf": 1})
        xf = xf[:,0,:]
        upt = torch.tensor(up, dtype=torch.float32)
        ypt = torch.tensor(yp, dtype=torch.float32)
        xft = torch.tensor(xf, dtype=torch.float32)

        U = np.hstack((up, yp))
        Y = xf

        ny_jax = Y.shape[1]
        N, nu_jax = U.shape 

        # input convex function model [Amos, Xu, Kolter, 2017]
        @jax.jit
        def output_fcn(x,params):
            W=params[0]
            y = W @ x.T
            return y.T
        model = StaticModel(ny_jax, nu_jax, output_fcn)
        
        params=[np.random.rand(ny_jax, nu_jax)]
        # params = [np.hstack((encoder.Wb_psi_u.detach().numpy(), encoder.Wb_psi_y.detach().numpy()))]
        
        model.init(params=params)
        model.loss(rho_th=0)

        model.optimization(adam_epochs=200, lbfgs_epochs=200)
        model.fit(Y,U)
        t0 = model.t_solve
        
        encoder = linear_encoder_net(nb=nb + 1,nu=nu,na=na + 1,ny=ny,nx=4,n_nodes_per_layer=2)
        with torch.no_grad():
            encoder.net.weight.copy_(torch.tensor(model.params[0], dtype=torch.float32))

        initialised_encoder = encoder
    
    else:
        # this initialises using the linear reconstruction map
        encoder_sys = SS_pre_encoder(
            nx=nx_model,
            na=na,
            nb=nb,
            e_net_kwargs={"n_nodes_per_layer": n_neurons_encoder},
            na_right=1,
            nb_right=1,
            e_net=default_encoder_net
            # e_net=linear_encoder_net,  # type: ignore
            # e_net=linear_encoder_init # type: ignore
        )
        
        temp_net = linear_encoder_init(
            A=A_bar_bla,
            B=B_bar_bla,
            C=C_bar_bla,
            D=D_bar_bla,
            nx=nx_model,
            nu=nu,
            ny=ny,
            na=na,
            nb=nb,
            n_nodes_per_layer=n_neurons_encoder,
        )

        # create normed system data for encoder fitting
        norm = System_data_norm_with_x()
        norm.fit(train_data)
        normed_data = norm.transform(train_data)  # type: ignore
        up, yp, uf, yf, xf = encoder_sys.make_training_data(normed_data, **{"nf": 1})
        xf = xf[:,0,:]
        upt = torch.tensor(up, dtype=torch.float32)
        ypt = torch.tensor(yp, dtype=torch.float32)
        xft = torch.tensor(xf, dtype=torch.float32)

        U = np.hstack((up, yp))
        Y = xf

        ny_jax = Y.shape[1]
        N, nu_jax = U.shape 

        # input convex function model [Amos, Xu, Kolter, 2017]
        act = nn.elu # activation function, must be convex and non decreasing on the domain of interest
        @jax.jit
        def output_fcn(x,params):
            W0y,W1z,W2z,W2y,b0,b1,b2=params
            z1 = act(W0y @ x.T + b0)
            z2 = act(W1z @ z1 + b1)
            y = W2z @ z2 + W2y @ x.T + b2
            return y.T

        model = StaticModel(ny_jax, nu_jax, output_fcn)
        n1,n2 = n_neurons_encoder,n_neurons_encoder  # number of neurons
        params=[np.random.rand(n1, nu_jax), #W0y
                np.random.rand(n2, n1), #W1z
                np.zeros((ny_jax, n2)), #W2z
                # np.random.randn(ny_jax, nu_jax), #W2y
                np.hstack((temp_net.Wb_psi_u.detach().numpy(), temp_net.Wb_psi_y.detach().numpy())),
                np.random.randn(n1, 1), #b0
                np.random.randn(n2, 1), #b1
                np.zeros((ny_jax, 1))] #b2

        model.init(params=params)
        model.loss(rho_th=0)

        model.optimization(adam_epochs=50, lbfgs_epochs=200, memory=50)
        model.fit(Y,U)
        t0 = model.t_solve
        
        encoder = default_encoder_net(nb=nb + 1,nu=nu,na=na + 1,ny=ny,nx=4,n_nodes_per_layer=n_neurons_encoder)
        with torch.no_grad():
            encoder.net.net_lin.weight.copy_(torch.tensor(model.params[3], dtype=torch.float32))
            encoder.net.net_lin.bias.zero_()
            
            encoder.net.net_non_lin.net[0].weight.copy_(torch.tensor(model.params[0], dtype=torch.float32)) # type: ignore
            encoder.net.net_non_lin.net[0].bias.copy_(torch.tensor(model.params[4][:,0], dtype=torch.float32)) # type: ignore
            encoder.net.net_non_lin.net[2].weight.copy_(torch.tensor(model.params[1], dtype=torch.float32)) # type: ignore
            encoder.net.net_non_lin.net[2].bias.copy_(torch.tensor(model.params[5][:,0], dtype=torch.float32)) # type: ignore
            encoder.net.net_non_lin.net[4].weight.copy_(torch.tensor(model.params[2], dtype=torch.float32)) # type: ignore
            encoder.net.net_non_lin.net[4].bias.copy_(torch.tensor(model.params[6][:,0], dtype=torch.float32)) # type: ignore

        initialised_encoder = encoder
elif encoder_initialisation_type == "linear_map":
    if FP_linear and flag_linear_msd:
        initialised_encoder = linear_encoder_init(
            A=A_bar_bla,
            B=B_bar_bla,
            C=C_bar_bla,
            D=D_bar_bla,
            nx=nx_model,
            nu=nu,
            ny=ny,
            na=na,
            nb=nb,
            n_nodes_per_layer=n_neurons_encoder,
            flag_linear_only = True
        )
    else:
        initialised_encoder = linear_encoder_init(
            A=A_bar_bla,
            B=B_bar_bla,
            C=C_bar_bla,
            D=D_bar_bla,
            nx=nx_model,
            nu=nu,
            ny=ny,
            na=na,
            nb=nb,
            n_nodes_per_layer=n_neurons_encoder,
            flag_linear_only = False
        )
else:
    raise ValueError(
        "encoder_initialisation_type must be either 'pre_trained', 'linear_map' or 'none'"
    )

## ------------- Train fit system -----------------
fit_sys = SSE_Interconnect(
    interconnect=interconnect,  # type: ignore
    na=na,
    nb=nb,
    e_net_kwargs={"n_nodes_per_layer": 16},
    na_right=1,
    nb_right=1,
)

if encoder_initialisation_type != "none":
    fit_sys.encoder = initialised_encoder  # type: ignore

list_val_measures = []
for i in [1, 5, 20, 50, 200]:
    list_val_measures.append("{0}-step-RMS".format(i))

fit_sys.fit(
    train_sys_data=train_data,
    val_sys_data=val_data,
    batch_size=batch_size,
    epochs=epochs,
    auto_fit_norm=True,
    loss_kwargs={"nf": nf},
    # validation_measure="sim-RMS",
    validation_measure="{0}-step-RMS".format(nf),
    optimizer_kwargs=optimizer_kwargs,
    list_val_measures = list_val_measures
)

## ------------- Save fit system -----------------
if save_flag:
    baseline_model_descriptors = ["FP", "{0}dof".format(FP_dof)]
    baseline_model_descriptors.append("linear" if FP_linear else "nonlinear")
    baseline_model_descriptors.append("ideal" if FP_type == "ideal" else "approximate")
    baseline_model_folder_name = "_".join(baseline_model_descriptors)

    encoder_initialiation_descriptors = [encoder_initialisation_type]
    encoder_initialisation_folder_name = "_".join(encoder_initialiation_descriptors)

    model_augmentation_descriptors = []
    if state_aug_type != "baseline":
        model_augmentation_descriptors.append("static")
        model_augmentation_descriptors.append(state_aug_type)
    elif state_aug_type == "baseline":
        model_augmentation_descriptors.append("baseline")

    model_augmentation_descriptors.append("e{0}".format(epochs))
    model_augmentation_file_name = "_".join(model_augmentation_descriptors)

    model_folder_path = os.path.join(
        root_dir,
        "models",
        "encoder_initialisation",
        system_folder_name,
        baseline_model_folder_name,
        "SNR_levels",
        "SNR{0}".format(snr),
        encoder_initialisation_folder_name,
    )
    os.makedirs(model_folder_path, exist_ok=True)

    # Determine version number for model file
    model_file_path = os.path.join(model_folder_path, model_augmentation_file_name)
    if not os.path.exists(model_file_path + "_v1"):
        # File doesn't exist, use _v1
        versioned_file_name = model_augmentation_file_name + "_v1"
    else:
        # File exists, find next available version
        version = 1
        while True:
            versioned_file_name = model_augmentation_file_name + "_v" + str(version + 1)
            versioned_file_path = os.path.join(model_folder_path, versioned_file_name)
            if not os.path.exists(versioned_file_path):
                break
            version += 1

    fit_sys.save_system(os.path.join(model_folder_path, versioned_file_name))

if not args.verbose:
    sys.stdout = save_stdout
