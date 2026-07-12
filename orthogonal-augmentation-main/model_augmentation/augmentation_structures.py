import torch
from torch import nn
from model_augmentation.utils import verifySystemType, verifyNetType, initialize_augmentation_net, calculate_normalization
from model_augmentation.augmentation_encoders import state_measure_encoder
from model_augmentation.fit_system import augmentation_encoder


# -------------------------------------------------------------------------------------------
# ------------------------------------- GENERIC FUNCTIONS ----------------------------------
# -------------------------------------------------------------------------------------------

def verifyAugmentationStructure(augmentation_struct, known_sys, neur_net, nx_hidden=0):
    # Verify if the augmentation structure is valid and calculate the encoder state depending on static/dynamic augmentation.
    if augmentation_struct is SSE_AdditiveAugmentation:
        static = True
        augm = 'additive'
    else:
        raise ValueError("'augmentation_structure' must be one of the types defined in 'model_augmentation.augmentationstructures'")

    initialize_augmentation_net(network=neur_net, augm_type=augm, nx=known_sys.Nx)

    if static:
        # Only learn the system states for static augmentation
        nx_encoder = known_sys.Nx
    else:
        raise NotImplementedError("Dynamic augmentation not implemented yet!")
    return nx_encoder


def get_augmented_fitsys(augmentation_structure, known_system, neur_net, aug_kwargs={}, e_net=None,
                         y_lag_encoder=None, u_lag_encoder=None, enet_kwargs={}, na_right=0, nb_right=0,
                         regLambda=0.0, orthLambda=0.0, l2_reg=0.0, norm_data=None, norm_x0_meas=False):
    """
    Function for generating model augmentation structures and getting everything ready for training.

    Args:
        augmentation_structure: from 'model_augmentation.augmentation_structures'.
            Additive, multiplicative or LFR-based structures both in static / dynamic form.
        known_system: from 'model_augmentation.system_models'.
            First-principle model, in LTI or nonlinear form.
        neur_net: torch.Model, preferably from 'model_augmentation.torch_nets', neural network for augmentation.
        aug_kwargs (dict, optional): extra arguments for the model augmentation structure.
        e_net (optional): encoder network, if None (default) no encoder is used (full-sate measurement).
        y_lag_encoder (int, optional): lag of the output for the encoder. Defaults to None, it sets automatically.
        u_lag_encoder (int, optional): lag of the input for the encoder. Defaults to None, it sets automatically.
        enet_kwargs (dict, optional): extra arguments for encoder network, e.g. n_hidden_layers, n_nodes_per_layer, etc.
        na_right (int, optional): encoder can use that many output values from the future. Defaults to 0.
            If 1, y[k] can be used for calculating x[k]. It is advised therefore to set na_right to 1.
        nb_right (int, optional): encoder can use that many input values from the future. Defaults to 0.
        regLambda (double, optional): regularization coefficient for physical parameters. Defaults to 0.
        orthLambda (double, optional): coefficient for orthogonal projection-based regularization. Defaults to 0.
        norm_data (System_data, optional): data for creating the normalization scheme for the model augmentation framework.
            Defaults to None, then normalization is not used. Usually the training data is used here.
        norm_x0_meas (bool, optional): if True, y[k] = x[k] output map assumed for the normalization scheme.
            Defaults to False.

    Returns:
        augmentation_encoder: a trainable model augmentation structure embedded into deepSI format
    """
    nx_encoder = verifyAugmentationStructure(augmentation_structure, known_system, neur_net)
    if y_lag_encoder is None: y_lag_encoder = nx_encoder + 1
    if u_lag_encoder is None: u_lag_encoder = nx_encoder + 1
    if e_net is None:
        y_lag_encoder = 1
        u_lag_encoder = 1
        na_right = 1
        e_net = state_measure_encoder
    std_x, std_y, std_u = calculate_normalization(norm_data, norm_x0_meas, known_system)
    return augmentation_encoder(nx=nx_encoder, na=y_lag_encoder, nb=u_lag_encoder, e_net=e_net,
                                e_net_kwargs=dict(std_x=std_x, std_y=std_y, std_u=std_u, **enet_kwargs),
                                augm_net=augmentation_structure, na_right=na_right, nb_right=nb_right,
                                augm_net_kwargs=dict(known_system=known_system, net=neur_net, regLambda=regLambda,
                                                     orthLambda=orthLambda, std_x=std_x, std_y=std_y, std_u=std_u,
                                                     l2_reg=l2_reg, **aug_kwargs))

# -------------------------------------------------------------------------------------------
# ---------------------------------- ADDITIVE AUGMENTATION ----------------------------------
# -------------------------------------------------------------------------------------------

class SSE_AdditiveAugmentation(nn.Module):
    """
    Simple augmentation structure implementing an additive scheme as
        x[k+1] = f(x[k], u[k]) + F(x[k], u[k]),
    where f is the first-principle mechanical model, and F is a neural network, x is the model state, u is the input.

    Arguments:
        known_system - first-principle model
        net - neural network (previously noted with F)
        regLambda - regularization coefficient for physical parameters in f
        orthLambda - orthogonalization coefficient that penalizes the cost fun. for F, which ouput is in the subspace of f
        std_x, std_u - standard deviation of the state and input (approximated based on the training set) for standardization
    """
    def __init__(self, known_system, net, std_x, std_u, regLambda=0, orthLambda=0, l2_reg=0, **kwargs):
        super(SSE_AdditiveAugmentation, self).__init__()

        # First verify if we have the correct system type and augmentation parameters
        verifySystemType(known_system)
        verifyNetType(net, 'static')

        # Save parameters
        self.sys = known_system
        self.xnet = net
        self.Nu = known_system.Nu
        self.Nx = known_system.Nx
        self.Ny = known_system.Ny

        # save normalization matrices
        self.Tx = torch.diag(torch.tensor(1 / std_x, dtype=torch.float))
        self.Tu = torch.diag(torch.tensor(1 / std_u, dtype=torch.float))
        self.Tx_inv = torch.diag(torch.tensor(std_x, dtype=torch.float))

        # check for regularization and orthogonalization coefficients (only if known_sys allows it)
        if hasattr(known_system, 'parm_corr_enab'):
            self.Pcorr_enab = known_system.parm_corr_enab
        else:
            self.Pcorr_enab = False
        if self.Pcorr_enab:
            self.regLambda = regLambda
            self.orthLambda = orthLambda

        self.l2_reg = l2_reg

    def calculate_xnet(self, x, u):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |

        x_norm = x @ self.Tx
        u_norm = u @ self.Tu

        xnet_input = torch.cat((x_norm.view(x.shape[0], -1), u_norm.view(u.shape[0], -1)), dim=1)

        return self.xnet(xnet_input) @ self.Tx_inv

    def forward(self, x, u):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |  - y  (Nd, Ny)
        x_plus = self.sys.f(x, u) + self.calculate_xnet(x, u)
        y_k = self.sys.h(x, u)
        return y_k, x_plus

    def calculate_orthogonalisation(self, x, u, U1):
        # in:                   | out:
        #  - x (Nd, Nx)         |  - cost
        #  - u (Nd, Nu)         |
        #  - U1 (Nd*Nx, Ntheta) |

        x_net = self.calculate_xnet(x, u).view(U1.shape[0], -1)
        # orthogonal_components = U1 @ U1.T @ x_net
        # cost = self.orthLambda * torch.linalg.vector_norm(orthogonal_components)**2
        orthogonal_components = U1.T @ x_net
        cost = self.orthLambda * torch.linalg.vector_norm(orthogonal_components)**2
        return cost
