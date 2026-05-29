import os

from deepSI import System_data, load_system_data
import matplotlib.pylab as plt
import numpy as np
from scipy.io import loadmat
from model_augmentation.systems.mass_spring_damper import Msd_ndof

from model_augmentation.utils.utils import determine_yes_no_query_binary_output

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

## ------------- Flags -----------------
system_dof = 2  # 2 or 3 (number of dof in system)
flag_linear_msd = False  # True or False (use linear mass spring damper system)

# FP (baseline) model parameters
FP_dof = 2  # 2 or 3 (number of dof in FP model)
FP_linear_options = [False]  # True or False (use linear FP model)
FP_types = ["ideal"]  # "ideal" or "approximate"

flag_save = True  # True or False (dont save data if False, e.g. for debugging)

## ------------- Load data -----------------
# These should depend on the data generated in msd_ndof_data_generation.py, but other values are not supported yet
ny = 1
nu = 1

system_descriptors = ["msd", "{0}dof".format(system_dof)]
system_descriptors.append("linear" if flag_linear_msd else "nonlinear")
system_descriptors.append("multisine")
system_folder_name = "_".join(system_descriptors)

data_file_path = os.path.join(
    root_dir, "data", "encoder_initialisation", system_folder_name
)
train = load_system_data(os.path.join(data_file_path, "train.npz"))
val = load_system_data(os.path.join(data_file_path, "val.npz"))
test = load_system_data(os.path.join(data_file_path, "test.npz"))

train_transient = load_system_data(
    os.path.join(data_file_path, "train_transient.npz")
)
val_transient = load_system_data(
    os.path.join(data_file_path, "val_transient.npz")
)
test_transient = load_system_data(
    os.path.join(data_file_path, "test_transient.npz")
)

train_transient_period = train_transient.u.shape[0] - train.u.shape[0]  # type: ignore
val_transient_period = val_transient.u.shape[0] - val.u.shape[0]  # type: ignore
test_transient_period = test_transient.u.shape[0] - test.u.shape[0]  # type: ignore


## ------------- Load and simulate baseline model -----------------
def simulate_baseline_on_dataset(data, nx, A, B, C, x0=None):
    # determine length
    u_arr = np.asarray(data.u)
    N = u_arr.shape[0]

    ylog = np.zeros(N)
    ulog = np.zeros(N)
    xlog = np.zeros((N, nx))

    # initial state
    if x0 is None:
        x0 = np.zeros((nx, 1))
    else:
        x0 = np.atleast_2d(x0).reshape((nx, 1))

    for i in range(N):
        u0 = u_arr[i]
        # make u_vec column vector (nu x 1)
        if np.isscalar(u0):
            u_vec = np.array([[u0]])
        else:
            u_vec = np.atleast_2d(u0).reshape(-1, 1)

        ulog[i] = u_arr[i]
        x1 = A @ x0 + B @ u_vec
        ylog[i] = (C @ x0).ravel()[0]
        xlog[i, :] = x0.ravel()
        x0 = x1.copy()

    return ylog, ulog, xlog


# Build baseline model folder names for all combinations of FP_linear options and FP_types.
baseline_model_folder_names = []
# figure init
nx = 2 * FP_dof
plt_len = 2000
fig = plt.figure()
for i in range(nx):
    plt.subplot(nx, 1, i + 1)
    plt.plot(test.x[:plt_len, i], label="system")  # type: ignore
plt.legend

for fp_lin in FP_linear_options:
    for fp_type in FP_types:
        baseline_model_descriptors = ["FP", f"{FP_dof}dof"]
        baseline_model_descriptors.append("linear" if fp_lin else "nonlinear")
        baseline_model_descriptors.append(
            "ideal" if fp_type == "ideal" else "approximate"
        )
        baseline_model_folder_name = "_".join(baseline_model_descriptors)

        if fp_lin:
            if fp_type == "ideal":
                data_file_path = os.path.join(
                    root_dir,
                    "data",
                    "mass_spring_damper",
                    "msd_{0}dof_Ts_01.mat".format(FP_dof),
                )
            elif fp_type == "approximate":
                data_file_path = os.path.join(
                    root_dir,
                    "data",
                    "mass_spring_damper",
                    "msd_{0}dof_non_ideal.mat".format(FP_dof),
                )
            else:
                raise ValueError("FP_type must be either 'ideal' or 'approximate'")

            mat_contents = loadmat(data_file_path, squeeze_me=False)

            nx = mat_contents["nx"][0, 0]
            ny = mat_contents["ny"][0, 0]
            nu = mat_contents["nu"][0, 0]
            Ts = mat_contents["Ts"][0, 0]

            A_bla = mat_contents["Ad"]
            B_bla = mat_contents["Bd"]
            C_bla = mat_contents["Cd"]
            D_bla = mat_contents["Dd"]

            ytrain, _, xtrain = simulate_baseline_on_dataset(
                data=train_transient, nx=nx, A=A_bla, B=B_bla, C=C_bla
            )
            yval, _, xval = simulate_baseline_on_dataset(
                data=val_transient, nx=nx, A=A_bla, B=B_bla, C=C_bla
            )
            ytest, _, xtest = simulate_baseline_on_dataset(
                data=test_transient, nx=nx, A=A_bla, B=B_bla, C=C_bla
            )

            xtrain = xtrain[train_transient_period:]
            xval = xval[val_transient_period:]
            xtest = xtest[test_transient_period:]
            
            # for plotting
            ytest = ytest[test_transient_period:]
            test_baseline = System_data(y=ytest, u=test.u)

        else:
            m = [0.5, 0.4, 0.1]
            k = [100, 100, 100]
            c = [0.5, 0.5, 0.5]
            a = [0, 1000, 0]
            d = [0.05, 0, 0]

            system = Msd_ndof(n=FP_dof, m=m, k=k, c=c, a=a, d=d, dt=0.01, input_ix=[0], output_ix=[1])
            up_factor = 10
            train_baseline = system.apply_experiment(
                sys_data=System_data(u=np.repeat(train_transient.u, up_factor)), save_state=True # type: ignore
            )
            val_baseline = system.apply_experiment(
                sys_data=System_data(u=np.repeat(val_transient.u, up_factor)), save_state=True # type: ignore
            )
            test_baseline = system.apply_experiment(
                sys_data=System_data(u=np.repeat(test_transient.u, up_factor)), save_state=True # type: ignore
            )

            # downsampling
            train_baseline = train_baseline[::up_factor]
            val_baseline = val_baseline[::up_factor]
            test_baseline = test_baseline[::up_factor]

            xtrain = train_baseline.x[train_transient_period:] # type: ignore
            xval = val_baseline.x[train_transient_period:] # type: ignore
            xtest = test_baseline.x[train_transient_period:] # type: ignore
            
            # for plotting
            test_baseline = test_baseline[test_transient_period:] # type: ignore
            test_baseline.y = test_baseline.y[:,0] # type: ignore

        ## ------------- Plots and metrics -----------------
        print("RMS " + baseline_model_folder_name + f" {test_baseline.RMS(test)}")

        for i in range(nx):
            plt.subplot(nx, 1, i + 1)
            plt.plot(
                test.x[:plt_len, i] - xtest[:plt_len, i],  # type: ignore
                label=baseline_model_folder_name,
            )
        plt.show()

        ## ------------- Save data -----------------
        if flag_save and determine_yes_no_query_binary_output(
            "Should this data be saved? [y/n] "
        ):
            data_file_path = os.path.join(
                root_dir,
                "data",
                "encoder_initialisation",
                system_folder_name,
                "baseline_simulations",
                baseline_model_folder_name,
            )
            os.makedirs(data_file_path, exist_ok=True)

            np.save(os.path.join(data_file_path, "x_train"), xtrain)
            np.save(os.path.join(data_file_path, "x_val"), xval)
            np.save(os.path.join(data_file_path, "x_test"), xtest)
        else:
            print("Data was not saved.")
