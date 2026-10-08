import os

from deepSI import exp_design, System_data
import matplotlib.pylab as plt
import numpy as np

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

flag_save = True  # True or False (dont save data if False, e.g. for debugging)

## ------------- Define system -----------------
dt = 0.1
up_factor = 10
dt_rk4 = dt / up_factor
if flag_linear_msd:
    m = [0.5, 0.4, 0.1]
    k = [100, 100, 100]
    c = [0.5, 0.5, 0.5]
    a = [0, 0, 0]
    d = [0, 0, 0]

else:
    m = [0.5, 0.4, 0.1]
    k = [100, 100, 100]
    c = [0.5, 0.5, 0.5]
    a = [0, 1000, 0]
    d = [0.1, 0, 0]

system = Msd_ndof(
    n=system_dof, m=m, k=k, c=c, a=a, d=d, dt=dt_rk4, input_ix=[0], output_ix=[1]
)

## ------------- Generate input -----------------
pmax = int(4999)
amp_scale = 10

periods_train = 2
Ntrain = 10000
Ntrain_total = Ntrain * (periods_train)
periods_val = 1
Nval = 10000
Nval_total = Nval * (periods_val)
periods_test = 1
Ntest = 10000
Ntest_total = Ntest * (periods_test)

seed_value = 42 if flag_linear_msd else 82
utrain = (
    exp_design.multisine(
        Ntrain, (periods_train + 1), pmax=pmax, n_crest_factor_optim=1, seed=seed_value
    )
    * amp_scale
)
uval = (
    exp_design.multisine(
        Nval, (periods_val + 1), pmax=pmax, n_crest_factor_optim=1, seed=seed_value + 1
    )
    * amp_scale
)
utest = (
    exp_design.multisine(
        Ntest,
        (periods_test + 1),
        pmax=pmax,
        n_crest_factor_optim=1,
        seed=seed_value + 2,
    )
    * amp_scale
)

Ttrain = np.arange(Ntrain_total) * dt
Tval = np.arange(Nval_total) * dt
Ttest = np.arange(Ntest_total) * dt

## ------------- Simulate system -----------------
train_transient = system.apply_experiment(
    sys_data=System_data(u=np.repeat(utrain, up_factor)), save_state=True
)
val_transient = system.apply_experiment(
    sys_data=System_data(u=np.repeat(uval, up_factor)), save_state=True
)
test_transient = system.apply_experiment(
    sys_data=System_data(u=np.repeat(utest, up_factor)), save_state=True
)

# downsampling
train_transient = train_transient[::up_factor]
val_transient = val_transient[::up_factor]
test_transient = test_transient[::up_factor]

## ------------- Remove transients -----------------
train = train_transient[Ntrain:]
train.y = train.y[:, 0]  # type: ignore
val = val_transient[Nval:]
val.y = val.y[:, 0]  # type: ignore
test = test_transient[Ntest:]
test.y = test.y[:, 0]  # type: ignore

## ------------- Plot -----------------
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"
Tlog = np.arange(Ntrain_total) * dt
plt_len = 10000
plt.plot(Tlog[:plt_len], train.y[:][:plt_len])
plt.xlabel(r"time [s]")
plt.ylabel(r"displacement [m]")
plt.show()

## ------------- Save data -----------------
if flag_save and determine_yes_no_query_binary_output("Should this data be saved? [y/n] "):
    system_descriptors = ["msd", "{0}dof".format(system_dof)]
    system_descriptors.append("linear" if flag_linear_msd else "nonlinear")
    system_descriptors.append("multisine")

    system_folder_name = "_".join(system_descriptors)
    data_file_path = os.path.join(
        root_dir, "data", "encoder_initialisation", system_folder_name
    )
    os.makedirs(data_file_path, exist_ok=True)

    train.save(os.path.join(root_dir, data_file_path, "train.npz"))
    val.save(os.path.join(root_dir, data_file_path, "val.npz"))
    test.save(os.path.join(root_dir, data_file_path, "test.npz"))

    train_transient.save(os.path.join(root_dir, data_file_path, "train_transient.npz"))
    val_transient.save(os.path.join(root_dir, data_file_path, "val_transient.npz"))
    test_transient.save(os.path.join(root_dir, data_file_path, "test_transient.npz"))

else:
    print("Data was not saved.")
