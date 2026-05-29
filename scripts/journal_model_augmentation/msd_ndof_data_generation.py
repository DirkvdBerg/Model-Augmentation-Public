import deepSI
from deepSI.system_data import System_data

import os
import matplotlib.pylab as plt
import numpy as np
from scipy.io import loadmat, savemat

from model_augmentation.systems.mass_spring_damper import (
    Msd_ndof,
    Msd_ndof_lpf_output,
    Msd_ndof_input_saturation,
)
from model_augmentation.utils.utils import (
    plot_fft_freq,
    plot_sys_data,
    determine_yes_no_query_binary_output,
)

## ------------- Flags -----------------
System_dof = 2  # 2 or 3 (number of dof in system)
FP_dof = 2  # 2 or 3 (number of dof in FP model)
flag_linear_msd = False  # True or False (use linear mass spring damper system)
flag_low_pass_filter = (
    "None"  # "Input", "Output" or "None" (use low pass filter in system)
)
flag_input_saturation = False  # True or False (use input saturation in system)
flag_save = False  # True or False (dont save data if False, e.g. for debugging)

## ------------- Define system -----------------
dt = 0.02
if flag_linear_msd:
    m = [0.5, 0.4, 0.1]
    k = [100, 100, 100]
    c = [0.5, 0.5, 0.5]
    a = [0, 0, 0]
else:
    m = [0.5, 0.4, 0.1]
    k = [100, 100, 100]
    c = [0.5, 0.5, 0.5]
    a = [0, 50, 0]

if flag_low_pass_filter == "Output":
    system = Msd_ndof_lpf_output(
        n=System_dof, m=m, k=k, c=c, a=a, dt=dt, input_ix=[0], output_ix=[1]
    )
elif flag_low_pass_filter == "Input":
    raise NotImplementedError("The LPF input MSD system has a bug in it.")
    # system = Msd_ndof_lpf_input(n=System_dof, m=m, k=k, c=c, a=a, dt=dt, input_ix=[0], output_ix=[1])
elif flag_low_pass_filter == "None":
    system = Msd_ndof(
        n=System_dof, m=m, k=k, c=c, a=a, dt=dt, input_ix=[0], output_ix=[1]
    )
else:
    raise ValueError("flag_low_pass_filter must be either 'Input', 'Output' or 'None'")

if flag_input_saturation:
    system = Msd_ndof_input_saturation(
        n=System_dof, m=m, k=k, c=c, a=a, dt=dt, input_ix=[0], output_ix=[1]
    )

## ------------- Define baseline model -----------------
data_file_path = os.path.join(
    os.getcwd(), "data\\mass_spring_damper\\msd_{0}dof.mat".format(FP_dof)
)
mat_contents = loadmat(data_file_path, squeeze_me=False)

nx = mat_contents["nx"][0, 0]
ny = mat_contents["ny"][0, 0]
nu = mat_contents["nu"][0, 0]
Ts = mat_contents["Ts"][0, 0]

A_bla = mat_contents["Ad"]
B_bla = mat_contents["Bd"]
C_bla = mat_contents["Cd"]
D_bla = mat_contents["Dd"]

## ------------- Generate input -----------------
pmax = 4999
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

utrain = (
    deepSI.deepSI.exp_design.multisine(
        Ntrain, (periods_train + 1), pmax=pmax, n_crest_factor_optim=1
    )
    * amp_scale
)
uval = (
    deepSI.deepSI.exp_design.multisine(
        Nval, (periods_val + 1), pmax=pmax, n_crest_factor_optim=1
    )
    * amp_scale
)
utest = (
    deepSI.deepSI.exp_design.multisine(
        Ntest, (periods_test + 1), pmax=pmax, n_crest_factor_optim=1
    )
    * amp_scale
)

Ttrain = np.arange(Ntrain_total) * dt
Tval = np.arange(Nval_total) * dt
Ttest = np.arange(Ntest_total) * dt

## ------------- Simulate system -----------------
train = system.apply_experiment(sys_data=deepSI.System_data(u=utrain), save_state=True)
train = train[Ntrain:]
train.y = train.y[:, 0]  # type: ignore
val = system.apply_experiment(sys_data=deepSI.System_data(u=uval), save_state=True)
val = val[Nval:]
val.y = val.y[:, 0]  # type: ignore
test = system.apply_experiment(sys_data=deepSI.System_data(u=utest), save_state=True)
test = test[Ntest:]
test.y = test.y[:, 0]  # type: ignore

## ------------- Add noise -----------------
# sigma_n = 52e-4
# noise =  np.random.normal(0, sigma_n, (Ntrain_total,))
# Py = np.sum(np.square(train.y + noise))/Ntrain_total
# Pn = np.sum(np.square(noise))/Ntrain_total
# SNR10 = 10*np.log10(Py/Pn)
# print("Py: {0}".format(Py))
# print("Pn: {0}".format(Pn))
# print("SNR: {0}".format(SNR10))

## ------------- RMS baseline model -----------------
ylog = np.zeros(Ntrain_total)
ulog = np.zeros(Ntrain_total)
x0 = np.zeros((nx, 1))
for i in range(Ntrain_total):
    u0 = train.u[i]  # type: ignore
    ulog[i] = u0
    x1 = A_bla @ x0 + B_bla * u0
    ylog[i] = (C_bla @ x0)[0, 0]
    x0 = x1.copy()

transient = 1000
test_baseline = deepSI.System_data(u=ulog[transient:], y=ylog[transient:])
print(f"RMS simulation baseline  {test_baseline.RMS(train[transient:])}")

u: np.ndarray = 30*np.tanh(train.u/30)
print(np.mean((u)**2,axis=0)**0.5)

Tlog = np.arange(Ntrain_total) * dt
plt_len = 10000
plt.plot(
    Tlog[transient : plt_len + transient], train.u[:][transient : plt_len + transient]
)
plt.plot(
    Tlog[transient : plt_len + transient], 30*np.tanh(train.u[:][transient : plt_len + transient]/30)
)

# plt.plot(
#     Tlog[transient : plt_len + transient], train.y[:][transient : plt_len + transient]
# )
# plt.plot(
#     Tlog[transient : plt_len + transient],
#     train.y[:][transient : plt_len + transient] - ylog[transient : plt_len + transient],
# )
plt.show()

# for i in range(0,6):
#     plt.subplot(6,1,i+1)
#     plt.plot(train.x[:,i])
# plt.plot(train.u)
# plt.show()

# plot_fft_freq(ylog, dt)
# plot_fft_freq(train.y, dt)
# plt.show()

## ------------- Save data -----------------
if flag_save and determine_yes_no_query_binary_output(
    "Should this data be saved? [y/n] "
):
    system_descriptors = ["new_msd", "{0}dof".format(System_dof)]
    system_descriptors.append("linear" if flag_linear_msd else "nonlinear")

    if flag_low_pass_filter == "Input":
        system_descriptors.append("lpf_input")
    elif flag_low_pass_filter == "Output":
        system_descriptors.append("lpf_output")

    if flag_input_saturation:
        system_descriptors.append("input_saturation")

    system_descriptors.append("multisine")

    folder_name = "_".join(system_descriptors)
    data_file_path = os.path.join(
        os.getcwd(), "data", "journal_model_augmentation", folder_name
    )
    os.makedirs(data_file_path, exist_ok=True)

    train.save(os.path.join(os.getcwd(), data_file_path, "train.npz"))
    val.save(os.path.join(os.getcwd(), data_file_path, "val.npz"))
    test.save(os.path.join(os.getcwd(), data_file_path, "test.npz"))

    # Save train and val as .mat files
    # savemat(os.path.join(data_file_path, "train.mat"), {
    #     'u': np.asarray(train.u),
    #     'y': np.asarray(train.y),
    #     'x': np.asarray(train.x)
    # })
    # savemat(os.path.join(data_file_path, "val.mat"), {
    #     'u': np.asarray(val.u),
    #     'y': np.asarray(val.y),
    #     'x': np.asarray(val.x)
    # })
else:
    print("Data was not saved.")
