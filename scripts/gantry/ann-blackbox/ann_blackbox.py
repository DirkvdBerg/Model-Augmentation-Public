"""Minimal full-ANN black box on the gantry records.

Structured line-for-line on Jan's `scripts/ecc_2025/msd_ndof_deepSI_encoder.py` (37 lines).
Every line that departs from that reference carries a DEV comment naming what forced it.
The line-by-line mapping is in `CORRESPONDENCE.md` next to this file.

No baseline, no interconnect, no projection: this is the black-box arm alone.
"""
__project_origin__ = "added"

# --- Jan L1-L4: imports -----------------------------------------------------
from deepSI.fit_systems.encoders import SS_encoder_general_hf, default_state_net, default_output_net  # Jan L1, verbatim
import deepSI                                          # Jan L2
import os                                              # Jan L3
import numpy as np                                     # Jan L4
import sys, json, argparse                             # DEV: Part 2 sweeps fs from the shell and must persist metrics before fit() returns (run 74045 lost everything to a wall clock)

## ------------- Load data -----------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # DEV: so the import below resolves when launched by path from the repo root
from data import load                                  # DEV(Jan L10-L12): the .mat loader plus the Part 2 rate change, in its own file so the diagnostic shares exactly one decimation path

p = argparse.ArgumentParser()                           # DEV: Part 2 needs one arm per rate with everything else held fixed
p.add_argument('--fs', type=float, default=4000.0)      # Part 2 sweep variable, the only one
p.add_argument('--train', default='T10_aprbs_60')       # brief section 9: the APRBS record carrying most prior results
p.add_argument('--val', default='V2_aprbs_Ylow')        # DEV: matched excitation class at an unseen Y. V1_standstill has y std 3e-6 m against T10's 5.7e-2 m, six orders down, so it cannot score a model trained on T10
p.add_argument('--nf', type=int, default=400)           # brief Part 2: training horizon fixed at 400 samples across all arms
p.add_argument('--epochs', type=int, default=500)
p.add_argument('--batch-size', type=int, default=256)   # DEV(Jan L32 = 2000): 2000 x nf=400 x 3 channels of BPTT gives 19 updates per epoch on a 48 k record; 256 gives 154, which is what the prior runs used
p.add_argument('--width', type=int, default=8)          # Jan L27: f and h nets, 2 x 8
p.add_argument('--ewidth', type=int, default=16)        # Jan L28: encoder net, 2 x 16
p.add_argument('--timeout', type=float, default=None)   # DEV: fit() only writes its npz/save_system after the loop; a wall-clock kill loses everything (run 74045). timeout makes fit() return normally instead
p.add_argument('--n-its', type=int, default=None)       # DEV: paired arms must be matched on UPDATE count, not epochs; nf changes updates-per-epoch by ~9x so epochs are not comparable
p.add_argument('--seed', type=int, default=None)        # DEV: a paired nf comparison is only clean if both arms start from the same weights
p.add_argument('--bla', choices=['off', 'dyn', 'full'], default='off')   # DEV: BLA initialisation, Ramkannan et al. IFAC 2023 (10.1016/j.ifacol.2023.10.010); see bla_init.py. Data-driven, so still a black box
p.add_argument('--bla-zero-nl', action='store_true')    # DEV: start the nonlinear branches at zero so epoch 0 IS the BLA; the paper leaves them random, and with poles at z=1 the difference is worth measuring
p.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'results'))
args = p.parse_args()

train_data = load(args.train, args.fs)                  # Jan L11
val_data = load(args.val, args.fs)                      # Jan L12

## ------------- Add noise -----------------
# DEV(Jan L14-L24, deliberately absent): the gantry records are noiseless Simulink output, so
# there is no SNR to set. Adding noise here would change the problem, not match the reference.

## ------------- Train fit system -----------------
if args.seed is not None:                               # DEV: see --seed
    import torch; torch.manual_seed(args.seed); np.random.seed(args.seed)
dof = 4                                                 # DEV(Jan L7 dof=3): the 8-state truth has 4 dof, X + Theta + Y + the hidden absorber
h_net_kwargs = f_net_kwargs = {"n_hidden_layers": 2, "n_nodes_per_layer": args.width}      # Jan L27
e_net_kwargs = {"n_hidden_layers": 2, "n_nodes_per_layer": args.ewidth}                    # Jan L28
hf_net_kwargs = dict(f_net=default_state_net, f_net_kwargs=f_net_kwargs, h_net_kwargs=h_net_kwargs, h_net=default_output_net)  # Jan L29, verbatim
fit_sys = SS_encoder_general_hf(nx=dof*2, na=dof*4+1, nb=dof*4+1, e_net_kwargs=e_net_kwargs, hf_net_kwargs=hf_net_kwargs)      # Jan L30, same formulas at dof=4: nx=8, na=nb=17
fit_sys.unique_code = f"fs{int(args.fs)}nf{args.nf}s{args.seed}"   # DEV: `name` is a property over unique_code and keys the _best/_last checkpoints; without this the paired arms overwrite each other

if args.bla != 'off':                                   # DEV: see --bla and bla_init.py
    # init_model must run first so the nets and the norm exist. It creates the optimizer too, and
    # fit() then skips its own init (fit_system.py:311-321), so no lr may be passed to fit (D-101).
    # No optimizer_kwargs here means torch.optim.Adam's default lr=1e-3, which is Jan L33's rate.
    fit_sys.init_model(sys_data=train_data, auto_fit_norm=True)
    from bla_init import apply_bla_init
    apply_bla_init(fit_sys, train_data, mode=args.bla, zero_nonlinear=args.bla_zero_nl)

nf = args.nf; epochs = args.epochs; batch_size = args.batch_size                           # Jan L32
os.makedirs(args.out, exist_ok=True)
tag = (f'fs{int(args.fs)}_nf{nf}' + (f'_s{args.seed}' if args.seed is not None else '')
       + (f'_bla{args.bla}' + ('z' if args.bla_zero_nl else '') if args.bla != 'off' else ''))
fit_sys.unique_code = tag                               # DEV: keep the checkpoint key aligned with the tag so arms never collide
fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs,
            auto_fit_norm=True, loss_kwargs={'nf': nf}, validation_measure="sim-RMS",      # Jan L33, verbatim up to here
            timeout=args.timeout, n_its=args.n_its)     # DEV: see --timeout and --n-its above

# ------------- Save fit system -----------------
fit_sys.save_system(os.path.join(args.out, f'ann_blackbox_{tag}'))                          # Jan L36-L38

# DEV(beyond Jan L38): the brief's acceptance criterion is each arm against its own epoch-0,
# and Theta lives in the x1 - x2 difference of two channels three orders larger, so a pooled
# RMS alone cannot show it. Both are written out here rather than reconstructed from the log.
sim = fit_sys.apply_experiment(val_data)
per_channel = sim.RMS(val_data, multi_average=False)
json.dump(dict(fs=args.fs, nf=nf, epochs=epochs, batch_size=batch_size, nx=dof*2, na=dof*4+1,
               train=args.train, val=args.val, width=args.width, ewidth=args.ewidth,
               epoch0_sim_rms=float(fit_sys.Loss_val[0]), best_sim_rms=float(fit_sys.bestfit),
               final_sim_rms=float(fit_sys.Loss_val[-1]), max_sim_rms=float(np.max(fit_sys.Loss_val)),
               best_epoch=float(fit_sys.epoch_id[int(np.argmin(fit_sys.Loss_val))]),
               n_val=len(fit_sys.Loss_val), rms_x1=float(per_channel[0]),
               rms_x2=float(per_channel[1]), rms_Y=float(per_channel[2]),
               loss_val=[float(v) for v in fit_sys.Loss_val],
               loss_train=[float(v) for v in fit_sys.Loss_train]),
          open(os.path.join(args.out, f'metrics_{tag}.json'), 'w'), indent=1)
# DEV: score against the bars re-derived in bars.py rather than leaving that to the reader.
# V2 values from results/bars.json; free run, oracle initial state, 4 kHz.
BARS = dict(floor=4.652e-05, baseline=1.883e-04, frozen_lti=4.781e-04, mean_pred=1.470e-01)
print(f'[{tag}] epoch0 {fit_sys.Loss_val[0]:.4e}  best {fit_sys.bestfit:.4e}  final {fit_sys.Loss_val[-1]:.4e}'
      f'  per-channel [x1 {per_channel[0]:.4e}, x2 {per_channel[1]:.4e}, Y {per_channel[2]:.4e}] m')
print(f'[{tag}] vs bars on {args.val}: ' + '  '.join(
    f'{k} {v:.3e} ({fit_sys.bestfit/v:.1f}x)' for k, v in BARS.items()))
