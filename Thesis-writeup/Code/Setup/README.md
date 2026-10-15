# Setup: numbers of Section V (Experiment design)

One minimal script per number that justifies a choice in Section V. Each reads only the thesis dataset
(argument 1, default `Coulomb-tanh-and-MSD-lowpass-noise`, the dataset of runs 201 to 207) and its
noise-free twin, writes `Thesis-writeup/Results/setup_<name>.csv`, and registers LaTeX macros in
`Results/setup_numbers.json`, rendered into `Writing/tables/setup_numbers.tex`. The text uses the macros;
numbers are never typed in.

Run from this folder, one at a time: `conda run -n GraduationProject python <script>.py [dataset]`.

| Script | Choice it justifies | Input | Macros |
|-|-|-|-|
| `snr.py` | noise level, stated as SNR (V-A) | training records + noise-free twins | `SetupSnr{Min,Med,Max}<axis>`, `SetupNoiseRms<axis>` |
| `band_edge.py` | multisine lower edge (V-B) | meta.truth (simulated FRF, simulation only); noise-free motion-only training records | `SetupCrossMin`, `SetupCrossMax`, `SetupMotionFNN` |
| `rate.py` | model rate 4 kHz (V-B) | meta.truth | `SetupPmLossFour`, `SetupPmLossTwo`, `SetupPmGate` |
| `filter_leak.py` | anti-alias rule on y - r (V-B) | training records + twins | `SetupAliasRaw`, `SetupAliasFilt`, `SetupFiltDist` |
| `loop_memory.py` | window 0.1 s (V-B) | meta.truth (simulation only) | `SetupTauMax`, `SetupDecay`, `SetupWindow` |
| `controller_rate.py` | K1 re-discretised at 4 kHz (V-B, open todo) | training records | `SetupKcheck`, `SetupKmismatch{Med,Max}` |
| `coverage.py` | training range, E records, coverage figure data (V-B) | all 36 records (references) | `SetupTrain{Y,Vel,Acc}`, `SetupE{Y,Acc,Vel}`, `SetupESixAcc` |

`_setup_common.py`: paths, record loading, the linear models built from `meta.truth` (rigid baseline,
simulated system with absorber, K1 by the generator's `ruleOfThumb`), CSV and macro writing.
Axis macro tags: `Xone`, `Xtwo`, `Y` (LaTeX names take no digits).
