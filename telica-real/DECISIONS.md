# telica-real decisions (TR log)

Local decision log for the real-Telica setup (handoff `tasks/handoffs/2026-09-22-telica-real-gantry.md`).
Replaces `docs/decisions.md` for this folder. Each entry is written BEFORE the code it governs.
Gate thresholds are pre-registered here before the gate is computed.

### [TR-001] Folder is self-contained; vendored code is a working-tree copy; one import root
**Date**: 2026-09-22
**What**: All code for this work lives in `telica-real/`. Framework and pipeline code are COPIED
into `telica-real/vendor/` from the working tree (HEAD `9372b09`, the handoff named its parent `a97a577`; tree dirty) and recorded in
`vendor/VENDORED.md`. Every script puts exactly one root on `sys.path`: `telica-real/vendor`
(plus its own folder for sibling modules). Data is read from `kamtin-data/` through the vendored
loader, never copied. Edits to vendored files are marked `# TELICA-REAL:` inline so they can be
grepped against the source.
**Rejected**: importing from `scripts/` and `model_augmentation/` in place (the handoff forbids
edits there, and an in-place import would let the user's ongoing work change this setup under it).

### [TR-002] Watchdog thresholds and G0 acceptance
**Date**: 2026-09-22
**What**: `tools/watchdog.ps1` launches the command through `cmd.exe /c`, samples every 5 s the
process tree's summed working set and CPU (share of all logical cores), system available RAM
(`Win32_OperatingSystem.FreePhysicalMemory`) and C: free space into `<Out>/resources.csv`, prints
`WATCHDOG ALERT` below 4 GB RAM or 3 GB disk, and prints `WATCHDOG KILL` and kills the tree
(`taskkill /T /F`) below 2.5 GB RAM or 2.0 GB disk. Thresholds are the handoff's, not re-derived.
It sets `PYTHONIOENCODING=utf-8`, `PYTHONUNBUFFERED=1`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`
for the child. Child stdout/stderr are inherited so output streams live.
**G0 PASS (pre-registered)**: (a) a 30 s dummy Python load under the watchdog exits normally and
leaves `resources.csv` with at least 5 samples; (b) the same load with a forced threshold (RAM kill
level above the machine's total RAM) prints `WATCHDOG KILL` within the first two samples and no
process of the tree survives (checked by PID afterwards).
**Measured context**: at session start the machine had 4.7 GB available RAM of 15.8 GB (editors,
OneDrive, browser) and 8.9 GB free on C:. The RAM kill level therefore sits only ~2 GB below the
idle state: every script must stay under ~1.5 GB working set.
**Amendment (import roots)**: two roots, both inside telica-real: `telica-real/vendor` (vendored
code, first) and `telica-real/` (this folder's own packages `params`, `friction`, ...), set by
`tr_env.py`. The repo is pip-installed in develop mode (`ModelAugmentation.egg-info`), so an
unbootstrapped script would import the ORIGINAL `model_augmentation`; `tr_env.check_no_leak()`
fails loudly on that.

### [TR-003] Parameter set: Telica datasheet where it has the value, Garcia 2013 Table I otherwise
**Date**: 2026-09-22
**What**: One module, `params/telica_params.py`, is the single source of every physical number
used by the baseline, the friction law and the controller path. Values:
- Datasheet `literature/gantry/telica-xyz-0750-0800-data.pdf` (ETEL ASME-YGNN-08-0750-0800W3 v1.0):
  p. 2 moving mass X 91.0 kg per gantry beam, Y 19 kg (Z 1.7 kg listed separately); static
  friction (maximal) X 2 x 43 N, Y 49 N; dynamic friction (maximal) X 2 x 136, Y 98 N/(m/s);
  p. 3 Kt X 109, Y 77.6 N/Arms (identical to `Telica 1.mat`, which the loader reads).
- Mapping onto the model: `m1+m2+mb+mh = 91.0` (the whole X-moving assembly carries the Y
  carriage), `mh = 19.0`; the split of the remaining 72.0 kg over m1:m2:mb is NOT on the datasheet
  and uses Garcia's proportions 10.2:10.7:22.8 (HEURISTIC, D-112's split; only `m_total` and
  `m_diff` of it are identifiable). cg1 = cg2 = 136 (per X motor), cy = 98, cc1 = cc2 = 43,
  ccy = 49. The friction values are MAXIMAL spec values, so they are upper bounds and inits, not
  estimates.
- Garcia 2013 Table I (`garcia2013`, PDF pp. 12-13) for what the datasheet lacks: Jb 1.0, Jh 0.05
  kg m^2, cb1 = cb2 = 9 Nm/(rad/s), kb1 = kb2 = 1987.5 Nm/rad, Lb 0.725 m, d 0.1 m. These describe
  a different gantry; they are placeholders that G4 may move (only through the identifiable
  combinations).
- Friction-law constants: V_BRK = 2.25e-3 m/s (D-209, `lee2020feeddrive` Table 5, a transfer);
  tanh width v0 = 1e-3 m/s (D-116, HEURISTIC).
**Rejected**: (1) Jb, Jh from Garcia's own bar formula (eq. 21) applied to the Telica masses: it
compounds the heuristic mass split and moves J_eff by 12 %; J only enters through J_eff, which G4
can recover. (2) Kamtin FP nominal masses (`gantry_ss.py`): contradicted by the datasheet (D-112).

### [TR-004] Force scale: represented as an input gain on the logged current, both readings kept
**Date**: 2026-09-22
**What**: The model works in physical newtons with datasheet masses, and the plant input is
`F = I_logged * Kt * s_F` per stage axis. Two readings are consistent with the data, and they are
kept side by side in `params/`:
- (A) datasheet masses are right, the logged current needs `s_F = [3.469, 3.469, 3.202]`
  (D-112 HEURISTIC; a factor 2 is the forcer pair D-112 reads from `Telica 1.mat` SubAxes, the
  remaining ~1.73 / 1.60 is unexplained, candidate sqrt(3)).
- (B) the logged current is Arms and `F = I*Kt` (s_F = 1), and the data's effective masses are
  26.2 kg (X) and 5.9 kg (Y), i.e. the datasheet masses would be wrong by 3.5x.
Only ratios `param / s_F` enter the closed-loop trajectories, so the two cannot be told apart by
fitting masses; they CAN be separated by any force-like parameter known in absolute newtons. The
datasheet friction maxima are such a handle, one-sided: recovered `cc` in (A)-units above
43 / 49 N contradicts (A), since the kinetic Coulomb force cannot exceed the maximal static one.
(A) is the default representation because it is the only one that is exact: (B) with
`s_X != s_Y` would need a mass matrix scaled row-wise, `S^-1 M`, which is not symmetric in the
Theta-Y block, i.e. not a mechanical mass matrix. This is a representation choice, not a verdict
on which reading is true.
**Datasheet cross-check recorded, not resolved**: the datasheet lists motors X1-L, X2-L, X1-R, X2-R
where L/R are the LEFT and RIGHT gantries (8 controlled axes: 4 X, 2 Y, 2 Z), and per-motor
Ic * Kt = 9.05 * 109 = 986 N against the continuous-force row 916 N, i.e. one motor per logged
X channel of one gantry at Kt = 109. If D-112's "forcer pair per logged channel" reads the X1-L/X1-R
sub-axes of `Telica 1.mat`, that pair may be the two gantries' motors, not two forcers on BHL; the
factor 2 of (A) is then unsupported. Stated for Kamtin, not settled here.

### [TR-005] G1 acceptance (pre-registered)
**Date**: 2026-09-22
**What**: G1 PASS iff, checked by `params/check_params.py` under the watchdog: (1) every name the
baseline block reads (the 14 raw scalars and Lb), the friction law reads (cc1, cc2, ccy, V_BRK,
V0_TANH) and the controller path reads (Kt_X, Kt_Y, s_F_X, s_F_Y, Ts) has a row in
`params.telica_params.ROWS`; (2) the vendored `gantry_ss` module constants and the vendored
`gantry_dynamic.controller` constants equal the params values exactly (float32 cast for
gantry_ss); (3) the vendored loader's `_A_TO_N` equals `params.a_to_n()` exactly and its Kt read
from `Telica 1.mat` equals the datasheet Kt; (4) no numeric literal of the old Garcia/Kamtin sets
(22.8, 10.1, 10.2, 10.7, 14.5, 20.3, 1987.5, 0.725) remains as an assignment in the vendored
`gantry_ss.py` or `gantry_dynamic/controller.py`.

### [TR-006] Friction port: Karnopp stage-frame force at both u-sites of Gantry_State_Block; V_BRK kept, step condition derived
**Date**: 2026-09-22
**What**: `friction/karnopp.py` ports the friction block of
`Matlab-scripts/Augmentation-coulomb/gantrySystemExtendedCoulomb.m` (lines 137-243: slip
`cc*sign(v)`, stuck set `|v| < V_BRK` solved through the Delassus matrix `G = P' M(Y)^-1 P`,
active-set breakaway saturating at `cc*sign(F_required)`, 4 iterations) to batched torch, and a
mixin `FrictionMixin` overrides `Gantry_State_Block._deriv_with` so friction enters as
`u_eff = u_log - P F_stage` at BOTH u-sites (the D-116 / D-138 placement). The friction solve uses
`d(qdd)/d(u_eff) = M(Y)^-1 = N(Y)/d(Y)` (the LFR identity; equal to plant_coulomb's
`Bw_v [Y M_r; Y^2 M_r] + Bu_v`, which reduces to `M_r` algebraically). No hidden MSD. Modes:
`karnopp` (the law), `tanh` (D-116 surrogate `cc tanh(v/v0)`), `sign`, `none`. The mixin works on
`Reduced_Gantry_State_Block` too, so G4 can train the ten combinations and `log cc` jointly.
**Superseded in part by TR-015**: on real data the G4 baseline uses `tanh` (Karnopp's stick branch
drives a limit cycle the machine does not show); Karnopp stays verified and available.
**Which form each later gate uses** (as first planned): G4 replays and the G4 recovery objective use `karnopp` (its
gradient w.r.t. cc is `sign(v)` on sliding rails and exactly zero on held rails, which is the
physics); `tanh` is kept for any objective that needs `dF/dv` at `v = 0` and is named where used.
**V_BRK and the step size (the D-209 reasoning, re-derived)**: D-209 made V_BRK a physical
breakaway velocity (2.25e-3 m/s, a transfer from `lee2020feeddrive`), independent of the step. The
step enters only as a VALIDITY condition of the discrete band: a sliding rail decelerating through
zero must land inside `(-V_BRK, V_BRK)` rather than jump over it, i.e. `a_max * h < 2 V_BRK`.
# THEORY: datasheet p. 2 "Maximum acceleration" X 30, Y 50 m/s^2 as a_max.
At h = 5e-5 s (20 kHz): 50 * 5e-5 = 2.5e-3 < 4.5e-3, margin 1.8x: VALID, V_BRK unchanged.
At h = 2.5e-4 s (4 kHz, the old pipeline default): 1.25e-2 (Y) and 7.5e-3 (X) > 4.5e-3: the band
can be jumped, so V_BRK does NOT transfer to 4 kHz; a 4 kHz model would need V_BRK >= 6.3e-3 m/s,
outside the lee2020 value and a different physical claim. This is one reason (with G3) to run the
friction model at 20 kHz.
**Rejected**: re-scaling V_BRK with h (the pre-D-209 `ts`-scaled band, which D-209 retired as
excitation-dependent); porting the pre-D-209 `plant_coulomb.py` band `(cc1+cc2)/m_total*ts`.

### [TR-007] G2 acceptance (pre-registered)
**Date**: 2026-09-22
**What**: MATLAB R2025a is callable (`matlab -batch`), so the reference is MATLAB. Trajectory: 8000
RK4 steps at h = 5e-5 s (0.4 s), ZOH stage forces, x0 at rest at Y = 0.1 m, Telica params
(TR-003, cc = 43/43/49 N): 0.05 s of sub-cc forces [20, -15, 25] N (all rails must stay stuck),
then `[250 sin(2 pi 5 t) + 30, 250 sin(2 pi 5 t + 0.5) - 20, 120 sin(2 pi 7 t + 1)]` N (slip,
reversal, re-stick, breakaway). References, all float64, same RK4 and ZOH:
- R2: `friction/matlab/gantry6Coulomb.m`, a 6-state copy of the `.m` with the absorber DOF deleted
  and the friction block VERBATIM (it also returns the stuck flags for the census).
- R1: the unmodified `gantrySystemExtendedCoulomb.m` at ma = 1e-6 and 2e-6 kg (ka, ca giving a
  200 Hz, zeta 0.5 absorber, L0 = 0), Richardson-extrapolated to ma = 0: `y0 = 2 y(1e-6) - y(2e-6)`.
PASS iff all of:
(a) cc = 0: the friction block equals the parent `Gantry_State_Block` bit-for-bit (max abs diff
    exactly 0.0) over 2000 steps of the same input, LPV branch (Y_op = None) and frozen branch,
    float64 and float32.
(b) R2: max |y_torch - y_R2| over all steps and channels <= 1e-12 m, and the per-rail census
    (stuck-and-held, sliding) identical step by step.
(c) R1: max |y_torch - y0| <= max(|y(1e-6) - y(2e-6)|_max, 1e-12 m), i.e. the torch port is
    closer to the ma -> 0 limit of the original file than the first-order absorber effect itself.
(d) Census non-degenerate: every rail has > 0 held steps and > 0 sliding steps, and all rails
    are held throughout the first 0.05 s.
The float64 tolerance 1e-12 m is ~1e4 x the D-116 cross-check residual (2.8e-18 m over 6000
steps): room for roundoff between the LFR rational form and MATLAB's backslash, none for a law
difference, since one flipped stick decision moves the state by orders more.

### [TR-008] Controller path: zpk file -> snapped integrators -> second-order sections, at 20 kHz only
**Date**: 2026-09-22
**What**: The controller is built from `kamtin-data/dFeedbackControllersTelica.mat` (zpk), not
from the num/den export: `controller/export_zpk_bhl.m` writes the exact zeros, poles and gains of
rows LX1, LX2, LY to `controller/telica_zpk_bhl.mat` (a derived constant file, 3 x 8th order;
the 6x6 has 0 nonzero off-diagonal entries). Measured in that export: the "integrator" poles sit
at |z| = 1 - 7.2e-9 (LX1), 1 + 8.3e-9 (LX2), 1 + 6.2e-8 (LY): the stored zpk carries rounding,
and two of the three are marginally UNSTABLE. They are snapped to exactly z = 1 (the schema
records "integrators by design"); over a 1 s record the snap changes the response by < 2e-4
relative (|1-p| N = 6.2e-8 x 2e4 = 1.2e-3 on the worst axis, a bound; the true integrator state
dominates). Realization: `scipy.signal.zpk2sos` pairing, one direct-form-II-transposed state
space per section, cascaded into one (A, B, C, D) per axis, block-diagonal over the three axes
(nc = 24). Rejected: the 8th-order companion form of the num/den export (roots of an 8th-order
polynomial with poles within 2e-3 of z = 1 are ill-conditioned; float32 training would move them).
**Units and sign**: input error `e = r - y` [m] (positive = behind the reference, = M2 * 1e-6);
output current [A]; force `F = diag(Kt s_F) i` [N] (TR-004). In the residual form
`u_plant = u_data + diag(Kt s_F) K (y_data - y_model)`: `y_data - y_model = e_model - e_data`, so
the same K with the same sign acts on it.
**Rate**: 20 kHz, i.e. the controller's own Ts = 5e-5 s, one controller update per log sample.
Reasons: (1) the zpk IS a 20 kHz discrete operator; any other rate needs a re-discretization,
which changes the operator, and at 4 kHz (Nyquist 2 kHz) the LX1/LX2 zeros at 1872/1877 Hz and
poles at 1104/1505 Hz sit next to Nyquist; (2) the error spectrum peaks at 2393 and 3057 Hz are
above a 4 kHz pipeline's Nyquist and would alias; (3) the Karnopp band is valid at 20 kHz only
(TR-006). The bank refuses any other step size. Cost: 5x the steps per horizon of the old 4 kHz
default, paid on the server.

### [TR-009] G3 acceptance (pre-registered)
**Date**: 2026-09-22
**What**: Replay on EVERY iter0 record of the split (11 train + 2 validation + 2 test operating
points), on the WHOLE record (standstill included, `load_telica_log_full`), controller fed the
LOGGED error `e = M2 * 1e-6`, compared with the logged `MF230`. Per axis j and record r the model
is `i_log = g i_K + Phi c` with `i_K` the zero-state replay and `Phi` the controller's own
zero-input-response basis (8 columns: the unknown controller state at the log start; the
integrator column is a constant), fitted by least squares; `g_rj` per record, `g_j` pooled.
(i) PASS iff for every axis `|g_j - 1| <= delta_j`, with the noise-derived half-width
    `delta_j = max(3 * std_r(g_rj) / sqrt(n_r), b_j)`, `b_j` the errors-in-variables attenuation
    from the logged-error quantization: `b_j = s2 / (var(i_K) + s2)`,
    `s2 = (Delta_e^2 / 12) * sum_k h_k^2` (Delta_e = smallest nonzero step of the logged M2,
    h = controller impulse response). If that fails, (i) can still pass only through a NAMED
    mechanism that is then modelled, and the modelled replay must meet the same interval.
    Checked alongside, reported not gated: saturation (max |MF230| against the datasheet peak
    current 27.9 Arms, and plateau fraction), feedforward leakage (max |MF30 - MF230| on iter0,
    must be exactly 0), cross-axis gains (i_log_j on all three replays: off-diagonal share), and
    the best integer lag in [-100, 100] samples.
(ii) PASS iff the residual-form loop with the Telica bank returns `u_data` EXACTLY (max abs diff
    0.0) when `y_model = y_data`, over a whole record, in float32 and float64; plus the units gate:
    with `y_model = y_data + d` for a known d, `u_cl - u_data` equals `diag(Kt s_F) K(-d)` from
    `scipy.signal.sosfilt` within 1e-6 relative (float64) and 1e-3 relative (float32).

### [TR-010] G3 attempt 1 FAILED; attempt 2 is mechanism identification, not a new gain statistic
**Date**: 2026-09-22
**What**: Attempt 1 (TR-009 model M1: zero-state replay + the controller's zero-input response)
falsified the leading hypothesis that the D-073 mismatch is the controller's unknown state at the
log start: g1 = 1.196 / 1.355 / 3.531, within 0.6 % of g0, identical across all 14 usable records.
The H1 ratio logged/replay is frequency dependent at coherence 0.95-1.0, so the pre-registered
scalar-gain statistic cannot pass by any re-estimation; the logged path is a DIFFERENT LINEAR
OPERATOR from the zpk as stored (deep X2 notch at 299 Hz absent in the log, Y notch region
different, integral band 1.3x / 1.4x / 3.6x stronger). Attempt 2 tests named mechanisms, each with
a pre-stated signature: (a) row mix-up: one of the six zpk rows flattens the ratio to ~1 on an
axis; (b) logical-frame controller with a decoupling transform: the other X rail's error adds
substantial VAF (> 0.01) to an X current. If neither holds, attempt 3 models the mechanism
"the implemented M2 -> MF230 path differs from the zpk file" directly: identify the logged path
from the iter0 I/O pairs (they are exact controller I/O pairs by the schema) on the TRAIN
records, and require the identified path to meet the TR-009 interval (pooled gain of the
identified replay against MF230) on the held-out validation and test iter0 records.

### [TR-011] G3 attempt 3: the logged path M2 -> MF230, identified on train iter0, validated held out
**Date**: 2026-09-22
**What**: Attempt 2 found neither a row mix-up nor a decoupling transform (RUNS g3_mechanism).
Mechanism adopted for attempt 3, named at the level the data supports: the controller that ran
during these logs is a linear time-invariant operator (coherence 0.95-1.0) that differs from the
zpk file AS STORED in its low-frequency tuning (1.0-4x magnitude, ~-33 deg extra phase over
5-50 Hz) and its notch placement (X2 299 Hz, Y 150-400 Hz). Its physical origin (a retuned
controller, a different sensor in the loop than the logged one, a drive-side filter) is NOT
identifiable from these channels and is left open for Kamtin.
**Why identifying it is the right training operator regardless of that origin**: the residual
form feeds back `K (y_data - y_model)` where y is the LOGGED position (q1 = M0 - M2) and the
correction lands on the LOGGED current. The operator the loop needs is therefore exactly the map
from the logged position error to the logged current, which iter0 measures as an exact I/O pair
(feedforward 0.0 A). If the machine's controller actually acts on another sensor, the identified
map is the effective controller that folds that sensor difference in; it is still the one that
reproduces the logged loop.
**Method**: per axis, the zpk's own structure (zpk2sos sections, same order 8, integrator FIXED
at z = 1, poles kept stable by a (1 - exp(rho), theta) / tanh parameterization, section-1
numerator carries the gain, others monic) is re-fitted by output-error least squares
(`scipy.optimize.least_squares`, init = the zpk) on the TRAIN iter0 records only: residual
`MF230[k] - K_theta(e)[k - d] - c_rec` from sample 1000 (50 ms transient) to the end, `c_rec` a
per-record constant (the integrator's initial state, eliminated in closed form), integer lag
d in [-3, 3] chosen on train. The train-degenerate record's Y axis is excluded.
**Acceptance (pre-registered, before the fit)**: TR-009 (i) applied to the IDENTIFIED replay on the
HELD-OUT validation + test iter0 records (4): pooled gain g_j of MF230 on [K_hat(e), Phi_hat]
within `1 +- delta_j`, `delta_j = max(3 std_r(g_rj) / sqrt(4), b_j)`. Reported alongside, not
gated: held-out VAF against the zpk replay's VAF, and the H1 ratio flatness (median abs log
ratio, 5 Hz-5 kHz, coherence > 0.9). If the identified K_hat destabilises the nominal loop that
is a G4 finding, not a G3 one. This is the LAST G3 attempt.

### [TR-012] G3 (ii) float32 fix: the controller computes in float64 inside a float32 rollout
**Date**: 2026-09-22
**What**: G3 attempt 3 passed (i) (RUNS g3_identify). The first (ii) run (g3_bank) passed the exact
identity in both dtypes and the float64 units test (2.25e-11), but the float32 units test gave
8.7e-3 relative against the 1e-3 tolerance. Mechanism: in float32 the rounded biquad coefficients
of the section that carries the integrator (a1 = -(1+p), a2 = p) move its pole off z = 1 by
O(1e-7 / (1 - p)), and a pole at 1 + 1e-7 grows ~0.2 % over 15k steps; the notch sections add
coefficient sensitivity on top. Fix, a numerical realization change and not a new hypothesis
about the controller: `ControllerBank` gains `compute_dtype`; with float64 it stores its matrices
and state in float64, casts the incoming error up and the feedback force back down to the
rollout dtype. Autograd passes through both casts. The acceptance of (ii) is unchanged and is
re-run as is. Counted in REPORT as a fourth documented change to G3 (numerical, part ii).
**Rejected**: loosening the float32 tolerance (it was pre-registered); running the whole pipeline
in float64 (config.py measured it as 2x memory for no Adam benefit; only the controller needs it).
**Outcome (2026-09-22)**: the float64 controller did NOT fix it (g3_bank_f64ctrl: 9.13e-3). The
TR-012 mechanism was wrong. Diagnostic g3_bank_diag found the real one: the float32 rollout agrees
to 4.6e-8 with the float64 reference fed the error AS FLOAT32 REPRESENTS IT. The rounding sits in
the INPUT: normalised absolute positions are O(1) in float32, so `y_data - y_model` carries
2.2 / 2.3 / 4.3 nm of rounding per sample (ystd 19 / 19 / 38 mm), fed through a controller of
~2.5e8 N/m. `compute_dtype` stays (harmless; a no-op in a float64 run).

### [TR-013] Real-data pipeline runs in float64
**Date**: 2026-09-22
**What**: `use_f64=True` for every real-data run of the vendored pipeline (G6 enforces it). Reason,
measured (TR-012 outcome): float32 normalised positions round the servo error by 2-4 nm per
sample, ABOVE the encoder resolution (9.77e-10 m/cnt, `Telica 1.mat`; logged M2 step 1.0e-9 m),
so a float32 closed loop injects noise larger than the data's own quantization through a
2.5e8 N/m controller: 0.9 % of the peak feedback force. The servo errors are ~1 um and the
residual the augmentation must learn is a fraction of that, so this is not a rounding detail.
config.py's "float64 buys nothing" was measured on SIMULATED data at 4 kHz with a 3rd-order
controller; it does not transfer.
**Rejected**: re-centring the normalisation per window (the model state carries absolute
position; a per-window offset changes the framework's state convention, a larger intervention
than a dtype switch the pipeline already supports).

### [TR-014] G4 data facts, method and acceptance (pre-registered)
**Date**: 2026-09-22
**Facts that shape the method** (runs g4_survey, g4_holding, summaries only): 159 readable logs
(`train/xpos_-60_ypos120/iter6.log` is empty); EVERY record is one positive move (X +40 mm at
1.0 m/s, 39.5 m/s^2; Y +80 mm at 1.5 m/s, 51 m/s^2), 0/159 with a reversal. So during sliding,
Coulomb friction and a constant force are the same regressor. The holding force at rest flips
sign between the pre-move and post-move dwell in 159 / 159 / 158 of 159 records, with implied
static external force ~0 (median -1 / -10 / +18 N, IQR spanning 0): it is friction memory (the
controller integrator stops where stiction catches the rail), which separates cc from a constant
force. Friction level held at rest: 83 / 102 / 81 N (reading A) = 24 / 29 / 25 N (reading B).
**Consequence for TR-004 (recorded, not settled)**: the data's ratio of held friction force to
inertial force is ~2x the datasheet's (static-friction max / moving mass), a RATIO no force scale
can reconcile. Reading A: the machine carries ~2x the datasheet's maximal static friction
(plausible: cable carriers and seals are not in a bare-axis spec). Reading B: the datasheet
moving masses are ~3.5x too high. The datasheet friction maxima are therefore NOT used as bounds.
**Method**:
(1) Recovery by equation error on TRAIN records (all iterations), sliding samples only: stage
positions and total force zero-phase low-passed (Butterworth 4, 200 Hz; HEURISTIC: 2.9x below
the lowest unmodelled error-spectrum peak, 586 Hz, and above the 5-50 Hz motion band), velocity
and acceleration by central differences, rows decimated 10x, samples within 5 ms of |v| < V_BRK
on an involved rail excluded (set-valued friction). Logical EOM rows X, Theta, Y, linear in
theta = [m_total, mh, m_diff, M11 = J_eff + mh d^2, mh d, cg1, cg2, cy, cb_sum, kb_sum, cc1, cc2,
ccy], F_ext = 0 (holding test). Force reading A. A parameter is MOVED from its TR-003 value only
if its relative standard error (OLS, rows weighted by the per-row residual std) is < 20 %
(HEURISTIC); otherwise it stays at the datasheet / Garcia value and is reported as not
identified from this excitation. Y is taken in the machine's coordinate (assumed centred on the
beam; an offset would appear as an unmodellable M(Y) term).
(2) Closed-loop replay, float64, 20 kHz, identified controller (G3), every readable record,
from 100 ms before motion to the end: DIRECT form (model driven by r and the logged feedforward,
u = u_ff + K(r - y_model)); and RESIDUAL form (the training form, u = u_data + K(y_data - y_model)).
Variants: V_rec (recovered + Karnopp cc), V_ds (TR-003 + Karnopp cc 43/43/49), V_nf (TR-003, no
friction), V_70821 (70821 parameters, simulations/server-output/telica_param_recovery_70821.out
Table 1-2, no friction).
(3) Metrics per record and axis over the motion segment: NRMSE_direct = rms(e_sim - e_meas) /
rms(e_meas); NRMSE_residual = rms(y_model - y_data) / rms(e_meas) (100 % = no better than
predicting perfect tracking). Noise floor: PSD of e_meas in the pre-motion standstill; residual
PSD (e_sim - e_meas) per band against it.
**Acceptance (G4 PASS iff all)**: (a) STABLE: every record, axis and both forms of V_rec finite
with max|error| <= 100 max|e_meas| (HEURISTIC; the 70821 divergence reached ~4e5 A); (b) BETTER
THAN 70821 on held-out (val + test, all iterations): median over (record, axis) of NRMSE_direct
of V_rec below that of V_70821; (c) PHYSICAL: the ten combinations positive (m_diff signed), M(Y)
positive definite and d(Y) != 0 over Y in [-0.25, 0.25] m, viscous and Coulomb coefficients >= 0,
each recovered cc <= the 97.5th percentile of that rail's held-friction estimate
(F_end - F_start)/2 (static >= kinetic; data-derived), m_total + mh and mh within +-50 % of the
datasheet (HEURISTIC; reading A); (d) the thresholds above are this entry's, unchanged after
seeing results.

### [TR-015] G4 attempt 1 FAILED; attempt 2 = held-friction cc + tanh stick zone (Karnopp stick rejected on real data)
**Date**: 2026-09-22
**Attempt 1 outcome** (runs g4_recover, g4_replay_rec): (a) bounded on 159/159 records in both forms,
but every record settles after the move into a sustained oscillation (X ~0.8 um near 100 Hz,
Y ~4.6 um near 190 Hz) that the machine does not show (measured standstill 7-11 nm); (c) FAILS:
recovered ccy = 170.8 N above the Y held-friction 97.5th percentile (90.3 N); cc1, cc2 not
identifiable from the regression (one-way moves: only cc1 + cc2 ~ 300 N enters the X row).
**Mechanism, measured** (run g4_diag_lc, 4 held-out iter0 records, attempt-1 parameters): the
frictionless loop with the identified controller is linearly stable (least-damped 186 Hz,
zeta 0.011; 311 Hz, zeta 0.033) and its replay settles to 12 / 12 / 15 nm; the Karnopp law
(set-valued stick, V_BRK = 2.25e-3) produces the 0.8 / 0.7 / 4.6 um limit cycle, and a 20x
narrower band (1e-4) still leaves 0.4 / 0.4 / 1.0 um; the D-116 tanh form (v0 = 1e-3) settles to
12 / 12 / 8 nm. So the oscillation is a stick-slip limit cycle of the relay-like stick branch
inside a high-gain loop. The machine, whose standstill error is at the 7-11 nm noise floor, does
not have that behaviour at these amplitudes: 0.4-4.6 um lies inside the pre-sliding range
(Armstrong: 2-5 um breakaway displacement, D-209 open item 3), where the rail is held by a stiff,
damped contact, not by a Coulomb relay. D-209's own pending check `V_BRK * T_dwell <= x_b` fails
on the Telica dwells (2.25e-3 m/s x ~0.25 s = 0.56 mm >> 5 um): a Karnopp stuck rail keeps any
residual velocity below V_BRK, because the stick solve holds the acceleration, not the velocity.
Side finding (same run): the stored zpk controller closes an UNSTABLE loop with this plant model
(|lambda| up to 1.0031 at 276-282 Hz, all three Y), the identified controller a stable one: the
70821 closed-loop divergence was the zpk file, not only the wrong plant.
**Attempt 2 (pre-registered, TR-014 acceptance unchanged)**:
(1) Friction form: the D-116 tanh law `F = cc tanh(v / v0)`, v0 = 1e-3 m/s (HEURISTIC, D-116),
in the baseline for the G4 replays and for G6 training. For |v| >> v0 (the whole sliding phase:
0.1-1.5 m/s) it equals Karnopp's slip branch `cc sign(v)` to 1e-40 relative; it differs only in
the stick zone, where it acts as heavy damping (cc / v0 ~ 8e4 Ns/m), the closest interpretable
stand-in for pre-sliding. Karnopp stays in `friction/` (verified in G2) and is REJECTED as the
baseline's stick model on real data by the measurement above. This deviates from the handoff's
"Karnopp Coulomb friction in the baseline" and is flagged for the user in REPORT.md.
Rejected: Karnopp with a data-derived band (1e-4 measured, still 0.4-1 um); Karnopp with a velocity
reset on stuck rails (changes the .m law, and a relay at standstill remains); a Dahl / LuGre
pre-sliding state (identifiability, D-209 rejection 3).
(2) cc from the measured held friction (train medians 84.1 / 100.3 / 80.5 N, run g4_holding,
reading A) instead of the regression; the ten linear parameters re-fitted by the same equation
error with cc fixed (`recover.py --cc-from-holding`). Criterion (c)'s cc bound is then met by
construction; it is still reported, and the regression's 170.8 N on Y is reported as rejected
(kinetic above static friction is unphysical; the excess is a direction-fixed force the one-way
moves cannot separate from Coulomb friction).
(3) For (b), V_70821 is replayed with the same code (frictionless, 70821 parameters).

### [TR-016] Frozen normalisation from the train split, filtered velocities (G6 item 3)
**Date**: 2026-09-22
**What**: `pipeline/real_data.py::build_frozen_norm` computes x_mean/std_x, u_mean/std_u, y0/ystd
once from the train records ([motion - 100 ms, end], every 10th sample), with velocities from a
200 Hz zero-phase low-pass of the positions (the handoff's item 3: differencing noisy positions
inflates dTheta ~190x), and stores `pipeline/norm_frozen.json`. Training, validation, test and the
server read that file; nothing recomputes statistics from the data being scored. The encoder's
scaling matrices are built from the same constants (a two-row synthetic System_data whose std
equals the frozen std exactly).
**Rejected**: per-split statistics (val/test would be normalised by themselves); a calibration
record (no single Telica record spans the X/Y range of the split).

### [TR-017] G5 learnability: metric, thresholds and ANN-setting rules (pre-registered)
**Date**: 2026-09-22
**Source**: the deep-research run (REPORT G5 lists the citations). Residual `eps = y_model - y_data`
of the G4 attempt-2 baseline in the RESIDUAL (training) form, stored at 5 kHz after an 8th-order
2 kHz zero-phase anti-alias filter (`learnability/make_residuals.py`).
**M1 (ceiling, gate)**: repeatable residual power across exact repeats of the reference.
# THEORY: pintelon2012sysid eqs. (2-36), (2-37), book p50 (sample mean and variance over M
# repeats); schoppe2016measuring eqs. (14)-(15) (signal power / noise power).
Groups: iter0 across the 15 operating points (feedforward 0) and iterETEL across the 15 (the same
ETEL feedforward rule everywhere); a group is valid only if the reference relative to its value at
motion start differs by <= 1 um between members (checked, not assumed). Window: from motion start,
common length. Per axis and DFT bin k: `Ebar = mean_m E_m(k)`, `s2 = var_m E_m(k)` (ddof 1),
`T(k) = M |Ebar|^2 / s2 ~ F(2, 2M-2)` under "no repeatable residual" (DERIVED by the research run
from the P&S construction, eqs. 2-39/2-40 p51; HEURISTIC in this use), Benjamini-Hochberg FDR
q = 0.01 over 1 Hz-2.5 kHz. `SP = sum_k (|Ebar|^2 - s2/M)`, `NP = sum_k s2` (Parseval, per-sample
powers). Records at different positions are NOT noise-only repeats: position-dependent structure
counts as NP here, so SP/NP is a CONSERVATIVE (lower) bound on learnable structure. M1 passes on an
axis iff >= 1 significant bin and SP/NP >= 1 (0 dB, HEURISTIC). The Y axis of
`train/xpos_-210_ypos-200/iter0` (MF230_Y constant) is excluded.
**M2 (floor, go decision)**: cross-validated model-error model on EXTERNAL regressors only.
# THEORY: ljung1999mem eqs. (12), (24) (MEM, FIR form); forssell1999closed Akaike's rule (report
# eqs. 71-72): in closed loop use a regressor uncorrelated with the noise, standard choice x = r.
Regressors (never u_data): stage reference acceleration and velocity (6), logged feedforward (3),
Y_ref times each reference acceleration (3, scheduling); FIR lags 0..99 at 5 kHz (20 ms), ridge
lambda = 1e-6 trace(X'X)/p (numerical only, HEURISTIC). Fitted on TRAIN records (all iterations,
motion segment), scored on HELD-OUT val + test records (all iterations): `d_n = (||eps_n||^2 -
||eps_n - eps_hat_n||^2) / N_n`, `H = mean_n d_n / NP_sample` (NP from the M1 iter0 group, per axis,
per sample: conservative); one-sided t-test of d_n > 0 over held-out records (records as units).
M2 passes on an axis iff H >= 1 and p < 0.01. Reported alongside: held-out R^2, and H against the
standstill floor (optimistic).
**Verdict**: GO ("server run warranted: yes") iff some axis passes M1 AND M2. CONDITIONAL (answer
"no, not yet") if M1 passes but M2 does not on every M1 axis. NO-GO if no axis passes M1.
**Descriptive only**: M3 correlation of eps with the reference acceleration of each axis (routing
evidence, P&S p442: no whiteness test on output-error residuals); M5 gradient SNR / noise scale at
initialisation (mccandlish2018empirical eq. 2.9 `B_simple = tr(Sigma)/|G|^2`; liu2020gsnr eq. 2), a
pipeline sanity check run with the G6 model, not part of the decision.
**ANN settings from the residual** (written to `learnability/ann_settings.json`):
- routing: the velocity rows (dX, dTheta, dY) of every logical DOF whose residual (X = mean of
  rails, Theta = difference / Lb, Y) passes M1 or M2, never Theta alone (X and Y always kept when
  any passes, project constraint D-103), plus every augmented-state row;
- nx_ann: rank of the Hankel matrix of the M2 FIR impulse responses from the reference
  accelerations (3 x 3 block), counting singular values above the 95th percentile of the largest
  singular value of Hankel matrices built from FIR perturbations drawn from the fit's covariance
  (inflated by the residual autocorrelation factor); rounded up to even, clipped to [2, 16].
  # THEORY: Ho-Kalman / Kung realization: the Hankel rank of an impulse response is the minimal
  # state dimension. The noise threshold is HEURISTIC.
- horizon nf_seconds: time at which the M2 impulse responses reach 99 % of their energy, rounded up
  to 5 ms, at least 5 ms (HEURISTIC);
- band: the frequency range of M1-significant bins with per-bin SP >= NP (reported; the ANN has no
  band parameter, the band describes where the residual carries repeatable power).

### [TR-018] ANN settings after G5: nx_ann = 2 (rule floor), horizon 40 ms (rule undetermined)
**Date**: 2026-09-23
**What**: The TR-017 settings rule ran as registered. Its order test was inconclusive: the Hankel
singular values of the model-error-model impulse response decay slowly with no gap (0.51 of the
largest at #12) and all lie under the coefficient-noise threshold, both in the registered 12-channel
FIR (collinear regressors: every record runs the same profile) and in a descriptive r_dd-only
40 ms FIR (run g5_order_sens). The response also has not decayed at either window edge (99 % energy
at 20 ms and at 40 ms). Reading: the residual is not a low-order LTI response to the reference; it
has slow content (M1 band from 3 Hz; PSD/floor 1e5 below 20 Hz), consistent with unmodelled
friction/compliance effects rather than one missing resonance. Settings used for G6 and the
server: nx_ann = 2 (the rule's clip floor, not an estimate), horizon nf_seconds = 0.04 (the
largest window measured, still a lower bound), routing the velocity rows of X, Theta and Y plus
both augmented rows (every logical DOF passes M1 and M2), band 3 Hz-2.5 kHz (descriptive).
**Rejected**: nx_ann = 8 from the simulation era (sized for the simulated 212 Hz absorber, which
does not exist here); a longer MEM to chase the horizon (collinearity grows with the window, and
the gate verdict does not depend on it). The server runner exposes NX_ANN and NF_SECONDS overrides
so the user can run a second arm (e.g. nx_ann 8, 0.1 s) without editing files.

### [TR-019] G6 acceptance and smoke-test shape (pre-registered)
**Date**: 2026-09-23
**What**: `pipeline/telica_augment.py` with `TELICA_SMOKE=1`: 2 train + 2 val + 2 test records cut
to 6000 samples (300 ms at 20 kHz), float64, CPU, eager, batch 8, one short window (nf = 5 ms =
100 steps), `n_its = 2`. PASS iff, in one run under the watchdog: (1) data, frozen norm, model
(G4 attempt-2 baseline with tanh friction, identified Telica bank in float64, ANN settings from
G5) build without error; (2) exactly 2 optimizer updates are taken (`fit_sys.batch_counter == 2`)
and at least one ANN parameter changes; (3) the closed-loop validation and the output-only
evaluation (augmented vs ANN-zeroed baseline, per record, metres) complete; (4) peak process RAM is
recorded (Windows peak working set) and the run ends without a watchdog kill. A memory probe
(forward + backward at two horizons, no optimizer step) in the same run gives the per-window-step
graph memory for the server batch estimate.

### [TR-020] Server run budget and memory sizing
**Date**: 2026-09-23
**What**: Measured (run g6_probe, float64, CPU eager, the full model with the identified bank):
the retained autograd graph costs 80.3 KB per window-step. At the G5 horizon (nf = 800 steps =
40 ms at 20 kHz) and batch 256 that is 16.4 GB unchunked; with `checkpoint_chunk = 200` (exact
gradient checkpointing, ~+33 % compute, D-169) the live graph is 256 x 200 x 80.3 KB = 4.1 GB, plus
the window arrays on the host (117 train records, stride 20: ~7e4 windows x 800 x 6 x 8 B =
~2.7 GB) and ~1 GB of process overhead. Batch 256 against the measured gradient-noise scale
B_simple = 6.7 (M5): the batch is far above it, so the gradient direction is not noise-limited.
Budget: 50 epochs, validation every 1400 updates (~5 epochs) on 4 validation records, 24 h wall;
`--mem=64gb` host. These are run-budget choices, labelled HEURISTIC; the server's own first
validations are the check. The runner accepts NX_ANN / NF_SECONDS / BATCH / CHUNK / EPOCHS / LR /
STRIDE / ITS_PER_VAL / COMPILE_MODE overrides for a second arm without editing files.
**Rejected**: running the 4 kHz simulation defaults (TR-008); float32 (TR-013).

### [TR-021] G6 verdict evidence
**Date**: 2026-09-23
**What**: TR-019 (2) "at least one ANN parameter changes" is read from the smoke run's deepSI
`_last` checkpoint against its `_best` (initial) checkpoint, because deepSI reloads the best weights
at the end of fit(), and in a 2-update run the best validation was the initial one (3.82e-7 m
before, 8.04e-7 m after). This reading uses no extra optimizer update (the handoff caps the smoke at
2). The in-process memory counter of that run read 0 (ctypes signature bug, fixed); the recorded
peak for the smoke run is the watchdog's sampled process-tree working set, and the probe run gives
the in-process peak for the same model at the server horizon.

### [TR-022] 5 kHz training rate: decimation rule, controller at 5 kHz, acceptance (pre-registered)
**Date**: 2026-09-23
**Why**: user request, to cut the server cost 4x. 5 kHz (D = 4): Nyquist 2.5 kHz stays above the
controller's highest features (zeros at 1.87 kHz) and leaves a transition band for the anti-alias
filter; the residual carrying power sits below 1 kHz (G4 PSD: residual/floor 75-1000 at 300-1000 Hz,
~2 above 1 kHz). What is given up: content above ~2 kHz (the 2.4 / 3.1 kHz error peaks, ~2x floor).
The Karnopp band condition (TR-006) no longer binds: the baseline uses tanh (TR-015).
**Decimation rule** (`rate.py`): positions and the reference zero-phase Butterworth order 8 at
2 kHz, then every 4th sample; forces and logged currents block-mean over each 4-sample hold interval
(ZOH-consistent input, D-087). Sample k at 5 kHz = sample 4k at 20 kHz. Logged error for controller
work = filtered r - filtered y, decimated.
**Controller at 5 kHz** (`controller/identify_5k.py`): initialised from the 20 kHz identified
controller by matched pole-zero mapping (p -> p^4, z -> z^4, gain matched at 100 Hz, integrator
kept at z = 1), then refined by the TR-011 output-error fit on decimated TRAIN iter0 (decimated
error -> block-mean MF230), lag in {-1, 0, 1}. Stored as `controller/telica_sos_identified_5k.npz`.
**Acceptance, 5 kHz variant becomes the server default iff all hold** (else 20 kHz stays):
(A) held-out val + test iter0 pooled gain within `1 +- delta_j` per axis (TR-009 construction at
    5 kHz);
(B) residual form with the 5 kHz bank: `u_cl == u_data` exactly (0.0) when `y_model = y_data`, and
    units rel. error <= 1e-6 against scipy `sosfilt`, float64;
(C) G4 replay of the attempt-2 baseline at Ts = 2e-4 s with the 5 kHz controller: stable on all
    readable records in both forms (TR-014 (a) bound), held-out NRMSE_direct median <= 1.1 x the
    20 kHz value (1.952 -> 2.147) and residual-form median <= 1.1 x 1.982 = 2.180 (HEURISTIC 10 %);
(D) pipeline at 5 kHz: static checks pass and a forward + backward probe at nf = 200 (40 ms) runs
    with NO optimizer step (the smoke test's 2 updates are not repeated).
Up to 3 documented attempts; the frozen normalisation is reused (physical statistics, rate
independent).

### [TR-023] Watchdog RAM levels lowered for the 5 kHz runs (user-authorised)
**Date**: 2026-09-23
**What**: The machine is in daytime use (browser, editor, other sessions) with ~2.0 GB available,
below the TR-002 kill level of 2.5 GB, so every launch was killed at start. The user authorised
lowering the level. For the TR-022 runs: `-RamKillGB 0.75 -RamAlertGB 1.25` (disk levels
unchanged, 2.0 / 3.0 GB). 0.75 GB is kept as a floor for the OS; the runs themselves need
0.5-1.0 GB. TR-002's levels stay the default of `watchdog.ps1`; the lower ones are passed per run
and recorded in RUNS.md.
**TR-022 amendment, attempt 2 (2026-09-23)**: attempt 1 (run g3_5k) passed (A) on every axis
(held-out gain 0.9988 / 1.0024 / 1.0020) but chose lag -1 everywhere, i.e. the current LEADING the
error by one 5 kHz sample: non-causal, so it cannot close the loop (`telica_bank` refuses it).
Mechanism: an alignment artifact of the decimation rule, not controller behaviour. The block-mean
current of interval [4k, 4k+4) is centred 1.5 fine samples after the point-sampled error at 4k.
The causal fit is almost as good (train VAF X 0.9978 at both lags, Y 0.9993 at 0 vs 0.9996 at -1).
Attempt 2 restricts the lag to the causal set {0, 1}; acceptance (A) unchanged.

**TR-022 outcome (2026-09-23): PASS, 5 kHz is the server default.** (A) causal refit, lag 0: held-out
gain 0.9988 / 1.0025 / 1.0013, VAF 0.998 / 0.998 / 0.9996 (Y better than the 20 kHz 0.940).
(B) identity 0.0, units 1.5e-11 (float64). (C) stable 159/159 both forms; held-out NRMSE_direct
1.947 (<= 2.147), residual 1.963 (<= 2.180); iter0 0.199 / 0.148 / 0.333; replay 57 s vs 208 s.
(D) static checks pass, forward + backward probe runs at nf 200 (81 KB per window-step, peak
697 MB). Server settings follow (TR-020 re-sized): nf 200 (40 ms), batch 256 -> 4.1 GB graph,
no checkpointing; stride 5 (the 1 ms spacing of the 20 kHz stride 20); host arrays ~0.2 GB. The
20 kHz path remains available (`FS_TRAIN=20000`).

### [TR-024] nx_ann = 8 (user decision) and the server runner aligned with the user's GPU template
**Date**: 2026-09-23
**What**: (1) nx_ann = 8, the user's choice (G5's order test was inconclusive: the rule floor was 2,
the recommendation 6). Routing: velocity rows of X, Theta, Y plus all 8 augmented rows
(`[3, 4, 5, 6, ..., 13]`); encoder window na = nb = 2 (6 + 8) + 1 = 29 samples (5.8 ms at 5 kHz).
(2) `pipeline/runners/server_run.sh` adopts the conventions of the user's working GPU runner
(`gantry_interconnect_dynamic` GPU script): partition `oahu`, the existing log folder
`logs/augmentation/augmentation-closed-loop/`, `srun --cpu-bind=cores`, `NUMEXPR_NUM_THREADS`,
GPU / host-CPU / toolchain (gcc, triton) diagnostics, the OS-keyed inductor cache, and its sanity
lines. Carried-over caveat, not measurable here: that template's timing (0.50 s/update, RTX 2080 Ti)
is float32; consumer GPUs run float64 at ~1/32 of float32, while this loop is dispatch-bound, so the
float64 penalty on such a card is unknown. The runner prints the card; an A100-class card (full
float64 rate) is preferred if the partition offers one.
**Rejected**: keeping nx_ann = 2 in the settings file (the user chose 8); float32 on the server
(TR-013: the 2-4 nm rounding would be above the encoder resolution).

### [TR-025] Joint estimation + OBC arms on real data: 13 protected directions, stick-zone mask (pre-registered)
**Date**: 2026-09-23
**What (user decisions: cc trainable, start at G4, stick-zone mask)**:
- Trainable block `ReducedGantryFrictionBlockCC`: the ten identifiable combinations (D-190, log
  coordinates, `m_diff` relative-linear) PLUS `log(cc / cc_G4)` for cc1, cc2, ccy: a 13-vector
  `free_params`, zero at the G4 attempt-2 values (combos, gauge and cc from
  `baseline/recovered_params_a2.json`; no detune). cc travels in the per-pass structure tuple as its
  11th element, so the rollout, the OBC basis (`transition_from_free`) and the per-step correction
  (`mats_and_tangent_from_free` -> `_rk4_with_tangent`) all see the same cc.
- Friction law tanh (TR-015) with the STICK-ZONE MASK: `F = cc_g tanh(v / v0)`,
  `cc_g = where(|v| > 3 v0, cc, cc.detach())` (HEURISTIC factor 3: tanh(3) = 0.995, i.e. the
  rail is within 0.5 % of the Coulomb level). The forward value is unchanged; cc receives gradient,
  and a forward-mode tangent, only from sliding samples. The state dependence (d F / d v) is not
  masked.
- Hand-written tangent `FrictionMixin._deriv_both_with` (the per-step OBC path, which compiles):
  the parent's term-by-term tangent plus `s_F = where(mask, d cc, 0) tanh(v/v0) + cc sech^2(v/v0)
  (P' s_qdot) / v0`, entering at both u-sites.
- Real-data reference set: every train sample at 5 kHz from [motion - 100 ms, end], q = P^-T y,
  qdot from the 200 Hz zero-phase low-passed positions (the TR-016 rule; the simulation's
  fourth-order difference of raw positions is refused on noisy data by its own guard), u the
  block-mean force, x_a = 0, frozen normalisation.
- Meters: drift is measured against the G4 values (`combo_ref`), not the `gantry_ss` "truth"
  (the vendored probe's own comment asks for this on real data); cc printed alongside.
- Arms via `OBC_ARM` in `pipeline/telica_augment.py`, identical except for the one field:
  noproj (joint estimation, no projection), obc (13-column tangent), obc_affine (+ frozen offset
  column); all with param_prior=False, start at G4.
**Gates, before any server run, no optimizer step (PASS iff all)**:
(T1) tangent: `transition_and_tangent_from_free` vs `torch.func.jvp` of `transition_from_free` on
     real reference tuples (sliding and standstill), random 13-direction: max rel. error <= 1e-10
     (float64);
(T2) primal: `transition_from_free(free_params, z)` equals the block's own step exactly (0.0);
(T3) mask: with every rail inside |v| < 3 v0 the cc gradient is exactly 0; with sliding it is
     nonzero; the stick-zone share of the cc gradient without the mask is reported;
(T4) OBC orthogonality: after the correction, the ANN field's component along the basis is
     <= 1e-8 relative (the construction's defining property), measured on the reference set;
(T5) per arm: the model builds, the basis builds at the G4 point (numerical rank and singular
     values reported; the lifecycle's `expected_rank` is set to that measured rank and stated),
     and forward + backward of the loss run.
**TR-025 T3a amendment (2026-09-23, test design, attempt 2)**: run g6_joint passed T1 (2.3e-16),
T2 (0.0), T3b, T4a (2.2e-13), T4b (6.4e-17) and T5 (all three arms), but T3a read 2.3 (against
1.2e5 while sliding) on ONE RK4 STEP started in the stick zone. The mask acts per derivative
evaluation; an RK4 step evaluates four stages, and with the forces during motion the stage
velocities leave |v| < 3 v0 within one 2e-4 s step. The pre-registered statement is therefore
tested where it holds: on single derivative evaluations (`_deriv_with`) at stick-zone states,
cc gradient exactly 0. The one-step value is reported, not gated.
**TR-025 outcome (2026-09-23): gates PASS (run g6_joint_a2).** T1 tangent vs jvp 2.3e-16; T2 exact;
T3a 0.0 per derivative evaluation, T3b nonzero; stick-zone share of the unmasked cc gradient on
closed-loop windows 5.6 / 11.5 / 3.5 % (X1 / X2 / Y), removed by the mask; T4a orthogonality
2.2e-13, T4b per-step correction == jvp 6.4e-17; T5 all three arms build and run forward +
backward. Basis at the G4 point: tangent rank 13/13, cond 6.1e4 (weakest directions kb_sum and
cb_sum, tiny columns: barely excited by one-way moves, as G4 found); column correlations
cb_sum~J_eff 0.99, cg1~cg2 -0.99, cc1~cc2 -0.99; affine rank 14/14, scaled cond 71.
`obc_expected_rank = 13` recorded in ann_settings.json (the lifecycle aborts below it).
