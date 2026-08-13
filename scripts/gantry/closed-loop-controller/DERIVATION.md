# The controller in closed form, and the closed-loop signal relations

Purpose: write the controller that generates the records as an explicit formula rather than as
a MATLAB object, then prove in Python that the formula reproduces MATLAB's controller on a real
record. Once that gate passes, the loop can be written into the derivation of the model
comparison instead of being an opaque part of the data generation.

Theory source: course 5SMB0 System Identification, `literature/experiment-design/System-identification/`,
Lecture 11 (Closed-loop Identification) for the signal relations, Lecture 13 (Best Linear
Approximation) for the input-class remark in section 5.

## 1. Frame, and what is stored

The controller acts in the **stage** frame, on the three measured stage coordinates
`[X1, X2, Y]`. `gtd_build_plant.m:22-28`:

```matlab
    sys = P.' * getss(cfg.n, M_op, cfg.C_damp, cfg.K) * P;

    Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
    for j = 1:3
        Cfb(j,j) = ruleOfThumb(cfg.fbw, sys(j,j), ts);
    end
    Cfb = ss(Cfb);
```

`P` maps a stage force to a logical force and `P'` maps a logical position back to stage, with
`P = [1, 1, 0; Lb/2, -Lb/2, 0; 0, 0, 1]` (`gtd_config.m:61`), so `sys` is stage in and stage
out. `Cfb` is **diagonal in the stage frame**, one independent SISO loop per rail plus one on Y.
This is the point at which the coordinate question is settled: any formula written for this loop
must be in stage coordinates, and the logical frame appears only inside `sys`.

`M_op` is built with the **full** payload mass `mh` and a **frozen** `Y_op` (D-039), so the
controller is LTI for the whole record even though the plant is not.

## 2. The controller, in closed form

`ruleOfThumb.m:2-14`:

```matlab
    s = tf('s');
    int_factor = 1/6;
    int = (s+2*pi*fbw*int_factor)/s;
    leadlag_factor = [1/3, 3];
    leadlag = (s+2*pi*fbw*leadlag_factor(1))/(s+2*pi*fbw*leadlag_factor(2));
    lowpass_factor = 10;
    lowpass = 2*pi*lowpass_factor*fbw/(s+2*pi*lowpass_factor*fbw);

    Cnorm = int*leadlag*lowpass;
    K = 1/abs(freqresp(sys*Cnorm, 2*pi*fbw));
    C = K*Cnorm;

    C = c2d(C, ts, 'tustin');
```

Write `w = 2*pi*f_bw` with `f_bw = 100` Hz (`gtd_config.m:62`). Then the shape is the same for
all three channels,

    Cnorm(s) = (s + w/6)/s  *  (s + w/3)/(s + 3w)  *  10w/(s + 10w)

which is a PID in series form: a lag-type integrator with its zero at `w/6`, a lead-lag pair
`w/3` over `3w` giving the phase margin at crossover, and a first-order roll-off at `10w`. As a
single rational function,

    Cnorm(s) = 10w (s + w/6)(s + w/3) / [ s (s + 3w)(s + 10w) ]

Only the gain is channel dependent, and it is fixed by requiring unit open-loop gain at the
bandwidth,

    K_j = 1 / | sys_jj(i w) * Cnorm(i w) |          j = 1, 2, 3

    C_j(s) = K_j * Cnorm(s)

so `f_bw = 100` Hz is the 0 dB crossover of each loop by construction. The plant enters the
controller **only** through the scalar `|sys_jj(i w)|`, which is why the design is frozen at one
`Y_op`: change `Y_op` and only these three numbers move.

Discretisation is Tustin at `ts = 5e-5` s, no prewarping, so

    C_j(z) = C_j(s)|,  s = (2/ts) (z - 1)/(z + 1)

and the controller actually applied is `Cfb(z) = diag(C_1(z), C_2(z), C_3(z))`.

## 3. The loop, written out

The generator forms, `gtd_run_simulation.m:33-34`:

```matlab
    u_fb    = lsim(plant.Cfb, r_sim - q_with);
    u_total = u_fb + f_ms;
```

so with `G` the (discretised) plant from stage force to stage position, `r` the reference and
`f_ms` the injected multisine,

    u_fb    = Cfb (r - q)
    u_total = u_fb + f_ms
    q       = G u_total

Eliminating `q`,

    (I + G Cfb) q = G Cfb r + G f_ms

    q       = T r + So G f_ms,        So = (I + G Cfb)^-1,   T = So G Cfb
    u_total = Si (Cfb r + f_ms),      Si = (I + Cfb G)^-1

This is exactly the unified closed-loop form of Lecture 11 slide 6 and 7, with the course's
`r = K r1 + r2` realised here as `Cfb r + f_ms`: the reference enters through the controller and
the multisine enters at the plant input. The course writes `y = Go So r + So v` and
`u = So r - K So v`; our `v = 0`, because the simulation is noiseless.

## 4. What the formulas say about the records, and about the offset

Three consequences, in decreasing order of how much they matter here.

**(a) The applied force is the reference, filtered by the loop.** `u_total = Si (Cfb r + f_ms)`.
`Cfb` has an integrator, so `Cfb r` is large at low frequency wherever `r` is not constant. That
is the origin of the measured `92.5 %` of input power below 20 Hz and the
`[+0.66, -0.46, +0.10] N` DC in `T10_aprbs_60` reported in
`Matlab-scripts/Augmentation-no-controller/README.md`. It is not noise and not a bug: it is
`Si Cfb r` and it is fully predicted by the formula above.

**(b) Replaying a recorded `u_total` open loop is exact, loop or no loop.** `q = G u_total` holds
whatever produced `u_total`. So the open-loop replay is not an approximation of the closed-loop
experiment, and the offset seen in the replay was never caused by "the controller being missing"
in the sense of a wrong equation. What the controller changes is **which** `u_total` you get, and
therefore which part of the plant the record excites. This is the correct version of the
supervisor's `waarom trekt de controller het niet naar 0`.

**(c) In the model comparison the loop rescales the error, it does not create it.** Take a
baseline `G_hat` and the truth `G`, and write `Delta = G - G_hat` and `w = Cfb r + f_ms`. Driven
open loop by the same recorded `u_total`, the output error is `e_ol = Delta Si w`. Run instead
**inside** the same loop from the same `r`, the difference collapses exactly, with no
linearisation:

    e_cl = So Delta So_hat w,      So = (I + G Cfb)^-1,   So_hat = (I + G_hat Cfb)^-1

Both carry one sensitivity factor on the right; the closed-loop error carries one **extra** `So`
on the left. So the loop rescales the error by `So`.

**The rescaling is not a suppression in the band that matters.** Measured on the frozen design
loop by `loop_sensitivity.py`:

| f [Hz] | 1 | 10 | 50 | 100 | 150 | 180 | 500 |
|-|-|-|-|-|-|-|-|
| `smax(So)`, Y_op = 0.10 | 0.0004 | 0.0214 | 1.0544 | 1.6695 | 1.7983 | 1.8043 | 1.1438 |
| `smax(So)`, Y_op = 0.00 | 0.0004 | 0.0217 | 1.0807 | 1.6569 | 1.7937 | 1.8076 | 1.1450 |

`|So| < 1` only below about 45 Hz. There is a sensitivity peak of about **1.8** covering the whole
`[130, 180]` Hz augmentation band, including the absorber at `f_a = 150` Hz. Closing the loop
around the comparison would therefore **amplify** the discrepancy of interest by roughly 1.8 while
suppressing its low-frequency part by two to three decades. Neither helps measure the absorber, so
this is an argument for keeping the comparison open loop rather than for restoring the controller.

## 5. Where the BLA does and does not apply

Lecture 13 defines the best linear approximation for a nonlinear plant under a **specified input
class**, random-phase multisines, and its central property is that the BLA is input-class
dependent. In closed loop the input is not the designed multisine but `Si (Cfb r + f_ms)`, so the
input class is shaped by the loop and the resulting linear approximation is a closed-loop BLA,
not the open-loop one.

Two things follow, and the second is the one that matters:

- The indirect closed-loop methods of Lecture 11 (classical, two-stage, coprime factor,
  slides 29 to 32) exist to restore consistency when `u` and the noise `v` are correlated through
  the loop. **Here `v = 0`, so there is nothing to restore.** `q = G u_total` is exact and no
  indirect construction is needed. Applying the two-stage method to this data would be machinery
  without a purpose.
- What survives from Lecture 13 is the input-class argument, and it argues against putting the
  controller back in: the honest way to get an open-loop BLA is to excite open loop, which is what
  `Matlab-scripts/Augmentation-no-controller/` now does.

## 6. The gate

`verify_controller.py` builds `Cfb(z)` in Python from section 2 alone, with no MATLAB object and
no exported matrices, then applies it to the stored `r_sim - y` of a closed-loop record and
compares against the stored `u_fb`. Passing means the formulas in section 2 and 3 are the
controller that generated the data, not a plausible reconstruction of it.

Known floor: `gtd_save_record.m:25-31` stores `u_fb`, `y` and `r_sim` in **single** precision,
while `gtd_run_simulation.m:33` computed `u_fb` in double from a double `q_with`. The
reconstruction therefore drives the controller with a quantised error signal while the reference
came from an unquantised one. The mismatch is a roughly constant bias in `e` of order one float32
step, and because `Cfb` has a pole at `z = 1` that bias is **integrated into a ramp**: on
`V1_standstill_Yp10` the Y residual is 99.99 % a straight line of `+0.199 N/s`, against
`+0.172 N/s` predicted from `kappa*w/54 * mean(e)` with nothing fitted. Only the channels with
negligible bias (V1 X1 and X2, whose positions sit at zero) show the broadband quantisation term.

This floor is a property of the archive, and it grows linearly with record length. It is removed
by `test_controller_exact.py` level L4, which compares against MATLAB's `lsim` run on exactly the
signal Python forms, reaching `1.6e-16`.

## 7. Result

See `RESULT.md`, written by the run.
