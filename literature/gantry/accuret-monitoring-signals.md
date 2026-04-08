# AccurET Monitoring Signals - ETEL Dual Gantry System

Source: `AccurET-Oper&Soft-VerV.pdf`
- Section 8.1.3 (pp. 139-140): monitoring diagram and description table
- Section 8.1.4 (p. 141): cogging compensation
- Page 398: `MF230` / `MF231` / `MF232` / `MF233` definitions

---

## Column format in logged data

```text
{Device}.{Monitor}:{controller}.{axis_index}
```

Example: `BHL_GTRX1.M2:0.0` -> left beam head, gantry X motor 1, position error, controller 0, axis 0.

---

## Device naming

| Token   | Meaning                          |
|---------|----------------------------------|
| `BHL`   | Beam Head Left                   |
| `BHR`   | Beam Head Right                  |
| `GTRX1` | Gantry X-axis motor 1            |
| `GTRX2` | Gantry X-axis motor 2 (dual drive) |
| `GTRY`  | Gantry Y-axis motor              |

Each beam head has two X motors (dual drive) and one Y motor - 6 axes total.

---

## Axis index mapping (`:controller.axis`)

All axes are on controller 0:

| Suffix  | Axis         |
|---------|--------------|
| `:0.0`  | BHL GTRX1    |
| `:0.1`  | BHL GTRX2    |
| `:0.2`  | BHL GTRY     |
| `:0.3`  | BHR GTRY     |
| `:0.10` | BHR GTRX1    |
| `:0.11` | BHR GTRX2    |

---

## Core monitor signal definitions

| Monitor | Alias | Name | Description | Unit |
|---------|-------|------|-------------|------|
| `M0`    | -     | Theoretical position | Position setpoint `Xc`. Does **not** take the `SET` command into account. LSL part in `ML0`. | dpi / rdpi |
| `M1`    | -     | Real position | Real position `X`. Takes mapping corrections into account, but does **not** take the `SET` command into account. LSL part in `ML1`. | dpi / rdpi |
| `M2`    | -     | Position control error | Tracking error `Xe = M0 - M1`. | dpi / rdpi |
| `MF1`   | -     | Position loop proportional gain value | `KF1` modified by gain scheduling. | - |
| `MF2`   | -     | Position loop speed feedback gain value | `KF2` modified by gain scheduling. | - |
| `MF30`  | `TIQ` | Theoretical current `Iq` after `KF60` limitation | Total current command **after** feedforward / compensation additions and **after** the `KF60` current saturation limit. | ci / A |
| `MF230` | -     | Theoretical current `Iq` after advanced filters | Controller-side current command **after** the advanced filters, before feedforward / cogging / torque-offset additions. | ci / A |
| `MF231` | -     | Theoretical current `Iq` ffwd part | Pure feedforward current contribution. | ci / A |
| `MF232` | -     | Theoretical current `Iq` with ffwd and cogging part | Current command after adding feedforward and cogging compensation. | ci / A |
| `MF233` | -     | Theoretical current `Iq` before `KF60` limitation | Current command just before the `KF60` limiter. | ci / A |
| `MF250` | -     | Cogging compensation value | Cogging compensation contribution added after the advanced filters. | ci / A |

---

## Current-command chain

From the monitoring diagram (p. 139), the current-command path is:

```text
MF32  --advanced filters-->  MF230

MF230 + MF231                          = feedback + feedforward sum
(MF230 + MF231) + MF250               = MF232
MF232 + torque-offset compensation    = MF233
sat_KF60(MF233)                       = MF30
```

Interpretation:
- `MF230` is the feedback/controller branch after the advanced filters.
- `MF231` is the feedforward branch.
- `MF250` is the cogging compensation branch.
- `MF233` is the total internal current command before saturation.
- `MF30` is the total current command after saturation.

Important nuance:
- `MF230` is **not** the final commanded current.
- `MF30` is **not** just a limited version of `MF230`; it is a limited version of the full summed command.

If cogging compensation is disabled, then `MF250 = 0`.

---

## What the logged signals represent

For each axis in the logged dataset:
- `M0` tells you what position the position loop wanted.
- `M2` tells you the tracking error.
- `M1` can be reconstructed as `M1 = M0 - M2`.
- `MF230` tells you the controller-generated current contribution.
- `MF30` tells you the actual current command sent onward after additions and limiting.

So for each axis, these signals let you inspect:
- reference position
- realized position
- tracking error
- controller contribution
- total applied current command

---

## Reconstructing controller and feedforward from the logged data

### What can be recovered directly

If your goal is to replay the measured experiment in a plant simulation, the safest input is:

```text
u_total(t) = MF30(t)
```

This is the best proxy for the actual control input applied by the drive at the current-command level.

You can also take:

```text
u_fb_est(t) = MF230(t)
```

as the controller-side contribution after the advanced filters.

### Feedforward estimate from available signals

With only `MF230` and `MF30`, the residual is:

```text
MF30 - MF230 = u_ff + u_cog + u_offset + u_sat_error
```

where:
- `u_ff` is feedforward
- `u_cog` is cogging compensation
- `u_offset` is the torque-offset branch
- `u_sat_error = MF30 - MF233` is the effect of `KF60` saturation

Therefore:

```text
u_ff_est = MF30 - MF230
```

is equal to the true feedforward only if all of the following hold:
- cogging compensation is off (`MF250 = 0`)
- torque-offset compensation is off
- `KF60` is not saturating, so `MF30 = MF233`

Under those assumptions:

```text
u_ff_est = MF30 - MF230 = MF231
```

If those assumptions do not hold, `MF30 - MF230` is only a residual, not pure feedforward.

### Practical conclusion for simulation

For one recorded gantry experiment:
- Use `MF30` as the actual input to replay in simulation.
- Use `MF230` as the best available estimate of the controller contribution.
- Use `MF30 - MF230` as an approximate feedforward only when you are confident there is no cogging / offset contribution and no `KF60` limiting.

For identifying the full ETEL controller and feedforward law for new simulations:
- these signals are **not sufficient** by themselves.
- for an exact decomposition you would ideally also log `MF231`, `MF233`, `MF250`, and preferably `MF31`.

---

## Selected other monitors (from p. 140 table)

| Monitor | Alias | Name | Unit |
|---------|-------|------|------|
| `M10`   | -     | Theoretical speed `Vc` after advanced filter | dsi / rdsi |
| `M11`   | -     | Real speed `V` after advanced filter, depth 0 | dsi / rdsi |
| `M14`   | -     | Theoretical acceleration `Ac` | dai / rdai |
| `MF20`  | `RCUR1` | Real current in phase 1 | ci / A |
| `MF21`  | `RCUR2` | Real current in phase 2 | ci / A |
| `MF22`  | `RCUR3` | Real current in phase 3 | ci / A |
| `MF24`  | -     | Phase PWM value | Incr. |
| `MF25`  | `M25` | Phase angle | - |
| `MF27`  | `TID` | Theoretical current `Id` reference | ci / A |
| `MF28`  | `RID` | Real current `Id` measured | ci / A |
| `MF31`  | `RIQ` | Real current | ci / A |

---

## Unit glossary

| Unit   | Meaning |
|--------|---------|
| dpi    | Drive position increment |
| rdpi   | Rotary drive position increment |
| dsi    | Drive speed increment |
| dai    | Drive acceleration increment |
| ci     | Current increment |
| upi    | User position increment |
| rupi   | Rotary user position increment |
| Incr.  | ISO relative value in `[-1, +1]` (multiply by `UBUS/2 = M91/200` to get Volts) |
