# AccurET Monitoring Signals — ETEL Dual Gantry System

Source: `AccurET-Oper&Soft-VerV.pdf`
- Section 8.1.3 (pp. 139–140): Monitorings diagram and description table
- Page 398: MF230 definition

---

## Column format in logged data

```
{Device}.{Monitor}:{controller}.{axis_index}
```

Example: `BHL_GTRX1.M2:0.0` → Left beam head, gantry X motor 1, position error, controller 0, axis 0.

---

## Device naming

| Token   | Meaning                          |
|---------|----------------------------------|
| `BHL`   | Beam Head Left                   |
| `BHR`   | Beam Head Right                  |
| `GTRX1` | Gantry X-axis motor 1            |
| `GTRX2` | Gantry X-axis motor 2 (dual drive) |
| `GTRY`  | Gantry Y-axis motor              |

Each beam head has two X-motors (dual/H-bridge drive) and one Y-motor — 6 axes total.

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

## Monitor signal definitions

| Monitor | Alias | Name | Description | Unit |
|---------|-------|------|-------------|------|
| `M0`    | —     | Theoretical position | Position setpoint Xc. Does **not** account for SET command. LSL part in ML0. | dpi / rdpi |
| `M1`    | —     | Real position | Real position X. Accounts for mapping corrections and SET command. LSL part in ML1. | dpi / rdpi |
| `M2`    | —     | Position control error | Tracking error Xe = M0 − M1 (difference between theoretical and real position). | dpi / rdpi |
| `MF1`   | —     | Position loop proportional gain value | KF1 modified by Gain Scheduling. | — |
| `MF2`   | —     | Position loop speed feedback gain value | KF2 modified by Gain Scheduling. | — |
| `MF30`  | TIQ   | Theoretical current Iq (after KF60 limiter) | Iq demand **after** the KF60 current saturation limit. | ci / A |
| `MF230` | —     | Theoretical current Iq (after advanced filters) | Iq demand **before** KF60 limiter — output of the advanced filter stage. | ci / A |

### MF230 vs MF30 — current loop chain

```
Position regulators → Advanced filters → [MF230] → KF60 limiter → [MF30] → ...
```

The difference `MF230 − MF30` reveals how much current demand is clipped by the KF60 saturation limit.

---

## Selected other monitors (from p.140 table)

| Monitor | Alias | Name | Unit |
|---------|-------|------|------|
| `M10`   | —     | Theoretical speed Vc (after advanced filter) | dsi / rdsi |
| `M11`   | —     | Real speed V (after advanced filter, depth 0) | dsi / rdsi |
| `M14`   | —     | Theoretical acceleration Ac | dai / rdai |
| `MF20`  | RCUR1 | Real current in phase 1 | ci / A |
| `MF21`  | RCUR2 | Real current in phase 2 | ci / A |
| `MF22`  | RCUR3 | Real current in phase 3 | ci / A |
| `MF24`  | —     | Phase PWM value | Incr. |
| `MF25`  | M25   | Phase angle | — |
| `MF27`  | TID   | Theoretical current Id reference | ci / A |
| `MF28`  | RID   | Real current Id measured | ci / A |
| `MF31`  | RIQ   | Real current F | ci / A |
| `MF250` | —     | Cogging compensation value | ci / A |

---

## Unit glossary

| Unit   | Meaning |
|--------|---------|
| dpi    | Drive position increment |
| rdpi   | Rdpi (rotary drive position increment) |
| dsi    | Drive speed increment |
| dai    | Drive acceleration increment |
| ci     | Current increment |
| upi    | User position increment |
| rupi   | Rotary user position increment |
| Incr.  | ISO relative value in [-1, +1] (multiply by UBUS/2 = M91/200 to get Volts) |
