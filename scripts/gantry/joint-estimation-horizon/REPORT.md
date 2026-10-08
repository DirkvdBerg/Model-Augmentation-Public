# joint-estimation-horizon report

```
(morning summary, written when the gates are done)
```

## G0 Infrastructure: PASS
Criteria JH-002. Runs g0_vendor_production, g0_repo_production, g0_vendor_noproj14, check_vendor.

| Criterion | Result | Threshold |
|-|-|-|
| (a) vendored CFG, epoch-0 closed-loop validation (Loss_val[0]) | **1.476803295949e-05 m** vs 1.4768032959e-05 (blackbox-closed-loop G1(c)) | 1e-6 rel |
| (a') same from the ORIGINAL repo packages, separate process | 1.476803295949e-05 m, identical to 13 digits | 1e-6 rel |
| (b) joint-estimation arm (OBC_ARM=noproj, 14 records, detuned start) | **2.296469801877e-05 m** vs 84032 logged 2.296473121532e-05: 1.4e-6 rel (1.6e-7 from 83795/84037) | 1e-5 rel |
| (c) import leak / vendor diff | 27 guarded modules all from vendor/; 140 vendored files, 0 unmarked differences | 0 |

Per record (a): V1 2.0190e-05, V2 2.0301e-05, V3 2.0585e-05, V4 2.0587e-05, VP1 3.415e-06,
VP2 3.531e-06 m. CPU, float32, eager (compile_mode None, the only override besides device).
Resources: peak tree 820 MB per run, 125 to 187 s; minimum available RAM 2.56 GB (the machine was
in daytime use; the 2.5 GB kill line was not crossed).

## G1 Information versus window length: PENDING
Criteria JH-003 (simulated), JH-004 (Telica).

## G2 The staged test: PENDING
Criteria JH-005.

## G3 Recommendation and server preparation: PENDING
Criteria JH-006.

## Window memory, methods A and B (JH-013, D-220): DONE 2026-09-27
Run `outputs/wm_ab/` (`run.log`, `window_memory.json`), script `memory/window_memory.py`, 203 s CPU.
- G-A PASS at all five Y (state-space PS against `common.loop`, max relative error 4.9e-11); loop stable (max |z| 0.9955).
- Closed-loop poles with K1 (same at every Y to 3 %): slowest three real, tau 10.4 to 11.1 ms (one per axis);
  54 Hz pair zeta 0.44 (tau 5.5 to 6.4 ms); the absorber's closed-loop mode 266 Hz, zeta 0.127, tau 4.7 ms
  (open loop zeta 0.03, tau 35 ms: the loop damps it); 72 Hz pair zeta 0.45. Literature multiples of the
  slowest: 1.5x = 17 ms (Gyorok), 4x = 44 ms (Beintema 2021). 0.1 s = 9 tau_slow.
- A-full: `B(T) * T` is constant from T = 0.05 s on (to 0.01 ms), so the missed energy is zero beyond about 50 ms
  and B(T) = D / T with the blind start D per residual class:
  | class | D [ms] (five Y) | B(0.1 s) | B(0.2 s) | B(0.4 s) |
  |-|-|-|-|-|
  | white | 4.6 to 5.0 | 4.6 to 5.0 % | 2.3 to 2.5 % | 1.2 % |
  | multisine band 106 to 297 Hz | 4.4 to 5.2 | 4.4 to 5.2 % | 2.2 to 2.6 % | 1.1 to 1.3 % |
  | velocity-type, low-pass 10 Hz | 11.2 to 11.4 | 11.2 to 11.4 % | 5.6 % | 2.8 % |
- Deviation from JH-013, stated: T10 and T1 turned out to measure the blind start (T1 = 10 T10 exactly, the 1/T law),
  not a tail. The memory statement is the constancy of `B(T) * T` above; T10 and T1 are kept in the JSON only.
- A-band against B (identical line inversion): `B(T) * T` differs by 0.5 to 0.8 ms at 0.1 s and 1.8 to 3.1 ms at
  0.4 s (white and band), 0.3 to 0.6 ms (low-pass). Both grow linearly with T (a non-decaying floor of 3 to
  4 % of the energy), the same in A-band and B, so the floor is the band-limited 1.5 Hz inversion, not the loop.
  The measured response (friction truth, BLA) adds 0.4 to 0.8 % of the energy as a slow tail over the
  frictionless analytic one; noise share of B below 0.03 %.
- G-band (white, band): A-band T10 / A-full T10 = 1.39 to 1.48 (PASS, under 1.5); low-pass FAILS (A-band T10 > 0.6 s). Choice rule (white, band): B / A-full T10 = 1.54 to 1.63,
  within 2 at every Y: A is the bound, B its data confirmation. B alone cannot set a bound (floor artefact) and
  cannot speak to the low-pass class (band starts at 1 Hz).
- Reading: in the residual closed-loop rollout a force residual stays visible for about 50 ms; 0.1 s spans that at
  every Y and for every residual class. A longer window only shrinks the blind start D / T, at a cost linear in T.
  Not covered: nonlinear stick dwell on the multisine-free routes (FRF records stick 2 to 10 % of the time).
