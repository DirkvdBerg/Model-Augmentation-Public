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
