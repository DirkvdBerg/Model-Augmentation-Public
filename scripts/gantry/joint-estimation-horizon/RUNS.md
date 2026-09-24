# joint-estimation-horizon runs

Every run goes through `tools/run.sh` (watchdog, JH-001). Hypothesis written BEFORE launch;
outcome after. Output of run `<name>` is in `outputs/<name>/` (`run.log`, `resources.csv`,
`watchdog_summary.txt`).

| Run | Gate | Script / args | Hypothesis (before launch) | Outcome |
|-|-|-|-|-|
| g0_vendor_production | G0 (a) | `tools/g0_epoch0.py --source vendor --arm production` | Vendored CFG model reproduces Loss_val[0] = 1.4768032959e-05 m within 1e-6 rel | 1.476803295949e-05 m (rel diff < 1e-10); per record identical to BB G1(c); 27 modules from vendor/; 128 s, peak 819 MB, min RAM 2.86 GB |
| g0_repo_production | G0 (a') | `tools/g0_epoch0.py --source repo --arm production` | Original repo packages give the same number as g0_vendor_production within 1e-6 rel | 1.476803295949e-05 m, identical to 13 digits; no module from vendor/ |
| g0_vendor_noproj14 | G0 (b) | `tools/g0_epoch0.py --source vendor --arm noproj14` | Joint-estimation arm on the 14-record set reproduces 84032's 2.296473121532472e-05 within 1e-5 rel | 2.296469801877e-05 m: 1.4e-6 rel from 84032, 1.6e-7 from 83795/84037 (2.29646944e-05); reduced block at the detuned start |
| compile_all | infra | `tools/compile_all.py` (MIN_RAM_GB=2.8, 30 MB) | Every session script compiles | 8 files, 0 errors; 3.6 s, peak 30 MB |
| g1_checks | G1 V1/V2 | `info/g1_info.py --checks-only` | ANN-free rollout = production rollout to <= 1e-12 of ystd; forward mode = central FD to <= 1e-4 | V1 PASS (0.0 exactly). V2 FAIL attempt 1: kb_sum 3.1e-4, cg1 5.8e-4, cg2 2.0e-4, cy 6.1e-4; all others <= 1.4e-6 (JH-008). 96 s, 821 MB |
| g1_sim | G1 | `info/g1_info.py` | Slow combinations (cg1, cg2, cy; rail time constants 1.0-1.3 s) need longer nf than the mass terms to reach sd < 1 %; x0-free costs most at nf 100-400 | |
| g1_telica | G1 (Telica) | `info/g1_telica.py` | Theta-row combinations (kb_sum, cb_sum, J_eff, d) never reach 1 % on Telica's one-move logs (telica-real G4 relative SE 0.3-1.8 on full records) | |
| g2_a | G2 (a) | `staged/g2_staged.py --arm a` | One-step baseline-only fit lands within tolerance of J+Delta* (R2 with curvature only) | |
| g2_i_3200 | G2 (i) | `staged/g2_staged.py --arm i --nf 3200` | Unrestricted long-window baseline-only fit is biased away from J+Delta* (the absorber's multistep effect enters the parameters) | |
| g2_ii_3200 | G2 (ii) | `staged/g2_staged.py --arm ii --nf 3200` | With c = (I-P0) r0 supplied, the long-window fit lands within 1.2 pp / 0.74 pp RMS of J+Delta* (first-order identity; gap = curvature, obliqueness, stencil artefacts) | |
| g2_i_400 | G2 (i-400) | `staged/g2_staged.py --arm i --nf 400` | Secondary: horizon dependence of the unrestricted bias | |
| g2_ii_400 | G2 (ii-400) | `staged/g2_staged.py --arm ii --nf 400` | Secondary: with c supplied the horizon should not matter at first order | |
| g1_checks2 | G1 V2 attempt 2 | `info/g1_info.py --checks-only` (h ladder) | The four damping and stiffness disagreements are FD rounding: error falls as h grows from 1e-6 and reaches <= 1e-4 | |
