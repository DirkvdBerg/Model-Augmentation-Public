# Decimating the noisy 20 kHz y to 4 kHz: literature and one recommended rule

Date 2026-09-28. Task: `tasks/handoffs/2026-09-28-decimation-anti-alias-research.md` (Q1 to Q5 of its section 8).
Research on paper only: no code, data or run was touched. Criteria: D-232 (`docs/decisions.md`), numbers from
`measure_aliasing.log`. Sources: held PDFs, 11 new reference rows in `docs/references.md` (section "Decimation and
anti-aliasing of noisy closed-loop data"), PDFs in `literature/sampling-rate/`.

## Recommendation

**y_dec = r[::5] + BW8(y - r)[::5], with BW8 a zero-phase Butterworth low-pass of order 8 at 1.6 kHz
(`sosfiltfilt`, float64, second-order sections). The same y_dec feeds the controller branch and the loss target.
u unchanged: block mean of 5.**

- Why this filter class: after forward-backward filtering the magnitude is 1/(1 + x^16), x = tan(pi f/fs) / tan(pi fc/fs).
  The DC gain is exactly 1, and the passband error falls off monotonically:

  | f | 300 Hz | 800 Hz | 1 kHz |
  |-|-|-|-|
  | passband error | 1.7e-12 | 1.2e-5 | 4.4e-4 |

  The multisine response (7 to 34 um at 106 to 297 Hz) and the 0.3 m position therefore pass unchanged to far below
  1 nm. Every class that failed in D-232 has a DC or passband error that multiplies large content (Q3).
- Why the error form y - r: with this filter it gives the same interior result as filtering y (the difference is
  (1 - BW8) r, below 1e-19 relative for a reference below 100 Hz). It removes the edge transient at record ends,
  where offsets and trends drive it (Gustafsson 1996, Sects. 5 and 6), and it filters the quantity precision-motion
  practice analyses (the servo error: Heertjes 2024 Def. 2.1; Voorhoeve 2018 Def. 5.1).
- Why the same y for branch and target: see Q2 (variance against bias). It also keeps the residual identity exact:
  u_model = u_data when y_model = y_data.
- Why u stays the block mean: see Q1. u is the known ZOH input, not a measurement. It carries no noise to
  anti-alias, and the block mean is its exact 4 kHz ZOH equivalent (D-087).

Expected effect on the D-232 criteria:

| Criterion | Expected | Basis |
|-|-|-|
| (i) Q2 / Q1 | 0.91 to 0.99; Y at the low end (about 0.92) | the 1.6 kHz FIR has the same -6 dB point after forward-backward and gave 0.91 to 0.99; NT-003 measured the same Butterworth at 0.97 to 1.0 on the earlier noisy set |
| (ii) Q3 / Q1 | below 0.1 (Q3 below 0.5 nm); Y is the risk | the filter's own error below 1 kHz is at most 4.4e-4 of the content at 1 kHz. What Q3 sees is the signal content above 2 kHz that point sampling folds below 1 kHz. NT-007 (b) measured the same filter on full positions at 0.000 nm without friction and 0.6 to 2.2 nm (all frequencies) with Karnopp friction |
| (iii) Q4 / Q4_point | multisine records 1.00 +- 0.01 (nm changes against 80 to 1000 nm floors); TR-T1 below 1, roughly 0.75 to 0.95 | the ~3 nm folded excess is removed without the FIR's 85 nm distortion, which cancelled the FIR's gain on TR-T1 |

Pre-stated fallback: if (i) falls below 0.90 on Y, use fc = 1.8 kHz. Its passband error at 1 kHz is 6.1e-5, and it
cuts less in-band noise.

Test, in `measure_aliasing.py` (another session runs it). The entries take `(y, r)`. The noise part is decimated as
`dec(y_noisy, r) - dec(y_noisefree, r)`. For the four existing linear entries this equals `dec(n)`, so their numbers
do not change.

```python
from scipy.signal import butter, sosfiltfilt

# HEURISTIC: order 8, fc = 0.8 x the 2 kHz Nyquist (TR-022's ratio at 5 kHz; scipy decimate's 0.8/q)
# THEORY: Butterworth, forward-backward magnitude 1/(1 + x^16), x = tan(pi f/fs)/tan(pi fc/fs):
#   DC gain exactly 1; error 1.7e-12 at 300 Hz, 1.2e-5 at 800 Hz, 4.4e-4 at 1 kHz
AA = butter(8, 1600.0, fs=FS, output='sos')

DECIM = {
    'point': lambda y, r: y[::D],
    'block': lambda y, r: y[:len(y) // D * D].reshape(-1, D, y.shape[1]).mean(axis=1),
    'fir':   lambda y, r: filtfilt(FIR, [1.0], y, axis=0)[::D],
    'cheby': lambda y, r: decimate(y, D, n=8, ftype='iir', axis=0, zero_phase=True),
    # THEORY: Gustafsson 1996 Sects. 5-6 (offsets and trends drive forward-backward edge transients),
    # so the filter acts on the servo error y - r (Heertjes 2024 Def. 2.1)
    'butter_err': lambda y, r: r[::D] + sosfiltfilt(AA, y - r, axis=0)[::D],
    # control, optional: should equal butter_err to < 1e-3 nm away from the record edges
    'butter':     lambda y, r: sosfiltfilt(AA, y, axis=0)[::D],
}
```

Changes in the record loop:
- `r = A['r_sim']`, and `assert np.array_equal(r, B['r_sim']) and r.shape == A['y'].shape`. The twins differ only in v.
- `q2 = rms((dec(A['y'], r) - dec(B['y'], r))[:N][HOLD // D:])`.
- `yf = dec(B['y'], r)[:N]`, `y_data = dec(A['y'], r)[:N]`, `ref_free = DECIM['point'](B['y'], r)[:N]`.

Implementation in `gantry_dynamic/data.py`, after the user's OK (D-232):
- The same rule for y in the thesis noisy mode.
- The per-record floor (noisy minus noise-free) decimated with the same rule.
- u untouched.

## Corrections to the handoff's premises

1. **The Chebyshev failure is its DC gain, not end transients.**
   - The scipy 1.17 source (`decimate`, read) uses `cheby1(8, 0.05, 0.8/q)` and then `sosfiltfilt`. An even-order
     Chebyshev I has DC gain 10^(-rp/20), so the result is 10^(-0.1/20) = 0.98855.
   - On 0.300 m that is 3.43 mm, exactly the TR-S1 Y value in the log (3.43e6 nm). On standstill X it bounds the
     offset to about 6 um, which is why X shows only 66 to 90 nm there.
2. **The FIR failure is not only "a 1e-6 error on 0.3 m positions".**
   - Its signal change is uniform across standstill and moving records (55 to 100 nm X, 150 to 170 nm Y).
   - On standstill X the offset is at most 6 um (item 1), and `firwin` normalises the DC gain to exactly 1.
   - So there the change comes from passband ripple on the dynamic content: about 5e-3 of the 34 um multisine
     response (inference; the Hamming ripple value was not verified from a textbook this run).
   - That content is in y - r, since r is constant at standstill. **The handoff's example rule (FIR on y - r) would
     therefore still fail (ii) on the multisine records.** The filter class has to change, not only the signal it
     acts on.
3. **The handoff omitted two sibling results on the recommended filter.**
   - `telica-real/rate.py` (TR-022, PASS on real data): zero-phase Butterworth 8 at 2 kHz (0.8 x Nyquist at 5 kHz) on
     positions and reference, point-sampled; forces block-mean.
   - `scripts/gantry/noise-training/` NT-003, NT-004a, NT-007 (b): the same Butterworth at 1.6 kHz on full positions,
     0.000 nm distortion without friction.
   - D-232 never tested this filter class.
4. **On criterion (ii)** (a finding, criterion unchanged): Q3's reference is the point-sampled noise-free y, which is
   itself aliased. Q3 therefore also charges a filter for removing signal content above 2 kHz that point sampling
   folds into the band.

## Q1: must u and y pass the same filter? Is block-mean u with a different rule for y consistent?

**Answer.**
- The "same filter on both channels" rule belongs to the band-limited (BL) setup, where u and y are both measured
  through acquisition channels. Our setup is a zero-order-hold (ZOH) setup: the input is known exactly and only the
  output is measured.
- In the ideal ZOH setup, Pintelon and Schoukens forbid an output anti-alias filter, because it removes the repeated
  ZOH spectra that the step-invariant model reproduces. For control, where the plant bandwidth is far below the
  sampling rate, they relax this.
- So block-mean u with low-pass-then-point-sampled y is consistent to the extent that F = 1 wherever the plant output
  has content. The misfit is |F - 1| |Y| over the signal band plus the out-of-band content F removes. For BW8 at
  1.6 kHz the first term is below 1e-11 relative up to 300 Hz. The second is nm-level friction content above 1 kHz.
- The cross-field check agrees. Econometrics requires each variable to carry its own observation operator: flows
  (force per interval) averaged, stocks (position) point-sampled. Averaging a stock and fitting a point model gives
  "temporal aggregation bias". That is D-232's block-mean failure on y.
- Closed loop and the residual form do not change the operator per channel. They change which copy of y is a bias
  channel (Q2).
- u must not follow y: filtering the known ZOH input moves it off its exact 4 kHz equivalent (D-087) and removes
  nothing, since u carries no measurement noise.

**Sources.**

| Source | Read | What it says | Verdict |
|-|-|-|-|
| Pintelon, Schoukens, *System Identification: A Frequency Domain Approach*, 2nd ed., Wiley-IEEE 2012 (held, `literature/BLA/`) | in full for these sections | Sect. 13.3.2.2 and Table 13-2, book p512: ZOH setup, single-channel, "NOT allowed to use an anti-alias filter as this would eliminate all the repeated spectral contributions of the ZOH"; BL setup: "anti-alias filters required" on both channels. Sect. 13.3.2.3, p512: control setups may "filter the output before feeding it back". Sect. 13.2.3.1, Eqs. (13-8) to (13-10), pp. 502-503: violation of the ZOH assumption; error small when \|G\| is small outside the band and \|w/w_s\| is small | SUPPORTS: the rule is per setup, and the relaxation conditions hold for a 1/(m s^2) plant at 100 to 300 Hz against 4 kHz |
| Ljung, "Estimation focus in system identification: prefiltering, noise models, and prediction", *38th IEEE CDC*, 1999, pp. 2810-2815, DOI 10.1109/CDC.1999.831359 (held) | in full | Eq. (6): the same L on all channels; Eq. (8): limit fit weighted by \|L\|^2 Phi_u; condition (9): open loop | SUPPORTS "same filter" for measured channels; DOES NOT ADDRESS closed loop or a known ZOH input |
| Christensen, Neri, Parra-Alvarez, "Estimation of continuous-time linear DSGE models from discrete-time measurements", *J. Econometrics*, 2024, DOI 10.1016/j.jeconom.2024.105871 (CREATES RP 2022-12, downloaded) | Sects. 2.3-2.4 and 4.2.2 in full | Eqs. 2.7-2.13: stocks point-sampled, flows integrated, measurement error additive on the stock (2.9). Sect. 4.2.2, p33: the wrong aggregation operator gives temporal aggregation bias | SUPPORTS block-mean u (flow) with point-type y (stock); rules out averaging y alone |
| Christiano, Eichenbaum, "Temporal aggregation and structural inference in macroeconomics", NBER TWP 60, 1986, DOI 10.3386/t0060 (downloaded) | grep level | Theorem 1, p13; Table 2.1, p17: averaged versus point-in-time measures | SUPPORTS the same, second source |
| Ljung, *System Identification: Theory for the User*, 2nd ed., Sects. 13.7, 14.4; Soderstrom and Stoica | not read (not held) | | gap |

## Q2: aliased output noise in closed-loop simulation error: variance or bias?

**Answer, per channel of the residual form** (u_model = u_data + K4 (y_data - y_model), loss on y_data - y_model):

1. **Loss target.**
   - In the CLOE-equivalent form, output noise enters the prediction error only through theta-independent terms
     (Forssell-Ljung Eq. B.4; Karimi-Landau Eq. 5.6), so aliased target noise raises variance and the loss floor,
     not bias.
   - The sampled noise without an anti-alias filter is itself a legitimate discrete-time noise (Pintelon-Schoukens
     Theorem 6.11).
   - Fang and Zhu measured the same split in an LTI example: aliasing raised the parameter spread, and the
     anti-alias-filtered model was biased.
2. **Controller branch.**
   - The branch is a bias channel. The model input must reconstruct a signal uncorrelated with the noise, which
     needs the exact controller (Van den Hof-Schrama Eqs. 33-34).
   - After decimation a leak remains: K4 applied to point-sampled v, minus the block mean of K20 v. It is correlated
     with the target noise. The closest stated result is Forssell-Ljung Remark (5): a correlated disturbance gives
     "arbitrarily bad" estimates.
   - NT-004 measured this leak at 26 to 38 % of the floor with point sampling, and 5 to 6 % with the 1.6 kHz filter
     on the branch copy.
3. **Where the size matters** (Forssell-Ljung Remark (3), Eqs. 105-106): the bias scales with the noise share of the
   input spectrum.
   - Multisine records: negligible (about 3 nm folded against 80 to 1000 nm floors).
   - Route and standstill records: a visible share (7 to 11 nm floors).
   - Point sampling is acceptable only where the floor is dominated by other errors.

**Why the recommendation filters the target as well, against the literature's "keep the target unfiltered":**
- (a) Filtering the target alone fits F G, not G (Ljung Eq. 6 logic; Fang et al. 2017 Eq. 34). But F differs from 1
  only above about 800 Hz. By Ljung Eq. (8) the fit moves only where F is not 1, and the physical parameters (masses,
  inertia, the 106 to 297 Hz absorber) are informed far below that. The bias is bounded by the removed content: 0 to
  2 nm (NT-007 (b)).
  Fang's bias arises when the filter cuts into the band where the plant responds. Their consistency condition, Eq. 34
  (second-hand), is that G_c be insignificant outside the Nyquist band, which a 100 Hz double integrator meets.
- (b) DERIVED, not sourced: in our residual form the target's folded noise is not strictly theta-independent.
  u_data's block mean carries a folded copy of K20's reaction to the same 2 to 10 kHz noise components (the block
  mean passes 0.57 at 2.2 kHz and 0.30 at 3 kHz). The model input and a point-sampled target therefore share them.
  Filtering the target removes its share.
- (c) The folded excess is 28 to 40 % of the noise rms, a variance factor of 1.6 to 2.0, against a deterministic
  change of about 2 nm or less above 1 kHz.
- (d) One y keeps the residual identity exact. The branch-only rule (NT-004a) needed a 0.005 N tolerance on it.
- (e) TR-022 filters branch and target on real data and passed.

The alternative (branch-only, NT-004a) is the literature's cleaner consistency argument. It is the choice to make if
the user weighs an unbiased target above variance.

**Sources.**

| Source | Read | What it says | Verdict |
|-|-|-|-|
| Forssell, Ljung, "Closed-loop identification revisited", *Automatica* 35(7):1215-1241, 1999, DOI 10.1016/S0005-1098(99)00022-9 (held, report version) | Corollary 4, Remarks 1-7, Eq. (20), proof Eqs. (B.4)-(B.7) | direct-method bias ∝ Phi_eu; target noise theta-independent in CLOE; a correlated constructed reference biases | SUPPORTS both channel verdicts; the mapping of the leak is DERIVED |
| Karimi, Landau, *Syst. Control Lett.* 34:159-167, 1998, DOI 10.1016/S0167-6911(97)00137-0 (held) | Eq. (5.6), p164 (grep) | CLOE fit independent of output noise | SUPPORTS (1) |
| Van den Hof, Schrama, *Automatica* 31(12):1751-1770, 1995 (held) | Eqs. (33)-(34), p1761 (grep) | reconstructing r = u + C y needs the exact controller | SUPPORTS (2) |
| Fang, Zhu, "Analysis of over-sampling based identification", *Automatica* 79:101-107, 2017, DOI 10.1016/j.automatica.2017.01.006 (held) | Sect. 4, Eqs. 28, 29, 38, 40; Sect. 5.1, Table 2 | aliasing costs variance; the filtered model was biased | SUPPORTS the split; their setting is the reverse of ours (slow controller, fast output) |
| Fang, Zhu, Hjalmarsson, "On anti-aliasing filtering and over-sampling scheme in system identification", *Comput. Chem. Eng.* 106:572-581, 2017, DOI 10.1016/j.compchemeng.2017.07.010 | SECOND-HAND (the project's 2026-08-27 browser read); re-read failed on 4 routes | Eq. (23): filter-then-decimate identifies sum (H_j/H_0) g_j, inconsistent; Eq. (34): approximately right when G_c is insignificant outside the Nyquist band; abstract: over-sampling anti-aliases in open and closed loop | SUPPORTS "a filter must be in the model or out of the plant's band"; DOES NOT ADDRESS the residual form |
| Pintelon, Schoukens 2012, Theorem 6.11, Fig. 6-6, book pp. 196-197 (held) | in full | noise sampled without an anti-alias filter is exactly a DT noise model | SUPPORTS (1) |

## Q3: low-passing a signal with a 0.3 m offset at nm precision

**Answer.**
- **Required spec.** Criterion (ii) asks for a relative error of about 2e-9 at DC and about 2e-5 over the multisine
  band.
- **Rejected classes.**
  - Equiripple (Chebyshev, elliptic, Parks-McClellan): a ripple of delta_p means a DC or in-band error of delta_p.
  - Windowed FIR at normal lengths: DC exact after normalisation, but in-band ripple stays.
  - Moving average, block mean and CIC (sinc^k): DC exact, but they droop: even centred, 0.991 at 300 Hz and 0.904 at
    1 kHz (Heertjes Eq. 2 form).
  - Halfband: its passband error equals its stopband error (Vaidyanathan Sect. V-A).
- **What passes.** Butterworth used forward-backward: exact unity DC gain, and an error of (f/fc)^16 after
  prewarping. Polyphase or multistage structures are efficiency devices; they do not change this accuracy argument.
- **Edges.** `sosfiltfilt` starts each pass in steady state and pads with an odd extension (scipy source), so a
  constant or a constant velocity at the edge produces no transient. Curvature at the edge does, and Gustafsson 1996
  names offsets and trends as the cause.
- **Filter the error.** Filtering y - r removes the edge issue and matches practice: MA and MSD act on the tracking
  error in lithography. The identity r + F(y - r) = F y + (1 - F) r is plain linearity; no source states it.
- **Caution.** Wigren and Schoukens: post-filtering data for nonlinear model estimation is "not straightforward". That
  concerns filtering both sides of a nonlinear model; here only the measured output is filtered, in a band where the
  plant output is essentially zero.

**Sources.**

| Source | Read | What it says | Verdict |
|-|-|-|-|
| Gustafsson, "Determining the initial states in forward-backward filtering", *IEEE Trans. Signal Process.* 44(4):988-992, 1996, DOI 10.1109/78.492552 (LiU preprint, downloaded) | in full | Sect. 1: forward-backward gives \|G\|^2; Sect. 2: least-squares initial states; Sects. 5-6: offsets and trends cause the edge transients | SUPPORTS the error form and the edge argument |
| Heertjes, "Frequency-domain properties of moving average and moving standard deviation with application to stage (motion) control", *ECC 2024*, pp. 3521-3526, DOI 10.23919/ECC64448.2024.10591183 (TU/e Pure, downloaded) | in full | Def. 2.1, Eq. (1): MA of the tracking error; Eq. (2): sinc with DC gain 1; fs 5 kHz; no anti-alias rule | SUPPORTS "the servo error is the analysis quantity"; DOES NOT ADDRESS anti-aliasing |
| Voorhoeve, PhD thesis, TU/e 2018 (held) | sections | Def. 5.1, Eqs. (5.1)-(5.3): MA and MSD of e = r - z; App. A, PDF p200: "No anti-aliasing filters are used ... for a zero-order hold measurement setup ... the conventional configuration" | SUPPORTS error-based analysis and the ZOH convention of Q1 |
| Vaidyanathan, "Multirate digital filters, filter banks, polyphase networks, and applications: a tutorial", *Proc. IEEE* 78(1):56-93, 1990, DOI 10.1109/5.52200 (downloaded) | sections | Sect. II: decimation filter; Sect. V-A: halfband errors equal | rules out the halfband; DOES NOT ADDRESS precision |
| scipy 1.17.0 source, `scipy/signal/_signaltools.py` (installed) | in full for `decimate`, `cheb1ap`, `filtfilt`, `sosfiltfilt`, `_validate_pad` | Chebyshev I design and forward-backward; steady-state edge initialisation; odd padding | explains the D-232 Chebyshev failure exactly |
| Wigren, Schoukens, "Three free data sets for development and benchmarking in nonlinear system identification", *ECC 2013*, pp. 2933-2938, DOI 10.23919/ECC.2013.6669201 (downloaded) | sections | p2: sample fast; post-filtering and mean removal not straightforward for nonlinear systems | CONTRADICTS naive filtering of both channels; the recommendation filters only the output, out of band |
| Butterworth formula, Oppenheim-Schafer, Crochiere-Rabiner, Hogenauer (CIC) | not read | standard definitions used | gap (textbook, browser route) |

## Q4: does the frequency-domain view make the question moot?

**Answer.**
- **Multisine records: nearly moot.**
  - The records are periodic: a 0.5 Hz grid gives 6 periods of 2 s in 12 s.
  - With a periodic reference, the sample mean and covariance over periods separate the periodic part from all
    noise, folded noise included, and the estimators stay consistent in feedback (Sect. 10.10).
  - Aliasing then only raises the noise variance on the excited lines, by at most about 3 dB (1.28 to 1.40 x rms)
    on an SNR of at least 53 dB in the band (EXCITATION-VALIDATION).
  - This matches D-232: about 3 nm folded against 80 to 1000 nm floors.
- **Routes: not moot.**
  - For arbitrary references the input is correlated with the noise. Consistency then needs a correct noise model
    (Sect. 10.10), or a known reference with nonparametric noise models (Sect. 13.5.4).
  - Our time-domain trainer uses neither. It does not average over periods, and every bin enters the loss, so the
    frequency-domain protection does not transfer to the trainer.

**Sources.**

| Source | Read | What it says | Verdict |
|-|-|-|-|
| Pintelon, Schoukens 2012, Sect. 10.10, book p411 (held) | in full | periodic reference: SML, SGTLS, SBTLS and SSUB strongly consistent in feedback; arbitrary reference: input correlated with the process noise | SUPPORTS moot on multisine lines, not on routes |
| Same book, Sect. 13.5.4, p515 (held) | in full | with nonparametric noise models, identification in feedback is as easy as generalized output error; arbitrary excitation needs a known reference | SUPPORTS; DOES NOT ADDRESS time-domain simulation error |

## Q5: guidance from neural state-space, SUBNET and benchmark practice

**Answer.**
- **SUBNET, augmentation and Forgione-Piga lines: none found.**
  - The held Beintema 2023 and CT-SUBNET papers were grepped for decimat, alias, resampl, downsampl, Nyquist and
    filter: no guidance.
  - Hoekstra et al. 2025 add white noise at the training rate, which avoids the question.
  - Forgione and Piga decimate EMPS by 5 through point sampling with no anti-alias filter (their
    `EMPS_preprocess.py`).
- **Benchmarks.**
  - Wiener-Hammerstein: hardware anti-alias filters on both channels (BL setup).
  - EMPS and the identification line built on inverse dynamic models (IDIM): zero-phase Butterworth on position,
    then parallel decimation of every regression column (Chebyshev), which works because it is applied to both
    sides of a linear-in-parameters regression.
  - Nano-drone benchmark: zero-phase Butterworth 4 after resampling, as noise filtering.
- So zero-phase Butterworth on positions is the common conditioning. Nothing addresses a noisy closed-loop target
  for simulation-error training. The gap is provisional (Ljung et al. 2020 and Schoukens-Noel 2017 not read).

**Sources.**

| Source | Read | What it says | Verdict |
|-|-|-|-|
| Beintema, Schoukens, Toth, "Deep subspace encoders", *Automatica* 2023; CT-SUBNET, ICLR 2023 (held) | grep in full | no decimation or anti-alias content; EMPS used at its native 1 ms | DOES NOT ADDRESS |
| Hoekstra, Verhoek, Toth, Schoukens, *Eur. J. Control* 86:101304, 2025, DOI 10.1016/j.ejcon.2025.101304 (held) | Sect. 4.1, p4 | noise added at Ts = 0.02 s, open loop | DOES NOT ADDRESS |
| Forgione, Piga, *Eur. J. Control* 2021, arXiv:2006.02915 | sections plus repo code | EMPS point-sampled by 5, float32 | DOES NOT ADDRESS (practice without justification) |
| Janot, Gautier, Brunot, "Data set and reference models of EMPS", 2019 Workshop on Nonlinear System Identification Benchmarks (downloaded) | in full | Sect. 3.1: Butterworth forward-backward, cutoff >= 5 omega_dyn; Sect. 3.2, Eq. (9): parallel decimation | SUPPORTS zero-phase Butterworth on position; its Chebyshev step is safe only on both sides of a regression |
| Gautier, Janot, Vandanjon, *IEEE TCST* 2013 (held) | sections | p4: filtfilt Butterworth on q; p5: parallel decimation at 0.8 fm / (2 nd); Table 5: no anti-aliasing degrades IDIM | SUPPORTS the same, second source |
| Schoukens, Suykens, Ljung, "Wiener-Hammerstein benchmark", *SYSID 2009* (downloaded) | in full | Sect. 3: 51.2 kHz, hardware anti-alias filter on input and output | SUPPORTS "same filter on both channels" for BL |

## Access status

TU/e browser access: NOT USED. The preflight was not run; the parent did not need it before the agents were cut off.
Items that would need it are listed below as `needs-browser-route`, not as unreachable.

## Evidence quality

- **Read in full for the cited sections:**
  - Pintelon-Schoukens 2012 (13.2.3.1, 13.3.2, 10.10, 13.5.4, 6.7.3, Ex. 13.6)
  - Forssell-Ljung 1999, Ljung 1999 CDC, Fang-Zhu 2017
  - Gustafsson 1996, Heertjes 2024, Janot 2019, WH 2009, Christensen et al. 2024
  - the scipy source
- **Grep or section level:** Karimi-Landau 1998, Van den Hof-Schrama 1995, Voorhoeve 2018, Vaidyanathan 1990,
  Wigren-Schoukens 2013, Christiano-Eichenbaum 1986, Gautier 2013, Beintema 2023, CT-SUBNET, Hoekstra 2025,
  Forgione-Piga 2021.
- **Second-hand:** Fang, Zhu, Hjalmarsson 2017 (from the project's earlier browser read).
- **Not read:** Ljung textbook Ch. 13-14; Soderstrom-Stoica; DSP textbooks; Butler 2011.
- **DERIVED in this report, not sourced:**
  - the folded-copy argument (b) in Q2
  - the FIR mechanism (Correction 2)
  - the expected criterion values
  - the r + F(y - r) identity

## Research Log

- **Frame.** Q1 and Q2 were delegated to agent A, Q3 and Q5 to agent B, and Q4 was done inline. Seeds: Forssell-Ljung
  1999, Ljung 1999, Pintelon-Schoukens 2012, Fang-Zhu-Hjalmarsson 2017, Fang-Zhu 2017, Sano-Tsuji 1993, Gautier 2013,
  Beintema 2023, Hoekstra 2025. Vocabularies: system identification, DSP and multirate, precision motion,
  econometrics (temporal aggregation, stock and flow), ML.
- **Local holdings mined first.** The noise-training study (SQ3, NT-003, NT-004a, NT-007) and `telica-real/rate.py`
  (TR-022) already held the recommended filter class with measurements. The handoff did not cite them.
- **Queries run.**
  - Parent: pypdf greps of the held Pintelon-Schoukens book (alias, decimat, Table 13-2, feedback); the scipy source;
    repo greps.
  - Agent A: OpenAlex 7 answered, 1 refused ("Search temporarily unavailable"); Crossref 8; arXiv 2 (valid zeros);
    Scholar 2 (sole source of Ghysels-Miller); MCP Unpaywall, CORE, BASE, OpenAIRE, fallback download (all failed for
    Fang 2017).
  - Agent B: Crossref 9; OpenAlex 5 (1 on-target, Heertjes 2024, via a quoted-phrase filter; 1 refused with 503
    "Anonymous search is paused"); arXiv 2 author-pair queries (found Gonzalez 2024 and the nano-drone benchmark);
    TU/e Pure OAI 3; nonlinearbenchmark.org 5 pages; GitHub raw 2.
  - dblp 0 in both agents.
- **What worked.**
  - Held PDFs first: the decisive Table 13-2, Sect. 10.10 and Fang-Zhu Table 2 were all on disk.
  - Reading the installed library source to explain a numerical failure exactly.
  - OpenAlex `title_and_abstract.search` with a quoted phrase ("moving standard deviation" wafer; "stocks and flows").
  - Benchmark code (IDIM script, EMPS preprocessing), which states filter parameters the papers omit.
- **What failed.**
  - Every Elsevier route for Fang et al. 2017 (403, a retired manuscript host); DiVA search behind a bot wall.
  - OpenAlex anonymous search paused mid-run (the `error` key guard caught it).
  - Two bundled agents took 33 minutes and were cut off before the textbooks.
- **Dead ends.** Oomen 2018 and dynoNet (no anti-alias content); CORE record 197887636 (no file).
- **Coverage gaps (provisional negatives).**
  - Ljung textbook Sects. 13.7, 14.4; DSP textbooks for the Butterworth and window-ripple formulas.
  - Butler 2011; Ljung et al. 2020; Schoukens-Noel 2017.
  - No source on averaging a noisy stock versus skip-sampling it (1 Scholar, 1 arXiv, 1 OpenAlex query).
  - No source on the residual-form leak.
- **Suggested skill fixes.**
  - Done this session, user-approved: literature research may use up to about 5 agents, one sub-question each, with
    a deadline and an access-hunt cap (SKILL.md "Subagent fan-out"; memory `feedback_no_agent_fanout`).
  - Proposed, not applied:
    - add OpenAlex's "Search temporarily unavailable" and 503 "Anonymous search is paused" bodies to the
      failure-mode table;
    - route 3b: RePEc and NBER PDFs from `locations[]` fetch first try, and nonlinearbenchmark.org PDFs download via
      `drive.google.com/uc?export=download&id=`;
    - DiVA: FULLTEXT01 can be PostScript (convert with ps2pdf), and its search is now bot-walled;
    - for numerical-implementation questions, read the installed library source before any paper.

## needs-browser-route (ranked)

1. Fang, Zhu, Hjalmarsson 2017, DOI 10.1016/j.compchemeng.2017.07.010: re-verify Sects. 3-4, Eq. (34), any closed-loop
   or output-only statement.
2. Ljung, *System Identification: Theory for the User*, 2nd ed., Sects. 13.7 (presampling filters), 14.4 (prefiltering
   and decimation), 13.4-13.5 (closed loop): book.
3. Butler, IEEE CSM 31(5):28-47, 2011, DOI 10.1109/MCS.2011.941882: servo-error logging practice.
4. Crochiere, Rabiner, Proc. IEEE 69(3):300-331, 1981, DOI 10.1109/PROC.1981.11969: multistage passband allocation.
5. Hogenauer, IEEE TASSP 29(2):155-162, 1981, DOI 10.1109/TASSP.1981.1163535: CIC droop.
6. Ghysels, Miller, J. Time Series Anal. 36(6), 2015, DOI 10.1111/jtsa.12129: mixed skip and average sampling.
