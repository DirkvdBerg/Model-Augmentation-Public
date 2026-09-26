# Independent review of the multisine-band evidence

Scope: nominal baseline and 10 % detuned baseline against the Coulomb + MSD truth. The frictionless model is treated only as the BLA pipeline gate. This review judges the present evidence in `EXCITATION-VALIDATION.md`; it does not use the Telica noise floor to choose the band.

## Q1 (nominal): are the closed-loop FRFs a sound basis?

- Verdict: sound with named fixes; the Y = 0 result supports a provisional band, but the present FRF set is not sufficient to freeze the training band.
- Argument: the relevant object is the closed-loop BLA `PS_truth` of the nonlinear Coulomb + MSD system, not an MSD-only FRF; subtracting `PS_nominal` gives the local closed-loop prediction mismatch that the excitation must reveal.
- Argument: the measurement is appropriately MIMO and in stage coordinates, with three plant-input forces and three servo-error outputs; figs. 6 and 7 show that a Y-only entry would miss the 50 to 100 Hz rail mismatch and the cross-entry features near 204 to 224 Hz.
- Argument: the robust periodic BLA has a credible numerical gate; friction-off agrees with the analytic loop to median `2e-5`, the responses are periodic to `2e-12`, and the production-level nonlinear distortion is 2 to 5 % of the BLA.
- Argument: the production rms is the relevant level; the one-sixth-level result moves the closed-loop peak from about 262 to 227 Hz and leaves the rails stuck 83 to 86 %, confirming that this Type-I BLA is level and spectrum dependent.
- Argument: the broadband measurement from 1 Hz to 1 kHz and the 1.5 Hz per-column grid resolve the observed 50 to 100 Hz friction mismatch, the 150 Hz anti-resonance, and the 204 to 224 Hz coupling notch with ample line density.
- Named fix: repeat the truth BLA at Y = -0.30, -0.15, +0.15, and +0.30 m with K1 designed at Y = 0 and frozen; combine these with the valid Y = 0 result using the largest material mismatch over all five training points.
- Named fix: after selecting the candidate band, measure at least one candidate-spectrum BLA at production rms; a nonlinear BLA obtained with the 1 Hz to 1 kHz spectrum is not automatically the BLA produced by a 50 to 230 Hz spectrum.
- Named fix: treat the standstill BLA as evidence for the absorber and local standstill friction response only; moving Coulomb friction requires velocity, reversal, and dwell coverage and cannot be certified from these FRFs.
- Would change it: K1 results showing that every FRF-difference feature remains inside the same frequency intervals and above the BLA uncertainty at all five Y points would make the basis sound for final band selection, subject to the candidate-spectrum check.
- Would change it: a candidate-spectrum BLA that shifts a material mismatch peak outside 50 to 230 Hz, or changes the friction regime materially, would require a different band or power allocation (?).

## Q1 (10 % detuned): are the closed-loop FRFs a sound basis?

- Verdict: sound with named fixes; the construction is technically appropriate, but the present operating-point set and controller do not yet match the training design.
- Argument: `E_det = PS_truth - PS_detuned` is the correct total mismatch for the detuned starting model, while `PS_nominal - PS_detuned` isolates the part caused by the known 10 % parameter perturbation.
- Argument: fig. 10 shows the detuning part peaking at about 60 to 100 Hz in all nine stage-coordinate entries; this is a MIMO result and is not inferred from one selected channel.
- Argument: the same validated truth BLA, production level, resolution, and closed-loop controller conditions that support the nominal comparison support this comparison at Y = 0; the detuned baseline FRF itself is analytic and introduces no extra BLA uncertainty.
- Named fix: run the four missing K1 BLAs and recompute `E_det` and the detuning part over the five training Y points; the existing Y = +/-0.30 curves used controllers redesigned at those positions and are not evidence for the planned K1 experiment.
- Named fix: retain BLA uncertainty in the decision; only mismatch features that are resolved relative to the measured BLA standard deviation should drive a band edge.
- Named fix: repeat the final comparison with the candidate excitation spectrum because the truth side of `E_det` is amplitude and spectrum dependent.
- Would change it: if the K1 sweep moves the 60 to 100 Hz maximum outside the proposed band, or makes it unresolved relative to BLA uncertainty, the current lower-edge argument fails.
- Would change it: if a candidate-spectrum BLA shows that 50 to 140 Hz no longer carries a material detuned-to-truth difference, retaining the 140 Hz lower edge becomes defensible (?).

## Q2 (nominal): do the references plus multisine cover the differences?

- Verdict: not sound as a complete nominal-coverage claim; the absorber conclusion is sound, but the Coulomb-friction conclusion is outside the validated range of the method.
- Argument: for one linear loop with the same controller, a reference acts as the equivalent plant-input injection `K r`, so the predicted mismatch `E K r` and multisine mismatch `E f` are the correct design-stage quantities.
- Argument: the J5 gate supports this route in 50 to 140 Hz and for the Y output in 140 to 230 Hz; there the predicted-to-simulated band-rms ratio is 0.9 to 1.8, and the Y absorber mismatch from the multisine is about 33,000 nm against only 40 to 90 nm from the references.
- Argument: the route fails where moving friction changes regime; it overpredicts by 3.9 to 8.9 in 20 to 50 Hz and underpredicts X by factors of about 3 to 7 in 140 to 230 Hz, with still larger failure above 230 Hz.
- Argument: fig. 10 therefore establishes a credible 50 to 140 Hz standstill-BLA gap, but it does not establish frequency coverage of Coulomb friction during motion; that is primarily coverage of sliding speeds, both directions, reversals, and dwells.
- Argument: the 40 dB mask is not a pass criterion; it is binary, normalized to each source's own maximum, insensitive to absolute delivered force and BLA uncertainty, and not linked to estimation precision or induced mismatch. Reporting 20 dB as a sensitivity case does not validate 40 dB.
- Argument: the present calculation uses current-record references and controller evaluations at their recorded operating points, whereas the planned data use different levels and jerk times and one frozen K1; class-averaging also does not represent the planned record mixture or a worst-case record.
- Fix: replace the binary mask by the continuous, data-derived excited-mismatch spectrum `E Phi_w E*`, with `w = K1 r + f`; report its cumulative energy by output and its reference and multisine contributions for every planned record class and training Y point.
- Fix: require every resolved FRF-difference peak, defined against the measured BLA standard deviation, to be either inside the multisine support or supported by a J5-style simulated gate for the relevant reference class; do not use an arbitrary relative-ASD cutoff as the decision rule.
- Fix: audit Coulomb learning separately in the velocity domain using the planned motion records; report visited sliding speeds in both signs, reversal counts, dwell time, and the fraction of samples in stick, breakaway, and sliding regimes.
- Would change it: a moving-record gate that predicts the X mismatch within the preregistered factor-two tolerance across the friction-dominated bands would justify using the FRF route for friction coverage.
- Would change it: planned-reference spectra under K1 that produce sufficient continuous mismatch energy at every resolved nominal peak could remove the need to extend the multisine below 140 Hz (?).

## Q2 (10 % detuned): do the references plus multisine cover the differences?

- Verdict: sound with named fixes as a qualitative gap diagnosis, not yet as a quantitative coverage certificate.
- Argument: the exact `K r` equivalence is strongest for the linear detuning part, and the J5 gate passes in the decisive 50 to 140 Hz interval; fig. 10 places the detuning maximum at 60 to 100 Hz while the current multisine begins at 140 Hz.
- Argument: the motion references reach only about 50 Hz on X and 70 Hz on Y under the stated 40 dB display rule, leaving much of the detuning maximum weakly excited; the combined binary covered share is only 29 to 43 % for the isolated detuning part.
- Argument: J4 does not refute this gap; noiseless separability of all ten parameter combinations means nonzero information exists, not that the experiment gives balanced precision or adequate signal against the unmodelled absorber and friction mismatch.
- Argument: J4 usefully shows why 140 to 230 Hz must remain covered; `J_eff`, `cb_sum`, and `d` receive 82 to 98 % of their information from the current multisine, while the point-to-point references carry most information for several low-frequency combinations.
- Named fix: recompute the continuous `E Phi_w E*` measure with K1, the planned references, their actual record weights, all five training Y points, and each candidate band at the same total rms.
- Named fix: show sensitivity to lower edges near the reference roll-off, for example 40, 50, 70, and 100 Hz, and select the edge from the continuous cumulative curves rather than from the 40 dB mask.
- Named fix: rerun the parameter-information calculation with the selected band; widening at fixed rms costs about 3 dB per line, so the 82 to 98 % information result for 140 to 230 Hz cannot simply be transferred to 50 to 230 Hz.
- Would change it: if the planned K1 references supply material excited-mismatch energy through 60 to 100 Hz for every relevant input, the current 140 Hz lower edge could be retained.
- Would change it: if fixed-power 50 to 230 Hz causes unacceptable loss of information for `J_eff`, `cb_sum`, or `d`, use a nonuniform power allocation or two subbands rather than returning automatically to 140 to 230 Hz (?).

## Recommendation

- Band: use 50 to 230 Hz as the provisional common candidate for both data sets; do not yet freeze it as `B_tr`.
- Reason: 140 to 230 Hz covers the 150 Hz absorber anti-resonance and the 204 to 224 Hz coupling notches and is indispensable for `J_eff`, `cb_sum`, and `d`, but it omits the resolved 50 to 100 Hz nominal rail mismatch and the 60 to 100 Hz detuning maximum that the present references do not reliably cover.
- Reason: 50 Hz is a defensible provisional lower edge because it is near the observed X-reference roll-off and below the full detuning maximum; it is not justified by the 40 dB number alone.
- Condition: preserve production rms and check the resulting force limits, friction regime, BLA, and parameter information; if the 3 dB per-line reduction harms the upper-band information, allocate power nonuniformly between 50 to 140 and 140 to 230 Hz (?).
- Upper edge: keep 230 Hz provisionally; the closed-loop peak near 262 Hz is not itself proof that training must excite it, but the proposed 230 to 300 Hz test record is needed to verify that the learned plant predicts that untrained closed-loop feature.

## Are the four planned BLA runs the right next computation?

- Verdict: yes, they are the necessary next computation because they repair the direct mismatch between the current evidence and the planned Y range and frozen K1 controller.
- Required use: combine the four runs with Y = 0, recompute figs. 6, 7, and 10 using the largest resolved mismatch over the five points, and discard the old +/-0.30 controller-redesigned curves from the band decision.
- Limitation: the four broadband runs are not sufficient to close the decision because they do not test the spectrum-dependent BLA under 50 to 230 Hz and do not update coverage to the planned K1 references.
- Follow-on: after the four runs, evaluate the planned references and the 50 to 230 Hz candidate with the continuous excited-mismatch and parameter-information measures; then run a candidate-spectrum truth BLA at the worst-case Y point or points identified by that sweep (?).
