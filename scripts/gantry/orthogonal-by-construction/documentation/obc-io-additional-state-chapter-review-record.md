# Chapter review and verification record

Date: 2026-09-15. Companion repository record for `obc-io-additional-state-chapter.tex`; not part of the thesis chapter body.

## Verification provenance

- Script: `../code/io_additional_state_equivalence.m`.
- Recorded output: `../code/logs/io_additional_state_equivalence.log`.
- Run: 2026-09-14, MATLAB R2025a Update 1, Symbolic Math Toolbox 25.1.
- The log records 18 passed checks. The surrogate has scalar output, two output delays, one input delay, two learned states, nonlinear polynomial terms and shared learned parameters.
- Script and log were inspected during review; MATLAB was not rerun for these editorial revisions. General claims rely on the chapter's proofs, not an assertion that finite surrogate checks verify all architectures or statistical theorems.
- The revised LaTeX source is compiled separately; successful typesetting is not mathematical verification.

## Covariance reasoning: preserve the distinction

Source: Györök et al., arXiv:2511.01321v3, 9 January 2026, Section 4.4, Theorem 18, Eqs. (38)–(40): https://arxiv.org/html/2511.01321v3#S4.SS4 . The chapter bibliography explicitly fixes this version for locators.

With fixed reference basis and consistently differentiated projection, the exact algebra gives

\[
\sum_i \phi_i^\top J_i=0.
\]

It need not give zero individual summands. It does give a zero cross block in the unweighted summed Jacobian Gram matrix on the same evaluations. The covariance expression also contains a noise-weighted factor whose cross block is

\[
\sum_i\phi_i^\top\Sigma J_i.
\]

A diagonal covariance with unequal output-channel variances does not generally make this weighted cross block zero. Even zero pointwise unweighted products are insufficient. For example, at one two-output evaluation choose

\[
\phi=\begin{bmatrix}1\\1\end{bmatrix},\qquad
J=\begin{bmatrix}1\\-1\end{bmatrix},\qquad
\Sigma=\operatorname{diag}(1,2).
\]

Then

\[
\phi^\top J=0,\qquad\phi^\top\Sigma J=-1.
\]

This elementary calculation is an algebraic illustration, not a newly run numerical experiment or a complete statistical counterexample to every conclusion of Theorem 18. Scalar-output noise or covariance proportional to the identity avoids this particular weighting issue; other justified channelwise separation conditions can also suffice. A theorem transfer to the three-output gantry needs a separate audit of noise weighting, estimator assumptions and identifiable Jacobian blocks. Do not redesign the approved Euclidean state-equation orthogonality solely to claim an unestablished covariance guarantee.

## Editorial decisions

The projection reduction is described as a compatibility result: it introduces no new projection constraint while accommodating additional learned memory. Empirical utility is not proved. Explicit regressor ordering replaces the erroneous same-order assertion. References may be drawn from trajectories, but the identity belongs to the construction evaluations. Stable-by-design parameterization is one possible stability route, not a necessity. Repository paths are retained here for traceability rather than placed in the thesis body.
