# Run 5 (v5, Section IV) delivery report, saved by the evaluator

Structure: roadmap; IV-A negation; IV-B protected space on reference tuples (new label sec:obc_space); IV-C corrected augmented model and training; IV-D scope and recovery condition. About 1,790 words; every original display kept, two short derivation displays added. Evaluator compile: IV on p. 7 to 9, as original.

Open points
1. gyorok2026obc (arXiv 2511.01321v3) missing from refs.bib; gyorok2025l4dc is a placeholder (verified metadata in a DISABLED block).
2. IV-B: offset excluded because it multiplies no estimated parameter (derived); D-190 records the supervisors' condition as non-overlap with the baseline output. Confirm with supervisors.
3. IV-B: tuples set added states to zero only because they are not measured; no decision gives a reason over e.g. encoder estimates.
4. IV-C: per-epoch refresh reason ("range changes little with the expansion point") rests on D-190 measurements on pre-thesis data.
5. IV-D: "acceleration sensitivity generically has rank three" measured only on an earlier absorber dataset.
6. IV-D: stated reason that Thms 16 and 18 do not transfer (proofs use scored-output orthogonality) differs from D-190's reason for Thm 18.
7. IV-A: "log/tanh coordinate map has a nonsingular Jacobian, so the range is the same" is the agent's derivation from D-229.
8. IV-D: recovery assumptions are a state-level adaptation of Gyorok's Assumption 6.
9. IV-D: confirm the second overlap measure is the "output-level overlap" of RESULTS-DESIGN R4.
10. Notation: Delta* clashes with the Delta block of Section II; u follows original Section III (actuator force), not Section II (generalised force); Gyorok's x_k is an I/O history inside eq:obc_gyorok.
11. Planned geometric projection figure not created.
12. Still open from the original todos: is one-step state orthogonality alone the intended contribution; is the orthogonal multisine design future work.
13. \kw{} markup not carried over; Phase 2 must re-mark.

Moved: to Setup tuple and record counts, rank tolerance, refresh setting, anti-alias filtering (D-232), arm list (Setup still lists stale affine-OBC); Discussion pointer now IV-D; left out the P-storage detail, hand-written RK4 forward sensitivity (appendix candidate), rank-loss retention policy, D-numbers, the removed affine arm.
