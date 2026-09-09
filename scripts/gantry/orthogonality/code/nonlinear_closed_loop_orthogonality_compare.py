"""Cross-engine comparison of the two nonlinear closed-loop orthogonality runs.

Reads only the two result JSONs. It does not re-derive anything, so it adds no
verification of its own: it reports whether two independent transcriptions of the
same model equations reached the same booleans and comparable numbers.

Run:  conda run -n GraduationProject python \
        scripts/gantry/orthogonality/code/nonlinear_closed_loop_orthogonality_compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "nonlinear_closed_loop_orthogonality"

# Numeric quantities both engines compute from their own transcription. Tolerances are
# per quantity: exact structural integers must match, floating results must agree to
# the stated relative tolerance.
SHARED_NUMBERS = {
    "horizon_scored_steps_per_window": 0.0,
    "state_transitions_per_window": 0.0,
    "scored_rows": 0.0,
    "rank_L_S0_unprofiled": 0.0,
    "rank_encoder_D0": 0.0,
    "rank_Sbar_after_profiling": 0.0,
    "constraint_nullspace_dim_at_eta_zero": 0.0,
    "constraint_jacobian_rank_at_eta_zero": 0.0,
    "long_horizon": 0.0,
    "long_windows": 0.0,
    "long_rows": 0.0,
    "projected_residual_norm_at_reference_K": 1e-12,
    "projected_residual_dnorm_dK_at_reference": 1e-12,
    "linearisation_spectral_radius_at_origin": 1e-12,
    "cond_L_S0": 1e-10,
    "cond_Sbar": 1e-10,
    "penalty_gradient_analytic": 1e-12,
}


def rel(a, b) -> float:
    if isinstance(a, list) or isinstance(b, list):
        a, b = list(a), list(b)
        if len(a) != len(b):
            return float("inf")
        return max((rel(x, y) for x, y in zip(a, b)), default=0.0)
    a, b = float(a), float(b)
    return abs(a - b) / max(1.0, abs(a), abs(b))


def main() -> None:
    ml = json.loads((OUT / "nonlinear_closed_loop_orthogonality_matlab.json").read_text())
    sy = json.loads((OUT / "nonlinear_closed_loop_orthogonality_sympy.json").read_text())

    mc, sc = ml["checks"], sy["checks"]
    shared = sorted(set(mc) & set(sc))
    agree = [k for k in shared if bool(mc[k]) == bool(sc[k])]
    disagree = [k for k in shared if bool(mc[k]) != bool(sc[k])]
    both_true = [k for k in agree if mc[k]]
    only_ml = sorted(set(mc) - set(sc))
    only_sy = sorted(set(sc) - set(mc))

    num_rows, num_bad = [], []
    for key, tol in SHARED_NUMBERS.items():
        if key in ml["numbers"] and key in sy["numbers"]:
            d = rel(ml["numbers"][key], sy["numbers"][key])
            num_rows.append((key, d, tol))
            if d > tol:
                num_bad.append(key)

    report = {
        "category": "cross-engine agreement on structural verification examples only",
        "matlab_engine": ml["engine"],
        "sympy_engine": sy["engine"],
        "n_shared_checks": len(shared),
        "n_agree": len(agree),
        "n_agree_and_both_true": len(both_true),
        "disagreements": disagree,
        "checks_only_in_matlab": only_ml,
        "checks_only_in_sympy": only_sy,
        "matlab_failed": ml.get("failed", []),
        "sympy_failed": sy.get("failed", []),
        "matlab_undecided_identities": ml.get("undecided_identities", []),
        "sympy_undecided_identities": sy.get("undecided_identities", []),
        "numeric_relative_differences": {k: d for k, d, _ in num_rows},
        "numeric_out_of_tolerance": num_bad,
        "matlab_timings_s": ml.get("timings_s", {}),
        "sympy_timings_s": sy.get("timings_s", {}),
        "caveat": (
            "Agreement checks a finite model and a finite set of identities transcribed "
            "twice from the same written equations. It is NOT verification of the gantry "
            "first-principles model, and two engines reading one transcription would not "
            "be the independent check algebra-tooling.md rule 1 asks for."),
    }
    (OUT / "nonlinear_closed_loop_orthogonality_compare.json").write_text(
        json.dumps(report, indent=2) + "\n")

    print(f"MATLAB : {ml['engine']}   {ml['n_checks']} checks, {len(ml['failed'])} failed")
    print(f"SymPy  : {sy['engine']}   {sy['n_checks']} checks, {len(sy['failed'])} failed")
    print(f"shared checks      : {len(shared)}")
    print(f"agree              : {len(agree)}  (both true: {len(both_true)})")
    print(f"disagree           : {disagree if disagree else 'none'}")
    print(f"only in MATLAB     : {only_ml if only_ml else 'none'}")
    print(f"only in SymPy      : {only_sy if only_sy else 'none'}")
    print("\nshared numeric quantities (relative difference vs tolerance):")
    for k, d, tol in sorted(num_rows, key=lambda r: -r[1]):
        print(f"  {k:<48} {d:.3e}   tol {tol:.0e}   {'OK' if d <= tol else 'OUT OF TOLERANCE'}")
    print(f"\nnumeric out of tolerance: {num_bad if num_bad else 'none'}")
    print(f"undecided identities: MATLAB {ml.get('undecided_identities', [])} "
          f"| SymPy {sy.get('undecided_identities', [])}")


if __name__ == "__main__":
    main()
