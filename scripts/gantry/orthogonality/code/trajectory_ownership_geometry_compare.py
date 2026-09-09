"""Compare the two engines' results for the revised-ownership reference cases.

Reads only the two result JSONs and adds no verification of its own, exactly as the 2026-09-07
comparison script does. The two source scripts transcribe the model equations independently and
neither reads anything the other produces; this file is the place where that independence is
turned into a number.

Run (after both engines):
    conda run -n GraduationProject python \
        scripts/gantry/orthogonality/code/trajectory_ownership_geometry_compare.py
"""
__project_origin__ = "added"

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'trajectory_ownership_geometry')

# The two engines name their checks by their own section numbering. This map is the ONLY place
# the correspondence is asserted, and it is deliberately explicit rather than derived from a
# string rule: a silent mismatch here would report agreement between two different statements.
PAIRS = {
    'M1.recurrence_equals_direct[lambda]': 'A1.recurrence_equals_direct[lambda]',
    'M1.recurrence_equals_direct[eta]': 'A1.recurrence_equals_direct[eta]',
    'M1.recurrence_equals_direct[xi_p]': 'A1.recurrence_equals_direct[xi_p]',
    'M1.recurrence_equals_direct[lambda,off]': 'A1.recurrence_equals_direct[lambda,off]',
    'M1.negctrl_controller_path_detected': 'A2.negctrl_controller_path_detected',
    'M2.off_independent_of[Kk]': 'A9.off_independent_of[Kk]',
    'M2.off_independent_of[Ff]': 'A9.off_independent_of[Ff]',
    'M2.off_independent_of[Gg]': 'A9.off_independent_of[Gg]',
    'M2.off_independent_of[Pp]': 'A9.off_independent_of[Pp]',
    'M2.off_independent_of[ca]': 'A9.off_independent_of[ca]',
    'M2.clamped_independent_of_latent_init': 'A3.latent_init_absent_from_clamped',
    'M2.clamped_latent_state_null': 'A3.clamped_latent_state_is_null_at_every_step',
    'M2.full_equals_off_at_zero_gains': 'A9.off_equals_full_at_zero_augmentation',
    'M2.clamped_equals_off_at_zero_direct': 'A9.clamped_equals_off_at_zero_direct_route',
    'M2.full_equals_clamped_at_zero_readout': 'A9.full_equals_clamped_at_zero_readout',
    'M3.E_equals_Gamma_J': 'A4.E_equals_Gamma_J',
    'M3.xip_only_through_initial_state': 'A4.xip_only_through_initial_state',
    'M4.latent_init_chain_rule': 'A3.latent_init_chain_rule',
    'M4.latent_init_reaches_output': 'A3.latent_init_reaches_output',
    'M5.structural_null_direction': 'A7.structural_null_direction',
    'M6.penalty_gradient_reaches_latent_init': 'A10.penalty_gradient_reaches_latent_init',
    'M6.penalty_has_nonzero_physical_derivative':
        'A10.penalty_has_nonzero_physical_derivative',
    'M6.mixed_derivative_witness': 'A10.mixed_derivative_witness',
}


def main():
    with open(os.path.join(D, 'trajectory_ownership_geometry_matlab.json')) as f:
        ml = json.load(f)
    with open(os.path.join(D, 'trajectory_ownership_geometry_sympy.json')) as f:
        sy = json.load(f)
    mi = {c['name']: c for c in ml['checks']}
    si = {c['name']: c for c in sy['checks']}

    rows, disagree, missing = [], [], []
    for m, s in PAIRS.items():
        if m not in mi or s not in si:
            missing.append((m, s))
            continue
        a, b = bool(mi[m]['ok']), bool(si[s]['ok'])
        rows.append(dict(matlab=m, sympy=s, matlab_ok=a, sympy_ok=b, agree=a == b,
                         both_true=a and b))
        if a != b:
            disagree.append((m, s, a, b))

    out = dict(
        engines=dict(matlab=ml.get('version'), sympy=sy.get('sympy_version')),
        matlab_checks=len(ml['checks']), sympy_checks=len(sy['checks']),
        shared_checks=len(rows),
        shared_in_agreement=sum(1 for r in rows if r['agree']),
        shared_true_in_both=sum(1 for r in rows if r['both_true']),
        disagreements=disagree, unmatched=missing,
        undecided=dict(matlab=ml.get('undecided'), sympy=sy.get('undecided')),
        seconds=dict(matlab=ml.get('seconds'), sympy=sy.get('seconds')),
        rows=rows,
        note=('The comparison covers the identities both engines transcribe. Checks unique to '
              'one engine (the SymPy compressed-construction, chunking, robustness and '
              'finite-difference numerics) are not comparable and are deliberately not counted '
              'as agreement.'))
    print(f"MATLAB {out['matlab_checks']} checks, SymPy {out['sympy_checks']} checks, "
          f"{out['shared_checks']} shared")
    print(f"shared in agreement : {out['shared_in_agreement']} of {out['shared_checks']}")
    print(f"shared true in both : {out['shared_true_in_both']}")
    print(f"disagreements       : {len(disagree)}")
    print(f"undecided           : matlab {out['undecided']['matlab']}, "
          f"sympy {out['undecided']['sympy']}")
    print(f"in-engine seconds   : matlab {out['seconds']['matlab']:.1f}, "
          f"sympy {out['seconds']['sympy']:.1f}")
    if missing:
        print(f'UNMATCHED NAMES: {missing}')
    p = os.path.join(D, 'trajectory_ownership_geometry_compare.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote ' + p)
    return 1 if (disagree or missing) else 0


if __name__ == '__main__':
    raise SystemExit(main())
