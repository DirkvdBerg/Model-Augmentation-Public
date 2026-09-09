"""Independent SymPy transcription of the nonlinear closed-loop orthogonality checks.

Category: STRUCTURAL VERIFICATION EXAMPLES on a generic model. NOT a transcription of
the gantry first-principles model, NOT gantry measurement, NOT evidence of physical
parameter recovery.

algebra-tooling.md rule 1 requires two engines reading two independent sources. This
file transcribes the model equations of the accompanying report from scratch. It does
NOT read, import or parse anything produced by nonlinear_closed_loop_orthogonality.m.
Agreement between the two therefore checks a finite model and a finite set of
identities; it is not automatic verification of the actual gantry equations.

Model (production timing: output computed BEFORE the state transition)
    y_k    = sig*x_k
    e_k    = ydata_k - y_k
    u_k    = udata_k + Cc*xc_k + Dc*e_k
    x_k+1  = kap*x_k + nu*x_k**2 + u_k + (Kk*a_k + Pp*u_k)
    a_k+1  = Ff*x_k + Gg*a_k
    xc_k+1 = Ac*xc_k + Bc*e_k
physical q = [kap, nu, sig], learned eta = [Kk, Ff, Gg, Pp], encoder xi = cc.

Run:  conda run -n GraduationProject python scripts/gantry/orthogonality/code/nonlinear_closed_loop_orthogonality.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
OUT = HERE / "nonlinear_closed_loop_orthogonality"
OUT.mkdir(exist_ok=True)

CHECKS: dict[str, bool] = {}
NUM: dict[str, object] = {}
TIM: dict[str, float] = {}
NOTE: dict[str, str] = {}
ASSUME: dict[str, str] = {}
UNDECIDED: list[str] = []

R = sp.Rational


def stage(name: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] stage {name}", flush=True)


_LAST = [time.perf_counter()]


def check(name: str, value) -> bool:
    now = time.perf_counter()
    CHECKS[name] = bool(value)
    print(f"    {name:<58} {bool(value)}   ({now - _LAST[0]:.1f}s)", flush=True)
    _LAST[0] = now
    return bool(value)


def is_zero(expr) -> bool:
    """PROVED identically zero. sympy returns None for undecidable comparisons;
    None is logged as undecided and reported as a failed check, never as proved."""
    M = expr if isinstance(expr, sp.MatrixBase) else sp.Matrix([expr])
    if M.rows * M.cols == 0:
        return True
    for e in M:
        e = sp.expand(e)
        if e == 0:
            continue
        verdict = e.equals(0)          # cheaper and stronger than a blind simplify
        if verdict is None:
            e2 = sp.simplify(e)
            if e2 == 0:
                continue
            verdict = e2.equals(0)
            if verdict is None:
                UNDECIDED.append(str(e2)[:160])
                return False
        if not verdict:
            return False
    return True


# Integer witness points: exhibiting one is a proof of non-vanishing as a function.
# Small integers are used deliberately. Fractional witnesses raised to the rollout's
# polynomial degree produce astronomically large numerators, which is what made SymPy
# overflow while factoring inside a radical (recorded in algebra-tooling.md clause 6).
_TRIALS = [
    [sp.Integer(v) for v in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)],
    [sp.Integer(v) for v in (3, -2, 5, -7, 2, -3, 11, 5, -2, 7, 3, -5)],
    [sp.Integer(v) for v in (-1, 4, -6, 8, -9, 12, -14, 16, -18, 21, -22, 26)],
]


def is_nonzero(expr) -> bool:
    """PROVED not identically zero by an explicit rational witness point."""
    M = expr if isinstance(expr, sp.MatrixBase) else sp.Matrix([expr])
    if M.rows * M.cols == 0:
        return False
    free = sorted(M.free_symbols, key=str)
    for trial in _TRIALS:
        sub = dict(zip(free, trial))
        for e in M:
            val = e.xreplace(sub)
            if val == 0:
                continue
            if val.equals(0) is False:         # PROVED nonzero at this witness point
                return True
    return False


def eq_at_points(expr, variables) -> bool:
    """EXACT rational-arithmetic equality at three independent witness points. This is
    a point check, NOT a proved identity; used only where the identity content is
    carried by a separate universal proof on a generic contribution vector."""
    M_ = expr if isinstance(expr, sp.MatrixBase) else sp.Matrix([expr])
    if M_.rows * M_.cols == 0:
        return True
    for trial in _TRIALS:
        sub = dict(zip(variables, trial))
        for e in M_:
            val = e.xreplace(sub)   # fully numeric: auto-evaluates exactly, fast
            if val == 0:
                continue
            if val.equals(0) is not True:      # None (undecided) also fails, as it must
                return False
    return True


def gram_schmidt(A: sp.Matrix) -> sp.Matrix:
    """Modified Gram-Schmidt over exact arithmetic. No generic symbolic SVD."""
    cols: list[sp.Matrix] = []
    for j in range(A.cols):
        v = A[:, j]
        for u in cols:
            v = sp.simplify(v - (u.T * v)[0] * u)
        n = sp.simplify(sp.sqrt((v.T * v)[0]))
        if n == 0:
            continue
        cols.append(sp.simplify(v / n))
    return sp.Matrix.hstack(*cols) if cols else sp.zeros(A.rows, 0)


# ------------------------------------------------------------------ model constants
Ac, Bc, Cc, Dc = R(4, 5), R(1, 4), R(3, 10), R(1, 5)
NW, TS = 2, 4
M = NW * TS
YDATA = [[R(1, 10), R(2, 10), R(-1, 10), R(3, 10)], [R(2, 10), R(-1, 10), R(1, 10), R(1, 10)]]
UDATA = [[R(1, 10), R(-2, 10), R(3, 10), R(1, 10)], [R(-1, 10), R(2, 10), R(1, 10), R(-3, 10)]]
HX = [R(1, 5), R(3, 5)]
HA = [R(2, 5), R(-1, 5)]
XC0 = sp.Integer(0)
LDIAG = [R(4, 4), R(6, 4), R(2, 4), R(8, 4), R(5, 4), R(3, 4), R(7, 4), R(4, 4)]
L = sp.diag(*LDIAG)

kap, nu, sig = sp.symbols("kap nu sig", real=True)
Kk, Ff, Gg, Pp = sp.symbols("Kk Ff Gg Pp", real=True)
cc = sp.Symbol("cc", real=True)
Q = [kap, nu, sig]
ETA = [Kk, Ff, Gg, Pp]
XI = [cc]
TH = Q + ETA + XI
NP = len(TH)
TH0 = [R(1, 2), R(1, 10), R(6, 5), R(1, 5), R(1, 4), R(2, 5), R(1, 10), sp.Integer(1)]
REF = dict(zip(TH, TH0))

xs, as_, xcs, yd, ud = sp.symbols("xs as xcs yd ud", real=True)
CHI = sp.Matrix([xs, as_, xcs])

ASSUME["parameters"] = "kap, nu, sig, Kk, Ff, Gg, Pp, cc real; cs nonzero; br positive."


def step(ch, ydk, udk, qq, ee, mode):
    """One closed-loop step. Returns (next chi, output at the CURRENT state)."""
    x_, a_, c_ = ch[0], ch[1], ch[2]
    yk = qq[2] * x_
    e_ = ydk - yk
    u_ = udk + Cc * c_ + Dc * e_
    if mode == "full":
        corr, an = ee[0] * a_ + ee[3] * u_, ee[1] * x_ + ee[2] * a_
    elif mode == "clamp":
        corr, an = ee[3] * u_, sp.Integer(0)
    elif mode == "off":
        corr, an = sp.Integer(0), sp.Integer(0)
    else:
        raise ValueError(mode)
    nxt = sp.Matrix([qq[0] * x_ + qq[1] * x_ ** 2 + u_ + corr, an, Ac * c_ + Bc * e_])
    return nxt, yk


BLOCKS = {}
for mode in ("full", "clamp", "off"):
    nxt, yk = step(CHI, yd, ud, Q, ETA, mode)
    BLOCKS[mode] = {
        "step": nxt,
        "out": yk,
        "Fchi": nxt.jacobian(CHI),
        "Fth": nxt.jacobian(TH),
        "Hchi": sp.Matrix([yk]).jacobian(CHI),
        "Hth": sp.Matrix([yk]).jacobian(TH),
    }


def rollout(qq, ee, xx, mode, latscale=None):
    """Fully expanded rollout. latscale applies the latent coordinate change a' = c a."""
    if latscale is not None:
        ee = [ee[0] / latscale, latscale * ee[1], ee[2], ee[3]]
    out, ends = [], []
    for w in range(NW):
        a0 = xx * HA[w] if mode == "full" else sp.Integer(0)
        if latscale is not None and mode == "full":
            a0 = latscale * xx * HA[w]
        ch = sp.Matrix([xx * HX[w], a0, XC0])
        for k in range(TS):
            out.append(sp.expand(qq[2] * ch[0]))
            if k < TS - 1:
                ch, _ = step(ch, YDATA[w][k], UDATA[w][k], qq, ee, mode)
                ch = sp.expand(ch)
        ends.append(ch)
    return sp.Matrix(out), sp.Matrix.hstack(*ends)


def rollout_sens(qq, ee, xx, mode, dropfb=False):
    """Same rollout via LOCAL derivative blocks and the sensitivity recurrence."""
    b = BLOCKS[mode]
    p = sp.zeros(M, 1)
    J = sp.zeros(M, NP)
    ends = []
    idx = 0
    for w in range(NW):
        a0 = xx * HA[w] if mode == "full" else sp.Integer(0)
        ch = sp.Matrix([xx * HX[w], a0, XC0])
        Tk = sp.Matrix([[sp.diff(ch[i], t) for t in TH] for i in range(3)])
        for k in range(TS):
            sub = {xs: ch[0], as_: ch[1], xcs: ch[2], yd: YDATA[w][k], ud: UDATA[w][k]}
            p[idx] = sp.expand(b["out"].subs(sub, simultaneous=True))
            J[idx, :] = sp.expand(b["Hchi"].subs(sub, simultaneous=True) * Tk
                                  + b["Hth"].subs(sub, simultaneous=True))
            idx += 1
            if k < TS - 1:
                Fc = b["Fchi"].subs(sub, simultaneous=True)
                Ft = b["Fth"].subs(sub, simultaneous=True)
                if dropfb:
                    Fc[0, 2] = sp.Integer(0)
                    Fc[2, 0] = sp.Integer(0)
                Tk = sp.expand(Fc * Tk + Ft)
                ch = sp.expand(b["step"].subs(sub, simultaneous=True))
        ends.append(ch)
    return p, J, sp.Matrix.hstack(*ends)


def main() -> None:
    t0 = time.perf_counter()

    # -------------------------------------------------- 1. local derivative blocks
    stage("1 derivative blocks")
    bf = BLOCKS["full"]
    check("jac_has_latent_readout", is_nonzero(bf["Fchi"][0, 1]))
    check("jac_has_latent_write", is_nonzero(bf["Fchi"][1, 0]))
    check("jac_has_latent_persistence", is_nonzero(bf["Fchi"][1, 1]))
    check("jac_has_controller_feedback",
          is_nonzero(bf["Fchi"][2, 0]) and is_nonzero(bf["Fchi"][0, 2]))
    check("jac_has_output_parameter", is_nonzero(bf["Hth"]))
    check("jac_physical_forcing_every_step",
          is_nonzero(bf["Fth"][0, 0]) and is_nonzero(bf["Fth"][0, 1]))
    NOTE["Fchi_full_11"] = str(bf["Fchi"][0, 0])

    # -------------------------------------------------- 2. rollout, two ways
    stage("2 rollout")
    p_full, end_full = rollout(Q, ETA, cc, "full")
    p_off, end_off = rollout(Q, ETA, cc, "off")
    p_clamp, end_clp = rollout(Q, ETA, cc, "clamp")
    pf_r, Jf_r, _ = rollout_sens(Q, ETA, cc, "full")
    po_r, Jo_r, _ = rollout_sens(Q, ETA, cc, "off")

    check("rollout_expansion_matches_recurrence_value",
          is_zero(p_full - pf_r) and is_zero(p_off - po_r))
    check("sensitivity_direct_equals_recurrence", is_zero(p_full.jacobian(TH) - Jf_r))
    check("sensitivity_direct_equals_recurrence_off", is_zero(p_off.jacobian(TH) - Jo_r))
    NUM["horizon_scored_steps_per_window"] = TS
    NUM["state_transitions_per_window"] = TS - 1
    NUM["scored_rows"] = M
    NOTE["horizon_rationale"] = (
        "TS=4 scored outputs -> 3 transitions: the latent state re-enters itself twice "
        "(Gg**2 appears in a_2) and the controller error feeds back three times.")

    # -------------------------------------------------- 3. interventions
    stage("3 interventions")
    d_total = p_full - p_off
    d_lat = p_full - p_clamp
    d_dir = p_clamp - p_off
    check("decomposition_exact", is_zero(d_lat + d_dir - d_total))
    NOTE["decomposition_caveat"] = (
        "ALGEBRAICALLY TRIVIAL: true for any three returned vectors, validating no "
        "simulator semantics. The independent intervention checks below do that.")
    check("off_prediction_independent_of_learned_parameters",
          is_zero(p_off.jacobian(ETA)))
    check("off_equals_full_when_all_augmentation_zero",
          is_zero(p_full.subs({Kk: 0, Pp: 0}) - p_off))
    check("clamp_equals_full_when_latent_readout_zero",
          is_zero(p_full.subs({Kk: 0}) - p_clamp))
    check("clamp_equals_off_when_direct_route_zero",
          is_zero(p_clamp.subs({Pp: 0}) - p_off))
    check("latent_route_is_nonzero", is_nonzero(d_lat))
    check("direct_route_is_nonzero", is_nonzero(d_dir))
    check("interventions_share_initial_physical_state",
          is_zero(sp.Matrix([p_full[0] - p_off[0], p_full[0] - p_clamp[0],
                             p_full[TS] - p_off[TS], p_full[TS] - p_clamp[TS]])))
    check("each_intervention_evolves_own_controller_state",
          is_nonzero(end_full[2, :] - end_off[2, :])
          and is_nonzero(end_full[2, :] - end_clp[2, :]))
    check("clamped_latent_state_is_null", is_zero(end_clp[1, :]))

    # -------------------------------------------------- 4. pointwise blind spot
    stage("4 pointwise")
    uj = sp.Matrix([R(2, 10), R(1, 10), R(-1, 10)])
    c_pointwise = Pp * uj                      # c_eta(x_j, a = 0, u_j)
    check("pointwise_penalty_blind_to_latent_readout", is_zero(sp.diff(c_pointwise, Kk)))
    check("pointwise_penalty_identically_zero_without_direct_route",
          is_zero(c_pointwise.subs({Pp: 0})))

    # -------------------------------------------------- 5. reference geometry
    stage("5 geometry")
    S0 = Jf_r[:, 0:3].subs(REF, simultaneous=True)
    E0 = Jf_r[:, 7].subs(REF, simultaneous=True)
    D0 = L * E0
    PE = D0 * D0.pinv()
    Im = sp.eye(M)
    ME = Im - PE
    Sbar = ME * L * S0
    PS = Sbar * Sbar.pinv()
    QE, Q0 = gram_schmidt(D0), gram_schmidt(Sbar)
    rS, rE, rS_un = Sbar.rank(), D0.rank(), (L * S0).rank()

    check("ME_symmetric_idempotent", is_zero(ME - ME.T) and is_zero(ME * ME - ME))
    check("PS_symmetric_idempotent", is_zero(PS - PS.T) and is_zero(PS * PS - PS))
    check("encoder_physical_projectors_orthogonal",
          is_zero(PS * PE) and is_zero(QE.T * Q0))
    check("QE_orthonormal", is_zero(QE.T * QE - sp.eye(QE.cols)))
    check("Q0_orthonormal", is_zero(Q0.T * Q0 - sp.eye(Q0.cols)))
    check("Q0_spans_profiled_physical_range",
          is_zero(Q0 * Q0.T - PS) and is_zero(ME * Sbar - Sbar))
    check("Q0_transpose_absorbs_ME", is_zero(Q0.T * ME - Q0.T))
    check("Sbar_transpose_absorbs_ME", is_zero(Sbar.T * ME - Sbar.T))
    check("zero_residual_iff_contribution_orthogonality",
          is_zero(Sbar.T - (Q0.T * Sbar).T * Q0.T) and (Q0.T * Sbar).rank() == rS)
    NUM["rank_L_S0_unprofiled"] = int(rS_un)
    NUM["rank_encoder_D0"] = int(rE)
    NUM["rank_Sbar_after_profiling"] = int(rS)
    ASSUME["rank_conditions"] = (
        f"Sbar has column rank {rS} after profiling a rank-{rE} encoder out of a "
        f"rank-{rS_un} unprofiled physical sensitivity; every pseudoinverse is used "
        "under exactly these ranks, verified exactly over the rationals.")

    # penalty value bound: beta*t**2 <= eps, beta > 0  =>  t <= sqrt(eps/beta)
    brp = sp.Symbol("brp", positive=True)
    tt = sp.Symbol("tt", nonnegative=True)
    sl = sp.Symbol("sl", nonnegative=True)
    inner = (brp * tt ** 2 + sl) / brp            # = eps/beta with eps >= beta*t**2
    check("penalty_value_bound",
          sp.simplify(inner - tt ** 2 - sl / brp) == 0
          and sp.ask(sp.Q.nonnegative(sl / brp)) is True
          and sp.simplify(sp.sqrt(inner) ** 2 - inner) == 0)
    NOTE["penalty_value_bound_route"] = (
        "eps/beta - t**2 = slack/beta >= 0 exactly, and sqrt is monotone on the "
        "nonnegatives, so sqrt(eps/beta) >= t. Stated as an exact identity plus "
        "monotonicity rather than as an opaque decision-procedure call.")

    # -------------------------------------------------- 6. penalty and gradients
    stage("6 penalty")
    br = sp.Symbol("br", positive=True)
    A1 = Q0[0:TS, :].T * L[0:TS, 0:TS]
    A2 = Q0[TS:M, :].T * L[TS:M, TS:M]

    # --- 6a. UNIVERSAL statements, proved as IDENTITIES for a generic contribution
    # vector and a generic smooth parameter dependence. They hold for any d, so in
    # particular for this example's d_total.
    dg = sp.Matrix(sp.symbols(f"dg0:{M}", real=True))
    eg = sp.Matrix(sp.symbols("eg0:4", real=True))
    kg = sp.Symbol("kg", real=True)
    Amap = sp.Matrix(M, 4, lambda i, j: R((i * 4 + j + 1) ** 2 % 7 - 3, 5))
    bq = sp.Matrix([R((i + 1) ** 3 % 5 - 2, 3) for i in range(M)])
    cvec = sp.Matrix([R((2 * (i + 1) + 1) % 4 - 1, 2) for i in range(M)])
    dgen = Amap * eg + (eg.T * eg)[0] * bq + kg * cvec
    rg = Q0.T * L * dg
    check("penalty_equals_projector_form_generic",
          is_zero(br * (rg.T * rg) - br * ((L * dg).T * PS * (L * dg))))
    check("exact_batching_residual_sum_generic",
          is_zero(rg - (A1 * dg[0:TS, 0] + A2 * dg[TS:M, 0])))
    # The basis-free RATIONAL residual rho = PS L d carries the same penalty value and
    # the same chunk decomposition. Everything downstream uses rho, which keeps the
    # example checks in exact rational arithmetic: Q0 carries surds, and expanding a
    # degree-16 surd expression overflowed SymPy (algebra-tooling.md clause 6).
    P1 = PS[:, 0:TS] * L[0:TS, 0:TS]
    P2 = PS[:, TS:M] * L[TS:M, TS:M]
    rhog = PS * L * dg
    check("projector_residual_equals_basis_residual_generic",
          is_zero(rg.T * rg - rhog.T * rhog))
    check("projector_residual_chunk_split_generic",
          is_zero(rhog - (P1 * dg[0:TS, 0] + P2 * dg[TS:M, 0])))
    rgen = Q0.T * L * dgen
    Vgen = br * (rgen.T * rgen)[0]
    ggen = sp.Matrix([sp.diff(Vgen, e) for e in eg])
    check("penalty_gradient_gauss_newton_form_generic",
          is_zero(ggen - 2 * br * (rgen.jacobian(list(eg)).T * rgen)))
    check("exact_batching_gradient_generic",
          is_zero(ggen - 2 * br * (
              (A1 * dgen[0:TS, 0].jacobian(list(eg))).T * rgen
              + (A2 * dgen[TS:M, 0].jacobian(list(eg))).T * rgen)))
    check("joint_route_has_nonzero_physical_gradient_generic",
          is_nonzero(sp.Matrix([sp.diff(Vgen, kg)])))
    NOTE["universal_vs_example"] = (
        "6a proves the projector, chain-rule and exact-batching identities as IDENTITIES "
        "for a generic contribution vector and a generic smooth parameter dependence. "
        "6b then verifies that this example composes correctly, exactly at rational "
        "witness points and numerically by finite differences. The degree-16 expanded "
        "polynomial identity for the full nonlinear d_total was deliberately NOT "
        "attempted: it adds nothing the universal identity does not already give.")

    # --- 6b. THIS nonlinear closed-loop example, at exact rational witness points.
    Ld = L * d_total
    rho = PS * Ld                     # rational residual, kept unexpanded on purpose
    V = br * (rho.T * rho)[0]
    gV_eta = sp.Matrix([sp.diff(V, e) for e in ETA])
    gV_q = sp.Matrix([sp.diff(V, e) for e in Q])
    gV_xi = sp.Matrix([sp.diff(V, e) for e in XI])
    rho1 = P1 * d_total[0:TS, 0]
    rho2 = P2 * d_total[TS:M, 0]
    wit = TH + [br]
    check("penalty_equals_projector_form_example",
          eq_at_points(br * (rho.T * rho) - br * (Ld.T * PS * Ld), wit))
    check("penalty_gradient_gauss_newton_form_example",
          eq_at_points(gV_eta - 2 * br * (rho.jacobian(ETA).T * rho), wit))
    check("exact_batching_residual_sum_example", eq_at_points(rho - (rho1 + rho2), wit))
    check("exact_batching_gradient_example",
          eq_at_points(gV_eta - 2 * br * (rho1.jacobian(ETA).T * rho
                                          + rho2.jacobian(ETA).T * rho), wit))
    check("ann_only_routing_differs_from_joint_objective",
          is_nonzero(gV_q) or is_nonzero(gV_xi))
    NOTE["routing"] = (
        "Default routing uses d V / d eta only. d V / d kappa and d V / d xi are NOT "
        "identically zero, so the routed update is not the gradient of any single "
        "scalar objective.")
    NOTE["example_check_strength"] = (
        "6b checks are EXACT rational-arithmetic equalities at three independent witness "
        "points, not proved identities. The identity content is carried by 6a.")

    _, Jf_drop, _ = rollout_sens(Q, ETA, cc, "full", dropfb=True)
    _, Jo_drop, _ = rollout_sens(Q, ETA, cc, "off", dropfb=True)
    check("detached_intermediate_state_changes_gradient",
          is_nonzero(Jf_drop[:, 3:7] - Jf_r[:, 3:7]))
    check("off_rollout_has_no_eta_path_even_detached",
          is_zero(Jo_drop[:, 3:7]) and is_zero(Jo_r[:, 3:7]))

    # Built in the RATIONAL chunk operators P1/P2: A1/A2 carry Q0's surds, and pinv on
    # a surd matrix overflowed SymPy (algebra-tooling.md clause 6).
    tsy = sp.Symbol("tsy", real=True)
    d1 = sp.Matrix([1, 0, 0, 0])
    rc = P1 * d1
    d2 = -P2.pinv() * rc
    check("chunk_two_can_cancel",
          P2.rank() == rS and is_zero(P1 * d1 + P2 * d2) and is_nonzero(rc))
    rr1, rr2 = tsy * (P1 * d1), tsy * (P2 * d2)
    Vsum = br * ((rr1.T * rr1)[0] + (rr2.T * rr2)[0])
    check("stacked_penalty_differs_from_per_chunk_sum",
          is_zero(sp.Matrix([br * ((rr1 + rr2).T * (rr1 + rr2))[0]]))
          and is_nonzero(sp.Matrix([Vsum]))
          and is_nonzero(sp.Matrix([sp.diff(Vsum, tsy)])))
    NOTE["cross_chunk"] = (
        "Constructed d with r_1 = -r_2 != 0: V = beta||sum_b r_b||**2 = 0 while "
        "beta*sum_b||r_b||**2 > 0 with nonzero gradient.")

    # -------------------------------------------------- 7. blind spot vs rollout
    stage("7 blindspot")
    sub_no_direct = {kap: TH0[0], nu: TH0[1], sig: TH0[2],
                     Ff: TH0[4], Gg: TH0[5], Pp: 0, cc: TH0[7]}
    rK = rho.subs(sub_no_direct, simultaneous=True)
    check("rollout_projection_detects_latent_readout",
          is_nonzero(rK) and is_nonzero(sp.diff(rK, Kk)))
    nrm = sp.sqrt((rK.T * rK)[0])
    NUM["projected_residual_norm_at_reference_K"] = float(nrm.subs({Kk: TH0[3]}))
    NUM["projected_residual_dnorm_dK_at_reference"] = float(
        sp.diff(nrm, Kk).subs({Kk: TH0[3]}))
    NOTE["blind_spot_case"] = (
        "With Pp = 0 the zero-latent pointwise residual is identically zero for every "
        "Kk, while ||Q0.T L d_total|| is nonzero with nonzero derivative in Kk.")

    # -------------------------------------------------- 8. latent invariance
    stage("8 invariance")
    csy = sp.Symbol("cs", nonzero=True, real=True)
    p_full_s, end_s = rollout(Q, ETA, cc, "full", latscale=csy)
    p_off_s, _ = rollout(Q, ETA, cc, "off", latscale=csy)
    p_clamp_s, _ = rollout(Q, ETA, cc, "clamp", latscale=csy)
    check("latent_rescaling_leaves_prediction_invariant", is_zero(p_full_s - p_full))
    check("latent_rescaling_leaves_off_and_clamp_invariant",
          is_zero(p_off_s - p_off) and is_zero(p_clamp_s - p_clamp))
    d_total_s = p_full_s - p_off_s
    check("latent_rescaling_leaves_effect_invariant", is_zero(d_total_s - d_total))
    rs = PS * L * d_total_s
    check("latent_rescaling_leaves_penalty_invariant",
          eq_at_points(sp.Matrix([br * (rs.T * rs)[0] - V]), TH + [br, csy]))
    check("latent_state_scales_by_c", is_zero(end_s[1, :] - csy * end_full[1, :]))
    check("latent_state_norm_is_not_invariant",
          is_nonzero(sp.Matrix([end_s[1, 0].subs({csy: 2}) - end_full[1, 0]])))
    check("readout_norm_is_not_invariant",
          is_nonzero(sp.Matrix([(Kk / csy).subs({csy: 2}) - Kk])))
    NOTE["invariance_scope"] = (
        "a' = c a with Kk' = Kk/c, Ff' = c Ff, Gg' = Gg, a_0' = c a_0 leaves "
        "predictions, all three interventions and the penalty unchanged. ||Kk|| and "
        "the latent state norm both change, so NEITHER is a coordinate-invariant "
        "diagnostic. The zero clamp value survives scaling but would NOT survive an "
        "affine translation of the latent coordinate.")

    TIM["symbolic_algebra_s"] = time.perf_counter() - t0

    # -------------------------------------------------- 9. numerical verification
    stage("9 numeric")
    t0 = time.perf_counter()
    step_f = sp.lambdify((CHI, (yd, ud), tuple(Q), tuple(ETA)),
                         BLOCKS["full"]["step"], "numpy")
    Fchi_f = sp.lambdify((CHI, (yd, ud), tuple(Q), tuple(ETA)),
                         BLOCKS["full"]["Fchi"], "numpy")
    Fth_f = sp.lambdify((CHI, (yd, ud), tuple(Q), tuple(ETA)),
                        BLOCKS["full"]["Fth"], "numpy")
    Hchi_f = sp.lambdify((CHI, (yd, ud), tuple(Q), tuple(ETA)),
                         BLOCKS["full"]["Hchi"], "numpy")
    Hth_f = sp.lambdify((CHI, (yd, ud), tuple(Q), tuple(ETA)),
                        BLOCKS["full"]["Hth"], "numpy")

    def num_rollout(thv, ydM, udM, hxv, hav, T, sens=True, dropfb=False):
        qn, en, cn = list(thv[:3]), list(thv[3:7]), thv[7]
        nwl = len(hxv)
        p = np.zeros(nwl * T)
        J = np.zeros((nwl * T, NP))
        st = np.zeros((3, nwl * T))
        idx = 0
        for w in range(nwl):
            ch = np.array([cn * hxv[w], cn * hav[w], 0.0])
            Tk = np.zeros((3, NP))
            Tk[0, 7] = hxv[w]
            Tk[1, 7] = hav[w]
            for k in range(T):
                sg = (ydM[w][k], udM[w][k])
                p[idx] = qn[2] * ch[0]
                st[:, idx] = ch
                if sens:
                    J[idx, :] = (np.asarray(Hchi_f(ch, sg, qn, en), float).reshape(1, 3) @ Tk
                                 + np.asarray(Hth_f(ch, sg, qn, en), float).reshape(1, NP))
                idx += 1
                if k < T - 1:
                    if sens:
                        Fc = np.asarray(Fchi_f(ch, sg, qn, en), float).reshape(3, 3)
                        if dropfb:
                            Fc[0, 2] = 0.0
                            Fc[2, 0] = 0.0
                        Tk = Fc @ Tk + np.asarray(Fth_f(ch, sg, qn, en), float).reshape(3, NP)
                    ch = np.asarray(step_f(ch, sg, qn, en), float).reshape(3)
        return p, J, st

    thn = np.array([float(v) for v in TH0])
    J_ref = np.array(Jf_r.subs(REF, simultaneous=True).evalf(), dtype=float)
    ydn = [[float(v) for v in row] for row in YDATA]
    udn = [[float(v) for v in row] for row in UDATA]
    _, J4, _ = num_rollout(thn, ydn, udn, [float(v) for v in HX],
                           [float(v) for v in HA], TS)
    NUM["lambdified_vs_symbolic_max_abs"] = float(np.max(np.abs(J4 - J_ref)))
    NUM["lambdified_vs_symbolic_max_rel"] = float(
        np.max(np.abs(J4 - J_ref)) / max(1e-30, np.max(np.abs(J_ref))))
    check("lambdified_functions_match_symbolic_reference",
          NUM["lambdified_vs_symbolic_max_abs"] < 1e-12)

    rng = np.random.default_rng(20260907)
    TL, nwL = 200, 3
    ydL = (0.15 * rng.standard_normal((nwL, TL))).tolist()
    udL = (0.10 * rng.standard_normal((nwL, TL))).tolist()
    hxL, haL = [0.20, 0.60, -0.35], [0.40, -0.20, 0.25]
    pL, JL, stL = num_rollout(thn, ydL, udL, hxL, haL, TL)
    NUM["long_horizon"] = TL
    NUM["long_windows"] = nwL
    NUM["long_rows"] = int(pL.size)
    NUM["long_max_abs_state"] = float(np.max(np.abs(stL)))
    en = thn[3:7]
    Alin = np.array([[thn[0] - (1 + en[3]) * 0.2 * thn[2], en[0], (1 + en[3]) * 0.3],
                     [en[1], en[2], 0.0],
                     [-0.25 * thn[2], 0.0, 0.8]])
    NUM["linearisation_spectral_radius_at_origin"] = float(
        np.max(np.abs(np.linalg.eigvals(Alin))))
    NOTE["regime"] = (
        "Finite-horizon numerical regime only: the linearisation at the origin has "
        "spectral radius below one and the realised 200-step closed-loop states stay "
        "bounded and small. This is NOT a stability proof for the nonlinear loop.")

    JL_fd = np.zeros_like(JL)
    for j in range(NP):
        h = 1e-6 * max(1.0, abs(thn[j]))
        tp, tm = thn.copy(), thn.copy()
        tp[j] += h
        tm[j] -= h
        pp, _, _ = num_rollout(tp, ydL, udL, hxL, haL, TL, sens=False)
        pm, _, _ = num_rollout(tm, ydL, udL, hxL, haL, TL, sens=False)
        JL_fd[:, j] = (pp - pm) / (2 * h)
    colscale = np.maximum(np.max(np.abs(JL), axis=0), 1e-30)
    NUM["long_recurrence_vs_fd_max_abs"] = float(np.max(np.abs(JL - JL_fd)))
    NUM["long_recurrence_vs_fd_max_rel_colscaled"] = float(
        np.max(np.abs(JL - JL_fd) / colscale))
    check("long_recurrence_matches_finite_differences",
          NUM["long_recurrence_vs_fd_max_rel_colscaled"] < 1e-6)
    _, JL_drop, _ = num_rollout(thn, ydL, udL, hxL, haL, TL, dropfb=True)
    NUM["long_detached_controller_max_rel"] = float(
        np.max(np.abs(JL_drop - JL_fd) / colscale))
    check("long_detached_controller_is_detected",
          NUM["long_detached_controller_max_rel"] > 1e-3)
    # Finite-difference check of the ANN-routed penalty gradient on THIS example.
    Ln_ = np.array(L.evalf(), dtype=float)
    Q0n_ = np.array(Q0.evalf(), dtype=float)
    beta_num = 0.75
    poff_n = np.array(p_off.subs(REF, simultaneous=True).evalf(), dtype=float).reshape(-1)
    hxn = [float(v) for v in HX]
    han = [float(v) for v in HA]

    def Vnum(ev):
        thq = thn.copy()
        thq[3:7] = ev
        pp_, _, _ = num_rollout(thq, ydn, udn, hxn, han, TS, sens=False)
        rr_ = Q0n_.T @ Ln_ @ (pp_ - poff_n)
        return float(beta_num * rr_ @ rr_)

    gA = np.array([float(sp.diff(V, e).subs({**REF, br: R(3, 4)}, simultaneous=True))
                   for e in ETA])
    gF = np.zeros(4)
    for j in range(4):
        h = 1e-6
        ep, em = thn[3:7].copy(), thn[3:7].copy()
        ep[j] += h
        em[j] -= h
        gF[j] = (Vnum(ep) - Vnum(em)) / (2 * h)
    NUM["penalty_gradient_fd_max_abs"] = float(np.max(np.abs(gA - gF)))
    NUM["penalty_gradient_fd_max_rel"] = float(
        np.max(np.abs(gA - gF)) / max(1e-30, np.max(np.abs(gF))))
    NUM["penalty_gradient_analytic"] = gA.tolist()
    check("penalty_gradient_matches_finite_differences",
          NUM["penalty_gradient_fd_max_rel"] < 1e-6)
    NUM["cond_L_S0"] = float(np.linalg.cond(np.array((L * S0).evalf(), dtype=float)))
    NUM["cond_Sbar"] = float(np.linalg.cond(np.array(Sbar.evalf(), dtype=float)))
    NUM["cond_long_physical_block"] = float(np.linalg.cond(JL[:, 0:3]))
    TIM["numeric_s"] = time.perf_counter() - t0

    # -------------------------------------------------- 10. hard-constraint experiment
    stage("10 slsqp")
    t0 = time.perf_counter()
    try:
        from scipy.optimize import minimize, NonlinearConstraint

        pf_e = sp.lambdify((tuple(ETA),),
                           p_full.subs({kap: TH0[0], nu: TH0[1], sig: TH0[2], cc: TH0[7]},
                                       simultaneous=True), "numpy")
        Jf_e = sp.lambdify((tuple(ETA),),
                           p_full.jacobian(ETA).subs(
                               {kap: TH0[0], nu: TH0[1], sig: TH0[2], cc: TH0[7]},
                               simultaneous=True), "numpy")
        Ln = np.array(L.evalf(), dtype=float)
        Q0n = np.array(Q0.evalf(), dtype=float)
        Wn = Ln.T @ Ln
        poffn = np.array(p_off.subs(REF, simultaneous=True).evalf(), dtype=float).reshape(-1)

        def pfe(e):
            return np.asarray(pf_e(tuple(e)), float).reshape(-1)

        def jfe(e):
            return np.asarray(Jf_e(tuple(e)), float).reshape(M, 4)

        eta_truth = np.array([0.30, 0.40, 0.50, 0.20])
        ymeas = pfe(eta_truth)
        fobj = lambda e: float((pfe(e) - ymeas) @ Wn @ (pfe(e) - ymeas))
        gobj = lambda e: 2 * jfe(e).T @ (Wn @ (pfe(e) - ymeas))
        ceq = lambda e: Q0n.T @ Ln @ (pfe(e) - poffn)
        jceq = lambda e: Q0n.T @ Ln @ jfe(e)

        J0 = jceq(np.zeros(4))
        rank0 = int(np.linalg.matrix_rank(J0, tol=1e-10))
        Zn = np.linalg.svd(J0)[2][rank0:].T              # null space, never solve()
        NUM["constraint_nullspace_dim_at_eta_zero"] = int(Zn.shape[1])
        NUM["constraint_jacobian_rank_at_eta_zero"] = rank0
        check("constraint_is_degenerate_at_ann_off", rank0 < 3)
        NOTE["ann_off_degeneracy"] = (
            f"At eta = 0 the constraint Jacobian has rank {rank0} of 3: the Ff and Gg "
            "columns vanish because with Kk = Pp = 0 the latent state cannot reach the "
            "output at all. The whole plane {Kk = 0, Pp = 0} is feasible with d = 0. "
            "That is why 'ANN-off is feasible' is NOT evidence, and why a null-space "
            "seed taken at eta = 0 is useless: it moves only along Ff and Gg.")

        # Feasible NONZERO corrections by root-solving the constraint on fixed-Kk slices.
        from scipy.optimize import fsolve
        feas = []
        for kk in (0.05, 0.1, 0.2, 0.3, 0.5, -0.05, -0.1, -0.2, -0.3, -0.5):
            for z0 in ([0.2, 0.3, 0.1], [-0.3, 0.5, -0.2], [0.6, -0.4, 0.3]):
                z, _, ok, _ = fsolve(lambda z: ceq(np.concatenate(([kk], z))), z0,
                                     fprime=lambda z: jceq(np.concatenate(([kk], z)))[:, 1:4],
                                     full_output=True, xtol=1e-13)[0:4]
                e_ = np.concatenate(([kk], np.atleast_1d(z)))
                v_ = float(np.max(np.abs(ceq(e_))))
                dn_ = float(np.linalg.norm(Ln @ (pfe(e_) - poffn)))
                if ok == 1 and v_ < 1e-10 and dn_ > 1e-3:
                    feas.append({"eta": e_.tolist(), "dnorm": dn_,
                                 "objective": fobj(e_), "max_abs_ceq": v_})
        check("feasible_nonzero_correction_exists", len(feas) > 0)
        if feas:
            feas.sort(key=lambda d: d["objective"])
            NUM["feasible_nonzero_points"] = feas[:6]
            NUM["best_feasible_nonzero_objective"] = feas[0]["objective"]

        nlc = NonlinearConstraint(ceq, 0.0, 0.0, jac=jceq)
        seeds = [np.zeros(4), 0.5 * Zn[:, 0], -0.5 * Zn[:, 0], eta_truth, 0.1 * np.ones(4)]
        seeds += [np.array(f["eta"]) for f in feas[:6]]
        best = None
        for i, s in enumerate(seeds):
            res = minimize(fobj, s, jac=gobj, constraints=[nlc], method="SLSQP",
                           options={"maxiter": 800, "ftol": 1e-14})
            dn = float(np.linalg.norm(Ln @ (pfe(res.x) - poffn)))
            if res.success and (best is None or res.fun < best["f"]):
                best = {"f": float(res.fun), "eta": res.x.tolist(), "flag": int(res.status),
                        "viol": float(np.max(np.abs(ceq(res.x)))), "dnorm": dn, "seed": i}
        NUM["slsqp"] = best
        NUM["slsqp_correction_norm_at_eta_zero"] = float(
            np.linalg.norm(Ln @ (pfe(np.zeros(4)) - poffn)))
        check("slsqp_constrained_optimum_retains_nonzero_correction",
              best is not None and best["viol"] < 1e-9 and best["dnorm"] > 1e-3)
        NOTE["slsqp_scope"] = (
            "SEPARATE method: hard equality constraint, NOT the ANN-only penalty. "
            "eta = 0 is feasible with zero correction and is explicitly rejected as "
            "evidence. Two DIFFERENT questions are reported separately: (1) does the "
            "feasible set contain nonzero corrections at all; (2) does the CONSTRAINED "
            "OPTIMUM retain one. A false answer to (2) with a true answer to (1) is a "
            "real reportable outcome about this objective and geometry, not a solver "
            "failure to paper over. No solver result proves global convergence or "
            "parameter recovery.")
    except ImportError:
        NOTE["slsqp_scope"] = "scipy unavailable; hard-constraint experiment skipped in SymPy."
    TIM["optimization_s"] = time.perf_counter() - t0

    failed = [k for k, v in CHECKS.items() if not v]
    result = {
        "category": "structural verification examples; generic nonlinear closed-loop model only",
        "not_a_gantry_transcription": True,
        "engine": f"sympy {sp.__version__}",
        "numpy": np.__version__,
        "checks": CHECKS,
        "numbers": NUM,
        "timings_s": TIM,
        "assumptions": ASSUME,
        "notes": NOTE,
        "n_checks": len(CHECKS),
        "failed": failed,
        "undecided_identities": UNDECIDED,
    }
    (OUT / "nonlinear_closed_loop_orthogonality_sympy.json").write_text(
        json.dumps(result, indent=2) + "\n")
    for k, v in CHECKS.items():
        print(f"{k:<58} {v}")
    print(f"\n{len(CHECKS)} checks, {len(failed)} failed. "
          f"algebra {TIM['symbolic_algebra_s']:.2f}s numeric {TIM['numeric_s']:.2f}s "
          f"optim {TIM['optimization_s']:.2f}s")
    if failed:
        print("FAILED: " + ", ".join(failed))
    print("UNDECIDED: " + (", ".join(UNDECIDED) if UNDECIDED else "none"))


if __name__ == "__main__":
    main()
