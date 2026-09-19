function check_garcia_constant_velocity()
%CHECK_GARCIA_CONSTANT_VELOCITY  Gate 1: replicate Garcia's own identification
%                                experiment against the implemented friction.
%
% D-204, gate 1. This is the strongest available test of the friction, because it
% does not inspect the implementation at all: it INVERTS it. For a prescribed
% constant-velocity state it solves for the input force that holds the velocity
% constant, exactly as Garcia held each axis at constant velocity and read the
% force, then regresses that force on velocity and reads the intercept.
%
% THEORY: garcia2013 (garcia2013_gantry-decoupling-control.pdf), Sect. "Parameter
% identification": "the Coulomb and viscous frictions of the Y-axis are estimated
% by displacing the payload at various constant velocities", with X1 and X2
% blocked; and the X frictions "by moving them in synchronization at various
% constant velocities". The identified values are the intercepts of that line.
%
% WHY THE INTERCEPT AND NOT THE WHOLE CURVE. With Y and delta_a held at zero the
% mass matrix is constant, so the required force is EXACTLY affine in v on each
% sign branch: the viscous, inertial and coupling terms are all odd-linear in v
% and contribute only to the SLOPE, while the Coulomb term is the sign jump and
% is the whole INTERCEPT. So the intercept is a clean read of cc even though the
% rails are coupled through M and through the stuck-rail solve.
%
% THREE EXPERIMENTS, because a single one cannot separate cc1 from cc2:
%   E1  Y alone      (dY = v, X rails at rest)     -> intercept = ccy
%   E2  X symmetric  (dX1 = dX2 = v)               -> u_X     : cc1 + cc2
%                                                     u_Theta : Lb/2*(cc1 - cc2)
%   E3  X antisymm.  (dX1 = v, dX2 = -v)           -> u_X     : cc1 - cc2
%                                                     u_Theta : Lb/2*(cc1 + cc2)
% E2 and E3 together give cc1 and cc2 SEPARATELY. That matters: cc1 ~= cc2 is
% recoverable ONLY because the friction is built in stage coordinates [X1,X2,Y]
% and projected with P. The u_Theta intercept of E2 is the sharpest frame
% evidence in the whole gate suite: under the wrong (logical-frame) law the Theta
% row would carry cc2*sign(dTheta) = cc2*sign(0) = 0 there, so the intercept
% would be 0 instead of Lb/2*(cc1-cc2) = -0.5619 Nm.
%
% It also closes, in its static form, the Theta torque prediction that
% README.md pre-registered and could only test on a standstill record, where
% sign(dX1) and sign(dX2) alternate at multisine rate and the mean torque
% averages to near zero.
%
% PASS CRITERIA (absolute, not tuned): every recovered Coulomb value within
% 1e-6 of Garcia's identified number, and every affine fit residual at round-off
% (below 1e-9 N), which is what says the "intercept = Coulomb" reading is valid
% rather than an artefact of curvature.
%
% Run:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-coulomb'); check_garcia_constant_velocity"

    % ── path bootstrap (mirrors generate_trajectory_data_coulomb.m) ─────────
    THIS_DIR  = fileparts(mfilename('fullpath'));
    REPO_ROOT = fileparts(fileparts(THIS_DIR));
    addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
    addpath(THIS_DIR);

    cfg = gtd_config('augmentation', true, 0.50);   % MA_FRAC matches the target dataset

    % Garcia's identified Coulomb forces. NOT a knob.
    CC1 = 16.80; CC2 = 18.35; CCY = 11.60;

    pp = struct();
    pp.m1 = cfg.m1; pp.m2 = cfg.m2; pp.mb = cfg.mb;
    pp.mh = cfg.mh - cfg.ma;              % the extended ODE takes mh_rigid
    pp.Lb = cfg.Lb; pp.Jb = cfg.Jb; pp.Jh = cfg.Jh; pp.d = cfg.d;
    pp.cg1 = cfg.cg1; pp.cg2 = cfg.cg2; pp.cb1 = cfg.cb1; pp.cb2 = cfg.cb2;
    pp.cy = cfg.cy; pp.kb1 = cfg.kb1; pp.kb2 = cfg.kb2;
    pp.ma = cfg.ma; pp.ka = cfg.ka; pp.ca = cfg.ca; pp.L0 = cfg.L0;
    pp.cc1 = CC1; pp.cc2 = CC2; pp.ccy = CCY; pp.ts = cfg.ts;

    m_total = pp.m1 + pp.m2 + pp.mb + pp.mh + pp.ma;
    v_eps   = coulomb_v_eps(CC1, CC2, m_total, pp.ts);

    fprintf('\n=== GATE 1: Garcia constant-velocity identification ===\n');
    fprintf('MA_FRAC 0.50, mh_rigid %.4f kg, m_total %.4f kg\n', pp.mh, m_total);
    fprintf('stick band v_eps = %.4e m/s\n', v_eps);

    % Velocities: every one is >> v_eps, so all probed rails genuinely slide.
    V = [1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1];
    fprintf('probe speeds %.0e .. %.0e m/s, all > %.0fx v_eps\n\n', ...
            min(V), max(V), min(V)/v_eps);

    ok = true;

    % ── E1: Y axis alone ────────────────────────────────────────────────────
    [iY, rY] = branch_intercept(@(v) state_Y(v), 3, 3, 3, V, pp);
    ok = report('E1  u_Y     ', iY, CCY, rY, 'ccy') && ok;

    % ── E2: X symmetric  (dX1 = dX2 = v) ────────────────────────────────────
    [iXs, rXs] = branch_intercept(@(v) state_Xsym(v),  [1 2], [1 2], 1, V, pp);
    [iTs, rTs] = branch_intercept(@(v) state_Xsym(v),  [1 2], [1 2], 2, V, pp);
    ok = report('E2  u_X     ', iXs, CC1 + CC2,              rXs, 'cc1+cc2')       && ok;
    ok = report('E2  u_Theta ', iTs, pp.Lb/2*(CC1 - CC2),    rTs, 'Lb/2*(cc1-cc2)') && ok;

    % ── E3: X antisymmetric  (dX1 = v, dX2 = -v) ────────────────────────────
    [iXa, rXa] = branch_intercept(@(v) state_Xanti(v, pp.Lb), [1 2], [1 2], 1, V, pp);
    [iTa, rTa] = branch_intercept(@(v) state_Xanti(v, pp.Lb), [1 2], [1 2], 2, V, pp);
    ok = report('E3  u_X     ', iXa, CC1 - CC2,              rXa, 'cc1-cc2')       && ok;
    ok = report('E3  u_Theta ', iTa, pp.Lb/2*(CC1 + CC2),    rTa, 'Lb/2*(cc1+cc2)') && ok;

    % ── separate cc1 and cc2 from E2 + E3 ───────────────────────────────────
    cc1_hat = (iXs + iXa) / 2;
    cc2_hat = (iXs - iXa) / 2;
    fprintf('\n--- recovered Coulomb forces (the actual deliverable) ---\n');
    ok = report('cc1         ', cc1_hat, CC1, 0, 'Garcia 16.80 N')  && ok;
    ok = report('cc2         ', cc2_hat, CC2, 0, 'Garcia 18.35 N')  && ok;
    ok = report('ccy         ', iY,      CCY, 0, 'Garcia 11.60 N')  && ok;

    fprintf('\nFrame evidence: cc1 and cc2 are recovered SEPARATELY and differ by\n');
    fprintf('%.4f N, and u_Theta carries a nonzero intercept in E2. Both are\n', CC2 - CC1);
    fprintf('impossible under a logical-frame law, where the Theta row would take\n');
    fprintf('sign(dTheta) = sign(0) = 0 in E2 and the intercept would vanish.\n');

    if ok
        fprintf('\nGATE 1: PASS\n\n');
    else
        fprintf('\nGATE 1: FAIL\n\n');
    end
end

% ===========================================================================

function x = state_Y(v)
    x = zeros(8,1); x(7) = v;                 % dY = v; X rails at rest
end

function x = state_Xsym(v)
    x = zeros(8,1); x(5) = v;                 % dX = v, dTheta = 0 -> dX1 = dX2 = v
end

function x = state_Xanti(v, Lb)
    x = zeros(8,1); x(6) = 2*v/Lb;            % dX = 0, dTheta = 2v/Lb -> dX1 = v, dX2 = -v
end

function [icept, resid] = branch_intercept(xfun, iu, ie, icomp, V, pp)
% Solve for the holding force at each +v and -v, fit u_icomp = a + b*v on each
% sign branch, and return the half-difference of the two intercepts. The two
% branches are fitted separately on purpose: a single fit through v = 0 would
% average the sign jump away, which is the whole quantity being measured.
    up = zeros(numel(V),1); un = zeros(numel(V),1);
    for k = 1:numel(V)
        uP = solve_hold(xfun( V(k)), iu, ie, pp);   up(k) = uP(icomp);
        uN = solve_hold(xfun(-V(k)), iu, ie, pp);   un(k) = uN(icomp);
    end
    A = [ones(numel(V),1), V(:)];
    cp = A \ up;   rp = norm(A*cp - up, inf);
    cn = A \ un;   rn = norm(A*cn - un, inf);
    icept = (cp(1) - cn(1)) / 2;
    resid = max(rp, rn);
end

function u = solve_hold(x, iu, ie, pp)
% Newton solve for the input components iu that zero the STAGE accelerations ie.
% The map u -> a is affine inside one active set, so this converges in a single
% step once the stuck/slip pattern has settled; the loop exists only for the
% pattern, and the final residual is asserted rather than assumed.
    u = zeros(3,1);
    h = 1e-5;
    for it = 1:40
        r = stage_acc(u, x, pp); r = r(ie);
        if norm(r, inf) < 1e-11, break; end
        J = zeros(numel(ie), numel(iu));
        for k = 1:numel(iu)
            uk = u; uk(iu(k)) = uk(iu(k)) + h;
            rk = stage_acc(uk, x, pp); rk = rk(ie);
            J(:,k) = (rk - r) / h;
        end
        u(iu) = u(iu) - J \ r;
    end
    r = stage_acc(u, x, pp);
    assert(norm(r(ie), inf) < 1e-8, ...
           'solve_hold did not converge: residual %.3e m/s^2', norm(r(ie), inf));
end

function a_stage = stage_acc(u, x, pp)
    dxdt = gantrySystemExtendedCoulomb(u, x, pp.m1, pp.m2, pp.mb, pp.mh, pp.Lb, ...
              pp.Jb, pp.Jh, pp.d, pp.cg1, pp.cg2, pp.cb1, pp.cb2, pp.cy, ...
              pp.kb1, pp.kb2, pp.ma, pp.ka, pp.ca, pp.L0, ...
              pp.cc1, pp.cc2, pp.ccy, pp.ts);
    P = [1, 1, 0; pp.Lb/2, -pp.Lb/2, 0; 0, 0, 1];
    a_stage = P.' * dxdt(5:7);
end

function ok = report(name, got, want, resid, what)
    err = abs(got - want);
    ok  = err < 1e-6;
    if resid > 0
        extra = sprintf('  fit resid %.2e', resid);
        ok = ok && (resid < 1e-9);
    else
        extra = '';
    end
    fprintf('%s = %+12.6f   expected %+12.6f (%s)   err %.2e   %s%s\n', ...
            name, got, want, what, err, tern(ok, 'OK  ', 'FAIL'), extra);
end

function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
