function check_leine_collocation()
%CHECK_LEINE_COLLOCATION  Gate 3: is the stick band large enough for the solver?
%
% D-204, gate 3. This is the one gate that tests a PUBLISHED criterion rather
% than a property we chose ourselves.
%
% THEORY: leine1998 (Leine, van Campen, de Kraker, van den Steen, "Stick-Slip
% Vibrations Induced by Alternate Friction Models", Nonlinear Dynamics
% 16(1):41-54, 1998, p. 7; free Version of Record at
% https://pure.tue.nl/ws/files/1478861/605090.pdf):
%
%   "The collocation points of the Runge-Kutta integration method during the
%    stick mode should all be situated within the stick band to avoid numerical
%    instability problems of the Karnopp model."
%
% RK4 evaluates the right-hand side at four collocation points per step:
%   s1 = x,   s2 = x + h/2*k1,   s3 = x + h/2*k2,   s4 = x + h*k3.
% Each sees its own stage velocity, so each makes its own stuck/slip decision.
% If a rail is inside the band at s1 (declared stuck) but outside it at s2, s3
% or s4, the single step mixes a held rail with a sliding one and the stick is
% not resolved consistently. That is precisely what Leine's sentence forbids.
%
% WHY THIS GATE EXISTS. The implementation sets
%     V_EPS = (cc1+cc2)/m_total * ts,
% which is exactly the velocity change dry friction produces in ONE step, i.e.
% the fixed-step analogue of Leine's Runge-Kutta tolerance. Leine requires the
% band to be STRICTLY LARGER than that quantity ("eta << v_dr", tolerance
% smaller than eta). So the shipped value sits ON the boundary of the published
% criterion rather than inside it. This gate measures how far outside the band
% the collocation points actually travel, and therefore what margin factor the
% criterion demands. That number is the deliverable, and it is what D-204's
% "3 to 10x margin" must be checked against rather than guessed.
%
% METHOD. One RK4 step is taken from each sampled RECORDED state with the
% recorded input held constant across the step. ZOH is exact here, not an
% approximation: the controller is discrete at 20 kHz and ts = 1/20000, so the
% force genuinely is constant over the step. Taking single steps from recorded
% states rather than free-running avoids replay drift, so every tested state is
% one the plant actually visited.
%
% REPORTED, and the second number is the actionable one:
%   - the fraction of stuck rail-steps whose later collocation points leave the
%     band (Leine violations);
%   - max |v_stage| over ALL collocation points of a stuck rail, divided by
%     v_eps. This is the margin factor the band needs to satisfy the criterion.
%
% PASS is defined honestly: a violation fraction of 0 passes outright. A nonzero
% fraction is NOT scored as a failure of the friction law, because gates 1 and 2
% already establish the law is right; it is scored as the band being undersized,
% and the required factor is printed.
%
% Run:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-coulomb'); check_leine_collocation"

    THIS_DIR  = fileparts(mfilename('fullpath'));
    REPO_ROOT = fileparts(fileparts(THIS_DIR));
    addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
    addpath(THIS_DIR);

    REC    = fullfile(REPO_ROOT, 'data', 'gantry', 'matlab', 'trajectory', ...
                      'augmentation_coulomb_karnopp', 'V1_standstill_Yp10.mat');
    STRIDE = 20;

    cfg = gtd_config('augmentation', true, 0.10);
    CC  = [16.80; 18.35; 11.60];
    mh  = cfg.mh - cfg.ma;
    P   = [1, 1, 0; cfg.Lb/2, -cfg.Lb/2, 0; 0, 0, 1];
    h   = cfg.ts;

    args = {cfg.m1, cfg.m2, cfg.mb, mh, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
            cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
            cfg.ma, cfg.ka, cfg.ca, cfg.L0, CC(1), CC(2), CC(3), cfg.ts};

    m_total = cfg.m1 + cfg.m2 + cfg.mb + mh + cfg.ma;
    [v_eps, v_margin] = coulomb_v_eps(CC(1), CC(2), m_total, cfg.ts);

    fprintf('\n=== GATE 3: Leine collocation criterion ===\n');
    assert(isfile(REC), 'record not found: %s', REC);
    S = load(REC);
    N = size(S.u_total, 1);
    idx = 1:STRIDE:N;
    fprintf('%d RK4 steps from recorded states, h = %.2e s\n', numel(idx), h);
    fprintf('v_eps = %.4e m/s   (margin %gx the per-step increment)\n\n', v_eps, v_margin);

    f = @(x, u) gantrySystemExtendedCoulomb(u, x, args{:});
    mh_ = mh; cfg_ = cfg;

    n_stuck = 0; n_viol = 0; worst_ratio = 0; n_broke = 0;
    viol_by_rail = zeros(3,1); stuck_by_rail = zeros(3,1);

    for t = idx
        xl = double(S.x_logical(t,:)).';
        x  = [xl(1:3); double(S.delta_a(t)); xl(4:6); double(S.vdelta_a(t))];
        u  = P * double(S.u_total(t,:)).';

        k1 = f(x, u);              s1 = x;
        s2 = x + (h/2)*k1;         k2 = f(s2, u);
        s3 = x + (h/2)*k2;         k3 = f(s3, u);
        s4 = x + h*k3;

        V = [P.'*s1(5:7), P.'*s2(5:7), P.'*s3(5:7), P.'*s4(5:7)];   % 3x4

        % Leine's criterion governs the STICK MODE only. A rail inside the band
        % whose required force exceeded cc has BROKEN AWAY: it is sliding, and
        % leaving the band is then correct physics rather than a numerical
        % defect. Held and broken-away rails are separated by recovering the
        % friction force the same way gate 2 does, from the cc=0 difference, and
        % testing saturation. Counting them together overstates the violation.
        F1 = recover_F(x, u, args, P, mh_, cfg_);
        for i = 1:3
            if abs(V(i,1)) < v_eps          % inside the band at the first point
                if abs(abs(F1(i)) - CC(i)) < 1e-9
                    n_broke = n_broke + 1;  % saturated: sliding, not stick mode
                    continue
                end
                n_stuck = n_stuck + 1;
                stuck_by_rail(i) = stuck_by_rail(i) + 1;
                r = max(abs(V(i,:))) / v_eps;
                worst_ratio = max(worst_ratio, r);
                if r > 1
                    n_viol = n_viol + 1;
                    viol_by_rail(i) = viol_by_rail(i) + 1;
                end
            end
        end
    end

    fprintf('--- rail-steps inside the band, classified ---\n');
    fprintf('  broke away (saturated: sliding, criterion does not apply) %8d\n', n_broke);
    fprintf('--- stuck AND HELD rail-steps examined ---\n');
    fprintf('  X1 %8d   X2 %8d   Y %8d   total %8d\n', ...
            stuck_by_rail(1), stuck_by_rail(2), stuck_by_rail(3), n_stuck);
    fprintf('--- collocation points leaving the band ---\n');
    fprintf('  X1 %8d   X2 %8d   Y %8d   total %8d  (%.2f%%)\n\n', ...
            viol_by_rail(1), viol_by_rail(2), viol_by_rail(3), n_viol, ...
            100*n_viol/max(1,n_stuck));

    fprintf('worst max|v_stage| over the 4 collocation points = %.3f x v_eps\n', worst_ratio);

    if n_stuck == 0
        fprintf('\nINCONCLUSIVE: no stuck rail-steps in the sample.\n');
        fprintf('\nGATE 3: INCONCLUSIVE\n\n');
    elseif n_viol == 0
        fprintf('Every collocation point of every stuck rail stayed inside the band.\n');
        fprintf('\nGATE 3: PASS\n\n');
    else
        fprintf('\nBand is UNDERSIZED against leine1998 at the shipped value.\n');
        fprintf('This is a SIZING result, not a friction-law failure: gates 1 and 2\n');
        fprintf('establish the law independently.\n');
        margin_scan(S, idx, P, args, h, v_eps, CC, mh_, cfg_);
        fprintf('\nGATE 3: FAIL (band undersized at the shipped value)\n\n');
    end
end

% ===========================================================================

function margin_scan(S, idx, P, args, h, v_eps0, CC, mh, cfg)
% The required margin is a FIXED POINT, not a one-shot read: enlarging the band
% declares more rails stuck, which changes the trajectory of the collocation
% points and hence the excursions. So the factor is swept rather than inferred
% from the worst ratio at the shipped value.
%
% The band is scaled by passing a scaled `ts` to the ODE. That is exact, not a
% hack: the function's own header states ts is "used ONLY to size the stick band
% V_EPS". The integration step h is unchanged throughout.
    fprintf('\n--- margin scan (band scaled; integration step h unchanged) ---\n');
    fprintf('  factor   v_eps [m/s]   stuck steps   violations   worst/band\n');
    for g = [1 3 10 12 20 30 50]
        a = args; a{23} = g * args{23};      % ts enters only through V_EPS
        f = @(x, u) gantrySystemExtendedCoulomb(u, x, a{:});
        ve = g * v_eps0;
        ns = 0; nv = 0; wr = 0;
        for t = idx
            xl = double(S.x_logical(t,:)).';
            x  = [xl(1:3); double(S.delta_a(t)); xl(4:6); double(S.vdelta_a(t))];
            u  = P * double(S.u_total(t,:)).';
            k1 = f(x, u);         s2 = x + (h/2)*k1;
            k2 = f(s2, u);        s3 = x + (h/2)*k2;
            k3 = f(s3, u);        s4 = x + h*k3;
            V  = [P.'*x(5:7), P.'*s2(5:7), P.'*s3(5:7), P.'*s4(5:7)];
            F1 = recover_F(x, u, a, P, mh, cfg);   % same held/broke split as above
            for i = 1:3
                if abs(V(i,1)) < ve
                    if abs(abs(F1(i)) - CC(i)) < 1e-9, continue; end   % broke away
                    ns = ns + 1;
                    r  = max(abs(V(i,:))) / ve;
                    wr = max(wr, r);
                    if r > 1, nv = nv + 1; end
                end
            end
        end
        fprintf('  %5dx   %.4e   %11d   %10d   %9.3f%s\n', ...
                g, ve, ns, nv, wr, tern(nv == 0, '   <- satisfies leine1998', ''));
    end
end

function F = recover_F(x, u, args, P, mh, cfg)
% Friction force inferred from its EFFECT, not read out of the block: the same
% call with cc = 0 differs only by the friction term.
    a0 = args; a0{20} = 0; a0{21} = 0; a0{22} = 0;
    dC = gantrySystemExtendedCoulomb(u, x, args{:});
    dF = gantrySystemExtendedCoulomb(u, x, a0{:});
    Y = x(3); da = x(4);
    m1=cfg.m1; m2=cfg.m2; mb=cfg.mb; ma=cfg.ma;
    Lb=cfg.Lb; Jb=cfg.Jb; Jh=cfg.Jh; d=cfg.d; L0=cfg.L0;
    M = [ m1+m2+mb+mh+ma, (m1-m2)*Lb/2-(mh+ma)*Y-ma*L0-ma*da, 0, 0;
          (m1-m2)*Lb/2-(mh+ma)*Y-ma*L0-ma*da, ...
              Jb+Jh+(m1+m2)*Lb^2/4+(mh+ma)*d^2+mh*Y^2+ma*(Y+L0+da)^2, -(mh+ma)*d, -ma*d;
          0, -(mh+ma)*d, mh+ma, ma;
          0, -ma*d, ma, ma];
    Minv_B = M \ [eye(3); zeros(1,3)];
    F = P \ (Minv_B \ (dF(5:8) - dC(5:8)));
end

function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
