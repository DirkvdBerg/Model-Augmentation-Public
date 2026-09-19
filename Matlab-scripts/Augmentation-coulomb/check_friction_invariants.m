function check_friction_invariants(dataset, record, ma_frac)
%CHECK_FRICTION_INVARIANTS  Gate 2: Coulomb's law holds at every sampled state.
%
% D-204, gate 2. Gate 1 validates the friction MAGNITUDE on prescribed
% constant-velocity states, which are all pure sliding. This gate validates the
% LAW on states the plant actually visited, where rails stick, slide and break
% away, and where the three rails are coupled through M(Y) and the stuck-set
% solve.
%
% HOW THE FRICTION FORCE IS RECOVERED, without reading it out of the
% implementation. The ODE is called twice at the same (x, u): once with Garcia's
% cc and once with cc = 0. The only difference between the two right-hand sides
% is the friction, and it enters as dxdt = A*x + B*(u - P*F), so
%
%     a_free - a_cc = Minv_B * (P*F)
%
% and P*F is recovered by least squares on that 4x3 system (Minv_B has full
% column rank), then F = P \ (P*F). Nothing inside the friction block is
% inspected; the force is inferred from its effect. That is what makes this a
% test rather than a restatement.
%
% THE FOUR INVARIANTS, which together ARE Coulomb's law with a Karnopp stick
% state. THEORY: garcia2013 Fig. 2 for the slip branch; Coulomb's law at rest
% (|F| <= cc) for the stick branch, which is what the Karnopp state implements.
%
%   I1  BOUND        |F_i| <= cc_i at every state, sliding or stuck. This is the
%                    half of Coulomb's law that a pure sign() model cannot
%                    violate but a stuck-set SOLVE can, since the solve is free
%                    to return any force until it is saturated.
%   I2  SLIP         |v_i| >= v_eps  =>  F_i = cc_i*sign(v_i) exactly.
%   I3  STICK        |v_i| <  v_eps  =>  either the rail is held, i.e. its stage
%                    acceleration is zero, or it broke away and |F_i| = cc_i.
%                    A stuck rail that is neither held nor saturated is the
%                    failure this invariant exists to catch.
%   I4  DISSIPATION  split by regime, because the two branches differ and a
%                    single "friction never does positive work" statement is
%                    FALSE for the stuck branch:
%                    I4a SLIP:  v_i*F_i >= 0 strictly. This is Garcia's law and
%                        it admits no exception.
%                    I4b STICK: |v_i*F_i| <= v_eps*cc_i. A rail inside the band
%                        is declared stuck and HELD, and its held force is
%                        whatever zeroes the acceleration, so its sign is
%                        unrelated to the residual velocity's sign. Holding a
%                        rail that still carries |v| < v_eps therefore does a
%                        bounded amount of positive work. This is the PRICE OF
%                        THE BAND, not a defect: it is bounded by v_eps*cc, and
%                        since v_eps scales with ts it vanishes with the step.
%                    Checked PER RAIL, not only on the sum, because two rails can
%                    hide a sign error between them in the total. The existing
%                    A3 gate checks the sum and therefore cannot see either the
%                    sign error or this bound.
%
% The regime census is printed and is part of the result: an invariant check that
% never saw a stuck rail, or never saw a breakaway, has not tested what it
% claims to, and the counts are reported so that cannot pass silently.
%
% Run:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-coulomb'); check_friction_invariants"

    THIS_DIR  = fileparts(mfilename('fullpath'));
    REPO_ROOT = fileparts(fileparts(THIS_DIR));
    addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
    addpath(THIS_DIR);

    % Defaults keep the original call form working; D-204 added the arguments so
    % the same gate can be pointed at the production-knob friction record.
    if nargin < 1 || isempty(dataset), dataset = 'augmentation_coulomb_karnopp'; end
    if nargin < 2 || isempty(record),  record  = 'V1_standstill_Yp10.mat';       end
    if nargin < 3 || isempty(ma_frac), ma_frac = 0.10;                            end
    REC    = fullfile(REPO_ROOT, 'data', 'gantry', 'matlab', 'trajectory', ...
                      dataset, record);
    STRIDE = 4;                       % 240000/4 = 60000 sampled states

    cfg = gtd_config('augmentation', true, ma_frac);
    CC  = [16.80; 18.35; 11.60];
    mh  = cfg.mh - cfg.ma;
    P   = [1, 1, 0; cfg.Lb/2, -cfg.Lb/2, 0; 0, 0, 1];

    base = {cfg.m1, cfg.m2, cfg.mb, mh, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
            cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
            cfg.ma, cfg.ka, cfg.ca, cfg.L0};
    argsC = [base, {CC(1), CC(2), CC(3), cfg.ts}];
    argsF = [base, {0, 0, 0, cfg.ts}];

    m_total = cfg.m1 + cfg.m2 + cfg.mb + mh + cfg.ma;
    [v_eps, v_margin] = coulomb_v_eps(CC(1), CC(2), m_total, cfg.ts);

    fprintf('\n=== GATE 2: friction invariants on visited states ===\n');
    assert(isfile(REC), 'record not found: %s', REC);
    S = load(REC);
    N = size(S.u_total, 1);
    idx = 1:STRIDE:N;
    fprintf('record %s\n', REC);
    fprintf('%d samples (stride %d of %d), v_eps = %.4e m/s\n\n', numel(idx), STRIDE, N, v_eps);

    % worst-case trackers
    w_bound = 0; w_slip = 0; w_held = 0; w_break = 0;
    w_diss_slip = 0; w_diss_stick = 0; w_diss_stick_rel = 0;
    n_slip = 0; n_held = 0; n_break = 0; n_bad_stick = 0;

    for t = idx
        xl = double(S.x_logical(t,:)).';        % [X;Theta;Y; dX;dTheta;dY]
        x  = [xl(1:3); double(S.delta_a(t)); xl(4:6); double(S.vdelta_a(t))];
        % u_total is in STAGE coordinates; the ODE takes logical forces.
        % gtd_save_record uses q_logical = (P')\q_stage, so power equivalence
        % gives u_logical = P*u_stage.
        u  = P * double(S.u_total(t,:)).';

        dC = gantrySystemExtendedCoulomb(u, x, argsC{:});
        dF = gantrySystemExtendedCoulomb(u, x, argsF{:});

        Minv_B = minv_b(x, cfg, mh);
        PF = Minv_B \ (dF(5:8) - dC(5:8));      % least squares, 4x3
        F  = P \ PF;

        v_stage = P.' * x(5:7);
        a_stage = P.' * dC(5:7);
        a_free  = P.' * dF(5:7);
        scale   = max(1, max(abs(a_free)));

        for i = 1:3
            % I1 bound
            w_bound = max(w_bound, abs(F(i)) - CC(i));

            if abs(v_stage(i)) >= v_eps
                % I2 slip
                n_slip = n_slip + 1;
                w_slip = max(w_slip, abs(F(i) - CC(i)*sign(v_stage(i))));
                % I4a slip dissipation: no exception permitted
                w_diss_slip = max(w_diss_slip, -v_stage(i)*F(i));
            else
                % I4b stick dissipation: bounded by the band, reported both
                % absolutely and as a fraction of its own bound v_eps*cc_i
                w_diss_stick     = max(w_diss_stick, abs(v_stage(i)*F(i)));
                w_diss_stick_rel = max(w_diss_stick_rel, ...
                                       abs(v_stage(i)*F(i)) / (v_eps*CC(i)));
                % I3 stick: held, or broke away at saturation
                held  = abs(a_stage(i)) / scale;
                broke = abs(abs(F(i)) - CC(i));
                if held < 1e-9
                    n_held = n_held + 1;
                    w_held = max(w_held, held);
                elseif broke < 1e-9
                    n_break = n_break + 1;
                    w_break = max(w_break, broke);
                else
                    n_bad_stick = n_bad_stick + 1;
                end
            end
        end
    end

    fprintf('--- regime census (rail-samples) ---\n');
    fprintf('  sliding          %10d\n', n_slip);
    fprintf('  stuck and held   %10d\n', n_held);
    fprintf('  stuck, broke away%10d\n', n_break);
    fprintf('  stuck, NEITHER   %10d   <- must be 0\n\n', n_bad_stick);

    ok = true;
    ok = pass('I1 bound       |F_i| <= cc_i', w_bound, 1e-9, 'N')            && ok;
    ok = pass('I2 slip        F_i = cc_i*sign(v_i)', w_slip, 1e-9, 'N')      && ok;
    ok = pass('I3 stick-held  |a_stage_i| (relative)', w_held, 1e-9, '-')    && ok;
    ok = pass('I3 breakaway   ||F_i| - cc_i|', w_break, 1e-9, 'N')           && ok;
    ok = pass('I4a slip diss.  max(-v_i*F_i)', w_diss_slip, 1e-9, 'W')       && ok;
    % I4b is a BOUND, not a zero: the band permits |v*F| up to v_eps*cc_i.
    ok = pass('I4b stick diss. as fraction of v_eps*cc_i', ...
              w_diss_stick_rel - 1, 1e-9, '-')                               && ok;
    fprintf('     stick |v*F| worst %.3e W against the band bound %.3e W\n', ...
            w_diss_stick, v_eps*CC(2));

    if n_slip == 0 || n_held == 0
        fprintf('\nINCONCLUSIVE: the sample never exercised both regimes.\n');
        ok = false;
    end
    if n_bad_stick > 0
        fprintf('\nFAIL: %d stuck rail-samples were neither held nor saturated.\n', n_bad_stick);
        ok = false;
    end

    if ok
        fprintf('\nGATE 2: PASS\n\n');
    else
        fprintf('\nGATE 2: FAIL\n\n');
    end
end

% ===========================================================================

function Minv_B = minv_b(x, cfg, mh)
    Y = x(3); da = x(4);
    m1=cfg.m1; m2=cfg.m2; mb=cfg.mb; ma=cfg.ma;
    Lb=cfg.Lb; Jb=cfg.Jb; Jh=cfg.Jh; d=cfg.d; L0=cfg.L0;
    M = [ m1+m2+mb+mh+ma, (m1-m2)*Lb/2-(mh+ma)*Y-ma*L0-ma*da, 0, 0;
          (m1-m2)*Lb/2-(mh+ma)*Y-ma*L0-ma*da, ...
              Jb+Jh+(m1+m2)*Lb^2/4+(mh+ma)*d^2+mh*Y^2+ma*(Y+L0+da)^2, -(mh+ma)*d, -ma*d;
          0, -(mh+ma)*d, mh+ma, ma;
          0, -ma*d, ma, ma];
    Minv_B = M \ [eye(3); zeros(1,3)];
end

function ok = pass(name, worst, tol, unit)
    ok = worst < tol;
    fprintf('%-38s worst %+.3e %-2s  tol %.0e   %s\n', ...
            name, worst, unit, tol, tern(ok, 'OK', 'FAIL'));
end

function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
