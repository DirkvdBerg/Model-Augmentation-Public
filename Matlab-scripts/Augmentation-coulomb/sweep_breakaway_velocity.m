function sweep_breakaway_velocity(dataset, record, ma_frac, V_BRK_LIST)
%SWEEP_BREAKAWAY_VELOCITY  D-209: size the stick band from the DATA, not from a quote.
%
% THE QUESTION. The 2026-09-21 meeting figures show friction changing sign on
% rails that are standing still, on the Telica profiles. The current band is
%
%     v_eps = V_EPS_MARGIN*(cc1+cc2)/m_total*ts = 2.69e-04 m/s
%
% a per-step velocity increment times a margin D-204 measured on the 140-230 Hz
% multisine. The Telica records carry NO multisine, so the rails dwell near zero
% far longer, their residual ripple exceeds that band, they are classified as
% SLIDING, and cc*sign(v) flips on controller noise. D-209 replaces the band
% with a physical breakaway velocity V_BRK. This sweep picks its value.
%
% CRITERION, and it is deliberately two-sided. The band is a DETECTION
% threshold, so every unit of it declares more rails stuck than the physics
% requires. So we report both:
%   BENEFIT  near-zero sign flips of the friction force (the artefact), and
%   COST     stick occupancy (how much of the record the band freezes).
% The recommendation is the SMALLEST V_BRK that removes the flipping. A larger
% one is not "safer": it buys nothing and freezes more of the record.
%
% WHY THE DIAGNOSTIC WINDOW IS FIXED AND THE BAND IS NOT. Flips are counted
% inside |v| < V_REF with V_REF held CONSTANT across the sweep and set above
% every candidate band. If the counting window moved with the knob, the metric
% would measure the knob instead of the artefact and every value would look
% equally good.
%
% HOW THE BAND IS SET WITHOUT TOUCHING THE ODE. `ts` enters
% gantrySystemExtendedCoulomb.m at exactly one place, the v_eps line (its own
% docstring: "ts integrator step [s], used ONLY to size the stick band V_EPS";
% verified by grep, `ts` appears only in the signature, the docstring, one
% comment and line 180). So passing
%
%     ts_eff = V_BRK * m_total / (V_EPS_MARGIN*(cc1+cc2))
%
% yields v_eps = V_BRK EXACTLY, and the sweep runs against the unmodified
% production ODE. Nothing here restates the friction law, which is the same
% discipline coulomb_v_eps.m exists to enforce: a gate that keeps its own copy
% of the formula stops testing the thing it names (D-204 amendment).
%
% HOW THE FRICTION FORCE IS RECOVERED. Exactly as check_friction_invariants.m
% (gate 2) does it: call the ODE twice at the same (x,u), once with Garcia's cc
% and once with cc = 0. The only difference is the friction, and it enters as
% dxdt = A*x + B*(u - P*F), so a_free - a_cc = Minv_B*(P*F), solved by least
% squares. Nothing inside the friction block is inspected; the force is inferred
% from its effect.
%
% WHAT THIS SWEEP IS NOT. It reclassifies the states of an EXISTING record,
% generated at the old band. It therefore answers "does this band still see a
% spurious sign change at these states", which is the artefact. It does NOT
% answer "what trajectory would this band produce", because that needs a
% regeneration per value. Take the recommended V_BRK from here, then regenerate
% ONE dataset and re-run gates A1, 2 and 3 to confirm. Stage 2 is not optional.
%
% Run:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-coulomb'); sweep_breakaway_velocity"

    THIS_DIR  = fileparts(mfilename('fullpath'));
    REPO_ROOT = fileparts(fileparts(THIS_DIR));
    addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
    addpath(THIS_DIR);

    if nargin < 1 || isempty(dataset),    dataset = 'augmentation_ma50_z03_telica_coulomb'; end
    if nargin < 2 || isempty(record),     record  = 'TP1_telica_y000.mat';                  end
    if nargin < 3 || isempty(ma_frac),    ma_frac = 0.50;                                   end
    if nargin < 4 || isempty(V_BRK_LIST), V_BRK_LIST = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2];      end

    % Fixed diagnostic window. Above every candidate band by design, so the
    % counting window never moves with the knob.
    V_REF  = 1e-2;    % m/s, "near zero" for flip counting
    V_DEEP = 1e-4;    % m/s, the deep-dwell subset
    STRIDE = 1;       % flips are a sample-to-sample quantity; do not subsample

    REC = fullfile(REPO_ROOT, 'data', 'gantry', 'matlab', 'trajectory', dataset, record);
    assert(isfile(REC), 'record not found: %s', REC);

    cfg = gtd_config('augmentation', true, ma_frac);
    CC  = [16.80; 18.35; 11.60];          % THEORY: garcia2013, identified. NOT a knob.
    mh  = cfg.mh - cfg.ma;
    P   = [1, 1, 0; cfg.Lb/2, -cfg.Lb/2, 0; 0, 0, 1];

    base = {cfg.m1, cfg.m2, cfg.mb, mh, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
            cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
            cfg.ma, cfg.ka, cfg.ca, cfg.L0};
    argsFric = [base, {0, 0, 0, cfg.ts}];              % cc = 0, band irrelevant

    m_total = cfg.m1 + cfg.m2 + cfg.mb + mh + cfg.ma;
    [v_eps_now, margin] = coulomb_v_eps(CC(1), CC(2), m_total, cfg.ts);

    % ts that makes v_eps land exactly on each requested V_BRK
    ts_of = @(vbrk) vbrk * m_total / (margin * (CC(1) + CC(2)));

    S = load(REC);
    N = size(S.u_total, 1);
    idx = 1:STRIDE:N;
    nb  = numel(V_BRK_LIST);

    fprintf('\n=== D-209 breakaway-velocity sweep ===\n');
    fprintf('record        : %s\n', REC);
    fprintf('samples       : %d (stride %d)\n', numel(idx), STRIDE);
    fprintf('current band  : %.4e m/s  (V_EPS_MARGIN = %g, ts = %.1e)\n', v_eps_now, margin, cfg.ts);
    fprintf('flip window   : |v| < %.1e m/s (fixed), deep dwell |v| < %.1e m/s\n', V_REF, V_DEEP);
    fprintf('candidates    : %s m/s\n\n', mat2str(V_BRK_LIST, 3));

    % per band, per rail counters
    n_stick   = zeros(nb, 3);   % samples classified stuck
    flips_ref = zeros(nb, 3);   % sign flips of F with |v| < V_REF
    flips_sl  = zeros(nb, 3);   % ... and BOTH samples classified sliding (the artefact)
    flips_dp  = zeros(nb, 3);   % ... with |v| < V_DEEP
    n_ref     = zeros(1, 3);    % samples in the fixed window (band-independent)
    n_deep    = zeros(1, 3);
    chatter   = zeros(nb, 3);   % max |F(k) - F(k-1)|

    prevF     = nan(nb, 3);
    prevStuck = false(nb, 3);

    t0 = tic;
    for kk = 1:numel(idx)
        t  = idx(kk);
        xl = double(S.x_logical(t,:)).';            % [X;Theta;Y; dX;dTheta;dY]
        x  = [xl(1:3); double(S.delta_a(t)); xl(4:6); double(S.vdelta_a(t))];
        % u_total is in STAGE coordinates; the ODE takes logical forces.
        % gtd_save_record uses q_logical = (P')\q_stage, so power equivalence
        % gives u_logical = P*u_stage.
        u  = P * double(S.u_total(t,:)).';

        dFree   = gantrySystemExtendedCoulomb(u, x, argsFric{:});   % cc = 0, band-independent
        Minv_B  = minv_b(x, cfg, mh);
        v_stage = P.' * x(5:7);

        for i = 1:3
            if abs(v_stage(i)) < V_REF,  n_ref(i)  = n_ref(i)  + 1; end
            if abs(v_stage(i)) < V_DEEP, n_deep(i) = n_deep(i) + 1; end
        end

        for b = 1:nb
            argsC = [base, {CC(1), CC(2), CC(3), ts_of(V_BRK_LIST(b))}];
            dC = gantrySystemExtendedCoulomb(u, x, argsC{:});
            PF = Minv_B \ (dFree(5:8) - dC(5:8));    % least squares, 4x3
            F  = P \ PF;

            for i = 1:3
                stuck_i = abs(v_stage(i)) < V_BRK_LIST(b);
                if stuck_i, n_stick(b,i) = n_stick(b,i) + 1; end

                if ~isnan(prevF(b,i))
                    chatter(b,i) = max(chatter(b,i), abs(F(i) - prevF(b,i)));
                    flipped = (sign(F(i)) ~= sign(prevF(b,i))) ...
                              && F(i) ~= 0 && prevF(b,i) ~= 0;
                    if flipped && abs(v_stage(i)) < V_REF
                        flips_ref(b,i) = flips_ref(b,i) + 1;
                        if ~stuck_i && ~prevStuck(b,i)
                            flips_sl(b,i) = flips_sl(b,i) + 1;
                        end
                        if abs(v_stage(i)) < V_DEEP
                            flips_dp(b,i) = flips_dp(b,i) + 1;
                        end
                    end
                end
                prevF(b,i)     = F(i);
                prevStuck(b,i) = stuck_i;
            end
        end

        if mod(kk, 20000) == 0
            fprintf('  ... %6d / %6d states  (%.0f s elapsed)\n', kk, numel(idx), toc(t0));
        end
    end

    names = {'X1', 'X2', 'Y'};
    fprintf('\nsamples inside the fixed windows (band-independent):\n');
    for i = 1:3
        fprintf('  %-3s |v|<%.0e : %7d (%5.2f%%)   |v|<%.0e : %7d (%5.2f%%)\n', ...
                names{i}, V_REF, n_ref(i), 100*n_ref(i)/numel(idx), ...
                V_DEEP, n_deep(i), 100*n_deep(i)/numel(idx));
    end

    fprintf('\n%-10s %-4s %10s %10s %10s %12s %12s\n', ...
            'V_BRK', 'rail', 'flips<Vref', 'of-which', 'flips<Vdeep', 'stick occ', 'max dF');
    fprintf('%-10s %-4s %10s %10s %10s %12s %12s\n', ...
            '[m/s]', '', '[-]', 'SLIDING', '[-]', '[%]', '[N]');
    for b = 1:nb
        for i = 1:3
            fprintf('%-10.1e %-4s %10d %10d %10d %11.2f%% %12.4f\n', ...
                    V_BRK_LIST(b), names{i}, flips_ref(b,i), flips_sl(b,i), ...
                    flips_dp(b,i), 100*n_stick(b,i)/numel(idx), chatter(b,i));
        end
    end

    % Recommendation: smallest band with zero SLIDING flips in the deep dwell.
    % Sliding flips are the artefact; a stuck rail's force is solved and may
    % legitimately change sign as the applied force does.
    rec = NaN;
    for b = 1:nb
        if all(flips_sl(b,:) == 0)
            rec = V_BRK_LIST(b);
            break
        end
    end
    fprintf('\n');
    if isnan(rec)
        fprintf('NO candidate removed the sliding sign flips. Extend V_BRK_LIST upward,\n');
        fprintf('and consider that some of the flipping may be REAL stick-slip: our\n');
        fprintf('f/F_C is 0.86/1.11/0.86, so the machine fails the stick-slip inequality\n');
        fprintf('v_brk >= (F_brk-F_C)/f by decades for any F_brk > F_C (D-209).\n');
    else
        fprintf('RECOMMENDED V_BRK = %.1e m/s  (smallest with zero SLIDING flips)\n', rec);
        fprintf('  vs current band %.2e m/s  -> factor %.1f\n', v_eps_now, rec/v_eps_now);
        fprintf('  published Stribeck-velocity range 1e-5 to 1e-2 m/s (lee2020feeddrive):  %s\n', ...
                ternary(rec >= 1e-5 && rec <= 1e-2, 'INSIDE', 'OUTSIDE, report this'));
    end
    fprintf('\nNEXT (not optional): set V_BRK in gantrySystemExtendedCoulomb.m, update\n');
    fprintf('coulomb_v_eps.m, regenerate ONE Telica record, then re-run gates A1, 2 and 3.\n');
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end

function Minv_B = minv_b(x, cfg, mh)
% The B-block of the ODE, rebuilt here so the recovery does not depend on any
% internal of the friction block. Same M(Y, delta_a) as
% gantrySystemExtendedCoulomb builds.
    Y = x(3); da = x(4);
    m1 = cfg.m1; m2 = cfg.m2; mb = cfg.mb; ma = cfg.ma;
    Lb = cfg.Lb; Jb = cfg.Jb; Jh = cfg.Jh; d = cfg.d; L0 = cfg.L0;
    M = [ m1+m2+mb+mh+ma,  (m1-m2)*Lb/2 - (mh+ma)*Y - ma*L0 - ma*da,  0,  0;
         (m1-m2)*Lb/2 - (mh+ma)*Y - ma*L0 - ma*da, ...
           Jb+Jh+(m1+m2)*Lb^2/4 + (mh+ma)*d^2 + mh*Y^2 + ma*(Y+L0+da)^2, -(mh+ma)*d, -ma*d;
          0, -(mh+ma)*d,  mh+ma,  ma;
          0, -ma*d,       ma,     ma];
    Minv_B = M \ [eye(3); zeros(1,3)];
end
