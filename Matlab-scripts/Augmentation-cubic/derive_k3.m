function out = derive_k3(D_TARGET)
% DERIVE_K3  Size the cubic spring constant BEFORE any dataset is generated.
%
%   out = DERIVE_K3()          uses the default target (10x the untrained
%                              baseline on the weakest validation record)
%   out = DERIVE_K3(D_TARGET)  target free-run sim-RMS [m] on that record
%
% WHY NOT T4's P3 RULE. T4 sized its spring by excitation preservation, because
% it put the spring in BOTH truth and model so no mismatch force ever existed.
% Here the spring is truth-only, so the ENTIRE spring force is a mismatch force
% acting on a K = 0 integrating axis, and the metric's sensitivity to it
% dominates. Sizing by P3 would land near 30 to 110 N/m^3 and drive the free run
% metres off, outside both the workspace and the range the ANN is trained on.
%
% THE LEVER (derived from the plant, not assumed). On an axis with mass m,
% damping c and zero stiffness, a force dF the model lacks obeys
% m*ddq + c*dq = dF in the open-loop free run, so terminal velocity is dF/c with
% time constant tau = m/c. Position error ramps LINEARLY at dF/c after about
% tau, reaching (dF/c)*(T - tau) by the end of a record of length T, and the RMS
% of a ramp is its endpoint over sqrt(3):
%
%       gain = (T - m/c) / (c * sqrt(3))     [metres of sim-RMS per newton]
%
% T5 measured the Y time constant at about 1 s against mh/cy = 1.01 s computed
% from the parameters, which is what makes this trustworthy rather than notional.
%
% ONLY THE DC PART OF THE FORCE RAMPS, and this is what the first version of
% this script got wrong. A record that sweeps Y symmetrically about zero has
% mean(k3*Y^3) close to zero, so it barely ramps however large its peak force
% is; a record parked off-centre has a constant force and ramps maximally. So
% the table below reports mean(F) (the part that sets the metric) separately
% from std(F) (the part that carries the SHAPE of the nonlinearity and is what
% makes k3 identifiable). Both matter, for different reasons.
%
% IT ALSO USES THE TRAJECTORY, NOT rec.Y_op. Y_op is the SCHEDULING operating
% point used to freeze M(Y) for the controller design, and it is 0 for every
% sweeping record even though those records reach |Y| = 0.3 m. Sizing from Y_op
% understates the exposure of two thirds of the record set.
%
% HEURISTIC, and labelled as such: the choice of "10x the untrained baseline on
% the weakest validation record" as the target. The gains above are derived; this
% target is an engineering judgement about how much headroom the experiment needs.
%
% Run from the repo root:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-cubic'); derive_k3"

    BASELINE_SIMRMS = 1.66e-4;   % untrained FP baseline, val sim-RMS [m]
    if nargin < 1 || isempty(D_TARGET)
        D_TARGET = 10 * BASELINE_SIMRMS;   % HEURISTIC: 10x headroom
    end

    THIS_DIR  = fileparts(mfilename('fullpath'));
    REPO_ROOT = fileparts(fileparts(THIS_DIR));
    addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
    addpath(THIS_DIR);

    cfg = gtd_config('augmentation', false, 0);   % USE_MSD = false: 6-state truth

    % --- per-axis mismatch-force sensitivity ---------------------------------
    mX = cfg.m1 + cfg.m2 + cfg.mb + cfg.mh;   % M(1,1) of gantrySystem
    cX = cfg.cg1 + cfg.cg2;                   % C(1,1)
    mY = cfg.mh;                              % M(3,3)
    cY = cfg.cy;                              % C(3,3)
    % The reported metric is an RMS over ALL THREE output channels (X1, X2, Y),
    % and the mismatch error lands almost entirely in one of them, so the metric
    % is smaller than the single-channel error by sqrt(NCH). Measured 2026-07-30:
    % without this factor the prediction overestimated by 1/0.577 on every
    % validation record (V1 0.55, V2 0.49, V4 0.58 of predicted). The first
    % sqrt(3) below is the RMS-of-a-ramp factor and is unrelated; they coincide
    % numerically only because there happen to be three channels.
    NCH = 3;
    gX = (cfg.t_record - mX/cX) / (cX * sqrt(3) * sqrt(NCH));
    gY = (cfg.t_record - mY/cY) / (cY * sqrt(3) * sqrt(NCH));

    fprintf('\n=== Mismatch-force sensitivity, per axis (derived from the plant) ===\n');
    fprintf('  X: m = %7.4g kg  c = %6.4g Ns/m  tau = %.4g s  gain = %.4g m/N\n', mX, cX, mX/cX, gX);
    fprintf('  Y: m = %7.4g kg  c = %6.4g Ns/m  tau = %.4g s  gain = %.4g m/N\n', mY, cY, mY/cY, gY);
    fprintf('  record length T = %g s. Only the DC part of the force ramps.\n', cfg.t_record);

    % --- per-record exposure at k3 = 1, then scale (degradation is linear) ----
    records = gtd_build_records(cfg);
    n = numel(records);
    [dcX, acX, dcY, acY, mxX, mxY] = deal(zeros(n,1));
    split = cell(n,1);

    for k = 1:n
        r = records(k);
        [rr, ~] = gtd_make_reference(r, cfg);   % STAGE coords [X1; X2; Y]
        % q_stage = P' * q_logical  =>  X = (X1+X2)/2, Y = ch3
        Xl = (rr(:,1) + rr(:,2)) / 2;
        Yl =  rr(:,3);
        fX = Xl.^3;    % force per unit k3
        fY = Yl.^3;
        dcX(k) = abs(mean(fX));  acX(k) = std(fX);  mxX(k) = max(abs(Xl));
        dcY(k) = abs(mean(fY));  acY(k) = std(fY);  mxY(k) = max(abs(Yl));
        split{k} = r.split;
    end

    % Predicted free-run sim-RMS degradation per unit k3, from the DC part only.
    deg_per_k3 = gX * dcX + gY * dcY;

    isval = strcmpi(split, 'val');
    cand  = deg_per_k3(isval & deg_per_k3 > 0);
    if isempty(cand)
        error('derive_k3:noval', ...
              ['No validation record carries a DC spring force. The cubic produces ' ...
               'no net force on a record centred at the origin, so this split cannot ' ...
               'be scored. Fix the record set before generating anything.']);
    end
    k3 = D_TARGET / min(cand);

    fprintf('\n=== k3 solve ===\n');
    fprintf('  target sim-RMS on the weakest val record : %.4g m   (HEURISTIC)\n', D_TARGET);
    fprintf('  weakest val degradation per unit k3      : %.4g m per (N/m^3)\n', min(cand));
    fprintf('  => k3                                    : %.4g N/m^3\n', k3);

    % --- the table -----------------------------------------------------------
    fprintf('\n=== Per-record exposure at k3 = %.4g N/m^3 (from the TRAJECTORY) ===\n', k3);
    fprintf('  DC ramps and sets the metric. AC carries the shape and is what makes k3 identifiable.\n\n');
    fprintf('  %-22s %-6s %7s %7s %11s %11s %11s %9s\n', ...
            'record', 'split', 'max|X|', 'max|Y|', 'DC_F[N]', 'AC_F[N]', 'simRMS[m]', 'vs base');
    deg = k3 * deg_per_k3;
    for k = 1:n
        fprintf('  %-22s %-6s %7.4f %7.4f %11.3e %11.3e %11.3e %8.1fx\n', ...
                records(k).id, split{k}, mxX(k), mxY(k), ...
                k3*(dcX(k)+dcY(k)), k3*(acX(k)+acY(k)), deg(k), deg(k)/BASELINE_SIMRMS);
    end

    % --- checks --------------------------------------------------------------
    excursion = max(deg) * sqrt(3);
    Fmax      = k3 * max(max(mxY).^3, max(mxX).^3);
    fprintf('\n=== Checks ===\n');
    fprintf('  worst predicted free-run excursion : %.4g m  (endpoint, = RMS*sqrt3)\n', excursion);
    fprintf('  Y travel limit (cfg.lim.pos_Y)     : %.4g m\n', cfg.lim.pos_Y);
    ok_ws = excursion <= cfg.lim.pos_Y;
    fprintf('  workspace check                    : %s\n', tf2s(ok_ws));
    if ~ok_ws
        fprintf(['      the model free run leaves the workspace, so it also leaves the\n' ...
                 '      range the ANN was trained on. Lower the target.\n']);
    end
    fprintf('  worst spring force                 : %.4g N\n', Fmax);
    fprintf('  Y multisine amplitude (cfg.A_Y)    : %.4g N\n', cfg.A_Y);
    fprintf('  force as %% of excitation           : %.3f %%  (T4''s P3 allowed 10%%)\n', ...
            100 * Fmax / cfg.A_Y);

    nAC = sum(k3*(acX + acY) > 0);
    nDC = sum(k3*(dcX + dcY) > 0);
    fprintf('  records carrying DC (score)        : %d of %d\n', nDC, n);
    fprintf('  records carrying AC (shape)        : %d of %d\n', nAC, n);
    fprintf('  val records carrying DC            : %d of %d\n', sum(isval & (dcX+dcY) > 0), sum(isval));

    out = struct('k3', k3, 'gain_X', gX, 'gain_Y', gY, 'D_target', D_TARGET, ...
                 'deg', deg, 'dcX', dcX, 'dcY', dcY, 'acX', acX, 'acY', acY, ...
                 'maxX', mxX, 'maxY', mxY, 'split', {split});
    fprintf('\nDone. Put the chosen k3 into generate_trajectory_data_cubic.m.\n\n');
end

function s = tf2s(b)
    if b, s = 'PASS'; else, s = 'FAIL'; end
end
