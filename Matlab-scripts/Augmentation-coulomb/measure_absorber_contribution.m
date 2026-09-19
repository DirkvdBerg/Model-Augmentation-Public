function measure_absorber_contribution(dataset, record, ma_frac, use_friction)
%MEASURE_ABSORBER_CONTRIBUTION  How much of y does the absorber actually produce?
%
% D-204 follow-up. The T3 probe measured that friction cuts delta_a (the
% absorber's INTERNAL state) to 14% of frictionless. That is not the quantity
% that decides whether the absorber is still recoverable: the relevant number is
% the absorber's contribution to the OUTPUT y, which is what the augmentation is
% fitted against and what the 0.12% figure from job 81088 refers to. Comparing a
% state-amplitude ratio against an output-contribution ratio is an error; this
% script measures the output contribution directly.
%
% METHOD, and why it needs no new model. The absorber's contribution is the
% difference between the plant WITH the real absorber and the same plant in the
% MASSLESS-ABSORBER LIMIT: ma -> 0 at FIXED total payload mass and FIXED modal
% frequency. Concretely ma_B = 1e-4 * mh_total with mh_rigid_B = mh_total - ma_B,
% and ka, ca derived from ma_B exactly as gtd_config derives them. The absorber's
% back-reaction on the gantry scales with ma, so at 1e-4 it is negligible, while
% mh_total, the 8-state structure, the mass-matrix conditioning and the friction
% block are all unchanged. The comparison is therefore a pure structural change.
%
% NOT DONE BY STIFFENING ka. The obvious alternative, locking the absorber with
% ka*1e6, was tried first and DIVERGED: it puts a 150 kHz mode into an explicit
% fixed-step RK4 running at 20 kHz, and the replay returned NaN with a singular
% mass matrix. Taking the mass to zero keeps the mode at 150 Hz and the problem
% non-stiff, which is why it is the right limit to take here.
%
% The RECORDED input is replayed OPEN LOOP through both plants. That is the right
% experiment: it asks what the absorber contributed to THIS trajectory, holding
% the excitation fixed. Closing the loop would let the controller react to the
% structural change and confound the answer.
%
% Integration is fixed-step RK4 at cfg.ts, matching the generator's ode4.
%
% Run:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-coulomb'); measure_absorber_contribution"

    THIS_DIR  = fileparts(mfilename('fullpath'));
    REPO_ROOT = fileparts(fileparts(THIS_DIR));
    addpath(genpath(fullfile(REPO_ROOT, 'Matlab-scripts', 'Augmentation')));
    addpath(THIS_DIR);

    if nargin < 1 || isempty(dataset),      dataset      = 'augmentation_ma50_b140-230_a6_z03_coulomb'; end
    if nargin < 2 || isempty(record),       record       = 'T3_standstill_Y000.mat'; end
    if nargin < 3 || isempty(ma_frac),      ma_frac      = 0.50; end
    if nargin < 4 || isempty(use_friction), use_friction = true; end

    MA_SMALL_FRAC = 1e-4;    % absorber mass as a fraction of mh_total in the B case
    NSTEP         = 120000;  % 6 s at 20 kHz; enough for a stationary rms

    cfg = gtd_config('augmentation', true, ma_frac);
    mh  = cfg.mh - cfg.ma;
    P   = [1, 1, 0; cfg.Lb/2, -cfg.Lb/2, 0; 0, 0, 1];
    if use_friction, CC = [16.80; 18.35; 11.60]; else, CC = [0; 0; 0]; end

    REC = fullfile(REPO_ROOT, 'data', 'gantry', 'matlab', 'trajectory', dataset, record);
    assert(isfile(REC), 'record not found: %s', REC);
    S = load(REC);
    N = min(NSTEP, size(S.u_total, 1));

    fprintf('\n=== absorber OUTPUT contribution ===\n');
    fprintf('record   %s / %s\n', dataset, record);
    fprintf('friction %s,  ma_frac %.2f,  %d steps (%.2f s)\n', ...
            string(use_friction), ma_frac, N, N*cfg.ts);

    mh_total = cfg.mh;                       % gtd_config's mh is the TOTAL payload
    wa       = 2*pi*cfg.fa;

    % A: the real absorber, exactly as generated
    maA = cfg.ma;  mhA = mh_total - maA;
    kaA = maA*wa^2;  caA = 2*cfg.zeta_a*sqrt(kaA*maA);

    % B: massless-absorber limit, same mh_total and same 150 Hz mode
    maB = MA_SMALL_FRAC*mh_total;  mhB = mh_total - maB;
    kaB = maB*wa^2;  caB = 2*cfg.zeta_a*sqrt(kaB*maB);

    fprintf('A: ma %.4f kg (mh_rigid %.4f)   B: ma %.3e kg (mh_rigid %.4f)\n', ...
            maA, mhA, maB, mhB);

    yA = replay(S, N, cfg, mhA, P, CC, maA, kaA, caA);   % real absorber
    yB = replay(S, N, cfg, mhB, P, CC, maB, kaB, caB);   % effectively none

    d = yA - yB;
    fprintf('\n%-8s %14s %14s %14s\n', 'channel', 'rms|y|', 'rms|contrib|', 'relative');
    for i = 1:3
        nm = {'X', 'Theta', 'Y'};
        fprintf('%-8s %14.4e %14.4e %13.3f%%\n', nm{i}, ...
                rms0(yA(:,i)), rms0(d(:,i)), 100*rms0(d(:,i))/rms0(yA(:,i)));
    end
    fprintf('\ntotal    %14.4e %14.4e %13.3f%%\n', ...
            rms0(yA(:)), rms0(d(:)), 100*rms0(d(:))/rms0(yA(:)));
end

% ===========================================================================

function y = replay(S, N, cfg, mh, P, CC, ma, ka, ca)
    args = {cfg.m1, cfg.m2, cfg.mb, mh, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
            cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
            ma, ka, ca, cfg.L0, CC(1), CC(2), CC(3), cfg.ts};
    f = @(x, u) gantrySystemExtendedCoulomb(u, x, args{:});
    h = cfg.ts;

    xl = double(S.x_logical(1,:)).';
    x  = [xl(1:3); double(S.delta_a(1)); xl(4:6); double(S.vdelta_a(1))];
    U  = (P * double(S.u_total(1:N,:)).').';     % stage -> logical forces
    y  = zeros(N, 3);
    for k = 1:N
        y(k,:) = (P.' * x(1:3)).';               % logical -> stage positions (q_stage = P'*q_logical)
        u  = U(k,:).';
        k1 = f(x, u);  k2 = f(x + (h/2)*k1, u);
        k3 = f(x + (h/2)*k2, u);  k4 = f(x + h*k3, u);
        x  = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
end

function r = rms0(v)
    r = sqrt(mean(v(:).^2));
end
