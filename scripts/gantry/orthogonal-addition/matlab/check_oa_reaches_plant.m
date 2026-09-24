function check_oa_reaches_plant(TEND)
% CHECK_OA_REACHES_PLANT  Class B gate (G3): the addition reaches the closed-loop q1 plant.
%   B1  gantry_oa with the addition OFF vs gantry_oa_ref (same OA-005 pruning, ORIGINAL chart), same
%       record, controller, reference and multisine: max |q1 difference| must be 0 (bit-identical).
%   B2  an X spring Ka0(1,1) = k at k and 2k: q1 differs, and the difference doubles (linear
%       regime, ratio in [1.9, 2.1]).
%   B3  one absorber slot active (m = 0.5 kg, 180 Hz, along Y at the payload): q1 differs.
% Records T3 (standstill) and T7 (fast Y sweep), first TEND seconds (default 1.0 s).
% The kamtin source model is never loaded here; its timestamp is asserted unchanged anyway.

if nargin < 1, TEND = 1.0; end
here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(fileparts(here))));
addpath(genpath(fullfile(repo, 'Matlab-scripts', 'Augmentation')));
addpath(genpath(fullfile(repo, 'kamtin-fp-model', '03 Simulink gantry')));
addpath(here);
src = fullfile(repo, 'kamtin-fp-model', '03 Simulink gantry', 'gantry_2025a.slx');
src_before = dir(src);

cfg = gtd_config('augmentation', false, 0.50);
cfg.f_low = 140;  cfg.f_high = 230;
cfg.A_sym = 6*cfg.A_sym; cfg.A_Y = 6*cfg.A_Y; cfg.A_anti = 0.5*cfg.A_sym*cfg.Lb;
cfg.out_dir = fullfile(here, '..', 'outputs', 'classB_cache');   % multisine cache only
cfg.fig_dir = fullfile(cfg.out_dir, 'figures');
NS = 4; Z3 = zeros(3);
off = struct('oa_Ma0',Z3,'oa_Ma1',Z3,'oa_Ma2',Z3,'oa_Ca0',Z3,'oa_Ka0',Z3,'oa_Ka1',Z3,'oa_Ka2',Z3, ...
             'oa_m',zeros(1,NS),'oa_w',ones(1,NS),'oa_b0',zeros(3,NS),'oa_b1',zeros(3,NS));

records = gtd_build_records(cfg);
tags = {'T3', 'T7'};
ok = true;
for tg = tags
    rec = records(strcmp(cellfun(@(s) strtok(s, '_'), {records.id}, 'UniformOutput', false), tg{1}));
    plant = gtd_build_plant(rec.Y_op, cfg);
    [r, t] = gtd_make_reference(rec, cfg);
    ms = gtd_make_multisine(rec, plant, cfg);
    [f, ~] = gtd_enforce_limits(plant, r, ms.f_stage, cfg);
    n = round(TEND / cfg.ts) + 1;
    r = r(1:n, :); t = t(1:n); f = f(1:n, :);

    q_orig = run(cfg, 'gantry_oa_ref', off, rec, r, t, f, plant);   % OA-005: pruned copy, original chart
    q_off  = run(cfg, 'gantry_oa',    off, rec, r, t, f, plant);
    d1 = max(abs(q_off(:) - q_orig(:)));
    fprintf('%s  B1 addition off vs gantry_oa_ref q1: max |dq| = %.3e m  (q rms %.3e)\n', ...
            rec.id, d1, rms(q_orig(:)));
    ok = ok && d1 == 0;

    k = 1e3;
    on = off; on.oa_Ka0(1,1) = k;
    q1 = run(cfg, 'gantry_oa', on, rec, r, t, f, plant);
    on.oa_Ka0(1,1) = 2*k;
    q2 = run(cfg, 'gantry_oa', on, rec, r, t, f, plant);
    e1 = max(abs(q1(:) - q_orig(:))); e2 = max(abs(q2(:) - q_orig(:)));
    fprintf('%s  B2 X spring k = %g: max |dq| %.3e;  2k: %.3e;  ratio %.4f\n', rec.id, k, e1, e2, e2/e1);
    ok = ok && e1 > 0 && e2/e1 > 1.9 && e2/e1 < 2.1;

    on = off; on.oa_m(1) = 0.5; on.oa_w(1) = 2*pi*180; on.oa_b0(:,1) = [0; -cfg.d; 1];
    q3 = run(cfg, 'gantry_oa', on, rec, r, t, f, plant);
    e3 = max(abs(q3(:) - q_orig(:)));
    fprintf('%s  B3 one absorber (0.5 kg, 180 Hz, Y at payload): max |dq| %.3e\n', rec.id, e3);
    ok = ok && e3 > 0;
end
src_after = dir(src);
same = isequal(src_before.datenum, src_after.datenum) && src_before.bytes == src_after.bytes;
fprintf('original model untouched: %d\n', same);
fprintf('\nCHECK_OA_REACHES_PLANT  %s\n', ternary(ok && same, 'PASS', 'FAIL'));
if ~(ok && same), error('check_oa_reaches_plant:fail', 'Class B failed'); end
end

function q = run(cfg, mdl, oa, rec, r, t, f, plant)
fn = fieldnames(oa);
for j = 1:numel(fn), assignin('base', fn{j}, oa.(fn{j})); end
c = cfg; c.mdl = mdl;
t0 = tic;
out = gtd_run_simulation(rec, r, t, f, plant, c);
q = out.q_with;
fprintf('    [%s, %.2f s simulated in %.1f s wall]\n', mdl, t(end), toc(t0));
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
