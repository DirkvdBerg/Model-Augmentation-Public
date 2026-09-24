function generate_oa_data(varargin)
% GENERATE_OA_DATA  Closed-loop dataset with the baseline truth plus an orthogonal addition (G3).
%
%   generate_oa_data('NAME', 'augmentation_oa_null')                          null control
%   generate_oa_data('NAME', 'augmentation_oa_xxx', 'ADDITION', 'design/x.mat') addition dataset
%   options: 'SELECT', {'T1','T3'}  exact record tags;  'PLOT', true
%
% Same records, seeds, references, multisine band and amplitude, and controller as
% augmentation_ma50_b140-230_a6_z03 (OA-003): gtd_config('augmentation', false, 0.50) with
% band [140 230] Hz and x6 amplitude re-applied. Differences: the plant is gantrySystemOA in the
% q1 loop of gantry_oa.slx (make_oa_model), and the hidden absorber of the production dataset is
% absent. ADDITION is a .mat holding oa_Ma0, oa_Ma1, oa_Ca0, oa_Ka0, oa_Ka1, oa_Ka2 (3x3) and
% oa_m, oa_w (1xNS), oa_b0, oa_b1 (3xNS); omitted = the null control (every term zero).
% Writes only to data/gantry/matlab/trajectory/<NAME>/ (new folder, refuses to overwrite).

NAME = ''; ADDITION = ''; SELECT = {}; PLOT = false; NS = 4;
for a = 1:2:numel(varargin)
    switch upper(varargin{a})
        case 'NAME',     NAME = varargin{a+1};
        case 'ADDITION', ADDITION = varargin{a+1};
        case 'SELECT',   SELECT = varargin{a+1};
        case 'PLOT',     PLOT = varargin{a+1};
        otherwise, error('generate_oa_data:opt', 'unknown option %s', varargin{a});
    end
end
assert(~isempty(NAME), 'NAME is required (a NEW dataset folder name)');

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(fileparts(here))));
addpath(genpath(fullfile(repo, 'Matlab-scripts', 'Augmentation')));
addpath(genpath(fullfile(repo, 'kamtin-fp-model', '03 Simulink gantry')));
addpath(here);
assert(exist('gtd_config', 'file') == 2, 'gtd_config not on the path');
assert(isfile(fullfile(here, 'gantry_oa.slx')), 'run make_oa_model first');

% ==== the addition ====
Z3 = zeros(3);
oa = struct('oa_Ma0',Z3,'oa_Ma1',Z3,'oa_Ma2',Z3,'oa_Ca0',Z3,'oa_Ka0',Z3,'oa_Ka1',Z3,'oa_Ka2',Z3, ...
            'oa_m',zeros(1,NS),'oa_w',ones(1,NS),'oa_b0',zeros(3,NS),'oa_b1',zeros(3,NS));
if ~isempty(ADDITION)
    if ~isfile(ADDITION), ADDITION = fullfile(fileparts(here), ADDITION); end
    L = load(ADDITION);
    fn = fieldnames(oa);
    for k = 1:numel(fn)
        if isfield(L, fn{k})
            assert(isequal(size(L.(fn{k})), size(oa.(fn{k}))), 'size mismatch in %s', fn{k});
            oa.(fn{k}) = double(L.(fn{k}));
        end
    end
end
is_null = all(structfun(@(v) all(v(:) == 0) || isequal(v, ones(1,NS)), oa));

% ==== config: the production dataset's knobs, non-MSD q1 truth ====
cfg = gtd_config('augmentation', false, 0.50);
cfg.f_low = 140;  cfg.f_high = 230;                 % BAND_OVERRIDE of the production dataset
AMP_SCALE = 6;                                      % AMP_SCALE of the production dataset
cfg.A_sym  = AMP_SCALE * cfg.A_sym;
cfg.A_Y    = AMP_SCALE * cfg.A_Y;
cfg.A_anti = 0.5 * cfg.A_sym * cfg.Lb;              % derived, as in generate_trajectory_data.m
cfg.mdl = 'gantry_oa';
ds_name = NAME;
if ~isempty(SELECT), ds_name = sprintf('%s_n%d', NAME, numel(SELECT)); end
cfg.out_dir = fullfile(repo, 'data', 'gantry', 'matlab', 'trajectory', ds_name);
cfg.fig_dir = fullfile(cfg.out_dir, 'figures');
assert(~isfolder(cfg.out_dir) || isempty(dir(fullfile(cfg.out_dir, '*.mat'))), ...
       'refusing to write: %s already holds .mat records', cfg.out_dir);
if ~exist(cfg.out_dir, 'dir'), mkdir(cfg.out_dir); end

fprintf('OA dataset generation\n  model   : %s (q1 loop = gantrySystemOA, %d absorber slots)\n', cfg.mdl, NS);
fprintf('  band    : %g to %g Hz, A_sym %.1f N, A_Y %.1f N, A_anti %.2f N m\n', ...
        cfg.f_low, cfg.f_high, cfg.A_sym, cfg.A_Y, cfg.A_anti);
fprintf('  out_dir : %s\n  addition: %s\n', cfg.out_dir, ternary(is_null, 'NONE (null control)', ADDITION));
fn = fieldnames(oa);
for k = 1:numel(fn), fprintf('    %-7s %s\n', fn{k}, mat2str(oa.(fn{k}), 6)); end

fid = fopen(fullfile(cfg.out_dir, 'PROVENANCE.txt'), 'w');
fprintf(fid, 'dataset   : %s\n', ds_name);
fprintf(fid, 'generator : scripts/gantry/orthogonal-addition/matlab/generate_oa_data.m\n');
fprintf(fid, 'model     : gantry_oa.slx (copy of kamtin gantry_2025a.slx, q1 chart -> gantrySystemOA)\n');
fprintf(fid, 'truth     : baseline gantrySystem.m + addition; the production hidden absorber is ABSENT\n');
fprintf(fid, 'addition  : %s\n', ternary(is_null, 'none (null control)', ADDITION));
for k = 1:numel(fn), fprintf(fid, '  %-7s %s\n', fn{k}, mat2str(oa.(fn{k}), 17)); end
fprintf(fid, 'knobs     : gtd_config(augmentation, use_msd=false, 0.50), band [140 230] Hz, amplitude x6\n');
fprintf(fid, 'x_aug     : delta_a = vdelta_a = 0 (not logged by the q1 loop; loader compatibility only)\n');
fprintf(fid, 'created   : %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fclose(fid);

records = gtd_build_records(cfg);
if ~isempty(SELECT)
    keep = false(1, numel(records));
    for k = 1:numel(records)
        keep(k) = any(strcmp(strtok(records(k).id, '_'), SELECT));
    end
    assert(any(keep), 'SELECT matched no record tag');
    records = records(keep);
end
fprintf('Generating %d records\n', numel(records));

for k = 1:numel(records)
    rec = records(k);
    t0 = tic;
    fprintf('\n=== %d/%d  %s  [%s] seed %d ===\n', k, numel(records), rec.id, rec.split, rec.seed);
    plant         = gtd_build_plant(rec.Y_op, cfg);
    [r, t]        = gtd_make_reference(rec, cfg);
    ms            = gtd_make_multisine(rec, plant, cfg);
    [f_safe, chk] = gtd_enforce_limits(plant, r, ms.f_stage, cfg);
    if chk.scale < 1
        fprintf('  scaled multisine to %.0f%% to meet limits\n', 100*chk.scale);
    end
    for j = 1:numel(fn), assignin('base', fn{j}, oa.(fn{j})); end
    out     = gtd_run_simulation(rec, r, t, f_safe, plant, cfg);
    out.amp = chk.scale * ms.A;
    oa_save_record(out, rec, cfg);
    if PLOT, gtd_plot_record(out, rec, cfg, false); end
    fprintf('  force peak [%.0f %.0f %.0f] N | force rms [%.1f %.1f %.1f] N | limiter scale %.3f | %.1f s\n', ...
            max(abs(out.u_total)), rms(out.u_total), chk.scale, toc(t0));
    fprintf('  saved %s.mat\n', rec.id);
end
fprintf('\nDone: %d records in %s\n', numel(records), cfg.out_dir);
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
