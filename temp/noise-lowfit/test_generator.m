% TEST_GENERATOR  D-238 checks of the changed generator, written only into temp/noise-lowfit/gen_test.
%   (1) reproduction: TR-T1_r1 regenerated (both versions, replica) into gen_test; its noise-free
%       version must equal the stored noise-free record bitwise outside meta.generator
%   (2) pilot: the noisy version of TR-T1_r1 runs through the post-check (printed by the generator)
%   (3) standstill shape at Y 0, nonlinear replica (tanh friction on), one 12 s draw from the
%       generator's own tdg_noise_psd / tdg_noise_draw: simulated / measured rms over the fitted
%       range within 5 % per axis; figure gen_test/standstill_shape.png
%   Run from this folder: matlab -batch "test_generator"

here = fileparts(mfilename('fullpath'));
repo = fullfile(here, '..', '..');
dd = fullfile(repo, 'Thesis-writeup', 'Code', 'Data');
addpath(dd);  addpath(fullfile(repo, 'scripts', 'gantry', 'coulomb-tanh-gain'));
D = fullfile(here, 'gen_test');
if ~exist(D, 'dir'), mkdir(D); end
cfg = tdg_config();
fprintf('noise model %s, noisy folder name %s\n', cfg.noise_model, cfg.name_noisy);

% (1) + (2)
file = 'TR-T1_r1';
cd(dd);
generate_thesis_data({file}, struct('out_dir', D, 'simulator', 'replica'));
cd(here);
A = load(fullfile(cfg.out_free, 'Training', [file '.mat']));
B = load(fullfile(D, cfg.name_free, 'Training', [file '.mat']));
d = cmp(A, B, '');  d = d(~startsWith(d, 'meta.generator'));
if isempty(d)
    fprintf('(1) noise-free %s: PASS, bitwise equal outside meta.generator\n', file);
else
    fprintf('(1) noise-free %s: FAIL, differs in %s\n', file, strjoin(d, ', '));
end
N = load(fullfile(D, cfg.name_noisy, 'Training', [file '.mat']), 'meta');
fprintf('(2) noisy %s: post-check pass %d, scale %.2f, noise model %s, std v %s m\n', file, ...
        N.meta.post_check.pass, N.meta.scale_realised, N.meta.noise.model, mat2str(N.meta.noise.std_v_m, 3));

% (3) standstill shape, nonlinear replica at Y 0
base = {cfg.m1, cfg.m2, cfg.mb, cfg.mh_rigid, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
        cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, cfg.ma, cfg.ka, cfg.ca, cfg.L0};
odeT = @(x, u) gantrySystemExtendedTanh(u, x, base{:}, cfg.cc1, cfg.cc2, cfg.ccy, cfg.friction_gain);
plant0 = gtd_build_plant(0, cfg, 1);
[Ac, Bc, Cc, Dc] = ssdata(plant0.Cfb);
ctl = struct('A', Ac, 'B', Bc, 'C', Cc, 'D', Dc);
Nr = cfg.N_record;  fs = cfg.fs;
nz = tdg_noise_psd(0, cfg, plant0.Cfb);
v = tdg_noise_draw(nz, 20261006);
q = cl_sim(struct('r_sim', zeros(Nr, 3), 'f_sim', zeros(Nr, 3), 'v_enc', v), odeT, 0, ctl, cfg.P, cfg.ts);
e = -(q + v);
keep = cfg.n_hold + 1:Nr;
[Pe, fe] = pwelch(e(keep, :), hann(2048), 1024, 2048, fs);
T = load(cfg.noise_target);  f = T.f(2:end);  df = f(2) - f(1);
Pm = zeros(numel(f), 3);  for j = 1:3, Pm(:, j) = real(T.P(2:end, j, j)); end
Pe = Pe(2:end, :);
excl = {[], [], [700 2000]};  names = {'X1', 'X2', 'Y'};
fig = figure('Visible', 'off', 'Position', [100 100 820 900]);
tl = tiledlayout(3, 1, 'TileSpacing', 'compact');
for j = 1:3
    m = true(size(f));
    if ~isempty(excl{j}), m = f < excl{j}(1) | f > excl{j}(2); end
    r = sqrt(sum(Pe(m, j)) / sum(Pm(m, j)));
    fprintf('(3) %-2s standstill rms over the fitted range: measured %.2e, simulated %.2e, ratio %.3f -> %s\n', ...
            names{j}, sqrt(sum(Pm(m, j)) * df), sqrt(sum(Pe(m, j)) * df), r, ternary(abs(r - 1) <= 0.05, 'PASS', 'FAIL'));
    nexttile;
    loglog(f, sqrt(Pm(:, j)), 'k', 'LineWidth', 1.3); hold on;
    loglog(f, sqrt(Pe(:, j)), 'Color', [0.16 0.47 0.84], 'LineWidth', 1.2);
    xlim([9 1e4]);  ylim([1e-11 5e-10]);  grid on;  ylabel(sprintf('%s  [m/\\surdHz]', names{j}));
    legend({'measured', 'generator, nonlinear replica'}, 'Location', 'southwest', 'FontSize', 8);
end
xlabel(tl, 'frequency [Hz]');
title(tl, 'Generator check: standstill servo error at Y = 0, tanh friction on, one 12 s draw', 'FontSize', 10);
exportgraphics(fig, fullfile(D, 'standstill_shape.png'), 'Resolution', 150);
fprintf('test_generator: done\n');

function d = cmp(a, b, p)
% field paths where a and b differ (recursive, isequaln on leaves)
    d = {};
    if isstruct(a) && isstruct(b) && isscalar(a) && isscalar(b)
        fa = fieldnames(a);  fb = fieldnames(b);
        for k = 1:numel(fa)
            q = [p '.' fa{k}];  if isempty(p), q = fa{k}; end
            if ~isfield(b, fa{k}), d{end+1} = [q ' (missing)']; continue; end %#ok<AGROW>
            d = [d, cmp(a.(fa{k}), b.(fa{k}), q)]; %#ok<AGROW>
        end
        extra = setdiff(fb, fa);
        for k = 1:numel(extra), d{end+1} = [p '.' extra{k} ' (extra)']; end %#ok<AGROW>
    elseif ~isequaln(a, b)
        d = {p};
    end
end

function v = ternary(c, a, b)
    if c, v = a; else, v = b; end
end
