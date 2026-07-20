% make_coulomb_model.m
% --------------------
% One-shot: copy the FP Simulink model to gantry_2025a_coulomb.slx (proper
% rename via save_system, which updates the internal model name) and ENABLE
% the Coulomb blocks (set 'Commented' to 'off') so the Simscape subsystem
% applies dry friction.
%
% Why a script and not a file copy: a .slx stores its model name internally
% and it must match the filename; a raw byte copy + rename will not load.
% save_system does the rename correctly.
%
% The 6 Coulomb blocks are found by TYPE/gain (not by hard-coded path), so this
% is robust to the subsystem layout: the 3 Signum blocks and the 3 Gain blocks
% whose gain expression is cc1 / cc2 / ccy.
%
% Run from repo root (from MATLAB):
%   cd('<path-to>/Baseline-LPV-Augmentation')
%   run('Matlab-scripts/coulomb-friction/make_coulomb_model.m')

src_name = 'gantry_2025a';
dst_name = 'gantry_2025a_coulomb';
here = fileparts(mfilename('fullpath'));   % Matlab-scripts/coulomb-friction
repo = fileparts(fileparts(here));         % repo root (up two)
dst_file = fullfile(here, [dst_name '.slx']);

addpath(genpath(fullfile(repo, 'kamtin-fp-model', '03 Simulink gantry')));
addpath(here);

% ------------------------------------------------------------------
% 1. Copy + rename via save_system (updates the internal model name)
% ------------------------------------------------------------------
% Close any stale copies first.
if bdIsLoaded(dst_name), close_system(dst_name, 0); end
if bdIsLoaded(src_name), close_system(src_name, 0); end

load_system(src_name);
if isfile(dst_file)
    warning('Overwriting existing %s', dst_file);
    delete(dst_file);
end
save_system(src_name, dst_file);     % writes gantry_2025a_coulomb.slx
close_system(src_name, 0);
fprintf('Copied %s -> %s\n', src_name, dst_file);

% ------------------------------------------------------------------
% 2. Enable the 6 Coulomb blocks (find by type/gain, path-independent)
% ------------------------------------------------------------------
load_system(dst_name);

signums = find_system(dst_name, 'LookUnderMasks', 'on', 'FollowLinks', 'on', ...
                      'BlockType', 'Signum');

all_gains = find_system(dst_name, 'LookUnderMasks', 'on', 'FollowLinks', 'on', ...
                        'BlockType', 'Gain');
cc_gains = {};
for i = 1:numel(all_gains)
    gexpr = get_param(all_gains{i}, 'Gain');
    if any(strcmp(gexpr, {'cc1', 'cc2', 'ccy'}))
        cc_gains{end+1} = all_gains{i}; %#ok<AGROW>
    end
end

blocks = [signums(:); cc_gains(:)];
fprintf('\nEnabling %d Coulomb blocks:\n', numel(blocks));
for i = 1:numel(blocks)
    prev = get_param(blocks{i}, 'Commented');
    set_param(blocks{i}, 'Commented', 'off');
    fprintf('  [%s -> off]  %s\n', prev, blocks{i});
end

if numel(signums) ~= 3 || numel(cc_gains) ~= 3
    warning(['Expected 3 Signum + 3 cc gains, found %d Signum + %d cc gains. ' ...
             'Inspect the model before trusting the result.'], ...
            numel(signums), numel(cc_gains));
end

save_system(dst_name);
fprintf('\nSaved %s with Coulomb enabled.\n', dst_file);
fprintf(['\nNext: open the model and confirm the Coulomb path is wired (the ' ...
         'Signum -> cc-gain outputs feed the force sum), then run\n' ...
         '  run(''Matlab-scripts/coulomb-friction/run_coulomb_validation.m'')\n']);
