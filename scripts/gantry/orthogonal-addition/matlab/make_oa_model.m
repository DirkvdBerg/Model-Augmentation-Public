function make_oa_model(NS, REF)
% MAKE_OA_MODEL  Build gantry_oa.slx: the baseline Simulink model whose q1 plant is gantrySystemOA.
%
% Follows Matlab-scripts/Augmentation-cubic/make_cubic_model.m:
%   1. copyfile kamtin-fp-model/03 Simulink gantry/gantry_2025a.slx -> this folder (the source is
%      never loaded from its own path, so it cannot be written);
%   2. rewrite the ONE chart that calls gantrySystemMFile (the q1 plant) so it forwards to
%      gantrySystemOA with ten extra PARAMETERS (oa_Ma0 ... oa_b1) resolved from the base workspace;
%   3. widen the q1 loop's state from 6 to 6 + 2*NS: Integrator initial condition and Selector1
%      input width. Selector1 still picks [1 2 3], so q1 is unchanged in meaning;
%   4. OA-005: comment out the Simscape loop (q) and the Coriolis loop (q2), which share no
%      signal with q1 and are never read on the non-MSD branch. Their compile does not fit in RAM.
% NS  = number of absorber slots (default 4, OA-003). Unused slots are exact zeros.
% REF = true builds gantry_oa_ref.slx instead: same pruning, chart and state UNTOUCHED (the
%       original gantrySystem, 6 states). It exists only for Class B check B1.
% Run: matlab -batch "addpath('matlab'); make_oa_model(4); make_oa_model(4, true)"

if nargin < 1, NS = 4; end
if nargin < 2, REF = false; end
here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(fileparts(fileparts(here))));
addpath(fullfile(repo, 'kamtin-fp-model', '03 Simulink gantry', 'functions'));
addpath(here);

src_path = fullfile(repo, 'kamtin-fp-model', '03 Simulink gantry', 'gantry_2025a.slx');
if REF, dst_mdl = 'gantry_oa_ref'; else, dst_mdl = 'gantry_oa'; end
dst_path = fullfile(here, [dst_mdl '.slx']);
assert(isfile(src_path), 'source model not found: %s', src_path);
src_before = dir(src_path);

if bdIsLoaded(dst_mdl), close_system(dst_mdl, 0); end
if isfile(dst_path), delete(dst_path); end
[ok, msg] = copyfile(src_path, dst_path);
assert(ok, 'copy failed: %s', msg);
fileattrib(dst_path, '+w');
fprintf('copied %s\n    -> %s\n', src_path, dst_path);
load_system(dst_path);

% ==== 4. prune the two parallel plants (OA-005) ====
prune = {'Single H-gantry','Demux','Mux','Sum','Sum1','LTI System','Error','Feedforward', ...
         'Reference','To Workspace','Scope2', ...
         'MATLAB Function2','Integrator1','Selector2','Gain1','Gain2','Sum4','Sum5', ...
         'LTI System2','Error2','Feedforward2','Reference2','To Workspace2'};
keep_q1 = {'MATLAB Function1','Integrator','Selector1','Gain3','Gain4','Sum2','Sum3', ...
           'LTI System1','Error1','Feedforward1','Reference1','To Workspace1'};
for k = 1:numel(prune)
    set_param([dst_mdl '/' prune{k}], 'Commented', 'on');
end
% every root block is either pruned or on the q1 loop: nothing left half-connected
blks = find_system(dst_mdl, 'SearchDepth', 1, 'Type', 'Block');
names = get_param(blks(2:end), 'Name');
unknown = setdiff(names, [prune, keep_q1]);
assert(isempty(unknown), 'unclassified root blocks: %s', strjoin(unknown, ', '));
fprintf('pruned %d blocks (q and q2 loops); q1 loop keeps %d blocks\n', numel(prune), numel(keep_q1));
assert(strcmp(get_param([dst_mdl '/To Workspace1'], 'VariableName'), 'q1'));

if ~REF
    % ==== 2. rewrite the q1 chart ====
    rt = sfroot;
    machine = rt.find('-isa', 'Stateflow.Machine', 'Name', dst_mdl);
    charts  = machine.find('-isa', 'Stateflow.EMChart');
    hits = [];
    for k = 1:numel(charts)
        if contains(charts(k).Script, 'gantrySystemMFile'), hits(end+1) = k; end %#ok<AGROW>
    end
    assert(numel(hits) == 1, 'expected exactly one chart calling gantrySystemMFile, found %d', numel(hits));
    target = charts(hits(1));
    fprintf('chart: %s\n==== original script ====\n%s\n', target.Path, target.Script);
    sig = regexp(target.Script, 'function\s+dxdt\s*=\s*\w+\(([^)]*)\)', 'tokens', 'once');
    assert(~isempty(sig), 'could not parse the chart signature');
    extra = {'oa_Ma0','oa_Ma1','oa_Ma2','oa_Ca0','oa_Ka0','oa_Ka1','oa_Ka2','oa_m','oa_w','oa_b0','oa_b1'};
    new_script = sprintf([ ...
        'function dxdt = gantrySystemMFile(%s, %s)\n' ...
        'dxdt = zeros(size(x));\n' ...
        'dxdt = gantrySystemOA(u,x,m1,m2,mb,mh,Lb,Jb,Jh,d,cg1,cg2,cb1,cb2,cy,kb1,kb2, %s);\n' ...
        'end'], strtrim(sig{1}), strjoin(extra, ', '), strjoin(extra, ', '));
    target.Script = new_script;
    fprintf('==== new script ====\n%s\n', new_script);
    for k = 1:numel(extra)
        dat = target.find('-isa', 'Stateflow.Data', 'Name', extra{k});
        assert(~isempty(dat), 'data %s not created by the script edit', extra{k});
        dat.Scope = 'Parameter';
    end
    for nm = {'x', 'dxdt'}
        dat = target.find('-isa', 'Stateflow.Data', 'Name', nm{1});
        dat.Props.Array.Size = '-1';
        fprintf('  %s size "%s" (inherited)\n', nm{1}, dat.Props.Array.Size);
    end

    % ==== 3. widen the q1 loop state, found by connectivity, not by name alone ====
    integ = [dst_mdl '/Integrator'];
    sel   = [dst_mdl '/Selector1'];
    pc = get_param(integ, 'PortConnectivity');
    dsts = cellstr(get_param([pc(end).DstBlock], 'Name'));
    fprintf('  Integrator feeds: %s\n', strjoin(dsts, ', '));
    assert(any(strcmp(dsts, 'MATLAB Function1')) && any(strcmp(dsts, 'Selector1')), ...
           'Integrator is not the q1 loop integrator');
    ic_old = get_param(integ, 'InitialCondition');
    set_param(integ, 'InitialCondition', sprintf('[0;0;Y;0;0;0;zeros(%d,1)]', 2*NS));
    fprintf('  Integrator IC: %s -> %s\n', ic_old, get_param(integ, 'InitialCondition'));
    w_old = get_param(sel, 'InputPortWidth');
    set_param(sel, 'InputPortWidth', sprintf('%d', 6 + 2*NS));
    fprintf('  Selector1 width: %s -> %s, indices %s\n', w_old, get_param(sel, 'InputPortWidth'), ...
            get_param(sel, 'Indices'));
end

save_system(dst_mdl);
close_system(dst_mdl, 0);
src_after = dir(src_path);
assert(isequal(src_before.datenum, src_after.datenum) && src_before.bytes == src_after.bytes, ...
       'the read-only source model changed');
fprintf('saved %s (REF %d, NS %d); source untouched (bytes %d, datenum equal)\n', ...
        dst_path, REF, NS, src_after.bytes);
end
