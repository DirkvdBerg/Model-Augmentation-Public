function make_cubic_model()
% MAKE_CUBIC_MODEL  Build the cubic-spring variant of the BASELINE Simulink model.
%
% Copies kamtin-fp-model/03 Simulink gantry/gantry_2025a.slx into this folder
% under a new name and edits the copy's "MATLAB Function1" chart so it calls
% gantrySystemCubic with one extra parameter, k3.
%
% THE ORIGINAL MODEL IS NEVER OPENED FOR WRITING, and kamtin-fp-model/ is read
% only. It is loaded, saved out under the new path, then closed with changes
% discarded; every subsequent edit targets the copy.
%
% WHICH CHART, AND WHY IT MATTERS. gantry_2025a runs THREE plants in parallel
% and logs them separately:
%   Single H-gantry (Simscape Multibody)        -> q
%   MATLAB Function1 -> gantrySystemMFile       -> q1   <-- THIS ONE
%   MATLAB Function2 -> ...CoriolisCentripetal  -> q2
% gtd_run_simulation.m reads q1 on the non-MSD branch, so the truth that ends up
% in the .mat records is the plain gantrySystem ODE. Selecting the chart by the
% substring 'gantrySystem' alone would match BOTH charts, because
% "gantrySystemCoriolisCentripetalMFile" also contains it. Match on
% 'gantrySystemMFile' and assert exactly one hit.
%
% Why no port wiring is needed: the chart carries 1 output, 2 inputs (u, x) and
% its parameters resolve from the base workspace, which is how
% gtd_run_simulation's push_params already feeds the model. So k3 is added as
% one more PARAMETER, not as an input port, and the diagram is structurally
% unchanged.
%
% NOTE ON ARGUMENT ORDER. The chart wrapper's own signature order differs from
% gantrySystem's: the wrapper takes (u,x,m1,m2,mb,mh,Jb,Jh,d,Lb,kb1,kb2,cg1,
% cg2,cb1,cb2,cy) and reorders when calling. The wrapper signature is preserved
% verbatim and k3 is APPENDED, so the existing parameters keep their identity
% and scope; only the forwarding call changes.
%
% Run from the repo root:
%   matlab -batch "addpath('Matlab-scripts/Augmentation-cubic'); make_cubic_model"

    here      = fileparts(mfilename('fullpath'));
    repo_root = fileparts(fileparts(here));
    addpath(genpath(fullfile(repo_root, 'kamtin-fp-model', '03 Simulink gantry')));
    addpath(here);

    src_mdl  = 'gantry_2025a';
    dst_mdl  = 'gantry_2025a_cubic';
    dst_path = fullfile(here, [dst_mdl '.slx']);

    fprintf('Building %s\n', dst_path);

    if isfile(dst_path)
        fprintf('  existing copy found, deleting it so this is reproducible\n');
        if bdIsLoaded(dst_mdl), close_system(dst_mdl, 0); end
        delete(dst_path);
    end

    % ---- 1. Copy the ORIGINAL at the FILESYSTEM level ------------------------
    % Deliberately NOT load_system + save_system, which is what make_kxy_model
    % does. That pattern opens the source model in memory, and kamtin-fp-model/
    % is READ ONLY by project policy. copyfile never opens it for writing at
    % all, so there is no path by which this script can modify the source.
    src_path = fullfile(repo_root, 'kamtin-fp-model', '03 Simulink gantry', [src_mdl '.slx']);
    % isfile, NOT exist(...,'file') == 2: exist returns 4 for Simulink models,
    % so the == 2 form rejects a file that is plainly there.
    assert(isfile(src_path), 'Source model not found: %s', src_path);
    src_info_before = dir(src_path);

    [ok, msg] = copyfile(src_path, dst_path);
    assert(ok, 'Could not copy the model: %s', msg);
    fileattrib(dst_path, '+w');        % the copy must be writable even if the source is not
    fprintf('  copied -> %s\n', dst_path);

    % ---- 2. Locate the RIGHT ODE chart in the COPY ---------------------------
    load_system(dst_path);
    rt      = sfroot;
    machine = rt.find('-isa', 'Stateflow.Machine', 'Name', dst_mdl);
    charts  = machine.find('-isa', 'Stateflow.EMChart');

    hits = [];
    for k = 1:numel(charts)
        if contains(charts(k).Script, 'gantrySystemMFile')
            hits(end+1) = k; %#ok<AGROW>
        end
    end
    if numel(hits) ~= 1
        error('make_cubic_model:chart', ...
              ['Expected exactly ONE chart calling gantrySystemMFile, found %d. ' ...
               'Do not guess: inspect the model before continuing.'], numel(hits));
    end
    target = charts(hits(1));
    fprintf('  chart found: %s\n', target.Path);

    old_script = target.Script;
    fprintf('\n  --- original chart script ---\n%s\n', old_script);

    % ---- 3. Rewrite the script: append k3, call the Cubic function -----------
    sig = regexp(old_script, 'function\s+dxdt\s*=\s*\w+\(([^)]*)\)', 'tokens', 'once');
    if isempty(sig)
        error('make_cubic_model:sig', 'Could not parse the chart function signature.');
    end
    args = strtrim(sig{1});

    new_script = sprintf([ ...
        'function dxdt = gantrySystemMFile(%s, k3)\n' ...
        'dxdt = zeros(6,1);\n' ...
        'dxdt = gantrySystemCubic(u,x,m1,m2,mb,mh,Lb,Jb,Jh,d,cg1,cg2,cb1,cb2,cy,kb1,kb2,k3);\n' ...
        'end'], args);

    target.Script = new_script;
    fprintf('  --- new chart script ---\n%s\n\n', new_script);

    % ---- 4. Force the new k3 data to PARAMETER scope -------------------------
    % Editing the signature makes Stateflow create k3 with the DEFAULT scope,
    % which is Input and would add an unconnected port. Demote it to Parameter
    % so it resolves from the base workspace like the others.
    dat = target.find('-isa', 'Stateflow.Data', 'Name', 'k3');
    if isempty(dat)
        error('make_cubic_model:noData', 'k3 data object was not created by the script edit.');
    end
    fprintf('  k3 data created with scope "%s"\n', dat.Scope);
    dat.Scope = 'Parameter';
    fprintf('  k3 scope set to "%s"\n', dat.Scope);

    % ---- 5. Report the resulting data inventory ------------------------------
    % ASSERTED, not assumed: the parameter count is read from the model rather
    % than hard-coded, and only the structure is required to be consistent.
    alldata = target.find('-isa', 'Stateflow.Data');
    nIn = 0; nOut = 0; nPar = 0; other = {};
    for k = 1:numel(alldata)
        switch alldata(k).Scope
            case 'Input',     nIn  = nIn  + 1;
            case 'Output',    nOut = nOut + 1;
            case 'Parameter', nPar = nPar + 1;
            otherwise,        other{end+1} = alldata(k).Scope; %#ok<AGROW>
        end
    end
    fprintf('\n  chart data inventory: %d input, %d output, %d parameter', nIn, nOut, nPar);
    if ~isempty(other), fprintf(', other: %s', strjoin(unique(other), ',')); end
    fprintf('\n');
    fprintf('  EXPECTED: 2 input (u, x), 1 output (dxdt), and k3 among the parameters\n');

    has_k3 = ~isempty(target.find('-isa', 'Stateflow.Data', 'Name', 'k3'));
    ok = (nIn == 2) && (nOut == 1) && has_k3 && isempty(other);
    fprintf('  structural check: %s\n', ternary(ok, 'PASS', 'FAIL'));

    save_system(dst_mdl);
    close_system(dst_mdl, 0);
    fprintf('\n  saved: %s\n', dst_path);

    % ---- 6. Prove the read-only source was not touched -----------------------
    src_info_after = dir(src_path);
    src_untouched  = isequal(src_info_before.datenum, src_info_after.datenum) && ...
                     isequal(src_info_before.bytes,   src_info_after.bytes);
    fprintf('  source model untouched (kamtin-fp-model is READ ONLY): %s\n', ...
            ternary(src_untouched, 'PASS', 'FAIL'));
    if ~src_untouched
        error('make_cubic_model:sourceModified', ...
              ['The source model under kamtin-fp-model/ changed. That tree is read only. ' ...
               'Restore it from git before doing anything else.']);
    end

    if ~ok
        error('make_cubic_model:structure', ...
              ['Chart data inventory is not as expected. The copy is on disk but ' ...
               'must not be used until this is understood.']);
    end
    fprintf('\nRESULT: PASS\n');
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end
