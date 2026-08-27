function make_kxy_model()
% MAKE_KXY_MODEL  Build the k_xy variant of the augmented Simulink model.
%
% Copies Matlab-scripts/Augmentation/gantry_additional_state_2025a.slx into this
% folder under a new name and edits the copy's "Extended ODE" chart so it calls
% gantrySystemExtendedKxy with one extra parameter, k_xy.
%
% THE ORIGINAL MODEL IS NEVER OPENED FOR WRITING. It is loaded read-only and
% saved out under the new path; every subsequent edit targets the copy.
%
% Why no port wiring is needed: inspection of the original chart XML shows the
% chart carries 1 output, 2 inputs (u, x) and 19 PARAMETER_DATA entries. Parameter
% scope resolves from the base workspace, which is how gtd_run_simulation's
% push_params already feeds the model. So k_xy is added as a 20th PARAMETER, not
% as an input port, and the Simulink diagram is structurally unchanged.
%
% Run from the repo root:
%   matlab -batch "addpath('Matlab-scripts/Augmentation'); addpath('Matlab-scripts/Augmentation-kxy'); make_kxy_model"

    here     = fileparts(mfilename('fullpath'));
    src_mdl  = 'gantry_additional_state_2025a';
    dst_mdl  = 'gantry_additional_state_kxy_2025a';
    dst_path = fullfile(here, [dst_mdl '.slx']);

    fprintf('Building %s\n', dst_path);

    if exist(dst_path, 'file')
        fprintf('  existing copy found, deleting it so this is reproducible\n');
        if bdIsLoaded(dst_mdl), close_system(dst_mdl, 0); end
        delete(dst_path);
    end

    % ---- 1. Load the ORIGINAL and save it out under the new name -------------
    load_system(src_mdl);
    save_system(src_mdl, dst_path);
    close_system(src_mdl, 0);          % discard: original untouched on disk
    fprintf('  copied and renamed -> %s\n', dst_mdl);

    % ---- 2. Locate the ODE chart in the COPY ---------------------------------
    load_system(dst_path);
    rt     = sfroot;
    machine = rt.find('-isa', 'Stateflow.Machine', 'Name', dst_mdl);
    charts  = machine.find('-isa', 'Stateflow.EMChart');

    target = [];
    for k = 1:numel(charts)
        if contains(charts(k).Script, 'gantrySystemExtended')
            target = charts(k);
            break
        end
    end
    if isempty(target)
        error('make_kxy_model:noChart', ...
              'No EM chart calling gantrySystemExtended found in %s', dst_mdl);
    end
    fprintf('  chart found: %s\n', target.Path);

    old_script = target.Script;
    fprintf('\n  --- original chart script ---\n%s\n', old_script);

    % ---- 3. Rewrite the script: add k_xy, call the Kxy function --------------
    % Keep the existing signature verbatim and APPEND k_xy, so the 19 existing
    % parameters keep their identity and scope. Only the forwarding call changes.
    sig_line = regexp(old_script, 'function\s+dxdt\s*=\s*\w+\(([^)]*)\)', 'tokens', 'once');
    if isempty(sig_line)
        error('make_kxy_model:sig', 'Could not parse the chart function signature.');
    end
    args = strtrim(sig_line{1});

    new_script = sprintf([ ...
        'function dxdt = gantrySystemExtendedMFile(%s, k_xy)\n' ...
        'dxdt = zeros(8,1);\n' ...
        'dxdt = gantrySystemExtendedKxy(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, cg1, cg2, cb1, cb2, cy, kb1, kb2, ma, ka, ca, L0, k_xy);\n' ...
        'end'], args);

    target.Script = new_script;
    fprintf('  --- new chart script ---\n%s\n\n', new_script);

    % ---- 4. Force the new k_xy data to PARAMETER scope -----------------------
    % Editing the signature makes Stateflow create k_xy with the DEFAULT scope,
    % which is Input and would add an unconnected port. Demote it to Parameter so
    % it resolves from the base workspace like the other 19.
    dat = target.find('-isa', 'Stateflow.Data', 'Name', 'k_xy');
    if isempty(dat)
        error('make_kxy_model:noData', ...
              'k_xy data object was not created by the script edit.');
    end
    fprintf('  k_xy data created with scope "%s"\n', dat.Scope);
    dat.Scope = 'Parameter';
    fprintf('  k_xy scope set to "%s"\n', dat.Scope);

    % ---- 5. Report the resulting data inventory -----------------------------
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
    fprintf('  EXPECTED: 2 input (u, x), 1 output (dxdt), 20 parameter (19 + k_xy)\n');

    ok = (nIn == 2) && (nOut == 1) && (nPar == 20);
    fprintf('  structural check: %s\n', ternary(ok, 'PASS', 'FAIL'));

    save_system(dst_mdl);
    close_system(dst_mdl, 0);
    fprintf('\n  saved: %s\n', dst_path);

    if ~ok
        error('make_kxy_model:structure', ...
              ['Chart data inventory is not as expected. The copy is on disk but ' ...
               'must not be used until this is understood.']);
    end
    fprintf('\nRESULT: PASS\n');
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end
