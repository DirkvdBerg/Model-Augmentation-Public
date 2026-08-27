function make_coulomb_model()
% MAKE_COULOMB_MODEL  Build the Coulomb-friction variant of the augmented model.
%
% Copies Matlab-scripts/Augmentation/gantry_additional_state_2025a.slx into this
% folder under a new name and edits the copy's "Extended ODE" chart so it calls
% gantrySystemExtendedCoulomb with three extra parameters, cc1, cc2, ccy.
%
% THE ORIGINAL MODEL IS NEVER OPENED FOR WRITING. It is loaded read-only and
% saved out under the new path; every subsequent edit targets the copy. This
% mirrors make_kxy_model.m, which is the proven route (b) from
% Matlab-scripts/Augmentation-kxy/README.md.
%
% Why the cc are PARAMETERS and not input ports: the chart resolves parameter
% scope from the base workspace, and gtd_run_simulation's push_params ALREADY
% pushes cc1, cc2, ccy (they have always been in gtd_config, they were simply
% never consumed). So no port wiring and no diagram change is needed.
%
% SOLVER CHANGE, and it is not cosmetic. The original model runs variable-step
% ode45 at RelTol 1e-4. sign() is discontinuous and the chart declares no
% zero-crossing signal, so ode45 would either crawl at the crossings or accept a
% badly-controlled step through them, and the output sample grid would depend on
% where those crossings landed. This copy is forced to FIXED-STEP ode4 (RK4) so
% that (i) the step is a knob we control, which is what makes the step-halving
% diagnostic in check_step_halving.m meaningful, (ii) the output grid is uniform
% so gtd_run_simulation's resample interpolation never fires, and (iii) the
% integrator matches the RK4 used on the Python side.
%
% Run from the repo root:
%   matlab -batch "addpath('Matlab-scripts/Augmentation'); addpath('Matlab-scripts/Augmentation-coulomb'); make_coulomb_model"

    here     = fileparts(mfilename('fullpath'));
    src_mdl  = 'gantry_additional_state_2025a';
    dst_mdl  = 'gantry_additional_state_coulomb_2025a';
    dst_path = fullfile(here, [dst_mdl '.slx']);

    FIXED_STEP = 5e-5;    % = 1/cfg.fs, the generator's sample period

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
    rt      = sfroot;
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
        error('make_coulomb_model:noChart', ...
              'No EM chart calling gantrySystemExtended found in %s', dst_mdl);
    end
    fprintf('  chart found: %s\n', target.Path);

    old_script = target.Script;
    fprintf('\n  --- original chart script ---\n%s\n', old_script);

    % ---- 3. Rewrite the script: add cc1/cc2/ccy, call the Coulomb function ---
    % Keep the existing signature verbatim and APPEND the three cc, so the 19
    % existing parameters keep their identity and scope. Only the call changes.
    sig_line = regexp(old_script, 'function\s+dxdt\s*=\s*\w+\(([^)]*)\)', 'tokens', 'once');
    if isempty(sig_line)
        error('make_coulomb_model:sig', 'Could not parse the chart function signature.');
    end
    args = strtrim(sig_line{1});

    % `ts` is appended alongside the three cc because the Karnopp stick band
    % V_EPS = (cc/m)*ts is sized from the integrator step. It resolves from the
    % base workspace for free: gtd_run_simulation's push_params already pushes
    % cfg.ts, so nothing under Augmentation/ has to change.
    new_script = sprintf([ ...
        'function dxdt = gantrySystemExtendedMFile(%s, cc1, cc2, ccy, ts)\n' ...
        'dxdt = zeros(8,1);\n' ...
        'dxdt = gantrySystemExtendedCoulomb(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, cg1, cg2, cb1, cb2, cy, kb1, kb2, ma, ka, ca, L0, cc1, cc2, ccy, ts);\n' ...
        'end'], args);

    target.Script = new_script;
    fprintf('  --- new chart script ---\n%s\n\n', new_script);

    % ---- 4. Force the new data to PARAMETER scope ---------------------------
    % Editing the signature makes Stateflow create each new name with the DEFAULT
    % scope, which is Input and would add unconnected ports. Demote to Parameter
    % so they resolve from the base workspace like the other 19.
    for nm = {'cc1', 'cc2', 'ccy', 'ts'}
        dat = target.find('-isa', 'Stateflow.Data', 'Name', nm{1});
        if isempty(dat)
            error('make_coulomb_model:noData', ...
                  '%s data object was not created by the script edit.', nm{1});
        end
        fprintf('  %s data created with scope "%s"', nm{1}, dat.Scope);
        dat.Scope = 'Parameter';
        fprintf('  ->  "%s"\n', dat.Scope);
    end

    % ---- 5. Force fixed-step ode4 (see the header) ---------------------------
    set_param(dst_mdl, 'SolverType', 'Fixed-step');
    set_param(dst_mdl, 'Solver',     'ode4');
    set_param(dst_mdl, 'FixedStep',  sprintf('%.12g', FIXED_STEP));
    fprintf('\n  solver: %s / %s, fixed step %s s\n', ...
            get_param(dst_mdl, 'SolverType'), get_param(dst_mdl, 'Solver'), ...
            get_param(dst_mdl, 'FixedStep'));

    % ---- 6. Report the resulting data inventory -----------------------------
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
    fprintf(['  EXPECTED: 2 input (u, x), 1 output (dxdt), '...
             '23 parameter (19 + cc1 + cc2 + ccy + ts)\n']);

    ok = (nIn == 2) && (nOut == 1) && (nPar == 23);
    fprintf('  structural check: %s\n', ternary(ok, 'PASS', 'FAIL'));

    save_system(dst_mdl);
    close_system(dst_mdl, 0);
    fprintf('\n  saved: %s\n', dst_path);

    if ~ok
        error('make_coulomb_model:structure', ...
              ['Chart data inventory is not as expected. The copy is on disk but ' ...
               'must not be used until this is understood.']);
    end
    fprintf('\nRESULT: PASS\n');
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end
