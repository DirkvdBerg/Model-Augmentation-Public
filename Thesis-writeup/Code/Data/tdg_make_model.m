function tdg_make_model()
% TDG_MAKE_MODEL  Build model/gantry_tdg_2025a.slx from the vendored coulomb model.
%
% Source: vendor/gantry_additional_state_coulomb_2025a.slx, a byte copy of
% Matlab-scripts/Augmentation-coulomb/ (VENDORED.md). The source is loaded, saved
% under the new name and only the COPY is edited. Four edits:
%
%   1. Keep only the recorded loop. The source integrates four independent closed
%      loops side by side (Simscape "Single H-gantry", MATLAB Function1 =
%      gantrySystem, MATLAB Function2 = gantrySystemCoriolisCentripetal, and the
%      "Extended ODE" loop). The generator records only the Extended ODE loop
%      (q_aug, delta_a_ode; gtd_run_simulation.m). The other three share no signal
%      with it, so deleting them cannot change it; check_model_equivalence.m proves
%      that bitwise. Reason: runtime (58 s per 12 s run with all four loops) and
%      self-containment (no Simscape Multibody dependency).
%   2. Encoder noise (D-225, NOISE-INJECTION.md; replaces D-218's noise force on Sum2):
%          Gain4 (P^T, stage q) -> Sum3 (r - q)      becomes
%          Gain4 -> EncoderSum (++) -> Sum3,  EncoderNoise (From Workspace v_in) -> EncoderSum
%      only on the branch into Sum3: the controller sees q + v, while To Workspace1
%      (q_aug) still logs the true q. v_in = [t, v] in STAGE coordinates [X1, X2, Y] [m],
%      interpolation off, sample time ts. v = 0 makes the edit an exact no-op.
%   3. Plant-force log: To Workspace "u_plant" on the Sum2 output (u_fb + u_ff, the force
%      the plant receives), so the saved u_total (reconstructed in gtd_run_simulation as
%      lsim(Cfb, r - y) + f with y = q + v) can be checked: u_plant = u_total proves that
%      the saved y is the signal the controller saw.
%   4. Friction law (D-224): the Extended ODE chart calls gantrySystemExtendedTanh with
%      the chart Parameter g_fric (cfg.friction_gain, pushed by gtd_run_simulation)
%      instead of the vendored Karnopp gantrySystemExtendedCoulomb. Its other arguments
%      are unchanged; ts stays a (now unused) chart Parameter so the data list keeps its
%      order. D-223 verified the tanh ODE in a bitwise replica of this loop.
    cfg = tdg_config();
    src = fullfile(cfg.here, 'vendor', [cfg.mdl_orig '.slx']);
    if ~exist(cfg.mdl_dir, 'dir'), mkdir(cfg.mdl_dir); end
    dst = fullfile(cfg.mdl_dir, [cfg.mdl '.slx']);
    if bdIsLoaded(cfg.mdl), close_system(cfg.mdl, 0); end
    if exist(dst, 'file'), delete(dst); end
    if bdIsLoaded(cfg.mdl_orig), close_system(cfg.mdl_orig, 0); end
    load_system(src);
    save_system(cfg.mdl_orig, dst);          % writes the copy; source file untouched
    close_system(cfg.mdl_orig, 0);
    load_system(dst);
    m = cfg.mdl;

    % == 1. keep only the Extended ODE loop ==
    keep = {'Reference1','Sum3','LTI System1','Feedforward1','Sum2','Gain3', ...
            'Extended ODE','Integrator','Selector1','Gain4','To Workspace1', ...
            'Selector5','To Workspace5','Error1'};
    top = find_system(m, 'SearchDepth', 1, 'Type', 'block');
    top = setdiff(top, {m});
    names = cellfun(@(b) get_param(b, 'Name'), top, 'UniformOutput', false);
    assert(all(ismember(keep, names)), 'a kept block is missing: %s', strjoin(setdiff(keep, names), ', '));
    del = top(~ismember(names, keep));
    for k = 1:numel(del)
        L = get_param(del{k}, 'LineHandles');
        hs = [L.Inport(:); L.Outport(:)];
        for h = hs(:)'
            if h > 0 && ishandle(h), delete_line(h); end
        end
    end
    for k = 1:numel(del), delete_block(del{k}); end
    fprintf('deleted %d blocks: %s\n', numel(del), strjoin(sort(names(~ismember(names, keep)))', ', '));
    % lines left dangling by the deletion (branch stubs of kept blocks) are removed
    ln = find_system(m, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'line');
    for h = ln(:)'
        if ishandle(h) && (get_param(h, 'SrcBlockHandle') < 0 || all(get_param(h, 'DstBlockHandle') < 0))
            delete_line(h);
        end
    end

    % == 2. encoder noise between Gain4 and Sum3 (D-225) ==
    delete_line(m, 'Gain4/1', 'Sum3/2');     % the branch into Sum3 only; Gain4 -> To Workspace1 stays
    add_block('simulink/Sources/From Workspace', [m '/EncoderNoise'], 'VariableName', 'v_in', ...
              'Interpolate', 'off', 'OutputAfterFinalValue', 'Holding final value', 'SampleTime', 'ts');
    add_block('simulink/Math Operations/Sum', [m '/EncoderSum'], 'Inputs', '++');
    add_line(m, 'Gain4/1', 'EncoderSum/1', 'autorouting', 'on');
    add_line(m, 'EncoderNoise/1', 'EncoderSum/2', 'autorouting', 'on');
    add_line(m, 'EncoderSum/1', 'Sum3/2', 'autorouting', 'on');

    % == 3. log the plant force ==
    add_block('simulink/Sinks/To Workspace', [m '/PlantForceLog'], 'VariableName', 'u_plant', ...
              'SaveFormat', 'Array', 'SampleTime', 'ts', 'MaxDataPoints', 'inf');
    add_line(m, 'Sum2/1', 'PlantForceLog/1', 'autorouting', 'on');

    % == 4. friction law: the chart calls the tanh ODE (D-224) ==
    ch = find(sfroot, '-isa', 'Stateflow.EMChart', 'Path', [m '/Extended ODE']);
    assert(numel(ch) == 1, 'Extended ODE chart not found');
    old = ch.Script;
    call_old = ['gantrySystemExtendedCoulomb(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, cg1, cg2, ' ...
                'cb1, cb2, cy, kb1, kb2, ma, ka, ca, L0, cc1, cc2, ccy, ts)'];
    assert(contains(old, call_old), 'unexpected chart script:\n%s', old);
    hdr = regexp(old, 'function dxdt = gantrySystemExtendedMFile\(([^)]*)\)', 'tokens', 'once');
    assert(~isempty(hdr) && endsWith(strtrim(hdr{1}), 'ts'), 'unexpected chart header:\n%s', old);
    ch.Script = sprintf(['function dxdt = gantrySystemExtendedMFile(%s, g_fric)\n' ...
        'dxdt = zeros(8,1);\n' ...
        'dxdt = gantrySystemExtendedTanh(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, cg1, cg2, ' ...
        'cb1, cb2, cy, kb1, kb2, ma, ka, ca, L0, cc1, cc2, ccy, g_fric);\n' ...
        'end\n'], hdr{1});
    dg = find(ch, '-isa', 'Stateflow.Data', 'Name', 'g_fric');
    if isempty(dg), dg = Stateflow.Data(ch); dg.Name = 'g_fric'; end
    dg.Scope = 'Parameter';
    dts = find(ch, '-isa', 'Stateflow.Data', 'Name', 'ts');
    assert(numel(dts) == 1 && strcmp(dts.Scope, 'Parameter'), 'chart data ts is not a Parameter');
    fprintf('chart script now:\n%s\n', ch.Script);
    d_all = find(ch, '-isa', 'Stateflow.Data');
    fprintf('chart data: %s\n', strjoin(arrayfun(@(x) sprintf('%s(%s)', x.Name, x.Scope), ...
            d_all, 'UniformOutput', false), ' '));

    set_param(m, 'SolverType', 'Fixed-step', 'Solver', 'ode4', 'FixedStep', sprintf('%.12g', cfg.ts));
    % report the recorded loop's wiring
    blk = find_system(m, 'SearchDepth', 1, 'Type', 'block');
    for k = 2:numel(blk)
        L = get_param(blk{k}, 'LineHandles');
        for h = L.Outport(:)'
            if h < 0, continue; end
            d = get_param(h, 'DstBlockHandle'); d = d(d > 0);
            fprintf('  %-14s -> %s\n', get_param(blk{k}, 'Name'), ...
                    strjoin(arrayfun(@(x) get_param(x, 'Name'), d, 'UniformOutput', false), ', '));
        end
    end
    save_system(m);
    close_system(m, 0);
    fprintf('saved %s\n', dst);
end
