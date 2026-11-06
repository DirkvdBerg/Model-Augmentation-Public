function build_model()
% BUILD_MODEL  Builds the recorded closed loop model/gantry_truth.slx from Jasper's model/gantry_2025a.slx.
%   Jasper's file is not changed; only the copy is edited.
    here = fileparts(mfilename('fullpath'));
    src = 'gantry_2025a';
    mdl = 'gantry_truth';
    dst = fullfile(here, 'model', [mdl '.slx']);
    for m = {src, mdl}
        if bdIsLoaded(m{1}), close_system(m{1}, 0); end
    end
    load_system(fullfile(here, 'model', [src '.slx']));
    save_system(src, dst);                     % the open model is now the copy
    close_system(mdl, 0);
    load_system(dst);

    % 1. Use Jasper's closed loop with the MATLAB Function plant
    keep = {'Reference1', 'Sum3', 'LTI System1', 'Feedforward1', 'Sum2', 'Gain3', 'MATLAB Function1', ...
            'Integrator', 'Selector1', 'Gain4', 'To Workspace1', 'Error1'};
    blocks = setdiff(find_system(mdl, 'SearchDepth', 1, 'Type', 'block'), {mdl});
    names = cellfun(@(b) get_param(b, 'Name'), blocks, 'UniformOutput', false);
    assert(all(ismember(keep, names)), 'block missing in Jasper''s model: %s', strjoin(setdiff(keep, names), ', '));
    remove = blocks(~ismember(names, keep));
    for k = 1:numel(remove)
        lines = get_param(remove{k}, 'LineHandles');
        for h = [lines.Inport(:); lines.Outport(:)]'
            if h > 0 && ishandle(h), delete_line(h); end
        end
    end
    for k = 1:numel(remove), delete_block(remove{k}); end
    for h = find_system(mdl, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'line')'
        if ishandle(h) && (get_param(h, 'SrcBlockHandle') < 0 || all(get_param(h, 'DstBlockHandle') < 0))
            delete_line(h);                    % branches left without a block
        end
    end

    % 2. The plant becomes the truth: Jasper's rigid gantry with the hidden absorber and the tanh
    %    friction (gantrySystemExtendedTanh.m), so the state grows from 6 to 8
    chart = find(sfroot, '-isa', 'Stateflow.EMChart', 'Path', [mdl '/MATLAB Function1']);
    assert(numel(chart) == 1, 'MATLAB Function1 not found');
    for name = {'ma', 'ka', 'ca', 'L0', 'cc1', 'cc2', 'ccy', 'g_fric'}
        data = Stateflow.Data(chart);
        data.Name = name{1};
        data.Scope = 'Parameter';
    end
    chart.Script = sprintf(['function dxdt = gantrySystemMFile(u,x,m1,m2,mb,mh,Jb,Jh,d,Lb,kb1,kb2,cg1,cg2,cb1,cb2,cy,' ...
        'ma,ka,ca,L0,cc1,cc2,ccy,g_fric)\n' ...
        'dxdt = zeros(8,1);\n' ...
        'dxdt = gantrySystemExtendedTanh(u,x,m1,m2,mb,mh,Lb,Jb,Jh,d,cg1,cg2,cb1,cb2,cy,kb1,kb2,ma,ka,ca,L0,cc1,cc2,ccy,g_fric);\n' ...
        'end\n']);
    set_param([mdl '/Integrator'], 'InitialCondition', '[0; 0; Y; 0; 0; 0; 0; 0]');
    set_param([mdl '/Selector1'], 'InputPortWidth', '8');

    % 3. Encoder noise: the controller sees the position plus the noise v_in, while To Workspace1 (q1)
    %    keeps the true position
    delete_line(mdl, 'Gain4/1', 'Sum3/2');
    add_block('simulink/Sources/From Workspace', [mdl '/EncoderNoise'], 'VariableName', 'v_in', ...
              'Interpolate', 'off', 'OutputAfterFinalValue', 'Holding final value', 'SampleTime', 'ts');
    add_block('simulink/Math Operations/Sum', [mdl '/EncoderSum'], 'Inputs', '++');
    add_line(mdl, 'Gain4/1', 'EncoderSum/1', 'autorouting', 'on');
    add_line(mdl, 'EncoderNoise/1', 'EncoderSum/2', 'autorouting', 'on');
    add_line(mdl, 'EncoderSum/1', 'Sum3/2', 'autorouting', 'on');

    % Fixed-step RK4 (ode4) instead of Jasper's ode45, as in the trained model: the model-data
    % mismatch is then the physics (absorber, friction), not solver error
    set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'ode4', 'FixedStep', 'ts');
    save_system(mdl);
    close_system(mdl, 0);
    fprintf('built %s\n', dst);
end
