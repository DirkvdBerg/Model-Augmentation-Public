function inspect_model(with_deps)
% INSPECT_MODEL  G5 preparation (read-only): dependencies of the two production generators and
% the wiring of the coulomb Simulink model around the controller. Loads the ORIGINAL model
% read-only and closes it without saving. Prints only.
    here = fileparts(mfilename('fullpath'));
    repo = fileparts(fileparts(fileparts(fileparts(fileparts(here)))));
    ms = fullfile(repo, 'Matlab-scripts');
    addpath(genpath(fullfile(ms, 'Augmentation')));
    addpath(fullfile(ms, 'Augmentation-coulomb'));
    addpath(fullfile(ms, 'Augmentation-telica-profiles'));
    addpath(fullfile(ms, 'Augmentation-telica-profiles-coulomb'));
    drivers = {fullfile(ms, 'Augmentation-coulomb', 'generate_trajectory_data_coulomb.m'), ...
               fullfile(ms, 'Augmentation-telica-profiles-coulomb', 'generate_trajectory_data_telica_coulomb.m')};
    for k = 1:numel(drivers) * (nargin > 0)
        [fl, pl] = matlab.codetools.requiredFilesAndProducts(drivers{k});
        fprintf('=== required files of %s\n', drivers{k});
        for j = 1:numel(fl), fprintf('  %s\n', strrep(fl{j}, [repo filesep], '')); end
        fprintf('  products: %s\n', strjoin({pl.Name}, ', '));
    end
    mdl = 'gantry_additional_state_coulomb_2025a';
    load_system(mdl);
    fprintf('=== model %s: solver %s %s step %s, StopTime %s\n', mdl, get_param(mdl,'SolverType'), ...
            get_param(mdl,'Solver'), get_param(mdl,'FixedStep'), get_param(mdl,'StopTime'));
    blks = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on');
    for j = 2:numel(blks)
        b = blks{j};
        bt = get_param(b, 'BlockType');
        extra = '';
        switch bt
            case {'FromWorkspace'}, extra = get_param(b, 'VariableName');
            case {'ToWorkspace'},   extra = get_param(b, 'VariableName');
            case {'Sum'},           extra = get_param(b, 'Inputs');
            case {'StateSpace'},    extra = [get_param(b,'A') ' ' get_param(b,'B')];
            case {'DiscreteStateSpace'}, extra = get_param(b,'A');
            case {'LTISystem'},     extra = get_param(b, 'sys');
            case {'Gain'},          extra = get_param(b, 'Gain');
            case {'Constant'},      extra = get_param(b, 'Value');
        end
        d = count(b, '/');
        if d <= 3
            fprintf('%s[%s] %s  %s\n', repmat('  ', 1, d), bt, strrep(b, newline, ' '), extra);
        end
    end
    lines = find_system(mdl, 'LookUnderMasks', 'all', 'FindAll', 'on', 'Type', 'line');
    fprintf('=== %d lines (top two levels)\n', numel(lines));
    for j = 1:numel(lines)
        L = lines(j);
        src = get_param(L, 'SrcBlockHandle'); dst = get_param(L, 'DstBlockHandle');
        if src < 0 || isempty(dst), continue; end
        sp = getfullname(src);
        if count(sp, '/') > 2, continue; end
        dn = arrayfun(@(h) strrep(getfullname(h), newline, ' '), dst(dst > 0), 'UniformOutput', false);
        fprintf('  %s (port %d) -> %s   [%s]\n', strrep(sp, newline, ' '), get_param(get_param(L,'SrcPortHandle'),'PortNumber'), ...
                strjoin(dn, ' | '), get_param(L, 'Name'));
    end
    rt = sfroot; m = rt.find('-isa', 'Stateflow.Machine', 'Name', mdl);
    ch = m.find('-isa', 'Stateflow.EMChart');
    for k = 1:numel(ch)
        fprintf('=== chart %s\n%s\n', ch(k).Path, ch(k).Script);
    end
    close_system(mdl, 0);
end
