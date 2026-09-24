function make_noise_model()
% MAKE_NOISE_MODEL  Build gantry_cn_noise_2025a.slx: the coulomb model with encoder noise (CN-009).
%
% Source: gantry_cn_base_2025a.slx, a byte copy of
% Matlab-scripts/Augmentation-coulomb/gantry_additional_state_coulomb_2025a.slx (never opened for
% writing here). The copy is saved under the new name and ONE edit is made in the "Extended ODE"
% loop, the only loop the generator records (q_aug):
%
%     Gain4 (P.', stage q) -> Sum3 (r - q)      becomes
%     Gain4 -> EncoderNoiseSum (++) -> Sum3,   EncoderNoise (From Workspace v_enc) -> EncoderNoiseSum
%
% so the controller LTI System1 (Cfb, discrete, ts) sees q + v while To Workspace1 (q_aug) still
% logs the true q. v_enc = [t, v] with interpolation off and sample time ts (sample and hold at
% the controller's own rate). v = 0 makes the edit an exact no-op (x + 0 == x).
    here = fileparts(mfilename('fullpath'));
    src_name = 'gantry_cn_base_2025a';
    dst_name = 'gantry_cn_noise_2025a';
    dst = fullfile(here, [dst_name '.slx']);
    if bdIsLoaded(dst_name), close_system(dst_name, 0); end
    if exist(dst, 'file'), delete(dst); end
    load_system(fullfile(here, [src_name '.slx']));
    save_system(src_name, dst);
    close_system(src_name, 0);
    load_system(dst);
    m = dst_name;
    fprintf('LTI System1 sys = %s\n', get_param([m '/LTI System1'], 'sys'));

    % which input port of Sum3 is fed by Gain4?
    pc = get_param([m '/Sum3'], 'PortConnectivity');
    g4 = get_param([m '/Gain4'], 'Handle');
    port = [];
    for k = 1:numel(pc)
        if ~isempty(pc(k).SrcBlock) && any(pc(k).SrcBlock == g4)
            port = str2double(pc(k).Type);
        end
    end
    assert(~isempty(port), 'Gain4 does not feed Sum3');
    fprintf('Sum3 inputs %s; Gain4 feeds input %d\n', get_param([m '/Sum3'], 'Inputs'), port);

    delete_line(m, 'Gain4/1', sprintf('Sum3/%d', port));
    add_block('simulink/Sources/From Workspace', [m '/EncoderNoise'], 'VariableName', 'v_enc', ...
              'Interpolate', 'off', 'OutputAfterFinalValue', 'Holding final value', 'SampleTime', 'ts');
    add_block('simulink/Math Operations/Sum', [m '/EncoderNoiseSum'], 'Inputs', '++');
    add_line(m, 'Gain4/1', 'EncoderNoiseSum/1', 'autorouting', 'on');
    add_line(m, 'EncoderNoise/1', 'EncoderNoiseSum/2', 'autorouting', 'on');
    add_line(m, 'EncoderNoiseSum/1', sprintf('Sum3/%d', port), 'autorouting', 'on');

    % report the rewired neighbourhood
    for b = {'Gain4', 'EncoderNoise', 'EncoderNoiseSum'}
        L = get_param([m '/' b{1}], 'LineHandles');
        for h = L.Outport
            dst_h = get_param(h, 'DstBlockHandle');
            fprintf('  %s -> %s\n', b{1}, strjoin(arrayfun(@(x) get_param(x, 'Name'), dst_h, ...
                    'UniformOutput', false), ', '));
        end
    end
    save_system(m);
    close_system(m, 0);
    fprintf('saved %s\n', dst);
end
