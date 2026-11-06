function out = run_simulation(r, t, f, v, loop, cfg)
% RUN_SIMULATION  One closed-loop run of the truth model with reference r, injected force f and encoder noise v (all N x 3, stage coordinates).
%   Returns the true position q, the measured position y = q + v that the controller saw, and the plant force u.
    % the model reads its parameters and inputs from the base workspace; the payload mass is its rigid
    % part, since the absorber mass ma enters through the absorber state
    names = {'mb','m1','m2','Jb','Jh','cg1','cg2','cy','cb1','cb2','kb1','kb2','Lb','d', ...
             'cc1','cc2','ccy','ma','L0','ka','ca','P','ts'};
    for k = 1:numel(names), assignin('base', names{k}, cfg.(names{k})); end
    assignin('base', 'mh', cfg.mh_rigid);
    assignin('base', 'g_fric', cfg.friction_gain);
    assignin('base', 'Cfb', loop.Cfb);
    assignin('base', 'Y', loop.Y_op);                  % initial state [0; 0; Y; 0; ...]
    assignin('base', 'r', r);
    assignin('base', 't', t);
    assignin('base', 'f', f);
    assignin('base', 'v_in', [t(:), v]);
    evalin('base', 'clear q1');                        % no output of an earlier run
    evalin('base', sprintf('sim(''%s'', %.12g);', cfg.mdl, t(end)));
    q = evalin('base', 'q1');
    assert(size(q, 1) == numel(t), 'the simulation returned %d samples, the record has %d', size(q, 1), numel(t));

    % the plant force is rebuilt from the same controller and inputs: u = K1 (r - y) + f
    y = q + v;
    out = struct('q', q, 'y', y, 'u', lsim(loop.Cfb, r - y) + f);
end
