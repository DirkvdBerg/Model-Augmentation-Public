function check_openloop_drift()
% CHECK_OPENLOOP_DRIFT  Is the open-loop Y drift physical rectification or a solver artefact?
%
% generate_openloop_record.m produced -1.85e-03 m of Y drift over 12 s from an input
% whose DC is 1e-16 N. Two candidates:
%   PHYSICAL   the plant is nonlinear (M depends on Y and delta_a), so oscillation at
%              150 Hz rectifies into a DC response. Then open-loop generation needs the
%              drift bounded somehow, and the friction question is live.
%   NUMERICAL  the RK4 holds u constant across each step while the true input varies,
%              so the effective input has a small bias. Then it is a harness bug.
%
% Step halving separates them: a numerical artefact shrinks with ts, a physical one does not.
% Also compares ZOH input against midpoint-sampled input at fixed ts, which isolates the
% input treatment from the integrator order.

    here = fileparts(mfilename('fullpath'));
    addpath(here); addpath(fullfile(here, 'data'));
    cfg = gtd_config('augmentation', true, 0.10);
    Y0 = 0.10;  T = 3.0;                       % 3 s is enough to see a trend
    p = {cfg.m1, cfg.m2, cfg.mb, cfg.mh_rigid, cfg.Lb, cfg.Jb, cfg.Jh, cfg.d, ...
         cfg.cg1, cfg.cg2, cfg.cb1, cfg.cb2, cfg.cy, cfg.kb1, cfg.kb2, ...
         cfg.ma, cfg.ka, cfg.ca, cfg.L0};
    f8 = @(x, u) gantrySystemExtended(u, x, p{:});

    rng(0);
    fr = (cfg.f_low : 1 : cfg.f_high)';
    ph = 2*pi*rand(numel(fr), 3);
    A_rms = [20, 20, 15];
    % analytic input so it can be evaluated at ANY time, not just on a grid
    uf = @(tq) ((cfg.P * (( cos(2*pi*tq*fr' + ph') ) .* 1)')');   % placeholder, replaced below

    % build a scale factor once, on a fine grid, so RMS matches across step sizes
    tg = (0 : cfg.ts : T-cfg.ts)';
    raw = zeros(numel(tg), 3);
    for c = 1:3, raw(:, c) = sum(cos(2*pi*tg*fr' + ph(:, c)'), 2); end
    sc = A_rms ./ std(raw, 0, 1);
    ustage_at = @(tq) arrayfun(@(c) sc(c) * sum(cos(2*pi*tq*fr + ph(:, c))), 1:3);
    ulog_at   = @(tq) (cfg.P * ustage_at(tq)')';

    fprintf('%-28s %14s %14s %14s\n', 'arm', 'Y drift [m]', 'X drift [m]', 'Th drift [rad]');

    res = struct('name', {}, 'd', {});
    for r = 0:2
        ts = cfg.ts / 2^r;  N = round(T / ts);
        d = rk4_run(f8, ulog_at, ts, N, Y0, 'zoh');
        fprintf('%-28s %14.4e %14.4e %14.4e\n', sprintf('RK4 ts/%d, ZOH input', 2^r), d(3), d(1), d(2));
        res(end+1) = struct('name', sprintf('ts/%d ZOH', 2^r), 'd', d); %#ok<AGROW>
    end
    d = rk4_run(f8, ulog_at, cfg.ts, round(T/cfg.ts), Y0, 'stage');
    fprintf('%-28s %14.4e %14.4e %14.4e\n', 'RK4 ts, input at RK stages', d(3), d(1), d(2));

    opt = odeset('RelTol', 1e-11, 'AbsTol', 1e-13);
    [~, X] = ode45(@(tq, xx) f8(xx, ulog_at(tq)'), [0 T], [0;0;Y0;0;0;0;0;0], opt);
    fprintf('%-28s %14.4e %14.4e %14.4e\n', 'ode45 RelTol 1e-11', ...
        X(end,3)-Y0, X(end,1), X(end,2));

    fprintf(['\nRead: if Y drift halves as ts halves it is NUMERICAL. If the three ZOH arms,\n' ...
             'the stage-sampled arm and ode45 all agree, the drift is PHYSICAL rectification.\n']);
end

function d = rk4_run(f8, ulog_at, ts, N, Y0, mode)
    x = [0;0;Y0;0;0;0;0;0];
    for k = 1:N
        tk = (k-1)*ts;
        if strcmp(mode, 'zoh')
            u1 = ulog_at(tk)'; u2 = u1; u3 = u1; u4 = u1;
        else
            u1 = ulog_at(tk)';        u2 = ulog_at(tk+0.5*ts)';
            u3 = u2;                  u4 = ulog_at(tk+ts)';
        end
        k1 = f8(x, u1);            k2 = f8(x + 0.5*ts*k1, u2);
        k3 = f8(x + 0.5*ts*k2, u3); k4 = f8(x + ts*k3, u4);
        x = x + (ts/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
    d = [x(1), x(2), x(3)-Y0];
end
