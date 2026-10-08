function [q, ufb, v] = cl_sim(S, ode, Y0, ctl, P, h)
% CL_SIM  One closed-loop run of the generator's recorded loop outside Simulink (D-223): K1 and the
% sources discrete at ts, the truth ODE by RK4 with the input held over the step, integrator start
% [0; 0; Y0; 0; ...]. S needs r_sim and f_sim (N x 3, stage) and may carry d_in (a force on the
% plant input, the D-218 records) and v_enc (encoder noise: the controller sees q + v, D-225).
% The loop is read from model/gantry_tdg_2025a.slx; run_tanh_sweep.m's header states it block by
% block. Returns the TRUE stage q, controller output u_fb and stage velocities, all N x 3, sampled
% at the start of each step as the To Workspace blocks log.
    N  = size(S.r_sim, 1);
    N0 = zeros(3, N);
    r  = S.r_sim.';  f = S.f_sim.';                          % 3 x N: column access
    dn = N0;  if isfield(S, 'd_in'),  dn = S.d_in.';  end
    vn = N0;  if isfield(S, 'v_enc'), vn = S.v_enc.'; end
    x  = [0; 0; Y0; 0; 0; 0; 0; 0];
    xc = zeros(size(ctl.A, 1), 1);
    q = zeros(3, N);  ufb = zeros(3, N);  v = zeros(3, N);
    PT = P.';
    for k = 1:N
        qk = PT * x(1:3);
        e  = r(:, k) - (qk + vn(:, k));                    % EncoderSum then Sum3
        uf = ctl.C * xc + ctl.D * e;
        xc = ctl.A * xc + ctl.B * e;
        u  = P * (f(:, k) + uf + dn(:, k));
        s1 = ode(x, u);
        s2 = ode(x + (h / 2) * s1, u);
        s3 = ode(x + (h / 2) * s2, u);
        s4 = ode(x + h * s3, u);
        q(:, k) = qk;  ufb(:, k) = uf;  v(:, k) = PT * x(5:7);
        x = x + (h / 6) * (s1 + 2 * s2 + 2 * s3 + s4);
    end
    q = q.';  ufb = ufb.';  v = v.';
end
