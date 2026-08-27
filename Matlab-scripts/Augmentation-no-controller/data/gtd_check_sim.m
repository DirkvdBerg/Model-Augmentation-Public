% GTD_CHECK_SIM  Smoke-test the Simulink integration on ONE record.
%   Run from repo root:  >> gtd_check_sim
%   Runs the full pipeline (plant -> reference -> multisine -> limits -> Simulink)
%   on a single standstill record and reports delta_a activation and the force
%   budget. This is the first step that touches the actual Simulink model; if it
%   errors with "Undefined function or variable 'X'", the model references a
%   base-workspace variable X that gtd_run_simulation does not push yet.

clear; clc;
cfg     = gtd_config('joint', true, 0.50);
records = gtd_build_records(cfg);
rec     = records(strcmp({records.id}, 'T3_standstill_Y000'));

plant  = gtd_build_plant(rec.Y_op, cfg);
[r, t] = gtd_make_reference(rec, cfg);
ms     = gtd_make_multisine(rec, plant, cfg);
[f_safe, chk] = gtd_enforce_limits(plant, r, ms.f_stage, cfg);
fprintf('limit scale = %.2f, pre-sim force peak [%.0f %.0f %.0f] N, yaw %.2f mm\n', ...
        chk.scale, chk.force_peak, chk.yaw_mm);

out = gtd_run_simulation(rec, r, t, f_safe, plant, cfg);

fprintf('q_with %dx%d over %.2f s\n', size(out.q_with,1), size(out.q_with,2), out.t_sim(end));
fprintf('delta_a rms: with %.3e / without %.3e  -> ratio %.1fx\n', ...
        rms(out.da_with), rms(out.da_without), rms(out.da_with)/max(rms(out.da_without),eps));
fprintf('total force peak [%.0f %.0f %.0f] / [%d %d %d] N\n', ...
        max(abs(out.u_total)), cfg.lim.force_peak);

assert(size(out.q_with,2) == 3 && all(isfinite(out.u_total(:))), 'bad simulation output');
fprintf('\nSim smoke test OK\n');
