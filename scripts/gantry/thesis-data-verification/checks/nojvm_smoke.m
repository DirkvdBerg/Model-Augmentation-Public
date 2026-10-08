function nojvm_smoke()
% NOJVM_SMOKE  C15 of IMPLEMENTATION.md: does MATLAB started with -nojvm (less RAM) simulate
% exactly as MATLAB with its JVM?
% Stated before the run: the same real record (TR-P3_r1: S-curve moves, A_prod multisine,
% its own noise draw, friction on) is simulated once per MATLAB mode; the printed 17-digit
% checksums of q_aug, delta_a_ode and u_total must be identical between the two modes.
% Each mode writes its checksum line to logs/smoke_jvm<0|1>.txt (named by usejava('jvm'),
% so a run can only write its own file; REVIEW.md finding 2); tools/pipeline_pilot.sh
% compares the two files. No data are saved.
% (A first version used a standstill at Y = 0 with zero force, whose output is exactly zero:
% a comparison of zeros proves nothing, so it was replaced before any decision used it.)
    here = fileparts(fileparts(mfilename('fullpath')));                % scripts/gantry/thesis-data-verification
    addpath(fullfile(fileparts(fileparts(fileparts(here))), 'Thesis-writeup', 'Code', 'Data'));   % the generator
    fprintf('JVM running: %d\n', usejava('jvm'));
    cfg = tdg_init(false);
    man = tdg_manifest(cfg);
    e = man(strcmp({man.file}, 'TR-P3_r1'));
    in = tdg_build_inputs(e, cfg, false);
    K1 = gtd_build_plant(0, cfg, 1).Cfb;
    d = tdg_noise_draw(tdg_noise_psd(e.Y_op, cfg, K1), e.seed_noise);
    o = gtd_run_simulation(e, in.r, in.t, in.f_safe, in.plant, cfg, d);
    line = sprintf('sum|q| %.17g, sum|da| %.17g, sum|u| %.17g', ...
            sum(abs(o.q_with(:))), sum(abs(o.da_with)), sum(abs(o.u_total(:))));
    fprintf('sim %.1f s, %s\n', o.sim_s, line);
    fid = fopen(fullfile(here, 'logs', sprintf('smoke_jvm%d.txt', usejava('jvm'))), 'w');
    fprintf(fid, '%s\n', line);  fclose(fid);
end
