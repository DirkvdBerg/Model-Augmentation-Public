% GTD_CHECK_PHASEA  Verify Phase A (config, records, plant, references).
%   Run from the repo root:  >> gtd_check_phaseA
%   Builds every record's reference and asserts it is within the enforced
%   limits, and builds the plant/controller at each operating point.

clear; clc;

cfg     = gtd_config('joint', true, 0.50);
records = gtd_build_records(cfg);

assert(numel(records) == 22, 'expected 22 records, got %d', numel(records));
fprintf('Records: %d   out_dir: %s\n', numel(records), cfg.out_dir);
fprintf('Limits : yaw %.3g m, vel %.1f, acc %.0f/%.0f, force peak [%d %d %d]\n', ...
        cfg.lim.diff, cfg.lim.vel, cfg.lim.acc_X, cfg.lim.acc_Y, cfg.lim.force_peak);

for k = 1:numel(records)
    plant = gtd_build_plant(records(k).Y_op, cfg); %#ok<NASGU>  % exercises getss + ruleOfThumb
    [r,t] = gtd_make_reference(records(k), cfg);
    assert(size(r,1) == cfg.N_record, '%s: %d samples, expected %d', ...
           records(k).id, size(r,1), cfg.N_record);
    gtd_validate_ref(r, t, records(k).id, cfg.lim);
end

% Confirm the mode toggle changes band + folder
cfg2 = gtd_config('augmentation', true, 0.50);
fprintf('augmentation band [%d %d]   out_dir: %s\n', cfg2.f_low, cfg2.f_high, cfg2.out_dir);

fprintf('\nPhase A OK\n');
