% plot_sim_results.m
% Plots model vs measured trajectory comparison for a parameter recovery run.
%
% Prerequisites:
%   Run export_sim_results.py first to generate the .mat file next to the .pt.
%
% Usage:
%   1. Set PT_FILE below to the path of your .pt checkpoint.
%   2. Run from repo root:  run('lpv_lfr_baseline/plots/plot_sim_results.m')
%
% Output:
%   figures/{run_id}/{split}/{traj_id}.png
%   Each figure: 3 rows x 3 cols
%     Row 1 — position [m]:  model (blue) vs measured (gray --)
%     Row 2 — error   [m]:  model - measured
%     Row 3 — force   [N]:  FX1 | FX2 | FY

% ── Set this to your .pt file ─────────────────────────────────────────────────
PT_FILE = 'C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation\simulations\param_recovery_base_extended\lfr_param_recovery_base_extended_T1_T2_T3_T4_T5_T6_T7_T8_e1500_63535.pt';

% ── Auto-discover sim_results_*.mat next to the .pt ──────────────────────────
[pt_dir, ~, ~] = fileparts(PT_FILE);
mat_files = dir(fullfile(pt_dir, 'sim_results_*.mat'));
if isempty(mat_files)
    error('No sim_results_*.mat found in %s\nRun export_sim_results.py first.', pt_dir);
end
mat_file = fullfile(pt_dir, mat_files(end).name);   % use most recent if multiple
fprintf('Loading: %s\n\n', mat_file);

data   = load(mat_file);
trajs  = data.trajs;
run_id = data.run_id;

save_root = fullfile(fileparts(mfilename('fullpath')), 'figures', run_id);
fprintf('Saving figures to: %s\n\n', save_root);

pos_labels   = {'X1 [m]',      'X2 [m]',      'Y [m]'};
err_labels   = {'\DeltaX1 [m]','\DeltaX2 [m]','\DeltaY [m]'};
force_labels = {'FX1 [N]',     'FX2 [N]',     'FY [N]'};

for i = 1:numel(trajs)
    tr       = trajs(i);
    traj_id  = tr.id;
    split    = tr.split;
    t        = tr.t;
    q1_meas  = tr.q1_measured;
    q1_mod   = tr.q1_model;
    u        = tr.u;
    err      = q1_mod - q1_meas;

    title_str = sprintf('%s  [%s]', strrep(traj_id, '_', ' '), split);

    fig = figure('Name', traj_id, 'NumberTitle', 'off', ...
                 'Units', 'centimeters', 'Position', [2, 2, 28, 16]);

    for col = 1:3
        % ── Row 1: model vs measured ──────────────────────────────────────
        subplot(3, 3, col);
        plot(t, q1_meas(:, col), '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.8); hold on;
        plot(t, q1_mod(:, col),  'b',                           'LineWidth', 0.8);
        ylabel(pos_labels{col}); grid on; box off;
        set(gca, 'XTickLabel', [], 'FontSize', 9);
        if col == 1
            legend('Measured', 'Model', 'Location', 'best', 'FontSize', 8);
        end
        if col == 2
            title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
        end

        % ── Row 2: error ──────────────────────────────────────────────────
        subplot(3, 3, 3 + col);
        plot(t, err(:, col), 'k', 'LineWidth', 0.8);
        yline(0, '--', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.6);
        ylabel(err_labels{col}); grid on; box off;
        set(gca, 'XTickLabel', [], 'FontSize', 9);

        % ── Row 3: forces ─────────────────────────────────────────────────
        subplot(3, 3, 6 + col);
        plot(t, u(:, col), 'r', 'LineWidth', 0.8);
        ylabel(force_labels{col}); grid on; box off;
        xlabel('Time [s]');
        set(gca, 'FontSize', 9);
    end

    % Save
    out_dir = fullfile(save_root, split);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    out_path = fullfile(out_dir, sprintf('%s.png', traj_id));
    exportgraphics(fig, out_path, 'Resolution', 150);
    fprintf('  Saved: %s\n', out_path);
    close(fig);
end

fprintf('\nDone.\n');
