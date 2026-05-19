% plot_trajectories.m
% Motion profile plots for all trajectories in a dataset.
%
% Each figure shows one trajectory:
%   Row 1 — Position [m]:  X1 | X2 | Y
%   Row 2 — Force   [N]:  FX1 | FX2 | FY
%
% Run from repo root:
%   run('lpv_lfr_baseline/plots/plot_trajectories.m')

% ── Dataset selector ──────────────────────────────────────────────────────────
DATASET = 'base_extended';

base = fullfile(fileparts(mfilename('fullpath')), '..', '..');

datasets = struct();
datasets.base          = struct('traj_dir', fullfile(base,'Matlab-output','parameter-recovery'));
datasets.multisine     = struct('traj_dir', fullfile(base,'Matlab-output','parameter-recovery-multisine'));
datasets.ref_injection = struct('traj_dir', fullfile(base,'Matlab-output','parameter-recovery-ref-injection'));
datasets.identification= struct('traj_dir', fullfile(base,'Matlab-output','identification-trajectories'));
datasets.base_extended = struct('traj_dir', fullfile(base,'Matlab-output','identification-trajectories-no-multisine'));

traj_specs = {
    'T1', 'T1_Y_sweep_conservative.mat';
    'T2', 'T2_X_sym_Y030.mat';
    'T3', 'T3_X_sym_Y000.mat';
    'T4', 'T4_X_antisym_Y020.mat';
    'T5', 'T5_X_sym_Y_sweep.mat';
    'T6', 'T6_Y_sweep_aggressive.mat';
    'T7', 'T7_X_antisym_Y_sweep.mat';
    'T8', 'T8_X_sym_anti_Y_sweep.mat';
};

traj_dir = datasets.(DATASET).traj_dir;
save_dir = fullfile(fileparts(mfilename('fullpath')), 'figures', DATASET);
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

fprintf('Dataset : %s\n', DATASET);
fprintf('Saving  : %s\n\n', save_dir);

for i = 1:size(traj_specs, 1)
    traj_id   = traj_specs{i, 1};
    traj_file = traj_specs{i, 2};

    mat = load(fullfile(traj_dir, traj_file));
    t   = mat.t_sim;     % (T, 1) seconds
    q1  = mat.q1;        % (T, 3) position  [X1, X2, Y]  m
    u   = mat.u_q1;      % (T, 3) force     [FX1, FX2, FY]  N

    title_str = strrep(strrep(traj_file, '.mat', ''), '_', ' ');

    fig = figure('Name', traj_id, 'NumberTitle', 'off', ...
                 'Units', 'centimeters', 'Position', [2, 2, 28, 12]);

    pos_labels   = {'X1 [m]',  'X2 [m]',  'Y [m]'};
    force_labels = {'FX1 [N]', 'FX2 [N]', 'FY [N]'};

    for col = 1:3
        % Row 1 — position
        subplot(2, 3, col);
        plot(t, q1(:, col), 'b', 'LineWidth', 0.8);
        ylabel(pos_labels{col}); grid on; box off;
        if col == 2
            title(title_str, 'FontWeight', 'bold');
        end
        set(gca, 'XTickLabel', []);

        % Row 2 — force
        subplot(2, 3, 3 + col);
        plot(t, u(:, col), 'r', 'LineWidth', 0.8);
        ylabel(force_labels{col}); grid on; box off;
        xlabel('Time [s]');
    end

    % Tight spacing between rows
    set(gcf, 'Units', 'normalized');
    for ax = findall(fig, 'Type', 'axes')
        set(ax, 'FontSize', 9);
    end

    out_path = fullfile(save_dir, sprintf('%s.png', traj_id));
    exportgraphics(fig, out_path, 'Resolution', 150);
    fprintf('  Saved: %s\n', out_path);
    close(fig);
end

fprintf('\nDone.\n');
