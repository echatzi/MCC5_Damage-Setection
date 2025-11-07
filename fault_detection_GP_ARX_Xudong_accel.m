% GP-based ARX Outlier Detection (Improved Segmented Evaluation, Parallel, fast)
clear all; close all; clc;

%% Configuration
addpath(genpath(fullfile(pwd, 'LPV_Core')));
disp('✅ LPV Toolbox added.');
data_dir = fullfile(pwd, 'Gearbox_VWCs', 'MCC5-THU');
regimes = {'speed', 'torque'};
response_vars = {'gearbox_vibration_x', 'gearbox_vibration_y','gearbox_vibration_z'};

% Detection thresholds
kl_threshold = 0.15;

% Windowing
gp_segment_duration_sec   = 1;     % window used for KL
overlap_ratio             = 0.9;
healthy_step_duration_sec = 20;    % segment length to break healthy files

% start parallel pool if not running
if isempty(gcp('nocreate'))
    parpool('local');
end

%% Loop over regimes
for r = 1:length(regimes)
    regime = regimes{r};
    fprintf('\n=== 🔍 Evaluating Regime: %s-Controlled ===\n', upper(regime));

    sensor_conf_matrices = cell(length(response_vars),1);
    if strcmp(regime, 'torque')
        vote_fraction_threshold = 1e-5;
    else
        vote_fraction_threshold = 0.04;   %originally 0.05
    end

    %% Load healthy validation + test data
    load(fullfile(data_dir, ['X_valid_', regime, '.mat']), 'X_valid', 'fs_target');
    load(fullfile(data_dir, ['X_test_healthy_', regime, '.mat']), 'X_test');

    fs = fs_target;
    gp_win_samples       = gp_segment_duration_sec * fs;
    hop_samples          = round(gp_win_samples * (1 - overlap_ratio));
    healthy_step_samples = round(healthy_step_duration_sec * fs);

    %% Loop over sensors
    for i = 1:length(response_vars)
        sensor = response_vars{i};
        y_true   = [];
        y_pred   = [];
        y_scores = [];

        % Load trained model + GP ONCE per sensor
        model_path = fullfile(data_dir, sprintf('arx_gp_%s_%s.mat', sensor, regime));
        load(model_path, 'best_model', 'mean_model', 'std_model');

        %% === HEALTHY FILES ===
        healthy_sets = [X_test(:)];

        for h = 1:length(healthy_sets)

            x = healthy_sets{h};
            Y = x.others.(sensor);
            N = length(Y);

            if strcmp(regime, 'torque')
                excitation = x.speed';
                control    = x.torque;
            else
                excitation = x.torque;
                control    = x.speed';
            end

            torque = x.torque;
            speed  = x.speed';

            step_samples = round(healthy_step_samples * (1 - overlap_ratio));
            seg_starts   = 1:step_samples:(N - healthy_step_samples + 1);
            nSeg         = numel(seg_starts);

            % preallocate per-file results
            file_y_true   = zeros(1, nSeg);
            file_y_pred   = zeros(1, nSeg);
            file_y_scores = zeros(1, nSeg);
            file_msgs     = cell(1, nSeg);

            parfor s = 1:nSeg
                seg_start = seg_starts(s);
                seg_end   = seg_start + healthy_step_samples - 1;

                % ---------- predict once for the whole 20 s segment ----------
                Y_long = Y(seg_start:seg_end);
                U_long = excitation(seg_start:seg_end);

                data_id_long = iddata(Y_long, U_long, 1/fs);
                Y_pred_long  = predict(best_model, data_id_long, 1).OutputData;
                residuals_long = Y_long - Y_pred_long;

                % slice corresponding control channels for GP input
                ctrl_long   = control(seg_start:seg_end);
                torque_long = torque(seg_start:seg_end);
                speed_long  = speed(seg_start:seg_end);

                fault_votes   = 0;
                total_windows = 0;

                % now slide 1 s windows over the precomputed residuals
                for start_idx = 1:hop_samples:(healthy_step_samples - gp_win_samples + 1)
                    idx_local = start_idx:(start_idx + gp_win_samples - 1);

                    res_win   = residuals_long(idx_local);
                    emp_mu    = mean(res_win);
                    emp_sigma = std(res_win) + 1e-6;

                    % GP input for this window
                    input_vector = [mean(speed_long(idx_local)), mean(torque_long(idx_local))];
                    pred_mu    = predict(mean_model, input_vector);
                    pred_sigma = sqrt(predict(std_model, input_vector)) + 1e-6;

                    kl_var  = log(emp_sigma / pred_sigma) + (pred_sigma^2) / (2 * emp_sigma^2) - 0.5;
                    kl_mean = (pred_mu - emp_mu)^2 / (2 * emp_sigma^2);
                    kl      = kl_var + kl_mean;

                    if kl > kl_threshold
                        fault_votes = fault_votes + 1;
                    end
                    total_windows = total_windows + 1;
                end

                vote_fraction  = fault_votes / total_windows;
                fault_detected = vote_fraction > vote_fraction_threshold;

                file_y_true(s)   = 0;               % healthy
                file_y_pred(s)   = fault_detected;
                file_y_scores(s) = vote_fraction;

                file_msgs{s} = sprintf('[%s] Healthy File %d | Segment %d | Sensor: %-25s | Fault Segments = %d/%d', ...
                    upper(regime), h, s, sensor, fault_votes, total_windows);
            end

            % print collected messages in order
            for s = 1:nSeg
                fprintf('%s\n', file_msgs{s});
            end

            % append to global
            y_true   = [y_true,   file_y_true];
            y_pred   = [y_pred,   file_y_pred];
            y_scores = [y_scores, file_y_scores];
        end

        %% === FAULTY FILES ===
        fault_structs = dir(fullfile(data_dir, 'X_*.mat'));
        fault_structs = fault_structs(~contains({fault_structs.name}, {'train','valid','test_healthy'}));
        fault_files   = {fault_structs.name};
        fault_files   = fault_files(contains(fault_files, [regime, '_circulation']));
        fault_files   = fault_files(randperm(length(fault_files)));
        num_fault_samples = 20;
        fault_files = fault_files(1:max(num_fault_samples, length(fault_files)));

        nFault = numel(fault_files);

        fault_y_true   = zeros(1, nFault);
        fault_y_pred   = zeros(1, nFault);
        fault_y_scores = zeros(1, nFault);
        fault_msgs     = cell(1, nFault);

        parfor f = 1:nFault

            fname = fault_files{f};
            data  = load(fullfile(data_dir, fname), 'T_fault_processed');
            x     = data.T_fault_processed;
            Y     = x.others.(sensor);
            N     = length(Y);

            if strcmp(regime, 'torque')
                excitation = x.speed';
                control    = x.torque;
            else
                excitation = x.torque;
                control    = x.speed';
            end

            torque = x.torque;
            speed  = x.speed';

            % ---------- predict once for the whole faulty file ----------
            data_id_long  = iddata(Y, excitation, 1/fs);
            Y_pred_long   = predict(best_model, data_id_long, 1).OutputData;
            residuals_long = Y - Y_pred_long;

            fault_votes   = 0;
            total_windows = 0;

            for start_idx = 1:hop_samples:(N - gp_win_samples + 1)
                idx_local = start_idx:(start_idx + gp_win_samples - 1);

                res_win   = residuals_long(idx_local);
                emp_mu    = mean(res_win);
                emp_sigma = std(res_win) + 1e-6;

                % use speed + torque as in your healthy part
                input_vector = [mean(speed(idx_local)), mean(torque(idx_local))];
                pred_mu    = predict(mean_model, input_vector);
                pred_sigma = sqrt(predict(std_model, input_vector)) + 1e-6;

                kl_var  = log(emp_sigma / pred_sigma) + (pred_sigma^2) / (2 * emp_sigma^2) - 0.5;
                kl_mean = (pred_mu - emp_mu)^2 / (2 * emp_sigma^2);
                kl      = kl_var + kl_mean;

                if kl > kl_threshold
                    fault_votes = fault_votes + 1;
                end
                total_windows = total_windows + 1;
            end

            vote_fraction  = fault_votes / total_windows;
            fault_detected = vote_fraction > vote_fraction_threshold;

            fault_y_true(f)   = 1;
            fault_y_pred(f)   = fault_detected;
            fault_y_scores(f) = vote_fraction;

            fault_msgs{f} = sprintf('[%s] Fault File %d | Sensor: %-25s | Fault Segments = %d/%d', ...
                upper(regime), f, sensor, fault_votes, total_windows);
        end

        % print fault messages
        for f = 1:nFault
            fprintf('%s\n', fault_msgs{f});
        end

        % append fault results
        y_true   = [y_true,   fault_y_true];
        y_pred   = [y_pred,   fault_y_pred];
        y_scores = [y_scores, fault_y_scores];

        %% === RESULTS ===
        fprintf('\n📊 Confusion Matrix for %s | Sensor: %s\n', upper(regime), sensor);
        conf_mat = confusionmat(y_true, y_pred);
        sensor_conf_matrices{i} = conf_mat;
        disp(array2table(conf_mat, ...
            'VariableNames', {'Pred_Healthy','Pred_Faulty'}, ...
            'RowNames', {'True_Healthy','True_Faulty'}));

        accuracy = sum(diag(conf_mat)) / sum(conf_mat(:));
        fprintf('Accuracy: %.2f%%\n', 100 * accuracy);
        fprintf('Error Rate: %.2f%%\n\n', 100 * (1 - accuracy));

        % AUC computation
        [~,~,~,AUC] = perfcurve(y_true, y_scores, 1);
        fprintf('AUC: %.3f\n\n', AUC);

        % ROC computation (keep outside parfor)
        [fpr, tpr, ~, AUC] = perfcurve(y_true, y_scores, 1);
        figure;
        plot(fpr, tpr, 'b-', 'LineWidth', 2);
        xlabel('False Positive Rate'); ylabel('True Positive Rate');
        title(sprintf('ROC Curve - %s | Sensor: %s | AUC = %.3f', upper(regime), sensor, AUC));
        grid on;

    end
end
