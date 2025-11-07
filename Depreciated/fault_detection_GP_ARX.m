% GP-based ARX Outlier Detection (Improved Segmented Evaluation)
clear all; close all; clc;

%% Configuration
addpath(genpath(fullfile(pwd, 'LPV_Core')));
disp('✅ LPV Toolbox added.');
data_dir = fullfile(pwd, 'Gearbox_VWCs', 'MCC5-THU');
regimes = {'speed', 'torque'};
response_vars = {'gearbox_vibration_x', 'gearbox_vibration_y','gearbox_vibration_z'};

% Detection thresholds
kl_threshold = 0.15;
vote_fraction_threshold = 0.05;  

% Windowing
gp_segment_duration_sec = 1;      % Used for GP variance prediction
overlap_ratio = 0.9;
healthy_step_duration_sec = 20;   % Segment length to break healthy files

%% Loop over regimes
for r = 1:length(regimes)
    regime = regimes{r};
    fprintf('\n=== 🔍 Evaluating Regime: %s-Controlled ===\n', upper(regime));

    sensor_conf_matrices = cell(length(response_vars),1);

    %% Load healthy validation + test data
    load(fullfile(data_dir, ['X_valid_', regime, '.mat']), 'X_valid', 'fs_target');
    load(fullfile(data_dir, ['X_test_healthy_', regime, '.mat']), 'X_test');

    fs = fs_target;
    gp_win_samples = gp_segment_duration_sec * fs;
    hop_samples = round(gp_win_samples * (1 - overlap_ratio));
    healthy_step_samples = round(healthy_step_duration_sec * fs);

    %% Loop over sensors
    for i = 1:length(response_vars)
        sensor = response_vars{i};
        y_true = []; y_pred = [];

        % Load trained model + GP
        load(fullfile(data_dir, sprintf('arx_gp_%s_%s.mat', sensor, regime)), 'best_model', 'mean_model', 'std_model');

        %% === HEALTHY FILES ===
        healthy_sets = [X_test(:)];

        for h = 1:length(healthy_sets)
            x = healthy_sets{h};
            Y = x.others.(sensor);
            N = length(Y);

            if strcmp(regime, 'torque')
                excitation = x.speed';
                control = x.torque;
            else
                excitation = x.torque;
                control = x.speed';
            end

            torque = x.torque;
            step_samples = round(healthy_step_samples * (1 - overlap_ratio));
            seg_starts = 1:step_samples:(N - healthy_step_samples + 1);

            for s = 1:length(seg_starts)
                seg_start = seg_starts(s);
                seg_end = seg_start + healthy_step_samples - 1;

                fault_votes = 0;
                total_windows = 0;

                for start_idx = seg_start:hop_samples:(seg_end - gp_win_samples + 1)
                    idx = start_idx:(start_idx + gp_win_samples - 1);

                    Y_seg = Y(idx);
                    U_seg = excitation(idx);
                    data_id = iddata(Y_seg, U_seg, 1/fs);
                    Y_pred = predict(best_model, data_id, 1).OutputData;
                    residuals = Y_seg - Y_pred;

                    emp_mu = mean(residuals);
                    emp_sigma = std(residuals) + 1e-6;

                    input_vector = [mean(control(idx)), mean(torque(idx))];
                    pred_mu = predict(mean_model, input_vector);
                    pred_sigma = sqrt(predict(std_model, input_vector)) + 1e-6;

                    kl_var = log(emp_sigma / pred_sigma) + (pred_sigma^2) / (2 * emp_sigma^2) - 0.5;
                    kl_mean = (pred_mu - emp_mu)^2 / (2 * emp_sigma^2);
                    kl = kl_var + kl_mean;

                    if kl > kl_threshold
                        fault_votes = fault_votes + 1;
                    end
                    total_windows = total_windows + 1;
                end

                fault_detected = (fault_votes / total_windows) > vote_fraction_threshold;

                fprintf('[%s] Healthy File %d | Segment %d | Sensor: %-25s | Fault Segments = %d/%d\n', ...
                    upper(regime), h, s, sensor, fault_votes, total_windows);

                y_true(end+1) = 0;
                y_pred(end+1) = fault_detected;
            end
        end

        %% === FAULTY FILES ===
        fault_structs = dir(fullfile(data_dir, 'X_*.mat'));
        fault_structs = fault_structs(~contains({fault_structs.name}, {'train','valid','test_healthy'}));
        fault_files = {fault_structs.name};
        fault_files = fault_files(contains(fault_files, [regime, '_circulation']));
        fault_files = fault_files(randperm(length(fault_files)));
        num_fault_samples = 20;
        fault_files = fault_files(1:max(num_fault_samples, length(fault_files)));

        for f = 1
            :length(fault_files)
            fname = fault_files{f};
            data = load(fullfile(data_dir, fname), 'T_fault_processed');
            x = data.T_fault_processed;
            Y = x.others.(sensor);
            N = length(Y);

            if strcmp(regime, 'torque')
                excitation = x.speed';
                control = x.torque;
            else
                excitation = x.torque;
                control = x.speed';
            end

            torque = x.torque;
            fault_votes = 0;
            total_windows = 0;

            for start_idx = 1:hop_samples:(N - gp_win_samples + 1)
                idx = start_idx:(start_idx + gp_win_samples - 1);

                Y_seg = Y(idx);
                U_seg = excitation(idx);
                data_id = iddata(Y_seg, U_seg, 1/fs);
                Y_pred = predict(best_model, data_id, 1).OutputData;
                residuals = Y_seg - Y_pred;

                emp_mu = mean(residuals);
                emp_sigma = std(residuals) + 1e-6;

                input_vector = [mean(control(idx)), mean(torque(idx))];
                pred_mu = predict(mean_model, input_vector);
                pred_sigma = sqrt(predict(std_model, input_vector)) + 1e-6;

                kl_var = log(emp_sigma / pred_sigma) + (pred_sigma^2) / (2 * emp_sigma^2) - 0.5;
                kl_mean = (pred_mu - emp_mu)^2 / (2 * emp_sigma^2);
                kl = kl_var + kl_mean;

                if kl > kl_threshold
                    fault_votes = fault_votes + 1;
                end
                total_windows = total_windows + 1;
            end

            fault_detected = (fault_votes / total_windows) > vote_fraction_threshold;

            fprintf('[%s] Fault File %d | Sensor: %-25s | Fault Segments = %d/%d\n', ...
                upper(regime), f, sensor, fault_votes, total_windows);

            y_true(end+1) = 1;
            y_pred(end+1) = fault_detected;
        end

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
%         pause;
    end
end
