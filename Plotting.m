
% GP-based ARX Outlier Detection (Improved Segmented Evaluation)
clear all; close all; clc;
load('SPEED23workspace.mat')

% Configuration
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
     %% Loop over sensors
    for i = 1:length(response_vars)
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
        
        % ROC computation
        [fpr, tpr, ~, AUC] = perfcurve(y_true, y_scores, 1);
        figure;
        plot(fpr, tpr, 'b-', 'LineWidth', 2);
        xlabel('False Positive Rate'); ylabel('True Positive Rate');
        title(sprintf('ROC Curve - %s | Sensor: %s | AUC = %.3f', upper(regime), sensor, AUC));
        grid on;
    end
end
