% GP-based ARX Outlier Detection Training Script
clear all; close all; clc;

%% Configuration
addpath(genpath(fullfile(pwd, 'LPV_Core')));
disp('✅ LPV Toolbox added.');
data_dir = fullfile(pwd, 'Gearbox_VWCs', 'MCC5-THU');
regimes = {'speed', 'torque'};
response_vars = {'gearbox_vibration_x', 'gearbox_vibration_y','gearbox_vibration_z'};

file_prefix = {'X_train_speed', 'X_train_torque'};
valid_prefix = {'X_valid_speed', 'X_valid_torque'};

% ARX Model Order Input
prompt = {'Enter minimum order (na = nb):', 'Enter maximum order (na = nb):', 'Enter pa (ignored for ARX):'};
dlgtitle = 'ARX Model Order';
% dims = [1 35];
% definput = {'7', '12', '2'};
% answer = inputdlg(prompt, dlgtitle, dims, definput);

order_min = 16;
order_max = 16;

% Windowing parameters
segment_length_sec = 1;
overlap_ratio = 0.75;

%% Loop over regimes
for r = 1:length(regimes)
    regime = regimes{r};
    fprintf('\n=== 🚀 Training for Regime: %s-Controlled ===\n', upper(regime));

    % Load training and validation sets
    load(fullfile(data_dir, [file_prefix{r}, '.mat']), 'X_train', 'fs_target');
    load(fullfile(data_dir, [valid_prefix{r}, '.mat']), 'X_valid');
    fs = fs_target;
    
    X_train_flat = flatten_struct_cells(X_train);
    X_valid_flat = flatten_struct_cells(X_valid);
    X_all = [X_train_flat; X_valid_flat];

    for i = 1:length(response_vars)
        output_var = response_vars{i};
        fprintf('\n🔁 Output: %s\n', output_var);

        Y_all_raw = X_all{:, output_var};

        if strcmp(regime, 'torque')
            excitation_all = X_all.speed;
        else
            excitation_all = X_all.torque;
        end

        speed_all  = X_all.speed;
        torque_all = X_all.torque;

        % Trim edges
        trim_sec = 3;
        Y_all = trim_edges(Y_all_raw,     trim_sec, fs);
        excitation_all = trim_edges(excitation_all,trim_sec, fs);
        speed_all = trim_edges(speed_all,     trim_sec, fs);
        torque_all = trim_edges(torque_all,    trim_sec, fs);

        % Split back to training/validation
        N_train = height(X_train_flat);
        ratio = N_train / (N_train + height(X_valid_flat));
        idx_cut = floor(length(Y_all) * ratio);

        Y_train          = Y_all(1:idx_cut);
        excitation_train = excitation_all(1:idx_cut);
        speed_train      = speed_all(1:idx_cut);
        torque_train     = torque_all(1:idx_cut);

        Y_valid          = Y_all(idx_cut+1:end);
        excitation_valid = excitation_all(idx_cut+1:end);
        speed_valid      = speed_all(idx_cut+1:end);
        torque_valid     = torque_all(idx_cut+1:end);


        % Fit best ARX model by grid search
        best_rmse = Inf;
        for na = order_min:order_max
            try
                model = arx(iddata(Y_train, excitation_train, 1/fs), [na na 0]);
                data_valid = iddata(Y_valid, excitation_valid, 1/fs);
                Y_pred = predict(model, data_valid, 1);
                residuals = Y_valid - Y_pred.OutputData;
                rmse_valid = sqrt(mean(residuals.^2));

                if rmse_valid < best_rmse
                    best_rmse = rmse_valid;
                    best_model = model;
                    best_order = na;
                    best_Y_pred = Y_pred;
                end
            catch
                warning('⚠️ ARX Estimation failed for order %d.', na);
            end
        end

        fprintf('✅ Best ARX na = nb = %d | RMSE = %.4f\n', best_model.na, best_rmse);

        % Simulate and compute residuals on training
        Y_hat_train = predict(best_model, iddata(Y_train, excitation_train, 1/fs), 1).OutputData;
        residuals_train = Y_train - Y_hat_train;

        % Sliding window statistics
        segment_length = round(fs * segment_length_sec);
        hop_size = round(segment_length * (1 - overlap_ratio));
        all_inputs = [];
        all_mu = [];
        all_sigma2 = [];
        N = length(residuals_train);
        for k = 1:hop_size:(N - segment_length + 1)
            idx = k:(k + segment_length - 1);
            all_inputs(end+1, :) = [ mean(speed_train(idx)), mean(torque_train(idx)) ];  % modified
            all_mu(end+1, 1)     =  mean(residuals_train(idx));                         
            all_sigma2(end+1, 1) =  var(residuals_train(idx)) + 1e-6;                  
        end

        % Train GP models (2D input)
        % === GP Fitting Configuration ===
        kernel_fn = 'matern32';   % Options: 'matern32', 'matern52', etc.
        basis_fn = 'linear';              % Options: 'linear', 'none'
        standardize_flag = true;
        
        
        % === Fit GP for mean of residuals
        mean_model = fitrgp(all_inputs, all_mu', ...
            'Basis', basis_fn, ...
            'KernelFunction', kernel_fn, ...
            'Standardize', standardize_flag);
        
        % === Fit GP for variance of residuals
        std_model = fitrgp(all_inputs, all_sigma2', ...
            'Basis', basis_fn, ...
            'KernelFunction', kernel_fn, ...
            'Standardize', standardize_flag);

        % Save models
        save(fullfile(data_dir, sprintf('arx_gp_%s_%s.mat', output_var, regime)), ...
            'best_model', 'mean_model', 'std_model');

        fprintf('✅ Trained and saved GP models for %s | Sensor: %s\n', upper(regime), output_var);

        % Plot ARX fit
        figure;
        plot(Y_train, 'b'); hold on;
        plot(Y_hat_train, 'r--');
        legend('True','ARX Prediction');
        title(sprintf('ARX Fit - %s [%s]', output_var, upper(regime)));
        grid on;

        % Plot GP surface for mean
        figure;
        [X1, X2] = meshgrid(linspace(min(all_inputs(:,1)), max(all_inputs(:,1)), 50), ...
                            linspace(min(all_inputs(:,2)), max(all_inputs(:,2)), 50));
        X_grid = [X1(:), X2(:)];
        Y_pred = predict(mean_model, X_grid);
        surf(X1, X2, reshape(Y_pred, size(X1)));
        xlabel('Scheduling Variable'); ylabel('Torque'); zlabel('Residual Mean');
        title(sprintf('GP Surface - Residual Mean [%s - %s]', upper(output_var), upper(regime)));
        shading interp; colorbar;

        % Plot GP surface for variance
        figure;
        Y_pred_var = predict(std_model, X_grid);
        surf(X1, X2, reshape(Y_pred_var, size(X1)));
        xlabel('Scheduling Variable'); ylabel('Torque'); zlabel('Residual Variance');
        title(sprintf('GP Surface - Residual Variance [%s - %s]', upper(output_var), upper(regime)));
        shading interp; colorbar;
    end
end

%% --- Helper Functions ---
function X_trimmed = trim_edges(X, trim_sec, fs)
    n_trim = round(trim_sec * fs);
    X_trimmed = X(n_trim+1:end-n_trim);
end

function X_table = flatten_struct_cells(X_struct_cell)
    all_tables = cell(size(X_struct_cell));
    for i = 1:length(X_struct_cell)
        x = X_struct_cell{i};
        this_table = [table(x.speed(:), x.torque(:), 'VariableNames', {'speed', 'torque'}), x.others];
        all_tables{i} = this_table;
    end
    X_table = vertcat(all_tables{:});
end
