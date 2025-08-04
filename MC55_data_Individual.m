%% Set main paths and parameters
clear all; clc; close all
data_dir = fullfile(pwd, 'Gearbox_VWCs', 'MCC5-THU');
save_dir = data_dir;

% Load all file names from .csv
S = dir(fullfile(data_dir, '*.csv'));
all_files = string({S.name});

% Identify healthy and faulty CSV files
healthy_files_all = all_files(startsWith(all_files, 'health'));
faulty_files_all = all_files(~startsWith(all_files, 'health'));

% Split regimes for healthy files
healthy_speed = healthy_files_all(contains(healthy_files_all, 'speed_circulation'));
healthy_torque = healthy_files_all(contains(healthy_files_all, 'torque_circulation'));

% Split regimes for faulty files
faulty_speed = faulty_files_all(contains(faulty_files_all, 'speed_circulation'));
faulty_torque = faulty_files_all(contains(faulty_files_all, 'torque_circulation'));

fs_original = 12800; % Hz
fs_target = 6400;    % Hz

column_names = { ...
    'speed', 'torque', ...
    'motor_vibration_x', 'motor_vibration_y', 'motor_vibration_z', ...
    'gearbox_vibration_x', 'gearbox_vibration_y', 'gearbox_vibration_z'};

%% Function for correcting speed and torque
process_signals = @(T) struct( ...
    'speed', extract_speed(T.speed), ...
    'torque', T.torque * 6, ...
    'others', T(:, 3:end));

%% === Process Healthy Splits for Both Regimes ===
regimes = {'speed', 'torque'};
healthy_sets = {healthy_speed, healthy_torque};

for r = 1:2
    regime = regimes{r};
    healthy_files = healthy_sets{r};

    rng(r); % Different random shuffle per regime
    num_total = numel(healthy_files);
    num_train = round(0.7 * num_total);
    num_valid = round(0.15 * num_total);
    num_test  = num_total - num_train - num_valid;

    shuffled = healthy_files(randperm(num_total));
    train_files = shuffled(1:num_train);
    valid_files = shuffled(num_train+1 : num_train+num_valid);
    test_files  = shuffled(num_train+num_valid+1 : end);

    X_train = load_and_process_multiple(data_dir, train_files, fs_original, fs_target, column_names, process_signals);
    X_valid = load_and_process_multiple(data_dir, valid_files, fs_original, fs_target, column_names, process_signals);
    X_test  = load_and_process_multiple(data_dir, test_files, fs_original, fs_target, column_names, process_signals);

    % Save with regime-specific filenames
    save(fullfile(save_dir, ['X_train_', regime, '.mat']), 'X_train', 'fs_target');
    save(fullfile(save_dir, ['X_valid_', regime, '.mat']), 'X_valid', 'fs_target');
    save(fullfile(save_dir, ['X_test_healthy_', regime, '.mat']), 'X_test', 'fs_target');

    fprintf('✅ Healthy %s regime splits saved.\n', upper(regime));
end

%% === Process All Faulty Files (Unchanged, all regimes) ===
for i = 1:length(faulty_files_all)
    fname = faulty_files_all(i);
    T_fault = load_single_file(data_dir, fname, fs_original, fs_target, column_names);
    T_fault_processed = process_signals(T_fault);

    mat_name = "X_" + erase(fname, '.csv') + ".mat";
    save(fullfile(save_dir, mat_name), 'T_fault_processed', 'fs_target');
    fprintf('✅ Saved fault data: %s --> %s\n', fname, mat_name);
end

disp('🎯 All datasets (healthy and faulty) processed and saved.');

%% --- Helper Functions ---

function speed_interp = extract_speed(raw)
    % Step 1: Binarize speed signal
    bin_signal = raw;
    bin_signal(bin_signal <= 2) = 0;
    bin_signal(bin_signal > 2)  = 1;

    % Step 2: Rising edges
    rising_edges_index = find(diff(bin_signal) == 1);

    % Step 3: Create time base (60s duration, matching signal length)
    time = linspace(0, 60, numel(bin_signal));

    % Step 4: Get rising edge times
    rising_time_point = time(rising_edges_index);

    % Step 5: Estimate speed (from intervals)
    period = diff(rising_time_point);
    frequency = 1 ./ period;
    speed = frequency * 60;

    % Step 6: Average time point per interval
    mean_time_point = movmean(rising_time_point, 2);
    mean_time_point = mean_time_point(2:end);

    % Step 7: Interpolate back to full time vector
    speed_interp = interp1(mean_time_point, speed, time, 'linear', 'extrap');
end

function data_all = load_and_process_multiple(folder, files, fs_orig, fs_tgt, colnames, processor)
    data_all = {};
    for i = 1:length(files)
        T = load_single_file(folder, files(i), fs_orig, fs_tgt, colnames);
        data_all{i} = processor(T);
    end
end
