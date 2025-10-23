function T_out = load_single_file(folder, fname, fs_original, fs_target, column_names)
    % Load full CSV file
    fpath = fullfile(folder, fname);
    T = readtable(fpath);

    % Check for expected columns
    if ~all(ismember(column_names, T.Properties.VariableNames))
        error('Missing expected columns in file: %s', fname);
    end

    % Select only needed columns
    T = T(:, column_names);

    % If sampling rates match, return directly
    if fs_original == fs_target
        T_out = T;
        return;
    end

    % Resample numeric columns
    T_out = table();
    for i = 1:numel(column_names)
        col = column_names{i};
        data = T.(col);

        if isnumeric(data)
            % Resample using MATLAB’s built-in function with anti-aliasing
            data_resampled = resample(data, fs_target, fs_original); 
            T_out.(col) = data_resampled;
        else
            warning('Column %s is not numeric and will be skipped.', col);
        end
    end
end
