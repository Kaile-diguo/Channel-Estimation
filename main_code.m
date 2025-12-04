%% 5G NR Channel Estimation Window Function Performance Evaluation Simulation - Optimized Version
% Author: Huang Jianing   Ou Lepeng
% Description: Compare the performance of different window functions in 5G NR channel estimation (significantly reduces BER)

clear all; close all; clc;
rng(42); % Set random seed for reproducibility

%% 1. Simulation Parameter Settings - Optimized
disp('=== 5G NR Channel Estimation Simulation - Window Function Performance Comparison (Optimized) ===');

% System parameters (5G NR compliant)
params.carrierFreq = 2.6e9;        % Carrier frequency 2.6 GHz
params.subcarrierSpacing = 30e3;   % Subcarrier spacing 30 kHz
params.FFTsize = 256;              % FFT size
params.numSubcarriers = 256;       % Number of subcarriers
params.CP_length = 64;             % Cyclic prefix length
params.modulation = 'QPSK';        % Modulation scheme
params.pilotDensity = 1/4;         % Pilot density

params.numOFDMsymbols = 14;        % Number of OFDM symbols

% Channel parameters
params.SNR_dB = 10;                % Signal-to-noise ratio (dB) - for single simulation display
params.SNR_range = -5:2:15;        % SNR range for Monte Carlo simulation
params.selected_SNRs = [-5, 0, 5, 10, 15]; % Selected SNR points for bar charts
params.channelType = 'UMa';        % Channel type: 'UMa' or 'Umi'

% Window function parameters
window_types = {'Rectangular', 'Hamming', 'Kaiser', 'Blackman', 'Chebyshev'};
num_windows = length(window_types);
kaiser_beta = 3;                   % Kaiser window beta parameter
chebyshev_ripple = 50;             % Chebyshev window sidelobe attenuation (dB)

% Simulation control
params.numMonteCarlo = 200;        % Number of Monte Carlo simulations

% New: Optimization parameters
params.enableNoiseReduction = true; % Enable noise reduction
params.noiseThreshold = 0.1;       % Noise threshold
params.enablePowerNormalization = true; % Enable power normalization
params.snrEstimationMethod = 'pilot'; % SNR estimation method: 'pilot' or 'theory'
params.equalizerType = 'MMSE';     % Equalizer type: 'MMSE' or 'ZF'

%% 2. Signal Generation - 5G NR QPSK Signal
disp('Step 1: Generating 5G NR QPSK signal...');

% Generate QPSK modulation symbols
numDataSymbols = params.numSubcarriers * params.numOFDMsymbols;
dataBits = randi([0 1], numDataSymbols*2, 1);
modSymbols = qpsk_modulate(dataBits);

% Reshape into OFDM grid
ofdmGrid = reshape(modSymbols, params.numSubcarriers, params.numOFDMsymbols);

% Add pilots
pilotIndices = 1:round(1/params.pilotDensity):params.numSubcarriers;
dataIndices = setdiff(1:params.numSubcarriers, pilotIndices);

% Generate random QPSK pilot symbols (independent from data symbols)
pilotSymbols = qpsk_modulate(randi([0 1], length(pilotIndices)*2, 1));
pilotSymbols = reshape(pilotSymbols, length(pilotIndices), 1);

% Insert pilots
ofdmGrid_withPilots = ofdmGrid;
for symIdx = 1:params.numOFDMsymbols
    ofdmGrid_withPilots(pilotIndices, symIdx) = pilotSymbols;
end

% OFDM modulation (IFFT)
txSignal_time = zeros(params.FFTsize + params.CP_length, params.numOFDMsymbols);
for symIdx = 1:params.numOFDMsymbols
    freqSymbol = ofdmGrid_withPilots(:, symIdx);
    timeSymbol = ifft(freqSymbol, params.FFTsize);
    txSignal_time(:, symIdx) = [timeSymbol(end-params.CP_length+1:end); timeSymbol];
end

% Flatten into time series
txSignal = txSignal_time(:);

% Figure 1: Generated 5G NR signal
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1);
plot(real(ofdmGrid_withPilots(:,1)), 'b.-', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
plot(imag(ofdmGrid_withPilots(:,1)), 'r.-', 'LineWidth', 1.5, 'MarkerSize', 8);
plot(pilotIndices, real(ofdmGrid_withPilots(pilotIndices,1)), 'go', 'MarkerSize', 8, 'LineWidth', 2);
xlabel('Subcarrier Index'); ylabel('Amplitude');
title('Frequency Domain Signal - Real(Blue) and Imaginary(Red)');
legend('Real', 'Imaginary', 'Pilot Positions', 'Location', 'best');
grid on;

subplot(1,2,2);
num_samples_show = min(3 * (params.FFTsize + params.CP_length), length(txSignal));
plot(real(txSignal(1:num_samples_show)), 'b-', 'LineWidth', 1.5);
hold on;
plot(imag(txSignal(1:num_samples_show)), 'r-', 'LineWidth', 1.5);
xlabel('Sample Index'); ylabel('Amplitude');
title(sprintf('Time Domain Signal (First %d Samples)', num_samples_show));
legend('Real', 'Imaginary', 'Location', 'best');
grid on;
sgtitle('Figure 1: Generated 5G NR QPSK Signal (with Pilots)');

%% 3. Improved Channel Propagation Model
disp('Step 2: Propagating signal through channel...');

% Set parameters based on selected channel type
if strcmp(params.channelType, 'UMa')
    numPaths = 12;
    pathDelays = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0] * 1e-6;
    avgPathGains = [0, -1.5, -3.0, -4.5, -7.5, -9.0, -12.0, -14.0, -16.0, -18.0, -20.0, -22.0];
    dopplerShift = 5; % Hz
else
    numPaths = 8;
    pathDelays = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35] * 1e-6;
    avgPathGains = [0, -2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -14.0];
    dopplerShift = 10; % Hz
end

% Sampling rate
sampleRate = params.subcarrierSpacing * params.FFTsize;
Ts = 1/sampleRate;

% Create more realistic time-varying multipath channel
rxSignal_channel = zeros(size(txSignal));
for p = 1:numPaths
    delay_samples = round(pathDelays(p) / Ts);
    path_gain = 10^(avgPathGains(p)/20);
    
    % Add random initial phase
    random_phase = 2*pi*rand();
    time_var = exp(1j*(2*pi*dopplerShift*(0:length(txSignal)-1)'*Ts + random_phase));
    
    if delay_samples < length(txSignal)
        rxSignal_channel(delay_samples+1:end) = rxSignal_channel(delay_samples+1:end) + ...
            path_gain * txSignal(1:end-delay_samples) .* time_var(1:end-delay_samples);
    end
end

% Add AWGN noise
signalPower = mean(abs(rxSignal_channel).^2);
noisePower = signalPower / (10^(params.SNR_dB/10));
noise = sqrt(noisePower/2) * (randn(size(rxSignal_channel)) + 1i*randn(size(rxSignal_channel)));
rxSignal_noisy = rxSignal_channel + noise;

% Calculate actual received SNR
actual_SNR = 10*log10(signalPower / noisePower);
fprintf('Target SNR: %.1f dB, Actual SNR: %.1f dB\n', params.SNR_dB, actual_SNR);

%% 4. Receiver Processing - OFDM Demodulation
disp('Step 3: Receiver OFDM demodulation...');

% Reshape into OFDM symbols
rxSignal_matrix = reshape(rxSignal_noisy, params.FFTsize + params.CP_length, params.numOFDMsymbols);

% Remove cyclic prefix and perform FFT
rxSignal_freq = zeros(params.numSubcarriers, params.numOFDMsymbols);
for symIdx = 1:params.numOFDMsymbols
    timeSymbol = rxSignal_matrix(params.CP_length+1:end, symIdx);
    rxSignal_freq(:, symIdx) = fft(timeSymbol, params.FFTsize);
end

%% 5. Improved Channel Estimation
disp('Step 4: Improved channel estimation...');

% Extract received signal at pilot positions
rxPilots = rxSignal_freq(pilotIndices, :);

% Improved LS estimation
H_LS_pilots = rxPilots ./ pilotSymbols;

% Use improved channel estimation algorithm
H_LS_full = zeros(params.numSubcarriers, params.numOFDMsymbols);
for symIdx = 1:params.numOFDMsymbols
    H_LS_full(:, symIdx) = improved_channel_estimation(rxPilots(:, symIdx), pilotSymbols, ...
        pilotIndices, params.numSubcarriers);
end

% True channel frequency response
maxDelaySamples = round(max(pathDelays) / Ts) + 1;
channelIR = zeros(maxDelaySamples, 1);
for p = 1:numPaths
    delay_samples = round(pathDelays(p) / Ts);
    if delay_samples < maxDelaySamples
        channelIR(delay_samples + 1) = channelIR(delay_samples + 1) + 10^(avgPathGains(p)/20);
    end
end
H_true = fft(channelIR, params.FFTsize);

% Figure 3: Initial channel estimation results
figure('Position', [100, 100, 1200, 400]);
subplot(1,3,1);
plot(abs(H_true), 'k-', 'LineWidth', 2.5);
hold on;
plot(abs(H_LS_full(:,1)), 'b--', 'LineWidth', 1.5);
xlabel('Subcarrier Index'); ylabel('Amplitude');
title('Channel Frequency Response Magnitude Comparison');
legend('True Channel', 'LS Estimated Channel', 'Location', 'best');
grid on;

subplot(1,3,2);
phase_true = unwrap(angle(H_true));
phase_est = unwrap(angle(H_LS_full(:,1)));
plot(phase_true, 'k-', 'LineWidth', 2.5);
hold on;
plot(phase_est, 'r--', 'LineWidth', 1.5);
xlabel('Subcarrier Index'); ylabel('Phase (Radians)');
title('Channel Frequency Response Phase Comparison');
legend('True Channel', 'LS Estimated Channel', 'Location', 'best');
grid on;

% Channel estimation error analysis
subplot(1,3,3);
estimation_error = H_LS_full(:,1) - H_true;
plot(abs(estimation_error), 'm-', 'LineWidth', 1.5);
xlabel('Subcarrier Index'); ylabel('Estimation Error Magnitude');
title(sprintf('Channel Estimation Error (MSE=%.4f)', mean(abs(estimation_error).^2)));
grid on;
sgtitle('Figure 3: Improved Channel Estimation Results and Error Analysis');

%% 6. Improved Window Function Application Method
disp('Step 5: Applying window functions in time domain...');

% Create different window functions
window_length = params.FFTsize;
window_rect = rectwin(window_length);
window_hamming = hamming(window_length);
window_kaiser = kaiser(window_length, kaiser_beta);
window_blackman = blackman(window_length);
window_chebyshev = chebwin(window_length, chebyshev_ripple);

% Apply window functions to channel estimation
H_windowed = zeros(params.numSubcarriers, num_windows, params.numOFDMsymbols);
windows = {window_rect, window_hamming, window_kaiser, window_blackman, window_chebyshev};

for symIdx = 1:params.numOFDMsymbols
    for wIdx = 1:num_windows
        % Method: Time domain windowing + noise reduction
        h_time = ifft(H_LS_full(:, symIdx));
        
        % Apply noise reduction
        if params.enableNoiseReduction
            h_time = time_domain_noise_reduction(h_time, params.noiseThreshold);
        end
        
        % Apply window function in time domain
        h_time_windowed = h_time .* windows{wIdx};
        
        % Transform back to frequency domain
        H_windowed(:, wIdx, symIdx) = fft(h_time_windowed);
        
        % Power normalization
        if params.enablePowerNormalization
            power_windowed = mean(abs(H_windowed(:, wIdx, symIdx)).^2);
            power_original = mean(abs(H_LS_full(:, symIdx)).^2);
            if power_windowed > 0
                H_windowed(:, wIdx, symIdx) = H_windowed(:, wIdx, symIdx) * sqrt(power_original/power_windowed);
            end
        end
    end
end

% Figure 4: Channel estimation after different window function processing
figure('Position', [100, 100, 1400, 900]);

for wIdx = 1:num_windows
    subplot(3, 2, wIdx);
    
    % Plot window function
    plot(windows{wIdx}, 'b-', 'LineWidth', 2);
    hold on;
    
    % Plot weighted channel estimation (first symbol)
    plot(abs(H_windowed(:, wIdx, 1)), 'r-', 'LineWidth', 1.5);
    
    xlabel('Subcarrier Index'); ylabel('Amplitude');
    title(sprintf('%s Window (Red: Weighted Channel Estimation)', window_types{wIdx}));
    legend('Window Function', 'Weighted Channel Estimation', 'Location', 'best');
    grid on;
    
    % Calculate and display MSE for this window function
    mse_value = mean(abs(H_windowed(:, wIdx, 1) - H_true).^2);
    text(0.05, 0.9, sprintf('MSE=%.4f', mse_value), 'Units', 'normalized', ...
        'BackgroundColor', 'white', 'FontSize', 9);
    
    % Add window function parameter information
    if strcmp(window_types{wIdx}, 'Kaiser')
        text(0.05, 0.8, sprintf('β=%.1f', kaiser_beta), 'Units', 'normalized', ...
            'BackgroundColor', 'white', 'FontSize', 9);
    elseif strcmp(window_types{wIdx}, 'Chebyshev')
        text(0.05, 0.8, sprintf('Sidelobe Attenuation=%d dB', chebyshev_ripple), ...
            'Units', 'normalized', 'BackgroundColor', 'white', 'FontSize', 9);
    end
end

% Compare all window functions in the last subplot
subplot(3, 2, 6);
colors = {'b', 'r', 'g', 'm', 'c'};
for wIdx = 1:num_windows
    plot(abs(H_windowed(:, wIdx, 1)), colors{wIdx}, 'LineWidth', 1.5);
    hold on;
end
plot(abs(H_true), 'k--', 'LineWidth', 2.5);
xlabel('Subcarrier Index'); ylabel('Amplitude');
title('Comparison of Channel Estimation After All Window Functions');
legend([window_types, {'True Channel'}], 'Location', 'best');
grid on;
sgtitle('Figure 4: Channel Estimation After Different Window Functions (with MSE Metrics)');

%% 7. Improved Demodulation and Metric Calculation
disp('Step 6: Demodulation using channel estimation after window function processing...');

% Initialize storage
eqSymbols_all = cell(num_windows, 1);
demodBits_all = cell(num_windows, 1);
BER_single = zeros(1, num_windows);
MSE_single = zeros(1, num_windows);

% Calculate actual usable SNR
SNR_linear = 10^(actual_SNR/10);

% Process each window function
for wIdx = 1:num_windows
    % Collect all equalized symbols and demodulated bits
    eqSymbols_temp = [];
    demodBits_temp = [];
    
    for symIdx = 1:params.numOFDMsymbols
        % Use channel estimation for this window function
        H_est = H_windowed(dataIndices, wIdx, symIdx);
        
        % Select equalization method based on equalizer type
        if strcmp(params.equalizerType, 'MMSE')
            % MMSE equalizer
            epsilon = 1e-10; % Avoid division by zero
            W_eq = conj(H_est) ./ (abs(H_est).^2 + 1/SNR_linear + epsilon);
        else
            % ZF equalizer
            W_eq = 1 ./ (H_est + 1e-10);
        end
        
        % Equalization
        eqSymbols = rxSignal_freq(dataIndices, symIdx) .* W_eq;
        
        % Amplitude normalization
        eq_power = mean(abs(eqSymbols).^2);
        if eq_power > 0
            eqSymbols = eqSymbols / sqrt(eq_power) * sqrt(2); % QPSK average power is 2
        end
        
        % QPSK demodulation
        demodBits = qpsk_demodulate(eqSymbols);
        
        % Collect results
        eqSymbols_temp = [eqSymbols_temp; eqSymbols];
        demodBits_temp = [demodBits_temp; demodBits];
    end
    
    % Store results
    eqSymbols_all{wIdx} = eqSymbols_temp;
    demodBits_all{wIdx} = demodBits_temp;
    
    % Calculate BER
    % Note: We need to compare with original data bits
    % Find corresponding data bit positions
    data_symbol_indices = [];
    for symIdx = 1:params.numOFDMsymbols
        data_symbol_indices = [data_symbol_indices; ...
            dataIndices' + (symIdx-1)*params.numSubcarriers];
    end
    
    % Get corresponding transmitted bits
    txDataBits = [];
    for i = 1:length(data_symbol_indices)
        symbol_idx = data_symbol_indices(i);
        bit_idx1 = (symbol_idx-1)*2 + 1;
        bit_idx2 = (symbol_idx-1)*2 + 2;
        txDataBits = [txDataBits; dataBits(bit_idx1:bit_idx2)];
    end
    
    % Ensure length matches
    min_len = min(length(demodBits_temp), length(txDataBits));
    BER_single(wIdx) = sum(demodBits_temp(1:min_len) ~= txDataBits(1:min_len)) / min_len;
    
    % Calculate MSE (first symbol)
    MSE_single(wIdx) = mean(abs(H_windowed(:, wIdx, 1) - H_true).^2);
end

% New: Calculate BER of raw signal (without equalization) as baseline
rx_data_no_eq = [];
for symIdx = 1:params.numOFDMsymbols
    rx_data_no_eq = [rx_data_no_eq; rxSignal_freq(dataIndices, symIdx)];
end
rx_bits_no_eq = qpsk_demodulate(rx_data_no_eq);
min_len_no_eq = min(length(rx_bits_no_eq), length(txDataBits));
BER_no_eq = sum(rx_bits_no_eq(1:min_len_no_eq) ~= txDataBits(1:min_len_no_eq)) / min_len_no_eq;

% Figure 5: Demodulation results constellation diagram
figure('Position', [100, 100, 1400, 800]);

for wIdx = 1:num_windows
    subplot(2, 3, wIdx);
    eqSymbols_plot = eqSymbols_all{wIdx};
    if length(eqSymbols_plot) > 1000
        eqSymbols_plot = eqSymbols_plot(1:1000); % Limit display points
    end
    scatter(real(eqSymbols_plot), imag(eqSymbols_plot), 20, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    
    % Plot theoretical constellation points
    qpskConstellation = [(1+1i)/sqrt(2), (1-1i)/sqrt(2), (-1+1i)/sqrt(2), (-1-1i)/sqrt(2)];
    plot(real(qpskConstellation), imag(qpskConstellation), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    
    xlim([-2, 2]); ylim([-2, 2]);
    xlabel('In-phase Component'); ylabel('Quadrature Component');
    title(sprintf('%s Window\nBER=%.4f, MSE=%.4f', window_types{wIdx}, BER_single(wIdx), MSE_single(wIdx)));
    grid on;
    axis square;
end

% Show unequalized constellation diagram in the last subplot for comparison
subplot(2, 3, 6);
if length(rx_data_no_eq) > 1000
    rx_data_plot = rx_data_no_eq(1:1000);
else
    rx_data_plot = rx_data_no_eq;
end
scatter(real(rx_data_plot), imag(rx_data_plot), 20, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot(real(qpskConstellation), imag(qpskConstellation), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlim([-2, 2]); ylim([-2, 2]);
xlabel('In-phase Component'); ylabel('Quadrature Component');
title(sprintf('Unequalized Received Signal\nBER=%.4f', BER_no_eq));
grid on;
axis square;
sgtitle(sprintf('Figure 5: QPSK Constellation After Equalization Using Different Window Functions (Actual SNR=%.1f dB)', actual_SNR));

%% 8. Monte Carlo Simulation - Performance Curves
disp('Step 7: Performing Monte Carlo simulation to generate performance curves...');

% Modify window function list: Replace Blackman window with "No Window" control group
window_types = {'Rectangular', 'Hamming', 'Kaiser', 'No Window', 'Chebyshev'};
num_windows = length(window_types);

% Preallocate storage
BER_all = zeros(length(params.SNR_range), num_windows);
MSE_all = zeros(length(params.SNR_range), num_windows);

% Create window functions (precomputed) - Replace Blackman window with all-ones vector (no window)
windows_mc = cell(1, num_windows);
windows_mc{1} = rectwin(params.FFTsize);
windows_mc{2} = hamming(params.FFTsize);
windows_mc{3} = kaiser(params.FFTsize, kaiser_beta);
windows_mc{4} = ones(params.FFTsize, 1);  % No window: all-ones vector
windows_mc{5} = chebwin(params.FFTsize, chebyshev_ripple);

% Simplify Monte Carlo simulation, use only the first symbol for speed
for snrIdx = 1:length(params.SNR_range)
    currentSNR = params.SNR_range(snrIdx);
    fprintf('  Processing SNR = %d dB...\n', currentSNR);
    
    % Temporary storage for results at current SNR
    BER_temp = zeros(params.numMonteCarlo, num_windows);
    MSE_temp = zeros(params.numMonteCarlo, num_windows);
    
    % Monte Carlo loop
    for mcIdx = 1:params.numMonteCarlo
        % Regenerate random data (simplified using only one symbol)
        numDataSymbols_mc = params.numSubcarriers * 1; % Use only one symbol
        dataBits_mc = randi([0 1], numDataSymbols_mc*2, 1);
        modSymbols_mc = qpsk_modulate(dataBits_mc);
        ofdmGrid_mc = reshape(modSymbols_mc, params.numSubcarriers, 1);
        
        % Insert pilots
        pilotSymbols_mc = qpsk_modulate(randi([0 1], length(pilotIndices)*2, 1));
        pilotSymbols_mc = reshape(pilotSymbols_mc, length(pilotIndices), 1);
        
        ofdmGrid_withPilots_mc = ofdmGrid_mc;
        ofdmGrid_withPilots_mc(pilotIndices, 1) = pilotSymbols_mc;
        
        % OFDM modulation
        txSignal_time_mc = zeros(params.FFTsize + params.CP_length, 1);
        freqSymbol = ofdmGrid_withPilots_mc(:, 1);
        timeSymbol = ifft(freqSymbol, params.FFTsize);
        txSignal_time_mc(:, 1) = [timeSymbol(end-params.CP_length+1:end); timeSymbol];
        txSignal_mc = txSignal_time_mc(:);
        
        % Pass through channel
        rxSignal_channel_mc = zeros(size(txSignal_mc));
        for p = 1:numPaths
            delay_samples = round(pathDelays(p) / Ts);
            path_gain = 10^(avgPathGains(p)/20);
            random_phase = 2*pi*rand();
            time_var_mc = exp(1j*(2*pi*dopplerShift*(0:length(txSignal_mc)-1)'*Ts + random_phase));
            
            if delay_samples < length(txSignal_mc)
                rxSignal_channel_mc(delay_samples+1:end) = rxSignal_channel_mc(delay_samples+1:end) + ...
                    path_gain * txSignal_mc(1:end-delay_samples) .* time_var_mc(1:end-delay_samples);
            end
        end
        
        % Add AWGN noise
        signalPower_mc = mean(abs(rxSignal_channel_mc).^2);
        noisePower_mc = signalPower_mc / (10^(currentSNR/10));
        noise_mc = sqrt(noisePower_mc/2) * (randn(size(rxSignal_channel_mc)) + 1i*randn(size(rxSignal_channel_mc)));
        rxSignal_noisy_mc = rxSignal_channel_mc + noise_mc;
        
        % OFDM demodulation
        rxSignal_matrix_mc = reshape(rxSignal_noisy_mc, params.FFTsize + params.CP_length, 1);
        timeSymbol_mc = rxSignal_matrix_mc(params.CP_length+1:end, 1);
        rxSignal_freq_mc = fft(timeSymbol_mc, params.FFTsize);
        
        % Create true channel frequency response for current realization
        channelIR_mc = zeros(maxDelaySamples, 1);
        for p = 1:numPaths
            delay_samples = round(pathDelays(p) / Ts);
            if delay_samples < maxDelaySamples
                channelIR_mc(delay_samples + 1) = channelIR_mc(delay_samples + 1) + 10^(avgPathGains(p)/20);
            end
        end
        H_true_mc = fft(channelIR_mc, params.FFTsize);
        
        % Initial LS channel estimation
        rxPilots_mc = rxSignal_freq_mc(pilotIndices);
        H_LS_pilots_mc = rxPilots_mc ./ pilotSymbols_mc;
        
        % Use improved channel estimation algorithm - ensure column vector return
        H_LS_full_mc = improved_channel_estimation(rxPilots_mc, pilotSymbols_mc, ...
            pilotIndices, params.numSubcarriers);
        
        % Ensure H_LS_full_mc is a column vector
        if size(H_LS_full_mc, 2) > 1
            H_LS_full_mc = H_LS_full_mc.';
        end
        
        % Apply different window functions
        for wIdx = 1:num_windows
            % For "No Window" case, use original LS estimation
            if strcmp(window_types{wIdx}, 'No Window')
                % Directly use LS estimation, no window processing
                H_windowed_mc = H_LS_full_mc;
            else
                % Other window functions: time domain windowing method
                h_time_mc = ifft(H_LS_full_mc);
                
                % Apply noise reduction
                if params.enableNoiseReduction
                    h_time_mc = time_domain_noise_reduction(h_time_mc, params.noiseThreshold);
                end
                
                h_time_windowed_mc = h_time_mc .* windows_mc{wIdx};
                H_windowed_mc = fft(h_time_windowed_mc);
                
                % Power normalization
                if params.enablePowerNormalization
                    power_windowed = mean(abs(H_windowed_mc).^2);
                    power_original = mean(abs(H_LS_full_mc).^2);
                    if power_windowed > 0
                        H_windowed_mc = H_windowed_mc * sqrt(power_original/power_windowed);
                    end
                end
            end
            
            % Ensure H_windowed_mc is a column vector
            if size(H_windowed_mc, 2) > 1
                H_windowed_mc = H_windowed_mc.';
            end
            
            % Equalization
            H_est_mc = H_windowed_mc(dataIndices);
            SNR_linear_mc = 10^(currentSNR/10);
            
            if strcmp(params.equalizerType, 'MMSE')
                epsilon = 1e-10;
                W_eq_mc = conj(H_est_mc) ./ (abs(H_est_mc).^2 + 1/SNR_linear_mc + epsilon);
            else
                W_eq_mc = 1 ./ (H_est_mc + 1e-10);
            end
            
            eqSymbols_mc = rxSignal_freq_mc(dataIndices) .* W_eq_mc;
            
            % Amplitude normalization
            eq_power = mean(abs(eqSymbols_mc).^2);
            if eq_power > 0
                eqSymbols_mc = eqSymbols_mc / sqrt(eq_power) * sqrt(2);
            end
            
            % Demodulation
            demodBits_mc = qpsk_demodulate(eqSymbols_mc);
            
            % Calculate BER
            % Get corresponding transmitted bits
            txDataBits_mc = [];
            for i = 1:length(dataIndices)
                symbol_idx = dataIndices(i);
                bit_idx1 = (symbol_idx-1)*2 + 1;
                bit_idx2 = (symbol_idx-1)*2 + 2;
                txDataBits_mc = [txDataBits_mc; dataBits_mc(bit_idx1:bit_idx2)];
            end
            
            min_len_mc = min(length(demodBits_mc), length(txDataBits_mc));
            BER_temp(mcIdx, wIdx) = sum(demodBits_mc(1:min_len_mc) ~= txDataBits_mc(1:min_len_mc)) / min_len_mc;
            
            % Calculate MSE - ensure vector sizes are consistent
            % Ensure H_true_mc is a column vector
            H_true_mc_col = H_true_mc(:);
            H_windowed_mc_col = H_windowed_mc(:);
            
            % Ensure same size
            if length(H_true_mc_col) ~= length(H_windowed_mc_col)
                % If sizes differ, adjust to smaller size
                min_len_mse = min(length(H_true_mc_col), length(H_windowed_mc_col));
                H_true_mc_col = H_true_mc_col(1:min_len_mse);
                H_windowed_mc_col = H_windowed_mc_col(1:min_len_mse);
            end
            
            % Calculate MSE
            MSE_temp(mcIdx, wIdx) = mean(abs(H_windowed_mc_col - H_true_mc_col).^2);
        end
    end
    
    % Calculate average BER and MSE at current SNR
    BER_all(snrIdx, :) = mean(BER_temp, 1);
    MSE_all(snrIdx, :) = mean(MSE_temp, 1);
end

%% 9. Plot Performance Curves
disp('Step 8: Plotting performance curves...');

% Figure 6: BER vs SNR curve
figure('Position', [100, 100, 1000, 600]);
colors = {'b-o', 'r-s', 'g-^', 'm-d', 'c-v'};
markerSize = 8;
lineWidth = 2;

for wIdx = 1:num_windows
    semilogy(params.SNR_range, BER_all(:, wIdx), colors{wIdx}, ...
        'MarkerSize', markerSize, 'LineWidth', lineWidth);
    hold on;
end

% Theoretical QPSK BER curve (AWGN channel)
SNR_linear_theory = 10.^(params.SNR_range/10);
BER_theory_QPSK = 0.5 * erfc(sqrt(SNR_linear_theory));
semilogy(params.SNR_range, BER_theory_QPSK, 'k--', 'LineWidth', 3, 'MarkerSize', 10);

xlabel('SNR (dB)'); ylabel('Bit Error Rate (BER)');
title(sprintf('%s Channel - BER Performance Curves for Different Window Functions', params.channelType));
grid on;
legend([window_types, {'Theoretical QPSK (AWGN)'}], 'Location', 'southwest');
xlim([min(params.SNR_range), max(params.SNR_range)]);
sgtitle('Figure 6: BER vs SNR Performance Curves for Different Window Functions');

%% 10. Plot Performance Line Charts
disp('Step 9: Plotting performance line charts...');

% Find indices of selected SNR points
selected_SNR_indices = zeros(1, length(params.selected_SNRs));
for i = 1:length(params.selected_SNRs)
    [~, selected_SNR_indices(i)] = min(abs(params.SNR_range - params.selected_SNRs(i)));
end

% Extract BER and MSE data at selected SNR points
BER_selected = BER_all(selected_SNR_indices, :);
MSE_selected = MSE_all(selected_SNR_indices, :);

% Set color scheme
colors_line = [0.2, 0.4, 0.8;   % Blue - Rectangular
               0.8, 0.2, 0.2;   % Red - Hamming
               0.2, 0.8, 0.4;   % Green - Kaiser
               0.8, 0.8, 0.2;   % Yellow - No Window (control group)
               0.6, 0.2, 0.8];  % Purple - Chebyshev

% Set line styles and markers
line_styles = {'-o', '-s', '-^', '-d', '-v'};
line_width = 2;
marker_size = 8;

% Figure 7: BER performance line chart
figure('Position', [100, 100, 1000, 600]);

% Plot BER curves for all window functions
for wIdx = 1:num_windows
    semilogy(params.SNR_range, BER_all(:, wIdx), line_styles{wIdx}, ...
        'Color', colors_line(wIdx, :), ...
        'LineWidth', line_width, ...
        'MarkerSize', marker_size, ...
        'MarkerFaceColor', colors_line(wIdx, :));
    hold on;
end

xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title(sprintf('%s Channel - BER Performance for Different Window Functions', params.channelType));
grid on;
legend([window_types], 'Location', 'southwest');
xlim([min(params.SNR_range), max(params.SNR_range)]);

% Add grid lines
grid on;
grid minor;

% Figure 8: MSE performance line chart
figure('Position', [100, 100, 1000, 600]);

% Plot MSE curves for all window functions
for wIdx = 1:num_windows
    semilogy(params.SNR_range, MSE_all(:, wIdx), line_styles{wIdx}, ...
        'Color', colors_line(wIdx, :), ...
        'LineWidth', line_width, ...
        'MarkerSize', marker_size, ...
        'MarkerFaceColor', colors_line(wIdx, :));
    hold on;
end

xlabel('SNR (dB)');
ylabel('Mean Square Error (MSE)');
title(sprintf('%s Channel - MSE Performance for Different Window Functions', params.channelType));
grid on;
legend(window_types, 'Location', 'northeast');
xlim([min(params.SNR_range), max(params.SNR_range)]);

% Add grid lines
grid on;
grid minor;

% Add annotation showing the best window function
% Find window function with minimum MSE at highest SNR
[~, best_mse_idx] = min(MSE_all(end, :));
best_snr_idx = length(params.SNR_range);
best_mse_value = MSE_all(best_snr_idx, best_mse_idx);

% Mark best performance point on the curve
text(params.SNR_range(best_snr_idx), best_mse_value * 0.8, ...
    sprintf('Best: %s Window\nMSE=%.2e', window_types{best_mse_idx}, best_mse_value), ...
    'FontSize', 10, 'Color', colors_line(best_mse_idx, :), ...
    'HorizontalAlignment', 'right', 'BackgroundColor', 'white', ...
    'EdgeColor', colors_line(best_mse_idx, :));


%% 10. Plot Horizontal Bar Charts
disp('Step 9: Plotting horizontal bar charts...');

% Find indices of selected SNR points
selected_SNR_indices = zeros(1, length(params.selected_SNRs));
for i = 1:length(params.selected_SNRs)
    [~, selected_SNR_indices(i)] = min(abs(params.SNR_range - params.selected_SNRs(i)));
end

% Extract BER and MSE data at selected SNR points
BER_selected = BER_all(selected_SNR_indices, :);
MSE_selected = MSE_all(selected_SNR_indices, :);

% Set color scheme
colors_bar = [0.2, 0.4, 0.8;   % Blue - Rectangular
              0.8, 0.2, 0.2;   % Red - Hamming
              0.2, 0.8, 0.4;   % Green - Kaiser
              0.8, 0.8, 0.2;   % Yellow - No Window (control group)
              0.6, 0.2, 0.8];  % Purple - Chebyshev

% Figure 7: BER horizontal bar chart
figure('Position', [100, 100, 1200, 600]);
h1 = barh(BER_selected, 'grouped');
xlabel('Bit Error Rate (BER)');
ylabel('SNR (dB)');
set(gca, 'YTickLabel', cellstr(num2str(params.selected_SNRs', '%d dB')));
title(sprintf('%s Channel - BER Performance of Different Window Functions at Various SNRs', params.channelType));
grid on;
set(gca, 'XScale', 'log');

% Set color for each window function
for wIdx = 1:num_windows
    h1(wIdx).FaceColor = colors_bar(wIdx, :);
    h1(wIdx).EdgeColor = 'k';
end

% Add legend
legend(window_types, 'Location', 'best');

% Add value labels - adjust position to make labels closer to bars
for snrIdx = 1:length(params.selected_SNRs)
    % Get all BER values for current SNR
    current_ber_values = BER_selected(snrIdx, :);
    
    % Add label for each window function
    for wIdx = 1:num_windows
        ber_value = current_ber_values(wIdx);
        
        % Format display text
        if ber_value > 0.001
            text_str = sprintf('%.4f', ber_value);
        else
            text_str = sprintf('%.2e', ber_value);
        end
        
        % Calculate text position
        % x position: slightly offset to right of bar, making labels close to bar ends
        if ber_value > 0
            % Adjust offset based on value size to make labels closer to bars
            if ber_value > 0.01
                x_pos = ber_value * 1.02;  % Large values: 2% offset
            elseif ber_value > 0.001
                x_pos = ber_value * 1.05;  % Medium values: 5% offset
            else
                x_pos = ber_value * 1.1;   % Small values: 10% offset
            end
        else
            x_pos = ber_value * 1.05;
        end
        
        % y position: calculate based on window function index, distributing labels vertically
        % Each SNR point has 5 window functions, distribute them evenly vertically
        base_y = snrIdx;  % Base y position for current SNR
        y_spacing = 0.15; % Vertical spacing
        y_pos = base_y - 0.3 + (wIdx-1) * y_spacing;  % Even distribution
        
        % Add text label with white background for better readability
        text(x_pos, y_pos, text_str, ...
            'FontSize', 8, 'Color', 'k', 'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'middle', 'BackgroundColor', 'white', ...
            'EdgeColor', 'k', 'Margin', 1, 'FontWeight', 'normal');
    end
end

% Figure 8: MSE horizontal bar chart
figure('Position', [100, 100, 1200, 600]);
h2 = barh(MSE_selected, 'grouped');
xlabel('Mean Square Error (MSE)');
ylabel('SNR (dB)');
set(gca, 'YTickLabel', cellstr(num2str(params.selected_SNRs', '%d dB')));
title(sprintf('%s Channel - MSE Performance of Different Window Functions at Various SNRs', params.channelType));
grid on;
set(gca, 'XScale', 'log');

% Set color for each window function
for wIdx = 1:num_windows
    h2(wIdx).FaceColor = colors_bar(wIdx, :);
    h2(wIdx).EdgeColor = 'k';
end

% Add legend
legend(window_types, 'Location', 'best');

% Add value labels - adjust position to make labels closer to bars
for snrIdx = 1:length(params.selected_SNRs)
    % Get all MSE values for current SNR
    current_mse_values = MSE_selected(snrIdx, :);
    
    % Add label for each window function
    for wIdx = 1:num_windows
        mse_value = current_mse_values(wIdx);
        
        % Format display text
        if mse_value > 0.001
            text_str = sprintf('%.4f', mse_value);
        else
            text_str = sprintf('%.2e', mse_value);
        end
        
        % Calculate text position
        % x position: slightly offset to right of bar, making labels close to bar ends
        if mse_value > 0
            % Adjust offset based on value size to make labels closer to bars
            if mse_value > 0.01
                x_pos = mse_value * 1.02;  % Large values: 2% offset
            elseif mse_value > 0.001
                x_pos = mse_value * 1.05;  % Medium values: 5% offset
            else
                x_pos = mse_value * 1.1;   % Small values: 10% offset
            end
        else
            x_pos = mse_value * 1.05;
        end
        
        % y position: calculate based on window function index, distributing labels vertically
        % Each SNR point has 5 window functions, distribute them evenly vertically
        base_y = snrIdx;  % Base y position for current SNR
        y_spacing = 0.15; % Vertical spacing
        y_pos = base_y - 0.3 + (wIdx-1) * y_spacing;  % Even distribution
        
        % Add text label with white background for better readability
        text(x_pos, y_pos, text_str, ...
            'FontSize', 8, 'Color', 'k', 'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'middle', 'BackgroundColor', 'white', ...
            'EdgeColor', 'k', 'Margin', 1, 'FontWeight', 'normal');
    end
end

%% 11. Results Summary
disp('=== Simulation Complete ===');
fprintf('\nPerformance Summary (SNR = %d dB, Actual SNR = %.1f dB):\n', params.SNR_dB, actual_SNR);
fprintf('%-15s %-10s %-10s %-15s\n', 'Window', 'BER', 'MSE', 'Relative Improvement(dB)');
fprintf('%-15s %-10s %-10s %-15s\n', '--------', '---', '---', '------------');

% Calculate improvement of each window function relative to no equalization
for wIdx = 1:num_windows
    ber_improvement = 10*log10(BER_no_eq/BER_single(wIdx));
    fprintf('%-15s %-10.4f %-10.4f %-10.1f dB\n', window_types{wIdx}, BER_single(wIdx), MSE_single(wIdx), ber_improvement);
end

fprintf('\nUnequalized Signal BER: %.4f\n', BER_no_eq);
fprintf('\nBest Window Function (based on single simulation):\n');
[~, minBER_idx] = min(BER_single);
[~, minMSE_idx] = min(MSE_single);
fprintf('Lowest BER: %s Window (BER = %.4f, Improvement %.1f dB)\n', ...
    window_types{minBER_idx}, BER_single(minBER_idx), ...
    10*log10(BER_no_eq/BER_single(minBER_idx)));
fprintf('Lowest MSE: %s Window (MSE = %.4f)\n', window_types{minMSE_idx}, MSE_single(minMSE_idx));

% Display Monte Carlo simulation results table
fprintf('\nMonte Carlo Simulation Results (Average BER @ SNR=10dB):\n');
snr10_idx = find(params.SNR_range == 10, 1);
if ~isempty(snr10_idx)
    fprintf('%-15s %-10s %-10s %-15s\n', 'Window', 'Average BER', 'Average MSE', 'Relative Improvement(dB)');
    fprintf('%-15s %-10s %-10s %-15s\n', '--------', '-------', '-------', '------------');
    for wIdx = 1:num_windows
        ber_avg = BER_all(snr10_idx, wIdx);
        mse_avg = MSE_all(snr10_idx, wIdx);
        ber_improvement_avg = 10*log10(BER_all(snr10_idx, 1)/ber_avg);
        fprintf('%-15s %-10.4f %-10.4f %-10.1f dB\n', ...
            window_types{wIdx}, ber_avg, mse_avg, ber_improvement_avg);
    end
end

fprintf('\nWindow Function Characteristics Analysis:\n');
fprintf('1. Rectangular: Narrowest main lobe, highest sidelobes - Suitable for smoothly varying channels\n');
fprintf('2. Hamming: Lower sidelobes, wider main lobe - Balanced main lobe width and sidelobe suppression\n');
fprintf('3. Kaiser: Can adjust main lobe width and sidelobe suppression via beta parameter\n');
fprintf('4. Blackman: Best sidelobe suppression, but widest main lobe\n');
fprintf('5. Chebyshev: Equiripple sidelobes, sidelobe attenuation can be precisely controlled\n');

fprintf('\nImprovement Measures Summary:\n');
fprintf('1. Fixed bit mapping and demapping issues\n');
fprintf('2. Improved channel estimation algorithm with smoothing\n');
fprintf('3. Used all OFDM symbols for equalization and demodulation\n');
fprintf('4. Improved equalizer to avoid division by zero errors\n');
fprintf('5. Amplitude normalization to ensure consistent symbol power\n');

% Save results
save('window_function_results_optimized.mat', 'params', 'window_types', ...
    'BER_all', 'MSE_all', 'BER_single', 'MSE_single', 'actual_SNR', 'BER_no_eq');
disp('Results saved to window_function_results_optimized.mat');

% Display final BER data comparison
fprintf('\nFinal BER Data Comparison:\n');
fprintf('Unequalized: %.4f\n', BER_no_eq);
for wIdx = 1:num_windows
    fprintf('%s Window: %.4f (Improvement: %.1f dB)\n', window_types{wIdx}, BER_single(wIdx), ...
        10*log10(BER_no_eq/BER_single(wIdx)));
end

%% ==================== Custom Function Definitions ====================

% QPSK modulation function
function modulated = qpsk_modulate(bits)
    % QPSK modulation using Gray coding
    if mod(length(bits), 2) ~= 0
        bits = [bits; 0];
    end
    
    bits_reshaped = reshape(bits, 2, [])';
    modulated = zeros(size(bits_reshaped, 1), 1);
    
    for i = 1:size(bits_reshaped, 1)
        if bits_reshaped(i, 1) == 0 && bits_reshaped(i, 2) == 0
            modulated(i) = (1 + 1i)/sqrt(2);
        elseif bits_reshaped(i, 1) == 0 && bits_reshaped(i, 2) == 1
            modulated(i) = (1 - 1i)/sqrt(2);
        elseif bits_reshaped(i, 1) == 1 && bits_reshaped(i, 2) == 0
            modulated(i) = (-1 + 1i)/sqrt(2);
        else
            modulated(i) = (-1 - 1i)/sqrt(2);
        end
    end
end

% QPSK demodulation function
function bits = qpsk_demodulate(symbols)
    bits = zeros(length(symbols)*2, 1);
    
    for i = 1:length(symbols)
        symbol = symbols(i);
        
        % Use decision regions for demodulation
        real_part = real(symbol);
        imag_part = imag(symbol);
        
        if real_part >= 0
            bits(2*i-1) = 0;
        else
            bits(2*i-1) = 1;
        end
        
        if imag_part >= 0
            bits(2*i) = 0;
        else
            bits(2*i) = 1;
        end
    end
end

% Time domain noise reduction function
function h_time_denoised = time_domain_noise_reduction(h_time, threshold)
    % Time domain noise reduction
    power_profile = abs(h_time).^2;
    max_power = max(power_profile);
    
    % Set threshold
    threshold_level = threshold * max_power;
    
    % Apply threshold
    h_time_denoised = h_time;
    h_time_denoised(power_profile < threshold_level) = 0;
end

% Improved channel estimation algorithm
function H_est = improved_channel_estimation(rxPilots, txPilots, pilotIndices, numSubcarriers)
    % LS estimation
    H_pilots = rxPilots ./ txPilots;
    
    % Use spline interpolation
    H_est = interp1(pilotIndices, H_pilots, 1:numSubcarriers, 'spline');
    
    % Smoothing
    H_est = smooth_channel_estimate(H_est);
end

% Channel estimation smoothing function
function H_smooth = smooth_channel_estimate(H_est)
    % Use moving average to smooth channel estimation
    window_size = 5;
    H_smooth = H_est;
    
    for i = 1:length(H_est)
        start_idx = max(1, i-floor(window_size/2));
        end_idx = min(length(H_est), i+floor(window_size/2));
        H_smooth(i) = mean(H_est(start_idx:end_idx));
    end
end