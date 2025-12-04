clear; clc; close all
addpath("DSA\")   % make sure SFT files are here

N = 128; 
k = 1; 
fprintf('N=%d, k=%d\n', N, k);

L = 200;
% -------------------------------------------------------------
% Generate exactly k-sparse frequency vector and time-domain signal
% -------------------------------------------------------------
[X_true, freqs_true, coeffs] = generate_sparse_freq_domain(N, k, L);
x = ifft(X_true);
snr_dB = 20;  % Desired signal-to-noise ratio in dB
x_n = add_awgn(x, snr_dB);

% fprintf('True freq bins (0-based):\n'); disp(freqs_true.');

% -------------------------------------------------------------
% FFT Ground Truth
% -------------------------------------------------------------
time_start = tic();
Xfft = fft(x);
time_elapsed = toc(time_start);
fprintf("\n-------------- FFT ----------------------\n")
fprintf('FFT computation time: %.4f seconds\n', time_elapsed);
[~, idx_fft] = maxk(abs(Xfft), k);
idx_fft = sort(idx_fft - 1);  % convert to 0-based

fprintf('FFT detected bins:\n');
disp(idx_fft.');

% -------------------------------------------------------------
% Run YOUR SFT implementation
% -------------------------------------------------------------
time_start = tic();
[idx_sft, val] = phase_encoding(x);
time_elapsed = toc(time_start);

correct = intersect(idx_fft, idx_sft);
missing = setdiff(idx_fft, idx_sft);
extra   = setdiff(idx_sft, idx_fft);

fprintf("\n--------------- Phase Encoding ---------------------\n")

fprintf('Phase encoding: %.4f seconds\n\n', time_elapsed);

fprintf('Phase encoding detected bins:\n');
disp(idx_sft.');

fprintf('Correct: %d / %d\n\n', numel(correct), k);
fprintf('Missing: %d / %d\n', numel(missing), k);

fprintf('Extra: %d / %d\n', numel(extra), k);

%%%%%%%%%%%%%%%%% add the binary search

time_start = tic();
[idx_sft, val] = crt_sft_topk(x,k,1e-4);
time_elapsed = toc(time_start);

correct = intersect(idx_fft, idx_sft);
missing = setdiff(idx_fft, idx_sft);
extra   = setdiff(idx_sft, idx_fft);

fprintf("\n---------------- CRTSFT --------------------\n")

fprintf('SparseFFT with CRT: %.4f seconds\n\n', time_elapsed);

fprintf('CRT detected bins:\n');
disp(idx_sft.');

fprintf('Correct: %d / %d\n\n', numel(correct), k);
fprintf('Missing: %d / %d\n', numel(missing), k);

fprintf('# SFT detected bins: %d\n', length(idx_sft));
fprintf('Extra: %d / %d\n', numel(extra), k);


time_start = tic();
zhat = NoiselessSparseFFT(x,k);
time_elapsed = toc(time_start);

[~, idx_sft] = maxk(abs(zhat), k);
idx_sft = sort(idx_sft - 1);  % convert to 0-based

correct = intersect(idx_fft, idx_sft);
missing = setdiff(idx_fft, idx_sft);
extra   = setdiff(idx_sft, idx_fft);

fprintf("\n---------------- NoiselessSFFT --------------------\n")

fprintf('SparseFFT: %.4f seconds\n\n', time_elapsed);

fprintf('Noiseless detected bins:\n');
disp(idx_sft.');

fprintf('Correct: %d / %d\n\n', numel(correct), k);
fprintf('Missing: %d / %d\n', numel(missing), k);

fprintf('# SFT detected bins: %d\n', length(idx_sft));
fprintf('Extra: %d / %d\n', numel(extra), k);


% -------------------------------------------------------------
% Helper: Generate sparse freq-domain vector
% -------------------------------------------------------------

function [X, freqs, coeffs] = generate_sparse_freq_domain(N, k, L)
    freqs = randperm(N, k).';
    coeffs = (randn(k,1) + 1j*randn(k,1)) * L;   % complex coeffs % coeffs = randi([-L,L],1,k);
    X = zeros(N,1);
    X(freqs) = coeffs;
    X = X/max(X);
    freqs = freqs - 1;
end

function x_noisy = add_awgn(x, snr_dB)
% ADD_AWGN - Add white Gaussian noise to a signal to achieve a desired SNR
%
%   x_noisy = add_awgn(x, snr_dB)
%
%   x       - input signal (can be complex)
%   snr_dB  - desired signal-to-noise ratio in dB
%
%   x_noisy - output signal with added noise


x = x(:);
N = length(x);

% Compute signal power
P_signal = mean(abs(x).^2);

% Compute noise power for desired SNR
P_noise = P_signal / 10^(snr_dB/10);

% Generate complex Gaussian noise
noise = sqrt(P_noise/2)*(randn(N,1) + 1j*randn(N,1));

% Add noise to signal
x_noisy = x + noise;


end
