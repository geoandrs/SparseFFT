clear; clc; close all
addpath("DSA\")   % make sure SFT files are here

h = helpers();

N = 128; 
k = 2; 
fprintf('N=%d, k=%d\n', N, k);

L = 200;
% -------------------------------------------------------------
% Generate exactly k-sparse frequency vector and time-domain signal
% -------------------------------------------------------------
[X_true, freqs_true, coeffs] = h.generate_sparse_freq(N, k, L);
x = ifft(X_true);
% x = x/max(x);
snr_dB = 20;  % Desired signal-to-noise ratio in dB
x_n = h.add_awgn(x, snr_dB);

figure
stem(abs(X_true))
hold on
stem(abs(fft(x)))

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
zhat = NoiselessSparseFFT(x,k);
time_elapsed = toc(time_start);

fprintf('Noiseless detected bins:\n');
idx_sft = find(zhat>0.1)-1;

% [~, idx_sft] = maxk(abs(zhat), k);
% idx_sft = sort(idx_sft - 1);  % convert to 0-based

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
