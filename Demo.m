%% Sparse FFT experiment: recovery probability vs sparsity k and signal length n
% Save this file as: sfft_n_k_experiment.m
clc; clear; close all;

% ------------------------------
% Experiment parameters
% ------------------------------
N_list   = [2^14, 2^15, 2^16, 2^17, 2^18, 2^19, 2^20, 2^21, 2^22];   % different signal lengths to test
K_list   = [2 3 4 5 6 7 8];           % sparsity levels to test (same for all n)
numTrials = 50;                     % Monte Carlo trials per (n,k) pair

% Threshold to decide "nonzero" in recovered spectrum
recovery_thresh = 1e-6;

numN = numel(N_list);
numK = numel(K_list);

% Storage:
%   P_exact(iN, iK): probability of exact recovery for N_list(iN), K_list(iK)
%   avg_frac_hits(iN, iK): average fraction of correct hits (recall)
P_exact         = zeros(numN, numK);
avg_frac_hits   = zeros(numN, numK);

% New detailed stats:
avg_num_predicted = zeros(numN, numK);   % E[|rec_support|]
avg_num_correct   = zeros(numN, numK);   % E[|correct_hits|]
avg_num_missed    = zeros(numN, numK);   % E[|missed|]
avg_num_false     = zeros(numN, numK);   % E[|false_alarms|]
avg_precision     = zeros(numN, numK);   % E[|correct_hits| / |rec_support|]

rng(0);  % for reproducibility

fprintf('Running sparse FFT recovery experiment over n and k...\n');
fprintf('Trials per (n,k): %d\n\n', numTrials);

for iN = 1:numN
    n = N_list(iN);
    fprintf('===== n = %d =====\n', n);

    for iK = 1:numK
        k = K_list(iK);

        % safety: k should be <= n (here it always is, but keep this for generality)
        if k > n
            warning('Skipping k = %d for n = %d because k > n.', k, n);
            continue;
        end

        exact_success_trials = zeros(numTrials,1);
        fraction_hits_trials = zeros(numTrials,1);

        % new per-trial stats
        num_predicted_trials = zeros(numTrials,1);
        num_correct_trials   = zeros(numTrials,1);
        num_missed_trials    = zeros(numTrials,1);
        num_false_trials     = zeros(numTrials,1);
        precision_trials     = zeros(numTrials,1);

        fprintf('  --- k = %d ---\n', k);

        for trial = 1:numTrials

            % ------------------------------
            % 1) Generate random k-sparse spectrum xhat_true (length n)
            % ------------------------------
            xhat_true = zeros(n,1);
            supp = randperm(n, k).';   % true support (indices 1..n)

            amps_real = randi([-5,5], k, 1);
            amps_imag = randi([-5,5], k, 1);
            amps = amps_real + 1i*amps_imag;

            xhat_true(supp) = amps;

            % Time-domain signal
            x = ifft(xhat_true);

            % ------------------------------
            % 2) Run sparse FFT
            % ------------------------------
            zhat = NoiselessSparseFFT(x, k);

            % ------------------------------
            % 3) Measure support recovery
            % ------------------------------
            true_support = find(abs(xhat_true) > 0);
            rec_support  = find(abs(zhat) > recovery_thresh);

            true_support_sorted = sort(true_support);
            rec_support_sorted  = sort(rec_support);

            correct_hits = intersect(true_support_sorted, rec_support_sorted);
            missed       = setdiff(true_support_sorted, rec_support_sorted);
            false_alarms = setdiff(rec_support_sorted, true_support_sorted);

            % sizes of sets
            c = numel(correct_hits);
            m = numel(rec_support);
            miss_cnt  = numel(missed);
            false_cnt = numel(false_alarms);

            % record per-trial stats
            num_correct_trials(trial)   = c;
            num_predicted_trials(trial) = m;
            num_missed_trials(trial)    = miss_cnt;
            num_false_trials(trial)     = false_cnt;

            if m == 0
                precision_trials(trial) = 0;
            else
                precision_trials(trial) = c / m;   % precision = TP / predicted
            end

            fraction_hits_trials(trial) = c / k;   % recall-like metric

            % exact success: all true found, no extras, and recovered size = k
            if isempty(missed) && isempty(false_alarms) && m == k
                exact_success_trials(trial) = 1;
            else
                exact_success_trials(trial) = 0;
            end
        end

        % aggregate stats over trials for this (n,k)
        P_exact(iN, iK)         = mean(exact_success_trials);
        avg_frac_hits(iN, iK)   = mean(fraction_hits_trials);

        avg_num_predicted(iN,iK) = mean(num_predicted_trials);
        avg_num_correct(iN,iK)   = mean(num_correct_trials);
        avg_num_missed(iN,iK)    = mean(num_missed_trials);
        avg_num_false(iN,iK)     = mean(num_false_trials);
        avg_precision(iN,iK)     = mean(precision_trials);

        fprintf('      Exact recovery probability: %.2f\n', P_exact(iN, iK));
        fprintf('      Avg fraction of correct hits (recall): %.2f\n', avg_frac_hits(iN, iK));
        fprintf('      Avg # predicted bins m: %.2f\n', avg_num_predicted(iN,iK));
        fprintf('      Avg # correct c: %.2f, missed: %.2f, false alarms: %.2f\n', ...
                avg_num_correct(iN,iK), avg_num_missed(iN,iK), avg_num_false(iN,iK));
        fprintf('      Avg precision (c/m): %.2f\n\n', avg_precision(iN,iK));
    end

    fprintf('\n');
end

% ------------------------------
% Print summary tables
% ------------------------------
fprintf('Summary: Probability of exact support recovery P_exact(n,k)\n');
for iN = 1:numN
    fprintf('n = %d:\n', N_list(iN));
    fprintf('   k    P_exact  avg_frac_hits  avg_pred  avg_corr  avg_miss  avg_false  avg_prec\n');
    for iK = 1:numK
        fprintf('%4d   %6.3f     %6.3f      %6.2f    %6.2f    %6.2f    %6.2f    %6.3f\n', ...
            K_list(iK), ...
            P_exact(iN,iK), ...
            avg_frac_hits(iN,iK), ...
            avg_num_predicted(iN,iK), ...
            avg_num_correct(iN,iK), ...
            avg_num_missed(iN,iK), ...
            avg_num_false(iN,iK), ...
            avg_precision(iN,iK));
    end
    fprintf('\n');
end

% ------------------------------
% Plots: curves vs k for each n (P_exact & recall)
% ------------------------------
figure;
hold on;
for iN = 1:numN
    plot(K_list, P_exact(iN,:), '-o', 'LineWidth', 1.5);
end
grid on;
xlabel('Sparsity k');
ylabel('Probability of exact support recovery');
title('Sparse FFT: P_{exact} vs k for different n');
legend(arrayfun(@(N) sprintf('n = %d', N), N_list, 'UniformOutput', false), ...
       'Location','best');

figure;
hold on;
for iN = 1:numN
    plot(K_list, avg_frac_hits(iN,:), '-o', 'LineWidth', 1.5);
end
grid on;
xlabel('Sparsity k');
ylabel('Average fraction of correct hits (recall)');
title('Sparse FFT: average fraction of correct support vs k for different n');
legend(arrayfun(@(N) sprintf('n = %d', N), N_list, 'UniformOutput', false), ...
       'Location','best');

% ------------------------------
% Plots: heatmap view over (n,k)
% ------------------------------
figure;
imagesc(K_list, N_list, P_exact);  % rows: n, cols: k
set(gca,'YDir','normal');
colorbar;
xlabel('Sparsity k');
ylabel('Signal length n');
title('Heatmap: Probability of exact recovery P_{exact}(n,k)');

figure;
imagesc(K_list, N_list, avg_frac_hits);
set(gca,'YDir','normal');
colorbar;
xlabel('Sparsity k');
ylabel('Signal length n');
title('Heatmap: Average fraction of correct hits (n,k)');

% optionally: heatmap for avg precision
figure;
imagesc(K_list, N_list, avg_precision);
set(gca,'YDir','normal');
colorbar;
xlabel('Sparsity k');
ylabel('Signal length n');
title('Heatmap: Average precision (correct / predicted)');


%% ================== IMPLEMENTATION BELOW ==================

function zhat = NoiselessSparseFFT(x,k)
% NOISELESSSPARSEFFT  Top-level sparse FFT driver.

n = length(x);
zhat = zeros(n,1);

for t = 0:floor(log2(k))
    kt = k / (2^t);
    alpha = 0.25;
    zhat = zhat + NoiselessSparseFFT_Inner(x,kt,zhat,alpha);
end

end


function w_hat = NoiselessSparseFFT_Inner(x,k,zhat,alpha)
% Inner routine: hashing + filtering + location/estimation

n = length(x);

% Number of hash bins B, capped so n/B >= 1
B = 2^nextpow2(512*k);
B = min(B, n);

% Width parameter for the Gaussian filter (acts like band width)
width = (1-alpha)*n/2/B;

filter = make_gaussian_filter(n,width);

% Random odd sigma and random shift b
odds  = 1:2:(n-1);
sigma = odds(randi(numel(odds)));
b     = randi([0, n-1]);

% Hash into bins
u_hat  = hash2bins(x,zhat,sigma,0,b,B,filter);
up_hat = hash2bins(x,zhat,sigma,1,b,B,filter);

% Find "significant" bins
J = find(abs(u_hat) >= 1-2e-12);
w_hat = zeros(n,1);

for kk = 1:length(J)
    jj = J(kk);

    ratio = u_hat(jj)/(up_hat(jj)+1e-12);
    phi_a = atan2(imag(ratio),real(ratio));

    sigma_inv = modinv(sigma, n);

    % integer frequency index
    kfreq = round(phi_a * n / (2*pi));
    idx = mod(sigma_inv * kfreq, n)+1;   % MATLAB 1-based index

    v = u_hat(jj);                       % estimated coefficient
    w_hat(idx) = v;
end

end


function uhat = hash2bins(x, zhat, sigma, a, b, B, filter)
% HASH2BINS  Permute, filter and downsample into B bins.

n = length(x);

% avoid zero step size
downsampling = max(1, round(n/B));

% 1. permute + filter
y = filter.g .* perm(x, sigma, a, b);

% 2. downsample
idx  = 1:downsampling:n;
yhat = n*downsampling*fft(y(idx));

% 3. alias cancellation
if ~isempty(zhat) && any(zhat ~= 0)

    % A. permute zhat in frequency domain, then downsample
    zhat_perm_ds = perm_freq_downsample(zhat, sigma, a, b, n, B);

    % B. circular convolution in frequency domain with explicit length
    M = length(yhat);   % number of bins (length we want)
    V = fft(filter.g_hat_prime, M) .* fft(zhat_perm_ds, M);
    v = ifft(V);

    % C. subtract
    yhat = yhat - v;
end

uhat = yhat;
end


function y = perm(x, sigma, a, b)
% PERM  Time-domain permutation with modulation.

n = length(x);
t = (0:n-1).';

idx0  = mod(sigma * (t - a), n);
idx   = idx0 + 1;   % 1-based
phase = exp(-2i*pi * (sigma * b .* t) / n);

y = x(idx) .* phase;
end


function zhat_perm_ds = perm_freq_downsample(zhat, sigma,a, b, n, B)
% PERM_FREQ_DOWNSAMPLE  Permute in frequency domain + downsample.

sigma_inv = modinv(sigma, n);

step = max(1, round(n/B));
k = (1:step:n).'-1;   % 0-based frequency indices

idx = mod(sigma_inv * k + b, n) + 1;
zhat_perm = step*zhat(idx);

phase = exp(-2i*pi * a * k / n);
zhat_perm_ds = zhat_perm .* phase;

end


function filt = make_gaussian_filter(N,Bwidth,sigma,center_bin)
% MAKE_GAUSSIAN_FILTER  Gaussian-tapered rectangular filter in freq domain.

if nargin < 3 || isempty(sigma), sigma = Bwidth/6; end
if nargin < 4 || isempty(center_bin), center_bin = 0; end

% ensure integer band width at least 1 and at most N
W = max(1, round(Bwidth));
W = min(W, N);

% 1. rectangular core + Gaussian taper
H = zeros(N,1);
start_bin = mod(center_bin - floor(W/2), N);
bins0 = mod(start_bin + (0:W-1), N);
H(bins0+1) = 1;

k = (0:N-1).';
d = mod(k - center_bin + N/2, N) - N/2;
gauss = exp(-0.5*(d./sigma).^2);
H = H .* gauss;

% 2. time-domain filter
g = ifft(H);

% 3. downsampled freq response
ds = max(1, round(N/W));
g_hat_prime = ds*H(1:ds:end);

filt.g = g;
filt.g_hat = H;
filt.g_hat_prime = g_hat_prime;
end


function inv = modinv(a, m)
% MODINV  modular inverse of a modulo m (returns value in 0..m-1)
% requires gcd(a,m) == 1
a = mod(a, m);
if a == 0
    error('modinv: a == 0 (no inverse)');
end
% extended euclid
old_r = a; r = m;
old_s = 1; s = 0;
while r ~= 0
    q = floor(old_r / r);
    [old_r, r] = deal(r, old_r - q * r);
    [old_s, s] = deal(s, old_s - q * s);
end
if old_r ~= 1
    error('modinv: no inverse exists (gcd ~= 1)');
end
inv = mod(old_s, m);
end
