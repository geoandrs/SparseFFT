% main_comparison.m
clc; clear; close all;

fprintf('=== COMPARISON OF SFT METHODS (Sparsity k=1) ===\n\n');

%% TEST 1: Phase Encoding & Binary Search (Powers of 2)
N = 2^24;
true_omega = 500; % Target Frequency
true_amp = 1.5 + 1i*0.5;
noise_level = 0.0; % Keep low for exact verification

% Generate Signal: f_j = A * exp(2*pi*i*omega*j/N)
t = 0:N-1;
signal = true_amp * exp(2*pi*1i*true_omega*t/N) + noise_level*(randn(1,N) + 1i*randn(1,N));

fprintf('--- Test Case 1: N=%d, True Omega=%d ---\n', N, true_omega);

% 1. Built-in FFT
tic;
fft_out = fft(signal);
[~, fft_idx] = max(abs(fft_out));
fft_omega = fft_idx - 1;
time_fft = toc;
fprintf('Built-in FFT:     Found w=%d \t(Time: %.6f s)\n', fft_omega, time_fft);

% 2. Phase Encoding
tic;
[pe_omega, pe_amp] = sft_phase_encoding(signal, N);
time_pe = toc;
fprintf('Phase Encoding:   Found w=%d \t(Time: %.6f s)\n', pe_omega, time_pe);

% 3. Binary Search (Corrected)
tic;
[bs_omega, bs_amp] = sft_binary_search(signal, N);
time_bs = toc;
bs_omega_rounded = round(bs_omega); 
fprintf('Binary Search:    Found w=%d \t(Time: %.6f s)\n', bs_omega_rounded, time_bs);

%% TEST 2: Aliased-Based Search (Coprime Product)
factors = [2, 3, 5, 7, 11];
N_alias = prod(factors);
true_omega_alias = 1984; 
t_alias = 0:N_alias-1;
signal_alias = true_amp * exp(2*pi*1i*true_omega_alias*t_alias/N_alias);

fprintf('\n--- Test Case 2 (Coprime N): N=%d, True Omega=%d ---\n', N_alias, true_omega_alias);

% Built-in FFT for comparison
tic;
fft_out_a = fft(signal_alias);
[~, idx_a] = max(abs(fft_out_a));
time_fft_a = toc;
fprintf('Built-in FFT:     Found w=%d \t(Time: %.6f s)\n', idx_a-1, time_fft_a);

% 4. Aliased Search
tic;
[as_omega, as_amp] = sft_aliased_search(signal_alias, N_alias, factors);
time_as = toc;
fprintf('Aliased Search:   Found w=%d \t(Time: %.6f s)\n', as_omega, time_as);

%% Method Implementations

function [omega, amplitude] = sft_phase_encoding(signal, N)
    val1 = signal(1); % f_0
    val2 = signal(2); % f_1
    
    if abs(val1) < 1e-9
        omega = 0; amplitude = 0; return;
    end
    
    ratio = val2 / val1;
    angle_rad = angle(ratio);
    if angle_rad < 0, angle_rad = angle_rad + 2*pi; end
    
    omega = round((angle_rad * N) / (2*pi));
    amplitude = val1;
end

function [omega, amplitude] = sft_binary_search(signal, N)
    % sft_binary_search
    % Recover integer frequency omega for a 1-sparse complex exponential:
    %   f(j) = A * exp(2*pi*i*omega*j/N)
    %
    % Uses O(log N) samples and decodes the bits of omega from phase
    % measurements at carefully chosen strides.
    %
    % Assumes: 
    %   - N is a power of 2
    %   - signal is noiseless or low-noise
    %   - spectrum is 1-sparse

    % ------- Safety checks -------
    m = round(log2(N));
    if 2^m ~= N
        error('sft_binary_search: N must be a power of 2.');
    end
    
    f0 = signal(1);   % f(0) = A * exp(0) = A
    if abs(f0) < 1e-12
        % If the DC sample is essentially zero, we can’t recover A or omega
        omega     = 0;
        amplitude = 0;
        return;
    end

    % ------- Bit-by-bit reconstruction of omega -------
    %
    % Let omega be written in binary:
    %   omega = b_{m-1} 2^{m-1} + ... + b_1 2^1 + b_0 2^0.
    %
    % For k = 0..m-1 we choose stride:
    %   s_k = N / 2^{k+1}.
    %
    % Then:
    %   r_k = f(s_k) / f(0) = exp(2*pi*i*omega*s_k/N)
    %       = exp(2*pi*i*omega / 2^{k+1}).
    %
    % The phase of r_k only depends on the LOWER (k+1) bits of omega.
    % If we already know bits b_0..b_{k-1} (encoded in "lower_bits"),
    % we can remove their effect and what remains is just b_k * pi:
    %
    %   r_k_corrected = r_k / exp(2*pi*i*lower_bits / 2^{k+1})
    %   => phase(r_k_corrected) ≈ 0      if b_k = 0
    %                         ≈ pi (or -pi) if b_k = 1
    %
    % So sign(real(r_k_corrected)) tells us b_k.

    lower_bits = 0;   % This will store sum_{j=0}^{k-1} b_j 2^j
    
    for k = 0:(m-1)
        % Stride for this bit
        stride = N / 2^(k+1);
        idx    = stride + 1;   % MATLAB is 1-based
        
        if idx > N
            error('sft_binary_search: stride index exceeds signal length.');
        end
        
        % Ratio r_k = f(stride) / f(0)
        val    = signal(idx);
        r_meas = val / f0;
        
        % Remove phase contribution of already-found lower bits
        expected_lower = exp(2*pi*1i * lower_bits / 2^(k+1));
        r_corr         = r_meas / expected_lower;
        
        % Decide bit b_k:
        %   if real(r_corr) >= 0  -> phase ~ 0    -> b_k = 0
        %   if real(r_corr) <  0  -> phase ~ pi   -> b_k = 1
        if real(r_corr) >= 0
            bit_k = 0;
        else
            bit_k = 1;
        end
        
        lower_bits = lower_bits + bit_k * 2^k;
    end

    % Wrap into [0, N-1]
    omega     = mod(lower_bits, N);
    amplitude = f0;   % A = f(0)
end

function [omega, amplitude] = sft_aliased_search(signal, N, factors)
    remainders = [];
    moduli = [];
    
    for k = 1:length(factors)
        P = factors(k);
        stride = N / P;
        indices = 1:stride:N;
        indices = indices(1:P);
        sub_signal = signal(indices);
        
        small_fft = fft(sub_signal);
        [~, idx] = max(abs(small_fft));
        rem = idx - 1; 
        
        remainders = [remainders; rem];
        moduli = [moduli; P];
    end
    omega = solve_crt(remainders, moduli);
    amplitude = signal(1);
end

function x = solve_crt(remainders, moduli)
    M = prod(moduli);
    x = 0;
    for i = 1:length(moduli)
        mi = moduli(i);
        ai = remainders(i);
        Mi = M / mi;
        [~, inv, ~] = gcd(Mi, mi);
        x = x + ai * Mi * inv;
    end
    x = mod(x, M);
end
