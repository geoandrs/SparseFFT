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
    % Corrected Binary Search / Successive Approximation
    % Resolves frequency by peeling off bits using increasing strides.
    
    omega_accum = 0;
    current_stride = 1;
    
    % Number of stages: log4(N) is ideal as we resolve 2 bits (quadrant) per step
    % We run until stride reaches N/4 to cover all bits
    max_stages = ceil(log(N)/log(2)); 
    
    for k = 1:max_stages
        % 1. Predict expected phase contribution from already found freq
        %    at the current stride.
        expected_phase_shift = exp(2*pi*1i * omega_accum * current_stride / N);
        
        % 2. Measure actual sample at current stride
        %    (Index is stride+1 because Matlab is 1-based)
        if current_stride >= N
            break; 
        end
        val = signal(current_stride + 1);
        
        % 3. Compute Residual (remove known frequency part)
        %    f_measured = A * exp(w * stride)
        %    f_resid    = f_measured / exp(w_accum * stride)
        %               = A * exp((w - w_accum) * stride)
        resid_val = val / expected_phase_shift;
        
        % 4. Determine Quadrant of the Residual
        %    This tells us the error term.
        %    Q0 (0):   0..0.25   -> Offset 0
        %    Q1 (i):   0.25..0.5 -> Offset 1
        %    Q2 (-1):  0.5..0.75 -> Offset 2
        %    Q3 (-i):  0.75..1.0 -> Offset 3
        
        % Check alignment with axes
        lhs_i = abs(1i * signal(1) - resid_val); % Compare with i*Amp
        rhs_i = abs(1i * signal(1) + resid_val); % Compare with -i*Amp
        closer_to_i_axis = lhs_i < rhs_i; % True if closer to i than -i
        
        lhs_1 = abs(signal(1) - resid_val);      % Compare with 1*Amp
        rhs_1 = abs(signal(1) + resid_val);      % Compare with -1*Amp
        closer_to_1_axis = lhs_1 < rhs_1; % True if closer to 1 than -1
        
        region_offset = 0;
        
        if closer_to_1_axis
            if closer_to_i_axis
                region_offset = 0; % Q1 (Top-Right) -> 0
            else
                region_offset = 3; % Q4 (Bottom-Right) -> 3
            end
        else
            if closer_to_i_axis
                region_offset = 1; % Q2 (Top-Left) -> 1
            else
                region_offset = 2; % Q3 (Bottom-Left) -> 2
            end
        end
        
        % 5. Update Omega
        %    The resolution of a quadrant at 'stride' is N / (4 * stride)
        correction = region_offset * (N / (4 * current_stride));
        omega_accum = omega_accum + correction;
        
        % 6. Increase Stride (Decimate)
        %    We multiply by 2? Or 4?
        %    Quadrants give 2 bits. Doubling stride shifts phase by 2x.
        %    If we resolved 2 bits, we can multiply stride by 4 ideally.
        %    But to be safe against boundary errors, stride * 2 is often used 
        %    with overlapping bins. For this specific 'k=1' simplified logic,
        %    doubling works well to refine precision.
        current_stride = current_stride * 2;
    end
    
    omega = omega_accum;
    amplitude = signal(1);
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