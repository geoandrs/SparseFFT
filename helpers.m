function h = helpers()
    h.generate_sparse_freq_domain = @generate_sparse_freq_domain;
    h.generate_sparse_freq = @generate_sparse_freq;
    h.add_awgn = @add_awgn;
    h.design_filter = @design_filter;
    h.make_multiple_filters = @make_multiple_filters;
    h.make_sparse_representation = @make_sparse_representation;
    h.gaussian_sinc_sparse = @gaussian_sinc_sparse;
    h.dpss_sparse = @dpss_sparse;
    h.sparse_cconv = @sparse_cconv;
    h.make_sparse_filter = @make_sparse_filter;
end

function [X, freqs, coeffs] = generate_sparse_freq(N, k, L)
    freqs = randperm(N, k).';
    coeffs = (ones(k,1) + 1j*ones(k,1)) * L;   % complex coeffs % coeffs = randi([-L,L],1,k);
    X = zeros(N,1);
    X(freqs) = coeffs;
    X = X/max(X);
    freqs = freqs - 1;
end

function [X, freqs, coeffs] = generate_sparse_freq_domain(N, k, L)
    % Uniformly spaced frequencies across the entire bandwidth
    spacing = floor(N / k);
    freqs = (1:k).' * spacing;      % spaced inside [spacing, ..., k*spacing]
    freqs = freqs - 1;              % shift to zero-based indexing

    % Random complex coefficients
    coeffs = L*ones(k,1);

    % Construct sparse spectrum
    X = zeros(N,1);
    X(freqs + 1) = coeffs;          % convert back to MATLAB 1-based indexing

    % Normalize
    X = X / max(abs(X));
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

function [t_filter, f_filter, fp_filter] = design_filter(N,samples,B)
    f_filter = zeros(N,1);
    f_filter(1:samples) = 1; % box window in the frequency domain
    t_filter = ifft(f_filter); % box window in time domain -> sinc

    % polyphase matrix: B filters, each length = N/B
    fp_filter = zeros(N/B, B);
    
    for b = 0:B-1
        fp_filter(:, b+1) = t_filter(1+b : B : end);
    end
end

function [t_filters, nz_idx_list, nz_vals_list] = make_multiple_filters(N,t_filter,num,samples)
    nz_idx_list = cell(num,1);
    nz_vals_list = cell(num,1);
        
    t_filters = t_filter;
    t_filter_moved = t_filter;
    [nz_idx_list{1}, nz_vals_list{1}] = make_sparse_representation(t_filter);
    for i = 2:num
        t_filter_moved = t_filter_moved.*exp(2j*pi*samples*(0:N-1).'/N);
        t_filters = [t_filters, t_filter_moved]; % accumulate the filters
        [nz_idx_list{i}, nz_vals_list{i}] = make_sparse_representation(t_filter_moved);
    end
end

function [nz_idx, nz_vals] = make_sparse_representation(h, threshold)
    if nargin < 2
        threshold = 1e-6;
    end
    nz = find(abs(h) > threshold);
    nz_idx = nz;               % 1-based indices (MATLAB)
    nz_vals = h(nz);
end

%% 1) Build Gaussian-tapered-sinc filter (frequency core B, length N)
function [h_trunc, H, nz_idx, nz_vals] = gaussian_sinc_sparse(N, B, sigma, center_bin, tol)
    % N         - full DFT length (size of signal)
    % B         - desired core width in frequency (bins)
    % sigma     - Gaussian taper width in bins (suggest B/6 .. B/3)
    % center_bin- 0-based center frequency (default 0)
    % tol       - truncation threshold relative to peak (e.g., 1e-3)
    if nargin < 4 || isempty(center_bin), center_bin = 0; end
    if nargin < 5 || isempty(tol), tol = 1e-3; end

    % make freq mask (centered rectangular core)
    H = zeros(N,1);
    start_bin = mod(center_bin - floor(B/2), N);
    bins0 = mod(start_bin + (0:B-1), N);  % 0-based
    H(bins0+1) = 1;

    % circular distance for gaussian
    k = (0:N-1).';
    d = mod(k - center_bin + N/2, N) - N/2;
    gauss = exp(-0.5*(d./sigma).^2);
    H = H .* gauss;

    % time-domain filter
    h = ifft(H);

    % truncation
    hmax = max(abs(h));
    keep = abs(h) >= tol*hmax;
    nz_idx = find(keep);          % 1-based indices
    nz_vals = h(nz_idx);

    % return truncated time-domain (for plotting)
    h_trunc = zeros(N,1);
    h_trunc(nz_idx) = nz_vals;
end


%% 2) Build DPSS filter (time-limited length L, concentrated over band W)
function [h_trunc, H, nz_idx, nz_vals] = dpss_sparse(N, L, W, center_bin, tol)
    % N        - full DFT length
    % L        - desired time support (number of taps)  (L << N for sparsity)
    % W        - half-bandwidth in normalized frequency (0..0.5) (e.g., W = B/(2*N))
    % center_bin - 0-based center frequency
    % tol      - truncation threshold (relative)
    if nargin < 4 || isempty(center_bin), center_bin = 0; end
    if nargin < 5 || isempty(tol), tol = 1e-3; end

    % use the built-in dpss (Signal Processing Toolbox)
    % The k=1 DPSS (first eigenvector) is the most concentrated.
    % time-window length L, NW = L*W (time-bandwidth product)
    NW = max(0.1, L * W);  % choose small positive if W=0
    [V,~] = dpss(L, NW, 1);   % V is Lx1 vector (first DPSS)
    h_time = V(:,1);          % length L, real valued

    % zero-pad to N and modulate to center_bin
    h_padded = zeros(N,1);
    % place centered around sample 0 .. L-1 or center; choose centered placement:
    start = floor((N - L)/2) + 1;
    h_padded(start:start+L-1) = h_time;

    % modulate to desired center frequency
    n = (0:N-1).';
    h_mod = h_padded .* exp(2j*pi * center_bin .* n ./ N);

    % freq response
    H = fft(h_mod);

    % truncation (you can also simply keep L nonzeros since that's the point)
    hmax = max(abs(h_mod));
    keep = abs(h_mod) >= tol*hmax;
    nz_idx = find(keep);
    nz_vals = h_mod(nz_idx);

    % truncated output for plotting
    h_trunc = zeros(N,1);
    h_trunc(nz_idx) = nz_vals;
end


%% 3) Sparse circular convolution helper
function y = sparse_cconv(x, nz_idx, nz_vals)
   N = length(x);
    y = zeros(N,1);

    for ii = 1:length(nz_idx)
        k = nz_idx(ii);   % 1-based index of the nonzero
        v = nz_vals(ii);

        % Compute target positions WITHOUT circshift
        idx = mod((0:N-1) - (k-1), N) + 1;

        % Accumulate
        y = y + v * x(idx);
    end
end

function h_time = make_sparse_filter(B, N, type)
% MAKE_SPARSE_FILTER  Create and plot short-DFT Gaussian or Sinc×Gaussian filter
%
% h_time = make_sparse_filter(B, N, type)
% B    = number of nonzero samples in DFT domain
% N    = full FFT size
% type = 'gaussian'  or  'sincgauss'
%
% Returns:
%     h_time = sparse time-domain impulse response (length N)

    % --- Frequency-domain support ---
    H = zeros(N,1);
    idx = 1:B;

    switch lower(type)
        case 'gaussian'
            % Gaussian in frequency → Gaussian in time
            sigma_f = B/6;
            H(idx) = exp(-0.5*((idx - B/2)/sigma_f).^2);

        case 'sincgauss'
            % Sinc × Gaussian (from the SFFT papers)
            % Gaussian envelope
            sigma_f = B/6;
            G = exp(-0.5*((idx - B/2)/sigma_f).^2);
            % Sinc kernel (centered)
            n = idx - mean(idx);
            S = sinc(n / (B/2));
            H(idx) = G .* S;

        otherwise
            error('Unknown filter type.');
    end

    % --- Normalize ---
    H = H / max(abs(H));

    % --- Time-domain version ---
    h_time = ifft(H, 'symmetric');

    % Make it sparse (most values are nearly zero)
    % (Useful to print how sparse it really is)
    fprintf('%s filter sparsity: %d/%d > threshold\n', ...
        type, nnz(abs(h_time) > 1e-6), N);

    % -------------------------------
    %         PLOTTING
    % -------------------------------
    figure;

    % Time domain
    subplot(2,1,1);
    stem(0:N-1, h_time, 'filled');
    title(['Time domain - ', type, ' filter']);
    xlabel('n'); ylabel('h[n]');

    % Frequency domain magnitude
    subplot(2,1,2);
    plot(0:N-1, abs(H));
    title(['Frequency domain - ', type, ' filter']);
    xlabel('k'); ylabel('|H[k]|');

end

