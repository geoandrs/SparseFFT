function zhat = NoiselessSparseFFT(x,k)

n = length(x);
zhat = zeros(n,1);

for t = 0:log2(k)
    kt = k/(2^t);
    alpha = 1/(2^(t+1));
    zhat = zhat + NoiselessSparseFFT_Inner(x,kt,zhat,alpha);
end

end

function w_hat = NoiselessSparseFFT_Inner(x,k,zhat,alpha)

n = length(x);
B = 2^nextpow2(16*k);
% width = (1-alpha)*n/2/B
filter = make_gaussian_filter(n,(1-alpha)*n/2/B);
% filter = helpers().design_filter(n,width,0.1);


odds = 1:2:(n-1);
sigma = odds(randi(numel(odds)));
b = randi([0, n-1]);

u_hat = hash2bins(x,zhat,sigma,0,b,B,filter);
up_hat = hash2bins(x,zhat,sigma,1,b,B,filter);

% while ~check_good_permutation(u_hat,up_hat,1e-3)
%     odds = 1:2:(n-1);
%     sigma = odds(randi(numel(odds)));
%     b = randi([0, n-1]);
% 
%     u_hat = hash2bins(x,zhat,sigma,0,b,B,filter);
%     up_hat = hash2bins(x,zhat,sigma,1,b,B,filter);
% end

w_hat = zeros(n,1);
J = find(abs(u_hat) >= 1-2e-12);

for kk = 1:length(J)
    jj = J(kk);
    ratio = u_hat(jj)/(up_hat(jj)+1e-12);
    phi_a = atan2(imag(ratio),real(ratio));
    sigma_inv = modinv(sigma, n);
    k = round(phi_a * n / (2*pi));   % integer frequency index
    idx = mod(sigma_inv * k, n)+1;     % 0-based index
    v = round(u_hat(jj));
    w_hat(idx) = v;
end


end

function uhat = hash2bins(x, zhat, sigma, a, b, B, filter)

n = length(x);
downsampling = round(n/B);

% 1. permute + filter
y = filter.g .* perm(x, sigma, a, b);

% 2. downsample
idx = 1:downsampling:n;
yhat = n*downsampling*fft(y(idx));

% figure
% subplot(511)
% stem(abs(fft(x)))
% subplot(512)
% stem(abs(fft(perm(x, sigma, a, b))))
% subplot(513)
% stem(abs(filter.g_hat))
% subplot(514)
% stem(n*abs(fft(y)))
% subplot(515)
% stem(abs(yhat))

% 3. alias cancellation
if ~isempty(zhat)

    % A. permute zhat in frequency domain, then downsample
    zhat_perm_ds = perm_freq_downsample(zhat, sigma, b, n, B);

    % B. circular convolution of length B
    % Corrected!
    v = cconv(filter.g_hat_prime, zhat_perm_ds, B);

    % v = ifft( fft(filter.g_hat_prime) .* fft(zhat_perm_ds) );

    % C. subtract
    yhat = yhat - v;
end

uhat = yhat;
end


function y = perm(x, sigma, a, b)
n = length(x);
t = (0:n-1).';
idx0 = mod(sigma * (t - a), n);
idx = idx0 + 1;
phase = exp(-2i*pi * (sigma * b .* t) / n);
y = x(idx) .* phase;
end


function zhat_perm_ds = perm_freq_downsample(zhat, sigma, b, n, B)

sigma_inv = modinv(sigma, n);
k = (1:B:n).'-1;

idx = mod(sigma_inv * k, n) + 1;
zhat_perm = zhat(idx);

phase = exp(-2i*pi * b * k / n);
zhat_perm_ds = zhat_perm .* phase;
end

function filt = make_gaussian_filter(N,B,sigma,center_bin)

if nargin < 3 || isempty(sigma), sigma = B/6; end
if nargin < 4 || isempty(center_bin), center_bin = 0; end

% 1. rectangular core + Gaussian taper
H = zeros(N,1);
start_bin = mod(center_bin - floor(B/2), N);
bins0 = mod(start_bin + (0:B-1), N);
H(bins0+1) = 1;

k = (0:N-1).';
d = mod(k - center_bin + N/2, N) - N/2;
gauss = exp(-0.5*(d./sigma).^2);
H = H .* gauss;

% 2. time-domain filter (NO TRUNCATION for correctness)
g = ifft(H);

% 3. downsampled freq response
g_hat_prime = H(1:B:end);

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

function ok = check_good_permutation(u_hat, up_hat, tol)
% Returns true if all bins with significant magnitude
% have |u_hat| ≈ |up_hat|.

if nargin < 3
    tol = 1e-6;
end

% Find candidate bins (only nonzero / large bins)
J = find(abs(u_hat) > 1e-3);

ok = true;

for jj = J
    mag1 = abs(u_hat(jj));
    mag2 = abs(up_hat(jj));

    if abs(mag1 - mag2) > tol
        ok = false;
        return;
    end
end
end
