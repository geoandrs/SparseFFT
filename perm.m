function y = perm(x, sigma, a, b)
% P  Apply permutation P_{sigma,a,b} to time-domain vector x
%   y = P_sigma_ab(x, sigma, a, b)
%
% This implements a commonly used permutation in SFT variants:
%   (P_{sigma,a,b} x)[t] = x[ sigma * (t - a) mod n ] * exp(-2pi i * b * t / n)
%
% Inputs:
%   x     - column vector (n x 1)
%   sigma - integer (should be coprime with n for invertibility; paper uses odd sigma)
%   a,b   - integers (shifts)
%
% Output:
%   y     - permuted and phased column vector (n x 1)
n = length(x);
t = (0:n-1).';               % 0-based time indices
idx0 = mod(sigma * (t - a), n);  % 0-based source indices
idx = idx0 + 1;              % convert to 1-based for MATLAB
% phase factor: exp(-2pi i * b * t / n)
phase = exp(-2i*pi * (sigma * b .* t) / n);
y = x(idx) .* phase;
end
