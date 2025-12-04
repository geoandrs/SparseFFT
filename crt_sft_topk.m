function [freqs, coeffs] = crt_sft_topk(x, k, thresh)
% CRT_SFT_THRESHOLD - CRT-based SFT, keep all peaks above threshold
%   x - input signal (length N)
%   thresh - magnitude threshold for peak detection
%
% Returns frequency bins (0-based) and estimated coefficients

x = x(:);  
N = length(x);  

% --- Prime-power moduli ---  
pf = prime_factorization(N);  
unique_pf = unique(pf);  
mods = zeros(1, length(unique_pf));  
for i = 1:length(unique_pf)  
    mods(i) = unique_pf(i)^sum(pf == unique_pf(i));  
end  
num_mods = length(mods);

if num_mods == 1
    Xfft = fft(x);
    [coeffs, freqs] = maxk(abs(Xfft), k);
    freqs = sort(freqs - 1);
    return
end

% --- Collect residues per modulus ---  
residues_list = cell(1, num_mods);  
coeffs_list   = cell(1, num_mods);  

for i = 1:num_mods  
    m = mods(i);  
    step = N/m;  
    if abs(step - round(step)) > 1e-12, error("N/mod must be integer."); end  
    step = round(step);  

    xs = x(mod((0:m-1)*step, N)+1);  

    Xs = fft(xs);  
    idx_above_thresh = find(abs(Xs) >= thresh);  
    residues_list{i} = idx_above_thresh - 1;  % 0-based  
    coeffs_list{i} = Xs(idx_above_thresh);  
end  

% --- Generate all CRT combinations ---  
all_combinations = cell(1, num_mods);  
for i = 1:num_mods  
    all_combinations{i} = residues_list{i};  
end  
[grid{1:num_mods}] = ndgrid(all_combinations{:});  
num_candidates = numel(grid{1});  

freqs_cand = zeros(num_candidates,1);  
coeffs_cand = zeros(num_candidates,1); 

for idx = 1:num_candidates  
    a = zeros(1,num_mods);  
    for j = 1:num_mods  
        a(j) = grid{j}(idx);  
    end  
    freqs_cand(idx) = chinese_remainder(a, mods);  

    % Approximate coefficient: mean of corresponding modulus coefficients  
    c = zeros(1,num_mods);  
    for j = 1:num_mods  
        c(j) = coeffs_list{j}(find(residues_list{j}==a(j),1));  
    end  
    coeffs_cand(idx) = mean(c);  
end  

% Return all candidates above threshold  
mask = abs(coeffs_cand) >= thresh;  
freqs = freqs_cand(mask);  
coeffs = coeffs_cand(mask);  

[freqs_unique, ~, ic] = unique(freqs);
coeffs_merged = zeros(size(freqs_unique));
for i = 1:length(freqs_unique)
    coeffs_merged(i) = mean(coeffs(ic == i));
end
freqs = freqs_unique;
coeffs = coeffs_merged;

% [coeffs,idd] = sort(coeffs,"descend");
% freqs = freqs(idd);
% 
% coeffs = coeffs(1:k);
% freqs = freqs(1:k);


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prime factorization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function primes_list = prime_factorization(n)
    primes_list = [];
    d = 2;
    while n > 1
        while mod(n,d) == 0
            primes_list(end+1) = d; %#ok<AGROW>
            n = n/d;
        end
        d = d + 1;
        if d*d > n && n > 1
            primes_list(end+1) = n;
            break;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chinese Remainder Theorem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = chinese_remainder(a, n)
    N = prod(n);
    x = 0;
    for i = 1:length(n)
        Ni = N / n(i);
        Mi = mod_inverse(Ni, n(i));
        x = x + a(i) * Mi * Ni;
    end
    x = mod(x, N);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modular Inverse
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function inv = mod_inverse(a, m)
    [g,x,~] = gcd_ext(a, m);
    if g ~= 1
        error("No modular inverse exists for a=%d mod m=%d", a, m);
    end
    inv = mod(x, m);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extended Euclid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [g, x, y] = gcd_ext(a, b)
    if b == 0
        g = a; x = 1; y = 0;
    else
        [g, x1, y1] = gcd_ext(b, mod(a,b));
        x = y1;
        y = x1 - floor(a/b)*y1;
    end
end

