function [freq,coeff] = phase_encoding(x)

    N = length(x);
    idx = randi([0,N-1]);
    next_idx = mod(idx+1,N);
    
    ratio = x(next_idx+1)/x(idx+1);
    
    freq = round(mod(atan2(imag(ratio),real(ratio))*N/2/pi,N));
    coeff = x(idx+1)*exp(-2j*pi*freq*(idx+1)/N);

end