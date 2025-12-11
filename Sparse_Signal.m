% This class defines a signal whose frequency content is sparse
classdef Sparse_Signal
    properties
        % Length / periodicity  of the signal (N) 
        N                                           % Postive Integer
        % Sparsity of the signal in frequency (k)
        k                                           % Positive Integer
        % List of frequency components (k numbers from 0 to N-1)
        frequencies                                 % k dim vector in [0 N-1]
        % List of fourier series coefficients 
        coefficients                                % k dim vector in Complex
        % Signal in time domain 
        value                                       % N dim vector in Complex
        % Noise variance / power 
        noise_var                                   % Non-negative real number
    end

    methods
        % output = sparse signal object 
        function signal_object = Sparse_Signal(signal_length,sparsity,frequencies,coefficients, noise_var)
            signal_object.N = signal_length;
            signal_object.k = sparsity;
            
            % Provide provision for random / user chosen frequencies 
            if isempty(frequencies)
                signal_object.frequencies = randi([0 signal_length-1],[1 sparsity]);
            else
                signal_object.frequencies = frequencies;
            end
            
            % Provide provision for random / user chosen coefficients
            if isempty(coefficients)
                signal_object.coefficients = randn([1 sparsity]) + 1i*randn([1 sparsity]);
            else
                signal_object.coefficients = coefficients;
            end
            
            % Creating the sparse vector containing fourier coefficients
            frequency_vector = zeros([1 signal_length]);
            frequency_vector(signal_object.frequencies + 1) = signal_object.coefficients;
            
            % Power of signma^2 in complex => sigma^2 / 2 in real and imag
            noise_vector = (randn([1 signal_length]) + 1i*randn([1 signal_length]))*sqrt(noise_var/2);
            signal_object.noise_var = noise_var;
            
            % IFFT scales down by N, so we multiply by N premptively
            signal_object.value = ifft(frequency_vector*signal_object.N /100) + noise_vector;
        end
        
        % output = frequency representation using built-in FFT
        function [frequency_representation, time] = Built_in_FFT(signal_object)
            signal = signal_object.value;
            % Mandatory warning message
            if signal_object.noise_var ~= 0
                disp("WARNING: Signal is noisy, output may be erroneous")
            end
            
            % Choosing random index = 1 (or n = 0) for quickest compute
            tic
            frequency_representation = fft(signal) / length(signal);
            time = toc;
        end

        % output = frequency representation for sparsity = 1 by phase encoding
        function [frequency_representation, time] = Phase_Encoding(signal_object)
            signal = signal_object.value;
            % Check if conditions are met
            if signal_object.k ~= 1
                disp("WARNING: Sparsity of signal is not equal to 1, output may be erroneous")
            elseif signal_object.noise_var ~= 0
                disp("WARNING: Signal is noisy, output may be erroneous")
            end
            frequency_representation = zeros([1 length(signal)]);
            % Choosing random index = 1 (or n = 0) for quickest compute
            tic
            
            frequency_representation(floor(mod(angle(signal(2)/signal(1)) * length(signal) / (2*pi) , length(signal)) + 1)) = signal(1);
            time = toc;
        end
    
        % output = frequency representation for sparsity = 1 by binary search
        function [frequency_representation, time] = CRT_single_frequency(signal_object)
            signal = signal_object.value;
            len = signal_object.N;
            % Check if conditions are met
            if signal_object.k ~= 1
                disp("WARNING: Sparsity of signal is not equal to 1, output may be erroneous")
            elseif signal_object.noise_var ~= 0
                disp("WARNING: Signal is noisy, output may be erroneous")
            end

            factors = getOptimalCoprimes(len);
            total_len = prod(factors);
            signal = [signal zeros([1 total_len - len])];
            residues = zeros([1 length(factors)]);
            frequency_representation = zeros([1 len]);
            tic
            
            for control = 1:length(factors)
                [~, ind] = max(abs(fft(signal(1:ceil(length(signal)/factors(control)):factors(control)*ceil(length(signal)/factors(control))))));
                residues(control) = ind-1;
            end
            frequency_representation(round((crt(residues, factors))*(len/total_len) + 1)) = signal(1);
            time = toc;
        end
    end     
end

