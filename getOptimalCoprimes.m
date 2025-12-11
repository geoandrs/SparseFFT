function best_factors = getOptimalCoprimes(N)
% GETOPTIMALCOPRIMES Returns the optimal set of coprime factors for N.
%
% Usage:
%   factors = getOptimalCoprimes(2000)
%
% Returns:
%   best_factors: A vector of coprime numbers (e.g., [2 3 5 7 11])

    % 1. Setup
    % Primes up to 100 cover almost all practical signal processing needs.
    P_LIST = primes(700); 
    
    % Internal trackers for the best solution found so far
    min_cost = inf;
    best_factors = [];
    
    % 2. Start Recursive Search
    solve_recursive(1, 1, 0, []);
    
    % --- NESTED SEARCH FUNCTION ---
    function solve_recursive(idx, current_prod, current_sum, current_factors)
        
        % Optimization: Stop if this path is already more expensive than our best found
        if current_sum >= min_cost
            return;
        end
        
        % Goal Check: If we reached the target N, save this result
        if current_prod >= N
            min_cost = current_sum;
            best_factors = current_factors;
            return;
        end
        
        % Safety: Stop if we run out of primes
        if idx > length(P_LIST)
            return;
        end
        
        p = P_LIST(idx);
        
        % PATH A: Skip this prime (don't use it)
        solve_recursive(idx + 1, current_prod, current_sum, current_factors);
        
        % PATH B: Use this prime (and try its powers p, p^2, p^3...)
        p_pow = p;
        while true
            new_sum = current_sum + p_pow;
            new_prod = current_prod * p_pow;
            
            % Stop if this specific power is too expensive
            if new_sum >= min_cost
                break;
            end
            
            % If valid, check if we are done or need to recurse
            if new_prod >= N
                min_cost = new_sum;
                best_factors = [current_factors, p_pow];
                break; % Found a valid set using this power, no need to go higher
            else
                % Keep searching with the next prime
                solve_recursive(idx + 1, new_prod, new_sum, [current_factors, p_pow]);
            end
            
            % Prepare next power (e.g., 2 -> 4 -> 8)
            if p_pow > N 
                break;
            end
            p_pow = p_pow * p;
        end
    end
end