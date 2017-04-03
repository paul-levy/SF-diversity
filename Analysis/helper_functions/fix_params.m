function params_out = fix_params(params_in)

% simply makes all input arguments positive
 
% R(Sf) = R0 + K_e * EXP(-(SF-mu)^2 / 2*(sig_e)^2) - K_i * EXP(-(SF-mu)^2 / 2*(sig_i)^2)
% params
% 1 (R0)        -- must be positive
% 2 (K_e)       -- " 
% 3 (K_i)       -- "
% 4 (mu)        -- "
% 5 (sig_e)     -- "
% 6 (sig_i)     -- "

params_out = abs(params_in); 

end

