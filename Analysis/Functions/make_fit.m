function [model, fit, NLL] = make_fit(cell, sf_bound, n_attempts, prev_fits, fit_descr_model)

% 1/30/17 - Added ability to fit descriptive model responses (e.g. Robbe V1
% model...) 
% We also make initial parameter guesses based on actual cell responses rather than
% model responses. Again, just to get things off the ground.

model = 'flexible_Gauss';

if exist('prev_fits', 'var')
    if ~isempty(prev_fits) % nargin() == 4
        how_many_more = 4; % we'll try our start parameters from previous families AND values from the regular fit
    else
        how_many_more = 2;
    end
else
    how_many_more = 2; % we'll try our start parameters from previous families
end

% if we're told to fit responses of the descriptive model (1) or not (0)
% then do as told; otherwise, don't fit descr. model responses
if ~exist('fit_descr_model', 'var')
    fit_descr_model = 0;
end

if fit_descr_model
    data.exp_trial  = cell.sfm.exp.trial;
    data.model_resp = cell.sfm.mod.sim.resp_full;
    data.sf_c       = cell.sfm.mod.sim.sf_c;
else
    data = cell.sfm.exp;
end
    

for contrast_hi = 1 : 2
    for family = 1 : 5
        
        %% init ranges for gaussian parameters
        baseline = cell.sfm.exp.sponRateMean;
        if baseline <= 0
            baseline_range = [ 0 3 ];
        else
            baseline_range = [ .8 * baseline 1.2 * baseline ];
        end
 
        maxResp = max(cell.sfm.exp.sfRateMean{family}{contrast_hi}) - baseline;
        amp_range = [ .8 * maxResp 1.2 * maxResp];
        
        log_bw = 1.5; % in octaves; reasonable guess
        
        [~, max_sf_index] = max(cell.sfm.exp.sfRateMean{family}{contrast_hi});
        mu = cell.sfm.exp.sf{family}{contrast_hi}(max_sf_index);
        
        if max_sf_index == 1
            mu_range = [ mu cell.sfm.exp.sf{family}{contrast_hi}(max_sf_index + 2) ];
        elseif max_sf_index == length(cell.sfm.exp.sf{family}{contrast_hi})
            mu_range = [ cell.sfm.exp.sf{family}{contrast_hi}(max_sf_index - 2) mu ];
        else
            mu_range = [ cell.sfm.exp.sf{family}{contrast_hi}(max_sf_index - 1) cell.sfm.exp.sf{family}{contrast_hi}(max_sf_index + 1) ];
        end
  
        denom = bw_log_to_lin(log_bw, mu);
        denom_range = [ 0.8 * denom 1.2 * denom ];
 
        % in octaves, at half-height
        min_bw = 1/4;
        max_bw = 20;
        sig_limits = [min_bw/(2*sqrt(2*log(2))) max_bw/(2*sqrt(2*log(2)))]; % by solving Gaussian at half-height
        
        for attempt = 1 : n_attempts + how_many_more

            if mod(attempt, 2) == 1
                options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'sqp');
            else
                options = optimoptions('fmincon', 'Display', 'none', 'Algorithm', 'interior-point');
            end
            
            %% pick initial values from ranges
            base_val = random_in_range(baseline_range);
            amp_val = random_in_range(amp_range);
            mu_val = random_in_range(mu_range);
            denom_l_val = random_in_range(denom_range);
            denom_r_val = random_in_range(denom_range);

            init_calc = [base_val amp_val mu_val denom_l_val denom_r_val];

            %% set initial values
            if attempt > n_attempts + 2
                init_guess{family}{contrast_hi} = prev_fits{family}{contrast_hi};
            elseif attempt > n_attempts % and not > n_attempts
                if family ~= 1
                    init_guess{family}{contrast_hi} = fit{family-1}{contrast_hi};
                elseif family == 1 && contrast_hi == 1
                    init_guess{family}{contrast_hi} = init_calc;
                elseif family == 1 && contrast_hi == 2
                    init_guess{family}{contrast_hi} = fit{5}{1}; 
                end
            else
                init_guess{family}{contrast_hi} = init_calc;
            end

            curr_init = init_guess{family}{contrast_hi};
            
            %% set bounds

            lower = zeros(5, 1);                            upper = zeros(5, 1);
            lower(1) = 0;                                   upper(1) = inf;                 % baseline amplitude
            lower(2) = 0;                                   upper(2) = 2*maxResp;           % amplitude above baseline
            lower(3) = sf_bound(1);                        	upper(3) = sf_bound(end);       % preferred spatial frequency preference
            lower(4) = sig_limits(1);                       upper(4) = sig_limits(2);       % "sigma" for LHS of curve
            lower(5) = sig_limits(1);                       upper(5) = sig_limits(2);       % "sigma" for RHS of curve

%         [curr_init, options, lower, upper] = make_initial_params(cell, family, contrast_hi, fit, sf_bound, n_attempts, prev_fits); 

            %% fit
            
            obj = @(params) compute_NLL(model, params, data, family, contrast_hi);
            [try_fit, try_NLL] = fmincon(obj, curr_init, [], [], [], [], lower, upper, [], options);

            if attempt == 1
                fit{family}{contrast_hi} = try_fit;
                NLL{family}{contrast_hi} = try_NLL;
                best_NLL = try_NLL;
            elseif abs(try_NLL) < abs(best_NLL)
                fit{family}{contrast_hi} = try_fit;
                NLL{family}{contrast_hi} = try_NLL;
                best_NLL = try_NLL;
            end

            fit{family}{contrast_hi} = fix_params(fit{family}{contrast_hi});
            
        end

    end
end

end