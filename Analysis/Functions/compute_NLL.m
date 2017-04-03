function [NLL] = compute_NLL(model, params, data, family, contrast_hi)

% Crucial! Objective function for minimization routines used in fitting
% descriptive model to sfMix tuning curves

% 1/30/17 - Adjusted to allow for fitting of descriptive model responses!


% if we pass in 'exp_trial', then we're fitting (functional) model responses
% see compute_NLL for more detail (exp_trial/model_resp)
if isfield(data, 'exp_trial')
    trial = data.exp_trial;
    spike_counts = data.model_resp;
    sf_c = data.sf_c;
    
    fit_desc_model = 1;
% otherwise, fitting real responses...
else
    trial = data.trial;
    spike_counts = data.trial.spikeCount;
    
    fit_desc_model = 0;
end

% if we're fitting the descriptive model, we already have responses
% organized properly!
if fit_desc_model

    % spike_counts{family}{contrast_hi} is [sf_centers X repeats]
    obs_count = [];
    sf_c_all = [];
    for sf_i = 1 : numel(sf_c)
            
        % why round? because we need integer values/spike counts, not rates...
        sp_counts = round(spike_counts{family}{contrast_hi}(sf_i, :));
        obs_count = [obs_count sp_counts];
        
        num_repeats = size(spike_counts{family}{contrast_hi}, 2);
        sf_c_all = [sf_c_all repmat(sf_c(sf_i), [1 num_repeats])];
        
    end
    
    pred_rate = eval(sprintf('%s(params, sf_c_all)', model));
        
    % what is stim_dur? well...it's basically the same for all presented
    % stimuli. Regardless, the range is on the order of 0.001 s, which is
    % minimal given a ~1s stimulus
    stim_dur_mode = mode(trial.duration);
    stim_dur = repmat(stim_dur_mode, size(sf_c_all));
    
% otherwise, do the work to get the spike counts 
else
    % set constants
    epsilon = 1e-4;   

    % get indices for trials we want to look at
    center_con = get_center_con(family, contrast_hi);
    ori_pref = mode(trial.ori{1}(:));

    con_check = abs(trial.con{1}(:) - center_con) < epsilon;
    ori_check = abs(trial.ori{1}(:) - ori_pref) < epsilon;

    sf_check = zeros(size(con_check));

    sf_centers  = data.sf{1}{1};

    for sf_i = 1 : length(sf_centers)
        sf_check = (trial.sf{1}(:) == sf_centers(sf_i)) | sf_check;
    end

    indices = find(con_check & ori_check & sf_check);

    obs_count = spike_counts(indices);
    pred_rate = eval(sprintf('%s(params, data.trial.sf{1}(indices))', model));
    

    stim_dur = trial.duration(indices);
    
end


% % Get predicted spike count distributions
% mu  = max(.1, stim_dur .* pred_rate);                                  % The predicted mean spike count
% var = mu + (varGain*(mu.^2));                                          % The corresponding variance of the spike count
% r   = (mu.^2)./(var - mu);                                             % The parameters r and p of the negative binomial distribution
% p   = r./(r + mu);
% 
% % Evaluate the model
% llh = nbinpdf(obs_count, r, p);                                        % The likelihood for each pass under the doubly stochastic model
% 
% NLL = sum(-log(llh));                                                  % The negative log-likelihood of the whole data-set      

NLL = sum(-log(poisspdf(obs_count, pred_rate .* stim_dur)));

end

