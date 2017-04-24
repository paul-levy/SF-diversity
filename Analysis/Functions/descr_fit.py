import numpy as np
import helper_functions as hfunc
import os
from scipy.stats import norm, mode, lognorm, nbinom
from numpy.matlib import repmat

def flexible_Gauss(params, stim_sf):
    # The descriptive model used to fit cell tuning curves - in this way, we
    # can read off preferred SF, octave bandwidth, and response amplitude

    respFloor       = params[0];
    respRelFloor    = params[1];
    sfPref          = params[2];
    sigmaLow        = params[3];
    sigmaHigh       = params[4];

    # Tuning function
    sf0   = stim_sf / sfPref;

    # set the sigma - left half (i.e. sf0<1) is sigmaLow; right half (sf0>1) is sigmaHigh
    sigma = sigmaLow * np.ones(len(stim_sf));
    sigma[sf0 > 1] = sigmaHigh;

    shape = np.exp(-np.square(np.log(sf0)) / (2*np.square(sigma)));
                
    return np.maximum(0.1, respFloor + respRelFloor*shape);

def descr_loss(params, data, family, contrast):
    
    # set constants
    epsilon = 1e-4;
    trial = data['sfm']['exp']['trial'];

    # get indices for trials we want to look at
    center_con = hfunc.get_center_con(family, contrast_hi);
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