import numpy as np
import model_responses as mod_resp
import helper_fcns as hfunc
import scipy.optimize as opt
from scipy.stats import norm, mode, lognorm, nbinom, poisson
from numpy.matlib import repmat
import os.path
import sys

import pdb

def invalid(params, bounds):
# given parameters and bounds, are the parameters valid?
  for p in range(len(params)):
    if params[p] < bounds[p][0] or params[p] > bounds[p][1]:
      return True;
  return False;
    

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

    respMetrics, stimVals, val_con_by_disp, validByStimVal, ignore = hfunc.tabulate_responses(data);

    # get indices for trials we want to look at
    valid_disp = validByStimVal[0];
    valid_con = validByStimVal[1];
    
    # family, contrast are in absolute terms (i.e. we pass over non-valid ones, so we can just index normally)
    curr_con = valid_con[contrast];
    curr_disp = valid_disp[family];

    indices = np.where(curr_con & curr_disp); 
    
    obs_count = trial['spikeCount'][indices];

    pred_rate = flexible_Gauss(params, trial['sf'][0][indices]);
    stim_dur = trial['duration'][indices];

    # poisson model of spiking
    poiss = poisson.pmf(obs_count, pred_rate * stim_dur);
    ps = np.sum(poiss == 0);
    if ps > 0:
      poiss = np.maximum(poiss, 1e-6); # anything, just so we avoid log(0)
    NLL = sum(-np.log(poiss));
    
    return NLL;

def fit_descr(cell_num, data_loc, n_repeats = 4):

    nParam = 5;
    
    # load cell information
    dataList = np.load(data_loc + 'dataList.npy').item();
    if os.path.isfile(data_loc + 'descrFits.npy'):
        descrFits = np.load(data_loc + 'descrFits.npy').item();
    else:
        descrFits = dict();
    data = np.load(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy').item();
    
    print('Doing the work, now');

    to_unpack = hfunc.tabulate_responses(data);
    [respMean, respVar] = to_unpack[0];
    [all_disps, all_cons, all_sfs] = to_unpack[1];
    val_con_by_disp = to_unpack[2];
    
    nDisps = len(all_disps);
    nCons = len(all_cons);

    if cell_num-1 in descrFits:
        bestNLL = descrFits[cell_num-1]['NLL'];
        currParams = descrFits[cell_num-1]['params'];
    else: # set values to NaN...
        bestNLL = np.ones((nDisps, nCons)) * np.nan;
        currParams = np.ones((nDisps, nCons, nParam)) * np.nan;
    
    for family in range(nDisps):
        for con in range(nCons):    
            
            if con not in val_con_by_disp[family]:
                continue;

            print('.');           
            # set initial parameters - a range from which we will pick!
            base_rate = hfunc.blankResp(data)[0];
            if base_rate <= 3:
                range_baseline = (0, 3);
            else:
                range_baseline = (0.5 * base_rate, 1.5 * base_rate);

            valid_sf_inds = ~np.isnan(respMean[family, :, con]);
            max_resp = np.amax(respMean[family, valid_sf_inds, con]);
            range_amp = (0.5 * max_resp, 1.5);
            
            theSfCents = all_sfs[valid_sf_inds];
            
            max_sf_index = np.argmax(respMean[family, valid_sf_inds, con]); # what sf index gives peak response?
            mu_init = theSfCents[max_sf_index];
            
            if max_sf_index == 0: # i.e. smallest SF center gives max response...
                range_mu = (mu_init/2,theSfCents[max_sf_index + 3]);
            elif max_sf_index+1 == len(theSfCents): # i.e. highest SF center is max
                range_mu = (theSfCents[max_sf_index-2], mu_init);
            else:
                range_mu = (theSfCents[max_sf_index-1], theSfCents[max_sf_index+1]); # go +-1 indices from center
                
            log_bw_lo = 0.75; # 0.75 octave bandwidth...
            log_bw_hi = 2; # 2 octave bandwidth...
            denom_lo = hfunc.bw_log_to_lin(log_bw_lo, mu_init)[0]; # get linear bandwidth
            denom_hi = hfunc.bw_log_to_lin(log_bw_hi, mu_init)[0]; # get lin. bw (cpd)
            range_denom = (denom_lo, denom_hi); # don't want 0 in sigma 
                
            # set bounds for parameters
            min_bw = 1/4; max_bw = 10; # ranges in octave bandwidth

            bound_baseline = (0, max_resp);
            bound_range = (0, 1.5*max_resp);
            bound_mu = (0.01, 10);
            bound_sig = (np.maximum(0.1, min_bw/(2*np.sqrt(2*np.log(2)))), max_bw/(2*np.sqrt(2*np.log(2)))); # Gaussian at half-height
            
            all_bounds = (bound_baseline, bound_range, bound_mu, bound_sig, bound_sig);

            for n_try in range(n_repeats):
                
                # pick initial params
                init_base = hfunc.random_in_range(range_baseline);
                init_amp = hfunc.random_in_range(range_amp);
                init_mu = hfunc.random_in_range(range_mu);
                init_sig_left = hfunc.random_in_range(range_denom);
                init_sig_right = hfunc.random_in_range(range_denom);
                         
                init_params = [init_base, init_amp, init_mu, init_sig_left, init_sig_right];
                         
                # choose optimization method
                if np.mod(n_try, 2) == 0:
                    methodStr = 'L-BFGS-B';
                else:
                    methodStr = 'TNC';
                
                obj = lambda params: descr_loss(params, data, family, con);
                wax = opt.minimize(obj, init_params, method=methodStr, bounds=all_bounds);
                
                # compare
                NLL = wax['fun'];
                params = wax['x'];

                if np.isnan(bestNLL[family, con]) or NLL < bestNLL[family, con] or invalid(currParams[family, con, :], all_bounds):
                    bestNLL[family, con] = NLL;
                    currParams[family, con, :] = params;

    # update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(data_loc + 'descrFits.npy'):
        print('reloading descrFits...');
        descrFits = np.load(data_loc + 'descrFits.npy').item();
    if cell_num-1 not in descrFits:
      descrFits[cell_num-1] = dict();
    descrFits[cell_num-1]['NLL'] = bestNLL;
    descrFits[cell_num-1]['params'] = currParams;

    np.save(data_loc + 'descrFits.npy', descrFits);
    print('saving for cell ' + str(cell_num));
                
if __name__ == '__main__':

    data_loc = '/home/pl1465/SF_diversity/altExp/analysis/structures/';

    if len(sys.argv) < 2:
      print('uhoh...you need at least one argument here');
      print('First be cell number, second [optional] is number of fit iterations');
      exit();

    print('Running cell ' + sys.argv[1] + '...');

    if len(sys.argv) > 2: # specify number of fit iterations
      print(' for ' + sys.argv[2] + ' iterations');
      fit_descr(int(sys.argv[1]), data_loc, int(sys.argv[2]));
    else: # all trials in each iteration
      fit_descr(int(sys.argv[1]), data_loc);

