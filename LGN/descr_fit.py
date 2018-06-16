import numpy as np
import helper_fcns as hf
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


def descr_loss(params, data, contrast, baseline = []):
    '''Given the model params (i.e. flexible gaussian params), the data, and the desired contrast
    (where contrast will be given as an index into the list of unique contrasts), return the loss
    '''
    respsSummary, stims, allResps = hf.tabulateResponses(data);

    f0 = allResps[0];
    obs_counts = f0[contrast];

    NLL = 0;
    all_sfs = np.unique(data['sf']);
    for i in range(len(all_sfs)):
      if all_sfs[i] > 0: # sf can be 0; ignore these...
        
        obs_spikes = obs_counts[i][~np.isnan(obs_counts[i])]; # only get the non-NaN values;
        if baseline:
          obs_spikes = np.maximum(0, obs_spikes - baseline); # cannot have <0 spikes!

        pred_spikes, _ = hf.DiffOfGauss(*params, stim_sf=all_sfs[i]*np.ones_like(obs_spikes));
        #pred_spikes = hf.flexible_Gauss(params, all_sfs[i]*np.ones_like(obs_spikes), minThresh=0);

        # poisson model of spiking
        poiss = poisson.pmf(np.round(obs_spikes), pred_spikes); # round since the values are nearly but not quite integer values (Sach artifact?)...
        ps = np.sum(poiss == 0);
        if ps > 0:
          poiss = np.maximum(poiss, 1e-6); # anything, just so we avoid log(0)
        NLL = NLL + sum(-np.log(poiss));
    
    return NLL;

def fit_descr_DoG(cell_num, data_loc, n_repeats = 4, baseline_sub = 0):

    nParam = 4;
    
    # load cell information
    dataList = hf.np_smart_load(data_loc + 'sachData.npy');
    fLname = 'descrFits';
    if baseline_sub:
      fLname = str(fLname + '_baseSub');
    fLname = str(data_loc + fLname + '.npy');
    if os.path.isfile(fLname):
        descrFits = hf.np_smart_load(fLname);
    else:
        descrFits = dict();
    data = dataList[cell_num-1]['data'];
    
    print('Doing the work, now');

    to_unpack = hf.tabulateResponses(data);
    if baseline_sub:
      base_mean, _ = hf.blankResp(data);
    else:
      base_mean = [];
    [f0, f1] = to_unpack[0];
    [all_cons, all_sfs] = to_unpack[1];
    [f0arr, f1arr] = to_unpack[2];
    
    nCons = len(all_cons);

    if cell_num-1 in descrFits:
        bestNLL = descrFits[cell_num-1]['NLL'];
        currParams = descrFits[cell_num-1]['params'];
    else: # set values to NaN...
        bestNLL = np.ones((nCons)) * np.nan;
        currParams = np.ones((nCons, nParam)) * np.nan;
    
    for con in range(nCons):    

        if all_cons[con] == 0: # skip 0 contrast...
            continue;

        print('.');

        maxResp = np.max(f0['mean'][con]);
        
        for n_try in range(n_repeats):

          # pick initial params
          init_gain = hf.random_in_range((0.5*maxResp, 0.9*maxResp))[0];
          init_charFreq = hf.random_in_range((1, 5))[0];
          init_gainSurr = hf.random_in_range((0.25, 0.75))[0];
          init_charFreqSurr = hf.random_in_range((0.25, 0.5))[0];

          init_params = [init_gain, init_charFreq, init_gainSurr, init_charFreqSurr];

          # choose optimization method
          if np.mod(n_try, 2) == 0:
              methodStr = 'L-BFGS-B';
          else:
              methodStr = 'TNC';

          obj = lambda params: descr_loss(params, data, con, base_mean);
          wax = opt.minimize(obj, init_params, method=methodStr); # unbounded...

          # compare
          NLL = wax['fun'];
          params = wax['x'];

          if np.isnan(bestNLL[con]) or NLL < bestNLL[con]:
              bestNLL[con] = NLL;
              currParams[con, :] = params;

    # update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(fLname):
        print('reloading descrFits...');
        descrFits = hf.np_smart_load(fLname);
    if cell_num-1 not in descrFits:
      descrFits[cell_num-1] = dict();
    descrFits[cell_num-1]['NLL'] = bestNLL;
    descrFits[cell_num-1]['params'] = currParams;

    np.save(fLname, descrFits);
    print('saving for cell ' + str(cell_num));

def fit_descr(cell_num, data_loc, n_repeats = 4):

    nParam = 5;
    
    # load cell information
    dataList = hf.np_smart_load(data_loc + 'sachData.npy');
    if os.path.isfile(data_loc + 'descrFits.npy'):
        descrFits = hf.np_smart_load(data_loc + 'descrFits.npy');
    else:
        descrFits = dict();
    data = dataList[cell_num-1]['data'];
    
    print('Doing the work, now');

    to_unpack = hf.tabulateResponses(data);
    [f0, f1] = to_unpack[0];
    [all_cons, all_sfs] = to_unpack[1];
    [f0arr, f1arr] = to_unpack[2];
    
    nCons = len(all_cons);

    if cell_num-1 in descrFits:
        bestNLL = descrFits[cell_num-1]['NLL'];
        currParams = descrFits[cell_num-1]['params'];
    else: # set values to NaN...
        bestNLL = np.ones((nCons)) * np.nan;
        currParams = np.ones((nCons, nParam)) * np.nan;
    
    for con in range(nCons):    

        if all_cons[con] == 0: # skip 0 contrast...
            continue;

        print('.');
        # set initial parameters - a range from which we will pick!
        base_rate = hf.blankResp(data)[0];
        if base_rate <= 3:
            range_baseline = (0, 3);
        else:
            range_baseline = (0.5 * base_rate, 1.5 * base_rate);

        valid_sf_inds = ~np.isnan(f0['mean'][con, :]);
        valid_sfs = all_sfs[valid_sf_inds];
        max_resp = np.amax(f0['mean'][con, valid_sf_inds]);
        range_amp = (0.5 * max_resp, 1.5);

        max_sf_index = np.argmax(f0['mean'][con, valid_sf_inds]); # what sf index gives peak response?
        mu_init = valid_sf_inds[max_sf_index];

        if max_sf_index == 0: # i.e. smallest SF center gives max response...
            range_mu = (mu_init/2, valid_sfs[max_sf_index + 3]);
        elif max_sf_index+1 == len(valid_sf_inds): # i.e. highest SF center is max
            range_mu = (valid_sfs[max_sf_index-2], mu_init);
        else:
            range_mu = ([max_sf_index-1], valid_sfs[max_sf_index+1]); # go +-1 indices from center

        log_bw_lo = 1; #  octave bandwidth...
        log_bw_hi = 3; # octave bandwidth...
        denom_lo = hf.bw_log_to_lin(log_bw_lo, mu_init)[0]; # get linear bandwidth
        denom_hi = hf.bw_log_to_lin(log_bw_hi, mu_init)[0]; # get lin. bw (cpd)
        range_denom = (denom_lo, denom_hi); # don't want 0 in sigma 

        # set bounds for parameters
        min_bw = 1/4; max_bw = 10; # ranges in octave bandwidth

        bound_baseline = (0, max_resp);
        bound_range = (0, 1.5*max_resp);
        bound_mu = (np.min(all_sfs[all_sfs>0]), np.max(all_sfs));
        bound_sig = (np.maximum(0.1, min_bw/(2*np.sqrt(2*np.log(2)))), max_bw/(2*np.sqrt(2*np.log(2)))); # Gaussian at half-height

        all_bounds = (bound_baseline, bound_range, bound_mu, bound_sig, bound_sig);

        for n_try in range(n_repeats):

            # pick initial params
            init_base = hf.random_in_range(range_baseline);
            init_amp = hf.random_in_range(range_amp);
            init_mu = hf.random_in_range(range_mu);
            init_sig_left = hf.random_in_range(range_denom);
            init_sig_right = hf.random_in_range(range_denom);

            init_params = [init_base, init_amp, init_mu, init_sig_left, init_sig_right];

            # choose optimization method
            if np.mod(n_try, 2) == 0:
                methodStr = 'L-BFGS-B';
            else:
                methodStr = 'TNC';

            obj = lambda params: descr_loss(params, data, con);
            wax = opt.minimize(obj, init_params, method=methodStr, bounds=all_bounds);

            # compare
            NLL = wax['fun'];
            params = wax['x'];

            if np.isnan(bestNLL[con]) or NLL < bestNLL[con] or invalid(currParams[con, :], all_bounds):
                bestNLL[con] = NLL;
                currParams[con, :] = params;

    # update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(data_loc + 'descrFits.npy'):
        print('reloading descrFits...');
        descrFits = hf.np_smart_load(data_loc + 'descrFits.npy');
    if cell_num-1 not in descrFits:
      descrFits[cell_num-1] = dict();
    descrFits[cell_num-1]['NLL'] = bestNLL;
    descrFits[cell_num-1]['params'] = currParams;

    np.save(data_loc + 'descrFits.npy', descrFits);
    print('saving for cell ' + str(cell_num));
                
if __name__ == '__main__':

    data_loc = '/home/pl1465/SF_diversity/LGN/sach-data/';
    #data_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/sach-data/';

    if len(sys.argv) < 2:
      print('uhoh...you need at least one argument here');
      print('First be cell number, second [optional] is number of fit iterations');
      exit();

    print('Running cell ' + sys.argv[1] + '...');

    if len(sys.argv) > 3: # specify baseline subtraction
      print(' for ' + sys.argv[2] + ' iterations' + ' with baseline sub? ' + sys.argv[3]);
      fit_descr_DoG(int(sys.argv[1]), data_loc, int(sys.argv[2]), int(sys.argv[3]));
    elif len(sys.argv) > 2: # specify number of fit iterations
      print(' for ' + sys.argv[2] + ' iterations');
      fit_descr_DoG(int(sys.argv[1]), data_loc, int(sys.argv[2]));
    else: # all trials in each iteration
      fit_descr_DoG(int(sys.argv[1]), data_loc);

