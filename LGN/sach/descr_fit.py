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

def descr_loss(params, f1, all_sfs, contrast, loss_type = 3, DoGmodel = 1):
    '''Given the model params (i.e. flexible gaussian params), the f1 mean/sem by contrast, and the desired contrast
    (where contrast will be given as an index into the list of unique contrasts), return the loss
    loss_type: 1 - poisson
               2 - sqrt
               3 - Sach sum{[(exp-obs)^2]/[k+sigma^2]} where
                   k := 0.01*max(obs); sigma := measured variance of the response
               4 - variance explained!
    '''
    obs_mean = f1['mean'][contrast, :];
    obs_sem = f1['sem'][contrast, :];

    NLL = 0;

    if DoGmodel == 1:
      pred_mean, _ = hf.DoGsach(*params, stim_sf=all_sfs);
    elif DoGmodel == 2:
      pred_mean, _ = hf.DiffOfGauss(*params, stim_sf=all_sfs);

    if loss_type == 1:
      # poisson model of spiking
      poiss = poisson.pmf(np.round(obs_mean), pred_mean); # round since the values are nearly but not quite integer values (Sach artifact?)...
      ps = np.sum(poiss == 0);
      if ps > 0:
        poiss = np.maximum(poiss, 1e-6); # anything, just so we avoid log(0)
      NLL = NLL + sum(-np.log(poiss));
    elif loss_type == 2:
      loss = np.sum(np.square(np.sqrt(obs_mean) - np.sqrt(pred_mean)));
      NLL = NLL + loss;
    elif loss_type == 3:
      k = 0.01*np.max(obs_mean);
      #sigma = np.ones_like(obs_mean);
      sigma = obs_sem;
      sq_err = np.square(obs_mean-pred_mean);
      NLL = NLL + np.sum(sq_err/(k+np.square(sigma)));
    elif loss_type == 4:
      resp_dist = lambda x, y: np.sum(np.square(x-y))/np.maximum(len(x), len(y))
      var_expl = lambda m, r, rr: 100 * (1 - resp_dist(m, r)/resp_dist(r, rr));
      # rr is the mean of all observed mean responses (and is last argument in below line)
      NLL = NLL - var_expl(pred_mean, obs_mean, np.mean(obs_mean) * np.ones_like(obs_mean));
    #print('NLL %.2f || params %s' % (NLL, str(params)));
    
    return NLL;

def fit_descr_DoG(cell_num, data_loc, n_repeats = 4, loss_type = 3, DoGmodel = 1):

    nParam = 4;
    
    # load cell information
    dataList = hf.np_smart_load(data_loc + 'sachData.npy');
    assert dataList!=[], "data file not found!"

    fLname = 'descrFits_d181012';
    if loss_type == 1:
      loss_str = '_poiss';
    elif loss_type == 2:
      loss_str = '_sqrt';
    elif loss_type == 3:
      loss_str = '_sach';
    elif loss_type == 4:
      loss_str = '_varExpl';
    if DoGmodel == 1:
      mod_str = '_sach';
    elif DoGmodel == 2:
      mod_str = '_tony';
    fLname = str(data_loc + fLname + loss_str + mod_str + '.npy');
    if os.path.isfile(fLname):
        descrFits = hf.np_smart_load(fLname);
    else:
        descrFits = dict();

    data = dataList[cell_num-1]['data'];
    
    print('Doing the work, now');

    to_unpack = hf.tabulateResponses(data);
    [_, f1] = to_unpack[0];
    [all_cons, all_sfs] = to_unpack[1];
    [_, f1arr] = to_unpack[2];
    
    nCons = len(all_cons);

    if cell_num-1 in descrFits:
      bestNLL = descrFits[cell_num-1]['NLL'];
      currParams = descrFits[cell_num-1]['params'];
      varExpl = descrFits[cell_num-1]['varExpl'];
      prefSf = descrFits[cell_num-1]['prefSf'];
      charFreq = descrFits[cell_num-1]['charFreq'];
    else: # set values to NaN...
      bestNLL = np.ones((nCons)) * np.nan;
      currParams = np.ones((nCons, nParam)) * np.nan;
      varExpl = np.ones((nCons)) * np.nan;
      prefSf = np.ones((nCons)) * np.nan;
      charFreq = np.ones((nCons)) * np.nan;

    # set bounds
    if DoGmodel == 1:
      bound_gainCent = (1e-3, None);
      bound_radiusCent= (1e-3, None);
      bound_gainSurr = (1e-3, None);
      bound_radiusSurr= (1e-3, None);
      allBounds = (bound_gainCent, bound_radiusCent, bound_gainSurr, bound_radiusSurr);
    elif DoGmodel == 2:
      bound_gainCent = (1e-3, None);
      bound_gainFracSurr = (1e-2, 1);
      bound_freqCent = (1e-3, None);
      bound_freqFracSurr = (1e-2, 1);
      allBounds = (bound_gainCent, bound_freqCent, bound_gainFracSurr, bound_freqFracSurr);
    
    for con in range(nCons):    
        if all_cons[con] == 0: # skip 0 contrast...
            continue;

        print('.');

        f1Means = f1['mean'][con];
        maxResp = np.max(f1Means);
        freqAtMaxResp = all_sfs[np.argmax(f1Means)];

        for n_try in range(n_repeats):
          # pick initial params
          if DoGmodel == 1:
            init_gainCent = hf.random_in_range((maxResp, 5*maxResp))[0];
            init_radiusCent = hf.random_in_range((0.05, 2))[0];
            init_gainSurr = init_gainCent * hf.random_in_range((0.1, 0.8))[0];
            init_radiusSurr = hf.random_in_range((0.5, 4))[0];
            init_params = [init_gainCent, init_radiusCent, init_gainSurr, init_radiusSurr];
          elif DoGmodel == 2:
            init_gainCent = maxResp * hf.random_in_range((0.9, 1.2))[0];
            init_freqCent = np.maximum(all_sfs[2], freqAtMaxResp * hf.random_in_range((1.2, 1.5))[0]); # don't pick all_sfs[0] -- that's zero (we're avoiding that)
            init_gainFracSurr = hf.random_in_range((0.7, 1))[0];
            init_freqFracSurr = hf.random_in_range((.25, .35))[0];
            init_params = [init_gainCent, init_freqCent, init_gainFracSurr, init_freqFracSurr];

          # choose optimization method
          if np.mod(n_try, 2) == 0:
              methodStr = 'L-BFGS-B';
          else:
              methodStr = 'TNC';

          obj = lambda params: descr_loss(params, f1, all_sfs, con, loss_type, DoGmodel);
          wax = opt.minimize(obj, init_params, method=methodStr, bounds=allBounds); # unbounded...

          # compare
          NLL = wax['fun'];
          params = wax['x'];

          if np.isnan(bestNLL[con]) or NLL < bestNLL[con]:
              bestNLL[con] = NLL;
              currParams[con, :] = params;
              varExpl[con] = hf.var_explained(data, params, con, DoGmodel);
              prefSf[con] = hf.dog_prefSf(params, all_sfs, DoGmodel);
              charFreq[con] = hf.dog_charFreq(params, DoGmodel);

    # update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(fLname):
        print('reloading descrFits...');
        descrFits = hf.np_smart_load(fLname);
    if cell_num-1 not in descrFits:
      descrFits[cell_num-1] = dict();
    descrFits[cell_num-1]['NLL'] = bestNLL;
    descrFits[cell_num-1]['params'] = currParams;
    descrFits[cell_num-1]['varExpl'] = varExpl;

    np.save(fLname, descrFits);
    print('saving for cell ' + str(cell_num));
                
if __name__ == '__main__':

    data_loc = '/home/pl1465/SF_diversity/LGN/sach/structures/';
    #data_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/sach-data/';

    if len(sys.argv) < 2:
      print('uhoh...you need at least one argument here');
      print('First be cell number, second [optional] is number of fit iterations');
      exit();

    print('Running cell ' + sys.argv[1] + '...');

    if len(sys.argv) > 4: # specify loss function, DoG model
      print(' for ' + sys.argv[2] + ' iterations' + ' with loss type? ' + sys.argv[3] + ' and DoG model ' + sys.argv[4]);
      fit_descr_DoG(int(sys.argv[1]), data_loc, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]));
    elif len(sys.argv) > 3: # specify loss type
      print(' for ' + sys.argv[2] + ' iterations' + ' loss type? ' + sys.argv[3]);
      fit_descr_DoG(int(sys.argv[1]), data_loc, int(sys.argv[2]), int(sys.argv[3]));
    elif len(sys.argv) > 2: # specify # iterations
      print(' for ' + sys.argv[2] + ' iterations');
      fit_descr_DoG(int(sys.argv[1]), data_loc, int(sys.argv[2]));
    else: # all trials in each iteration
      fit_descr_DoG(int(sys.argv[1]), data_loc);

