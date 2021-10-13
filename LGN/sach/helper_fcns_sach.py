import math
import numpy as np
import scipy.optimize as opt
import os, sys
import random
from time import sleep
from scipy.stats import poisson, sem
import pdb

# Don't duplicate work that's already done in the "main" helper_fcns

maindir = os.path.abspath('../../'); # it's two directories up
sys.path.insert(0, maindir);
##############
### Imports from "main" helper_fcns
##############
# -- basic things
from helper_fcns import nan_rm, np_smart_load, bw_lin_to_log, bw_log_to_lin, resample_array, descrFit_name, random_in_range
from helper_fcns import descrLoss_name, descrMod_name, descrFit_name
from helper_fcns import flatten_list as flatten
# -- rvc
from helper_fcns import rvc_mod_suff, rvc_fit_name, get_rvc_model
from helper_fcns import naka_rushton, get_rvcResp, rvc_fit
# -- sf 
from helper_fcns import dog_charFreq, dog_get_param, dog_init_params, deriv_gauss, compute_SF_BW, fix_params
from helper_fcns import DiffOfGauss, DoGsach, dog_prefSfMod, dog_charFreqMod
from helper_fcns import DoG_loss, get_descrResp
from helper_fcns import flexible_Gauss_np as flexible_Gauss
from helper_fcns import descr_prefSf as dog_prefSf # to keep the function call here unchanged

##############
### Code written *here*, i.e. just for Sach stuff
##############

# load_modParams - [UNUSED] load the 4 parameters from the Tony fits...

# var_explained - compute the variance explained given data/model fit

# dog_fit - used to fit the Diff of Gauss responses -- either separately for each con, or jointly for all cons within a given dispersion

# blankResp - return mean/std of blank responses (i.e. baseline firing rate) for Sach's experiment
# tabulateResponses - Organizes measured and model responses for Sach's experiment

# writeDataTxt - write [sf mean sem] for a given cell/contrast
# writeCellTxt - call writeDataTxt for all contrasts for a cell

def load_modParams(which_cell, contrast, loadPath='/home/pl1465/SF_diversity/LGN/sach/structures/tonyFits/'):
   
  nParams = 4;

  loadName = 'cell%d_con%d.txt+.fit' % (which_cell, contrast);
  fits = open(str(loadPath + loadName), 'r');
  allLines = fits.readlines();
  firstLine = allLines[0].split();
  fL = [float(x) for x in firstLine]

  return fL[0:nParams]; 

#######

def var_expl_direct(obs_mean, pred_mean):
  # Just compute variance explained given the data and model responses (assumed same SF for each)
  resp_dist = lambda x, y: np.sum(np.square(x-y))/np.maximum(len(x), len(y))
  var_expl = lambda m, r, rr: 100 * (1 - resp_dist(m, r)/resp_dist(r, rr));

  obs_grand_mean = np.mean(obs_mean) * np.ones_like(obs_mean); # make sure it's the same shape as obs_mean
    
  return var_expl(pred_mean, obs_mean, obs_grand_mean);

def var_explained(data, modParams, whichInd, DoGmodel=1, rvcModel=None):
  ''' given a set of responses and model parameters, compute the variance explained by the model (DoGsach)
      --- whichInd is either the contrast index (if doing SF tuning)
                              or SF index (if doing RVCs)
  '''
  resp_dist = lambda x, y: np.sum(np.square(x-y))/np.maximum(len(x), len(y))
  var_expl = lambda m, r, rr: 100 * (1 - resp_dist(m, r)/resp_dist(r, rr));

  respsSummary, stims, allResps = tabulateResponses(data); # Need to fit on f1 
  f1 = respsSummary[1];
  if rvcModel is None: # SF
    all_sfs = stims[1];
    obs_mean = f1['mean'][whichInd, :];
  else:
    all_cons = stims[0];
    obs_mean = f1['mean'][:, whichInd];

  if rvcModel is None: # then we're doing vExp for SF tuning
    pred_mean = get_descrResp(modParams, all_sfs, DoGmodel);
  else: # then we've getting RVC responses!
    pred_mean = get_rvcResp(modParams, cons, rvcMod)

  obs_grand_mean = np.mean(obs_mean) * np.ones_like(obs_mean); # make sure it's the same shape as obs_mean
    
  return var_expl(pred_mean, obs_mean, obs_grand_mean);

## - for fitting DoG models

def dog_fit(resps, all_cons, all_sfs, DoGmodel, loss_type, n_repeats, joint=False, ref_varExpl=None, veThresh=65, fracSig=1):
  ''' Helper function for fitting descriptive funtions to SF responses
      if joint=True, (and DoGmodel is 1 or 2, i.e. not flexGauss), then we fit assuming
      a fixed ratio for the center-surround gains and [freq/radius]
      - i.e. of the 4 DoG parameters, 2 are fit separately for each contrast, and 2 are fit 
        jointly across all contrasts!
      - note that ref_varExpl (optional) will be of the same form that the output for varExpl will be

      inputs: self-explanatory, except for resps, which should be "f1" from tabulateResponses 
      outputs: bestNLL, currParams, varExpl, prefSf, charFreq, [overallNLL, paramList; if joint=True]
  '''
  nCons = len(all_cons);
  if DoGmodel == 0:
    nParam = 5;
  else:
    nParam = 4;

  # unpack responses
  resps_mean = resps['mean'];
  resps_sem = resps['sem'];

  # next, let's compute some measures about the responses
  max_resp = np.nanmax(resps_mean.flatten());
  min_resp = np.nanmin(resps_mean.flatten());
  ############
  ### WARNING - we're subtracting min_resp-1 from all responses
  ############  
  resps_mean = np.subtract(resps_mean, min_resp-1); # i.e. make the minimum response 1 spk/s...

  # and set up initial arrays
  bestNLL = np.ones((nCons, )) * np.nan;
  currParams = np.ones((nCons, nParam)) * np.nan;
  varExpl = np.ones((nCons, )) * np.nan;
  prefSf = np.ones((nCons, )) * np.nan;
  charFreq = np.ones((nCons, )) * np.nan;
  if joint==True:
    overallNLL = np.nan;
    params = np.nan;

  ### set bounds
  if DoGmodel == 0:
    min_bw = 1/4; max_bw = 10; # ranges in octave bandwidth
    bound_baseline = (0, max_resp);
    bound_range = (0, 1.5*max_resp);
    bound_mu = (0.01, 10);
    bound_sig = (np.maximum(0.1, min_bw/(2*np.sqrt(2*np.log(2)))), max_bw/(2*np.sqrt(2*np.log(2)))); # Gaussian at half-height
    if fracSig:
      bound_sigFrac = (0.2, 2);
      allBounds = (bound_baseline, bound_range, bound_mu, bound_sig, bound_sigFrac);
    else:
      allBounds = (bound_baseline, bound_range, bound_mu, bound_sig, bound_sig);
  elif DoGmodel == 1: # SACH
    bound_gainCent = (1e-3, None);
    bound_radiusCent= (1e-3, None);
    bound_gainSurr = (1e-2, 1); # multiplier on gainCent, thus the center must be weaker than the surround
    bound_radiusSurr = (1, 25); # multiplier on radiusCent, thus the surr. radius must be larger than the center
    if joint==True: # TODO: Is this ok with reparameterization?
      bound_gainRatio = (1e-3, 1); # the surround gain will always be less than the center gain
      bound_radiusRatio= (1, None); # the surround radius will always be greater than the ctr r
      # we'll add to allBounds later, reflecting joint gain/radius ratios common across all cons
      allBounds = (bound_gainRatio, bound_radiusRatio);
    else:
      allBounds = (bound_gainCent, bound_radiusCent, bound_gainSurr, bound_radiusSurr);
  elif DoGmodel == 2:
    bound_gainCent = (1e-3, None);
    bound_freqCent = (1e-3, 2e1);
    if joint==True:
      bound_gainRatio = (1e-3, 1); # surround gain always less than center gain
      bound_freqRatio = (1e-1, 1); # surround freq always less than ctr freq
      # we'll add to allBounds later, reflecting joint gain/radius ratios common across all cons
      allBounds = (bound_gainRatio, bound_freqRatio);
    elif joint==False:
      bound_gainFracSurr = (1e-3, 1);
      bound_freqFracSurr = (1e-1, 1);
      allBounds = (bound_gainCent, bound_freqCent, bound_gainFracSurr, bound_freqFracSurr);

  ### organize responses -- and fit, if joint=False
  allResps = []; allRespsSem = []; valCons = []; start_incl = 0; incl_inds = [];
  base_rate = np.min(resps_mean.flatten());
  for con in range(nCons):
    if all_cons[con] == 0: # skip 0 contrast...
        continue;
    else:
      valCons.append(con);
    resps_curr = resps_mean[con, :];
    sem_curr   = resps_sem[con, :];

    if ref_varExpl is None:
      start_incl = 1; # hacky...
    if start_incl == 0:
      if ref_varExpl[con] < veThresh:
        continue; # i.e. we're not adding this; yes we could move this up, but keep it here for now
      else:
        start_incl = 1; # now we're ready to start adding to our responses that we'll fit!

    ### prepare for the joint fitting, if that's what we've specified!
    if joint==True:
      allResps.append(resps_curr);
      allRespsSem.append(sem_curr);
      incl_inds.append(con);
      # and add to the bounds list!
      if DoGmodel == 1:
        allBounds = (*allBounds, bound_gainCent, bound_radiusCent);
      elif DoGmodel == 2:
        allBounds = (*allBounds, bound_gainCent, bound_freqCent);
      continue;

    ### otherwise, we're really going to fit here! [i.e. if joint is False]
    # first, specify the objection function!
    obj = lambda params: DoG_loss(params, resps_curr, all_sfs, resps_std=sem_curr, loss_type=loss_type, DoGmodel=DoGmodel, joint=joint);

    #pdb.set_trace();

    for n_try in range(n_repeats):
      ###########
      ### pick initial params
      ###########
      init_params = dog_init_params(resps_curr, base_rate, all_sfs, all_sfs, DoGmodel, fracSig=fracSig, bounds=allBounds)

      # choose optimization method
      if np.mod(n_try, 2) == 0:
          methodStr = 'L-BFGS-B';
      else:
          methodStr = 'TNC';
          
      try:
        wax = opt.minimize(obj, init_params, method=methodStr, bounds=allBounds);
      except:
        continue; # the fit has failed (bound issue, for example); so, go back to top of loop, try again
      
      # compare
      NLL = wax['fun'];
      params = wax['x'];

      if np.isnan(bestNLL[con]) or NLL < bestNLL[con]:
        bestNLL[con] = NLL;
        currParams[con, :] = params;
        curr_mod = get_descrResp(params, all_sfs, DoGmodel);
        varExpl[con] = var_expl_direct(resps_curr[all_sfs>0], curr_mod[all_sfs>0]); # do not include 0/cdeg SF conditoin
        prefSf[con] = dog_prefSf(params, dog_model=DoGmodel, all_sfs=all_sfs[all_sfs>0]); # do not include 0 c/deg SF condition
        charFreq[con] = dog_charFreq(params, DoGmodel=DoGmodel);

  if joint==False: # then we're DONE
    return bestNLL, currParams, varExpl, prefSf, charFreq, None, None; # placeholding None for overallNLL, params [full list]

  ### NOW, we do the fitting if joint=True
  if joint==True:
    ### now, we fit!
    for n_try in range(n_repeats):
      # we'll also add to allInitParams later, to estimate the center gain/radius
      if DoGmodel == 1:
        # -- but first, we estimate the gain ratio and the radius ratio
        allInitParams = [random_in_range((0.2, 0.8))[0], random_in_range((2, 3.5))[0]];
      elif DoGmodel == 2:
        # -- but first, we estimate the gain ratio and the cfreq ratio
        allInitParams = [random_in_range((0.2, 0.8))[0], random_in_range((0.2, 0.4))[0]];
      for resps_curr in allResps:
        # now, we need to guess the local parameters - 0:2 will give us center gain/shape
        curr_init = dog_init_params(resps_curr, all_sfs, DoGmodel)[0:2];
        allInitParams = [*allInitParams, curr_init[0], curr_init[1]];

      # choose optimization method
      if np.mod(n_try, 2) == 0:
          methodStr = 'L-BFGS-B';
      else:
          methodStr = 'TNC';

      obj = lambda params: DoG_loss(params, allResps, all_sfs, resps_std=allRespsSem, loss_type=loss_type, DoGmodel=DoGmodel, joint=joint);
      wax = opt.minimize(obj, allInitParams, method=methodStr, bounds=allBounds);

      # compare
      NLL = wax['fun'];
      params_curr = wax['x'];

      if np.isnan(overallNLL) or NLL < overallNLL:
        overallNLL = NLL;
        params = params_curr;

    ### then, we must unpack the fits to actually fill in the "true" parameters for each contrast
    gain_rat, shape_rat = params[0], params[1];
    for con in range(len(incl_inds)):
      # get the current parameters and responses by unpacking the associated structures
      local_gain = params[2+con*2]; 
      local_shape = params[3+con*2]; # shape, as in radius/freq, depending on DoGmodel
      if DoGmodel == 1: # i.e. sach
        curr_params = [local_gain, local_shape, local_gain*gain_rat, local_shape*shape_rat];
      elif DoGmodel == 2: # i.e. Tony
        curr_params = [local_gain, local_shape, gain_rat, shape_rat];
      # -- then the responses, and overall contrast index
      resps_curr = allResps[con];
      sem_curr   = allRespsSem[con];
      
      # now, compute!
      conInd = incl_inds[con];
      bestNLL[conInd] = DoG_loss(curr_params, resps_curr, all_sfs, resps_std=sem_curr, loss_type=loss_type, DoGmodel=DoGmodel, joint=False); # now it's NOT joint!
      currParams[conInd, :] = curr_params;
      curr_mod = get_descrResp(curr_params, all_sfs, DoGmodel);
      varExpl[conInd] = var_expl_direct(resps_curr, curr_mod);
      prefSf[conInd] = dog_prefSf(curr_params, dog_model=DoGmodel, all_sfs=all_sfs);
      charFreq[conInd] = dog_charFreq(curr_params, DoGmodel=DoGmodel);    

    # and NOW, we can return!
    return bestNLL, currParams, varExpl, prefSf, charFreq, overallNLL, params;
##


#####

#####

def blankResp(data):
  blanks = np.where(data['cont'] == 0);

  mu = np.mean(data['f0'][blanks]);
  std = np.std(data['f0'][blanks]);

  return mu, std;

def tabulateResponses(data, resample=False):
  ''' Given the dictionary containing all of the data, organize the data into the proper responses
  Specifically, we know that Sach's experiments varied contrast and spatial frequency
  Thus, we will organize responses along these dimensions
  '''
  all_cons = np.unique(data['cont']);
  all_cons = all_cons[all_cons>0];
  all_sfs = np.unique(data['sf']);

  f0 = dict();
  f0mean= np.nan * np.zeros((len(all_cons), len(all_sfs))); 
  f0sem = np.nan * np.zeros((len(all_cons), len(all_sfs))); 
  f1 = dict();
  f1mean = np.nan * np.zeros((len(all_cons), len(all_sfs))); 
  f1sem = np.nan * np.zeros((len(all_cons), len(all_sfs))); 

  # rather than getting just the mean/s.e.m., we can also record/transfer the firing rate of each individual stimulus run
  f0arr = dict();
  f1arr = dict();
  
  for con in range(len(all_cons)):
    val_con = np.where(data['cont'] == all_cons[con]);
    f0arr[con] = dict();
    f1arr[con] = dict();
    for sf in range(len(all_sfs)):
      val_sf = np.where(data['sf'][val_con] == all_sfs[sf]);

      if resample: # we'll do it manually, since we want to keep f0/f1 resamplings aligned
        non_nan = np.where(~np.isnan(data['f1arr'][val_con][val_sf]))[-1]; # we accidentally create a singleton 1st dim. with this indexing; ignore it
        new_inds = np.random.choice(non_nan, len(non_nan));
        f0arr[con][sf] = nan_rm(data['f0arr'][val_con][val_sf][0][new_inds]); # internal [0] is again due to poor indexing
        f1arr[con][sf] = nan_rm(data['f1arr'][val_con][val_sf][0][new_inds])
      else:
        f0arr[con][sf] = nan_rm(data['f0arr'][val_con][val_sf]);
        f1arr[con][sf] = nan_rm(data['f1arr'][val_con][val_sf]);

      # take mean, since some conditions have repeats - just average them
      f0mean[con, sf] = np.mean(f0arr[con][sf]); #np.mean(data['f0'][val_con][val_sf]);
      f0sem[con, sf] = sem(f0arr[con][sf]); #np.mean(data['f0sem'][val_con][val_sf]);
      f1mean[con, sf] = np.mean(f1arr[con][sf]); #np.mean(data['f1'][val_con][val_sf]);
      f1sem[con, sf] = sem(f1arr[con][sf]); #np.mean(data['f1sem'][val_con][val_sf]);


  f0['mean'] = f0mean;
  f0['sem'] = f0sem;
  f1['mean'] = f1mean;
  f1['sem'] = f1sem;

  return [f0, f1], [all_cons, all_sfs], [f0arr, f1arr];

def writeDataTxt(cellNum, f1, sfs, contrast, save_loc):
  
  obs_mean = f1['mean'][contrast, :];
  obs_sem = f1['sem'][contrast, :];
   
  write_name = 'cell%d_con%d.txt' % (cellNum, contrast);
  file = open(str(save_loc + write_name), 'w');

  for i in range(len(sfs)):
    file.write('%.3f %.3f %.3f\n' % (sfs[i], obs_mean[i], obs_sem[i]));

  file.close();

def writeCellTxt(cellNum, load_path, save_loc):

  dataList = np_smart_load(load_path + 'sachData.npy');
  data = dataList[cellNum-1]['data'];

  resps, conds, _ = tabulateResponses(data);
  f1 = resps[1];
  all_cons = conds[0];
  all_sfs = conds[1];
  
  for i in range(len(all_cons)):
    writeDataTxt(cellNum, f1, all_sfs, i, save_loc);
