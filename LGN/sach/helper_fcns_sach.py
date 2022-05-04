import math
import descr_fit as df
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
from helper_fcns import polar_vec_mean, phase_fit_name, phase_advance, project_resp, get_phAdv_model
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

def dog_fit(resps, all_cons, all_sfs, DoGmodel, loss_type, n_repeats, joint=0, ref_varExpl=None, veThresh=-np.nan, fracSig=1, ftol=2.220446049250313e-09, jointMinCons=3):
  ''' Helper function for fitting descriptive funtions to SF responses
      if joint=True, (and DoGmodel is 1 or 2, i.e. not flexGauss), then we fit assuming
      a fixed ratio for the center-surround gains and [freq/radius]
      - i.e. of the 4 DoG parameters, 2 are fit separately for each contrast, and 2 are fit 
        jointly across all contrasts!
      - note that ref_varExpl (optional) will be of the same form that the output for varExpl will be
      - note that jointMinCons is the minimum # of contrasts that must be included for a joint fit to be run (e.g. 2)

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
  #resps_mean = np.subtract(resps_mean, min_resp-1); # i.e. make the minimum response 1 spk/s...

  # and set up initial arrays
  bestNLL = np.ones((nCons, ), dtype=np.float32) * np.nan;
  currParams = np.ones((nCons, nParam), dtype=np.float32) * np.nan;
  varExpl = np.ones((nCons, ), dtype=np.float32) * np.nan;
  prefSf = np.ones((nCons, ), dtype=np.float32) * np.nan;
  charFreq = np.ones((nCons, ), dtype=np.float32) * np.nan;
  if joint>0:
    overallNLL = np.nan;
    params = np.nan;
    success = False;
  else:
    success = np.zeros((nCons, ), dtype=np.bool_);

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
    bound_gainCent = (1, 3*max_resp);
    bound_radiusCent= (1e-2, 1.5);
    bound_gainSurr = (1e-2, 1); # multiplier on gainCent, thus the center must be weaker than the surround
    bound_radiusSurr = (1, 10); # (1,10) # multiplier on radiusCent, thus the surr. radius must be larger than the center
    if joint>0:
      if joint == 1: # original joint (fixed gain and radius ratios across all contrasts)
        bound_gainRatio = (1e-3, 1); # the surround gain will always be less than the center gain
        bound_radiusRatio= (1, 10); # the surround radius will always be greater than the ctr r
        # we'll add to allBounds later, reflecting joint gain/radius ratios common across all cons
        allBounds = (bound_gainRatio, bound_radiusRatio);
      elif joint == 2: # fixed surround radius for all contrasts
        allBounds = (bound_radiusSurr, );
      elif joint == 3: # fixed center AND surround radius for all contrasts
        allBounds = (bound_radiusCent, bound_radiusSurr);
    else:
      allBounds = (bound_gainCent, bound_radiusCent, bound_gainSurr, bound_radiusSurr);
  elif DoGmodel == 2:
    bound_gainCent = (1e-3, None);
    bound_freqCent = (1e-3, 2e1);
    bound_gainFracSurr = (1e-3, 2); # surround gain always less than center gain NOTE: SHOULD BE (1e-3, 1)
    bound_freqFracSurr = (5e-2, 1); # surround freq always less than ctr freq NOTE: SHOULD BE (1e-1, 1)
    if joint>0:
      if joint == 1: # original joint (fixed gain and radius ratios across all contrasts)
        bound_gainRatio = (1e-3, 3);
        bound_freqRatio = (1e-1, 1); 
        # we'll add to allBounds later, reflecting joint gain/radius ratios common across all cons
        allBounds = (bound_gainRatio, bound_freqRatio);
      elif joint == 2: # fixed surround radius for all contrasts
        allBounds = (bound_freqFracSurr,);
      elif joint == 3: # fixed center AND surround radius for all contrasts
        allBounds = (bound_freqCent, bound_freqFracSurr);
    elif joint==0:
      bound_gainFracSurr = (1e-3, 1);
      bound_freqFracSurr = (1e-1, 1);
      allBounds = (bound_gainCent, bound_freqCent, bound_gainFracSurr, bound_freqFracSurr);

  ### organize responses -- and fit, if joint=0
  allResps = []; allRespsSem = []; allSfs = []; valCons = []; start_incl = 0; incl_inds = [];
  base_rate = np.min(resps_mean.flatten());
  for con in range(nCons):
    if all_cons[con] == 0: # skip 0 contrast...
        continue;
    else:
      valCons.append(con);
    resps_curr = resps_mean[con, :];
    sem_curr   = resps_sem[con, :];

    ### prepare for the joint fitting, if that's what we've specified!
    if joint>0:
      if ref_varExpl is None:
        start_incl = 1; # hacky...
      if start_incl == 0:
        if ref_varExpl[con] < veThresh:
          continue; # i.e. we're not adding this; yes we could move this up, but keep it here for now
        else:
          start_incl = 1; # now we're ready to start adding to our responses that we'll fit!

      allResps.append(resps_curr);
      allRespsSem.append(sem_curr);
      allSfs.append(all_sfs);
      incl_inds.append(con);
      # and add to the bounds list!
      if DoGmodel == 1:
        if joint == 1: # add the center gain and center radius for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_radiusCent);
        if joint == 2: # add the center and surr. gain and center radius for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_radiusCent, bound_gainSurr);
        if joint == 3:  # add the center and surround gain for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_gainSurr);
      elif DoGmodel == 2:
        if joint == 1: # add the center gain and center radius for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_freqCent);
        if joint == 2: # add the center and surr. gain and center radius for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_freqCent, bound_gainFracSurr);
        if joint == 3:  # add the center and surround gain for each contrast 
          allBounds = (*allBounds, bound_gainCent, bound_gainFracSurr);

      continue;

    ### otherwise, we're really going to fit here! [i.e. if joint is False]
    # first, specify the objection function!
    obj = lambda params: DoG_loss(params, resps_curr, all_sfs, resps_std=sem_curr, loss_type=loss_type, DoGmodel=DoGmodel, joint=joint);

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
        success[con] = wax['success'];

  if joint==0: # then we're DONE
    return bestNLL, currParams, varExpl, prefSf, charFreq, None, None, success; # placeholding None for overallNLL, params [full list]

  ### NOW, we do the fitting if joint=True
  if joint>0:
    if len(allResps)<jointMinCons: # need at least jointMinCons contrasts!
      return bestNLL, currParams, varExpl, prefSf, charFreq, overallNLL, params, success;
    ### now, we fit!
    for n_try in range(n_repeats):
      # first, estimate the joint parameters; then we'll add the per-contrast parameters after
      # --- we'll estimate the joint parameters based on the high contrast response
      ref_resps = allResps[-1];
      ref_init = dog_init_params(ref_resps, base_rate, all_sfs, all_sfs, DoGmodel);
      if joint == 1: # gain ratio (i.e. surround gain) [0] and shape ratio (i.e. surround radius) [1] are joint
        allInitParams = [ref_init[2], ref_init[3]];
      elif joint == 2: #  surround radius [0] (as ratio) is joint
        allInitParams = [ref_init[3]];
      elif joint == 3: # center radius [0] and surround radius [1] ratio are joint
        allInitParams = [ref_init[1], ref_init[3]];

      # now, we cycle through all responses and add the per-contrast parameters
      for resps_curr in allResps:
        curr_init = dog_init_params(resps_curr, base_rate, all_sfs, all_sfs, DoGmodel);
        if joint == 1:
          allInitParams = [*allInitParams, curr_init[0], curr_init[1]];
        elif joint == 2: # then we add center gain, center radius, surround gain (i.e. params 0:3
          allInitParams = [*allInitParams, curr_init[0], curr_init[1], curr_init[2]];
        elif joint == 3: # then we add center gain and surround gain (i.e. params 0, 2)
          allInitParams = [*allInitParams, curr_init[0], curr_init[2]];

      methodStr = 'L-BFGS-B';
      obj = lambda params: DoG_loss(params, allResps, allSfs, resps_std=allRespsSem, loss_type=loss_type, DoGmodel=DoGmodel, joint=joint, n_fits=len(allResps)); # if joint, it's just one fit!
      wax = opt.minimize(obj, allInitParams, method=methodStr, bounds=allBounds, options={'ftol': ftol});

      # compare
      NLL = wax['fun'];
      params_curr = wax['x'];

      if np.isnan(overallNLL) or NLL < overallNLL:
        overallNLL = NLL;
        params = params_curr;
        success = wax['success'];

    ### Done with multi-start fits; now, unpack the fits to fill in the "true" parameters for each contrast
    # --- first, get the global parameters
    if joint == 1:
      gain_rat, shape_rat = params[0], params[1];
    elif joint == 2:
      surr_shape = params[0]; # radius or frequency, if Tony model
    elif joint == 3:
      center_shape, surr_shape = params[0], params[1]; # radius or frequency, if Tony model
    for con in range(len(allResps)):
      # --- then, go through each contrast and get the "local", i.e. per-contrast, parameters
      if joint == 1: # center gain, center shape
        center_gain = params[2+con*2]; 
        center_shape = params[3+con*2]; # shape, as in radius/freq, depending on DoGmodel
        curr_params = [center_gain, center_shape, gain_rat, shape_rat];
      elif joint == 2: # center gain, center radus, surround gain
        center_gain = params[1+con*3]; 
        center_shape = params[2+con*3];
        surr_gain = params[3+con*3];
        curr_params = [center_gain, center_shape, surr_gain, surr_shape];
      elif joint == 3: # center gain, surround gain
        center_gain = params[2+con*2]; 
        surr_gain = params[3+con*2];
        curr_params = [center_gain, center_shape, surr_gain, surr_shape];
      # -- then the responses, and overall contrast index
      resps_curr = allResps[con];
      sem_curr   = allRespsSem[con];

      # now, compute!
      conInd = incl_inds[con];
      bestNLL[conInd] = DoG_loss(curr_params, resps_curr, all_sfs, resps_std=sem_curr, loss_type=loss_type, DoGmodel=DoGmodel, joint=0); # now it's NOT joint!
      currParams[conInd, :] = curr_params;
      curr_mod = get_descrResp(curr_params, all_sfs, DoGmodel);
      varExpl[conInd] = var_expl_direct(resps_curr, curr_mod);
      prefSf[conInd] = dog_prefSf(curr_params, dog_model=DoGmodel, all_sfs=all_sfs[all_sfs>0]);
      charFreq[conInd] = dog_charFreq(curr_params, DoGmodel=DoGmodel);    

    # and NOW, we can return!
    return bestNLL, currParams, varExpl, prefSf, charFreq, overallNLL, params, success;
##


#####

#####

def blankResp(data, get_dc=0):
  blanks = np.where(data['cont'] == 0);

  key = 'f0' if get_dc else 'f1';
  mu = np.mean(data[key][blanks]);
  std = np.std(data[key][blanks]);

  return mu, std;

def tabulateResponses(data, resample=False, sub_f1_blank=False, phAdjusted=0, dir=1):
  ''' Given the dictionary containing all of the data, organize the data into the proper responses
  Specifically, we know that Sach's experiments varied contrast and spatial frequency
  Thus, we will organize responses along these dimensions
  - NOTE: If phAdjusted=1, then we return the phase-adjusted responses (amplitudes)!
          if phAdjusted=0, then we return vec-corrected but NOT phase-amplitude adjusted 
          if phAdjusted=-1, then we do the (dumb, non-vector) scalar average
  ----  : We discovered on 22.04.07 that Sach's mean F1 phase/amplitude were not done using proper vector math (i.e. he simply took the mean of the amplitudes)
  ----  : So, we not only do the proper vector math but also apply the phase-amplitude relationship correction that we apply for my own LGN data
  - NOTE (ONLY APPLIED PRE-PHASE CORRECTION): Sach's data has marked offset, even on F1 -- if sub_f1_blank is True, we'll subtract that off
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
  f1arr_prePhCorr = dict();

  to_sub = blankResp(data, get_dc=False)[0] if sub_f1_blank else 0;

  if phAdjusted==1:
    # phAdv_model is used to project the responses; all_opts is organized by SF (ascending)
    phAdv_model, all_opts = df.phase_advance_fit(data, None, 'phAdv_dummy', dir=dir, to_save=0);
    #phAdv_model, all_opts_neg = df.phase_advance_fit(data, None, 'phAdv_dummy', dir=-1, to_save=0);

  for con in range(len(all_cons)):
    val_con = np.where(data['cont'] == all_cons[con]);
    f0arr[con] = dict();
    f1arr[con] = dict();
    f1arr_prePhCorr[con] = dict();
    for sf in range(len(all_sfs)):
      val_sf = np.where(data['sf'][val_con] == all_sfs[sf]);

      if resample: # we'll do it manually, since we want to keep f0/f1 resamplings aligned
        non_nan = np.where(~np.isnan(data['f1arr'][val_con][val_sf]))[-1]; # we accidentally create a singleton 1st dim. with this indexing; ignore it
        new_inds = np.random.choice(non_nan, len(non_nan));
        f0arr[con][sf] = nan_rm(data['f0arr'][val_con][val_sf][0][new_inds]); # internal [0] is again due to poor indexing
        f1amps = nan_rm(data['f1arr'][val_con][val_sf][0][new_inds] - to_sub)
        f1phs = nan_rm(data['f1pharr'][val_con][val_sf][0][new_inds]);
        if phAdjusted==1:
          f1arr[con][sf] = project_resp([f1amps], [f1phs], phAdv_model, [all_opts[sf]], disp=0)[0];
        elif phAdjusted==0:
          mean_amp, mean_ph,_,_ = polar_vec_mean([f1amps], [f1phs]);
          f1arr[con][sf] = np.multiply(f1amps, np.cos(np.deg2rad(mean_ph) - np.deg2rad(f1phs)));
        elif phAdjusted==-1:
          f1arr[con][sf] = f1amps;
      else: # TODO: Could make this and the prior block less redundant
        f0arr[con][sf] = nan_rm(data['f0arr'][val_con][val_sf]);
        f1amps = nan_rm(data['f1arr'][val_con][val_sf] - to_sub)
        f1phs = nan_rm(data['f1pharr'][val_con][val_sf]);
        if phAdjusted==1:
          if con>(len(all_cons)-3): # i.e. a high contrast..
            pdb.set_trace();
          f1arr[con][sf] = project_resp([f1amps], [f1phs], phAdv_model, [all_opts[sf]], disp=0)[0];
          f1arr_prePhCorr[con][sf] = f1amps;
        elif phAdjusted==0:
          mean_amp, mean_ph,_,_ = polar_vec_mean([f1amps], [f1phs]);
          f1arr[con][sf] = np.multiply(f1amps, np.cos(np.deg2rad(mean_ph) - np.deg2rad(f1phs)));
        elif phAdjusted==-1:
          f1arr[con][sf] = f1amps;

      # take mean, since some conditions have repeats - just average them
      # --- this applies regardless of phAdjustment, since the amplitudes would then be corrected
      f0mean[con, sf] = np.mean(f0arr[con][sf]); #np.mean(data['f0'][val_con][val_sf]);
      f0sem[con, sf] = sem(f0arr[con][sf]); #np.mean(data['f0sem'][val_con][val_sf]);
      f1mean[con, sf] = np.mean(f1arr[con][sf]); #np.mean(data['f1'][val_con][val_sf]);
      #f1mean_prePhCorr[con, sf] = polar_vec_mean(f1amps, f1phs);
      f1sem[con, sf] = sem(f1arr[con][sf]); #np.mean(data['f1sem'][val_con][val_sf]);

  f0['mean'] = f0mean;
  f0['sem'] = f0sem;
  f1['mean'] = f1mean;
  #f1['mean_prePhCorr'] = f1mean_prePhCorr;
  f1['sem'] = f1sem;

  return [f0, f1], [all_cons, all_sfs], [f0arr, f1arr]#; [f0arr, f1arr, f1arr_prePhCorr];

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
