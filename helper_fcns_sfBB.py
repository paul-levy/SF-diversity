import numpy as np
import itertools
import helper_fcns as hf
from scipy.stats import sem

import pdb

### Similar to helper_fcns, but meant specifically for the sfBB_* series of experiments

### Helper/Basic ###
# get_resp_str - return 'dc' or 'f1' depending in which response measure we're using
# get_mask_base_inds - the answer is in the name! 0 index for mask, 1 for base (when array is [...,2]
# get_valid_trials - get the list of valid trials corresponding to a particular stimulus condition
# resample_all_cond - resample an array of arbitrary size, specifying across which axis

### Organize responses
# get_baseOnly_resp - get the response to the base stimulus ONLY
# get_mask_resp - get the response to the mask OR mask+base at either the base or mask TF
# adjust_f1_byTrial - to the vector math adjustment on each trial
# phase_advance_core

### Analysis ###
# --- Computing F1 response (including phase), computing F1::F0 ratio, etc
# get_vec_avg_response
# compute_f1f0
# get_all_responses - organize all responses in a dictionary!

### HELPER/BASIC ###

def get_resp_str(respMeasure): 
  # return 'dc' or 'f1' depending in which response measure we're using
  if respMeasure == 0:
    return 'dc';
  elif respMeasure == 1:
    return 'f1';
  else:
    return 'ERROR';

def get_mask_base_inds():
  return 0,1; # we return responses as mask (i.e. [:,0]), then base (i.e. [:,1])

def get_valid_trials(expInfo, maskOn, baseOn, whichCon=None, whichSf=None, baseCon=0, baseSf=0, returnStimConds=0):
  # Given a stimulus condition (index, not stim. value), return the corresponding set of trial numbers

  byTrial = expInfo['trial'];
  conDig = 2
  maskInd, baseInd = get_mask_base_inds();
  # Gather all possible stimulus conditions
  maskSfs  = expInfo['maskSF'];
  maskCons = np.round(expInfo['maskCon'], conDig);
  baseSfs  = expInfo['baseSF'];
  baseCons = np.round(expInfo['baseCon'], conDig);

  # find the right trials
  whichComps = np.logical_and(byTrial['maskOn']==maskOn, byTrial['baseOn']==baseOn)
  if maskOn == 0: # i.e. it doesn't matter which mask
    whichSfs = np.ones_like(whichComps);
    whichCons = np.ones_like(whichComps);
  else:
    whichSfs = byTrial['sf'][maskInd, :] == maskSfs[whichSf]
    whichCons = np.round(byTrial['con'][maskInd, :], conDig) == maskCons[whichCon]
  if baseOn == 0:
    whichBaseSf = np.ones_like(whichComps);
    whichBaseCon = np.ones_like(whichComps);
  else:
    # - including the base stimulus (it doesn't matter whether we have baseOn or not...
    whichBaseSf = byTrial['sf'][baseInd, :] == baseSfs[baseSf];
    whichBaseCon = np.round(byTrial['con'][baseInd, :], conDig) == baseCons[baseCon];
  whichBase = np.logical_and(whichBaseSf, whichBaseCon);

  val_trials = np.where(np.logical_and(np.logical_and(whichComps, whichBase), np.logical_and(whichSfs, whichCons)))[0]

  if returnStimConds:
    nTrials = len(val_trials);
    # - phase
    stimPh = np.zeros((nTrials, 2)); # ,2 for [mask, base]
    stimPh[:,maskInd] = byTrial['ph'][maskInd, val_trials];
    stimPh[:,baseInd] = byTrial['ph'][baseInd, val_trials];
    # - TF
    stimTf = np.zeros((nTrials, 2)); # ,2 for [mask, base]
    stimTf[:,maskInd] = byTrial['tf'][maskInd, val_trials];
    stimTf[:,baseInd] = byTrial['tf'][baseInd, val_trials];
    return val_trials, stimPh, stimTf; # why not stimSf, stimCon -- those are specified in the function call!
  else:
    return val_trials;    

def resample_all_cond(resample, array, axis=-1):
  ''' Assuming array of responses [X,Y,...,Z], resample accordingly
      NOTE: We assume we resample at axis = I, then all axes>I are yoked
      NOTE: This function modifies the input array -- ensure that this is what we want (i.e. copy array beforehand)
  '''
  
  ######
  ## Yes, the code here is a bit obtuse - however, there are a few tricks which are as follows:
  ## - 1. We use itertools.product on *[range(x) for x in shape], which translates to itertools.product(range(shape[0]), ..., range(shape[axis]))
  ## ------ i.e. we can iterate over an arbitrary number of dimensions, all of which are BEFORE the specified axis on which we resample
  ## - 2. non_nan = ....np.atleast_2d()
  ## ------ to account for the possibility that axis is NOT the last axis, we want to make sure that we get the # of non_nan entries along that axis, and not any
  ## ------ further axes (i.e. if axis=3, we don't want to know how many non-nan are in axis=4)
  ## ------ By expanding to at least 2d, we assure that (..., axis=0) works even if axis=-1
  ## - 3. preSamp_dim + (X, ) is making into a tuple what we normally pass in as an array [X] to allow the flexibility of preSamp_dim to remain as a tuple
  ######

  if resample:
    preSamp_shape = array.shape[0:axis];
    # This line creates a loop where we iterate through each dimension in the array before "axis"
    # -- i.e. if array is [4, 3, 2], we will go from [0,0,:] to [3,2,:]
    for preSamp_dim in itertools.product(*[range(x) for x in preSamp_shape]):
      non_nan = np.where(~np.any(np.isnan(np.atleast_2d(array[preSamp_dim])), axis=0))[0];
      new_inds = np.random.choice(non_nan, len(non_nan));
      array[preSamp_dim + (range(len(non_nan)),)] = array[preSamp_dim + (new_inds,)]

    return array;
  else:
    return array;

### ORGANIZING RESPONSES ###

def get_baseOnly_resp(expInfo, dc_resp=None, f1_base=None, val_trials=None, vecCorrectedF1=1, onsetTransient=None, F1useSem=True):
  ''' returns the distribution of responses, mean/s.e.m. and unique sfXcon for each base stimulus in the sfBB_* series
      -- dc_resp; f1_base --> use to overwrite the spikes in expInfo (e.g. model responses)
      ---- if passing in the above, should also include val_trials (list of valid trial indices), since
           the model is not evaulated for all trials
      -- vecCorrectedF1 --> Rather than averaging amplitudes, take vector mean/addition (correcting for responses with more varying phase, i.e. noise!)
      -- if vecCorrectedF1 is on, and if onsetTransient is not None, we'll use the onset transient correction for compute F1 amplitudes
  '''

  # TODO: Replace the "val_trials" stuff here with get_valid_trials() func

  if dc_resp is not None:
    dc_resp = dc_resp;
  else:
    dc_resp = expInfo['spikeCounts']

  if f1_base is not None:
    f1_base = f1_base;
  else:
    f1_base = expInfo['f1_base'];
  if val_trials is None: # otherwise, we've defined which trials matter
    val_trials = np.arange(expInfo['trial']['con'].shape[1]); # i.e. how many trials

  # now, check if val_trials is the same length as dc_resp and/or f1_base - if so, we'll need to handle the indexing differently
  # - more ternary operator
  dc_len_match = 1 if len(dc_resp) == len(val_trials) else 0;
  f1_len_match = 1 if len(f1_base) == len(val_trials) else 0;
  
  byTrial = expInfo['trial'];

  baseOnlyTr = np.logical_and(byTrial['baseOn'], ~byTrial['maskOn']) # baseON and maskOFF
  baseSf_all, baseCon_all = byTrial['sf'][1, baseOnlyTr], byTrial['con'][1, baseOnlyTr]

  sf_con_pairs = np.stack((baseSf_all, baseCon_all), axis=1)
  unique_pairs = np.unique(sf_con_pairs, axis=0);

  baseResp_dc = []; baseResp_f1 = [];
  baseSummary_dc = np.zeros((len(unique_pairs), 2)); 
  if vecCorrectedF1: # [i, 0, :] is means [r,phi]; [i, 1, :] is std/var for [r,phi]
    baseSummary_f1 = np.zeros((len(unique_pairs), 2, 2));
  else:
    baseSummary_f1 = np.zeros((len(unique_pairs), 2));

  for ii, up in enumerate(unique_pairs):

      # we have the unique pairs, now cycle through and do the same thing here we did with the other base stimulus....
      baseSf_curr, baseCon_curr = up;
      base_match = np.logical_and(byTrial['sf'][1,:]==baseSf_curr,
                                           byTrial['con'][1,:]==baseCon_curr);

      # NOTE: We do all indexing in logicals until the end, where we align it into the length of val_trials
      # - unfortunately, need to do different indexing depending on if len(responses) = or not to len(val_trials)...
      baseOnly_curr = np.intersect1d(np.where(np.logical_and(baseOnlyTr, base_match))[0], val_trials);
      baseOnly_curr_match = np.where(np.logical_and(baseOnlyTr, base_match)[val_trials])[0];
      # - ternary operator (should be straightforward)
      dc_indexing = baseOnly_curr_match if dc_len_match else baseOnly_curr;
      f1_indexing = baseOnly_curr_match if f1_len_match else baseOnly_curr;
      # - then get the responses...
      baseDC, baseF1 = dc_resp[dc_indexing], f1_base[f1_indexing];

      if vecCorrectedF1: # overwrite baseF1 from above...
        vec_means, vec_byTrial, _, _, _, _ = get_vec_avg_response(expInfo, baseOnly_curr, onsetTransient=onsetTransient, useSem=F1useSem);
        _, baseInd = get_mask_base_inds(); # we know we're getting base response...
        try:
          # NOTE: If vecCorrectedF1, we return baseResp_f1 --> length 2, [0] is R, [1] is phi in polar coordinates
          baseResp_f1.append(np.vstack((vec_byTrial[0][:, baseInd], vec_byTrial[1][:, baseInd])));
          # NOTE: If vecCorrectedF1, we return means [r, phi] in [ii, 0];
          # -------------------------------- sem/var [r, phi] in [ii, 1];
          baseSummary_f1[ii, 0, :] = [vec_means[0][baseInd], vec_means[1][baseInd]];
          baseSummary_f1[ii, 1, :] = [vec_means[2][baseInd], vec_means[3][baseInd]];
        except: # why? If there were not valid trials for this condition...
          baseResp_f1.append(np.vstack(([], [])));
          baseSummary_f1[ii, 0, :] = [np.nan, np.nan]
          baseSummary_f1[ii, 1, :] = [np.nan, np.nan]
      else:
        baseResp_f1.append(baseF1);
        baseSummary_f1[ii, :] = [np.mean(baseF1), sem(baseF1)];

      baseResp_dc.append(baseDC); 
      baseSummary_dc[ii, :] = [np.mean(baseDC), sem(baseDC)];

  return [baseResp_dc, baseResp_f1], [baseSummary_dc, baseSummary_f1], unique_pairs;

def get_mask_resp(expInfo, withBase=0, maskF1 = 1, returnByTr=0, dc_resp=None, f1_base=None, f1_mask=None, val_trials=None, vecCorrectedF1=1, onsetTransient=None, resample=False, phAdvCorr=True, opt_params=None, debugPhAdv=False, F1useSem=True):
  ''' return the DC, F1 matrices [mean, s.e.m.] for responses to the mask only in the sfBB_* series 
      For programs (e.g. sfBB_varSF) with multiple base conditions, the order returned here is guaranteed
      to be the same as the unique base conditions given in get_baseOnly_resp
      -- dc_resp; f1_base/mask --> use to overwrite the spikes in expInfo (e.g. model responses)
      ---- if passing in the above, should also include val_trials (list of valid trial indices), since
           the model is not evaulated for all trials
      -- vecCorrectedF1 --> Rather than averaging amplitudes, take vector mean/addition (correcting for responses with more varying phase, i.e. noise!)

  '''

  if dc_resp is not None:
    dc_resp = dc_resp;
  else:
    dc_resp = expInfo['spikeCounts']
  if f1_base is not None:
    f1_base = f1_base;
  else:
    f1_base = expInfo['f1_base'];
  if f1_mask is not None:
    f1_mask = f1_mask;
  else:
    f1_mask = expInfo['f1_mask'];
  if val_trials is None: # otherwise, we've defined which trials matter
    val_trials = np.arange(expInfo['trial']['con'].shape[1]); # i.e. how many trials

  # now, check if val_trials is the same length as dc_resp and/or f1_base/mask - if so, we'll need to handle the indexing differently
  # - more ternary operator
  dc_len_match = 1 if len(dc_resp) == len(val_trials) else 0;
  f1Base_len_match = 1 if len(f1_base) == len(val_trials) else 0;
  f1Mask_len_match = 1 if len(f1_mask) == len(val_trials) else 0;

  maxTr = 20; # we assume that the max # of trials in any condition will be this value
  conDig = 3; # round contrast to nearest thousandth (i.e. 0.001)
  byTrial = expInfo['trial'];

  maskCon, maskSf = expInfo['maskCon'], expInfo['maskSF'];
  # if we want with the base, we'll have to consider how many unique base conditions there are

  respsDC = []; respsF1 = [];

  if withBase == 0:
    # first, the logical which gives mask-only trials
    baseMatch = np.logical_and(byTrial['maskOn'], ~byTrial['baseOn']);
    nBase = 1;
  elif withBase == 1:
    # first, the logical which gives mask+base trials
    baseMatch = np.logical_and(byTrial['maskOn'], byTrial['baseOn']);
    _, _, baseConds = get_baseOnly_resp(expInfo, vecCorrectedF1=vecCorrectedF1);
    nBase = len(baseConds);

  maskResp_dc = []; maskResp_f1 = [];
  maskResp_dcAll = []; maskResp_f1All = [];

  for up in np.arange(nBase):

    # make a 3d matrix of base+mask responses - SF x CON x [mean, SEM]
    maskCon, maskSf = np.unique(np.round(expInfo['maskCon'], conDig)), expInfo['maskSF'];
    respMatrixDC = np.nan * np.zeros((len(maskCon), len(maskSf), 2));
    respMatrixDCall = np.nan * np.zeros((len(maskCon), len(maskSf), maxTr));
    if vecCorrectedF1: # we add an extra dimension to account for (r, phi) rather than just amplitude (sqrt(r^2 + phi^2))
      respMatrixF1 = np.nan * np.zeros((len(maskCon), len(maskSf), 2, 2));
      respMatrixF1all = np.nan * np.zeros((len(maskCon), len(maskSf), maxTr, 2));
    else:
      respMatrixF1 = np.nan * np.zeros((len(maskCon), len(maskSf), 2));
      respMatrixF1all = np.nan * np.zeros((len(maskCon), len(maskSf), maxTr));

    if withBase == 1: # then subset based on the particular base condition
      # we have the unique pairs, now cycle through and do the same thing here we did with the other base stimulus....
      baseSf_curr, baseCon_curr = baseConds[up];
      currTr = np.logical_and(baseMatch, np.logical_and(byTrial['sf'][1,:]==baseSf_curr,
                                                                             byTrial['con'][1,:]==baseCon_curr));
    else:
      currTr = baseMatch;

    for mcI, mC in enumerate(maskCon):
        conOk = (np.round(byTrial['con'][0,:], conDig) == mC)
        for msI, mS in enumerate(maskSf):
            sfOk = (byTrial['sf'][0,:] == mS)
            # NOTE: We do all indexing in logicals until the end, where we align it into the length of val_trials
            # - unfortunately, need to do different indexing depending on if len(responses) = or not to len(val_trials)...
            trialsOk = np.intersect1d(np.where(np.logical_and(currTr, np.logical_and(conOk, sfOk)))[0], val_trials);
            trialsOk_match = np.where(np.logical_and(currTr, np.logical_and(conOk, sfOk))[val_trials])[0];

            # - ternary operator (should be straightforward)
            dc_indexing = trialsOk_match if dc_len_match else trialsOk;
            f1Mask_indexing = trialsOk_match if f1Mask_len_match else trialsOk;
            f1Base_indexing = trialsOk_match if f1Base_len_match else trialsOk;
            ### then get the responses...DC, first
            currDC = dc_resp[dc_indexing];
            nTr = len(currDC);
            respMatrixDCall[mcI, msI, 0:nTr] = currDC;
            dcMean = np.mean(currDC);
            respMatrixDC[mcI, msI, :] = [dcMean, sem(currDC)]

            if vecCorrectedF1:
              maskInd, baseInd = get_mask_base_inds();
              if maskF1 == 1:
                whichInd = maskInd;
                whichTrials = f1Mask_indexing;
              else:
                whichInd = baseInd;
                whichTrials = f1Base_indexing;
              vec_means, vec_byTrial, _, _, _, _ = get_vec_avg_response(expInfo, whichTrials, onsetTransient=onsetTransient, useSem=F1useSem);
              try:
                # NOTE: vec_byTrial is list: R [0] and phi [1] (relative to stim onset)
                respMatrixF1all[mcI, msI, 0:nTr, 0] = vec_byTrial[0][:, whichInd];
                respMatrixF1all[mcI, msI, 0:nTr, 1] = vec_byTrial[1][:, whichInd];
                # NOTE: If vecCorrectedF1, we return r [mean, sem] in [..., 0, :];
                # ---------------------------------phi [mean, circVar] in [..., 1, :];
                respMatrixF1[mcI, msI, 0, :] = [vec_means[0][whichInd], vec_means[2][whichInd]]
                respMatrixF1[mcI, msI, 1, :] = [vec_means[1][whichInd], vec_means[3][whichInd]];
              except: # we'll end up here if the condition we're analyzing in a "subset of data" case has NO valid trials
                pass; # we're already pre-populated with NaN
            else: # i.e. for model responses, in particular
              if maskF1 == 1:
                currF1 = f1_mask[f1Mask_indexing];
              else:
                currF1 = f1_base[f1Base_indexing];
              respMatrixF1all[mcI, msI, 0:nTr] = currF1;
              respMatrixF1[mcI, msI, :] = [np.mean(currF1), sem(currF1)]

    # Now, here (after organizing all of the responses by con x sf), we can apply any vecF1 correction, if applicable
    if phAdvCorr and vecCorrectedF1:
      if opt_params is None: # otherwise, we can pass in the values in advance
        opt_params, phAdv_model = phase_advance_fit_core(respMatrixF1[:,:,0], respMatrixF1[:,:,1], maskCon, maskSf);
        
        if debugPhAdv:
          return opt_params;
      else:
        phAdv_model = hf.get_phAdv_model();
      for msI, mS in enumerate(maskSf):
        curr_params = opt_params[msI]; # the phAdv model applies per-SF
        for mcI, mC in enumerate(maskCon):
          curr_r, curr_phi = respMatrixF1[mcI, msI, :, 0]
          refPhi = phAdv_model(*curr_params, curr_r);
          new_r = np.multiply(curr_r, np.cos(np.deg2rad(refPhi)-np.deg2rad(curr_phi)));
          respMatrixF1[mcI, msI,0,0] = new_r;

    maskResp_dc.append(respMatrixDC); maskResp_f1.append(respMatrixF1);
    maskResp_dcAll.append(respMatrixDCall); maskResp_f1All.append(respMatrixF1all);

  if returnByTr == 0:
    if nBase == 1:
      return maskResp_dc[0], maskResp_f1[0];
    else:
      return maskResp_dc, maskResp_f1;
  elif returnByTr == 1:
    if nBase == 1:
      return maskResp_dc[0], maskResp_f1[0], maskResp_dcAll[0], maskResp_f1All[0];
    else:
      return maskResp_dc, maskResp_f1, maskResp_dcAll, maskResp_f1All;

def adjust_f1_byTrial(expInfo, onsetTransient=None, maskF1byPhAmp=None):
  ''' Correct the F1 ampltiudes for each trial (in order) by:
      - Projecting the full, i.e. (r, phi) FT vector onto the (vector) mean phase
        across all trials of a given condition
      NOTE: As of 22.07.17, only used in model_responses_pytorch (since descr_fits done on condition averages, not trials)
      Return: adjMask, adjBase (each an (nTr, ) vector)
  '''
  dir = -1;
  stimDur = 1
  conDig = 2
  maskInd, baseInd = get_mask_base_inds();

  # Gather all possible stimulus conditions
  maskSfs = expInfo['maskSF'];
  maskCons = np.round(expInfo['maskCon'], conDig);
  # TODO: This assumes only one base condition (i.e. sfBB_core); adapt for sfBB_var*
  baseSf = expInfo['baseSF']; 
  baseCon = np.round(expInfo['baseCon'], conDig);

  updMask = np.nan * np.zeros_like(expInfo['f1_mask']);
  updBase = np.nan * np.zeros_like(expInfo['f1_base']);

  for maskOn in range(2):
    for baseOn in range(2):
      if (maskOn + baseOn) == 0: # these are just blank trials...
        continue;

      if maskOn == 1: # if the mask is on, we'll need to specify which trials
        allSfs = maskSfs;
        allCons = maskCons;
      else:
        allSfs = [0]; allCons = [0]; # the values won't matter, anyway, so just pass in [0], [0]
      for whichSf, maskSf in enumerate(allSfs):
        for whichCon, maskCon in enumerate(allCons):
          # Choose which condition to consider
          val_trials, stimPh, stimTf = get_valid_trials(expInfo, maskOn, baseOn, whichCon, whichSf, returnStimConds=1);
          # Then get the vec avg'd responses, phase information, etc
          vec_avgs, vec_byTrial, rel_amps, _, _, _ = get_vec_avg_response(expInfo, val_trials, dir=dir, stimDur=stimDur, onsetTransient=onsetTransient);
          # - unpack the vec_avgs, and per trial
          mean_r, mean_phi = vec_avgs[0], vec_avgs[1];
          resp_amp, phase_rel_stim = vec_byTrial[0], vec_byTrial[1];
          # finally, project the response as usual
          resp_proj = np.multiply(resp_amp, np.cos(np.deg2rad(mean_phi)-np.deg2rad(phase_rel_stim)));
          if maskF1byPhAmp is not None and maskOn==1 and baseOn==0: # this only applies with no base stimulus!
            updMask[val_trials] = maskF1byPhAmp[whichCon, whichSf, 0:len(val_trials), 0]; # just unpack from the already processed maskF1 responses (corrected by phAmp) --> must be by trial 
          else:
            updMask[val_trials] = resp_proj[:,maskInd];
          updBase[val_trials] = resp_proj[:,baseInd];

  return updMask, updBase;

def phase_advance_fit_core(allAmps, allPhis, maskCons, maskSfs):
  ''' For the Bauman+Bonds experiment, we can only make the phase-amplitude adjustment for mask F1 responses
      - So, this function will take the F1 responses directly [con X sf] and fit the phAmp relationship per SF
  '''

  # allAmp will be [nSf] list of responses, per contrast, each [mean, std]
  # allPhi will be [nSf] ... each [mean, var]
  allAmp = []; allPhi = []; allCons = []; allPhiVar = [];
  for i_sf, _ in enumerate(maskSfs):
    curr_amp = []; curr_phi = []; curr_tf = []; curr_phiVar = [];
    for i_con, con_val in enumerate(maskCons):
      curr_amp.append([allAmps[i_con, i_sf, 0], allAmps[i_con, i_sf, 1]]);
      curr_phi.append([allPhis[i_con, i_sf, 0], allPhis[i_con, i_sf, 1]]);
      curr_phiVar.append([allPhis[i_con, i_sf, 1]]);
    allAmp.append(curr_amp);
    allPhi.append(curr_phi);
    allCons.append(maskCons);
    allPhiVar.append(curr_phiVar);
  phAdv_model, all_opts, all_phAdv, all_loss = hf.phase_advance(allAmp, allPhi, allCons, tfs=None, phiVar=allPhiVar);

  return all_opts, phAdv_model;

### ANALYSIS ###

def get_vec_avg_response(expInfo, val_trials, dir=1, psth_binWidth=1e-3, stimDur=1, onsetTransient=None, refPhi=None, useSem=True):
  ''' Return [r_mean, phi_mean, r_var, phi_var], [resp_amp, phase_rel_stim], rel_amps, phase_rel_stim, stimPhs, resp_phase
      -- Note that the above values are at mask/base TF, respectively (i.e. not DC)
      ---- r_var is s.e.m. if useSem, otherwise std

      If onsetTransient is not None, we'll do the manual FFT

      Given the data and the set of valid trials, first compute the response phase
      and stimulus phase - then determine the response phase relative to the stimulus phase
  '''
  # organize the spikes
  msTenthToS = 1e-4; # the spike times are in 1/10th ms, so multiply by 1e-4 to convert to S
  spikeTimes = [expInfo['spikeTimes'][trNum]*msTenthToS for trNum in val_trials]

  # -- get stimulus info
  byTrial = expInfo['trial'];
  maskInd, baseInd = get_mask_base_inds();
  baseTf, maskTf = byTrial['tf'][baseInd, val_trials], byTrial['tf'][maskInd, val_trials]
  basePh, maskPh = byTrial['ph'][baseInd, val_trials], byTrial['ph'][maskInd, val_trials]
  baseSf, maskSf = byTrial['sf'][baseInd, val_trials], byTrial['sf'][maskInd, val_trials]
  baseCon, maskCon = byTrial['con'][baseInd, val_trials], byTrial['con'][maskInd, val_trials]

  # -- compute PSTH
  psth, bins = hf.make_psth(spikeTimes, stimDur=stimDur)
  tfAsInts = np.zeros((len(val_trials),2), dtype='int32')
  tfAsInts[:, maskInd] = maskTf;
  tfAsInts[:, baseInd] = baseTf;
  if onsetTransient is None: # then just do this normally...
    amps, rel_amps, full_fourier = hf.spike_fft(psth, tfs = tfAsInts, stimDur = stimDur)
    # get the phase of the response
    resp_phase = np.array([np.angle(full_fourier[x][tfAsInts[x, :]], True) for x in range(len(full_fourier))]); # true --> in degrees
    resp_amp = np.array([amps[tfAsInts[ind, :]] for ind, amps in enumerate(amps)]); # after correction of amps (19.08.06)
  else:
    man_fft = [hf.manual_fft(psth_curr, tfs=np.array([int(np.unique(maskTf)), int(np.unique(baseTf))]), onsetTransient=onsetTransient, stimDur=stimDur) for psth_curr in psth]
    # -- why [2]? That's the full spectrum; then [1:] is to get mask,base resp_phases respectively // [0] is DC
    resp_phase = np.squeeze(np.array([np.angle(curr_fft[2][1:], True) for curr_fft in man_fft])); # true --> in degrees
    # -- why [3]? That's the array of amplitudes; then [1:] is to get mask,base respectively // [0] is DC
    resp_amp = np.squeeze(np.array([curr_fft[3][1:] for curr_fft in man_fft])); # true --> in degrees
    rel_amps = np.copy(resp_amp); # they're the same...

  stimPhs = np.zeros_like(resp_amp);
  try:
    stimPhs[:, maskInd] = maskPh
    stimPhs[:, baseInd] = basePh
  except: # why would this happen? If we passed in val_trials for "subset of the data analysis" and this condition is empty...
    pass;

  phase_rel_stim = np.mod(np.multiply(dir, np.add(resp_phase, stimPhs)), 360);
  r_mean, phi_mean, r_var, phi_var = hf.polar_vec_mean(np.transpose(resp_amp), np.transpose(phase_rel_stim), sem=useSem) # return s.e.m. rather than std (default)
  if refPhi is not None:
    # then do the phase projection here! TODO: CHECK IF PHI_MEAN/refPhi in deg or rad
    r_mean = np.multiply(r_mean, np.cos(np.deg2rad(refPhi)-np.deg2rad(phi_mean)));

  return [r_mean, phi_mean, r_var, phi_var], [resp_amp, phase_rel_stim], rel_amps, phase_rel_stim, stimPhs, resp_phase;

def compute_f1f0(trial_inf, vecCorrectedF1=1):
  ''' Using the stimulus closest to optimal in terms of SF (at high contrast), get the F1/F0 ratio
      This will be used to determine simple versus complex
  '''
  stimDur = 1; # for now (latest check as of 21.05.10), all sfBB_* experiments have a 1 second stim dur

  ######
  # why are we keeping the trials with max response at F0 (always) and F1 (if present)? Per discussion with Tony, 
  # we should evaluate F1/F0 at the SF  which has the highest response as determined by comparing F0 and F1, 
  # i.e. F1 might be greater than F0 AND have a different than F0 - in the case, we ought to evalaute at the peak F1 frequency
  ######
  ## first, get F0 responses (mask only)
  f0_counts, f1_rates, f0_all, f1_rates_all = get_mask_resp(trial_inf, withBase=0, maskF1=1, returnByTr=1, vecCorrectedF1=vecCorrectedF1);
  f0_blank = trial_inf['blank']['mean'];
  f0_rates = np.divide(f0_counts - f0_blank, stimDur);
  f0_rates_all = np.divide(f0_all - f0_blank, stimDur);
  if vecCorrectedF1:
    f1_rates = f1_rates[..., 0]; # throw away the phase information
    f1_rates_all = f1_rates_all[..., 0]; # throw away the phase information
    
  # get prefSfEst
  all_rates = [f0_rates, f1_rates]
  prefSfEst_ind = np.array([np.argmax(resps[-1, :, 0]) for resps in all_rates]); # get peak resp for f0 and f1
  all_sfs = trial_inf['maskSF'];
  prefSfEst = all_sfs[prefSfEst_ind];

  ######
  # make the comparisons 
  ######
  f0f1_highCon = [f0_rates[-1,:,0], f1_rates[-1,:,0]];
  f0f1_highConAll = [f0_rates_all[-1,:,:], f1_rates_all[-1,:,:]];
  # determine which of the peakInd X F0/F1 combinations has the highest overall response
  peakRespInd = np.argmax([np.nanmean(x[y]) for x,y in zip(f0f1_highCon, prefSfEst_ind)]);

  # added/changed 23.01.23 --> compare apples to apples (best dcSF, best f1SF)
  #f0rate, f1rate = [x[ind] for x,ind in zip(f0f1_highCon, prefSfEst_ind)];
  #f0all, f1all = [x[ind, :] for x,ind in zip(f0f1_highConAll, prefSfEst_ind)];
  # --- previous method: compare at max response SF (i.e. dcMax or f1Max, depending)
  # then get the true peak ind
  indToAnalyze = prefSfEst_ind[peakRespInd];
  f0rate, f1rate = [x[indToAnalyze] for x in f0f1_highCon];
  f0all, f1all = [x[indToAnalyze, :] for x in f0f1_highConAll];
  # note: the below lines will help us avoid including trials for which the f0 is negative (after baseline subtraction!)
  # i.e. we will not include trials with below-baseline f0 responses in our f1f0 calculation
  val_inds = np.logical_and(~np.isnan(f0all), ~np.isnan(f1all));
  f0rate_posInd = np.where(np.logical_and(val_inds, f0all>0))[0];
  f0rate_pos = f0all[f0rate_posInd];
  f1rate_pos = f1all[f0rate_posInd];
  # NOTE: 23.01.06 --> CHANGED to mean(f1)/mean(f0) rather than mean(f1/f0)...
  return np.nanmean(f1rate_pos)/np.nanmean(f0rate_pos), f0all, f1all, f0_rates, f1_rates;
  #return np.nanmean(np.divide(f1rate_pos, f0rate_pos)), f0all, f1all, f0_rates, f1_rates;

#######################
#### More analysis
#######################
def get_all_responses(cellNum, data_loc, dataList, descrFits=None, fitBase=None, vecCorrected=1, expName='sfBB_core', f1_r_std_on_r=True):
  ''' Major helper function which will enable many analyses!
      Creates dictionary which we return!
      - if descrFits is not None, then we include information based on the descr. (not full comp. model) fit!
      ---- as of 23.01.22, descrFits is 'descrFitsHPC_221126vEs_phAdj_sqrt_ddogs_JTflankShiftCopyCtrRaSlope.npy'
      ---- BUT, we should pass it in preemptively rather than loading it here (will be used with mp.pool)

      NOTES:
      --- resps organized as [CON x SF x [mn x sem]]
      --- if f1_r_std_on_r, then rather than computing the vector variance, compute only the var/std on the resp magnitudes
  '''

  expSummary = dict(); # split by DC, F1 for resp. measures

  unitNm = dataList['unitName'][cellNum-1];
  try:
    curr_descrFits = descrFits[cellNum-1];
  except:
    curr_descrFits = None
  cell = hf.np_smart_load('%s%s_sfBB.npy' % (data_loc, unitNm));
  expInfo = cell[expName]
  byTrial = expInfo['trial'];
  f1f0_rat = compute_f1f0(expInfo)[0];

  expSummary['f1f0_ratio'] = f1f0_rat;

  ### Get the responses - base only, mask+base [base F1], mask only (mask F1)
  baseDistrs, baseSummary, baseConds = get_baseOnly_resp(expInfo);
  # - unpack DC, F1 distribution of responses per trial
  baseDC, baseF1 = baseDistrs;
  baseDC_mn, baseF1_mn = np.mean(baseDC), np.mean(baseF1);
  if vecCorrected:
      baseDistrs, baseSummary, _ = get_baseOnly_resp(expInfo, vecCorrectedF1=1, F1useSem=False);
      baseF1_mn = baseSummary[1][0][0,:]; # [1][0][0,:] is r,phi mean
      baseF1_var = baseSummary[1][0][1,:]; # [1][0][0,:] is r,phi std/(circ.) var
      baseF1_r, baseF1_phi = baseDistrs[1][0][0], baseDistrs[1][0][1];
  # - unpack the SF x CON of the base (guaranteed to have only one set for sfBB_core)
  baseSf_curr, baseCon_curr = baseConds[0];
  expSummary['baseSf'] = baseSf_curr;
  expSummary['baseCon'] = baseCon_curr;

  # now get the mask+base response (f1 at base TF)
  respMatrixDC, respMatrixF1 = get_mask_resp(expInfo, withBase=1, maskF1=0, vecCorrectedF1=vecCorrected); # i.e. get the base response for F1
  # and get the mask only response (f1 at mask TF)
  respMatrixDC_onlyMask, respMatrixF1_onlyMask = get_mask_resp(expInfo, withBase=0, maskF1=1, vecCorrectedF1=vecCorrected); # i.e. get the maskONLY response
  # and get the mask+base response (but f1 at mask TF)
  _, respMatrixF1_maskTf = get_mask_resp(expInfo, withBase=1, maskF1=1, vecCorrectedF1=vecCorrected); # i.e. get the maskONLY response

  # -- if vecCorrected, let's just take the "r" elements, not the phi information
  if vecCorrected:
      respMatrixF1 = respMatrixF1[:,:,0,:]; # just take the "r" information (throw away the phi)
      respMatrixF1_onlyMask = respMatrixF1_onlyMask[:,:,0,:]; # just take the "r" information (throw away the phi)
      respMatrixF1_maskTf = respMatrixF1_maskTf[:,:,0,:]; # just take the "r" information (throw away the phi)

  ## Reference tuning...
  refDC, refF1 = get_mask_resp(expInfo, withBase=0, vecCorrectedF1=vecCorrected); # i.e. mask only, at mask TF
  maskSf, maskCon = expInfo['maskSF'], expInfo['maskCon'];
  expSummary['maskSf'] = maskSf;
  expSummary['maskCon'] = maskCon;
  # - get DC tuning curves
  refDC_sf = refDC[-1, :, :]; # highest contrast
  prefSf_ind = np.argmax(refDC_sf[:, 0]);
  prefSf_DC = maskSf[prefSf_ind];
  refDC_rvc = refDC[:, prefSf_ind, :];
  # - get F1 tuning curves (adjust for vecCorrected?)
  if vecCorrected: # get only r, not phi
      refF1 = refF1[:,:,0,:];
  refF1_sf = refF1[-1, :, :];
  prefSf_ind = np.argmax(refF1_sf[:, 0]);
  prefSf_F1 = maskSf[prefSf_ind];
  refF1_rvc = refF1[:, prefSf_ind, :];

  ######
  for measure in [0,1]:
      if measure == 0:
          dc_dict = dict();
          dc_dict['blank_mean'] = expInfo['blank']['mean'];
          dc_dict['blank_std'] = expInfo['blank']['std'];
          dc_dict['blank_stderr'] = expInfo['blank']['stderr'];
          dc_dict['bothResp'] = respMatrixDC;
          dc_dict['maskResp'] = respMatrixDC_onlyMask;
          if len(baseDC)==1: # i.e. just one base (not var* experiment)
            baseDC = baseDC[0];
          dc_dict['baseResp_all'] = baseDC
          dc_dict['baseResp_mean'] = np.nanmean(baseDC)
          dc_dict['baseResp_std'] = np.nanstd(baseDC)
          # fano: var/mean (and var=square(std))
          dc_dict['baseResp_fano'] = np.square(dc_dict['baseResp_std'])/dc_dict['baseResp_mean']
          dc_dict['baseResp_stderr'] = sem(hf.nan_rm(baseDC))
          # --- also compute the difference from expected response
          # ----- i.e. R(m+b) - R(m) - R(b)
          dc_dict['diffFromSumResp'] = respMatrixDC[:,:,0] - respMatrixDC_onlyMask[:,:,0] - dc_dict['baseResp_mean']
          # --- and, more importantly --> deviations from base response when mask+base are present
          baseDiffs = dc_dict['bothResp'][:,:,0] - dc_dict['baseResp_mean']
          baseDiffs_zscr = baseDiffs/dc_dict['baseResp_std']; # in z-scored units
          baseDiffs_norm = baseDiffs/dc_dict['baseResp_mean']; # difference divided by response measure
          dc_dict['baseDiffs'] = np.stack((baseDiffs, baseDiffs_zscr, baseDiffs_norm), axis=-1); # [con X sf x [raw, zscr]]
          #refAll = refDC[:,:,0];
          #refSf = refDC_sf;
          #refRVC = refDC_rvc;
          #refSf_pref = prefSf_DC;
          dc_dict['sfPref_est'] = prefSf_DC;
          # NOTE: the descrFits info is duplicated in jointList --> perhaps deprecate here?
          if 'dc' in curr_descrFits:
            dc_dict['sfPref'] = curr_descrFits['dc']['mask']['prefSf'][-1]; # high con.
            dc_dict['charFreq'] = curr_descrFits['dc']['mask']['charFreq'][-1]; # high con.
            dc_dict['sfVarExpl'] = curr_descrFits['dc']['mask']['varExpl'][-1]; # high con.
          ### ignored for now...
          if fitBase is not None:
            modelsAsObj = [mod_A_dc, mod_B_dc]
            data_A = respMatrix_A_dc;
            data_B = respMatrix_B_dc;
            data_A_onlyMask = respMatrix_A_dc_onlyMask;
            data_B_onlyMask = respMatrix_B_dc_onlyMask;
            data_A_baseTf = None;
            data_B_baseTf = None;
            mod_mean_A = baseMean_mod_dc[0];
            mod_mean_B = baseMean_mod_dc[1];
      elif measure == 1:
          f1_dict = dict();
          f1_dict['bothResp_maskTf'] = respMatrixF1_maskTf;
          f1_dict['bothResp_baseTf'] = respMatrixF1;
          f1_dict['maskResp'] = respMatrixF1_onlyMask;
          if vecCorrected:
              mean_r, mean_phi = baseF1_mn;
              std_r, var_phi = baseF1_var;
              vec_r, vec_phi = baseF1_r, baseF1_phi;
              f1_dict['baseResp_mean'] = mean_r
              f1_dict['baseResp_std'] = std_r if not f1_r_std_on_r else np.nanstd(baseDistrs[1][0][0]); # just the r values
              f1_dict['baseResp_circVar'] = var_phi
              # fano: var/mean (and var=square(std))
              f1_dict['baseResp_fano'] = np.square(std_r)/mean_r;
          else: # should be unused...
              f1_dict['baseResp_mean'] = baseF1_mn
              f1_dict['baseResp_var'] = baseF1_var
          #refAll = refF1[:,:,0];
          #refSf = refF1_sf;
          #refRVC = refF1_rvc;
          f1_dict['sfPref_est'] = prefSf_F1
          # NOTE: the descrFits info is duplicated in jointList --> perhaps deprecate here?
          if 'f1' in curr_descrFits:
            f1_dict['sfPref'] = curr_descrFits['f1']['mask']['prefSf'][-1]; # high con.
            f1_dict['charFreq'] = curr_descrFits['f1']['mask']['charFreq'][-1]; # high con.
            f1_dict['sfVarExpl'] = curr_descrFits['f1']['mask']['varExpl'][-1]; # high con.
          # change in mean base response - raw and norm. by baseResp std
          baseDiffs = f1_dict['bothResp_baseTf'][:,:,0] - f1_dict['baseResp_mean']
          baseDiffs_zscr = baseDiffs/f1_dict['baseResp_std']; # in z-scored units
          baseDiffs_norm = baseDiffs/f1_dict['baseResp_mean']; # difference divided by response measure
          f1_dict['baseDiffs'] = np.stack((baseDiffs, baseDiffs_zscr, baseDiffs_norm), axis=-1); # [con X sf x [raw, zscr]]
          if fitBase is not None:
            modelsAsObj = [mod_A_f1, mod_B_f1]
            data_A = respMatrix_A_f1_maskTf;
            data_B = respMatrix_B_f1_maskTf;
            data_A_onlyMask = respMatrix_A_f1_onlyMask;
            data_B_onlyMask = respMatrix_B_f1_onlyMask;
            data_A_baseTf = respMatrix_A_f1;
            data_B_baseTf = respMatrix_B_f1;
            mod_mean_A = baseMean_mod_f1[0][0];
            mod_mean_B = baseMean_mod_f1[1][0];

  expSummary['dc'] = dc_dict;
  expSummary['f1'] = f1_dict;

  return expSummary;
