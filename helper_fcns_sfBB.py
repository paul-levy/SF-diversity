import numpy as np
import helper_fcns

import pdb

### Similar to helper_fcns, but meant specifically for the sfBB_* series of experiments

### Organizing ###
# get_resp_str - return 'dc' or 'f1' depending in which response measure we're using
# get_mask_base_inds - the answer is in the name! 0 index for mask, 1 for base (when array is [...,2]
# get_baseOnly_resp - get the response to the base stimulus ONLY
# get_mask_resp - get the response to the mask OR mask+base at either the base or mask TF

### Anaylsis ###
# compute_f1f0

### ORGANIZING ###

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

def get_baseOnly_resp(expInfo, dc_resp=None, f1_base=None, val_trials=None):
  ''' returns the distribution of responses, mean/s.e.m. and unique sfXcon for each base stimulus in the sfBB_* serie
      -- dc_resp; f1_base --> use to overwrite the spikes in expInfo (e.g. model responses)
      ---- if passing in the above, should also include val_trials (list of valid trial indices), since
           the model is not evaulated for all trials
  '''

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

  byTrial = expInfo['trial'];

  baseOnlyTr = np.logical_and(byTrial['baseOn'], ~byTrial['maskOn']) # baseON and maskOFF
  baseSf_all, baseCon_all = byTrial['sf'][1, baseOnlyTr], byTrial['con'][1, baseOnlyTr]

  sf_con_pairs = np.stack((baseSf_all, baseCon_all), axis=1)
  unique_pairs = np.unique(sf_con_pairs, axis=0);

  baseResp_dc = []; baseResp_f1 = [];
  baseSummary_dc = np.zeros((len(unique_pairs), 2)); baseSummary_f1 = np.zeros((len(unique_pairs), 2));

  for ii, up in enumerate(unique_pairs):

      # we have the unique pairs, now cycle through and do the same thing here we did with the other base stimulus....
      baseSf_curr, baseCon_curr = up;
      base_match = np.logical_and(byTrial['sf'][1,:]==baseSf_curr,
                                           byTrial['con'][1,:]==baseCon_curr);

      # NOTE: We do all indexing in logicals until the end, where we align it into the length of val_trials
      baseOnly_curr = np.where(np.logical_and(baseOnlyTr, base_match)[val_trials])[0];
      baseDC, baseF1 = dc_resp[baseOnly_curr], f1_base[baseOnly_curr];

      baseResp_dc.append(baseDC); baseResp_f1.append(baseF1);

      baseSummary_dc[ii, :] = [np.mean(baseDC), np.std(baseDC)/len(baseDC)];
      baseSummary_f1[ii, :] = [np.mean(baseF1), np.std(baseF1)/len(baseF1)];

  return [baseResp_dc, baseResp_f1], [baseSummary_dc, baseSummary_f1], unique_pairs;


def get_mask_resp(expInfo, withBase=0, maskF1 = 1, returnByTr=0, dc_resp=None, f1_base=None, f1_mask=None, val_trials=None):
  ''' return the DC, F1 matrices [mean, s.e.m.] for responses to the mask only in the sfBB_* series 
      For programs (e.g. sfBB_varSF) with multiple base conditions, the order returned here is guaranteed
      to be the same as the unique base conditions given in get_baseOnly_resp
      -- dc_resp; f1_base/mask --> use to overwrite the spikes in expInfo (e.g. model responses)
      ---- if passing in the above, should also include val_trials (list of valid trial indices), since
           the model is not evaulated for all trials
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
    _, _, baseConds = get_baseOnly_resp(expInfo);
    nBase = len(baseConds);

  maskResp_dc = []; maskResp_f1 = [];
  maskResp_dcAll = []; maskResp_f1All = [];

  for up in np.arange(nBase):

    # make a 3d matrix of base+mask responses - SF x CON x [mean, SEM]
    maskCon, maskSf = np.unique(np.round(expInfo['maskCon'], conDig)), expInfo['maskSF'];
    respMatrixDC = np.nan * np.zeros((len(maskCon), len(maskSf), 2));
    respMatrixF1 = np.nan * np.zeros((len(maskCon), len(maskSf), 2));
    respMatrixDCall = np.nan * np.zeros((len(maskCon), len(maskSf), maxTr));
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
            trialsOk = np.where(np.logical_and(currTr, np.logical_and(conOk, sfOk))[val_trials])[0];

            currDC = dc_resp[trialsOk];
            nTr = len(currDC);

            if maskF1 == 1:
              currF1 = f1_mask[trialsOk];
            else:
              currF1 = f1_base[trialsOk];

            respMatrixDCall[mcI, msI, 0:nTr] = currDC;
            respMatrixF1all[mcI, msI, 0:nTr] = currF1;

            dcMean, f1Mean = np.mean(currDC), np.mean(currF1)
            respMatrixDC[mcI, msI, :] = [dcMean, np.std(currDC)/len(currDC)]
            respMatrixF1[mcI, msI, :] = [f1Mean, np.std(currF1)/len(currF1)]

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


### ANALYSIS ###

def compute_f1f0(trial_inf):
  ''' Using the stimulus closest to optimal in terms of SF (at high contrast), get the F1/F0 ratio
      This will be used to determine simple versus complex
  '''
  stimDur = 1; # for now, all sfBB_* experiments have a 1 second stim dur

  ######
  # why are we keeping the trials with max response at F0 (always) and F1 (if present)? Per discussion with Tony, 
  # we should evaluate F1/F0 at the SF  which has the highest response as determined by comparing F0 and F1, 
  # i.e. F1 might be greater than F0 AND have a different than F0 - in the case, we ought to evalaute at the peak F1 frequency
  ######
  ## first, get F0 responses (mask only)
  f0_counts, f1_rates, f0_all, f1_rates_all = get_mask_resp(trial_inf, withBase=0, maskF1=1, returnByTr=1);
  f0_blank = trial_inf['blank']['mean'];
  f0_rates = np.divide(f0_counts - f0_blank, stimDur);
  f0_rates_all = np.divide(f0_all - f0_blank, stimDur);

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
  # then get the true peak ind
  indToAnalyze = prefSfEst_ind[peakRespInd];
 
  f0rate, f1rate = [x[indToAnalyze] for x in f0f1_highCon];
  f0all, f1all = [x[indToAnalyze, :] for x in f0f1_highConAll];
  # note: the below lines will help us avoid including trials for which the f0 is negative (after baseline subtraction!)
  # i.e. we will not include trials with below-baseline f0 responses in our f1f0 calculation
  val_inds = np.logical_and(~np.isnan(f0all), ~np.isnan(f1all));
  f0rate_posInd = np.where(np.logical_and(val_inds, f0rate>0))[0];
  f0rate_pos = f0all[f0rate_posInd];
  f1rate_pos = f1all[f0rate_posInd];

  return np.nanmean(np.divide(f1rate_pos, f0rate_pos)), f0all, f1all, f0_rates, f1_rates;

#def organize_modResp(modParams):

