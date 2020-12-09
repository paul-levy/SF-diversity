import numpy as np
import helper_fcns as hf

import pdb

### Similar to helper_fcns, but meant specifically for the sfBB_* series of experiments

### Helper/Basic ###
# get_resp_str - return 'dc' or 'f1' depending in which response measure we're using
# get_mask_base_inds - the answer is in the name! 0 index for mask, 1 for base (when array is [...,2]
# get_valid_trials - get the list of valid trials corresponding to a particular stimulus condition

### Organize responses
# get_baseOnly_resp - get the response to the base stimulus ONLY
# get_mask_resp - get the response to the mask OR mask+base at either the base or mask TF
# adjust_f1_byTrial - to the vector math adjustment on each trial

### Anaylsis ###
# --- Computing F1 response (including phase), computing F1::F0 ratio, etc
# get_vec_avg_response
# compute_f1f0

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

### ORGANIZING RESPONSES ###

def get_baseOnly_resp(expInfo, dc_resp=None, f1_base=None, val_trials=None, vecCorrectedF1=0):
  ''' returns the distribution of responses, mean/s.e.m. and unique sfXcon for each base stimulus in the sfBB_* serie
      -- dc_resp; f1_base --> use to overwrite the spikes in expInfo (e.g. model responses)
      ---- if passing in the above, should also include val_trials (list of valid trial indices), since
           the model is not evaulated for all trials
      -- vecCorrectedF1 --> Rather than averaging amplitudes, take vector mean/addition (correcting for responses with more varying phase, i.e. noise!)
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
      baseOnly_curr = np.intersect1d(np.where(np.logical_and(baseOnlyTr, base_match))[0], val_trials);
      #baseOnly_curr = np.where(np.logical_and(baseOnlyTr, base_match)[val_trials])[0];
      baseDC, baseF1 = dc_resp[baseOnly_curr], f1_base[baseOnly_curr];

      if vecCorrectedF1: # overwrite baseF1 from above...
        vec_means, vec_byTrial, _, _, _, _ = get_vec_avg_response(expInfo, baseOnly_curr);
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
        baseSummary_f1[ii, :] = [np.mean(baseF1), np.std(baseF1)/len(baseF1)];

      baseResp_dc.append(baseDC); 
      baseSummary_dc[ii, :] = [np.mean(baseDC), np.std(baseDC)/len(baseDC)];

  return [baseResp_dc, baseResp_f1], [baseSummary_dc, baseSummary_f1], unique_pairs;


def get_mask_resp(expInfo, withBase=0, maskF1 = 1, returnByTr=0, dc_resp=None, f1_base=None, f1_mask=None, val_trials=None, vecCorrectedF1=0):
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
            trialsOk = np.intersect1d(np.where(np.logical_and(currTr, np.logical_and(conOk, sfOk)))[0], val_trials);
            #trialsOk = np.where(np.logical_and(currTr, np.logical_and(conOk, sfOk))[val_trials])[0];

            #if len(val_trials)<200 and len(trialsOk) > 0:
            #  pdb.set_trace();

            currDC = dc_resp[trialsOk];
            nTr = len(currDC);
            respMatrixDCall[mcI, msI, 0:nTr] = currDC;
            dcMean = np.mean(currDC);
            respMatrixDC[mcI, msI, :] = [dcMean, np.std(currDC)/len(currDC)]

            if vecCorrectedF1:
              maskInd, baseInd = get_mask_base_inds();
              if maskF1 == 1:
                whichInd = maskInd;
              else:
                whichInd = baseInd;
              vec_means, vec_byTrial, _, _, _, _ = get_vec_avg_response(expInfo, trialsOk);
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
            else:
              if maskF1 == 1:
                currF1 = f1_mask[trialsOk];
              else:
                currF1 = f1_base[trialsOk];
              respMatrixF1all[mcI, msI, 0:nTr] = currF1;
              respMatrixF1[mcI, msI, :] = [np.mean(currF1), np.std(currF1)/len(currF1)]

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

def adjust_f1_byTrial(expInfo):
  ''' Correct the F1 ampltiudes for each trial (in order) by:
      - Projecting the full, i.e. (r, phi) FT vector onto the (vector) mean phase
        across all trials of a given condition
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
          vec_avgs, vec_byTrial, rel_amps, _, _, _ = get_vec_avg_response(expInfo, val_trials, dir=dir, stimDur=stimDur);
          # - unpack the vec_avgs, and per trial
          mean_r, mean_phi = vec_avgs[0], vec_avgs[1];
          resp_amp, phase_rel_stim = vec_byTrial[0], vec_byTrial[1];
          # finally, project the response as usual
          resp_proj = np.multiply(resp_amp, np.cos(np.deg2rad(mean_phi)-np.deg2rad(phase_rel_stim)));
          updMask[val_trials] = resp_proj[:,maskInd];
          updBase[val_trials] = resp_proj[:,baseInd];

  return updMask, updBase;

### ANALYSIS ###

def get_vec_avg_response(expInfo, val_trials, dir=-1, psth_binWidth=1e-3, stimDur=1):
  ''' Return [r_mean, phi_mean, r_std, phi_var], [resp_amp, phase_rel_stim], rel_amps, phase_rel_stim, stimPhs, resp_phase

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
  amps, rel_amps, full_fourier = hf.spike_fft(psth, tfs = tfAsInts, stimDur = stimDur)

  # get the phase of the response
  resp_phase = np.array([np.angle(full_fourier[x][tfAsInts[x, :]], True) for x in range(len(full_fourier))]); # true --> in degrees
  resp_amp = np.array([amps[tfAsInts[ind, :]] for ind, amps in enumerate(amps)]); # after correction of amps (19.08.06)
  stimPhs = np.zeros_like(resp_amp);
  try:
    stimPhs[:, maskInd] = maskPh
    stimPhs[:, baseInd] = basePh
  except: # why would this happen? If we passed in val_trials for "subset of the data analysis" and this condition is empty...
    pass;

  phase_rel_stim = np.mod(np.multiply(dir, np.add(resp_phase, stimPhs)), 360);
  r_mean, phi_mean, r_sem, phi_var = hf.polar_vec_mean(np.transpose(resp_amp), np.transpose(phase_rel_stim), sem=1) # return s.e.m. rather than std (default)

  return [r_mean, phi_mean, r_sem, phi_var], [resp_amp, phase_rel_stim], rel_amps, phase_rel_stim, stimPhs, resp_phase;

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

