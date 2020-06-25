import numpy as np
import sys
import helper_fcns as hf
import scipy.optimize as opt
import os
from time import sleep
from scipy.stats import sem, poisson
import warnings
import pdb

basePath = os.getcwd() + '/';
data_suff = 'structures/';

expName = hf.get_datalist(sys.argv[3]); # sys.argv[3] is experiment dir
#expName = 'dataList_glx_mr.npy'
#df_f0 = 'descrFits_200507_sqrt_flex.npy';
#df_f0 = 'descrFits_190503_sach_flex.npy';
df_f0 = None;
#dogName =  'descrFits_191023';
dogName =  'descrFits_200507';
#phAdvName = 'phaseAdvanceFits_200507'
phAdvName = 'phaseAdvanceFits_191023'
rvcName_f0   = 'rvcFits_200507_f0' # .npy will be added later, as will suffix assoc. w/particular RVC model
rvcName_f1   = 'rvcFits_191023' # _pos.npy will be added later, as will suffix assoc. w/particular RVC model
#rvcName_f1   = 'rvcFits_200507' # _pos.npy will be added later, as will suffix assoc. w/particular RVC model
## model recovery???
modelRecov = 0;
if modelRecov == 1:
  normType = 1; # use if modelRecov == 1 :: 1 - flat; 2 - wght; ...
  dogName =  'mr%s_descrFits_190503' % hf.fitType_suffix(normType);
  rvcName = 'mr%s_rvcFits_f0.npy' % hf.fitType_suffix(normType);
else:
  normType = 0;

##########
### TODO:
##########
# - Fix rvc_adjusted_fit to still fit F1 without phase_advance_fit (i.e. for altExp)
# - Redo all of (1) for more general use!

##########
### Table of contents
##########

## phase_advance_fit
## rvc_adjusted_fit  - fit RVC for each SF/disp condition, choosing F0 or F1 given simple/complex determination
## fit_RVC_f0        - forces f0 spikes for RVC

## invalid
## DoG_loss
## fit_descr_DoG

### 1: Recreate Movshon, Kiorpes, Hawken, Cavanaugh '05 figure 6 analyses
''' These plots show response versus contrast data (with model fit) AND
    phase/amplitude plots with a model fit for response phase as a function of response amplitude
   
    First, given the FFT-derived response amplitude and phase, determine the response phase relative
    to the stimulus by taking into account the stimulus phase. 
    Then, make a simple linear model fit (line + constant offset) of the response phase as a function
    of response amplitude.

    The key non-intuitive step in the analysis is as follows: Rather than simply fitting an RVC curve
    with the FFT-derived response amplitudes, we determine the "true"/expected response phase given
    the measured response amplitude. Then, we project the observed vector onto a line which represents
    the "true"/expected response vector (i.e. the response vector with the expected phase, given the amplitude).
    Thus, the "fixed" response amplitude is = measuredAmp * cos(phiM - phiE)
    (phiM/E are the measured and expected response phases, respectively)

    This value will always be <= measuredAmp, which is what we want - if the response phase deviated
    from what it should be, then there was noise, and that noise shouldn't contribute to our response.
  
    Then, with these set of response amplitudes, fit an RVC (same model as in the '05 paper)

    For mixture stimuli, we will do the following (per conversations with JAM and EPS): Get the mean amplitude/phase
    of each component for a given condition (as done for single gratings) -- using the phase/amplitude relationship established
    for that component when presented in isolation, perform the same projection.
    To fit the RVC curve, then, simply fit the model to the sum of the adjusted individual component responses.
    The Sach DoG curves should also be fit to this sum.
'''

def phase_advance_fit(cell_num, data_loc, expInd, phAdvName=phAdvName, to_save=1, disp=0, dir=1, expName=expName):
  ''' Given the FFT-derived response amplitude and phase, determine the response phase relative
      to the stimulus by taking into account the stimulus phase. 
      Then, make a simple linear model fit (line + constant offset) of the response phase as a function
      of response amplitude.
      vSAVES loss/optimized parameters/and phase advance (if default "to_save" value is kept)
      RETURNS phAdv_model, all_opts

      Do ONLY for single gratings
  '''

  assert disp==0, "In phase_advance_fit; we only fit ph-amp relationship for single gratings."
  assert expInd>2, "In phase_advance_fit; we can only fit ph-amp relationship for experiments with \
                    careful component TF; expInd 1, 2 do not meet this requirement."

  dataList = hf.np_smart_load(data_loc + expName);
  cellStruct = hf.np_smart_load(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  phAdvName = hf.phase_fit_name(phAdvName, dir);

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, _, _ = hf.tabulate_responses(data, expInd);
  allCons = stimVals[1];
  allSfs = stimVals[2];

  # for all con/sf values for this dispersion, compute the mean amplitude/phase per condition
  allAmp, allPhi, allTf, _, _ = hf.get_all_fft(data, disp, expInd, dir=dir, all_trials=0); # all_trials=1 for debugging (0 is default)
     
  # now, compute the phase advance
  conInds = valConByDisp[disp];
  conVals = allCons[conInds];
  nConds = len(allAmp); # this is how many conditions are present for this dispersion
  # recall that nConds = nCons * nSfs
  allCons = [conVals] * nConds; # repeats list and nests
  phAdv_model, all_opts, all_phAdv, all_loss = hf.phase_advance(allAmp, allPhi, allCons, allTf);

  if os.path.isfile(data_loc + phAdvName):
      phFits = hf.np_smart_load(data_loc + phAdvName);
  else:
      phFits = dict();

  # update stuff - load again in case some other run has saved/made changes
  if os.path.isfile(data_loc + phAdvName):
      print('reloading phAdvFits...');
      phFits = hf.np_smart_load(data_loc + phAdvName);
  if cell_num-1 not in phFits:
    phFits[cell_num-1] = dict();
  phFits[cell_num-1]['loss'] = all_loss;
  phFits[cell_num-1]['params'] = all_opts;
  phFits[cell_num-1]['phAdv'] = all_phAdv;

  if to_save:
    np.save(data_loc + phAdvName, phFits);
    print('saving phase advance fit for cell ' + str(cell_num));

  return phAdv_model, all_opts;

def rvc_adjusted_fit(cell_num, data_loc, expInd, descrFitName_f0=None, rvcName=rvcName_f1, descrFitName_f1=None, to_save=1, disp=0, dir=1, expName=expName, force_f1=False, rvcMod=0):
  ''' Piggy-backing off of phase_advance_fit above, get prepared to project the responses onto the proper phase to get the correct amplitude
      Then, with the corrected response amplitudes, fit the RVC model
      - as of 19.11.07, we will fit non-baseline subtracted responses 
          (F1 have baseline of 0 always, but now we will not subtract baseline from F0 responses)
  '''
  #########
  ### load data/metadata
  #########
  dataList = hf.np_smart_load(data_loc + expName);
  cellName = dataList['unitName'][cell_num-1];
  cellStruct = hf.np_smart_load(data_loc + cellName + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  rvcNameFinal = hf.rvc_fit_name(rvcName, rvcMod, dir);
  expInd = hf.get_exp_ind(data_loc, cellName)[0];

  #########
  ### compute f1/f0, gather data of experiment/trials
  #########

  # before anything, let's get f1/f0 ratio
  # note: we are now using "none" for the compute_f1f0 in rvc_adj_fit - this ensures we don't rely on already existing descrFits to get this calculation
  f1f0 = hf.compute_f1f0(data, cell_num, expInd, data_loc, descrFitName_f0=None, descrFitName_f1=None)[0];

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, valByStimVal, _ = hf.tabulate_responses(data, expInd);
  allCons = stimVals[1];
  allSfs = stimVals[2];
  try:
    valCons = allCons[valConByDisp[disp]];
  except:
    warnings.warn('This experiment does not have dispersion level %d; returning empty arrays' % disp);
    return [], [], [], [];

  #########
  ### Now, we fit the RVC
  #########
  # first, load the file if it already exists
  if os.path.isfile(data_loc + rvcNameFinal):
      rvcFits = hf.np_smart_load(data_loc + rvcNameFinal);
      try:
        rvcFits_curr = rvcFits[cell_num-1][disp];
      except:
        rvcFits_curr = None;
  else:
      rvcFits_curr = None;

  ######
  # simple cell
  ######
  if f1f0 > 1 or force_f1 is True:
    # calling phase_advance fit, use the phAdv_model and optimized paramters to compute the true response amplitude
    # given the measured/observed amplitude and phase of the response
    # NOTE: We always call phase_advance_fit with disp=0 (default), since we don't make a fit
    # for the mixtrue stimuli - instead, we use the fits made on single gratings to project the
    # individual-component-in-mixture responses
    phAdv_model, all_opts = phase_advance_fit(cell_num, data_loc=data_loc, expInd=expInd, dir=dir, to_save = 0); # don't save
    allAmp, allPhi, _, allCompCon, allCompSf = hf.get_all_fft(data, disp, expInd, dir=dir, all_trials=1);
    # get just the mean amp/phi and put into convenient lists
    allAmpMeans = [[x[0] for x in sf] for sf in allAmp]; # mean is in the first element; do that for each [mean, std] pair in each list (split by sf)
    allAmpTrials = [[x[2] for x in sf] for sf in allAmp]; # trial-by-trial is third element 

    allPhiMeans = [[x[0] for x in sf] for sf in allPhi]; # mean is in the first element; do that for each [mean, var] pair in each list (split by sf)
    allPhiTrials = [[x[2] for x in sf] for sf in allPhi]; # trial-by-trial is third element 

    adjMeans   = hf.project_resp(allAmpMeans, allPhiMeans, phAdv_model, all_opts, disp, allCompSf, allSfs);
    adjByTrial = hf.project_resp(allAmpTrials, allPhiTrials, phAdv_model, all_opts, disp, allCompSf, allSfs);
    consRepeat = [valCons] * len(adjMeans);

    if disp > 0: # then we need to sum component responses and get overall std measure (we'll fit to sum, not indiv. comp responses!)
      adjSumResp  = [np.sum(x, 1) if x else [] for x in adjMeans];
      adjSemTr    = [[sem(np.sum(hf.switch_inner_outer(x), 1)) for x in y] for y in adjByTrial]
      adjSemCompTr  = [[sem(hf.switch_inner_outer(x)) for x in y] for y in adjByTrial];
      rvc_model, all_opts, all_conGains, all_loss = hf.rvc_fit(adjSumResp, consRepeat, adjSemTr, mod=rvcMod, fix_baseline=True, prevFits=rvcFits_curr);
    elif disp == 0:
      adjSemTr   = [[sem(x) for x in y] for y in adjByTrial];
      adjSemCompTr = adjSemTr; # for single gratings, there is only one component!
      rvc_model, all_opts, all_conGains, all_loss = hf.rvc_fit(adjMeans, consRepeat, adjSemTr, mod=rvcMod, fix_baseline=True, prevFits=rvcFits_curr);
  ######
  # complex cell
  ######
  else: ### FIT RVC TO F0 -- as of 19.11.07, this will NOT be baseline subtracted
    # as above, we pass in None for descrFitNames to ensure no dependence on existing descrFits in rvcFits
    spikerate = hf.get_adjusted_spikerate(data, cell_num, expInd, data_loc, rvcName=None, descrFitName_f0=None, descrFitName_f1=None, baseline_sub=False);
    # recall: rvc_fit wants adjMeans/consRepeat/adjSemTr organized as nSfs lists of nCons elements each (nested)
    respsOrg = hf.organize_resp(spikerate, data, expInd, respsAsRate=True)[3];
    #  -- so now, we organize
    adjMeans = []; adjSemTr = []; adjByTrial = [];
    curr_cons = valConByDisp[disp];
    # now, note that we also must add in the blank responses (0% contrast)
    blankMean, blankSem, blankByTr = hf.blankResp(data, expInd, returnRates=True);
    for sf_i, sf_val in enumerate(allSfs):
      # each list we add here should be of length nCons+1 (+1 for blank);
      # create empties
      mnCurr = []; semCurr = []; adjCurr = []; 
      for con_i in curr_cons:
        curr_resps = hf.nan_rm(respsOrg[disp, sf_i, con_i, :]);
        if np.array_equal([], curr_resps) or np.array_equal(np.nan, curr_resps):
          if np.array_equal(mnCurr, []): # i.e. we haven't added anything yet
            # then first, let's add NaN for the blank condition (we don't want to confuse things by adding the true blank values)
            mnCurr.append(np.nan); semCurr.append(np.nan); adjCurr.append(np.nan);
          mnCurr.append(np.nan); semCurr.append(np.nan); adjCurr.append(np.nan);
          continue;
        else:
          if np.array_equal(mnCurr, []): # i.e. we haven't added anything yet
            # then first, let's add the blank values
            mnCurr.append(blankMean); semCurr.append(blankSem); adjCurr.append(blankByTr);
          mnCurr.append(np.mean(curr_resps)); semCurr.append(sem(curr_resps));
          val_tr = hf.get_valid_trials(data, disp, con_i, sf_i, expInd, stimVals=stimVals, validByStimVal=valByStimVal)[0];
          adjCurr.append(np.array(spikerate[val_tr]));
      adjMeans.append(mnCurr); adjSemTr.append(semCurr);
      adjByTrial.append(adjCurr); # put adjByTrial in same format as adjMeans/adjSemTr!!!
    # to ensure correct mapping of response to contrast, let's add "0" to the front of the list of contrasts
    consRepeat = [np.hstack((0, allCons[curr_cons]))] * len(adjMeans);
    rvc_model, all_opts, all_conGains, all_loss = hf.rvc_fit(adjMeans, consRepeat, adjSemTr, mod=rvcMod, prevFits=rvcFits_curr);
    # adjByTrial = spikerate;
    adjSemCompTr = []; # we're getting f0 - therefore cannot get individual component responses!

  if os.path.isfile(data_loc + rvcNameFinal):
      rvcFits = hf.np_smart_load(data_loc + rvcNameFinal);
  else:
      rvcFits = dict();

  # update stuff - load again in case some other run has saved/made changes
  if os.path.isfile(data_loc + rvcNameFinal):
    print('reloading rvcFits...');
    rvcFits = hf.np_smart_load(data_loc + rvcNameFinal);
  if cell_num-1 not in rvcFits:
    rvcFits[cell_num-1] = dict();
    rvcFits[cell_num-1][disp] = dict();
  else: # cell_num-1 is a key in rvcFits
    if disp not in rvcFits[cell_num-1]:
      rvcFits[cell_num-1][disp] = dict();

  rvcFits[cell_num-1][disp]['loss'] = all_loss;
  rvcFits[cell_num-1][disp]['params'] = all_opts;
  rvcFits[cell_num-1][disp]['conGain'] = all_conGains;
  rvcFits[cell_num-1][disp]['adjMeans'] = adjMeans;
  rvcFits[cell_num-1][disp]['adjByTr'] = adjByTrial
  rvcFits[cell_num-1][disp]['adjSem'] = adjSemTr;
  rvcFits[cell_num-1][disp]['adjSemComp'] = adjSemCompTr;

  if to_save:
    np.save(data_loc + rvcNameFinal, rvcFits);
    print('saving rvc fit for cell %d, disp %d' % (cell_num, disp));

  return rvc_model, all_opts, all_conGains, adjMeans;

### 1.1 RVC fits without adjusted responses (organized like SF tuning)

def fit_RVC_f0(cell_num, data_loc, n_repeats=500, fLname = rvcName_f0, dLname=expName, modelRecov=modelRecov, normType=normType, rvcMod=0):
  # TODO: Should replace spikes with baseline subtracted spikes?
  # NOTE: n_repeats not used (19.05.06)
  # normType used iff modelRecv == 1

  if rvcMod == 0:
    nParam = 3; # RVC model is 3 parameters only
  else:
    nParam = 5; # naka rushton is 4, peirce modification is 5 (but for NR, we just add a fixed parameter for #5)

  # load cell information
  dataList = hf.np_smart_load(data_loc + dLname);
  assert dataList!=[], "data file not found!"
  cellStruct = hf.np_smart_load(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  # get expInd, load rvcFits [if existing]
  expInd, expName = hf.get_exp_ind(data_loc, dataList['unitName'][cell_num-1]);
  print('Making RVC (F0) fits for cell %d in %s [%s]\n' % (cell_num,data_loc,expName));

  name_final = '%s%s.npy' % (fLname, hf.rvc_mod_suff(rvcMod));
  if os.path.isfile(data_loc + name_final):
    rvcFits = hf.np_smart_load(data_loc + name_final);
  else:
    rvcFits = dict();

  # now, get the spikes (recovery, if specified) and organize for fitting
  if modelRecov == 1:
    recovSpikes = hf.get_recovInfo(cellStruct, normType)[1];
  else:
    recovSpikes = None;
  ### Note: should replace with get_adjusted_spikerate? can do if passing in None for descrFits (TODO?)
  spks = hf.get_spikes(data, get_f0=1, rvcFits=None, expInd=expInd, overwriteSpikes=recovSpikes); # we say None for rvc (F1) fits
  _, _, resps_mean, resps_all = hf.organize_resp(spks, cellStruct, expInd, respsAsRate=False); # spks is spike count, not rate
  resps_sem = sem(resps_all, axis=-1, nan_policy='omit');
  
  print('Doing the work, now');

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, _, _ = hf.tabulate_responses(data, expInd);
  all_disps = stimVals[0];
  all_cons = stimVals[1];
  all_sfs = stimVals[2];
  
  nDisps = len(all_disps);
  nSfs = len(all_sfs);

  # Get existing fits
  if cell_num-1 in rvcFits:
    bestLoss = rvcFits[cell_num-1]['loss'];
    currParams = rvcFits[cell_num-1]['params'];
    conGains = rvcFits[cell_num-1]['conGain'];
  else: # set values to NaN...
    bestLoss = np.ones((nDisps, nSfs)) * np.nan;
    currParams = np.ones((nDisps, nSfs, nParam)) * np.nan;
    conGains = np.ones((nDisps, nSfs)) * np.nan;

  for d in range(nDisps): # works for all disps
    val_sfs = hf.get_valid_sfs(data, d, valConByDisp[d][0], expInd); # any valCon will have same sfs
    for sf in val_sfs:
      curr_conInd = valConByDisp[d];
      curr_conVals = all_cons[curr_conInd];
      curr_resps, curr_sem = resps_mean[d, sf, curr_conInd], resps_sem[d, sf, curr_conInd];
      # wrap in arrays, since rvc_fit is written for multiple rvc fits at once (i.e. vectorized)
      _, params, conGain, loss = hf.rvc_fit([curr_resps], [curr_conVals], [curr_sem], n_repeats=n_repeats, mod=rvcMod);

      if (np.isnan(bestLoss[d, sf]) or loss < bestLoss[d, sf]) and params[0] != []: # i.e. params is not empty
        bestLoss[d, sf] = loss[0];
        currParams[d, sf, :] = params[0][:]; # "unpack" the array
        conGains[d, sf] = conGain[0];

  # update stuff - load again in case some other run has saved/made changes
  if os.path.isfile(data_loc + name_final):
    print('reloading RVC (F0) fits...');
    rvcFits = hf.np_smart_load(data_loc + name_final);
  if cell_num-1 not in rvcFits:
    rvcFits[cell_num-1] = dict();
  rvcFits[cell_num-1]['loss'] = bestLoss;
  rvcFits[cell_num-1]['params'] = currParams;
  rvcFits[cell_num-1]['conGain'] = conGains;

  np.save(data_loc + name_final, rvcFits);

#####################################

### 2: Descriptive tuning fit to (adjusted, if needed) responses
# previously, only difference of gaussian models; now (May 2019), we've also added the original flexible (i.e. two-halved) Gaussian model
# this is meant to be general for all experiments, so responses can be F0 or F1, and the responses will be the adjusted ones if needed

def invalid(params, bounds):
# given parameters and bounds, are the parameters valid?
  for p in range(len(params)):
    if params[p] < bounds[p][0] or params[p] > bounds[p][1]:
      return True;
  return False;

def fit_descr_DoG(cell_num, data_loc, n_repeats=100, loss_type=3, DoGmodel=1, is_f0=0, get_rvc=1, dir=+1, gain_reg=0, fLname = dogName, dLname=expName, modelRecov=modelRecov, normType=normType, rvcName=rvcName_f1, rvcMod=0, joint=0):
  ''' This function is used to fit a descriptive tuning function to the spatial frequency responses of individual neurons 
      note that we must fit to non-negative responses - thus f0 responses cannot be baseline subtracted, and f1 responses should be zero'd (TODO: make the f1 calc. work)
  '''
  if DoGmodel == 0:
    nParam = 5;
  else:
    nParam = 4;

  if joint==True:
    try: # load non_joint fits as a reference (see hf.dog_fit or S. Sokol thesis for details)
      modStr  = hf.descrMod_name(DoGmodel);
      ref_fits = hf.np_smart_load(data_loc + hf.descrFit_name(loss_type, descrBase=fLname, modelName=modStr));
      ref_varExpl = ref_fits[cell_num-1]['varExpl'][0];
    except:
      ref_varExpl = None;
    # now set the name for the joint list
    fLname = '%s_joint' % fLname
  else:
    ref_varExpl = None; # set to None as default      

  # load cell information
  dataList = hf.np_smart_load(data_loc + dLname);
  assert dataList!=[], "data file not found!"
  cellStruct = hf.np_smart_load(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  # get expInd, load rvcFits [if existing, and specified]
  expInd, expName = hf.get_exp_ind(data_loc, dataList['unitName'][cell_num-1]);
  print('Making descriptive SF fits for cell %d in %s [%s]\n' % (cell_num,data_loc,expName));
  if is_f0 == 0 and get_rvc == 1: # NOTE: as of 19.09.16 (well, earlier, actually), rvcFits are on F0 or F1, depending on simple/complex designation - in either case, they are both already as rates!
    rvcFits = hf.get_rvc_fits(data_loc, expInd, cell_num, rvcName=rvcName, rvcMod=rvcMod); # see default arguments in helper_fcns.py
  else:
    rvcFits = None;

  modStr  = hf.descrMod_name(DoGmodel);
  fLname  = hf.descrFit_name(loss_type, descrBase=fLname, modelName=modStr);
  if os.path.isfile(data_loc + fLname):
    descrFits = hf.np_smart_load(data_loc + fLname);
  else:
    descrFits = dict();

  # now, get the spikes (adjusted, if needed) and organize for fitting
  # TODO: Add recovery spikes...
  if modelRecov == 1:
    recovSpikes = hf.get_recovInfo(cellStruct, normType)[1];
  else:
    recovSpikes = None;
  ### Note: can replace with get_adjusted_spikerate, since we do not need descrFits for that anymore (TODO?)
  spks = hf.get_adjusted_spikerate(data, cell_num, expInd, data_loc, rvcName, rvcMod, baseline_sub=False);
  #spks = hf.get_spikes(data, get_f0 = is_f0, rvcFits=rvcFits, expInd=expInd, overwriteSpikes=recovSpikes);

  # Note that if rvcFits is not None, then spks will be rates already
  # ensure the spikes array is a vector of overall response, not split by component 
  spks_sum = np.array([np.sum(x) for x in spks]);

  _, _, resps_mean, resps_all = hf.organize_resp(spks_sum, cellStruct, expInd, respsAsRate=True);  
  resps_sem = sem(resps_all, axis=-1, nan_policy='omit');
  base_rate = hf.blankResp(cellStruct, expInd, spks_sum, spksAsRate=True)[0]; 
  
  print('Doing the work, now');

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, validByStimVal, _ = hf.tabulate_responses(data, expInd);
  all_disps = stimVals[0];
  all_cons = stimVals[1];

  nDisps = len(all_disps);
  nCons = len(all_cons);

  # then, set the default values (NaN for everything)
  bestNLL = np.ones((nDisps, nCons)) * np.nan;
  currParams = np.ones((nDisps, nCons, nParam)) * np.nan;
  varExpl = np.ones((nDisps, nCons)) * np.nan;
  prefSf = np.ones((nDisps, nCons)) * np.nan;
  charFreq = np.ones((nDisps, nCons)) * np.nan;
  if joint==True:
    totalNLL = np.ones((nDisps, )) * np.nan;
    paramList = np.ones((nDisps, ), dtype='O') * np.nan;

  ### here is where we do the real fitting!
  for d in range(nDisps): # works for all disps
    # a separate fitting call for each dispersion
    nll, prms, vExp, pSf, cFreq, totNLL, totPrm = hf.dog_fit([resps_mean, resps_all, resps_sem, base_rate], DoGmodel, loss_type, d, expInd, stimVals, validByStimVal, valConByDisp, n_repeats, joint, gain_reg=gain_reg, ref_varExpl=ref_varExpl)

    # before we update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(data_loc + fLname):
      print('reloading descrFits...');
      descrFits = hf.np_smart_load(data_loc + fLname);
      if cell_num-1 in descrFits:
        bestNLL = descrFits[cell_num-1]['NLL'];
        currParams = descrFits[cell_num-1]['params'];
        varExpl = descrFits[cell_num-1]['varExpl'];
        prefSf = descrFits[cell_num-1]['prefSf'];
        charFreq = descrFits[cell_num-1]['charFreq'];
        if joint==True:
          totalNLL = descrFits[cell_num-1]['totalNLL'];
          paramList = descrFits[cell_num-1]['paramList'];
      else:
        descrFits[cell_num-1] = dict();
    else:
      descrFits[cell_num-1] = dict();

    # now, what we do, depends on if joint or not
    if joint==True:
      if np.isnan(totalNLL[d]) or totNLL < totalNLL[d]: # then UPDATE!
        totalNLL[d] = totNLL;
        paramList[d] = totPrm;
        bestNLL[d, :] = nll;
        currParams[d, :, :] = prms;
        varExpl[d, :] = vExp;
        prefSf[d, :] = pSf;
        charFreq[d, :] = cFreq;
    else:
      # must check separately for each contrast
      for con in reversed(range(nCons)):
        if con not in valConByDisp[d]:
          continue;
        if np.isnan(bestNLL[d, con]) or nll[con] < bestNLL[d, con]: # then UPDATE!
          bestNLL[d, con] = nll[con];
          currParams[d, con, :] = prms[con];
          varExpl[d, con] = vExp[con];
          prefSf[d, con] = pSf[con];
          charFreq[d, con] = cFreq[con];

    descrFits[cell_num-1]['NLL'] = bestNLL;
    descrFits[cell_num-1]['params'] = currParams;
    descrFits[cell_num-1]['varExpl'] = varExpl;
    descrFits[cell_num-1]['prefSf'] = prefSf;
    descrFits[cell_num-1]['charFreq'] = charFreq;
    if joint==True:
      descrFits[cell_num-1]['totalNLL'] = totalNLL;
      descrFits[cell_num-1]['paramList'] = paramList;

    np.save(data_loc + fLname, descrFits);
    print('saving for cell ' + str(cell_num));


### Fin: Run the stuff!

if __name__ == '__main__':

    if len(sys.argv) < 2:
      print('uhoh...you need at least one argument(s) here');
      exit();

    cell_num   = int(sys.argv[1]);
    disp       = int(sys.argv[2]);
    data_dir   = sys.argv[3];
    ph_fits    = int(sys.argv[4]);
    rvc_fits   = int(sys.argv[5]);
    rvcF0_fits = int(sys.argv[6]);
    rvc_model  = int(sys.argv[7]);
    descr_fits = int(sys.argv[8]);
    dog_model  = int(sys.argv[9]);
    loss_type  = int(sys.argv[10]);
    is_joint   = int(sys.argv[11]);
    if len(sys.argv) > 12:
      dir = float(sys.argv[12]);
    else:
      dir = None;
    if len(sys.argv) > 13:
      gainReg = float(sys.argv[13]);
    else:
      gainReg = 0;
    print('Running cell %d in %s' % (cell_num, expName));

    # get the full data directory
    dataPath = basePath + data_dir + data_suff;
    # get the expInd
    dL = hf.np_smart_load(dataPath + expName);
    unitName = dL['unitName'][cell_num-1];
    expInd = hf.get_exp_ind(dataPath, unitName)[0];

    if data_dir == 'LGN/':
      force_f1 = True; # must be F1!
    else:
      force_f1 = False; # let simple/complex be the determing factor!

    # then, put what to run here...
    if dir == None:
      if ph_fits == 1:
        phase_advance_fit(cell_num, data_loc=dataPath, expInd=expInd, disp=disp);
      if rvc_fits == 1:
        rvc_adjusted_fit(cell_num, data_loc=dataPath, expInd=expInd, descrFitName_f0=df_f0, disp=disp, force_f1=force_f1, rvcMod=rvc_model);
      if descr_fits == 1:
        fit_descr_DoG(cell_num, data_loc=dataPath, gain_reg=gainReg, DoGmodel=dog_model, loss_type=loss_type, rvcMod=rvc_model, joint=is_joint);
    else:
      if ph_fits == 1:
        phase_advance_fit(cell_num, data_loc=dataPath, expInd=expInd, disp=disp, dir=dir);
      if rvc_fits == 1:
        rvc_adjusted_fit(cell_num, data_loc=dataPath, expInd=expInd, descrFitName_f0=df_f0, disp=disp, dir=dir, force_f1=force_f1, rvcMod=rvc_model);
      if descr_fits == 1:
        fit_descr_DoG(cell_num, data_loc=dataPath, gain_reg=gainReg, dir=dir, DoGmodel=dog_model, loss_type=loss_type, rvcMod=rvc_model, joint=is_joint);

    if rvcF0_fits == 1:
      fit_RVC_f0(cell_num, data_loc=dataPath, rvcMod=rvc_model);
