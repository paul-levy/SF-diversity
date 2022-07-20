import numpy as np
import sys
import helper_fcns as hf
import scipy.optimize as opt
import copy
import os
from time import sleep
from scipy.stats import sem, poisson
import warnings
import pdb

basePath = os.getcwd() + '/';
data_suff = 'structures/';

if 'pl1465' in basePath:
  hpcSuff = 'HPC';
else:
  hpcSuff = '';

expName = hf.get_datalist(sys.argv[3], force_full=1); # sys.argv[3] is experiment dir
df_f0 = 'descrFits%s_200507_sqrt_flex.npy';
#dogName = 'descrFits%s_220531' % hpcSuff;
dogName = 'descrFits%s_220720vEs' % hpcSuff;
if sys.argv[3] == 'LGN/':
  phAdvName = 'phaseAdvanceFits%s_220531' % hpcSuff
  rvcName_f1 = 'rvcFits%s_220531' % hpcSuff;
  #phAdvName = 'phaseAdvanceFits%s_220519' % hpcSuff
  #rvcName_f1 = 'rvcFits%s_220519' % hpcSuff;
  #phAdvName = 'phaseAdvanceFits_211108'
  #rvcName_f1 = 'rvcFits_211108'; # FOR LGN
  rvcName_f0 = 'rvcFits_211108_f0'; # _pos.npy will be added later, as will suffix assoc. w/particular RVC model
else:
  phAdvName = 'phaseAdvanceFits%s_220718' % hpcSuff
  rvcName_f1 = 'rvcFits%s_220718' % hpcSuff;
  #phAdvName = 'phaseAdvanceFits%s_220609' % hpcSuff
  #rvcName_f1 = 'rvcFits%s_220609' % hpcSuff;
  #phAdvName = 'phaseAdvanceFits%s_210914' % hpcSuff
  #rvcName_f1 = 'rvcFits%s_210914' % hpcSuff; # FOR V1
  #phAdvName = 'phaseAdvanceFits%s_210914' % hpcSuff
  #rvcName_f1 = 'rvcFits%s_210914' % hpcSuff; # FOR V1
  rvcName_f0 = 'rvcFits%s_220609_f0' % hpcSuff; # _pos.npy will be added later, as will suffix assoc. w/particular RVC model
  #rvcName_f0 = 'rvcFits_210914_f0'; # _pos.npy will be added later, as will suffix assoc. w/particular RVC model

'''
modelRecov = 0; # 
## model recovery??? OUTDATED, as of 21.11, and only applicable in RVC fits, anyway
if modelRecov == 1:
  normType = 1; # use if modelRecov == 1 :: 1 - flat; 2 - wght; ...
  dogName =  'mr%s_descrFits_190503' % hf.fitType_suffix(normType);
  rvcName = 'mr%s_rvcFits_f0.npy' % hf.fitType_suffix(normType);
else:
  normType = 0;
'''

##########
### TODO:
##########
# - Redo all of (1) for more general use!

##########
## Table of contents
##########

## phase_advance_fit
## rvc_adjusted_fit  - fit RVC for each SF/disp condition, choosing F0 or F1 given simple/complex determination
## fit_RVC_f0        - forces f0 spikes for RVC

## invalid
## fit_descr_empties -- creates the arrays to be filled
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

def phase_advance_fit(cell_num, expInd, data_loc, phAdvName=phAdvName, to_save=1, disp=0, dir=1, expName=expName, returnMod=1):
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

  if not isinstance(cell_num, int):
    cell_num, cellName = cell_num;
  else:
    dataList = hf.np_smart_load(data_loc + expName);
    assert dataList!=[], "data file not found!"
    cellName = dataList['unitName'][cell_num-1];
  cellStruct = hf.np_smart_load(data_loc + cellName + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  phAdvName = hf.phase_fit_name(phAdvName, dir);

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, _, _ = hf.tabulate_responses(data, expInd);
  allCons = stimVals[1];
  allSfs = stimVals[2];

  # for all con/sf values for this dispersion, compute the mean amplitude/phase per condition
  allAmp, allPhi, allTf, _, _ = hf.get_all_fft(data, disp, expInd, dir=dir, all_trials=0); # all_trials=1 for debugging (0 is default)
  aa, ap, _, _, _ = hf.get_all_fft(data, disp, expInd, dir=dir, all_trials=1);
     
  # now, compute the phase advance
  conInds = valConByDisp[disp];
  conVals = allCons[conInds];
  nConds = len(allAmp); # this is how many conditions are present for this dispersion
  # recall that nConds = nCons * nSfs
  allCons = [conVals] * nConds; # repeats list and nests
  # get list of mean amp, mean phi, std. mean, var phi
  # --- we can try to use this in fitting the phase-amplitude relationship...
  oyvey = [[hf.polar_vec_mean([aa[i_sf][i_con][2]], [ap[i_sf][i_con][2]]) for i_con in range(len(ap[i_sf]))] for i_sf in range(len(aa))];
  phiVars = [[oyvey[x][y][3] for y in range(len(oyvey[x]))] for x in range(len(oyvey))];
  phAdv_model, all_opts, all_phAdv, all_loss = hf.phase_advance(allAmp, allPhi, allCons, allTf, ampSem=None, phiVar=phiVars);

  # update stuff - load again in case some other run has saved/made changes
  curr_fit = dict();
  curr_fit['loss'] = all_loss;
  curr_fit['params'] = all_opts;
  curr_fit['phAdv'] = all_phAdv;
  curr_fit['cellNum'] = cell_num;

  if to_save:

    pass_check=False
    n_tries=100;
    while not pass_check:
    
      if os.path.isfile(data_loc + phAdvName):
        print('reloading phAdvFits...');
        phFits = hf.np_smart_load(data_loc + phAdvName);
      else:
        phFits = dict();
      phFits[cell_num-1] = curr_fit;
      np.save(data_loc + phAdvName, phFits);
      print('saving phase advance fit for cell ' + str(cell_num));

      # now check...
      check = hf.np_smart_load(data_loc + phAdvName);
      try:
        if 'loss' in check[cell_num-1].keys(): # just check that any relevant key is there
          pass_check = True;
          print('...cell %02d passed!' % cell_num);
      except:
        pass; # then we didn't pass --> keep trying
      # --- and if neither pass_check was triggered, then we go back and reload, etc
      n_tries -= 1;
      if n_tries <= 0:
        pass_check = True;
        print('never really passed...');

  if returnMod: # default
    return phAdv_model, all_opts;
  else:
    return curr_fit;

def rvc_adjusted_fit(cell_num, expInd, data_loc, descrFitName_f0=None, rvcName=rvcName_f1, descrFitName_f1=None, to_save=1, disp=0, dir=1, expName=expName, force_f1=False, rvcMod=0, vecF1=0, returnMod=1, n_repeats=25, nBoots=0):
  ''' With the corrected response amplitudes, fit the RVC model; save the results and/or return the resulting fits

       ###############
       ### OUTLINE
       ###############
       ### 1. Load data, set up metadata
       ### 2. Compute f1/f0, organize the data (i.e. which experiment conditions, which spiking responses)
       ### 3a. Set up to make the RVC fits, including establishing the loop in which we bootstrap sample or just run once through
       ### 3b-i.  Simple cell - do the fits
       ### 3b-ii. Complex cell - do the fits
       ### 3c. Reload the rvcFits; add the latest boot iter, if applicable
       ### 4. Save everything (and/or return the results)
       ###############

      - as of 19.11.07, we will fit non-baseline subtracted responses 
          (F1 have baseline of 0 always, but now we will not subtract baseline from F0 responses)
      - If vecF1=0: Piggy-back off of phase_advance_fit above, get prepared to project the responses onto the proper phase to get the correct amplitude
      - If vecF1=1: Do the vector-averaging across conditions to get the mean response per condition
      -- the vecF1=1 approach was started in late 2020/early 2021, a smarter approach for V1 than doing phase-adjustment, per se

      For asMulti fits (i.e. when done in parallel) we do the following to reduce multiple loading of files/race conditions
      --- we'll pass in [cell_num, cellName] as cell_num [to avoid loading datalist]

      If nBoots=0 (default, we just fit once to the original data, i.e. not resamples)

      Update, as of 21.09.13: For each cell, we'll have boot and non-boot versions of each field (e.g. NLL, prefSf) to make analysis (e.g. for hf.jl_perCell() call) easier
      -- To accommodate this, we will always load the descrFit structure for each cell and simply tack on the new information (e.g. boot_prefSf or prefSf [i.e. non-boot])
  '''
  assert expInd>2, "In rvc_adjusted_fit; we can only evaluate F1 for experiments with \
                    careful component TF; expInd 1, 2 do not meet this requirement.\nUse fit_RVC_f0 instead"
  
  # Set up whether we will bootstrap straight away
  resample = False if nBoots <= 0 else True;
  nBoots = 1 if nBoots <= 0 else nBoots;

  #########
  ### 1. load data/metadata
  #########
  if not isinstance(cell_num, int):
    cell_num, cellName = cell_num;
  else:
    dataList = hf.np_smart_load(data_loc + expName);
    cellName = dataList['unitName'][cell_num-1];
  cellStruct = hf.np_smart_load(data_loc + cellName + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  if vecF1==1:
    dir=None;
  rvcNameFinal = hf.rvc_fit_name(rvcName, rvcMod, dir, vecF1);

  #########
  ### 2. compute f1/f0, gather data of experiment/trials
  #########

  # before anything, let's get f1/f0 ratio
  # note: we are now using "none" for the compute_f1f0 in rvc_adj_fit - this ensures we don't rely on already existing descrFits to get this calculation
  f1f0 = hf.compute_f1f0(data, cell_num, expInd, data_loc, descrFitName_f0=None, descrFitName_f1=None)[0];

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, valByStimVal, _ = hf.tabulate_responses(data, expInd);
  allCons = stimVals[1];
  allSfs = stimVals[2];
  try:
    valConInds = valConByDisp[disp];
    valCons = allCons[valConInds];
  except:
    warnings.warn('This experiment does not have dispersion level %d; returning empty arrays' % disp);
    return [], [], [], [];

  #########
  ### 3a. Now, we fit the RVC
  #########
  # first, load the file if it already exists
  prevFits_toSave = dict(); # will be used to ensure we save both boot and non-boot results
  if os.path.isfile(data_loc + rvcNameFinal):
      rvcFits = hf.np_smart_load(data_loc + rvcNameFinal);
      try:
        rvcFits_curr = rvcFits[cell_num-1][disp] if not resample else None;
        prevFits_toSave = rvcFits[cell_num-1][disp]; # why are we saving this regardless? we'll include the boot and non-boot versions
      except:
        rvcFits_curr = None;
  else:
    rvcFits = dict();
    rvcFits_curr = None;
    rvcFits[cell_num-1] = dict();

  # Prepare by creating empty lists to append results (if boot)
  if resample:
    boot_loss = []; boot_opts = []; boot_conGains = []; boot_varExpl = [];
    boot_adjMeans = []; boot_adjByTrial = []; boot_adjByTrialCorr = []; boot_adjSemTr = []; boot_adjSemCompTr = [];

  for boot_i in range(nBoots):
    ######
    # 3b-i. simple cell
    ######
    adjByTrialCorr = None # create the corrected adjByTrialCorr as None, so we know if we've actually made the corrections (iff getting F1 resp AND vecF1 correction)
    if f1f0 > 1 or force_f1 is True:
      # calling phase_advance fit, use the phAdv_model and optimized paramters to compute the true response amplitude
      # given the measured/observed amplitude and phase of the response
      # NOTE: We always call phase_advance_fit with disp=0 (default), since we don't make a fit
      # for the mixtrue stimuli - instead, we use the fits made on single gratings to project the
      # individual-component-in-mixture responses

      # if we are doing phase adjustments
      if vecF1==0:
        phAdv_model, all_opts = phase_advance_fit(cell_num, data_loc=data_loc, expInd=expInd, dir=dir, to_save = 0); # don't save
        allAmp, allPhi, _, allCompCon, allCompSf = hf.get_all_fft(data, disp, expInd, dir=dir, all_trials=1, resample=resample);
        # get just the mean amp/phi and put into convenient lists
        allAmpMeans = [[x[0] for x in sf] for sf in allAmp]; # mean is in the first element; do that for each [mean, std] pair in each list (split by sf)
        allAmpTrials = [[x[2] for x in sf] for sf in allAmp]; # trial-by-trial is third element 

        allPhiMeans = [[x[0] for x in sf] for sf in allPhi]; # mean is in the first element; do that for each [mean, var] pair in each list (split by sf)
        allPhiTrials = [[x[2] for x in sf] for sf in allPhi]; # trial-by-trial is third element 
       
        adjMeans   = hf.project_resp(allAmpMeans, allPhiMeans, phAdv_model, all_opts, disp, allCompSf, allSfs);
        adjByTrial = hf.project_resp(allAmpTrials, allPhiTrials, phAdv_model, all_opts, disp, allCompSf, allSfs);
        # -- adjByTrial is series of nested lists: [nSfs x nConsValid x nComps x nRepeats]

      # if we are doing vector math 
      elif vecF1==1:
        adjByTrial = hf.adjust_f1_byTrial(cellStruct, expInd, dir=-1, whichSpikes=1, binWidth=1e-3)
        # then, sum up the valid components per stimulus component
        allCons = np.vstack(data['con']).transpose();
        blanks = np.where(allCons==0);
        adjByTrialCorr = np.copy(adjByTrial);
        adjByTrialCorr[blanks] = 0; # just set it to 0 if that component was blank during the trial
        adjByTrialSum = np.sum(adjByTrialCorr, axis=1);
        # get the mean resp organized by sfMix condition
        # NOTE: 22.06.22 --> note that respsAsRate=True, here --> why? if F1, it's already a rate, so we shouldn't divide out the stimDur when computing means
        adjMeans, adjByTrialSum = hf.organize_resp(adjByTrialSum, cellStruct, expInd, respsAsRate=True, resample=resample, cellNum=cell_num)[2:];
        # will need to transpose, since axis orders get switched when mixing single # slice with array slice of dim, too
        adjMeans = np.transpose(adjMeans[disp,:,valConInds]);
        adjByTrialSum = np.transpose(adjByTrialSum[disp,:,valConInds,:], (1,0,2)); 
        # -- adjByTrialSum is series of [nSfs x nConsValid x nRepeats], i.e. we've already summed over each component within each repeat
      consRepeat = [valCons] * len(adjMeans);

      ### NOTE: From vecF1==0 case, we know that 
      ### --- adjMeans is list of lists, len 11 (or #sfs); each sublist of len 9 (or #con) (i.e. sfs x con)
      ### --- adjByTrial as with adjMeans, but further subdivided to have nTr in inner-most list

      if disp > 0: # then we need to sum component responses and get overall std measure (we'll fit to sum, not indiv. comp responses!)
        adjSumResp  = [np.sum(x, 1) if x else [] for x in adjMeans] if vecF1 == 0 else adjMeans;
        # --- adjSemTr is [nSf x nValCon], i.e. s.e.m. per condition
        adjSemTr    = [[sem(np.sum(hf.switch_inner_outer(x), 1)) for x in y] for y in adjByTrial] if vecF1 == 0 else [[sem(hf.nan_rm(x)) for x in y] for y in adjByTrialSum]
        adjSemCompTr  = [[sem(hf.switch_inner_outer(x)) for x in y] for y in adjByTrial] if vecF1 == 0 else None;
        rvc_model, all_opts, all_conGains, all_loss = hf.rvc_fit(adjSumResp, consRepeat, adjSemTr, mod=rvcMod, fix_baseline=True, prevFits=rvcFits_curr, n_repeats=n_repeats);
      elif disp == 0:
        if vecF1 == 0:
          adjSemTr   = [[sem(x) for x in y] for y in adjByTrial]; # keeping for backwards compatability? check when this one works
        elif vecF1 == 1:
          adjSemTr   = [[sem(hf.nan_rm(x)) for x in y] for y in adjByTrialSum]; # keeping for backwards compatability? check when this one works
        adjSemCompTr = adjSemTr; # for single gratings, there is only one component!
        rvc_model, all_opts, all_conGains, all_loss = hf.rvc_fit(adjMeans, consRepeat, adjSemTr, mod=rvcMod, fix_baseline=True, prevFits=rvcFits_curr, n_repeats=n_repeats);

      currResp = adjSumResp if disp > 0 else adjMeans
      # --- if the responses we're fitting are [], then we just give np.nan for varExpl
      '''
      if len(np.unique([len(x) for x in all_opts])) > 1:
        print('###########');
        print('OH NO: CELL %d' % cell_num);
        print('###########');
      '''
      try:
        varExpl = [hf.var_explained(hf.nan_rm(np.array(dat)), hf.nan_rm(hf.get_rvcResp(prms, valCons, rvcMod)), None) if dat != [] else np.nan for dat, prms in zip(currResp, all_opts)];
      except:
        varExpl = [np.nan];
        pass;

    ######
    # 3b-ii. complex cell
    ######
    else: ### FIT RVC TO F0 -- as of 19.11.07, this will NOT be baseline subtracted
      # as above, we pass in None for descrFitNames to ensure no dependence on existing descrFits in rvcFits
      spikerate = hf.get_adjusted_spikerate(data, cell_num, expInd, data_loc, rvcName=None, descrFitName_f0=None, descrFitName_f1=None, baseline_sub=False);
      # recall: rvc_fit wants adjMeans/consRepeat/adjSemTr organized as nSfs lists of nCons elements each (nested)
      respsOrg = hf.organize_resp(spikerate, data, expInd, respsAsRate=True, resample=resample, cellNum=cell_num)[3];
      #  -- so now, we organize
      adjMeans = []; adjSemTr = []; adjByTrial = [];
      curr_cons = valConByDisp[disp];
      # now, note that we also must add in the blank responses (0% contrast)
      blankMean, blankSem, blankByTr = hf.blankResp(data, expInd, returnRates=True, resample=resample);
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
      rvc_model, all_opts, all_conGains, all_loss = hf.rvc_fit(adjMeans, consRepeat, adjSemTr, mod=rvcMod, prevFits=rvcFits_curr, n_repeats=n_repeats);
      try:
        varExpl = [hf.var_explained(hf.nan_rm(np.array(dat)), hf.nan_rm(hf.get_rvcResp(prms, np.hstack((0, allCons[curr_cons])), rvcMod)), None) for dat, prms in zip(adjMeans, all_opts)];
      except:
        varExpl = [np.nan];
        pass;
      # adjByTrial = spikerate;
      adjSemCompTr = []; # we're getting f0 - therefore cannot get individual component responses!

    ######
    # 3c. Reload the rvcFits, tack on the latest boot iter results, if applicable
    ######
    ### End of if/else for simple vs. complex (i.e. do this for all cells)
    # update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(data_loc + rvcNameFinal) and to_save:
      print('reloading rvcFits...(with a pause)');
      #sleep(hf.random_in_range([2,5])[0]) # just sleep to avoid multiply saves simultaneously
      rvcFits = hf.np_smart_load(data_loc + rvcNameFinal);
    if cell_num-1 not in rvcFits:
      rvcFits[cell_num-1] = dict();
      rvcFits[cell_num-1][disp] = dict();
    else: # cell_num-1 is a key in rvcFits
      if disp not in rvcFits[cell_num-1]:
        rvcFits[cell_num-1][disp] = dict();

    # Last thing before going to the top of the boot_i loop
    if resample:
      boot_loss.append(all_loss);
      boot_opts.append(all_opts);
      boot_conGains.append(all_conGains);
      boot_varExpl.append(varExpl);
      boot_adjMeans.append(adjMeans);
      boot_adjByTrial.append(adjByTrial);
      boot_adjByTrialCorr.append(adjByTrialCorr);
      boot_adjSemTr.append(adjSemTr);
      boot_adjSemCompTr.append(adjSemCompTr);

  ######
  # 4. Save everything
  ######
  ### After boot_i loop
  if resample: # i.e. bootstrap
    prevFits_toSave['boot_loss'] = boot_loss;
    prevFits_toSave['boot_params'] = boot_opts;
    prevFits_toSave['boot_conGain'] = boot_conGains;
    prevFits_toSave['boot_varExpl'] = boot_varExpl;
    # We should default to not saving these -- will get very large with high nBoots
    #prevFits_toSave['boot_adjMeans'] = boot_adjMeans;
    #prevFits_toSave['boot_adjByTr'] = boot_adjByTrial if boot_adjByTrialCorr is None else boot_adjByTrialCorr;
    #prevFits_toSave['boot_adjSem'] = boot_adjSemTr;
    #prevFits_toSave['boot_adjSemComp'] = boot_adjSemCompTr;
  else:
    prevFits_toSave['loss'] = all_loss;
    prevFits_toSave['params'] = all_opts;
    prevFits_toSave['conGain'] = all_conGains;
    prevFits_toSave['varExpl'] = varExpl;
    prevFits_toSave['adjMeans'] = adjMeans;
    prevFits_toSave['adjByTr'] = adjByTrial if adjByTrialCorr is None else adjByTrialCorr;
    prevFits_toSave['adjSem'] = adjSemTr;
    prevFits_toSave['adjSemComp'] = adjSemCompTr;

  if to_save:

    pass_check=False
    n_tries=100;
    while not pass_check:
    
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

      rvcFits[cell_num-1][disp] = prevFits_toSave;
      np.save(data_loc + rvcNameFinal, rvcFits);
      print('saving rvc fit [%s] for cell %d, disp %d' % (rvcNameFinal, cell_num, disp));

      # now check...
      check = hf.np_smart_load(data_loc + rvcNameFinal);
      if resample: # check that the boot stuff is there
        try:
          if np.any(['boot' in x for x in check[cell_num-1][disp].keys()]):
            pass_check = True;
        except:
          pass; # then we didn't pass --> keep trying
      else:
        try:
          if 'loss' in check[cell_num-1][disp].keys(): # just check that any relevant key is there
            pass_check = True;
            print('...cell %02d passed!' % cell_num);
        except:
          pass; # then we didn't pass --> keep trying
      # --- and if neither pass_check was triggered, then we go back and reload, etc
      n_tries -= 1;
      if n_tries <= 0:
        pass_check = True;
        print('never really passed...');

  if returnMod:
    return rvc_model, all_opts, all_conGains, adjMeans;
  else:
    return prevFits_toSave;

### 1.1 RVC fits without adjusted responses (organized like SF tuning)

def fit_RVC_f0(cell_num, data_loc, n_repeats=25, fLname = rvcName_f0, dLname=expName, rvcMod=0, to_save=1, returnDict=0, nBoots=0): # n_repeats was 100, before 21.09.01
  # TODO: Should replace spikes with baseline subtracted spikes?
  # NOTE: n_repeats not used (19.05.06); modelRecov, normType now deprecated (21.11.15)

  # Set up whether we will bootstrap straight away
  resample = False if nBoots <= 0 else True;
  nBoots = 1 if nBoots <= 0 else nBoots;

  if rvcMod == 0:
    nParam = 3; # RVC model is 3 parameters only
  else:
    nParam = 5; # naka rushton is 4, peirce modification is 5 (but for NR, we just add a fixed parameter for #5)

  # load cell information
  if not isinstance(cell_num, int):
    cell_num, cellName, overwriteExpName = cell_num;
  else:
    dataList = hf.np_smart_load(data_loc + dLname);
    assert dataList!=[], "data file not found!"
    cellName = dataList['unitName'][cell_num-1]
  cellStruct = hf.np_smart_load(data_loc + cellName + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  # get expInd, load rvcFits [if existing]
  try:
    overwriteExpName = dataList['expType'][cell_num-1]
  except:
    overwriteExpName = None if not overwriteExpName else overwriteExpName; # we would've already established it above
  expInd, expName = hf.get_exp_ind(data_loc, cellName, overwriteExpName);
  print('Making RVC (F0) fits for cell %d in %s [%s]\n' % (cell_num,data_loc,expName));
 
  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, _, _ = hf.tabulate_responses(data, expInd);
  all_disps = stimVals[0];
  all_cons = stimVals[1];
  all_sfs = stimVals[2];

  nDisps = len(all_disps);
  nSfs = len(all_sfs);

  name_final = '%s%s.npy' % (fLname, hf.rvc_mod_suff(rvcMod));
  if os.path.isfile(data_loc + name_final):
    rvcFits = hf.np_smart_load(data_loc + name_final);
  else:
    rvcFits = dict();

  # DEPRECATED //now, get the spikes (recovery, if specified) and organize for fitting
  recovSpikes = None;
  '''
  if modelRecov == 1:
    recovSpikes = hf.get_recovInfo(cellStruct, normType)[1];
  else:
    recovSpikes = None;
  '''

  ### Note: should replace with get_adjusted_spikerate? can do if passing in None for descrFits (TODO?)
  spks = hf.get_spikes(data, get_f0=1, rvcFits=None, expInd=expInd, overwriteSpikes=recovSpikes); # we say None for rvc (F1) fits
  
  # Get existing fits
  prevFits_toSave = dict(); # ensuring we can save boot and non-boot results
  prevFits = None;
  if cell_num-1 in rvcFits:
    bestLoss = rvcFits[cell_num-1]['loss'];
    currParams = rvcFits[cell_num-1]['params'];
    conGains = rvcFits[cell_num-1]['conGain'];
    varExpl = rvcFits[cell_num-1]['varExpl']; 
    prevFits = rvcFits[cell_num-1] if not resample else None;
    prevFits_toSave = rvcFits[cell_num-1];
  else: # set values to NaN...
    bestLoss = np.ones((nDisps, nSfs)) * np.nan;
    currParams = np.ones((nDisps, nSfs, nParam)) * np.nan;
    conGains = np.ones((nDisps, nSfs)) * np.nan;
    varExpl = np.ones((nDisps, nSfs)) * np.nan;

  if resample:
    boot_bestLoss = []; boot_currParams = []; boot_conGains = []; boot_varExpl = [];

  # now, we can bootstrap the responses, if needed
  for boot_i in range(nBoots):

    if resample: # then we need to reset the arrays each time around
      bestLoss = np.ones((nDisps, nSfs)) * np.nan;
      currParams = np.ones((nDisps, nSfs, nParam)) * np.nan;
      conGains = np.ones((nDisps, nSfs)) * np.nan;
      varExpl = np.ones((nDisps, nSfs)) * np.nan;

    _, _, resps_mean, resps_all = hf.organize_resp(spks, cellStruct, expInd, respsAsRate=False, resample=resample, cellNum=cell_num); # spks is spike count, not rate
    resps_sem = sem(resps_all, axis=-1, nan_policy='omit');

    for d in range(nDisps): # works for all disps
      val_sfs = hf.get_valid_sfs(data, d, valConByDisp[d][0], expInd); # any valCon will have same sfs
      for sf in val_sfs:
        curr_conInd = valConByDisp[d];
        curr_conVals = all_cons[curr_conInd];
        curr_resps, curr_sem = resps_mean[d, sf, curr_conInd], resps_sem[d, sf, curr_conInd];
        # wrap in arrays, since rvc_fit is written for multiple rvc fits at once (i.e. vectorized)
        _, params, conGain, loss = hf.rvc_fit([curr_resps], [curr_conVals], [curr_sem], n_repeats=n_repeats, mod=rvcMod, prevFits=prevFits, cond=(d,sf));

        if (np.isnan(bestLoss[d, sf]) or loss < bestLoss[d, sf]) and params[0] != []: # i.e. params is not empty
          bestLoss[d, sf] = loss[0];
          currParams[d, sf, :] = params[0][:]; # "unpack" the array
          conGains[d, sf] = conGain[0];
          varExpl[d, sf] = hf.var_explained(hf.nan_rm(curr_resps), hf.nan_rm(hf.get_rvcResp(params[0], curr_conVals, rvcMod)), None);

    # end of each boot iteration; before we go around, we append
    if resample:
      boot_bestLoss.append(bestLoss);
      boot_currParams.append(currParams);
      boot_conGains.append(conGains);
      boot_varExpl.append(varExpl);
    
  # end of all boot iterations
  if resample:
    prevFits_toSave['boot_loss'] = boot_bestLoss;
    prevFits_toSave['boot_params'] = boot_currParams;
    prevFits_toSave['boot_conGain'] = boot_conGains;
    prevFits_toSave['boot_varExpl'] = boot_varExpl;
  else:
    prevFits_toSave['loss'] = bestLoss;
    prevFits_toSave['params'] = currParams;
    prevFits_toSave['conGain'] = conGains;
    prevFits_toSave['varExpl'] = varExpl;

  if to_save:
    # update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(data_loc + name_final):
      print('reloading RVC (F0) fits...');
      rvcFits = hf.np_smart_load(data_loc + name_final);
    rvcFits[cell_num-1] = prevFits_toSave;
    np.save(data_loc + name_final, rvcFits);

  if returnDict:
    return prevFits_toSave;

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

def fit_descr_empties(nDisps, nCons, nParam, joint=0, nBoots=1, flt32=True):
  ''' Just create the empty numpy arrays that we'll fill up 
      -- flt32: numpy default is float64 -- float32 will be 2x smaller on memory
      -- TODO: Do I need this control statement or can it be cleaned up?
  '''
  nBoots = 1 if nBoots <= 0 else nBoots;
  dt = np.float32 if flt32 else np.float64;

  ## Yes, I'm pre-pending the nBoots dimensions, but if it's one, we'll use np.squeeze at the end to remove that
  # Set the default values (NaN for everything)
  bestNLL = np.ones((nBoots, nDisps, nCons), dtype=dt) * np.nan;
  currParams = np.ones((nBoots, nDisps, nCons, nParam), dtype=dt) * np.nan;
  varExpl = np.ones((nBoots, nDisps, nCons), dtype=dt) * np.nan;
  prefSf = np.ones((nBoots, nDisps, nCons), dtype=dt) * np.nan;
  charFreq = np.ones((nBoots, nDisps, nCons), dtype=dt) * np.nan;
  if joint>0:
    totalNLL = np.ones((nBoots, nDisps, ), dtype=dt) * np.nan;
    paramList = np.ones((nBoots, nDisps, ), dtype='O') * np.nan;
    success = np.zeros((nBoots, nDisps, ), dtype=np.bool_);
  else:
    totalNLL = None;
    paramList = None;
    success = np.zeros((nBoots, nDisps, nCons), dtype=np.bool_);

  return bestNLL, currParams, varExpl, prefSf, charFreq, totalNLL, paramList, success;
 
def fit_descr_DoG(cell_num, data_loc, n_repeats=1, loss_type=3, DoGmodel=1, force_dc=False, get_rvc=1, dir=+1, gain_reg=0, fLname = dogName, dLname=expName, modRecov=False, rvcName=rvcName_f1, rvcMod=0, joint=0, vecF1=0, to_save=1, returnDict=0, force_f1=False, fracSig=1, debug=0, nBoots=0, cross_val=None, vol_lam=0, no_surr=False, jointMinCons=3, phAmpOnMean=False, phAdvName=phAdvName, resp_thresh=(-1e5,0), sfModRef=1, veThresh=60): # n_repeats was 100, before 21.09.01
  ''' This function is used to fit a descriptive tuning function to the spatial frequency responses of individual neurons 
      note that we must fit to non-negative responses - thus f0 responses cannot be baseline subtracted, and f1 responses should be zero'd (TODO: make the f1 calc. work)

      ###############
      ### OUTLINE
      ###############
      ### 1. Set up if joint and/or bootstrap; load data, metadata
      ### --- if joint = 1, then we fix both radiil; if joint = 2, then the center radius is free at each contrast, but the surround is fixed
      ### 2. Compute f1/f0, organize the data (i.e. which experiment conditions, which spiking responses)
      ### 3a. Set up to make the RVC fits, including establishing the loop in which we bootstrap sample or just run once through
      ### 3b-i.  Simple cell - do the fits
      ### 3b-ii. Complex cell - do the fits
      ### 3c. Reload the rvcFits; add the latest boot iter, if applicable
      ### 4. Save everything (and/or return the results)
      ###############

      NOTE: 22.06.06 - if phAmpOnMean (and not vecF1; will only happen if LGN), then we correct each condition's response mean by projecting on the vector mean for that condition
      ---------------- can only use with: no resampling; no cross-val but resampling; cross_val==2.0; i.e. we cannot, as of 22.06.06, use with 0<cross_val<1
      ---------------- however, this means that resps_all (as passed into hf.dog_fit) will be done on phAdvByTrial responses (i.e. corrected trial-by-trial)
      ---------------- resps_all is just used for var/mean calculations, etc, and has no real influence on the fit, so we're OK to only correct resps_mean

      For asMulti fits (i.e. when done in parallel) we do the following to reduce multiple loading of files/race conditions
      --- we'll pass in [cell_num, cellName] as cell_num [to avoid loading datalist]
   
      If nBoots=0 (default, we just fit once to the original data, i.e. not resamples)
      -- As of 21.10.31, if cross_val is not None [e.g. 0.7], then we'll use the bootstrap infrastructure, but make cross-validated holdout fits (and analysis of loss on holdout data)

      Update, as of 21.09.13: For each cell, we'll have boot and non-boot versions of each field (e.g. NLL, prefSf) to make analysis (e.g. for hf.jl_perCell() call) easier
      -- To accommodate this, we will always load the descrFit structure for each cell and simply tack on the new information (e.g. boot_prefSf or prefSf [i.e. non-boot])
      jointMinCons defaults to 3, meaning we only fit a cell jointly if there are at least 3 valid contrasts
  '''
  ######
  # 1. Establish whether joint fits, bootstrapping; load data, metadata
  ######
 
  # Set up whether we will bootstrap straight away
  resample = False if nBoots <= 0 else True;
  if cross_val is not None:
    # we also set to true if cross_val is not None...
    resample = True
  nBoots = 1 if nBoots <= 0 else nBoots;

  nParam = hf.nParams_descrMod(DoGmodel);

  ### load data/metadata
  if not isinstance(cell_num, int):
    cell_num, cellName, overwriteExpName = cell_num;
  else:
    dataList = hf.np_smart_load(data_loc + dLname);
    assert dataList!=[], "data file not found!"
    cellName = dataList['unitName'][cell_num-1];
  cellStruct = hf.np_smart_load(data_loc + cellName + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  # get expInd, load rvcFits [if existing, and specified]
  try:
    overwriteExpName = dataList['expType'][cell_num-1]
  except:
    overwriteExpName = None if not overwriteExpName else overwriteExpName; # we would've already established it above
  expInd, expName = hf.get_exp_ind(data_loc, cellName, overwriteExpName);
  print('Making descriptive SF fits for cell %d in %s [%s]\n' % (cell_num,data_loc,expName));

  phAdj = None if vecF1==1 else 1;
  if joint>0:
    try: # load non_joint fits as a reference (see hf.dog_fit or S. Sokol thesis for details)
      modStr  = hf.descrMod_name(DoGmodel);
      ref_fits = hf.np_smart_load(data_loc + hf.descrFit_name(loss_type, descrBase=fLname, modelName=modStr, joint=0, phAdj=phAdj));
      #ref_varExpl = None; # as of 22.01.14, no longer restricting which conditions are fit jointly
      if DoGmodel==sfModRef: # we've already loaded fits whose varExp we want as ref.
        ref_varExpl = ref_fits[cell_num-1]['varExpl'][0]; # reference varExpl for single gratings
      else:
        vExp_ref_fits = hf.np_smart_load(data_loc + hf.descrFit_name(loss_type, descrBase=fLname, modelName=hf.descrMod_name(sfModRef), joint=0, phAdj=phAdj));
        ref_varExpl = vExp_ref_fits[cell_num-1]['varExpl'][0];
      isolFits = ref_fits[cell_num-1];
      print('properly loaded isol fits');
    except:
      ref_varExpl = None;
      isolFits = None;
  else:
    ref_varExpl = None; # set to None as default
    isolFits = None;

  if force_dc is False and get_rvc == 1: # NOTE: as of 19.09.16 (well, earlier, actually), rvcFits are on F0 or F1, depending on simple/complex designation - in either case, they are both already as rates!
    rvcFits = hf.get_rvc_fits(data_loc, expInd, cell_num, rvcName=rvcName, rvcMod=rvcMod, direc=dir, vecF1=vecF1); # see default arguments in helper_fcns.py
  else:
    rvcFits = None;

  modStr  = hf.descrMod_name(DoGmodel);
  fLname  = hf.descrFit_name(loss_type, descrBase=fLname, modelName=modStr, joint=joint, phAdj=phAdj); # why add modRecov at the end? we want to load the non-modRecov fits first
  prevFits = None; # default to none, but we'll try to load previous fits...
  prevFits_toSave = dict();
  if os.path.isfile(data_loc + fLname):
    descrFits = hf.np_smart_load(data_loc + fLname);
    try:
      prevFits_toSave = descrFits[cell_num-1]; # why are we saving this regardless? we'll include the boot and non-boot versions
    except:
      prevFits_toSave = dict();
    if cell_num-1 in descrFits:
      if modRecov or not resample: # again, if we are resampling and NOT doing model recovery, we do NOT want previous fits
        prevFits = copy.deepcopy(descrFits[cell_num-1]); # why deep copy? Well, we meddle with descrFits, but don't want to overwrite..
  else:
    descrFits = dict();

  ######
  # 2. Now, get the spikes (adjusted, if needed) and organize for fitting
  ######
  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, validByStimVal, _ = hf.tabulate_responses(data, expInd);
  all_disps = stimVals[0];
  all_cons = stimVals[1];

  nDisps = len(all_disps);
  nCons = len(all_cons);

  make_cv_train_subset = False; # make the test-length-matched subsample of the training data? 
  if cross_val == 2.0:
    nBoots = len(valConByDisp[0])*len(stimVals[2]); # stimVals[2] is sfs
    print('# boots is %03d' % nBoots);
  if cross_val is not None:
    if cross_val>0 and cross_val<1:
      make_cv_train_subset = True;

  # -- we'll ask which response measure is returned (DC or F1) so that we can pass in None for base_rate if it's F1 (will become 0)
  # ---- NOTE: We pass in rvcMod=-1 so that we know we're passing in the already loaded rvcFit for that cell
  spks, which_measure = hf.get_adjusted_spikerate(data, cell_num, expInd, data_loc, rvcName=rvcFits, rvcMod=-1, baseline_sub=False, force_dc=force_dc, force_f1=force_f1, return_measure=1, vecF1=vecF1);

  # Note that if rvcFits is not None, then spks will be rates already
  # ensure the spikes array is a vector of overall response, not split by component 
  spks_sum = np.array([np.sum(x) for x in spks]);

  # IF we are re-sampling for bootstrapping, that will happen in hf.organize_resp (such that the resampling occurs within each condition separately)
  # First, set the default values (NaN for everything)
  # --- we'll adjust nBoots if cross_val is not None (should be fraction, i.e. 0.6 or 0.7)
  if cross_val is not None and resample:
    _, _, _, r_all_noresamp = hf.organize_resp(spks_sum, cellStruct, expInd, respsAsRate=True, resample=False, cellNum=cell_num);
    n_nonNan = np.nanmax(np.sum(~np.isnan(r_all_noresamp), axis=-1));
    if nBoots<=1:
      nBoots = n_nonNan; # at least as many boots as there are trials
  bestNLL, currParams, varExpl, prefSf, charFreq, totalNLL, paramList, success = fit_descr_empties(nDisps, nCons, nParam, joint, nBoots);

  #import cProfile

  ######
  # 3a. Start the loop (will go once unless bootstrapping), organize the responses (resampling, if applicable)
  ######
  if cross_val is not None and resample:
    # also create infra. to save nll, vExp
    test_nll = np.copy(bestNLL);
    test_vExp = np.copy(varExpl);
    if make_cv_train_subset:
      tr_subset_nll = np.copy(bestNLL);
      tr_subset_vExp = np.copy(varExpl);

  if phAmpOnMean and phAdj: # should happen iff LGN data
    phAdvFits = hf.np_smart_load(data_loc + hf.phase_fit_name(phAdvName, dir=dir));
    all_opts = phAdvFits[cell_num-1]['params'];
    respsPhAdv_mean_ref = hf.organize_phAdj_byMean(data, expInd, all_opts, stimVals, valConByDisp);

  print('# boots --> %03d' % nBoots);
  for boot_i in range(nBoots):
    if nBoots > 1:
      ftol = 5e-9; # artificially lower value (was prev. 1e-5) that has minimal impact on parameter values; will avoid too many fit iterations
      if np.mod(boot_i, int(np.floor(nBoots/5))) == 0:
        print('iteration %d of %d' % (boot_i, nBoots));
    else:
      ftol = 2.220446049250313e-09; # the default value, per scipy guide (scipy.optimize, for L-BFGS-B)

    cross_val_curr = None if cross_val is None else (cross_val, -1); # why -1? To make resampling rather than sequential sampling
    if cross_val == 2.0: # this is a code to specify restricting by condition:
      val_sfs = hf.get_valid_sfs(cellStruct, disp=disp, con=valConByDisp[0][0], expInd=expInd, stimVals=stimVals, validByStimVal=validByStimVal);
      con_ind, sf_ind = np.floor(np.divide(boot_i, len(val_sfs))).astype('int'), np.mod(boot_i, len(val_sfs)).astype('int');
      print('holding out con/sf indices %02d/%02d' % (con_ind, sf_ind));
      resps_all = np.copy(r_all_noresamp);
      try:
        resps_all[disp, val_sfs[sf_ind], valConByDisp[disp][con_ind]] = np.nan;
      except:
        print('Failure on con/sf %02d/%02d for cell %02d' % (con_ind, sf_ind, cell_num));
        continue;
      if phAmpOnMean:
        resps_mean = np.copy(respsPhAdv_mean_ref);
        resps_mean[disp, val_sfs[sf_ind], valConByDisp[disp][con_ind]] = np.nan;
      else:
        resps_mean = np.nanmean(resps_all, axis=-1);
    else: # NOTE: As of 22.06.06, cannot do cross_val by trial with phAmpOnMean
      _, _, resps_mean, resps_all = hf.organize_resp(spks_sum, cellStruct, expInd, respsAsRate=True, resample=resample, cellNum=cell_num, cross_val=cross_val_curr);
      if cross_val is None and phAmpOnMean: # then we replace resps_mean with the corrected version
        resps_mean = hf.organize_phAdj_byMean(data, expInd, all_opts, stimVals, valConByDisp, resample=resample);

    if cross_val is not None and resample:
      ########
      ### cross-val stuff
      # --- find out which data were held out by:
      # --- turning all NaN into a set value (e.g. -1e3), find where the training/all arrays differ, get those values
      #########
      nan_val = -1e3;
      training = np.copy(resps_all);
      all_data = np.copy(r_all_noresamp);
      test_data = np.nan * np.zeros_like(resps_all);
      # the above needs to be split by case (i.e. is this phAmpOnMean or not)
      training[np.isnan(training)] = nan_val;
      all_data[np.isnan(all_data)] = nan_val;
      heldout = np.abs(all_data - training) > 1e-6; # if the difference is g.t. this, it means they are different value
      test_data[heldout] = all_data[heldout]; # then put the heldout values here
      print('boot %d' % boot_i);
 
      # if the below are true, then we also need to 
      if cross_val==2.0 and phAmpOnMean: # As of 22.06.06, iff we're doing cross-val by condition AND phAmpOnMean, then we'll compute test/train based on the 
        training_phAmp = np.copy(resps_mean);
        all_data_phAmp = np.copy(respsPhAdv_mean_ref);
        test_data_phAmp = np.nan * np.zeros_like(resps_mean);
        training_phAmp[np.isnan(training_phAmp)] = nan_val;
        all_data_phAmp[np.isnan(all_data_phAmp)] = nan_val;
        heldout = np.abs(all_data_phAmp - training_phAmp) > 1e-6; # if the difference is g.t. this, it means they are different value
        test_data_phAmp[heldout] = all_data_phAmp[heldout]; # then put the heldout values here
      else:
        test_data_phAmp = None;

    resps_sem = sem(resps_all, axis=-1, nan_policy='omit');
    base_rate = hf.blankResp(cellStruct, expInd, spks_sum, spksAsRate=True)[0] if which_measure==0 else None;
    baseline = 0 if base_rate is None else base_rate; # what's the baseline to add

    ######
    # 3b. Loop for each dispersion, making fit
    ######
    ### here is where we do the real fitting!
    for d in range(1): #nDisps): # works for all disps
      # a separate fitting call for each dispersion
      if debug:
        nll, prms, vExp, pSf, cFreq, totNLL, totPrm, DEBUG, succ = hf.dog_fit([resps_mean, resps_all, resps_sem, base_rate], DoGmodel, loss_type, d, expInd, stimVals, validByStimVal, valConByDisp, n_repeats, joint, gain_reg=gain_reg, ref_varExpl=ref_varExpl, prevFits=prevFits, fracSig=fracSig, debug=1, vol_lam=vol_lam, modRecov=modRecov, no_surr=no_surr, jointMinCons=jointMinCons, isolFits=isolFits, ftol=ftol, resp_thresh=resp_thresh, veThresh=veThresh)
      else:
        nll, prms, vExp, pSf, cFreq, totNLL, totPrm, succ = hf.dog_fit([resps_mean, resps_all, resps_sem, base_rate], DoGmodel, loss_type, d, expInd, stimVals, validByStimVal, valConByDisp, n_repeats, joint, gain_reg=gain_reg, ref_varExpl=ref_varExpl, prevFits=prevFits, fracSig=fracSig, vol_lam=vol_lam, modRecov=modRecov, no_surr=no_surr, jointMinCons=jointMinCons, isolFits=isolFits, ftol=ftol, resp_thresh=resp_thresh, veThresh=veThresh)

      if cross_val is not None and resample:
        # compute the loss, varExpl on the heldout (i.e. test) data
        test_mn = np.nanmean(test_data, axis=-1) if test_data_phAmp is None else test_data_phAmp;
        test_sem = sem(test_data, axis=-1, nan_policy='omit');
        # assumes not joint??
        test_nlls = np.nan*np.zeros_like(nll);
        test_vExps = np.nan*np.zeros_like(vExp);
        if make_cv_train_subset:
          tr_subset_nlls = np.nan*np.zeros_like(nll);
          tr_subset_vExps = np.nan*np.zeros_like(vExp);

        # set up ref_params, ref_rc_val; will only be used IF applicable
        ref_params = None; ref_rc_val = None;
        try:
          if DoGmodel==3:
            try:
              all_xc = prms[d,:,1]; # xc values
              ref_params = [np.nanmin(all_xc), 1];
            except:
              pass;
          else:
            ref_params = prms[-1]; # high contrast condition
            ref_rc_val = totPrm[2] if joint>0 else None; # even then, only used for joint==5
        except:
          ref_params = None; ref_rc_val = None;
        
        for ii, prms_curr in enumerate(prms):
          # we'll iterate over the parameters, which are fit for each contrast (the final dimension of test_mn)
          if np.any(np.isnan(prms_curr)):
            continue;
          non_nans = np.where(~np.isnan(test_mn[d,:,ii]))[0];
          curr_sfs = stimVals[2][non_nans];
          resps_curr = test_mn[d, non_nans, ii]
          if make_cv_train_subset:
            # now, let's also make a size-matched subset of the training data to see if the small N is the source of the noise
            len_test = np.array([np.sum(~np.isnan(test_data[d, nn, ii])) for nn in non_nans]);
            len_train = np.array([np.sum(~np.isnan(resps_all[d, nn, ii])) for nn in non_nans]);
            to_repl = False if np.max(len_test)<np.max(len_train) else True; # if the training set is SMALLER then the test set, then we should allow resampling; otherwise, don't
            try:
              train_subset_curr = np.array([np.nanmean(np.random.choice(hf.nan_rm(resps_all[d, nn, ii]), num, replace=to_repl)) for nn,num in zip(non_nans, len_test)]);
            except: # we could have all-NaN subset?
              train_subset_curr = None;

          test_nlls[ii] = hf.DoG_loss(prms_curr, resps_curr, curr_sfs, resps_std=test_sem[d, non_nans, ii], loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg, joint=0, baseline=baseline, ref_params=ref_params, ref_rc_val=ref_rc_val) # why not enforce max? b/c fewer resps means more varied range of max, don't want to wrongfully penalize
          test_vExps[ii] = hf.var_explained(resps_curr, prms_curr, curr_sfs, DoGmodel, baseline=baseline, ref_params=ref_params, ref_rc_val=ref_rc_val);
          # and evaluate loss, vExp on a subset of the TRAINING data that has the same # of trials as the test data
          # --- why? per Tony + Eero (22.04.21), we want to see if the large discrepancy in loss has to do with noise in smaller samples
          if make_cv_train_subset and train_subset_curr is not None:
            tr_subset_nlls[ii] = hf.DoG_loss(prms_curr, train_subset_curr, curr_sfs, resps_std=test_sem[d, non_nans, ii], loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg, joint=0, baseline=baseline, ref_params=ref_params, ref_rc_val=ref_rc_val) # why not enforce max? b/c fewer resps means more varied range of max, don't want to wrongfully penalize
            tr_subset_vExps[ii] = hf.var_explained(train_subset_curr, prms_curr, curr_sfs, DoGmodel, baseline=baseline, ref_params=ref_params, ref_rc_val=ref_rc_val);
        test_nll[boot_i, d] = test_nlls;
        test_vExp[boot_i, d] = test_vExps;
        if make_cv_train_subset:
          tr_subset_nll[boot_i, d] = tr_subset_nlls;
          tr_subset_vExp[boot_i, d] = tr_subset_vExps;

      # Update the fits! Now, what we do depends on:
      # -- joint?
      # -- bootstrap? (i.e. if resample, we ALWAYS replace)
      # ---- why do we need to update? When we pass in prevFits, we take into account the previous fits
      if joint>0:
        if resample or np.isnan(totalNLL[boot_i, d]) or totNLL < totalNLL[boot_i, d]: # then UPDATE!
          totalNLL[boot_i, d] = totNLL;
          paramList[boot_i, d] = totPrm;
          success[boot_i, d] = succ;
          bestNLL[boot_i, d, :] = nll;
          currParams[boot_i, d, :, :] = prms;
          varExpl[boot_i, d, :] = vExp;
          prefSf[boot_i, d, :] = pSf;
          charFreq[boot_i, d, :] = cFreq;
      else:
        # must check separately for each contrast
        for con in reversed(range(nCons)):
          if con not in valConByDisp[d]:
            continue;
          if resample or np.isnan(bestNLL[boot_i, d, con]) or nll[con] < bestNLL[boot_i, d, con]: # then UPDATE!
            bestNLL[boot_i, d, con] = nll[con];
            currParams[boot_i, d, con, :] = prms[con];
            varExpl[boot_i, d, con] = vExp[con];
            prefSf[boot_i, d, con] = pSf[con];
            charFreq[boot_i, d, con] = cFreq[con];
            success[boot_i, d, con] = succ[con];

  ###### [END OF BOTH DISP AND BOOT LOOPS]
  # 4. Pack the results, depending on boot or not; then return and/or save, if needed
  ######

  # After all disp iterations, we pack everything up and get ready to return the results
  if not resample: # i.e. not bootstrap, then we'll call np.squeeze to remove the extra 0th dimension
    bestNLL = np.squeeze(bestNLL, axis=0);
    currParams = np.squeeze(currParams, axis=0);
    varExpl = np.squeeze(varExpl, axis=0);
    prefSf = np.squeeze(prefSf, axis=0);
    charFreq = np.squeeze(charFreq, axis=0);
    success = np.squeeze(success, axis=0);
    if joint>0:
      paramList = np.squeeze(paramList, axis=0);
      totalNLL = np.squeeze(totalNLL, axis=0);

  # Pack the dict to save (prevFits_toSave, since it has both boot and non-boot results)
  if resample: # i.e. boot
    # we save the cross-validated stuff separately
    if cross_val is not None:
      # first, pre-define empty lists for all of the needed results, if they are not yet defined
      if 'boot_NLL_cv_test' not in prevFits_toSave:
        prevFits_toSave['boot_cv_lambdas'] = [];
        prevFits_toSave['boot_NLL_cv_test'] = [];
        prevFits_toSave['boot_vExp_cv_test'] = [];
        if make_cv_train_subset:
          prevFits_toSave['boot_NLL_cv_train_subset'] = [];
          prevFits_toSave['boot_vExp_cv_train_subset'] = [];
        prevFits_toSave['boot_NLL_cv_train'] = [];
        prevFits_toSave['boot_vExp_cv_train'] = [];
        # --- these are all implicitly based on training data
        prevFits_toSave['boot_cv_params'] = [];
        prevFits_toSave['boot_cv_prefSf'] = [];
        prevFits_toSave['boot_cv_charFreq'] = [];

      # we'll keep track of which lambda values we use so that we can compare the CV vExpl, NLL across regularization terms (for both train and test)
      prevFits_toSave['boot_cv_lambdas'].append(vol_lam);
      prevFits_toSave['boot_NLL_cv_test'].append(test_nll);
      prevFits_toSave['boot_vExp_cv_test'].append(test_vExp);
      if make_cv_train_subset:
        prevFits_toSave['boot_NLL_cv_train_subset'].append(tr_subset_nll);
        prevFits_toSave['boot_vExp_cv_train_subset'].append(tr_subset_vExp);
      prevFits_toSave['boot_NLL_cv_train'].append(bestNLL);
      prevFits_toSave['boot_vExp_cv_train'].append(varExpl);
      prevFits_toSave['boot_cv_params'].append(currParams);
      prevFits_toSave['boot_cv_prefSf'].append(prefSf);
      prevFits_toSave['boot_cv_charFreq'].append(charFreq);
      if joint>0:
        prevFits_toSave['boot_totalNLL'] = totalNLL;
        prevFits_toSave['boot_paramList'] = paramList;
    else:
      prevFits_toSave['boot_NLL'] = bestNLL;
      prevFits_toSave['boot_params'] = currParams;
      prevFits_toSave['boot_varExpl'] = varExpl;
      prevFits_toSave['boot_prefSf'] = prefSf;
      prevFits_toSave['boot_charFreq'] = charFreq;
      prevFits_toSave['boot_success'] = success;
      if joint>0:
        prevFits_toSave['boot_totalNLL'] = totalNLL;
        prevFits_toSave['boot_paramList'] = paramList;

  else:
    prevFits_toSave['NLL'] = bestNLL;
    prevFits_toSave['params'] = currParams;
    prevFits_toSave['varExpl'] = varExpl;
    prevFits_toSave['prefSf'] = prefSf;
    prevFits_toSave['charFreq'] = charFreq;
    prevFits_toSave['success'] = success;
    if joint>0:
      prevFits_toSave['totalNLL'] = totalNLL;
      prevFits_toSave['paramList'] = paramList;

  # -- and save, if that's what we're doing here
  if to_save: # i.e. this is the final boot iteration

    pass_check=False;
    n_tries = 100;
    while not pass_check: # keep saving/reloading until the fit has properly saved everything...

      # reload in case another thread/call has changed descrFits
      if os.path.isfile(data_loc + fLname):
        descrFits = hf.np_smart_load(data_loc + fLname);
        if descrFits == []:
          continue; # don't try to save the results if we didn't load the dictionary properly
      else:
        descrFits = dict();

      # then save
      # --- first, if model recovery, change the name here; why? Keeping the same name for earlier on allows the 
      if modRecov:
        fLname = fLname.replace('.npy', '_modRecov.npy');
      descrFits[cell_num-1] = prevFits_toSave;
      np.save(data_loc + fLname, descrFits);   

      # now check...
      check = hf.np_smart_load(data_loc + fLname);
      if resample: # check that the boot stuff is there
        if np.any(['boot' in x for x in check[cell_num-1].keys()]):
          pass_check = True;
      else:
        try:
          if 'NLL' in check[cell_num-1].keys(): # just check that any relevant key is there
            pass_check = True;
        except:
          pass; # then we didn't pass...
      # --- and if neither pass_check was triggered, then we go back and reload, etc
      n_tries -= 1;
      if n_tries <= 0:
        pass_check = True;

  # -- and return, if specified
  if returnDict:
    return prevFits_toSave;

### Fin: Run the stuff!

if __name__ == '__main__':

    if len(sys.argv) < 11:
      print('uhoh...you need at least 11 arguments here');
      exit();

    cell_num   = int(sys.argv[1]);
    if cell_num < -99: 
      # i.e. 3 digits AND negative, then we'll treat the first two digits as where to start, and the second two as when to stop
      # -- in that case, we'll do this as multiprocessing
      asMulti = 1;
      end_cell = int(np.mod(-cell_num, 100));
      start_cell = int(np.floor(-cell_num/100));
    else:
      asMulti = 0;
    disp       = int(sys.argv[2]);
    data_dir   = sys.argv[3];
    ph_fits    = int(sys.argv[4]);
    rvc_fits   = int(sys.argv[5]);
    rvcF0_fits = int(sys.argv[6]);
    rvc_model  = int(sys.argv[7]);
    descr_fits = int(sys.argv[8]);
    dog_model  = int(sys.argv[9]);
    loss_type  = int(sys.argv[10]);
    nBoots     = int(sys.argv[11]);
    joint   = int(sys.argv[12]);
    if len(sys.argv) > 13:
      dir = int(sys.argv[13]);
    else:
      dir = 1; # default
    # --- first, 
    if len(sys.argv) > 14:
      modRecov  = int(sys.argv[14]);
    else:
      modRecov = False;
    if len(sys.argv) > 15:
      cross_val  = float(sys.argv[15]);
      if cross_val <= 0: # but if it's <=0, we set it back to None
        cross_val = None;
    else:
      cross_val = None;
    if len(sys.argv) > 16:
      vol_lam    = float(sys.argv[16]);
    else:
      vol_lam = 0;
    if len(sys.argv) > 17: # DEPRECATED as of 21.11
      gainReg = float(sys.argv[17]);
    else:
      gainReg = 0;
    print('Running cell %d in %s' % (cell_num, expName));

    jointMinCons = 2 if data_dir=='V1_orig/' else 3;
    phAmpOnMean = 1 if data_dir=='LGN/' and ph_fits==1 else 0;
    sfModRef = 1 if data_dir=='LGN/' else 1; # which model to use as reference for inclusion
    
    resp_thresh = (-1e5,0); #(5,2); # at least 2 responses must be g.t.e. 5 spks/s (22.07.05 addition/try)
    #veThresh = -np.Inf; # allow all...
    veThresh = 60; 
    
    # get the full data directory
    dataPath = basePath + data_dir + data_suff;
    # get the expInd
    dL = hf.np_smart_load(dataPath + expName);
    if asMulti:
      unitNames = [dL['unitName'][ind-1] for ind in range(start_cell, end_cell+1)];
      try:
        overwriteNames = [dL['expType'][ind-1] for ind in range(start_cell, end_cell+1)];
        expInds = [hf.get_exp_ind(dataPath, name, overwriteName)[0] for name, overwriteName in zip(unitNames, overwriteNames)];
      except:
        expInds = [hf.get_exp_ind(dataPath, name)[0] for name in unitNames];
    else:
      unitName = dL['unitName'][cell_num-1];
      try: 
        overwriteName = dL['expType'][cell_num-1];
        expInd = hf.get_exp_ind(dataPath, unitName, overwriteName)[0];
      except:
        expInd = hf.get_exp_ind(dataPath, unitName)[0];

    if data_dir == 'LGN/':
      force_f1 = True; # must be F1!
    else:
      force_f1 = False; # let simple/complex be the determing factor!
    if data_dir == 'V1_orig/' or data_dir == 'altExp/':
      force_dc = True;
    else:
      force_dc = False;

    vecF1 = 1 if ph_fits == -1 else 0;
    print('ph fits | rvc_fits | vecF1 = %d|%d|%d' % (ph_fits, rvc_fits, vecF1));
    if vecF1:
      dir = 0;
    fracSig = 1;
    #fracSig = 0 if data_dir == 'LGN/' else 1; # we only enforce the "upper-half sigma as fraction of lower half" for V1 cells!
    
    if asMulti:
      from functools import partial
      import multiprocessing as mp
      nCpu = mp.cpu_count()-1; # heuristics say you should reqeuest at least one fewer processes than their are CPU
      print('***cpu count: %02d***' % nCpu);

      ### Phase fits
      if ph_fits == 1 and disp==0: # only call this if disp=0!
        print('phase advance fits!!!');
        with mp.Pool(processes = nCpu) as pool:
          ph_perCell = partial(phase_advance_fit, data_loc=dataPath, disp=disp, dir=dir, returnMod=0, to_save=0);
          phFits = pool.starmap(ph_perCell, zip(zip(range(start_cell, end_cell+1), dL['unitName']), expInds));
          pool.close();

        ### do the saving HERE!
        phAdvName = hf.phase_fit_name(phAdvName, dir);
        if os.path.isfile(dataPath + phAdvName):
          print('reloading phAdvFits...');
          phFitNPY = hf.np_smart_load(dataPath + phAdvName);
        else:
          phFitNPY = dict();
        for iii, phFit in enumerate(phFits):
          phFitNPY[iii] = phFit;
        np.save(dataPath + phAdvName, phFitNPY)

      ### RVC fits
      if rvc_fits == 1:
        print('rvc fits!!!');
        with mp.Pool(processes = nCpu) as pool:
          rvc_perCell = partial(rvc_adjusted_fit, data_loc=dataPath, descrFitName_f0=df_f0, disp=disp, dir=dir, force_f1=force_f1, rvcMod=rvc_model, vecF1=vecF1, returnMod=0, to_save=0, nBoots=nBoots);
          rvcFits = pool.starmap(rvc_perCell, zip(zip(range(start_cell, end_cell+1), dL['unitName']), expInds));
          pool.close();

        ### do the saving HERE!
        rvcNameFinal = hf.rvc_fit_name(rvcName_f1, rvc_model, dir, vecF1);
        if os.path.isfile(dataPath + rvcNameFinal):
          print('reloading rvcFits...');
          rvcFitNPY = hf.np_smart_load(dataPath + rvcNameFinal);
        else:
          rvcFitNPY = dict();

        for iii, rvcFit in enumerate(rvcFits):
          if iii not in rvcFitNPY:
            rvcFitNPY[iii] = dict(); # create the key
          rvcFitNPY[iii][disp] = rvcFit;
        np.save(dataPath + rvcNameFinal, rvcFitNPY)

      ### Descriptive fits
      if descr_fits == 1:

        if nBoots > 1:
          if dog_model==1: # if if just DoG and not d-DoG-S
            n_repeats = 5 if joint>0 else 7;
          else:
            n_repeats = 3 if joint>0 else 5;
        else:
          if dog_model==1: # if if just DoG and not d-DoG-S
            n_repeats = 25 if joint>0 else 50; # was previously be 3, 15, then 7, 15
          else:
            #n_repeats = 12 if joint>0 else 15; # was previously be 3, 15
            n_repeats = 20 if joint>0 else 15; # was previously be 3, 15
          
        print('descr fits! --> mod %d, joint %d, %03d boots, cross_val %.2f' % (dog_model, joint, nBoots, cross_val if cross_val is not None else -99));

        with mp.Pool(processes = nCpu) as pool:
          dir = dir if vecF1 == 0 else None # so that we get the correct rvcFits
          descr_perCell = partial(fit_descr_DoG, data_loc=dataPath, n_repeats=n_repeats, gain_reg=gainReg, dir=dir, DoGmodel=dog_model, loss_type=loss_type, rvcMod=rvc_model, joint=joint, vecF1=vecF1, to_save=0, returnDict=1, force_dc=force_dc, force_f1=force_f1, fracSig=fracSig, nBoots=nBoots, cross_val=cross_val, vol_lam=vol_lam, modRecov=modRecov, jointMinCons=jointMinCons, phAmpOnMean=phAmpOnMean, resp_thresh=resp_thresh, sfModRef=sfModRef, veThresh=veThresh);
          dogFits = pool.map(descr_perCell, zip(range(start_cell, end_cell+1), dL['unitName'], dL['expType']));
          pool.close();

        print('debug');
        ### do the saving HERE!
        phAdj = None if vecF1==1 else 1;
        dogNameFinal = hf.descrFit_name(loss_type, descrBase=dogName, modelName=hf.descrMod_name(dog_model), modRecov=modRecov, joint=joint, phAdj=phAdj);
        if os.path.isfile(dataPath + dogNameFinal):
          dogFitNPY = hf.np_smart_load(dataPath + dogNameFinal);
        else:
          dogFitNPY = dict();
          
        for iii, dogFit in enumerate(dogFits):
          dogFitNPY[iii] = dogFit;
        np.save(dataPath + dogNameFinal, dogFitNPY)

      ### rvcF0 fits
      if rvcF0_fits == 1:
        with mp.Pool(processes = nCpu) as pool:
          rvc_perCell = partial(fit_RVC_f0, data_loc=dataPath, rvcMod=rvc_model, to_save=0, returnDict=1, nBoots=nBoots);
          rvcFits = pool.map(rvc_perCell, zip(range(start_cell, end_cell+1), dL['unitName'], dL['expType']));
          pool.close();

        ### do the saving HERE!
        rvcNameFinal = '%s%s.npy' % (rvcName_f0, hf.rvc_mod_suff(rvc_model));
        if os.path.isfile(dataPath + rvcNameFinal):
          print('reloading rvcFits...');
          rvcFitNPY = hf.np_smart_load(dataPath + rvcNameFinal);
        else:
          rvcFitNPY = dict();
        for iii, rvcFit in enumerate(rvcFits):
          rvcFitNPY[iii] = rvcFit;
        np.save(dataPath + rvcNameFinal, rvcFitNPY)
    else: # if not multi (i.e. parallel...)
      # then, put what to run here...
      if ph_fits == 1 and disp==0:
        phase_advance_fit(cell_num, expInd=expInd, data_loc=dataPath, disp=disp, dir=dir);
      if rvc_fits == 1:
        rvc_adjusted_fit(cell_num, expInd=expInd, data_loc=dataPath, descrFitName_f0=df_f0, disp=disp, dir=dir, force_f1=force_f1, rvcMod=rvc_model, vecF1=vecF1, nBoots=nBoots);
      if descr_fits == 1:
        if nBoots > 1:
          if dog_model<3: # i.e. DoG, not d-DoG-S
            n_repeats = 8 if joint>0 else 10;
          else:
            n_repeats = 2 if joint>0 else 5; # fewer if repeat
        else:
          #n_repeats = 15 if joint>0 else 20; # was previously be 3, 15, then 7, 15
          n_repeats = 3 if joint>0 else 20; # was previously be 3, 15, then 7, 15

        #import cProfile, re
        #cProfile.run('fit_descr_DoG(cell_num, data_loc=dataPath, gain_reg=gainReg, dir=dir, DoGmodel=dog_model, loss_type=loss_type, rvcMod=rvc_model, joint=joint, vecF1=vecF1, fracSig=fracSig, nBoots=nBoots, cross_val=cross_val, vol_lam=vol_lam, modRecov=modRecov)');
        if hf.is_mod_DoG(dog_model) and nBoots<10:
          sleep(hf.random_in_range((0, 20))[0]); # why? DoG fits run so quickly that successive load/save calls take place in an overlapping way and we lose the result of some calls
        dir = dir if vecF1 == 0 else None # so that we get the correct rvcFits
        fit_descr_DoG(cell_num, data_loc=dataPath, n_repeats=n_repeats, gain_reg=gainReg, dir=dir, DoGmodel=dog_model, loss_type=loss_type, rvcMod=rvc_model, joint=joint, vecF1=vecF1, force_dc=force_dc, force_f1=force_f1, fracSig=fracSig, nBoots=nBoots, cross_val=cross_val, vol_lam=vol_lam, modRecov=modRecov, jointMinCons=jointMinCons, phAmpOnMean=phAmpOnMean, resp_thresh=resp_thresh, sfModRef=sfModRef, veThresh=veThresh);

      if rvcF0_fits == 1:
        fit_RVC_f0(cell_num, data_loc=dataPath, rvcMod=rvc_model, nBoots=nBoots);
