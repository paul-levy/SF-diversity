import numpy as np
import sys
import helper_fcns as hf
import scipy.optimize as opt
import os
from time import sleep
from scipy.stats import sem, poisson
import warnings
import pdb

# personal mac
#basePath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/';
# prince cluster
basePath = '/home/pl1465/SF_diversity/';
# LCV/cns machines
basePath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/';

data_suff = 'structures/';

expName = 'dataList.npy'
dogName =  'descrFits_190129';
phAdvName = 'phaseAdvanceFitsTest'
rvcName   = 'rvcFits'

### TODO:
# Redo all of (1) for more general use!

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

def phase_advance_fit(cell_num, data_loc, expInd, phAdvName=phAdvName, to_save = 1, disp=0, dir=1):
  ''' Given the FFT-derived response amplitude and phase, determine the response phase relative
      to the stimulus by taking into account the stimulus phase. 
      Then, make a simple linear model fit (line + constant offset) of the response phase as a function
      of response amplitude.
      vSAVES loss/optimized parameters/and phase advance (if default "to_save" value is kept)
      RETURNS phAdv_model, all_opts

      Do ONLY for single gratings
  '''

  assert disp==0, "In phase_advance_fit; we only fit ph-amp relationship for single gratings."

  dataList = hf.np_smart_load(data_loc + 'dataList.npy');
  cellStruct = hf.np_smart_load(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  phAdvName = hf.phase_fit_name(phAdvName, dir);

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, _, _ = hf.tabulate_responses(data, expInd);
  allCons = stimVals[1];
  allSfs = stimVals[2];

  # for all con/sf values for this dispersion, compute the mean amplitude/phase per condition
  allAmp, allPhi, allTf, _, _ = hf.get_all_fft(data, disp, expInd, dir=dir); 
     
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

def rvc_adjusted_fit(cell_num, data_loc, rvcName=rvcName, to_save=1, disp=0, dir=-1):
  ''' Piggy-backing off of phase_advance_fit above, get prepare to project the responses onto the proper phase to get the correct amplitude
      Then, with the corrected response amplitudes, fit the RVC model
  '''
  dataList = hf.np_smart_load(data_loc + 'dataList.npy');
  cellStruct = hf.np_smart_load(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  rvcNameFinal = hf.phase_fit_name(rvcName, dir);

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, _, _ = hf.tabulate_responses(data, expInd);
  allCons = stimVals[1];
  allSfs = stimVals[2];
  try:
    valCons = allCons[valConByDisp[disp]];
  except:
    warnings.warn('This experiment does not have dispersion level %d; returning empty arrays' % disp);
    return [], [], [], [];
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
    rvc_model, all_opts, all_conGains, all_loss = hf.rvc_fit(adjSumResp, consRepeat, adjSemTr);
  elif disp == 0:
    adjSemTr   = [[sem(x) for x in y] for y in adjByTrial];
    adjSemCompTr = adjSemTr; # for single gratings, there is only one component!
    rvc_model, all_opts, all_conGains, all_loss = hf.rvc_fit(adjMeans, consRepeat, adjSemTr);

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
    print('saving rvc fit for cell ' + str(cell_num));

  return rvc_model, all_opts, all_conGains, adjMeans;

### 2: Difference of gaussian fit to (adjusted, if needed) responses
# this is meant to be general for all experiments, so responses can be F0 or F1, and the responses will be the adjusted ones if needed

def invalid(params, bounds):
# given parameters and bounds, are the parameters valid?
  for p in range(len(params)):
    if params[p] < bounds[p][0] or params[p] > bounds[p][1]:
      return True;
  return False;

def DoG_loss(params, resps, sfs, loss_type = 3, DoGmodel=1, dir=-1, resps_std=None, gain_reg = 0):
  '''Given the model params (i.e. sach or tony formulation)), the responses, sf values
  return the loss
  loss_type: 1 - lsq
             2 - sqrt
             3 - poiss
             4 - Sach sum{[(exp-obs)^2]/[k+sigma^2]} where
                 k := 0.01*max(obs); sigma := measured variance of the response
  DoGmodel: 1 - sach
            2 - tony
  '''
  # NOTE: See version in LGN/sach/ for how to fit trial-by-trial responses (rather than avg. resp)
  if DoGmodel == 1:
    pred_spikes, _ = hf.DoGsach(*params, stim_sf=sfs);
  elif DoGmodel == 2:
    pred_spikes, _ = hf.DiffOfGauss(*params, stim_sf=sfs);

  loss = 0;
  if loss_type == 1: # lsq
    loss = np.square(resps - pred_spikes);
    loss = loss + loss;
  elif loss_type == 2: # sqrt
    loss = np.square(np.sqrt(resps) - np.sqrt(pred_spikes));
    loss = loss + loss;
  elif loss_type == 3: # poisson model of spiking
    poiss = poisson.pmf(np.round(resps), pred_spikes); # round since the values are nearly but not quite integer values (Sach artifact?)...
    ps = np.sum(poiss == 0);
    if ps > 0:
      poiss = np.maximum(poiss, 1e-6); # anything, just so we avoid log(0)
    loss = loss + sum(-np.log(poiss));
  elif loss_type == 4: # sach's loss function
    k = 0.01*np.max(resps);
    if resps_std is None:
      sigma = np.ones_like(resps);
    else:
      sigma = resps_std;
    sq_err = np.square(resps-pred_spikes);
    loss = loss + np.sum((sq_err/(k+np.square(sigma)))) + gain_reg*(params[0] + params[2]); # regularize - want gains as low as possible
  return loss;

def fit_descr_DoG(cell_num, data_loc, n_repeats=1000, loss_type=3, DoGmodel=1, dir=+1, gain_reg=0, fLname = dogName):
  ''' 
      NOTE: as of now, fLname is overwritten (we just use default from helper_fcns, meaning no date appending)
  '''
  nParam = 4;

  # load cell information
  dataList = hf.np_smart_load(data_loc + 'dataList.npy');
  assert dataList!=[], "data file not found!"
  cellStruct = hf.np_smart_load(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  # get expInd, load rvcFits [if existing]
  expInd, expName = hf.get_exp_ind(data_loc, dataList['unitName'][cell_num-1]);
  print('Making DoG fits for cell %d in %s [%s]\n' % (cell_num,data_loc,expName));
  rvcFits = hf.get_rvc_fits(data_loc, expInd, cell_num); # see default arguments in helper_fcns.py

  if DoGmodel == 1:
    modStr = 'sach';
  elif DoGmodel == 2:
    modStr = 'tony';
  fLname  = hf.descrFit_name(loss_type, modelName=modStr);
  if os.path.isfile(data_loc + fLname):
      descrFits = hf.np_smart_load(data_loc + fLname);
  else:
      descrFits = dict();

  # now, get the spikes (adjusted, if needed) and organize for fitting
  spks = hf.get_spikes(data, rvcFits, expInd);
  _, _, resps_mean, resps_all = hf.organize_resp(spks, cellStruct, expInd);
  resps_sem = sem(resps_all, axis=-1, nan_policy='omit');
  
  print('Doing the work, now');

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, _, _ = hf.tabulate_responses(data, expInd);
  all_disps = stimVals[0];
  all_cons = stimVals[1];
  all_sfs = stimVals[2];

  nDisps = len(all_disps);
  nCons = len(all_cons);

  if cell_num-1 in descrFits:
    bestNLL = descrFits[cell_num-1]['NLL'];
    currParams = descrFits[cell_num-1]['params'];
    varExpl = descrFits[cell_num-1]['varExpl'];
    prefSf = descrFits[cell_num-1]['prefSf'];
    charFreq = descrFits[cell_num-1]['charFreq'];
  else: # set values to NaN...
    bestNLL = np.ones((nDisps, nCons)) * np.nan;
    currParams = np.ones((nDisps, nCons, nParam)) * np.nan;
    varExpl = np.ones((nDisps, nCons)) * np.nan;
    prefSf = np.ones((nDisps, nCons)) * np.nan;
    charFreq = np.ones((nDisps, nCons)) * np.nan;

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

  for d in range(nDisps): # TODO: check nDisps is ok (and not range(1))
    for con in range(nCons):
      if con not in valConByDisp[d]:
        continue;

      valSfInds = hf.get_valid_sfs(data, d, con, expInd);
      valSfVals = all_sfs[valSfInds];

      print('.');
      respConInd = np.where(np.asarray(valConByDisp[d]) == con)[0];
      resps_curr = resps_mean[d, valSfInds, con];
      sem_curr   = resps_sem[d, valSfInds, con];
      maxResp       = np.max(resps_curr);
      freqAtMaxResp = all_sfs[np.argmax(resps_curr)];

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
        obj = lambda params: DoG_loss(params, resps_curr, valSfVals, resps_std=sem_curr, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg);
        wax = opt.minimize(obj, init_params, method=methodStr, bounds=allBounds);

        # compare
        NLL = wax['fun'];
        params = wax['x'];

        if np.isnan(bestNLL[d, con]) or NLL < bestNLL[d, con]:
          bestNLL[d, con] = NLL;
          currParams[d, con, :] = params;
          varExpl[d, con] = hf.var_explained(resps_curr, params, valSfVals, DoGmodel);
          prefSf[d, con] = hf.dog_prefSf(params, DoGmodel, valSfVals);
          charFreq[d, con] = hf.dog_charFreq(params, DoGmodel);

    # update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(data_loc + fLname):
      print('reloading descrFits...');
      descrFits = hf.np_smart_load(data_loc + fLname);
    if cell_num-1 not in descrFits:
      descrFits[cell_num-1] = dict();
    descrFits[cell_num-1]['NLL'] = bestNLL;
    descrFits[cell_num-1]['params'] = currParams;
    descrFits[cell_num-1]['varExpl'] = varExpl;
    descrFits[cell_num-1]['prefSf'] = prefSf;
    descrFits[cell_num-1]['charFreq'] = charFreq;
    descrFits[cell_num-1]['gainRegFactor'] = gain_reg;

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
    descr_fits = int(sys.argv[6]);
    dog_model  = int(sys.argv[7]);
    if len(sys.argv) > 8:
      dir = float(sys.argv[8]);
    else:
      dir = None;
    if len(sys.argv) > 9:
      gainReg = float(sys.argv[9]);
    else:
      gainReg = 0;
    print('Running cell %d' % cell_num);

    # get the full data directory
    dataPath = basePath + data_dir + data_suff;
    # get the expInd
    dL = hf.np_smart_load(dataPath + 'dataList.npy');
    unitName = dL['unitName'][cell_num-1];
    expInd = hf.get_exp_ind(dataPath, unitName)[0];

    # then, put what to run here...
    if dir == None:
      if ph_fits == 1:
        phase_advance_fit(cell_num, data_loc=dataPath, expInd=expInd, disp=disp);
      if rvc_fits == 1:
        rvc_adjusted_fit(cell_num, data_loc=dataPath, disp=disp);
      if descr_fits == 1:
        fit_descr_DoG(cell_num, data_loc=dataPath, gain_reg=gainReg, DoGmodel=dog_model);
    else:
      if ph_fits == 1:
        phase_advance_fit(cell_num, data_loc=dataPath, expInd=expInd, disp=disp, dir=dir);
      if rvc_fits == 1:
        rvc_adjusted_fit(cell_num, data_loc=dataPath, disp=disp, dir=dir);
      if descr_fits == 1:
        fit_descr_DoG(cell_num, data_loc=dataPath, gain_reg=gainReg, dir=dir, DoGmodel=dog_model);
