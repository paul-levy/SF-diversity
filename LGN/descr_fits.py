import numpy as np
import sys
import scipy.optimize as opt
import os
from time import sleep
from scipy.stats import sem
import warnings
import pdb

import sys
sys.path.insert(0, '../');
import helper_fcns as hf

# LCV/CNS
dataPath = '/users/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/LGN/structures/';
save_loc = '/users/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/LGN/structures/';
# personal mac
# dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/structures/';
# save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/structures/';
# prince cluster
#dataPath = '/home/pl1465/SF_diversity/LGN/analysis/structures/';
#save_loc = '/home/pl1465/SF_diversity/LGN/analysis/structures/';

expName   = 'dataList.npy'
phAdvName = 'phaseAdvanceFits'
rvcName   = 'rvcFits'
dogName   =  'descrFits_181102';
expInd    = 3;

def flatten(l):
  flatten = lambda l: [item for sublist in l for item in sublist];
  return flatten(l);

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

def phase_advance_fit(cell_num, data_loc = dataPath, phAdvName=phAdvName, to_save = 1, disp=0, dir=-1):
  ''' Given the FFT-derived response amplitude and phase, determine the response phase relative
      to the stimulus by taking into account the stimulus phase. 
      Then, make a simple linear model fit (line + constant offset) of the response phase as a function
      of response amplitude.
      vSAVES loss/optimized parameters/and phase advance (if default "to_save" value is kept)
      RETURNS phAdv_model, all_opts

      Do ONLY for single gratings
  '''

  dataList = hf.np_smart_load(data_loc + 'dataList.npy');
  cellStruct = hf.np_smart_load(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  phAdvName = hf.fit_name(phAdvName, dir);

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, _, _ = hf.tabulate_responses(data, expInd=expInd);
  allCons = stimVals[1];
  allSfs = stimVals[2];

  # for all con/sf values for this dispersion, compute the mean amplitude/phase per condition
  allAmp, allPhi, allTf, _, _ = hf.get_all_fft(data, disp, dir=dir); 
     
  # now, compute the phase advance
  conInds = valConByDisp[disp];
  conVals = allCons[conInds];
  nConds = len(allAmp); # this is how many conditions are present for this dispersion
  # recall that nConds = nCons * nSfs
  allCons = [conVals] * nConds; # repeats list and nests
  phAdv_model, all_opts, all_phAdv, all_loss = hf.phase_advance(allAmp, allPhi, conVals, allTf);

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

def rvc_adjusted_fit(cell_num, data_loc=dataPath, rvcName=rvcName, to_save=1, disp=0, dir=-1):
  ''' Piggy-backing off of phase_advance_fit above, get prepare to project the responses onto the proper phase to get the correct amplitude
      Then, with the corrected response amplitudes, fit the RVC model
  '''
  dataList = hf.np_smart_load(data_loc + 'dataList.npy');
  cellStruct = hf.np_smart_load(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  rvcNameFinal = hf.fit_name(rvcName, dir);

  # first, get the set of stimulus values:
  _, stimVals, valConByDisp, _, _ = hf.tabulate_responses(data, expInd=expInd);
  allCons = stimVals[1];
  allSfs = stimVals[2];
  valCons = allCons[valConByDisp[disp]];

  # calling phase_advance fit, use the phAdv_model and optimized paramters to compute the true response amplitude
  # given the measured/observed amplitude and phase of the response
  # NOTE: We always call phase_advance_fit with disp=0 (default), since we don't make a fit
  # for the mixtrue stimuli - instead, we use the fits made on single gratings to project the
  # individual-component-in-mixture responses
  phAdv_model, all_opts = phase_advance_fit(cell_num, dir=dir, to_save = 0); # don't save
  allAmp, allPhi, _, allCompCon, allCompSf = hf.get_all_fft(data, disp, dir=dir, all_trials=1);
  # get just the mean amp/phi and put into convenient lists
  allAmpMeans = [[x[0] for x in sf] for sf in allAmp]; # mean is in the first element; do that for each [mean, std] pair in each list (split by sf)
  allAmpTrials = [[x[2] for x in sf] for sf in allAmp]; # trial-by-trial is third element 

  allPhiMeans = [[x[0] for x in sf] for sf in allPhi]; # mean is in the first element; do that for each [mean, var] pair in each list (split by sf)
  allPhiTrials = [[x[2] for x in sf] for sf in allPhi]; # trial-by-trial is third element 

  adjMeans   = hf.project_resp(allAmpMeans, allPhiMeans, phAdv_model, all_opts, disp, allCompSf, allSfs);
  adjByTrial = hf.project_resp(allAmpTrials, allPhiTrials, phAdv_model, all_opts, disp, allCompSf, allSfs);
  consRepeat = [valCons] * len(adjMeans);
  
  if disp == 1: # then we need to sum component responses and get overall std measure (we'll fit to sum, not indiv. comp responses!)
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

### 2: Difference of gaussian fit to (adjusted) responses - i.e. F1 as a function of spatial frequency

def invalid(params, bounds):
# given parameters and bounds, are the parameters valid?
  for p in range(len(params)):
    if params[p] < bounds[p][0] or params[p] > bounds[p][1]:
      return True;
  return False;

def DoG_loss(params, resps, sfs, loss_type = 3, DoGmodel=1, dir=-1, resps_std=None, gain_reg = 0):
  '''Given the model params (i.e. sach or tony formulation)), the responses, sf values
  return the loss
  loss_type: 1 - poisson
             2 - sqrt
             3 - Sach sum{[(exp-obs)^2]/[k+sigma^2]} where
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
  if loss_type == 1:
    # poisson model of spiking
    poiss = poisson.pmf(np.round(resps), pred_spikes); # round since the values are nearly but not quite integer values (Sach artifact?)...
    ps = np.sum(poiss == 0);
    if ps > 0:
      poiss = np.maximum(poiss, 1e-6); # anything, just so we avoid log(0)
    loss = loss + sum(-np.log(poiss));
  elif loss_type == 2:
    loss = np.square(np.sqrt(resps) - np.sqrt(pred_spikes));
    loss = loss + loss;
  elif loss_type == 3:
    k = 0.01*np.max(resps);
    if resps_std is None:
      sigma = np.ones_like(resps);
    else:
      sigma = resps_std;
    sq_err = np.square(resps-pred_spikes);
    loss = loss + np.sum((sq_err/(k+np.square(sigma)))) + gain_reg*(params[0] + params[2]); # regularize - want gains as low as possible
  return loss;

def fit_descr_DoG(cell_num, data_loc=dataPath, n_repeats=1000, loss_type=3, DoGmodel=1, disp=0, rvcName=rvcName, dir=-1, gain_reg=0, fLname = dogName):

  nParam = 4;

  # load cell information
  dataList = hf.np_smart_load(data_loc + 'dataList.npy');
  assert dataList!=[], "data file not found!"

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

  cellStruct = hf.np_smart_load(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy');
  data = cellStruct['sfm']['exp']['trial'];
  rvcNameFinal = hf.phase_fit_name(rvcName, dir);
  rvcFits = hf.np_smart_load(data_loc + rvcNameFinal);
  adjResps = rvcFits[cell_num-1][disp]['adjMeans'];
  adjSem = rvcFits[cell_num-1][disp]['adjSem'];
  if 'adjByTr' in rvcFits[cell_num-1][disp]:
    adjByTr = rvcFits[cell_num-1][disp]['adjByTr'];
  if disp == 1:
    adjResps = [np.sum(x, 1) if x else [] for x in adjResps];
    if adjByTr:
      adjByTr = [np.sum(x, 1) if x else [] for x in adjByTr];
  adjResps = np.array(adjResps); # indexing multiple SFs will work only if we convert to numpy array first
  adjSem = np.array([np.array(x) for x in adjSem]); # make each inner list an array, and the whole thing an array
  
  print('Doing the work, now');

  # first, get the set of stimulus values:
  resps, stimVals, valConByDisp, _, _ = hf.tabulate_responses(data, expInd=expInd); # LGN is expInd=3
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

  for d in range(1): # should be nDisps - just setting to 1 for now (i.e. fitting single gratings and mixtures separately)
    for con in range(nCons):
      if con not in valConByDisp[disp]:
        continue;

      valSfInds = hf.get_valid_sfs(data, disp, con, expInd);
      valSfVals = all_sfs[valSfInds];

      print('.');
      # adjResponses (f1) in the rvcFits are separate by sf, values within contrast - so to get all responses for a given SF, 
      # access all sfs and get the specific contrast response
      respConInd = np.where(np.asarray(valConByDisp[disp]) == con)[0];
      pdb.set_trace();
      ### interlude...
      spks = hf.get_spikes(data, rvcFits=rvcFits[cell_num-1], expInd=expInd);
      _, _, mnResp, alResp =hf.organize_resp(spks, data, expInd);
      ###
      resps = flatten([x[respConInd] for x in adjResps[valSfInds]]);
      resps_sem = [x[respConInd] for x in adjSem[valSfInds]];
      if isinstance(resps_sem[0], np.ndarray): # i.e. if it's still array of arrays...
        resps_sem = flatten(resps_sem);
      #resps_sem = None;
      maxResp = np.max(resps);
      freqAtMaxResp = all_sfs[np.argmax(resps)];

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

        obj = lambda params: DoG_loss(params, resps, valSfVals, resps_std=resps_sem, loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, gain_reg=gain_reg);
        wax = opt.minimize(obj, init_params, method=methodStr, bounds=allBounds);

        # compare
        NLL = wax['fun'];
        params = wax['x'];

        if np.isnan(bestNLL[disp, con]) or NLL < bestNLL[disp, con]:
          bestNLL[disp, con] = NLL;
          currParams[disp, con, :] = params;
          varExpl[disp, con] = hf.var_explained(resps, params, valSfVals);
          prefSf[disp, con] = hf.dog_prefSf(params, DoGmodel, valSfVals);
          charFreq[disp, con] = hf.dog_charFreq(params, DoGmodel);

    # update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(fLname):
      print('reloading descrFits...');
      descrFits = hf.np_smart_load(fLname);
    if cell_num-1 not in descrFits:
      descrFits[cell_num-1] = dict();
    descrFits[cell_num-1]['NLL'] = bestNLL;
    descrFits[cell_num-1]['params'] = currParams;
    descrFits[cell_num-1]['varExpl'] = varExpl;
    descrFits[cell_num-1]['prefSf'] = prefSf;
    descrFits[cell_num-1]['charFreq'] = charFreq;
    descrFits[cell_num-1]['gainRegFactor'] = gain_reg;

    np.save(fLname, descrFits);
    print('saving for cell ' + str(cell_num));

### Fin: Run the stuff!

if __name__ == '__main__':

    if len(sys.argv) < 2:
      print('uhoh...you need at least one argument(s) here');
      exit();

    cell_num = int(sys.argv[1]);
    disp = int(sys.argv[2]);
    ph_fits = int(sys.argv[3]);
    rvc_fits = int(sys.argv[4]);
    descr_fits = int(sys.argv[5]);
    dog_model = int(sys.argv[6]);
    if len(sys.argv) > 7:
      dir = float(sys.argv[7]);
    else:
      dir = None;
    if len(sys.argv) > 8:
      gainReg = float(sys.argv[8]);
    else:
      gainReg = 0;
    print('Running cell %d' % cell_num);

    # then, put what to run here...
    if dir == None:
      if ph_fits == 1:
        phase_advance_fit(cell_num, disp=disp);
      if rvc_fits == 1:
        rvc_adjusted_fit(cell_num, disp=disp);
      if descr_fits == 1:
        fit_descr_DoG(cell_num, gain_reg=gainReg, disp=disp, DoGmodel=dog_model);
    else:
      if ph_fits == 1:
        phase_advance_fit(cell_num, disp=disp, dir=dir);
      if rvc_fits == 1:
        rvc_adjusted_fit(cell_num, disp=disp, dir=dir);
      if descr_fits == 1:
        fit_descr_DoG(cell_num, gain_reg=gainReg, disp=disp, dir=dir, DoGmodel=dog_model);
