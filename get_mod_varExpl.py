# coding: utf-8

import os
import sys
import numpy as np
import itertools
from functools import partial
import multiprocessing as mp

import helper_fcns as hf
import helper_fcns_sfBB as hf_sf
import model_responses_pytorch as mrpt

import warnings
warnings.filterwarnings('once');

import pdb

# using fits where the filter sigma is sigmoid?
_sigmoidSigma = 5; # put a value (5, as of 21.03.10) or None (see model_responses_pytorch.py for details)
_applyLGNtoNorm = 1;
recenter_norm = 0;

f1_expCutoff = 2; # if 1, then all but V1_orig/ are allowed to have F1; if 2, then altExp/ is also excluded

force_full = 1;

def get_mod_varExpl(cellNum, expDir, normType, lgnFrontEnd, rvcAdj, whichKfold=None, lgnConType=1, excType=1, lossType=1, rvcMod=1, fixRespExp=2, hpcFit=1, schedule=False, rvcDir=1, vecF1=0, _applyLGNtoNorm=True, _newMethod=True, vecCorrected_bb=1):
  ''' Compute the explained variance for the computational model of SF tuning (i.e. model_responses_pytorch)
      --- Computes varExpl on the basis of averaged responses per condition
  '''
  loc_base = os.getcwd() + '/';
  data_loc = loc_base + expDir + 'structures/';
  isBB = True if expDir=='V1_BB/' else False;
  isCV = False if whichKfold is None else True;

  loc_str = 'HPC' if ('pl1465' in loc_base or hpcFit) else '';

  expName = hf.get_datalist(expDir, force_full=force_full, new_v1=True);
  fitBase = 'fitList%s_pyt_nr230104%s%s' % (loc_str, '_noRE' if fixRespExp is not None else '', '_noSched' if schedule==False else '')

  if rvcAdj == -1:
    vecCorrected = 1;
  else: # should always be here if V1_BB, V1_orig, altExp
    vecCorrected = 0;

  ### RVCFITS
  if expDir == 'V1/':
    rvcBase = 'rvcFits%s_221126' % loc_str; # direc flag & '.npy' are added
  elif expDir == 'V1_BB/':
    rvcBase = None
  else:
    rvcBase = 'rvcFits%s_220928' % loc_str; # direc flag & '.npy' are added

  fitName = hf.fitList_name(fitBase, normType, lossType, lgnFrontEnd, lgnConType, vecCorrected, excType=excType, CV=isCV, lgnForNorm=_applyLGNtoNorm)

  try: # keeping for backwards compatability
    dataList = np.load(str(data_loc + expName), encoding='latin1').item();
  except:
    dataList = hf.np_smart_load(str(data_loc + expName))
  try:
    fitList = hf.np_smart_load(data_loc + fitName);
    if cellNum-1 not in fitList: # need to have this key in there
      return np.nan, [];
  except: # if we can't load the fitList, nothing to do here! return np.nan
    return np.nan, [];

  cellName = dataList['unitName'][cellNum-1];

  expStr = '_sfBB.npy' if isBB else '_sfm.npy'
  try: # keeping for backwards compatability
    expData  = np.load(str(data_loc + cellName + expStr), encoding='latin1').item();
  except:
    expData  = hf.np_smart_load(str(data_loc + cellName + expStr));
  expInd   = hf.exp_name_to_ind(dataList['expType'][cellNum-1]) if not isBB else -1;
  if isBB:
    expName = 'sfBB_core';
    expInfo = expData[expName];
    maskInd, baseInd = hf_sf.get_mask_base_inds();

  if expInd > f1_expCutoff or isBB:
    if isBB:
      f1f0_rat = hf_sf.compute_f1f0(expInfo)[0];
    else:
      f1f0_rat = hf.compute_f1f0(expData['sfm']['exp']['trial'], cellNum, expInd, loc_data=None)[0];
    respMeasure = int(f1f0_rat > 1);
  else:
    respMeasure = 0; # default to DC (since this might be an expt where we can only analyze DC)
  respStr = hf_sf.get_resp_str(respMeasure);
  modFit = fitList[cellNum-1][respStr]['params'];
  if isCV:
    modFit = modFit[whichKfold]

  # ### Organize data & model responses --- differently handled for BB vs. other experiments
  if isBB:
    trInf, resps = mrpt.process_data(expInfo, expInd=expInd, respMeasure=respMeasure);
    val_trials = trInf['num']; # these are the indices of valid, original trials

    #####
    # now get the mask+base response (f1 at base TF)
    #####
    respMatrix = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, vecCorrectedF1=vecCorrected_bb)[respMeasure]; # i.e. get the base response for F1
    # and get the mask only response (f1 at mask TF)
    respMatrix_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, vecCorrectedF1=vecCorrected_bb)[respMeasure]; # i.e. get the maskONLY response
    # and get the mask+base response (but f1 at mask TF)
    respMatrix_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, vecCorrectedF1=vecCorrected_bb)[respMeasure]; # i.e. get the maskONLY response
    if respMeasure == 0: # i.e. dc
      respMatrix_maskTf = None

    # -- if vecCorrected, let's just take the "r" elements, not the phi information
    if vecCorrected_bb and respMeasure==1:
        respMatrix = respMatrix[:,:,0,:]; # just take the "r" information (throw away the phi)
        respMatrix_onlyMask = respMatrix_onlyMask[:,:,0,:]; # just take the "r" information (throw away the phi)
        respMatrix_maskTf = respMatrix_maskTf[:,:,0,:]; # just take the "r" information (throw away the phi)

    resps = [respMatrix_onlyMask, respMatrix, respMatrix_maskTf]; #need to plot data_baseTf for f1
    respMean = np.array(hf.flatten_list([hf.flatten_list(x[:,:,0]) if x is not None else [] for x in resps]));
  else:
    # get the correct, adjusted F1 response
    trialInf = expData['sfm']['exp']['trial'];
    if expInd > f1_expCutoff and respMeasure == 1:
      respOverwrite = hf.adjust_f1_byTrial(trialInf, expInd);
    else:
      respOverwrite = None;
    # ---- DATA - organize data responses, first
    _, stimVals, val_con_by_disp, validByStimVal, _ = hf.tabulate_responses(expData, expInd);
    if rvcAdj >= 0:
      if rvcAdj == 1:
        rvcFlag = '';
        rvcFits = hf.get_rvc_fits(data_loc, expInd, cellNum, rvcName=rvcBase, rvcMod=rvcMod, direc=rvcDir, vecF1=vecF1);
        asRates = False; #True;
        force_dc = False
      elif rvcAdj == 0:
        rvcFlag = '_f0';
        rvcFits = hf.get_rvc_fits(data_loc, expInd, cellNum, rvcName='None');
        asRates = False;
        force_dc = True
      # rvcMod=-1 tells the function call to treat rvcName as the fits, already (we loaded above!)
      spikes_rate, meas = hf.get_adjusted_spikerate(expData['sfm']['exp']['trial'], cellNum, expInd, data_loc, rvcName=rvcFits, rvcMod=-1, descrFitName_f0=None, baseline_sub=False, return_measure=True, force_dc=force_dc);
    elif rvcAdj == -1: # i.e. ignore the phase adjustment stuff...
      if respMeasure == 1 and expInd > f1_expCutoff:
        spikes_byComp = respOverwrite;
        # then, sum up the valid components per stimulus component
        allCons = np.vstack(expData['sfm']['exp']['trial']['con']).transpose();
        blanks = np.where(allCons==0);
        spikes_byComp[blanks] = 0; # just set it to 0 if that component was blank during the trial
        spikes_rate = np.sum(spikes_byComp, axis=1);
        asRates = False; # TODO: Figure out if really as rates or not...
        rvcFlag = '_f1';
      else:
        spikes_rate = hf.get_adjusted_spikerate(expData['sfm']['exp']['trial'], cellNum, expInd, data_loc, rvcName=None, force_dc=True, baseline_sub=False); 
        rvcFlag = '_f0';
        asRates = True;
    # finally, organize into mean per condition!
    _, _, respOrg, respAll = hf.organize_resp(spikes_rate, expData, expInd, respsAsRate=asRates);
    respMean = respOrg;

  ### now, set-up the model
  model = mrpt.sfNormMod(modFit, expInd=expInd, excType=excType, normType=normType, lossType=lossType, newMethod=_newMethod, lgnFrontEnd=lgnFrontEnd, lgnConType=lgnConType, applyLGNtoNorm=_applyLGNtoNorm, toFit=False, normFiltersToOne=False)

  if isBB:
    # getting both responses just to make the below easier...
    resp_dc = model.forward(trInf, respMeasure=0, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm).detach().numpy();
    resp_f1 = model.forward(trInf, respMeasure=1, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm).detach().numpy();

    # ------ note: for all model responses, flag vecCorrectedF1 != 1 so that we make sure to use the passed-in model responses
    # ---- model A responses
    respModel = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, dc_resp=resp_dc, f1_base=resp_f1[:,baseInd], f1_mask=resp_f1[:,maskInd], val_trials=val_trials, vecCorrectedF1=0)[respMeasure]; # i.e. get the base response for F1
    # and get the mask only response (f1 at mask TF)
    respModel_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, dc_resp=resp_dc, f1_base=resp_f1[:,baseInd], f1_mask=resp_f1[:,maskInd], val_trials=val_trials, vecCorrectedF1=0)[respMeasure]; # i.e. get the maskONLY response
    # and get the mask+base response (but f1 at mask TF)
    respModel_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, dc_resp=resp_dc, f1_base=resp_f1[:,baseInd], f1_mask=resp_f1[:,maskInd], val_trials=val_trials, vecCorrectedF1=0)[respMeasure]; # i.e. get the maskONLY response
    if respMeasure == 0: # there isn't a maskTF in this case (cannot separate mask/base in responses)
      respModel_maskTf = None;

    mod_resps = [respModel_onlyMask, respModel, respModel_maskTf];
    modAvgs = np.array(hf.flatten_list([hf.flatten_list(x[:,:,0]) if x is not None else [] for x in mod_resps]));
  else:
    dw = mrpt.dataWrapper(trialInf, respMeasure=respMeasure, expInd=expInd, respOverwrite=respOverwrite);
    modResps_temp = model.forward(dw.trInf, respMeasure=respMeasure, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm).detach().numpy();
    # package the responses into the right trials (the model skips simulating some [blank/orientation] trials)
    if respMeasure == 1: # make sure the blank components have a zero response (we'll do the same with the measured responses)
      blanks = np.where(dw.trInf['con']==0);
      modResps_temp[blanks] = 0;
      # next, sum up across components
      modResps_temp = np.sum(modResps_temp, axis=1)
    # finally, make sure this fills out a vector of all responses (just have nan for non-modelled trials)
    nTrialsFull = len(trialInf['num']);
    modResps = np.nan * np.zeros((nTrialsFull, ));
    modResps[dw.trInf['num']] = modResps_temp;

    # organize responses so that we can package them for evaluating varExpl...
    stimDur = hf.get_exp_params(expInd).stimDur;
    # TODO: This is a work around for which measures are in rates vs. counts (DC vs F1, model vs data...)
    # --- but check V1/1 -- why is varExpl still bad????
    divFactor = stimDur if respMeasure == 0 else 1;

    # now organize the responses
    modresps_org = hf.organize_resp(np.divide(modResps, divFactor), expData, expInd)[3];
    modAvgs = np.nanmean(modresps_org, axis=3)

  nn = np.logical_and(~np.isnan(respMean), ~np.isnan(modAvgs));
  varExpl = hf.var_explained(respMean[nn], modAvgs[nn], None);

  # RETURN THE VAL...we'll save elsewhere
  #print('%s%d: %.1f%% [%s]' % (expDir, cellNum, varExpl, respStr))
  return varExpl, respStr;

if __name__ == '__main__':

  expDir       = sys.argv[1];
  kfold        = int(sys.argv[2]);
  if kfold < 0:
    kfold = None;
    todoCV = False;
  else:
    todoCV = True

  ##########
  # EDIT HERE
  ##########
  fitTypes = [1,2];
  lgnTypes = [0,1,4];
  # --- unlikely to edit
  fixRespExp = 2;
  schedule = False
  lossType = 1;
  excType = 1;
  _LGNforNorm = 1
  lgnConType = 1
  hpcStr = 'HPC'; # are we doing HPC fits or no?
  vecCorrected = 0;
  if expDir=='V1/' or expDir=='V1_BB/':
    rvcAdj = 1
  else:
    rvcAdj = 0;
  ##########
  ### END OF....EDIT HERE
  ##########

  fitBase = 'fitList%s_pyt_nr230104%s%s' % (hpcStr, '_noRE' if fixRespExp is not None else '', '_noSched' if schedule==False else '')

  dataListName = hf.get_datalist(expDir, force_full=force_full, new_v1=True); # argv[2] is expDir
  nCpu = 20; # mp.cpu_count()-1; # heuristics say you should reqeuest at least one fewer processes than their are CPU

  loc_base = os.getcwd() + '/'; # ensure there is a "/" after the final directory
  loc_data = loc_base + expDir + 'structures/';
  dataList = hf.np_smart_load(str(loc_data + dataListName));
  dataNames = dataList['unitName'];
  len_to_use = len(dataNames);
  cellNums = np.arange(1, 1+len_to_use);

  for fitType,lgnFrontOn in itertools.product(fitTypes, lgnTypes):
    perCell = partial(get_mod_varExpl, expDir=expDir, normType=fitType, lgnFrontEnd=lgnFrontOn, rvcAdj=rvcAdj, whichKfold=kfold, fixRespExp=fixRespExp, schedule=schedule, lgnConType=lgnConType, excType=excType, lossType=lossType);
    with mp.Pool(processes = nCpu) as pool:
      vExp_perCell = pool.map(perCell, cellNums); # use starmap if you to pass in multiple args
      pool.close();

    fitListName = hf.fitList_name(base=fitBase, fitType=fitType, lossType=lossType, lgnType=lgnFrontOn, lgnConType=lgnConType, vecCorrected=vecCorrected, CV=todoCV, excType=excType, lgnForNorm=_LGNforNorm)
    fitListNPY = hf.np_smart_load(loc_data + fitListName);

    for iii, (vExp, respStr) in enumerate(vExp_perCell):
      if vExp != np.nan and respStr != []:
        if todoCV:
          if 'varExpl_func' not in fitListNPY[iii][respStr]: # then create the vector for varExpl...
            nFold = len(fitListNPY[iii][respStr]['NLL_train']);
            fitListNPY[iii][respStr]['varExpl_func'] = np.nan * np.zeros((nFold,));
          fitListNPY[iii][respStr]['varExpl_func'][kfold] = vExp;
        else:
          fitListNPY[iii][respStr]['varExpl_func'] = vExp;
    # --- finally, save
    np.save(loc_data + fitListName, fitListNPY)


