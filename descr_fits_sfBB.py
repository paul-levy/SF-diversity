import numpy as np
import sys
import helper_fcns as hf
import helper_fcns_sfBB as hf_sf
import os
from time import sleep
from scipy.stats import sem, poisson
import warnings
import pdb

basePath = os.getcwd() + '/';
data_suff = 'V1_BB/structures/';

expName = hf.get_datalist('V1_BB/');

sfName = 'descrFits_210721';
rvcName = 'rvcFits_210721';

def make_descr_fits(cellNum, data_path=basePath+data_suff, fit_rvc=1, fit_sf=1, rvcMod=1, sfMod=0, loss_type=2, vecF1=1, onsetCurr=None, rvcName=rvcName, sfName=sfName, jointSf=False, toSave=1, fracSig=1):
  ''' Separate fits for DC, F1 
      -- for DC: [maskOnly, mask+base]
      -- for F1: [maskOnly, mask+base {@mask TF}] 
      For asMulti fits (i.e. when done in parallel) we do the following to reduce multiple loading of files/race conditions
      --- we'll pass in the previous fits as fit_rvc and/or fit_sf
      --- we'll pass in [cellNum, cellName] as cellNum
  '''
  
  expName = 'sfBB_core';

  if not isinstance(cellNum, int):
    cellNum, unitNm = cellNum;
    print('cell %d {%s}' % (cellNum, unitNm));
  else:
    dlName = hf.get_datalist('V1_BB/');
    dataList = hf.np_smart_load(data_path + dlName);
    unitNm = dataList['unitName'][cellNum-1];
  print('loading cell');
  cell = hf.np_smart_load('%s%s_sfBB.npy' % (data_path, unitNm));
  expInfo = cell[expName]
  byTrial = expInfo['trial'];

  if fit_rvc == 1 or fit_rvc is not None: # load existing rvcFits, if there
    rvcNameFinal = hf.rvc_fit_name(rvcName, rvcMod, None, vecF1);
    if fit_rvc == 1:
      if os.path.isfile(data_path + rvcNameFinal):
        rvcFits = hf.np_smart_load(data_path + rvcNameFinal);
    else: # otherwise, we have passed it in as fit_sf to avoid race condition during multiprocessing (i.e. multiple threads trying to load the same file)
      rvcFits = fit_rvc;
    try:
      rvcFits_curr = rvcFits[cellNum-1];
    except:
      rvcFits_curr = None;

  if fit_sf == 1 or fit_sf is not None:
    modStr = hf.descrMod_name(sfMod);
    sfNameFinal = hf.descrFit_name(loss_type, descrBase=sfName, modelName=modStr); # descrLoss order is lsq/sqrt/poiss/sach
    if fit_sf == 1:
      if os.path.isfile(data_path + sfNameFinal):
        sfFits = hf.np_smart_load(data_path + sfNameFinal);
    else: # otherwise, we have passed it in as fit_sf to avoid race condition during multiprocessing (i.e. multiple threads trying to load the same file)
      sfFits = fit_sf;
    try:
      sfFits_curr = sfFits[cellNum-1];
      print('---prev sf!');
    except:
      sfFits_curr = None;
      print('---NO PREV sf!');
  
  #########
  ### Get the responses - base only, mask+base [base F1], mask only (mask F1)
  ### _____ MAKE THIS A FUNCTION???
  #########
  # 1. Get the mask only response (f1 at mask TF)
  respMatrixDC_onlyMask, respMatrixF1_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, vecCorrectedF1=vecF1, onsetTransient=onsetCurr); # i.e. get the maskONLY response
  # 2f1. get the mask+base response (but f1 at mask TF)
  _, respMatrixF1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, vecCorrectedF1=vecF1, onsetTransient=onsetCurr); # i.e. get the maskONLY response
  # 2dc. get the mask+base response (f1 at base TF)
  respMatrixDC, _ = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, vecCorrectedF1=vecF1, onsetTransient=onsetCurr); # i.e. get the base response for F1

  # -- if vecF1, let's just take the "r" elements, not the phi information
  if vecF1:
    respMatrixF1_onlyMask = respMatrixF1_onlyMask[:,:,0,:]; # just take the "r" information (throw away the phi)
    respMatrixF1_maskTf = respMatrixF1_maskTf[:,:,0,:]; # just take the "r" information (throw away the phi)

  # pre-define curr_rvc, curr_sf as dictionaries, in case we aren't fitting
  curr_rvc = dict();
  curr_sf = dict();
  
  for measure in [0,1]:
    if measure == 0:
      baseline = expInfo['blank']['mean'];
      mask_only = respMatrixDC_onlyMask;
      mask_base = respMatrixDC;
      fix_baseline = False
    elif measure == 1:
      baseline = 0;
      mask_only = respMatrixF1_onlyMask;
      mask_base = respMatrixF1_maskTf;
      fix_baseline = True;
    resp_str = hf_sf.get_resp_str(respMeasure=measure);

    whichResp = [mask_only, mask_base];
    whichKey = ['mask', 'both'];

    curr_rvc[resp_str] = dict();
    curr_sf[resp_str] = dict();

    if fit_rvc == 1 or fit_rvc is not None:
      ''' Fit RVCs responses (see helper_fcns.rvc_fit for details) for:
          --- F0: mask alone (7 sfs)
                  mask + base together (7 sfs)
          --- F1: mask alone (7 sfs; at maskTf)
                  mask+ + base together (7 sfs; again, at maskTf)
          NOTE: Assumes only sfBB_core
      '''
      cons = expInfo['maskCon'];
      # first, mask only; then mask+base
      for wR, wK in zip(whichResp, whichKey):
        adjMeans = np.transpose(wR[:,:,0]); # just the means
        consRepeat = [cons] * len(adjMeans);
        try:
          rvcFit_curr = rvcFits_curr[resp_str][wK];
        except:
          rvcFit_curr = None
        _, all_opts, all_conGains, all_loss = hf.rvc_fit(adjMeans, consRepeat, var=None, mod=rvcMod, fix_baseline=fix_baseline, prevFits=rvcFit_curr);

        curr_rvc[resp_str][wK] = dict();
        curr_rvc[resp_str][wK]['loss'] = all_loss;
        curr_rvc[resp_str][wK]['params'] = all_opts;
        curr_rvc[resp_str][wK]['conGain'] = all_conGains;
        curr_rvc[resp_str][wK]['adjMeans'] = adjMeans;
        # compute variance explained!
        varExpl = [hf.var_explained(hf.nan_rm(dat), hf.nan_rm(hf.get_rvcResp(prms, cons, rvcMod)), None) for dat, prms in zip(adjMeans, all_opts)];
        curr_rvc[resp_str][wK]['varExpl'] = adjMeans;
        # END of rvc fit

    if fit_sf == 1 or fit_sf is not None:
      ''' Fit SF tuning responses (see helper_fcns.dog_fit for details) for:
          --- F0: mask alone (7 cons)
                  mask + base together (7 cons)
          --- F1: mask alone (7 cons; at maskTf)
                  mask+ + base together (7 cons; again, at maskTf)
          NOTE: Assumes only sfBB_core
      '''
      cons, sfs = expInfo['maskCon'], expInfo['maskSF']
      stimVals = [[0], cons, sfs];
      valConByDisp = [np.arange(0,len(cons))]; # all cons are valid in sfBB experiment

      for wR, wK in zip(whichResp, whichKey):
        try:
          sfFit_curr = sfFits_curr[resp_str][wK];
        except:
          sfFit_curr = None

        # -- by default, loss_type=2 (meaning sqrt loss); why expand dims and transpose? dog fits assumes the data is in [disp,sf,con] and we just have [con,sf]
        nll, prms, vExp, pSf, cFreq, totNLL, totPrm = hf.dog_fit([np.expand_dims(np.transpose(wR[:,:,0]), axis=0), None, np.expand_dims(np.transpose(wR[:,:,1]), axis=0), baseline], sfMod, loss_type=2, disp=0, expInd=None, stimVals=stimVals, validByStimVal=None, valConByDisp=valConByDisp, prevFits=sfFit_curr, noDisp=1, fracSig=fracSig, n_repeats=50) # noDisp=1 means that we don't index dispersion when accessins prevFits

        curr_sf[resp_str][wK] = dict();
        curr_sf[resp_str][wK]['NLL'] = nll;
        curr_sf[resp_str][wK]['params'] = prms;
        curr_sf[resp_str][wK]['varExpl'] = vExp;
        curr_sf[resp_str][wK]['prefSf'] = pSf;
        curr_sf[resp_str][wK]['charFreq'] = cFreq;
        if jointSf==True:
          curr_sf[resp_str][wK]['totalNLL'] = totNLL;
          curr_sf[resp_str][wK]['paramList'] = totPrm;
        # END of sf fit

  ###########
  # NOW, save (if saving)
  ###########
  # if we are saving, save; otherwise, return the curr_rvc, curr_sf fits
  if toSave:
    if fit_rvc:
      # load fits again in case some other run has saved/made changes
      if os.path.isfile(data_path + rvcNameFinal):
        print('reloading rvcFits...');
        rvcFits = hf.np_smart_load(data_path + rvcNameFinal);
      if cellNum-1 not in rvcFits:
        rvcFits[cellNum-1] = dict();

      # now save
      rvcFits[cellNum-1][resp_str][wK] = curr_rvc
      np.save(data_path+rvcNameFinal, rvcFits);
      print('Saving %s, %s @ %s' % (resp_str, wK, rvcNameFinal))

    if fit_sf:
      # load fits again in case some other run has saved/made changes
      if os.path.isfile(data_path + sfNameFinal):
        print('reloading sfFits...');
        sfFits = hf.np_smart_load(data_path + sfNameFinal);
      if cellNum-1 not in sfFits:
        sfFits[cellNum-1] = dict();

      sfFits[cellNum-1] = curr_sf
          
      # now save
      np.save(data_path + sfNameFinal, sfFits);
      print('Saving %s, %s @ %s' % (resp_str, wK, sfNameFinal))

    ### End of saving (both RVC and SF)
  else:
    return curr_rvc, curr_sf

if __name__ == '__main__':

  if len(sys.argv) < 3:
    print('uhoh...you need at least 3 arguments here');
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
  fit_rvc    = int(sys.argv[2]);
  fit_sf     = int(sys.argv[3]);
  rvc_mod    = int(sys.argv[4]);
  sf_mod     = int(sys.argv[5]);
  loss_type  = int(sys.argv[6]); # default will be 2 (i.e. sqrt)

  fracSig = 1; # why fracSig =1? For V1 fits, we want to contrasin the upper-half sigma of the two-half gaussian as a fraction of the lower half

  if asMulti:
    from functools import partial
    import multiprocessing as mp
    nCpu = mp.cpu_count();

    # to avoid race conditions, load the previous fits beforehand; and the datalist
    rvcNameFinal = hf.rvc_fit_name(rvcName, rvc_mod, None, vecF1=1); # DEFAULT is vecF1 adjustment
    modStr = hf.descrMod_name(sf_mod);
    sfNameFinal = hf.descrFit_name(loss_type, descrBase=sfName, modelName=modStr); # descrLoss order is lsq/sqrt/poiss/sach

    pass_rvc = hf.np_smart_load('%s%s%s' % (basePath, data_suff, rvcNameFinal)) if fit_rvc else None;
    pass_sf = hf.np_smart_load('%s%s%s' % (basePath, data_suff, sfNameFinal)) if fit_sf else None;

    dataList = hf.np_smart_load('%s%s%s' % (basePath, data_suff, hf.get_datalist('V1_BB/')));

    #make_descr_fits((10, dataList['unitName'][10]), fit_rvc=pass_rvc, fit_sf=pass_sf, rvcMod=rvc_mod, sfMod=sf_mod, toSave=0, fracSig=fracSig)

    with mp.Pool(processes = nCpu) as pool:
      # if we're doing as parallel, do NOT save

      fit_perCell = partial(make_descr_fits, fit_rvc=pass_rvc, fit_sf=pass_sf, rvcMod=rvc_mod, sfMod=sf_mod, toSave=0, fracSig=fracSig); 
      fits = zip(*pool.map(fit_perCell, zip(range(start_cell, end_cell+1), dataList['unitName'])));
      rvc_fits, sf_fits = fits; # unpack

      ### do the saving HERE!
      dataPath = basePath+data_suff;
      # --- RVC
      if fit_rvc:
        rvcNameFinal = hf.rvc_fit_name(rvcName, rvc_mod, None, vecF1=1);
        if os.path.isfile(dataPath + rvcNameFinal):
          print('reloading rvcFits...');
          rvcFits = hf.np_smart_load(dataPath + rvcNameFinal);
        else:
          rvcFits = dict();
        for iii, rvcFit in enumerate(rvc_fits):
          rvcFits[iii] = rvcFit;
        np.save(dataPath + rvcNameFinal, rvcFits);
      # --- SF
      if fit_sf:
        modStr = hf.descrMod_name(sf_mod); # we pass this sf_mod argument on the command line
        sfNameFinal = hf.descrFit_name(lossType=2, descrBase=sfName, modelName=modStr); # descrLoss order is lsq/sqrt/poiss/sach
        if os.path.isfile(dataPath + sfNameFinal):
          print('reloading sfFits...');
          sfFits = hf.np_smart_load(dataPath + sfNameFinal);
        else:
          sfFits = dict();
        for iii, sfFit in enumerate(sf_fits):
          sfFits[iii] = sfFit;
        np.save(dataPath + sfNameFinal, sfFits);

  else:
    make_descr_fits(cell_num, fit_rvc=fit_rvc, fit_sf=fit_sf, rvcMod=rvc_mod, sfMod=sf_mod);

