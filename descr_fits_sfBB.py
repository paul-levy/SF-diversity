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

sfName = 'descrFits_210304';
rvcName = 'rvcFits_210304';

def make_descr_fits(cellNum, data_path=basePath+data_suff, fit_rvc=1, fit_sf=1, rvcMod=1, sfMod=0, loss_type=2, vecF1=1, onsetCurr=None, rvcName=rvcName, sfName=sfName, jointSf=False):
  ''' Separate fits for DC, F1 
      -- for DC: [maskOnly, mask+base]
      -- for F1: [maskOnly, mask+base {@mask TF}] '''
  
  expName = 'sfBB_core';

  dlName = hf.get_datalist('V1_BB/');
  dataList = hf.np_smart_load(data_path + dlName);
  unitNm = dataList['unitName'][cellNum-1];
  cell = hf.np_smart_load('%s%s_sfBB.npy' % (data_path, unitNm));
  expInfo = cell[expName]
  byTrial = expInfo['trial'];

  if fit_rvc == 1: # load existing rvcFits, if there
    rvcNameFinal = hf.rvc_fit_name(rvcName, rvcMod, None, vecF1);
    if os.path.isfile(data_path + rvcNameFinal):
      rvcFits = hf.np_smart_load(data_path + rvcNameFinal);
      try:
        rvcFits_curr = rvcFits[cellNum-1];
      except:
        rvcFits_curr = None
    else:
      rvcFits = dict();
      rvcFits_curr = None

  if fit_sf == 1:
    modStr = hf.descrMod_name(sfMod);
    sfNameFinal = hf.descrFit_name(loss_type, descrBase=sfName, modelName=modStr); # descrLoss order is lsq/sqrt/poiss/sach
    if os.path.isfile(data_path + sfNameFinal):
      sfFits = hf.np_smart_load(data_path + sfNameFinal);
      try:
        sfFits_curr = sfFits[cellNum-1];
      except:
        sfFits_curr = None
    else:
      sfFits = dict();
      sfFits_curr = None
  
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

    if fit_rvc == 1:
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

        # load fits again in case some other run has saved/made changes
        if os.path.isfile(data_path + rvcNameFinal):
          print('reloading rvcFits...');
          rvcFits = hf.np_smart_load(data_path + rvcNameFinal);
        if cellNum-1 not in rvcFits:
          rvcFits[cellNum-1] = dict();
          rvcFits[cellNum-1][resp_str] = dict();
          rvcFits[cellNum-1][resp_str][wK] = dict();
        else: # cellNum-1 is a key in rvcFits
          if resp_str not in rvcFits[cellNum-1]:
            rvcFits[cellNum-1][resp_str] = dict();
            rvcFits[cellNum-1][resp_str][wK] = dict();
          elif wK not in rvcFits[cellNum-1][resp_str]:
            rvcFits[cellNum-1][resp_str][wK] = dict();

        # now save
        rvcFits[cellNum-1][resp_str][wK]['loss'] = all_loss;
        rvcFits[cellNum-1][resp_str][wK]['params'] = all_opts;
        rvcFits[cellNum-1][resp_str][wK]['conGain'] = all_conGains;
        rvcFits[cellNum-1][resp_str][wK]['adjMeans'] = adjMeans;
        # compute variance explained!
        varExpl = [hf.var_explained(hf.nan_rm(dat), hf.nan_rm(hf.get_rvcResp(prms, cons, rvcMod)), None) for dat, prms in zip(adjMeans, all_opts)];
        rvcFits[cellNum-1][resp_str][wK]['varExpl'] = varExpl
        np.save(data_path+rvcNameFinal, rvcFits);
        print('Saving %s, %s @ %s' % (resp_str, wK, rvcNameFinal))

    if fit_sf == 1:
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
        nll, prms, vExp, pSf, cFreq, totNLL, totPrm = hf.dog_fit([np.expand_dims(np.transpose(wR[:,:,0]), axis=0), None, np.expand_dims(np.transpose(wR[:,:,1]), axis=0), baseline], sfMod, loss_type=2, disp=0, expInd=None, stimVals=stimVals, validByStimVal=None, valConByDisp=valConByDisp, prevFits=sfFit_curr)

        # load fits again in case some other run has saved/made changes
        if os.path.isfile(data_path + sfNameFinal):
          print('reloading sfFits...');
          sfFits = hf.np_smart_load(data_path + sfNameFinal);
        if cellNum-1 not in sfFits:
          sfFits[cellNum-1] = dict();
          sfFits[cellNum-1][resp_str] = dict();
          sfFits[cellNum-1][resp_str][wK] = dict();
        else: # cellNum-1 is a key in sfFits
          if resp_str not in sfFits[cellNum-1]:
            sfFits[cellNum-1][resp_str] = dict();
            sfFits[cellNum-1][resp_str][wK] = dict();
          else:
            if wK not in sfFits[cellNum-1][resp_str]:
              sfFits[cellNum-1][resp_str][wK] = dict();

        sfFits[cellNum-1][resp_str][wK]['NLL'] = nll;
        sfFits[cellNum-1][resp_str][wK]['params'] = prms;
        sfFits[cellNum-1][resp_str][wK]['varExpl'] = vExp;
        sfFits[cellNum-1][resp_str][wK]['prefSf'] = pSf;
        sfFits[cellNum-1][resp_str][wK]['charFreq'] = cFreq;
        if jointSf==True:
          sfFits[cellNum-1][resp_str][wK]['totalNLL'] = totNLL;
          sfFits[cellNum-1][resp_str][wK]['paramList'] = totPrm;
        np.save(data_path + sfNameFinal, sfFits);
        print('Saving %s, %s @ %s' % (resp_str, wK, sfNameFinal))

if __name__ == '__main__':

  if len(sys.argv) < 3:
    print('uhoh...you need at least 3 arguments here');
    exit();

  cell_num   = int(sys.argv[1]);
  fit_rvc    = int(sys.argv[2]);
  fit_sf     = int(sys.argv[3]);
  rvc_mod    = int(sys.argv[4]);
  sf_mod     = int(sys.argv[5]);

  make_descr_fits(cell_num, fit_rvc=fit_rvc, fit_sf=fit_sf, rvcMod=rvc_mod, sfMod=sf_mod);
