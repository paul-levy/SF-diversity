
import numpy as np
import sys
import helper_fcns as hf
import helper_fcns_sfBB as hf_sf
import os, itertools
from time import sleep
from scipy.stats import sem, poisson
import warnings
import pdb

basePath = os.getcwd() + '/';
data_suff = 'V1_BB/structures/';

if 'pl1465' in basePath:
  hpcSuff = 'HPC';
else:
  hpcSuff = '';

expName = hf.get_datalist('V1_BB/', force_full=1);

sfName = 'descrFits%s_220811vEs' % hpcSuff;
#sfName = 'descrFits%s_220801vEs' % hpcSuff;
rvcName = 'rvcFits%s_220718' % hpcSuff;
#sfName = 'descrFits%s_220609' % hpcSuff;
#rvcName = 'rvcFits%s_220609' % hpcSuff;

def make_descr_fits(cellNum, data_path=basePath+data_suff, fit_rvc=1, fit_sf=1, rvcMod=1, sfMod=0, loss_type=2, vecF1=1, onsetCurr=None, rvcName=rvcName, sfName=sfName, toSave=1, fracSig=1, nBoots=0, n_repeats=25, jointSf=0, resp_thresh=(-1e5,0), veThresh=-1e5, sfModRef=1, phAdvCorr=True, phAdj=True):
  ''' Separate fits for DC, F1 
      -- for DC: [maskOnly, mask+base]
      -- for F1: [maskOnly, mask+base {@mask TF}] 
      For asMulti fits (i.e. when done in parallel) we do the following to reduce multiple loading of files/race conditions
      --- we'll pass in the previous fits as fit_rvc and/or fit_sf
      --- we'll pass in [cellNum, cellName] as cellNum
  '''
  rvcFits_curr_toSave = None; sfFits_curr_toSave = None; # default to None, in case we don't actually do those fits
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
      else:
        rvcFits = dict();
    else: # otherwise, we have passed it in as fit_sf to avoid race condition during multiprocessing (i.e. multiple threads trying to load the same file)
      rvcFits = fit_rvc;
    try:
      rvcFits_curr_toSave = rvcFits[cellNum-1];
    except:
      rvcFits_curr_toSave = dict();

  if fit_sf == 1 or fit_sf is not None:
    modStr = hf.descrMod_name(sfMod);
    sfNameFinal = hf.descrFit_name(loss_type, descrBase=sfName, modelName=modStr, joint=jointSf, phAdj=phAdj); # descrLoss order is lsq/sqrt/poiss/sach
    if fit_sf == 1:
      if os.path.isfile(data_path + sfNameFinal):
        sfFits = hf.np_smart_load(data_path + sfNameFinal);
      else:
        sfFits = dict();
    else: # otherwise, we have passed it in as fit_sf to avoid race condition during multiprocessing (i.e. multiple threads trying to load the same file)
      sfFits = fit_sf;
    try:
      sfFits_curr_toSave = sfFits[cellNum-1];
      print('---prev sf!');
    except:
      sfFits_curr_toSave = dict();
      print('---NO PREV sf!');

  # Set up whether we will bootstrap straight away
  resample = False if nBoots <= 0 else True;
  if cross_val is not None:
    # we also set to True if cross_val is not None
    resample = True;
  nBoots = 1 if nBoots <= 0 else nBoots;
  if nBoots>1:
    ftol = 5e-9;
    #ftol = 1e-6;
  else:
    ftol = 2.220446049250313e-09; # the default value, per scipy guide (scipy.optimize, for L-BFGS-B) 
    
  #########
  ### Get the responses - base only, mask+base [base F1], mask only (mask F1)
  ### _____ MAKE THIS A FUNCTION???
  #########
  # 1. Get the mask only response (f1 at mask TF)
  _, _, gt_respMatrixDC_onlyMask, gt_respMatrixF1_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, vecCorrectedF1=vecF1, onsetTransient=onsetCurr, returnByTr=1, phAdvCorr=phAdvCorr); # i.e. get the maskONLY response
  # 2f1. get the mask+base response (but f1 at mask TF)
  _, _, _, gt_respMatrixF1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, vecCorrectedF1=vecF1, onsetTransient=onsetCurr, returnByTr=1, phAdvCorr=phAdvCorr); # i.e. get the maskONLY response
  # 2dc. get the mask+base response (f1 at base TF)
  _, _, gt_respMatrixDC, _ = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, vecCorrectedF1=vecF1, onsetTransient=onsetCurr, returnByTr=1, phAdvCorr=phAdvCorr); # i.e. get the base response for F1
 
  if cross_val == 2.0:
    nBoots = np.multiply(*gt_respMatrixDC_onlyMask.shape[0:2], );
    print('# boots is %03d' % nBoots);
    resample = False; # then we DO NOT want to resample

  for boot_i in range(nBoots):
    ######
    # 3a. Ensure we have the right responses (incl. resampling, taking mean)
    ######
    # --- Note that all hf_sf.resample_all_cond(resample, arr, axis=X) calls are X=2, because all arrays are [nSf x nCon x respPerTrial x ...]
    # - o. DC responses are easy - simply resample, then take the mean/s.e.m. across all trials of a given condition
    respMatrixDC_onlyMask_resample = hf_sf.resample_all_cond(resample, np.copy(gt_respMatrixDC_onlyMask), axis=2)
    respMatrixDC_onlyMask = np.stack((np.nanmean(respMatrixDC_onlyMask_resample, axis=-1), sem(respMatrixDC_onlyMask_resample, axis=-1, nan_policy='omit')), axis=-1)
    respMatrixDC_resample = hf_sf.resample_all_cond(resample, np.copy(gt_respMatrixDC), axis=2);
    respMatrixDC = np.stack((np.nanmean(respMatrixDC_resample, axis=-1), sem(respMatrixDC_resample, axis=-1, nan_policy='omit')), axis=-1)
    # - F1 responses are different - vector math (i.e. hf.polar_vec_mean call)
    # --- first, resample the data, then do the vector math
    # - i. F1, only mask
    respMatrixF1_onlyMask_resample = hf_sf.resample_all_cond(resample, np.copy(gt_respMatrixF1_onlyMask), axis=2);
    # --- however, polar_vec_mean must be computed by condition, to handle NaN (which might be unequal across conditions):
    respMatrixF1_onlyMask = np.empty(respMatrixF1_onlyMask_resample.shape[0:2] + respMatrixF1_onlyMask_resample.shape[3:]);
    respMatrixF1_onlyMask_phi = np.copy(respMatrixF1_onlyMask);
    for conds in itertools.product(*[range(x) for x in respMatrixF1_onlyMask_resample.shape[0:2]]):
      r_mean, phi_mean, r_sem, phi_sem = hf.polar_vec_mean([hf.nan_rm(respMatrixF1_onlyMask_resample[conds + (slice(None), 0)])], [hf.nan_rm(respMatrixF1_onlyMask_resample[conds + (slice(None), 1)])], sem=1) # return s.e.m. rather than std (default)
      # - and we only care about the R value (after vec. avg.)
      respMatrixF1_onlyMask[conds] = [r_mean[0], r_sem[0]]; # r...[0] to unpack (it's nested inside array, since f is vectorized
      respMatrixF1_onlyMask_phi[conds] = [phi_mean[0], phi_sem[0]];
    if phAdvCorr and vecF1:
      maskCon, maskSf = expInfo['maskCon'], expInfo['maskSF'];
      opt_params, phAdv_model = hf_sf.phase_advance_fit_core(respMatrixF1_onlyMask, respMatrixF1_onlyMask_phi, maskCon, maskSf);
      for msI, mS in enumerate(maskSf):
        curr_params = opt_params[msI]; # the phAdv model applies per-SF
        for mcI, mC in enumerate(maskCon):
          curr_r, curr_phi = respMatrixF1_onlyMask[mcI, msI, 0], respMatrixF1_onlyMask_phi[mcI, msI, 0];
          refPhi = phAdv_model(*curr_params, curr_r);
          new_r = np.multiply(curr_r, np.cos(np.deg2rad(refPhi)-np.deg2rad(curr_phi)));
          respMatrixF1_onlyMask[mcI, msI, 0] = new_r;
      
    # - ii. F1, both (@ maskTF)
    respMatrixF1_maskTf_resample = hf_sf.resample_all_cond(resample, np.copy(gt_respMatrixF1_maskTf), axis=2);
    # --- however, polar_vec_mean must be computed by condition, to handle NaN (which might be unequal across conditions):
    respMatrixF1_maskTf = np.empty(respMatrixF1_maskTf_resample.shape[0:2] + respMatrixF1_maskTf_resample.shape[3:]);
    respMatrixF1_maskTf_phi = np.copy(respMatrixF1_maskTf);
    for conds in itertools.product(*[range(x) for x in respMatrixF1_maskTf_resample.shape[0:2]]):
      r_mean, phi_mean, r_sem, phi_var = hf.polar_vec_mean([hf.nan_rm(respMatrixF1_maskTf_resample[conds + (slice(None), 0)])], [hf.nan_rm(respMatrixF1_maskTf_resample[conds + (slice(None), 1)])], sem=1) # return s.e.m. rather than std (default)
      # - and we only care about the R value (after vec. avg.)
      respMatrixF1_maskTf[conds] = [r_mean[0], r_sem[0]]; # r...[0] is to unpack (it's nested inside array, since func is vectorized
      respMatrixF1_maskTf_phi[conds] = [phi_mean[0], phi_sem[0]];
    if phAdvCorr and vecF1:
      maskCon, maskSf = expInfo['maskCon'], expInfo['maskSF'];
      opt_params, phAdv_model = hf_sf.phase_advance_fit_core(respMatrixF1_maskTf, respMatrixF1_maskTf_phi, maskCon, maskSf);
      for msI, mS in enumerate(maskSf):
        curr_params = opt_params[msI]; # the phAdv model applies per-SF
        for mcI, mC in enumerate(maskCon):
          curr_r, curr_phi = respMatrixF1_maskTf[mcI, msI, 0], respMatrixF1_maskTf_phi[mcI, msI, 0];
          refPhi = phAdv_model(*curr_params, curr_r);
          new_r = np.multiply(curr_r, np.cos(np.deg2rad(refPhi)-np.deg2rad(curr_phi)));
          respMatrixF1_maskTf[mcI, msI, 0] = new_r;

    if cross_val == 2.0:
      n_sfs, n_cons = gt_respMatrixF1_maskTf.shape[0], gt_respMatrixF1_maskTf.shape[1];
      con_ind, sf_ind = np.floor(np.divide(boot_i, n_sfs)).astype('int'), np.mod(boot_i, n_sfs).astype('int');
      print('holding out con/sf indices %02d/%02d' % (con_ind, sf_ind));

    for measure in [0,1]:
      if measure == 0:
        baseline = np.nanmean(hf.resample_array(resample, expInfo['blank']['resps']));
        if cross_val == 2.0:
          # --- so, "nan" out that condition in all of the responses
          mask_only_ref = np.copy(respMatrixDC_onlyMask);
          mask_base_ref = np.copy(respMatrixDC);
          # the above establish the reference values; below, we nan out the current condition
          mask_only = np.copy(respMatrixDC_onlyMask);
          mask_base = np.copy(respMatrixDC);
          mask_only[sf_ind, con_ind] = np.nan;
          mask_base[sf_ind, con_ind] = np.nan;
        else:
          mask_only = respMatrixDC_onlyMask;
          mask_base = respMatrixDC;
        mask_only_all = None; # as of 22.06.15
        fix_baseline = False
      elif measure == 1:
        baseline = 0;
        if cross_val == 2.0:
          # --- so, "nan" out that condition in all of the responses
          mask_only_ref = np.copy(respMatrixF1_onlyMask);
          mask_base_ref = np.copy(respMatrixF1_maskTf);
          # the above establish the reference values; below, we nan out the current condition
          mask_only = np.copy(respMatrixF1_onlyMask);
          mask_base = np.copy(respMatrixF1_maskTf);
          mask_only[sf_ind, con_ind] = np.nan;
          mask_base[sf_ind, con_ind] = np.nan;
        else:
          mask_only = respMatrixF1_onlyMask;
          mask_base = respMatrixF1_maskTf;
        mask_only_all = None; # as of 22.06.15; ignore the above line
        fix_baseline = True;
      resp_str = hf_sf.get_resp_str(respMeasure=measure);

      if cross_val is not None: # set up what is the test data!
        nan_val = -1e3;
        # make sure we nan out any existing NaN values in the reference data
        mask_only_ref[np.isnan(mask_only_ref)] = nan_val
        # make sure we nan out any NaN values in the training data
        mask_only_tr = np.copy(mask_only);
        mask_only_tr[np.isnan(mask_only_tr)] = nan_val;
        heldout = np.abs(mask_only_tr - mask_only_ref) > 1e-6 # if the idff. is g.t. this, it means they are different values
        test_mask_only = np.nan * np.zeros_like(mask_only_tr);
        test_mask_base = np.nan * np.zeros_like(mask_only_tr);
        test_mask_only[heldout] = mask_only_ref[heldout]
        test_mask_base[heldout] = mask_base_ref[heldout]
        heldouts = [test_mask_only]; # update to include ...base IF below lines (which*) include both
      else:
        heldouts = [None];
        
      whichAll = [mask_only_all];
      whichResp = [mask_only]#, mask_base];
      whichKey = ['mask']#, 'both'];

      if fit_rvc == 1 or fit_rvc is not None:
        ''' Fit RVCs responses (see helper_fcns.rvc_fit for details) for:
            --- F0: mask alone (7 sfs)
                    mask + base together (7 sfs)
            --- F1: mask alone (7 sfs; at maskTf)
                    mask+ + base together (7 sfs; again, at maskTf)
            NOTE: Assumes only sfBB_core
        '''
        if resp_str not in rvcFits_curr_toSave:
          rvcFits_curr_toSave[resp_str] = dict();

        cons = expInfo['maskCon'];
        # first, mask only; then mask+base
        for wR, wK, wA in zip(whichResp, whichKey, whichAll):
          # create room for an empty dict, if not already present
          if wK not in rvcFits_curr_toSave[resp_str]:
            rvcFits_curr_toSave[resp_str][wK] = dict();

          adjMeans = np.transpose(wR[:,:,0]); # just the means
          # --- in new version of code [to allow boot], we can get masked array; do the following to save memory
          adjMeans = adjMeans.data if isinstance(adjMeans, np.ma.MaskedArray) else adjMeans;
          consRepeat = [cons] * len(adjMeans);
   
          # get a previous fit, if present
          try:
            rvcFit_curr = rvcFits_curr_toSave[resp_str][wK] if not resample else None;
          except:
            rvcFit_curr = None
          # do the fitting!
          _, all_opts, all_conGains, all_loss = hf.rvc_fit(adjMeans, consRepeat, var=None, mod=rvcMod, fix_baseline=fix_baseline, prevFits=rvcFit_curr, n_repeats=n_repeats);

          # compute variance explained!
          varExpl = [hf.var_explained(hf.nan_rm(dat), hf.nan_rm(hf.get_rvcResp(prms, cons, rvcMod)), None) for dat, prms in zip(adjMeans, all_opts)];
          # now, package things
          if resample:
            if boot_i == 0: # i.e. first time around
              # - then we create empty lists to which we append the result of each success iteration
              # --- note that we do not include adjMeans here (don't want nBoots iterations of response means saved!)
              rvcFits_curr_toSave[resp_str][wK]['boot_loss'] = [];
              rvcFits_curr_toSave[resp_str][wK]['boot_params'] = [];
              rvcFits_curr_toSave[resp_str][wK]['boot_conGain'] = [];
              rvcFits_curr_toSave[resp_str][wK]['boot_varExpl'] = [];
            # then -- append!
            rvcFits_curr_toSave[resp_str][wK]['boot_loss'].append(all_loss)
            rvcFits_curr_toSave[resp_str][wK]['boot_params'].append(all_opts)
            rvcFits_curr_toSave[resp_str][wK]['boot_conGain'].append(all_conGains)
            rvcFits_curr_toSave[resp_str][wK]['boot_varExpl'].append(varExpl)
          else: # we will never be here more than once, since if not resample, then nBoots = 1
            rvcFits_curr_toSave[resp_str][wK]['loss'] = all_loss;
            rvcFits_curr_toSave[resp_str][wK]['params'] = all_opts;
            rvcFits_curr_toSave[resp_str][wK]['conGain'] = all_conGains;
            rvcFits_curr_toSave[resp_str][wK]['adjMeans'] = adjMeans
            rvcFits_curr_toSave[resp_str][wK]['varExpl'] = varExpl;

        ######## 
        # END of rvc fit (for this measure, boot iteration)
        ######## 

      if fit_sf == 1 or fit_sf is not None:
        ''' Fit SF tuning responses (see helper_fcns.dog_fit for details) for:
            --- F0: mask alone (7 cons)
                    mask + base together (7 cons)
            --- F1: mask alone (7 cons; at maskTf)
                    mask+ + base together (7 cons; again, at maskTf)
            NOTE: Assumes only sfBB_core
        '''
        if resp_str not in sfFits_curr_toSave:
          sfFits_curr_toSave[resp_str] = dict();

        cons, sfs = expInfo['maskCon'], expInfo['maskSF']
        stimVals = [[0], cons, sfs];
        valConByDisp = [np.arange(0,len(cons))]; # all cons are valid in sfBB experiment

        for wR, wK, wA, heldout in zip(whichResp, whichKey, whichAll, heldouts):
          if wK not in sfFits_curr_toSave[resp_str]:
            sfFits_curr_toSave[resp_str][wK] = dict();

          # get a previous fit, if present
          try:
            sfFit_curr = sfFits_curr_toSave[resp_str][wK] if not resample else None;
          except:
            sfFit_curr = None

          # try to load isolated fits...
          if jointSf>0:
            try: # load non_joint fits as a reference (see hf.dog_fit or S. Sokol thesis for details)
              modStr  = hf.descrMod_name(sfMod);
              ref_fits = hf.np_smart_load(data_path + hf.descrFit_name(loss_type, descrBase=sfName, modelName=modStr, joint=0, phAdj=phAdj));
              isolFits = ref_fits[cellNum-1][resp_str][wK]['params'];
              if sfMod==sfModRef:
                ref_varExpl = ref_fits[cellNum-1][resp_str][wK]['varExpl'];
              else:
                # otherwise, load the DoG
                vExp_ref_fits = hf.np_smart_load(data_path + hf.descrFit_name(loss_type, descrBase=sfName, modelName=hf.descrMod_name(sfModRef), joint=0, phAdj=phAdj));
                ref_varExpl = vExp_ref_fits[cellNum-1][resp_str][wK]['varExpl'];
            except:
               isolFits = None;
               ref_varExpl = None;
          else:
            isolFits = None;
            ref_varExpl = None;

          allCurr = np.expand_dims(np.transpose(wA, [1,0,2]), axis=0) if wA is not None else None;
          # -- by default, loss_type=2 (meaning sqrt loss); why expand dims and transpose? dog fits assumes the data is in [disp,sf,con] and we just have [con,sf]
          nll, prms, vExp, pSf, cFreq, totNLL, totPrm, success = hf.dog_fit([np.expand_dims(np.transpose(wR[:,:,0]), axis=0), allCurr, np.expand_dims(np.transpose(wR[:,:,1]), axis=0), baseline], sfMod, loss_type=2, disp=0, expInd=None, stimVals=stimVals, validByStimVal=None, valConByDisp=valConByDisp, prevFits=sfFit_curr, noDisp=1, fracSig=fracSig, n_repeats=n_repeats, isolFits=isolFits, ref_varExpl=ref_varExpl, veThresh=veThresh, joint=jointSf, ftol=ftol, resp_thresh=resp_thresh) # noDisp=1 means that we don't index dispersion when accessins prevFits
          
          if resample or cross_val is not None:
            if boot_i == 0: # i.e. first time around
              if cross_val is not None:
                # first, pre-define empty lists for all of the needed results, if they are not yet defined
                sfFits_curr_toSave[resp_str][wK]['boot_NLL_cv_test'] = np.empty((nBoots,) + nll.shape, dtype=np.float32);
                sfFits_curr_toSave[resp_str][wK]['boot_vExp_cv_test'] = np.empty((nBoots,) + vExp.shape, dtype=np.float32);
                sfFits_curr_toSave[resp_str][wK]['boot_NLL_cv_train'] = np.empty((nBoots,) + nll.shape, dtype=np.float32);
                sfFits_curr_toSave[resp_str][wK]['boot_vExp_cv_train'] = np.empty((nBoots,) + vExp.shape, dtype=np.float32);
                # --- these are all implicitly based on training data
                sfFits_curr_toSave[resp_str][wK]['boot_cv_params'] = np.empty((nBoots,) + prms.shape, dtype=np.float32);
                sfFits_curr_toSave[resp_str][wK]['boot_cv_prefSf'] = np.empty((nBoots,) + pSf.shape, dtype=np.float32);
                sfFits_curr_toSave[resp_str][wK]['boot_cv_charFreq'] = np.empty((nBoots,) + cFreq.shape, dtype=np.float32);
              else: # otherwise, the things we put only if we didn't have cross-validation
                # - pre-allocate empty array of length nBoots (save time over appending each time around)
                sfFits_curr_toSave[resp_str][wK]['boot_loss'] = np.empty((nBoots,) + nll.shape, dtype=np.float32);
                sfFits_curr_toSave[resp_str][wK]['boot_params'] = np.empty((nBoots,) + prms.shape, dtype=np.float32);
                sfFits_curr_toSave[resp_str][wK]['boot_varExpl'] = np.empty((nBoots,) + vExp.shape, dtype=np.float32);
                sfFits_curr_toSave[resp_str][wK]['boot_prefSf'] = np.empty((nBoots,) + pSf.shape, dtype=np.float32);
                sfFits_curr_toSave[resp_str][wK]['boot_charFreq'] = np.empty((nBoots,) + cFreq.shape, dtype=np.float32);
              # and the below apply whether or not we did cross-validation!
              if jointSf>0:
                sfFits_curr_toSave[resp_str][wK]['boot_totalNLL'] = np.empty((nBoots,) + totNLL.shape, dtype=np.float32);
                sfFits_curr_toSave[resp_str][wK]['boot_paramList'] = np.empty((nBoots,) + totPrm.shape, dtype=np.float32);
                sfFits_curr_toSave[resp_str][wK]['boot_success'] = np.empty((nBoots, ), dtype=np.bool_);
              else: # only if joint=0 will success be an array (and not just one value)
                sfFits_curr_toSave[resp_str][wK]['boot_success'] = np.empty((nBoots, ) + success.shape, dtype=np.bool_);
            
            # then -- put in place (we reach here for all boot_i)
            if cross_val is not None:
              ### then, we need to compute test loss/vExp!
              test_nlls = np.nan*np.zeros_like(nll);
              test_vExps = np.nan*np.zeros_like(vExp);
              # set up ref_params, ref_rc_val; will only be used IF applicable
              ref_params = None; ref_rc_val = None;
              if sfMod==3:
                try:
                  all_xc = prms[:,1]; # xc
                  ref_params = [np.nanmin(all_xc), 1];
                except:
                  pass;
              else:
                try:
                  ref_params = prms[-1]; # high contrast condition
                  ref_rc_val = totPrm[2] if jointSf>0 else None; # even then, only used for jointSf==5
                except:
                  pass;

              for ii, prms_curr in enumerate(prms):
                # we'll iterate over the parameters, which are fit for each contrast (the final dimension of test_mn)
                if np.any(np.isnan(prms_curr)):
                  continue;
                non_nans = np.where(~np.isnan(heldout[:,ii,0]))[0]; # check -- from the heldout data at contrast ii, which sfs are not NaN
                curr_sfs = stimVals[2][non_nans]; # get those sf values
                resps_curr = heldout[non_nans, ii, 0]; # and get those responses, to pass into DoG_loss
                test_nlls[ii] = hf.DoG_loss(prms_curr, resps_curr, curr_sfs, resps_std=None, loss_type=loss_type, DoGmodel=sfMod, dir=dir, joint=0, baseline=baseline, ref_params=ref_params, ref_rc_val=ref_rc_val) # why not enforce max? b/c fewer resps means more varied range of max, don't want to wrongfully penalize
                test_vExps[ii] = hf.var_explained(resps_curr, prms_curr, curr_sfs, sfMod, baseline=baseline, ref_params=ref_params, ref_rc_val=ref_rc_val);

              ### done with computing test loss, so save everything
              sfFits_curr_toSave[resp_str][wK]['boot_NLL_cv_test'][boot_i] = test_nlls
              sfFits_curr_toSave[resp_str][wK]['boot_vExp_cv_test'][boot_i] = test_vExps
              sfFits_curr_toSave[resp_str][wK]['boot_NLL_cv_train'][boot_i] = nll
              sfFits_curr_toSave[resp_str][wK]['boot_vExp_cv_train'][boot_i] = vExp
              # --- these are all implicitly based on training data
              sfFits_curr_toSave[resp_str][wK]['boot_cv_params'][boot_i] = prms
              sfFits_curr_toSave[resp_str][wK]['boot_cv_prefSf'][boot_i] = pSf
              sfFits_curr_toSave[resp_str][wK]['boot_cv_charFreq'][boot_i] = cFreq;
            else:
              sfFits_curr_toSave[resp_str][wK]['boot_loss'][boot_i] = nll;
              sfFits_curr_toSave[resp_str][wK]['boot_params'][boot_i] = prms;
              sfFits_curr_toSave[resp_str][wK]['boot_charFreq'][boot_i] = cFreq;
              sfFits_curr_toSave[resp_str][wK]['boot_varExpl'][boot_i] = vExp
              sfFits_curr_toSave[resp_str][wK]['boot_prefSf'][boot_i] = pSf;
            # these apply regardless of c-v or not
            sfFits_curr_toSave[resp_str][wK]['boot_success'][boot_i] = success;
            if jointSf>0:
              try:
                sfFits_curr_toSave[resp_str][wK]['boot_totalNLL'][boot_i] = totNLL;
                sfFits_curr_toSave[resp_str][wK]['boot_paramList'][boot_i] = totPrm;
              except:
                pass;
          else: # otherwise, we'll only be here once
            sfFits_curr_toSave[resp_str][wK]['NLL'] = nll.astype(np.float32);
            sfFits_curr_toSave[resp_str][wK]['params'] = prms.astype(np.float32);
            sfFits_curr_toSave[resp_str][wK]['varExpl'] = vExp.astype(np.float32);
            sfFits_curr_toSave[resp_str][wK]['prefSf'] = pSf.astype(np.float32);
            sfFits_curr_toSave[resp_str][wK]['charFreq'] = cFreq.astype(np.float32);
            sfFits_curr_toSave[resp_str][wK]['success'] = success#.astype(np.bool_);
            if jointSf>0:
              try:
                sfFits_curr_toSave[resp_str][wK]['totalNLL'] = totNLL.astype(np.float32);
              except:
                sfFits_curr_toSave[resp_str][wK]['totalNLL'] = totNLL; # if it's None, or [], or...
              try:
                sfFits_curr_toSave[resp_str][wK]['paramList'] = totPrm.astype(np.float32);
              except:
                sfFits_curr_toSave[resp_str][wK]['paramList'] = totPrm; # if it's None, or [], or...
        ######## 
        # END of sf fit (for this measure, boot iteration)
        ######## 

    ######## 
    # END of measure (i.e. handled both measures, go back for more boot_iter, if specified)
    ######## 
  ######## 
  # END of all boot iters (i.e. handled both measures, go back for more boot_iter, if specified)
  ######## 


  ###########
  # NOW, save (if saving); otherwise, we return the values
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
      rvcFits[cellNum-1] = rvcFits_curr_toSave;
      np.save(data_path+rvcNameFinal, rvcFits);
      print('Saving %s, %s @ %s' % (resp_str, wK, rvcNameFinal))

    if fit_sf:

      pass_check=False;
      while not pass_check: # keep saving/reloading until the fit has properly saved everything...

        # load fits again in case some other run has saved/made changes
        if os.path.isfile(data_path + sfNameFinal):
          print('reloading sfFits...');
          sfFits = hf.np_smart_load(data_path + sfNameFinal);
        if cellNum-1 not in sfFits:
          sfFits[cellNum-1] = dict();

        sfFits[cellNum-1] = sfFits_curr_toSave;

        # now save
        np.save(data_path + sfNameFinal, sfFits);
        print('Saving %s, %s @ %s' % (resp_str, wK, sfNameFinal))

        # now check...
        check = hf.np_smart_load(data_path + sfNameFinal);
        if resample: # check that the boot stuff is there
          if 'boot_params' in check[cellNum-1]['dc']['mask'].keys():
            pass_check = True;
        else:
          if 'NLL' in check[cellNum-1]['dc']['mask'].keys(): # just check that any relevant key is there
            pass_check = True;
        # --- and if neither pass_check was triggered, then we go back and reload, etc

    ### End of saving (both RVC and SF)
  else:
    return rvcFits_curr_toSave, sfFits_curr_toSave;

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
  nBoots     = int(sys.argv[7]);
  jointSf    = int(sys.argv[8]);
  if len(sys.argv) > 9:
    cross_val  = float(sys.argv[9]);
    if cross_val <= 0:
      cross_val = None;
  else:
    cross_val = None;

  phAdj = True; # as of late July 2022, we automatically apply phAdj to V1_BB data
    
  #resp_thresh = (5,2); # at least 2 responses must be g.t.e. 5 spks/s
  veThresh = 60; # i.e. let all through
  resp_thresh = (-1e5,0);
  #veThresh = -1e5; # i.e. let all through
  
  print('%d - %d - %d - %d - %d - %d - %d' % (fit_rvc, fit_sf, rvc_mod, sf_mod, loss_type, nBoots, jointSf));
  print('rvc? %d -- sf? %d -- sfMod,joint %d,%d' % (fit_rvc, fit_sf, sf_mod, jointSf))
  fracSig = 1; # why fracSig =1? For V1 fits, we want to contrasin the upper-half sigma of the two-half gaussian as a fraction of the lower half

  if asMulti:
    from functools import partial
    import multiprocessing as mp
    nCpu = mp.cpu_count();

    # to avoid race conditions, load the previous fits beforehand; and the datalist
    rvcNameFinal = hf.rvc_fit_name(rvcName, rvc_mod, None, vecF1=1); # DEFAULT is vecF1 adjustment
    modStr = hf.descrMod_name(sf_mod);
    sfNameFinal = hf.descrFit_name(loss_type, descrBase=sfName, modelName=modStr, joint=jointSf, phAdj=phAdj); # descrLoss order is lsq/sqrt/poiss/sach

    pass_rvc = hf.np_smart_load('%s%s%s' % (basePath, data_suff, rvcNameFinal)) if fit_rvc else None;
    pass_sf = hf.np_smart_load('%s%s%s' % (basePath, data_suff, sfNameFinal)) if fit_sf else None;

    dataList = hf.np_smart_load('%s%s%s' % (basePath, data_suff, hf.get_datalist('V1_BB/')));

    if nBoots > 1:
      n_repeats = 3 if jointSf>0 else 5; # fewer if repeat
    else:
      n_repeats = 20 if jointSf>0 else 15; # was previously be 3, 15, then 7, 15

    with mp.Pool(processes = nCpu) as pool:
      # if we're doing as parallel, do NOT save
      fit_perCell = partial(make_descr_fits, fit_rvc=pass_rvc, fit_sf=pass_sf, rvcMod=rvc_mod, sfMod=sf_mod, toSave=0, fracSig=fracSig, loss_type=loss_type, nBoots=nBoots, jointSf=jointSf, n_repeats=n_repeats, resp_thresh=resp_thresh, veThresh=veThresh, phAdj=phAdj);
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
        sfNameFinal = hf.descrFit_name(lossType=2, descrBase=sfName, modelName=modStr, joint=jointSf, phAdj=phAdj); # descrLoss order is lsq/sqrt/poiss/sach
        if os.path.isfile(dataPath + sfNameFinal):
          print('reloading sfFits...');
          sfFits = hf.np_smart_load(dataPath + sfNameFinal);
        else:
          sfFits = dict();
        for iii, sfFit in enumerate(sf_fits):
          sfFits[iii] = sfFit;
        np.save(dataPath + sfNameFinal, sfFits);

  else:
    if nBoots > 1:
      n_repeats = 2 if jointSf>0 else 5; # fewer if boot
    else:
      n_repeats = 5 if jointSf>0 else 12; # was previously be 3, 15, then 7, 15

    make_descr_fits(cell_num, fit_rvc=fit_rvc, fit_sf=fit_sf, rvcMod=rvc_mod, sfMod=sf_mod, loss_type=loss_type, nBoots=nBoots, jointSf=jointSf, n_repeats=n_repeats, resp_thresh=resp_thresh, veThresh=veThresh, phAdj=phAdj);

