import numpy as np
import helper_fcns_sach as hf
import scipy.optimize as opt
from scipy.stats import norm, mode, lognorm, nbinom, poisson, sem
from numpy.matlib import repmat
import os.path
import sys
import copy

import pdb

#################
### RVC
#################

def phase_advance_fit(cell_num, data_loc, phAdvName, dir=1, to_save=1, returnMod=1):
  ''' Modeled after the call in the "standard" helper_fcns, but specific to accessing responses of Sach data '''

  # load cell information
  if data_loc is not None:
    dataList = hf.np_smart_load(data_loc + 'sachData.npy');
    assert dataList!=[], "data file not found!"
    data = dataList[cell_num-1]['data'];
  else: # if we make data_loc=None, then we've already passed in the data in cell_num
    data = cell_num; 

  if to_save:
    phAdvFinal = hf.phase_fit_name(phAdvName, dir);
    # first, load the file if it already exists
    if os.path.isfile(data_loc + phAdvFinal):
        try:
          phFits = hf.np_smart_load(data_loc + phAdvFinal);
        except:
          phFits = np.load(data_loc + phAdvFinal, allow_pickle=True).item();
        try:
          phFits_curr = phFits[cell_num-1];
        except:
          phFits_curr = None;
    else:
        phFits = dict();
        phFits_curr = None;

  print('Doing the work, now');

  all_f1_tr = [hf.nan_rm(x) for x in data['f1arr']]; # nan rm first
  all_f1ph_tr = [hf.nan_rm(x) for x in data['f1pharr']]; # nan rm first
  allAmp_mean, allPhi_mean, allAmp_std, allPhi_var = hf.polar_vec_mean(all_f1_tr, all_f1ph_tr);

  # then, we'll need to organize into list of lists -- by SF (outer), then by asc. con (inner), with [mean, std OR var] per each
  cons = np.unique(data['cont']);
  cons = cons[cons>0]; # exclude 0% contrast from this analysis (it's noise!!!)
  sfs = np.unique(data['sf']);
  allAmp = []; allPhi = []; allCons = []; allTf = []; allPhiVar = [];
  for sf_val in sfs:
    curr_amp = []; curr_phi = []; curr_tf = []; curr_phiVar = [];
    ok_sf = data['sf']==sf_val;
    for con_val in cons:
      which_ind = np.where(np.logical_and(data['cont']==con_val, ok_sf))[0][0];
      curr_amp.append([allAmp_mean[which_ind], allAmp_std[which_ind]]);
      curr_phi.append([allPhi_mean[which_ind], allPhi_var[which_ind]]);
      curr_tf.append([data['tf'][which_ind]]); # should be same for all conditions
      curr_phiVar.append([allPhi_var[which_ind]]);
    allAmp.append(curr_amp);
    allPhi.append(curr_phi);
    allCons.append(cons);
    allTf.append(curr_tf);
    allPhiVar.append(curr_phiVar);

  phAdv_model, all_opts, all_phAdv, all_loss = hf.phase_advance(allAmp, allPhi, allCons, allTf, phiVar=allPhiVar);

  # update stuff - load again in case some other run has saved/made changes
  curr_fit = dict();
  curr_fit['loss'] = all_loss;
  curr_fit['params'] = all_opts;
  curr_fit['phAdv'] = all_phAdv;
  curr_fit['cellNum'] = cell_num;

  if to_save:
    pass_check = False;
    while not pass_check:
      if os.path.isfile(data_loc + phAdvFinal):
        print('reloading phAdvFits...');
        phFits = hf.np_smart_load(data_loc + phAdvFinal);
      else:
        phFits = dict();
      phFits[cell_num-1] = curr_fit;
      np.save(data_loc + phAdvFinal, phFits);
      print('saving phase advance fit for cell ' + str(cell_num));

      # now check...
      check = hf.np_smart_load(data_loc + phAdvFinal);
      if cell_num-1 in check:
        if 'loss' in check[cell_num-1].keys(): # just check that any relevant key is there
          pass_check = True;
      # --- and if neither pass_check was triggered, then we go back and reload, etc

  if returnMod: # default
    return phAdv_model, all_opts;
  else:
    return curr_fit;

def rvc_fit(cell_num, data_loc, rvcName, rvcMod=0, nBoots=0, phAdjusted=0, dir=1, to_save=1, cross_val=None):
  ''' IF phAdjusted=1, then we Piggy-backing off of phase_advance_fit above, get prepared to project the responses onto the proper phase to get the correct amplitude
      Then, with the corrected response amplitudes, fit the RVC model
      - as of 19.11.07, we will fit non-baseline subtracted responses 
          (F1 have baseline of 0 always, but now we will not subtract baseline from F0 responses)
  '''

  # Set up whether we will bootstrap straight away
  resample = False if nBoots <= 0 else True;
  nBoots = 1 if nBoots <= 0 else nBoots;

  # load cell information
  dataList = hf.np_smart_load(data_loc + 'sachData.npy');
  assert dataList!=[], "data file not found!"
  data = dataList[cell_num-1]['data'];

  if phAdjusted<1: # either vec corr or neither
    dir=None;
  vecF1 = 1 if phAdjusted==0 else None;
  rvcNameFinal = hf.rvc_fit_name(rvcName, rvcMod, dir=dir, vecF1=vecF1);
  # first, load the file if it already exists
  if os.path.isfile(data_loc + rvcNameFinal):
      try:
        rvcFits = hf.np_smart_load(data_loc + rvcNameFinal);
      except:
        rvcFits = np.load(data_loc + rvcNameFinal, allow_pickle=True).item();
      try:
        rvcFits_curr = rvcFits[cell_num-1];
      except:
        rvcFits_curr = dict();
  else:
      rvcFits = dict();
      rvcFits_curr = dict();

  print('Doing the work, now');

  # running tabulate once to know shape of experiment (e.g. nCons, nSfs)
  to_unpack = hf.tabulateResponses(data); # no need to pass in phAdj, dir since we ignore responses
  [all_cons, all_sfs] = to_unpack[1];
  v_sfs = all_sfs;
  n_v_sfs = len(v_sfs);

  if resample:
    boot_loss = []; boot_params = []; boot_conGain = []; boot_varExpl = [];

  for boot_i in range(nBoots):

    to_unpack = hf.tabulateResponses(data, resample, phAdjusted=phAdjusted, dir=dir);
    [_, f1] = to_unpack[0];
    [_, f1arr] = to_unpack[2];

    resps = [f1['mean'][:, x] for x in range(n_v_sfs)];
    respsSEM = [f1['sem'][:, x] for x in range(n_v_sfs)];
    cons = [all_cons] * len(resps); # tile
    rvc_model, all_opts, all_conGain, all_loss = hf.rvc_fit(resps, cons, var=respsSEM, mod=rvcMod, prevFits=rvcFits_curr);
    varExpl = [hf.var_expl_direct(dat, hf.get_rvcResp(prms, all_cons, rvcMod)) for dat, prms in zip(resps, all_opts)];

    if resample:
      boot_loss.append(all_loss); boot_params.append(all_opts); boot_conGain.append(all_conGain); boot_varExpl.append(varExpl);
  # end of boot

  if resample:
    rvcFits_curr['boot_loss'] = boot_loss;
    rvcFits_curr['boot_params'] = boot_params;
    rvcFits_curr['boot_conGain'] = boot_conGain;
    rvcFits_curr['boot_varExpl'] = boot_varExpl;
  else:
    rvcFits_curr['loss'] = all_loss;
    rvcFits_curr['params'] = all_opts;
    rvcFits_curr['conGain'] = all_conGain;
    rvcFits_curr['varExpl'] = varExpl;

  if to_save:
    if os.path.isfile(data_loc + rvcNameFinal):
      print('reloading rvcFits...');
      try:
        rvcFits = hf.np_smart_load(data_loc + rvcNameFinal);
      except:
        rvcFits = np.load(data_loc + rvcNameFinal, allow_pickle=True).item();

    if cell_num-1 not in rvcFits:
      rvcFits[cell_num-1] = dict();
    rvcFits[cell_num-1] = rvcFits_curr;
      
    np.save(data_loc + rvcNameFinal, rvcFits);

  # now, just return
  return rvcFits_curr;

#################
### DESCRIPTIVE SF (DoG)
#################

def invalid(params, bounds):
# given parameters and bounds, are the parameters valid?
  for p in range(len(params)):
    if params[p] < bounds[p][0] or params[p] > bounds[p][1]:
      return True;
  return False;

def fit_descr_empties(nCons, nParam, joint=0, nBoots=1):
  ''' Just create the empty numpy arrays that we'll fill up 
  '''
  nBoots = 1 if nBoots <= 0 else nBoots;

  ## Yes, I'm pre-pending the nBoots dimensions, but if it's one, we'll use np.squeeze at the end to remove that
  # Set the default values (NaN for everything)
  bestNLL = np.ones((nBoots, nCons), dtype=np.float32) * np.nan;
  currParams = np.ones((nBoots, nCons, nParam), dtype=np.float32) * np.nan;
  varExpl = np.ones((nBoots, nCons), dtype=np.float32) * np.nan;
  prefSf = np.ones((nBoots, nCons), dtype=np.float32) * np.nan;
  charFreq = np.ones((nBoots, nCons), dtype=np.float32) * np.nan;
  if joint>0:
    totalNLL = np.ones((nBoots, )) * np.nan;
    paramList = np.ones((nBoots, ), dtype='O') * np.nan;
    success = np.zeros((nBoots, ), dtype=np.bool_);
  else:
    totalNLL = None;
    paramList = None;
    success = np.zeros((nBoots, nCons), dtype=np.bool_);

  return bestNLL, currParams, varExpl, prefSf, charFreq, totalNLL, paramList, success;

def fit_descr_DoG(cell_num, data_loc, dogBase, n_repeats = 15, loss_type = 3, DoGmodel = 1, joint=0, fracSig=0, nBoots=0, forceOverwrite=False, phAdj=1, to_save=1, cross_val=None, veThresh=-np.nan):

  # Set up whether we will bootstrap straight away
  resample = False if nBoots <= 0 else True;
  nBoots = 1 if nBoots <= 0 else nBoots;

  if DoGmodel == 0:
    nParam = 5;
  else: # we should not fit the d-DoG-S model in the LGN!
    nParam = 4;

  # load cell information
  dataList = hf.np_smart_load(data_loc + 'sachData.npy');
  assert dataList!=[], "data file not found!"

  HPC = 'HPC' if 'pl1465' in data_loc else '';

  if joint>0:
    try: # load non_joint fits as a reference (see hf.dog_fit or S. Sokol thesis for details)
      modStr  = hf.descrMod_name(DoGmodel);
      fitName = hf.descrFit_name(loss_type, descrBase=dogBase, modelName=modStr, joint=0, phAdj=phAdj)
      try:
        ref_fits = hf.np_smart_load(data_loc + fitName);
      except:
        ref_fits = np.load(data_loc + fitName, allow_pickle=True).item();
      ref_varExpl = ref_fits[cell_num-1]['varExpl'];
    except:
      ref_varExpl = None;
  else:
    ref_varExpl = None; # set to None as default

  mod_str = hf.descrMod_name(DoGmodel);
  fLname = str(data_loc + hf.descrFit_name(loss_type, dogBase, mod_str, joint=joint, phAdj=phAdj));

  if os.path.isfile(fLname):
    try:
      descrFits = hf.np_smart_load(fLname);
    except:
      descrFits = np.load(fLname, allow_pickle=True).item();
  else:
    descrFits = dict();
  try:
    descrFits_curr = descrFits[cell_num-1];
  except:
    descrFits_curr = dict();

  data = dataList[cell_num-1]['data'];

  # First, set the default values (NaN for everything); if no resample, we'll remove the singleton dimension
  to_unpack_ref = hf.tabulateResponses(data, resample=False, phAdjusted=phAdj, dir=dir, cross_val=None);
  [_, f1_ref] = to_unpack_ref[0];
  [all_cons, all_sfs] = to_unpack_ref[1];
  n_v_sfs = len(all_sfs);
  [_, f1arr_ref] = to_unpack_ref[2];
  f1arr_ref = hf.unpack_f1arr(f1arr_ref);
  nCons = len(all_cons);
  # pre-fill the values
  if cross_val == 2.0:
    nBoots = n_v_sfs * len(all_cons); # why? need one boot for each condition left out
    print('nBoots is %03d' % nBoots);

  bestNLL, currParams, varExpl, prefSf, charFreq, totalNLL, paramList, success = fit_descr_empties(nCons, nParam, joint, nBoots);

  print('Doing the work, now');

  make_cv_train_subset = False;
  if cross_val is not None:
    if cross_val>0 and cross_val<1:
      make_cv_train_subset = True;
    # also create infra. to save nll, vExp
    test_nll = np.copy(bestNLL);
    test_vExp = np.copy(varExpl);
    if make_cv_train_subset:
      tr_subset_nll = np.copy(bestNLL);
      tr_subset_vExp = np.copy(varExpl);
  
  for boot_i in range(nBoots):

    if cross_val == 2.0:
      con_ind, sf_ind = np.floor(np.divide(boot_i, n_v_sfs)).astype('int'), np.mod(boot_i, n_v_sfs).astype('int');
      print('holding out con/sf indices %02d/%02d' % (con_ind, sf_ind));
      # copy the dictionary, and make the requisite condition NaN
      f1 = copy.deepcopy(f1_ref);
      f1['mean'][con_ind, sf_ind] = np.nan;
      f1['sem'][con_ind, sf_ind] = np.nan;
      f1arr = copy.copy(f1arr_ref);
      f1arr[con_ind, sf_ind] = np.nan;
    else:
      # we need to COPY data so as to not overwrite the original data through consecutive resampling
      dt = copy.deepcopy(data);
      to_unpack = hf.tabulateResponses(dt, resample, phAdjusted=phAdj, dir=dir, cross_val=cross_val);
      [_, f1] = to_unpack[0];
      [_, f1arr] = to_unpack[2];
      # unpack the dictionary of responses (nested --> f1arr[con][sf]) into conXsfXrespXrepeat
      # NOTE: Check that this unpacks properly IF f1arr dimensions are not all equal
      f1arr = hf.unpack_f1arr(f1arr);

    if cross_val is not None and resample:
      ########
      ### cross-val stuff
      # --- find out which data were held out by:
      # --- turning all NaN into a set value (e.g. -1e3), find where the training/all arrays differ, get those values
      #########
      nan_val = -1e3;
      training = np.copy(f1arr);
      training[np.isnan(training)] = nan_val;
      all_data = np.copy(f1arr_ref);
      all_data[np.isnan(all_data)] = nan_val;
      heldout = np.abs(all_data - training) > 1e-6; # if the difference is g.t. this, it means they are different value
      test_data = np.nan * np.zeros_like(f1arr);
      test_data[heldout] = all_data[heldout]; # then put the heldout values here

    # now, we fit!
    nll, prms, vExp, pSf, cFreq, totNLL, totPrm, succ = hf.dog_fit(f1, all_cons, all_sfs, DoGmodel, loss_type, n_repeats, joint=joint, ref_varExpl=ref_varExpl, fracSig=fracSig, jointMinCons=0, veThresh=veThresh);
    ref_rc_val = None if joint==0 else totPrm[2];

    if resample:
      bestNLL[boot_i] = nll;
      currParams[boot_i] = prms;
      varExpl[boot_i] = vExp;
      prefSf[boot_i] = pSf;
      charFreq[boot_i] = cFreq;
      success[boot_i] = succ;
      if joint>0:
        totalNLL[boot_i] = totNLL;
        paramList[boot_i] = totPrm;

      if cross_val is not None:
        # compute the loss, varExpl on the heldout (i.e. test) data
        test_mn = np.nanmean(test_data, axis=-1);
        test_sem = sem(test_data, axis=-1, nan_policy='omit');
        # assumes not joint??
        test_nlls = np.nan*np.zeros_like(nll);
        test_vExps = np.nan*np.zeros_like(vExp);
        if make_cv_train_subset:
          tr_subset_nlls = np.nan*np.zeros_like(nll);
          tr_subset_vExps = np.nan*np.zeros_like(vExp);

        # will be used iff joint==5
        for ii, prms_curr in enumerate(prms):
          # we'll iterate over the parameters, which are fit for each contrast (the final dimension of test_mn)
          if np.any(np.isnan(prms_curr)):
            continue;
          non_nans = np.where(~np.isnan(test_mn[ii]))[0];
          curr_sfs = all_sfs[non_nans]; # as values
          resps_curr = test_mn[ii, non_nans]
          if make_cv_train_subset:
            # now, let's also make a size-matched subset of the training data to see if the small N is the source of the noise
            len_test = np.array([np.sum(~np.isnan(test_data[ii, nn])) for nn in non_nans]);
            len_train = np.array([np.sum(~np.isnan(training[ii, nn])) for nn in non_nans]);
            to_repl = False if np.max(len_test)<np.max(len_train) else True; # if the training set is SMALLER then the test set, then we should allow resampling; otherwise, don't
            try:
              train_subset_curr = np.array([np.nanmean(np.random.choice(hf.nan_rm(f1arr[ii, nn]), num, replace=to_repl)) for nn,num in zip(non_nans, len_test)]);
            except: # we could have all-NaN subset?
              train_subset_curr = None;

          test_nlls[ii] = hf.DoG_loss(prms_curr, resps_curr, curr_sfs, resps_std=test_sem[ii, non_nans], loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, joint=0, ref_rc_val=ref_rc_val) # why not enforce max? b/c fewer resps means more varied range of max, don't want to wrongfully penalizes
          test_vExps[ii] = hf.var_explained(resps_curr, prms_curr, whichSfs=curr_sfs, DoGmodel=DoGmodel, ref_rc_val=ref_rc_val, dataAreResps=True);
          # and evaluate loss, vExp on a subset of the TRAINING data that has the same # of trials as the test data
          # --- why? per Tony + Eero (22.04.21), we want to see if the large discrepancy in loss has to do with noise in smaller samples
          if make_cv_train_subset and train_subset_curr is not None:
            tr_subset_nlls[ii] = hf.DoG_loss(prms_curr, train_subset_curr, curr_sfs, resps_std=test_sem[ii, non_nans], loss_type=loss_type, DoGmodel=DoGmodel, dir=dir, ref_rc_val=ref_rc_val) # why not enforce max? b/c fewer resps means more varied range of max, don't want to wrongfully penalize
            tr_subset_vExps[ii] = hf.var_explained(train_subset_curr, prms_curr, whichSfs=curr_sfs, DoGmodel=DoGmodel, ref_rc_val=ref_rc_val, dataAreResps=True);
        test_nll[boot_i] = test_nlls;
        test_vExp[boot_i] = test_vExps;
        if make_cv_train_subset:
          tr_subset_nll[boot_i] = tr_subset_nlls;
          tr_subset_vExp[boot_i] = tr_subset_vExps;

  ##########
  ### End of boot loop
  ##########
  # remove any singleton dimensions
  bestNLL = np.squeeze(bestNLL, axis=0) if bestNLL.shape[0]==1 else bestNLL;
  currParams = np.squeeze(currParams, axis=0) if currParams.shape[0]==1 else currParams;
  varExpl = np.squeeze(varExpl, axis=0) if varExpl.shape[0]==1 else varExpl;
  prefSf = np.squeeze(prefSf, axis=0) if prefSf.shape[0]==1 else prefSf;
  charFreq = np.squeeze(charFreq, axis=0) if charFreq.shape[0]==1 else charFreq;
  success = np.squeeze(success, axis=0) if success.shape[0]==1 else success;
  if joint>0:
    paramList = np.squeeze(paramList, axis=0) if paramList.shape[0]==1 else paramList;
    totalNLL = np.squeeze(totalNLL, axis=0) if totalNLL.shape[0]==1 else totalNLL;

  if cell_num-1 not in descrFits:
    descrFits[cell_num-1] = dict(); # and previously created NaN for everything
  else:
    if not resample: # do not reload these values if it is resample...
      try: # If we made a mistake and saved boot_* first, we might not be able to do this; hence the try/except
        bestNLL = descrFits_curr['NLL'];
        currParams = descrFits_curr['params'];
        varExpl = descrFits_curr['varExpl'];
        prefSf = descrFits_curr['prefSf'];
        charFreq = descrFits_curr['charFreq'];
        success = descrFits_curr['success'];
        if joint>0:
          totalNLL = descrFits_curr['totalNLL'];
          paramList = descrFits_curr['paramList'];
      except:
        pass

  if not resample: # otherwise, we do not want to update...
    # now, what we do, depends on if joint or not [AGAIN -- all of this is only for not resample]
    if joint>0:
      if np.isnan(totalNLL) or totNLL < totalNLL: # then UPDATE!
        totalNLL = totNLL;
        paramList = totPrm;
        bestNLL = nll;
        currParams = prms;
        varExpl = vExp;
        prefSf = pSf;
        charFreq = cFreq;
        success = succ;
    else:
      # must check separately for each contrast
      for con in range(nCons):
        if np.isnan(bestNLL[con]) or nll[con] < bestNLL[con] or forceOverwrite: # then UPDATE!
          #print('\tcell %02d, con %02d: loss = %.2f' % (cell_num, con, nll[con]));
          bestNLL[con] = nll[con];
          currParams[con, :] = prms[con];
          varExpl[con] = vExp[con];
          prefSf[con] = pSf[con];
          charFreq[con] = cFreq[con];
          success[con] = succ[con];

  if resample:
    if cross_val is None:
      descrFits_curr['boot_NLL'] = bestNLL;
      descrFits_curr['boot_params'] = currParams;
      descrFits_curr['boot_varExpl'] = varExpl;
      descrFits_curr['boot_prefSf'] = prefSf;
      descrFits_curr['boot_charFreq'] = charFreq;
      descrFits_curr['boot_success'] = success;
      if joint>0:
        descrFits_curr['boot_totalNLL'] = totalNLL;
        descrFits_curr['boot_paramList'] = paramList;
    else: # cross-val stuff is saved differently...
      descrFits_curr['boot_NLL_cv_test'] = test_nll;
      descrFits_curr['boot_vExp_cv_test'] = test_vExp;
      if make_cv_train_subset:
        descrFits_curr['boot_NLL_cv_train_subset'] = tr_subset_nll;
        descrFits_curr['boot_vExp_cv_train_subset'] = tr_subset_vExp;
      descrFits_curr['boot_NLL_cv_train'] = bestNLL;
      descrFits_curr['boot_vExp_cv_train'] = varExpl;
      # based on training data (implicitly)
      descrFits_curr['boot_cv_params'] = currParams;
      descrFits_curr['boot_cv_prefSf'] = prefSf;
      descrFits_curr['boot_cv_charFreq'] = charFreq;
      if joint>0:
        descrFits_curr['boot_totalNLL'] = totalNLL;
        descrFits_curr['boot_paramList'] = paramList;
  else:
    descrFits_curr['NLL'] = bestNLL;
    descrFits_curr['params'] = currParams;
    descrFits_curr['varExpl'] = varExpl;
    descrFits_curr['prefSf'] = prefSf;
    descrFits_curr['charFreq'] = charFreq;
    descrFits_curr['success'] = success;
    if joint>0:
      descrFits_curr['totalNLL'] = totalNLL;
      descrFits_curr['paramList'] = paramList;

  if to_save:
    # before we update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(fLname):
      print('reloading descrFits...');
      try:
        descrFits = hf.np_smart_load(fLname);
      except:
        descrFits = np.load(fLname, allow_pickle=True).item();
    descrFits[cell_num-1] = descrFits_curr;

    np.save(fLname, descrFits);
    print('saving for cell ' + str(cell_num));

  return descrFits_curr;

if __name__ == '__main__':

    basePath = os.getcwd() + '/';
    data_suff = 'structures/';
    data_loc = basePath + data_suff;

    HPC = 'HPC' if 'pl1465' in data_loc else '';

    rvcBase = 'rvcFits%s_220531' % HPC;
    phBase = 'phAdv%s_220531' % HPC;
    dogBase = 'descrFits%s_s220810vEs' % HPC;
    #dogBase = 'descrFits%s_s220730vE' % HPC;
    #rvcBase = 'rvcFits%s_220412' % HPC;
    #phBase = 'phAdv%s_220412' % HPC;

    veThresh = 60;
    
    fracSig = 0; # should be unconstrained, per Tony (21.05.19) for LGN fits

    print('Running cell ' + sys.argv[1] + '...');
    
    cellNum   = int(sys.argv[1])
    if cellNum < -99: 
      # i.e. 3 digits AND negative, then we'll treat the first two digits as where to start, and the second two as when to stop
      # -- in that case, we'll do this as multiprocessing
      asMulti = 1;
      end_cell = int(np.mod(-cellNum, 100));
      start_cell = int(np.floor(-cellNum/100));
    else:
      asMulti = 0;
    rvcModel  = int(sys.argv[2]);
    loss_type = int(sys.argv[3]);
    DoGmodel  = int(sys.argv[4]);
    joint     = int(sys.argv[5]);
    nBoots    = int(sys.argv[6]);
    phAdj     = int(sys.argv[7]); # +1 for phAdj; 0 for vec mean; -1 for scalar mean (BAD)
    if len(sys.argv) > 7:
      cross_val  = float(sys.argv[8]);
      if cross_val <= 0: # but if it's <=0, we set it back to None
        cross_val = None;
    else:
      cross_val = None;

    dir = 1; # by default...
      
    if asMulti:

      from functools import partial
      import multiprocessing as mp
      nCpu = mp.cpu_count()-1; # heuristics say you should reqeuest at least one fewer processes than their are CPU
      print('***cpu count: %02d***' % nCpu);
      vecF1 = 1 if phAdj==0 else None;
      
      if rvcModel >= 0:
        if phAdj==1:
          with mp.Pool(processes = nCpu) as pool:
            ph_perCell = partial(phase_advance_fit, data_loc=data_loc, phAdvName=phBase, dir=dir, to_save=0, returnMod=0);
            phFits = pool.map(ph_perCell, range(start_cell, end_cell+1));
            pool.close();

          ### do the saving HERE!
          phAdvName = hf.phase_fit_name(phBase, dir);
          if os.path.isfile(data_loc + phAdvName):
            print('reloading phAdvFits...');
            phFitNPY = hf.np_smart_load(data_loc + phAdvName);
          else:
            phFitNPY = dict();
          for iii, phFit in enumerate(phFits):
            phFitNPY[iii] = phFit;
            np.save(data_loc + phAdvName, phFitNPY)
 
        # Now, do RVC!
        with mp.Pool(processes = nCpu) as pool:
          rvc_perCell = partial(rvc_fit, data_loc=data_loc, rvcName=rvcBase, rvcMod=rvcModel, nBoots=nBoots, phAdjusted=phAdj, dir=dir, to_save=0); 
          rvcFits = pool.map(rvc_perCell, range(start_cell, end_cell+1));
          pool.close();

        ### do the saving HERE!
        rvcName = hf.rvc_fit_name(rvcBase, rvcModel, dir=dir, vecF1=vecF1);
        if os.path.isfile(data_loc + rvcName):
          print('reloading rvcFits...');
          rvcFitNPY = hf.np_smart_load(data_loc + rvcName);
        else:
          rvcFitNPY = dict();
        for iii, phFit in enumerate(rvcFits):
          rvcFitNPY[iii] = phFit;
          np.save(data_loc + rvcName, rvcFitNPY)
            
      if DoGmodel >= 0:
        print('DoG model is %d' % DoGmodel);
        if nBoots > 1:
          if DoGmodel==1: # if if just DoG and not d-DoG-S
            n_repeats = 5 if joint>0 else 7;
          else:
            n_repeats = 2 if joint>0 else 5;
        else:
          if DoGmodel==1: # if if just DoG and not d-DoG-S
            n_repeats = 25 if joint>0 else 50; # was previously be 3, 15, then 7, 15
          else:
            n_repeats = 5 if joint>0 else 15; # was previously be 3, 15

        with mp.Pool(processes = nCpu) as pool:
          descr_perCell = partial(fit_descr_DoG, data_loc=data_loc, dogBase=dogBase, n_repeats=n_repeats, loss_type=loss_type, DoGmodel=DoGmodel, joint=joint, fracSig=fracSig, nBoots=nBoots, phAdj=phAdj, to_save=0, cross_val=cross_val, veThresh=veThresh);
          dogFits = pool.map(descr_perCell, range(start_cell, end_cell+1));
          pool.close();

        ### do the saving HERE!
        dogNameFinal = hf.descrFit_name(loss_type, descrBase=dogBase, modelName=hf.descrMod_name(DoGmodel), joint=joint, phAdj=phAdj);
        if os.path.isfile(data_loc + dogNameFinal):
          dogFitNPY = hf.np_smart_load(data_loc + dogNameFinal);
        else:
          dogFitNPY = dict();
          
        for iii, dogFit in enumerate(dogFits):
          dogFitNPY[iii] = dogFit;
        np.save(data_loc + dogNameFinal, dogFitNPY)
         
    else: # by cell (i.e. not multi...)
    
      if rvcModel >= 0:
        if phAdj==1:
          phase_advance_fit(cellNum, data_loc, phBase)
        rvc_fit(cellNum, data_loc, rvcBase, rvcModel, nBoots=nBoots, phAdjusted=phAdj, dir=dir); 

      if DoGmodel >= 0:
        if nBoots > 1:
          n_repeats = 2 if joint>0 else 5; # fewer if repeat
        else:
          n_repeats = 5 if joint>0 else 12; # was previously be 3, 15, then 7, 15

        fit_descr_DoG(cellNum, data_loc, dogBase, n_repeats, loss_type, DoGmodel, joint, fracSig=fracSig, nBoots=nBoots, phAdj=phAdj, cross_val=cross_val, veThresh=veThresh);

