import numpy as np
import helper_fcns_sach as hf
import scipy.optimize as opt
from scipy.stats import norm, mode, lognorm, nbinom, poisson
from numpy.matlib import repmat
import os.path
import sys

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
  allAmp = []; allPhi = []; allCons = []; allTf = [];
  for sf_val in sfs:
    curr_amp = []; curr_phi = []; curr_tf = [];
    ok_sf = data['sf']==sf_val;
    for con_val in cons:
      which_ind = np.where(np.logical_and(data['cont']==con_val, ok_sf))[0][0];
      curr_amp.append([allAmp_mean[which_ind], allAmp_std[which_ind]]);
      curr_phi.append([allPhi_mean[which_ind], allPhi_var[which_ind]]);
      curr_tf.append([data['tf'][which_ind]]); # should be same for all conditions
    allAmp.append(curr_amp);
    allPhi.append(curr_phi);
    allCons.append(cons);
    allTf.append(curr_tf);

  phAdv_model, all_opts, all_phAdv, all_loss = hf.phase_advance(allAmp, allPhi, allCons, allTf);

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

def rvc_fit(cell_num, data_loc, rvcName, rvcMod=0, nBoots=0, phAdjusted=0, dir=1):
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

  rvcNameFinal = hf.rvc_fit_name(rvcName, rvcMod);
  # first, load the file if it already exists
  if os.path.isfile(data_loc + rvcNameFinal):
      try:
        rvcFits = hf.np_smart_load(data_loc + rvcNameFinal);
      except:
        rvcFits = np.load(data_loc + rvcNameFinal, allow_pickle=True).item();
      try:
        rvcFits_curr = rvcFits[cell_num-1];
      except:
        rvcFits_curr = None;
  else:
      rvcFits = dict();
      rvcFits_curr = None;

  print('Doing the work, now');

  # running tabulate once to know shape of experiment (e.g. nCons, nSfs)
  to_unpack = hf.tabulateResponses(data);
  [all_cons, all_sfs] = to_unpack[1];
  v_sfs = all_sfs;
  n_v_sfs = len(v_sfs);

  if resample:
    boot_loss = []; boot_params = []; boot_conGain = []; boot_varExpl = [];

  for boot_i in range(nBoots):

    to_unpack = hf.tabulateResponses(data, resample);
    [_, f1] = to_unpack[0];
    [_, f1arr] = to_unpack[2];

    resps = [f1['mean'][:, x] for x in range(n_v_sfs)];
    respsSEM = [f1['sem'][:, x] for x in range(n_v_sfs)];
    cons = [all_cons] * len(resps); # tile
    rvc_model, all_opts, all_conGain, all_loss = hf.rvc_fit(resps, cons, var=respsSEM, mod=rvcMod, prevFits=rvcFits_curr);
    varExpl = [hf.var_expl_direct(dat, hf.get_rvcResp(prms, all_cons, rvcMod)) for dat, prms in zip(resps, all_opts)];

    if resample:
      boot_loss.append(all_loss); boot_params.append(all_opts); boot_conGain.append(all_conGain); boot_varExpl.append(varExpl);

  if os.path.isfile(data_loc + rvcNameFinal):
    print('reloading rvcFits...');
    try:
      rvcFits = hf.np_smart_load(data_loc + rvcNameFinal);
    except:
      rvcFits = np.load(data_loc + rvcNameFinal, allow_pickle=True).item();

  if cell_num-1 not in rvcFits:
    rvcFits[cell_num-1] = dict();

  if resample:
    rvcFits[cell_num-1]['boot_loss'] = boot_loss;
    rvcFits[cell_num-1]['boot_params'] = boot_params;
    rvcFits[cell_num-1]['boot_conGain'] = boot_conGain;
    rvcFits[cell_num-1]['boot_varExpl'] = boot_varExpl;
  else:
    rvcFits[cell_num-1]['loss'] = all_loss;
    rvcFits[cell_num-1]['params'] = all_opts;
    rvcFits[cell_num-1]['conGain'] = all_conGain;
    rvcFits[cell_num-1]['varExpl'] = varExpl;
 
  np.save(data_loc + rvcNameFinal, rvcFits);

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
      -- TODO: Do I need this control statement or can it be cleaned up?
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

def fit_descr_DoG(cell_num, data_loc, n_repeats = 15, loss_type = 3, DoGmodel = 1, joint=0, fracSig=0, nBoots=0, forceOverwrite=False):

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

    fLname = 'descrFits%s_s220412' % HPC;
    #fLname = 'descrFits_s211206';
    #fLname = 'descrFits_s211006';

    if joint>0:
      try: # load non_joint fits as a reference (see hf.dog_fit or S. Sokol thesis for details)
        modStr  = hf.descrMod_name(DoGmodel);
        fitName = hf.descrFit_name(loss_type, descrBase=fLname, modelName=modStr, joint=0)
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
    fLname = str(data_loc + hf.descrFit_name(loss_type, fLname, mod_str, joint=joint));

    if os.path.isfile(fLname):
      try:
        descrFits = hf.np_smart_load(fLname);
      except:
        descrFits = np.load(fLname, allow_pickle=True).item();
    else:
      descrFits = dict();

    data = dataList[cell_num-1]['data'];
    
    # First, set the default values (NaN for everything); if no resample, we'll remove the singleton dimension
    to_unpack = hf.tabulateResponses(data); # a quick call to figure out how many cons
    [all_cons, all_sfs] = to_unpack[1];
    nCons = len(all_cons);
    # pre-fill the values
    bestNLL, currParams, varExpl, prefSf, charFreq, totalNLL, paramList, success = fit_descr_empties(nCons, nParam, joint, nBoots);
    
    print('Doing the work, now');

    for boot_i in range(nBoots):

      to_unpack = hf.tabulateResponses(data, resample);
      [_, f1] = to_unpack[0];
      [_, f1arr] = to_unpack[2];

      # now, we fit!
      nll, prms, vExp, pSf, cFreq, totNLL, totPrm, succ = hf.dog_fit(f1, all_cons, all_sfs, DoGmodel, loss_type, n_repeats, joint=joint, ref_varExpl=ref_varExpl, fracSig=fracSig, jointMinCons=0);
      #except: # why? Some of sach's data have NO conditions which meet the ref_varExpl threshold, so we won't be able to fit the joint model
      #  continue; # just continue, since the values will be pre-filled, anyway...

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

    # before we update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(fLname):
      print('reloading descrFits...');
      try:
        descrFits = hf.np_smart_load(fLname);
      except:
        descrFits = np.load(fLname, allow_pickle=True).item();
    if cell_num-1 not in descrFits:
      descrFits[cell_num-1] = dict(); # and previously created NaN for everything
    else:
      if not resample: # do not reload these values if it is resample...
        try: # If we made a mistake and saved boot_* first, we might not be able to do this; hence the try/except
          bestNLL = descrFits[cell_num-1]['NLL'];
          currParams = descrFits[cell_num-1]['params'];
          varExpl = descrFits[cell_num-1]['varExpl'];
          prefSf = descrFits[cell_num-1]['prefSf'];
          charFreq = descrFits[cell_num-1]['charFreq'];
          success = descrFits[cell_num-1]['success'];
          if joint>0:
            totalNLL = descrFits[cell_num-1]['totalNLL'];
            paramList = descrFits[cell_num-1]['paramList'];
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
      descrFits[cell_num-1]['boot_NLL'] = bestNLL;
      descrFits[cell_num-1]['boot_params'] = currParams;
      descrFits[cell_num-1]['boot_varExpl'] = varExpl;
      descrFits[cell_num-1]['boot_prefSf'] = prefSf;
      descrFits[cell_num-1]['boot_charFreq'] = charFreq;
      descrFits[cell_num-1]['boot_success'] = success;
      if joint>0:
        descrFits[cell_num-1]['boot_totalNLL'] = totalNLL;
        descrFits[cell_num-1]['boot_paramList'] = paramList;
    else:
      descrFits[cell_num-1]['NLL'] = bestNLL;
      descrFits[cell_num-1]['params'] = currParams;
      descrFits[cell_num-1]['varExpl'] = varExpl;
      descrFits[cell_num-1]['prefSf'] = prefSf;
      descrFits[cell_num-1]['charFreq'] = charFreq;
      descrFits[cell_num-1]['success'] = success;
      if joint>0:
        descrFits[cell_num-1]['totalNLL'] = totalNLL;
        descrFits[cell_num-1]['paramList'] = paramList;

    np.save(fLname, descrFits);
    print('saving for cell ' + str(cell_num));
                
if __name__ == '__main__':

    basePath = os.getcwd() + '/';
    data_suff = 'structures/';
    data_loc = basePath + data_suff;

    HPC = 'HPC' if 'pl1465' in data_loc else '';

    #rvcBase = 'rvcFits%s_220219' % HPC;
    rvcBase = 'rvcFits%s_220412' % HPC;
    phBase = 'phAdv%s_220412' % HPC;

    fracSig = 0; # should be unconstrained, per Tony (21.05.19) for LGN fits

    print('Running cell ' + sys.argv[1] + '...');

    cellNum   = int(sys.argv[1])
    rvcModel  = int(sys.argv[2]);
    loss_type = int(sys.argv[3])
    DoGmodel  = int(sys.argv[4])
    joint     = int(sys.argv[5]);
    nBoots    = int(sys.argv[6]);

    if rvcModel >= 0:
      phase_advance_fit(cellNum, data_loc, phBase)
      rvc_fit(cellNum, data_loc, rvcBase, rvcModel, nBoots=nBoots); 

    if DoGmodel >= 0:
      if nBoots > 1:
        n_repeats = 2 if joint>0 else 5; # fewer if repeat
      else:
        n_repeats = 5 if joint>0 else 12; # was previously be 3, 15, then 7, 15

      fit_descr_DoG(cellNum, data_loc, n_repeats, loss_type, DoGmodel, joint, fracSig=fracSig, nBoots=nBoots);

