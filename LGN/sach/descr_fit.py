import numpy as np
import helper_fcns as hf
import scipy.optimize as opt
from scipy.stats import norm, mode, lognorm, nbinom, poisson
from numpy.matlib import repmat
import os.path
import sys

import pdb

#################
### RVC
#################

def rvc_fit(cell_num, data_loc, rvcName, rvcMod=0, nBoots=0):
  ''' Piggy-backing off of phase_advance_fit above, get prepared to project the responses onto the proper phase to get the correct amplitude
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

def fit_descr_DoG(cell_num, data_loc, n_repeats = 4, loss_type = 3, DoGmodel = 1, joint=False, fracSig=0, nBoots=0, forceOverwrite=True):

    # Set up whether we will bootstrap straight away
    resample = False if nBoots <= 0 else True;
    nBoots = 1 if nBoots <= 0 else nBoots;

    if DoGmodel == 0:
      nParam = 5;
    else:
      nParam = 4;
    
    # load cell information
    dataList = hf.np_smart_load(data_loc + 'sachData.npy');
    assert dataList!=[], "data file not found!"

    fLname = 'descrFits_s210920';
    #fLname = 'descrFits_s210520';
    #fLname = 'descrFits_s210304';

    if joint==True:
      try: # load non_joint fits as a reference (see hf.dog_fit or S. Sokol thesis for details)
        modStr  = hf.descrMod_name(DoGmodel);
        fitName = hf.descrFit_name(loss_type, descrBase=fLname, modelName=modStr)
        try:
          ref_fits = hf.np_smart_load(data_loc + fitName);
        except:
          ref_fits = np.load(data_loc + fitName, allow_pickle=True).item();
        ref_varExpl = ref_fits[cell_num-1]['varExpl'];
      except:
        ref_varExpl = None;
      fLname = '%s_joint' % fLname
    else:
      ref_varExpl = None; # set to None as default

    mod_str = hf.descrMod_name(DoGmodel);
    fLname = str(data_loc + hf.descrFit_name(loss_type, fLname, mod_str));

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

    bestNLL = np.ones((nBoots, nCons)) * np.nan;
    currParams = np.ones((nBoots, nCons, nParam)) * np.nan;
    varExpl = np.ones((nBoots, nCons)) * np.nan;
    prefSf = np.ones((nBoots, nCons)) * np.nan;
    charFreq = np.ones((nBoots, nCons)) * np.nan;
    if joint==True:
      totalNLL = np.ones((nBoots, )) * np.nan;
      paramList = np.ones((nBoots, )) * np.nan;

    print('Doing the work, now');

    for boot_i in range(nBoots):

      to_unpack = hf.tabulateResponses(data, resample);
      [_, f1] = to_unpack[0];
      [_, f1arr] = to_unpack[2];

      # now, we fit!
      nll, prms, vExp, pSf, cFreq, totNLL, totPrm = hf.dog_fit(f1, all_cons, all_sfs, DoGmodel, loss_type, n_repeats, joint=joint, ref_varExpl=ref_varExpl, fracSig=fracSig);

      if resample:
        bestNLL[boot_i] = nll;
        currParams[boot_i] = prms;
        varExpl[boot_i] = vExp;
        prefSf[boot_i] = pSf;
        charFreq[boot_i] = cFreq;
        if joint==True:
          totalNLL[boot_i] = totNLL;
          paramList[boot_i] = paramList;

    ##########
    ### End of boot loop
    ##########
    # before we update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(fLname):
      print('reloading descrFits...');
      try:
        descrFits = hf.np_smart_load(fLname);
      except:
        descrFits = np.load(fLname, allow_pickle=True).item();
    if cell_num-1 not in descrFits:
      descrFits[cell_num-1] = dict(); # and previously created NaN for everything
    else: # overwrite the default NaN for everything
      if not resample: # otherwise, we do not want to update...
        bestNLL = descrFits[cell_num-1]['NLL'];
        currParams = descrFits[cell_num-1]['params'];
        varExpl = descrFits[cell_num-1]['varExpl'];
        prefSf = descrFits[cell_num-1]['prefSf'];
        charFreq = descrFits[cell_num-1]['charFreq'];
        if joint==True:
          totalNLL = descrFits[cell_num-1]['totalNLL'];
          paramList = descrFits[cell_num-1]['paramList'];

        # remove any singleton dimensions
        bestNLL = np.squeeze(bestNLL, axis=0) if bestNLL.shape[0]==1 else bestNLL;
        currParams = np.squeeze(currParams, axis=0) if currParams.shape[0]==1 else currParams;
        varExpl = np.squeeze(varExpl, axis=0) if varExpl.shape[0]==1 else varExpl;
        prefSf = np.squeeze(prefSf, axis=0) if prefSf.shape[0]==1 else prefSf;
        charFreq = np.squeeze(charFreq, axis=0) if charFreq.shape[0]==1 else charFreq;
        if joint==True:
          paramList = np.squeeze(paramList, axis=0) if paramList.shape[0]==1 else paramList;
          totalNLL = np.squeeze(totalNLL, axis=0) if totalNLL.shape[0]==1 else totalNLL;

        # now, what we do, depends on if joint or not [AGAIN -- all of this is only for not resample]
        if joint==True:
          if np.isnan(totalNLL) or totNLL < totalNLL: # then UPDATE!
            totalNLL = totNLL;
            paramList = totPrm;
            bestNLL = nll;
            currParams = prms;
            varExpl = vExp;
            prefSf = pSf;
            charFreq = cFreq;
        else:
          # must check separately for each contrast
          for con in range(nCons):
            if np.isnan(bestNLL[con]) or nll[con] < bestNLL[con] or forceOverwrite: # then UPDATE!
              bestNLL[con] = nll[con];
              currParams[con, :] = prms[con];
              varExpl[con] = vExp[con];
              prefSf[con] = pSf[con];
              charFreq[con] = cFreq[con];

    if resample:
      descrFits[cell_num-1]['boot_NLL'] = bestNLL;
      descrFits[cell_num-1]['boot_params'] = currParams;
      descrFits[cell_num-1]['boot_varExpl'] = varExpl;
      descrFits[cell_num-1]['boot_prefSf'] = prefSf;
      descrFits[cell_num-1]['boot_charFreq'] = charFreq;
      if joint==True:
        descrFits[cell_num-1]['boot_totalNLL'] = totalNLL;
        descrFits[cell_num-1]['boot_paramList'] = paramList;
    else:
      descrFits[cell_num-1]['NLL'] = bestNLL;
      descrFits[cell_num-1]['params'] = currParams;
      descrFits[cell_num-1]['varExpl'] = varExpl;
      descrFits[cell_num-1]['prefSf'] = prefSf;
      descrFits[cell_num-1]['charFreq'] = charFreq;
      if joint==True:
        descrFits[cell_num-1]['totalNLL'] = totalNLL;
        descrFits[cell_num-1]['paramList'] = paramList;

    np.save(fLname, descrFits);
    print('saving for cell ' + str(cell_num));
                
if __name__ == '__main__':

    data_loc = '/users/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/LGN/sach/structures/';
    #rvcBase = 'rvcFits_210520'
    rvcBase = 'rvcFits_210920'

    fracSig = 0; # should be unconstrained, per Tony (21.05.19) for LGN fits

    print('Running cell ' + sys.argv[1] + '...');

    cellNum   = int(sys.argv[1])
    rvcModel  = int(sys.argv[2]);
    n_repeats = int(sys.argv[3])
    loss_type = int(sys.argv[4])
    DoGmodel  = int(sys.argv[5])
    is_joint  = int(sys.argv[6]);
    nBoots    = int(sys.argv[7]);

    if rvcModel >= 0:
      rvc_fit(cellNum, data_loc, rvcBase, rvcModel, nBoots=nBoots); 
    if DoGmodel >= 0:
      fit_descr_DoG(cellNum, data_loc, n_repeats, loss_type, DoGmodel, is_joint, fracSig=fracSig, nBoots=nBoots);

