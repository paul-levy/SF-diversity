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

def rvc_fit(cell_num, data_loc, rvcName, rvcMod=0):
  ''' Piggy-backing off of phase_advance_fit above, get prepared to project the responses onto the proper phase to get the correct amplitude
      Then, with the corrected response amplitudes, fit the RVC model
      - as of 19.11.07, we will fit non-baseline subtracted responses 
          (F1 have baseline of 0 always, but now we will not subtract baseline from F0 responses)
  '''

  # load cell information
  dataList = hf.np_smart_load(data_loc + 'sachData.npy');
  assert dataList!=[], "data file not found!"
  data = dataList[cell_num-1]['data'];

  rvcNameFinal = hf.rvc_fit_name(rvcName, rvcMod);
  # first, load the file if it already exists
  if os.path.isfile(data_loc + rvcNameFinal):
      rvcFits = hf.np_smart_load(data_loc + rvcNameFinal);
      try:
        rvcFits_curr = rvcFits[cell_num-1];
      except:
        rvcFits_curr = None;
  else:
      rvcFits = dict();
      rvcFits_curr = None;

  print('Doing the work, now');

  to_unpack = hf.tabulateResponses(data);
  [_, f1] = to_unpack[0];
  [all_cons, all_sfs] = to_unpack[1];
  [_, f1arr] = to_unpack[2];

  v_sfs = all_sfs;
  n_v_sfs = len(v_sfs);

  resps = [f1['mean'][:, x] for x in range(n_v_sfs)];
  respsSEM = [f1['sem'][:, x] for x in range(n_v_sfs)];
  cons = [all_cons] * len(resps); # tile

  rvc_model, all_opts, all_conGain, all_loss = hf.rvc_fit(resps, cons, var=respsSEM, mod=rvcMod, prevFits=rvcFits_curr);

  if os.path.isfile(data_loc + rvcNameFinal):
    print('reloading rvcFits...');
    rvcFits = hf.np_smart_load(data_loc + rvcNameFinal);
  if cell_num-1 not in rvcFits:
    rvcFits[cell_num-1] = dict();
    rvcFits[cell_num-1] = dict();

  rvcFits[cell_num-1]['loss'] = all_loss;
  rvcFits[cell_num-1]['params'] = all_opts;
  rvcFits[cell_num-1]['conGain'] = all_conGain;

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

def fit_descr_DoG(cell_num, data_loc, n_repeats = 4, loss_type = 3, DoGmodel = 1, joint=False):

    nParam = 4;
    
    # load cell information
    dataList = hf.np_smart_load(data_loc + 'sachData.npy');
    assert dataList!=[], "data file not found!"

    fLname = 'descrFits_s191201';
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

    data = dataList[cell_num-1]['data'];
    
    print('Doing the work, now');

    to_unpack = hf.tabulateResponses(data);
    [_, f1] = to_unpack[0];
    [all_cons, all_sfs] = to_unpack[1];
    [_, f1arr] = to_unpack[2];
    
    nCons = len(all_cons);

    # then, set the default values (NaN for everything)
    bestNLL = np.ones((nCons)) * np.nan;
    currParams = np.ones((nCons, nParam)) * np.nan;
    varExpl = np.ones((nCons)) * np.nan;
    prefSf = np.ones((nCons)) * np.nan;
    charFreq = np.ones((nCons)) * np.nan;
    if joint==True:
      totalNLL = np.nan;
      paramList = np.nan;

    # now, we fit!
    nll, prms, vExp, pSf, cFreq, totNLL, totPrm = hf.dog_fit(f1, all_cons, all_sfs, DoGmodel, loss_type, n_repeats, joint=joint);

    # before we update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(fLname):
      print('reloading descrFits...');
      descrFits = hf.np_smart_load(fLname);
    if cell_num-1 not in descrFits:
      descrFits[cell_num-1] = dict(); # and previously created NaN for everything
    else: # overwrite the default NaN for everything
      bestNLL = descrFits[cell_num-1]['NLL'];
      currParams = descrFits[cell_num-1]['params'];
      varExpl = descrFits[cell_num-1]['varExpl'];
      prefSf = descrFits[cell_num-1]['prefSf'];
      charFreq = descrFits[cell_num-1]['charFreq'];
      if joint==True:
        totalNLL = descrFits[cell_num-1]['totalNLL'];
        paramList = descrFits[cell_num-1]['paramList'];

    # now, what we do, depends on if joint or not
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
        if np.isnan(bestNLL[con]) or nll[con] < bestNLL[con]: # then UPDATE!
          bestNLL[con] = nll[con];
          currParams[con, :] = prms[con];
          varExpl[con] = vExp[con];
          prefSf[con] = pSf[con];
          charFreq[con] = cFreq[con];

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
    rvcBase = 'rvcFits_191201'

    print('Running cell ' + sys.argv[1] + '...');

    cellNum = int(sys.argv[1])
    rvcModel = int(sys.argv[2]);
    n_repeats = int(sys.argv[3])
    loss_type = int(sys.argv[4])
    DoGmodel = int(sys.argv[5])
    is_joint = int(sys.argv[6]);

    #rvc_fit(cellNum, data_loc, rvcBase, rvcModel); 
    fit_descr_DoG(cellNum, data_loc, n_repeats, loss_type, DoGmodel, is_joint);

