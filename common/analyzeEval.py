import numpy as np
import helper_fcns as hf
from model_responses import SFMsimulate, SFMGiveBof
import sys
import pdb

nFits = 2; # 0 (flat), 1 (weighted)

# this file will contain functions/routines used to compare different forms of the model, or evaluate the models for comparing between cells (or, e.g., fits in different areas)

def compute_diffs(lossType, expInd=1, baseStr='holdoutFits', date='181121'):

  _, nDisps, _, nCons, nCells, dataLoc = hf.get_exp_params(expInd);
  if expInd == 1: # main V1 branch
    dataLoc = dataLoc + 'Structures/';
  else:
    dataLoc = dataLoc + 'structures/';

  # then the loss type
  if lossType == 1:
    lossSuf = '_sqrt.npy';
  elif lossType == 2:
    lossSuf = '_poiss.npy';
  elif lossType == 3:
    lossSuf = '_modPoiss.npy';

  losses = np.nan * np.zeros((nCells, nDisps, nCons, nFits));
  preds = np.nan * np.zeros((nCells, nDisps, nCons, nFits));

  flat   = np.load(dataLoc + '%s_%s_flat%s' % (baseStr, date, lossSuf), encoding='latin1').item();
  weight = np.load(dataLoc + '%s_%s_wght%s' % (baseStr, date, lossSuf), encoding='latin1').item();

  for i in range(nCells):

    try:

      losses[i, :, :, 0] = np.mean(flat[i]['NLL'], 2)
      preds[i, :, :, 0] = np.mean(flat[i]['holdoutNLL'], 2)

      losses[i, :, :, 1] = np.mean(weight[i]['NLL'], 2)
      preds[i, :, :, 1] = np.mean(weight[i]['holdoutNLL'], 2)

    except:
      continue;  

  diffs_loss = losses[:, :, :, 0] - losses[:, :, :, 1];
  diffs_pred = preds[:, :, :, 0] - preds[:, :, :, 1];

  return diffs_loss, diffs_pred, losses, preds;

def measure_chiSq(lossType, expInd=1, date='181121'):

  _, nDisps, _, nCons, nCells, dataLoc = hf.get_exp_params(expInd);
  if expInd == 1: # main V1 branch
    dataLoc = dataLoc + 'Structures/';
  else:
    dataLoc = dataLoc + 'structures/';
  dataList = hf.np_smart_load(str(dataLoc + 'dataList.npy'));

  # then the loss type
  if lossType == 1:
    lossSuf = '_sqrt.npy';
  elif lossType == 2:
    lossSuf = '_poiss.npy';
  elif lossType == 3:
    lossSuf = '_modPoiss.npy';

  chi_wght = np.nan * np.zeros((nCells, ));
  chi_flat = np.nan * np.zeros((nCells, ));

  flat   = hf.np_smart_load(dataLoc + 'fitList_%s_flat%s' % (date, lossSuf));
  weight = hf.np_smart_load(dataLoc + 'fitList_%s_wght%s' % (date, lossSuf));

  for i in range(nCells):
    print('cell %d' % i);
    S = hf.np_smart_load(str(dataLoc + dataList['unitName'][i] + '_sfm.npy'));

    # first, the data
    _, _, sfmixExpResp, allSfMixExp = hf.organize_modResp(S['sfm']['exp']['trial']['spikeCount'], S['sfm']['exp']['trial'])    
    exp_responses = [sfmixExpResp, np.nanvar(allSfMixExp, axis=3)];

    ## then, the model
    # flat normalization
    if i in flat:
      flat_params = flat[i]['params'];
      ignore, modResp = SFMGiveBof(flat_params, S, normType=1, lossType=lossType, expInd=expInd);
      _, _, sfmixModResp, allSfMixMod = hf.organize_modResp(modResp, S['sfm']['exp']['trial'])
      mod_responses = [sfmixModResp, np.nanvar(allSfMixMod)];

      chi_flat[i] = hf.chiSq(exp_responses, mod_responses);
      print('chi: %.1f' % chi_flat[i]);

    # weighted normalization
    if i in weight:
      wght_params = weight[i]['params'];
      ignore, modResp = SFMGiveBof(wght_params, S, normType=2, lossType=lossType, expInd=expInd);
      _, _, sfmixModResp, allSfMixMod = hf.organize_modResp(modResp, S['sfm']['exp']['trial'])
      mod_responses = [sfmixModResp, np.nanvar(allSfMixMod)];

      chi_wght[i] = hf.chiSq(exp_responses, mod_responses);
      print('\tchi: %.1f' % chi_wght[i]);

  n_flat_params = len(flat[0]['params']);
  n_wght_params = len(weight[0]['params']);    

  chiNorm_flat = np.divide(chi_flat, n_flat_params);
  chiNorm_wght = np.divide(chi_flat, n_wght_params);

  chiAnalysis = dict();
  chiAnalysis['flat_norm'] = chiNorm_flat;
  chiAnalysis['wght_norm'] = chiNorm_wght;
  chiAnalysis['flat'] = chi_flat;
  chiAnalysis['wght'] = chi_wght;

  np.save('chiAnalysis', chiAnalysis);

  return chiNorm_flat, chiNorm_wght, chi_flat, chi_wght;

def evaluate_RVC(cellStructure, modParams, disp=1, sf_c=None, normType=1, expInd=1, conSteps=11):
  ''' given the cell structure, and model parameters, let's simulate an RVC to get a sense for the model's gain control strength
   optional parameters are dispersion level (default=1, i.e. single gratings)
                       and spatial frequency center (default is None, in which case we use the model's prefSF)
                       and normType, expInd
  ''' 
  lowCon = 0.01;
  highCon = 1;

  conRange = np.geomspace(lowCon, highCon, conSteps);
  rvc = np.nan * np.zeros((conSteps, ));

  if sf_c is None:
    sf_c = modParams[0]; # 0th model parameter is prefSf

  for i in range(conSteps):
    rvc[i] = SFMsimulate(modParams, cellStructure, disp, conRange[i], sf_c, unweighted=0, normType=normType, expInd=expInd)[0];

  return rvc;

def compare_gainControl(fitBase, normType, lossType, expInd, conSteps=11):
  ''' given a fitBase (str) with associated norm/loss indices, and experiment indices, get the RVC for all cells
  '''
  _, _, _, _, nCells, dir = hf.get_exp_params(expInd);

  if expInd == 1:
    dir = str(dir+'Structures/');
  else:
    dir = str(dir+'structures/');

  dataList = hf.np_smart_load(str(dir + 'dataList.npy'));

  fl_str = hf.fitList_name(fitBase, normType, lossType);
  fitList = hf.np_smart_load(str(dir + fl_str));

  rvcs = np.nan * np.zeros((nCells, conSteps));

  for i in range(nCells):
    print('cell %d' % i);
    cellStr = hf.np_smart_load(str(dir + dataList['unitName'][i] + '_sfm.npy'));
    currParams = fitList[i]['params'];

    rvcs[i, :] = evaluate_RVC(cellStr, currParams, normType=normType, expInd=expInd, conSteps=conSteps);

  if expInd == 1:
    np.save(str('rvcV1_' + fl_str), rvcs);
  elif expInd == 3:
    np.save(str('rvcLGN_' + fl_str), rvcs);

  return rvcs;

if __name__ == '__main__':

  lossType = int(sys.argv[1]);
  expInd   = int(sys.argv[2]);

  diffs_loss, diffs_pred, losses, preds = compute_diffs(lossType, expInd=expInd);

  # compute averages per "slice" of cell X disp X con
  mu_byCon = np.nanmean(np.nanmean(diffs_loss, 0), 0);
  mu_byCell = np.nanmean(np.nanmean(diffs_loss, 1), 1);
  mu_byDisp = np.nanmean(np.nanmean(diffs_loss, 0), 1);

  md_byCon = np.nanmedian(np.nanmedian(diffs_loss, 0), 0);
  md_byCell = np.nanmedian(np.nanmedian(diffs_loss, 1), 1);
  md_byDisp = np.nanmedian(np.nanmedian(diffs_loss, 0), 1);

  pdb.set_trace();
