import numpy as np
import helper_fcns as hf
import sys
import pdb

nFits = 2; # 0 (flat), 1 (weighted)

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

  return diffs_loss, diffs_pred, losses, preds, flat, weight;

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
