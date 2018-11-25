import numpy as np
import sys
import pdb

dataLoc = '../Structures/';
nCells = 59;
nDisps = 5;
nCons = 2;
nFits = 2; # 0 (flat), 1 (weighted)
to_print = 0;

def compute_diffs(lossType, dataLoc=dataLoc, baseStr='holdoutFits', date='181121', nCells=nCells, nDisps=nDisps, nCons=nCons):

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

if __name__ == '__main__':

  lossType = int(sys.argv[1]);

  diffs_loss, diffs_pred, losses, preds = compute_diffs(lossType);

  # compute averages per "slice" of cell X disp X con
  mu_byCon = np.nanmean(np.nanmean(diffs_loss, 0), 0);
  mu_byCell = np.nanmean(np.nanmean(diffs_loss, 1), 1);
  mu_byDisp = np.nanmean(np.nanmean(diffs_loss, 0), 1);

  md_byCon = np.nanmedian(np.nanmedian(diffs_loss, 0), 0);
  md_byCell = np.nanmedian(np.nanmedian(diffs_loss, 1), 1);
  md_byDisp = np.nanmedian(np.nanmedian(diffs_loss, 0), 1);

  pdb.set_trace();
