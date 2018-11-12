import numpy as np
import sys
import pdb

dataLoc = '../Structures/';
nCells = 59;
nDisps = 5;
nCons = 2;
nFits = 2; # 0 (flat), 1 (weighted)
to_print = 0;

lossType = int(sys.argv[1]);

# then the loss type
if lossType == 1:
  lossSuf = '_sqrt.npy';
elif lossType == 2:
  lossSuf = '_poiss.npy';
elif lossType == 3:
  lossSuf = '_modPoiss.npy';

losses = np.nan * np.zeros((nCells, nDisps, nCons, nFits));
preds = np.nan * np.zeros((nCells, nDisps, nCons, nFits));

for i in range(nCells):

  try:
    flat   = np.load(dataLoc + 'hQ_%d__flat%s' % (i+1, lossSuf)).item();
    weight = np.load(dataLoc + 'hQ_%d__wght%s' % (i+1, lossSuf)).item();

    for d in range(nDisps):

      losses[i, d, :, 0] = [np.mean(x) for x in flat['NLL'][d]];
      preds[i, d, :, 0] = [np.mean(x) for x in flat['holdoutNLL'][d]];

      losses[i, d, :, 1] = [np.mean(x) for x in weight['NLL'][d]];
      preds[i, d, :, 1] = [np.mean(x) for x in weight['holdoutNLL'][d]];

      if to_print:

        print('Cell %d' % (i+1));
        print('\thigh contrast:');
        print('\t\tloss (f|w): %.3f, %.3f' % (flat_loss[0], weight_loss[0]));  
        print('\t\tpred error (f|w): %.3f, %.3f' % (flat_pred[0], weight_pred[0]));  
        print('\tlow contrast:');
        print('\t\tloss (f|w): %.3f, %.3f' % (flat_loss[1], weight_loss[1]));  
        print('\t\tpred error (f|w): %.3f, %.3f' % (flat_pred[1], weight_pred[1]));  

  except:
    continue;  
   
diffs_loss = losses[:, :, :, 0] - losses[:, :, :, 1]
diffs_pred = preds[:, :, :, 0] - preds[:, :, :, 1]

# compute averages per "slice" of cell X disp X con
mu_byCon = np.nanmean(np.nanmean(diffs_loss, 0), 0);
mu_byCell = np.nanmean(np.nanmean(diffs_loss, 1), 1);
mu_byDisp = np.nanmean(np.nanmean(diffs_loss, 0), 1);

md_byCon = np.nanmedian(np.nanmedian(diffs_loss, 0), 0);
md_byCell = np.nanmedian(np.nanmedian(diffs_loss, 1), 1);
md_byDisp = np.nanmedian(np.nanmedian(diffs_loss, 0), 1);

pdb.set_trace();

