import numpy as np
import pdb

dataLoc = '../Structures/';
nCells = 59;
to_print = 0;

losses = [];
preds = [];

for i in range(nCells):

  try:
    flat   = np.load(dataLoc + 'holdQuick_%d__flat_sqrt.npy' % (i+1)).item();
    weight = np.load(dataLoc + 'holdQuick_%d__wght_sqrt.npy' % (i+1)).item();

    # 0 is for only single gratings! (i.e. dispersion 0)
    flat_loss = [np.mean(x) for x in flat['NLL'][0]];
    flat_pred = [np.mean(x) for x in flat['holdoutNLL'][0]];

    weight_loss = [np.mean(x) for x in weight['NLL'][0]];
    weight_pred = [np.mean(x) for x in weight['holdoutNLL'][0]];

    losses.append(np.subtract(flat_loss, weight_loss));
    preds.append(np.subtract(flat_pred, weight_pred));

    if to_print:

      print('Cell %d' % (i+1));
      print('\thigh contrast:');
      print('\t\tloss (f|w): %.3f, %.3f' % (flat_loss[0], weight_loss[0]));  
      print('\t\tpred error (f|w): %.3f, %.3f' % (flat_pred[0], weight_pred[0]));  
      print('\tlow contrast:');
      print('\t\tloss (f|w): %.3f, %.3f' % (flat_loss[1], weight_loss[1]));  
      print('\t\tpred error (f|w): %.3f, %.3f' % (flat_pred[1], weight_pred[1]));  

  except:
    losses.append([]);
    preds.append([]);

# high contrast
hc_loss = [x[0] if x!=[] else np.nan for x in losses];
hc_pred = [x[0] if x!=[] else np.nan for x in preds];
# low contrast
lc_loss = [x[-1] if x!=[] else np.nan for x in losses];
lc_pred = [x[-1] if x!=[] else np.nan for x in preds];

pdb.set_trace();
