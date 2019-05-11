import numpy as np
import helper_fcns as hf
import model_responses as mod_resp
import scipy.optimize as opt
from scipy.stats import norm, mode, poisson, nbinom
import sys
import os

import pdb

### EDIT HERE
dataListName = 'dataList.npy';
#dataListName = 'dataList_glx.npy'
#dataListName = 'dataList_mr.npy'
#dataListName = 'dataList_glx_mr.npy'

rvcBaseName = 'rvcFits_f0'; # a base set of RVC fits used for initializing c50 in opt...
fitListBase = lambda s: 'holdout_fitList%s_190510cB' % s
modRecov = 0;

# TODO: make a smart way of holding out a set of conditions
# But...for now, we just default to keeping only single gratings (see below)

### now, set everything up
# load input paramaters

cellNum      = int(sys.argv[1]);
expDir       = sys.argv[2];
lossType     = int(sys.argv[3]);
fitType      = int(sys.argv[4]);
initFromCurr = int(sys.argv[5]);
trackSteps   = int(sys.argv[6]);

loc_base = os.getcwd() + '/'; # ensure there is a "/" after the final directory
loc_data = loc_base + expDir + 'structures/';

if 'pl1465' in loc_base:
  loc_str = 'HPC';
else:
  loc_str = '';

dataList = hf.np_smart_load(str(loc_data + dataListName));
dataNames = dataList['unitName'];

expInd = hf.exp_name_to_ind(dataList['expType'][cellNum-1]);

print('loading data structure from %s...' % loc_data);
S = hf.np_smart_load(str(loc_data + dataNames[cellNum-1] + '_sfm.npy')); # why -1? 0 indexing...
print('...finished loading');
trial_inf = S['sfm']['exp']['trial'];

### why have we loaded everything? so that we can create the holdout set
# ASSUMPTION: Hold out all dispersed stimuli
_, stimVals, val_con_by_disp, _, _ = hf.tabulate_responses(S, expInd);
allDisps, allCons, allSfs = stimVals;
dispVals = allDisps[1:];

holdoutCond = [];
for d in range(len(dispVals)):
  dEff = d+1;
  val_cons = val_con_by_disp[dEff];
  for c in val_cons:
    val_sfs = hf.get_valid_sfs(S, dEff, c, expInd);
    for v in val_sfs:
      holdoutCurr = [dEff, c, v];
      #print('holding out d/c/v: %d/%d/%d' % (dEff,c,v));
      #print('\tholding out d/c/v: %d/%.2f/%.2f' % (allDisps[dEff],allCons[c],allSfs[v]));
      holdoutCond.append(holdoutCurr); 
      
### NOW run it!
mod_resp.setModel(cellNum, expDir, lossType, fitType, initFromCurr, trackSteps=False, holdOutCondition=holdoutCond, modRecov=modRecov, rvcBase=rvcBaseName, fL_name=fitListBase(loc_str));
