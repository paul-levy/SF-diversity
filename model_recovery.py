import model_responses as mod_resp
import helper_fcns as hf
import numpy as np
import os

import pdb

### In this model recovery analysis, we'll pick a set of parameters for our model neuron, simulate to the full experiment, and
### fit these model responses with the model - we aim to show the uniqueness of a given model parameterization

### In this code, we'll pick the parameters/experiments, and save the generated responses
### That means we create a dataList and accompanying *_sfm.npy files
### The actually model recovery (i.e. optimization based on simulated responses) will be by calling
### model_responses.py as usual, but with the dataList we create for model recovery 

###########
# 1. choose model parameters and experiment from which to simulate
# Note: I will use the pre/suffix mr for model_recovery
###########
print('#########   1    #########');

### TO EDIT
overwriteMR = 1; # 1 if you want to overwrite the existing MR files; 0 otherwise
expDir = ['V1_orig/'];
lossType = 4; # for fitting
inhAsym  = [0]; # we always set inhAsym = 0 (i.e. truly flat normalization)
cellNum = [29]; # which cell within the dataList (access cellNum-1)
# normType - we assume that we will fit both 
# ASSUMPTIONS: Only one dataList out? need to fix that...
dataList_in = ['dataList.npy'];
dataList_mr = 'dataList_mr.npy'
fitList_mr = 'mr_fitList190502cA';
# Now, parameters
# first, a recall of the parameters:
  # 00 = preferred spatial frequency   (cycles per degree) || [>0.05]
  # 01 = derivative order in space || [>0.1]
  # 02 = normalization constant (log10 basis) || unconstrained
  # 03 = response exponent || >1
  # 04 = response scalar || >1e-3
  # 05 = early additive noise || [0, 1]; was [0.001, 1] - see commented out line below
  # 06 = late additive noise || >0.01
  # 07 = variance of response gain || >1e-3
  # if fitType == 2
  # 08 = mean of normalization weights gaussian || [>-2]
  # 09 = std of ... || >1e-3 or >5e-1
  # if fitType == 3
  # 08 = the offset of the c50 tuning curve which is bounded between [v_sigOffset, 1] || [0, 0.75]
  # 09 = standard deviation of the gaussian to the left of the peak || >0.1
  # 10 = "" to the right "" || >0.1
  # 11 = peak (i.e. sf location) of c50 tuning curve 
# Now set params
params_1 = [2, 1, -0.465, 2, 250, 0, 0.01, 0.1, 0.9, 0.5];
params_2 = [2, 1, -0.465, 2, 250, 0, 0.01, 0.1, 0.9, 1.5];
params_3 = [2, 1, -0.200, 2, 250, 0, 0.01, 0.1, 0.9, 0.5];
params_4 = [0.5, 1, -0.200, 2, 250, 0, 0.01, 0.1, 0.9, 0.5];
params_5 = [5, 3, -0.5, 2, 250, 0, 0.01, 0.1, 0.9, 0.5];
params_6 = [5, 3, -0.5, 2, 250, 0, 0.01, 0.1, -0.5, 1.5];
params_7 = [0.5, 2, -0.200, 4, 250, 0, 0.01, 0.1, -0.1, 3.5];
params_8 = [0.5, 2, -0.200, 1, 150, 0, 0.01, 0.1, -0.5, 3.5];
params_9 = [0.5, 2, -0.200, 3, 250, 0, 0.01, 0.1, -0.1, 0.75];
params_10 = [0.5, 2, -0.200, 1.5, 550, 0, 0.01, 0.1, -0.5, 5];
params = [params_1, params_2, params_3, params_4, params_5, params_6, params_7, params_8, params_9, params_10];
nRecov = len(params);
# name it?
recovName = [];
# if you didn't name them individually, then just number them
if recovName == []:
  recovName = [str(x+1) for x in np.arange(nRecov)]
### STOP EDIT

### (FIXED)
# get the right directories
if len(expDir) == 1:
  expDirs = expDir * nRecov;
# and the right cell
if len(cellNum) == 1:
  cellNums = cellNum * nRecov;
# and the dataList
if len(dataList_in) == 1:
  dataLists = dataList_in * nRecov;
# and the inhibitory asymmetry
if len(inhAsym) == 1:
  inhAsyms = inhAsym * nRecov;

# get the directory, set up the dataList
loc_base = os.getcwd() + '/'; # ensure there is a "/" after the final directory
dL_mr = dict();
# set up the datalist components
mr_names   = [];
mr_unitArea = [];
mr_expType = [];

for param, nm, dl, expDr, cellNum, inhAsym in zip(params, recovName, dataLists, expDirs, cellNums, inhAsyms):
  loc_data = loc_base + expDr + 'structures/';
  dl_in = hf.np_smart_load(loc_data + dl);
  regFileName = dl_in['unitName'][cellNum-1] + '_sfm.npy';
  mrFileName = ('mr%s_' % nm) + regFileName;
  # copy that whole data structure, but add "mr_" at the beginning
  if not os.path.isfile(loc_data + mrFileName) or overwriteMR:
    print('moving %s to %s...' % (regFileName, mrFileName));
    mvStr = 'cp %s %s' % (loc_data + regFileName, loc_data + mrFileName);
    os.system(mvStr);
    print('...done!\n');
    curr = hf.np_smart_load(loc_data + mrFileName);
    # add a model recovery section
    curr['sfm']['mod']['recovery'] = dict();
    curr['sfm']['mod']['recovery']['paramsWght'] = param;
    baseParams = param[0:len(param)-2]; # all params except the two relating to normalization tuning
    baseParams.append(inhAsym);
    curr['sfm']['mod']['recovery']['paramsFlat'] = baseParams;
    # now, re-save
    np.save(loc_data + mrFileName, curr);
  # add to the dataList comps
  mr_names.append(mrFileName.replace('_sfm.npy', '')); # we don't include _sfm.npy in saved filenames
  mr_unitArea.append(dl_in['unitArea'][cellNum-1]);
  mr_expType.append(dl_in['expType'][cellNum-1]);

# ASSERTION/ASSUMPTION: for now, we pick just one directory to save the expDir
dL_mr['unitName'] = mr_names;
dL_mr['unitArea'] = mr_unitArea;
dL_mr['expType'] = mr_expType;
np.save(loc_base + expDirs[0] + 'structures/' + dataList_mr, dL_mr);

###########
# 2. simulate responses and save
###########
# NOTE: This will break if not all of the cells are in the same expDir...but fix this later

print('#########   2    #########');

# load dataList
dL_mr = hf.np_smart_load(loc_base + expDirs[0] + 'structures/' + dataList_mr);
nCells = len(dL_mr['unitName']);
for c in range(nCells):
  curr = hf.np_smart_load(loc_data + dL_mr['unitName'][c] + '_sfm.npy');
  recov = curr['sfm']['mod']['recovery'];
  expInd = hf.exp_name_to_ind(dL_mr['expType'][c]);              

  types = ['Wght', 'Flat'];
  paramStrs = ['params%s' % x for x in types];
  respStrs = ['resp%s' % x for x in types];
  normTypes = [2, 1];
  for (paramStr, respStr, norm) in zip(paramStrs, respStrs, normTypes):
    currResp = mod_resp.SFMGiveBof(recov[paramStr], curr, normType=norm, lossType=lossType, expInd=expInd)[1]; # 0th return is NLL
    curr['sfm']['mod']['recovery'][respStr] = np.random.poisson(currResp); 
    # simulate from poisson model - this makes integer spike counts and introduces some variability
  # now save it!
  np.save(loc_data + dL_mr['unitName'][c] + '_sfm.npy', curr);


###########
# 3. fit model
# now, run model_responses while specifying the correct dataList/fitList
###########
print('\n\n\n********YOU HAVE MADE IT TO FITTING STAGE*********\n\n');
