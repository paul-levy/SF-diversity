# coding: utf-8

#### NOTE: Based on plot_diagnose_vLGN.py (and by extension, plot_sfBB.py)
## Here, we'll make each contrast (or sf) a separate plot
# --- thus, we'll have 6 columns and nCon or nSf rows (i.e. 7)
# --- --- the 6 columns come from the 3 rows each for DC and F1 plots (as in plot_sfBB.py)

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns
sns.set(style='ticks')
from scipy.stats import poisson, nbinom
from scipy.stats.mstats import gmean

import helper_fcns as hf
import helper_fcns_sfBB as hf_sf
import model_responses_pytorch as mrpt

import warnings
warnings.filterwarnings('once');

import pdb

# using fits where the filter sigma is sigmoid?
_sigmoidRespExp = None; # 3 or None, as of 21.03.14
_sigmoidSigma = 5; # put a value (5) or None (see model_responses_pytorch.py for details)
_sigmoidGainNorm = 5;
recenter_norm = 1; # recenter the tuned normalization around 1?
useCoreFit = 1; # if useCoreFit, then we'll plot the model response to the sfBB_var* experiments, if applicable
_globalMin = 1e-10;

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_style.mplstyle');
from matplotlib import rcParams

###
for i in range(2):
    # must run twice for changes to take effect?
    from matplotlib import rcParams, cm
    rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = ['Helvetica']
    rcParams['font.style'] = 'oblique'
    rcParams['font.size'] = 30;
    rcParams['pdf.fonttype'] = 3 # should be 42, but there are kerning issues
    rcParams['ps.fonttype'] = 3 # should be 42, but there are kerning issues
    rcParams['lines.linewidth'] = 3;
    rcParams['lines.markeredgewidth'] = 0; # remove edge??                                                                                                                               
    rcParams['axes.linewidth'] = 3;
    rcParams['lines.markersize'] = 12; # 8 is the default                                                                                                                                
    rcParams['font.style'] = 'oblique';

    rcParams['xtick.major.size'] = 25
    rcParams['xtick.minor.size'] = 12
    rcParams['ytick.major.size'] = 25
    rcParams['ytick.minor.size'] = 0; # i.e. don't have minor ticks on y...                                                                                                              

    rcParams['xtick.major.width'] = 2
    rcParams['xtick.minor.width'] = 2
    rcParams['ytick.major.width'] = 2
    rcParams['ytick.minor.width'] = 0

cellNum  = int(sys.argv[1]);
excType  = int(sys.argv[2]);
lossType = int(sys.argv[3]);
expDir   = sys.argv[4]; 
normTypesIn = int(sys.argv[5]); # two-digit number, extracting 1st for modA, 2nd for modB
conTypesIn = int(sys.argv[6]); # two-digit number, extracting 1st for modA, 2nd for modB
lgnFrontEnd = int(sys.argv[7]); # two-digit number, extracting 1st for modA, 2nd for modB
diffPlot = int(sys.argv[8]);
intpMod  = int(sys.argv[9]);
kMult  = float(sys.argv[10]);
vecCorrected = int(sys.argv[11]);
onsetTransient = int(sys.argv[12]);
onsetMod = 1;

if len(sys.argv) > 13:
  fixRespExp = float(sys.argv[13]);
  if fixRespExp <= 0: # this is the code to not fix the respExp
    fixRespExp = None;
else:
  fixRespExp = None; # default (see modCompare.ipynb for details)

if len(sys.argv) > 14:
  respVar = int(sys.argv[14]);
else:
  respVar = 1;

if len(sys.argv) > 15:
  useHPCfit = int(sys.argv[15]);
else:
  useHPCfit = 0;
if len(sys.argv) > 16:
  whichKfold = int(sys.argv[16]);
  if whichKfold<0:
    whichKfold = None
else:
  whichKfold = None;
isCV = False if whichKfold is None else True;

## Unlikely to be changed, but keep flexibility
baselineSub = 0;
fix_ylim = 0;
singleGratsOnly = False

## used for interpolation plot
sfSteps  = 45; # i.e. how many steps between bounds of interest
conSteps = -1;
#nRpts    = 100; # how many repeats for stimuli in interpolation plot?
nRpts    = 5; # how many repeats for stimuli in interpolation plot?
#nRpts    = 3000; # how many repeats for stimuli in interpolation plot? USE FOR PUBLICATION/PRESENTATION QUALITY, but SLOW
nRptsSingle = 5; # when disp = 1 (which is most cases), we do not need so many interpolated points

loc_base = os.getcwd() + '/';
data_loc = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + 'figures/';

if 'pl1465' in loc_base or useHPCfit:
  loc_str = 'HPC';
else:
  loc_str = '';

if _sigmoidRespExp is not None:
  rExpStr = 're';
else:
  rExpStr = '';

### DATALIST
expName = hf.get_datalist(expDir);
### ONSETS
if onsetTransient > 0:
  onsetDur = onsetTransient;
  halfWidth = 15; # by default, we'll just use 15 ms...
  onset_key = (onsetDur, halfWidth);
  try:
    if onsetMod == 0:
      onsetMod_str = '';
    elif onsetMod == 1:
      onsetMod_str = '_zeros'
    onsetTransients = hf.np_smart_load(data_loc + 'onset_transients%s.npy' % onsetMod_str); # here's the set of all onset transients
    onsetCurr = onsetTransients[cellNum-1][onset_key]['transient'];
  except:
    onsetCurr = None
else:
  onsetCurr = None;

### FITLIST
_applyLGNtoNorm = 0;
# -- some params are sigmoid, we'll use this to unpack the true parameter
_sigmoidScale = 10
_sigmoidDord = 5;
if excType==1 or excType == 2:
  #fitBase = 'fitList%s_pyt_nr221109e_noRE_noSched%s' % (loc_str, '_sg' if singleGratsOnly else '') 
  #fitBase = 'fitList%s_pyt_nr221109e_noSched%s' % (loc_str, '_sg' if singleGratsOnly else '') 
  #fitBase = 'fitList%s_pyt_nr221119d_noRE_noSched%s' % (loc_str, '_sg' if singleGratsOnly else '') 
  fitBase = 'fitList%s_pyt_nr230104_noRE_noSched%s' % (loc_str, '_sg' if singleGratsOnly else '') 
else:
  fitBase = None;

if fitBase is not None:
  _CV=isCV
  if vecCorrected:
    vecCorrected = 1;
  else:
    vecCorrected = 0;

  ### Model types
  # 0th: Unpack the norm types, con types, lgnTypes
  normA, normB = int(np.floor(normTypesIn/10)), np.mod(normTypesIn, 10)
  conA, conB = int(np.floor(conTypesIn/10)), np.mod(conTypesIn, 10)
  lgnA, lgnB = int(np.floor(lgnFrontEnd/10)), np.mod(lgnFrontEnd, 10)

  fitNameA = hf.fitList_name(fitBase, normA, lossType, lgnA, conA, 0, fixRespExp=fixRespExp, kMult=kMult, excType=excType, CV=_CV)
  fitNameB = hf.fitList_name(fitBase, normB, lossType, lgnB, conB, 0, fixRespExp=fixRespExp, kMult=kMult, excType=excType, CV=_CV)
  # what's the shorthand we use to refer to these models...
  wtStr = 'wt';
  # -- the following two lines assume that we only use wt (norm=2) or wtGain (norm=5)
  aWtStr = 'wt%s' % ('' if normA==2 else 'Gn');
  bWtStr = 'wt%s' % ('' if normB==2 else 'Gn');
  lgnStrA = hf.lgnType_suffix(lgnA, conA);
  lgnStrB = hf.lgnType_suffix(lgnB, conB);
  modA_str = '%s%s' % ('fl' if normA==1 else aWtStr, lgnStrA if lgnA>0 else 'V1');
  modB_str = '%s%s' % ('fl' if normB==1 else bWtStr, lgnStrB if lgnB>0 else 'V1');

  fitListA = hf.np_smart_load(data_loc + fitNameA);
  fitListB = hf.np_smart_load(data_loc + fitNameB);

  try:
    # using context manager approach to save memory
    with hf.np_smart_load(data_loc + fitNameA.replace('.npy', '_details.npy')) as fit_detailsA_all:
      fit_detailsA = fit_detailsA_all[cellNum-1];
    with hf.np_smart_load(data_loc + fitNameB.replace('.npy', '_details.npy')) as fit_detailsB_all:
      fit_detailsB = fit_detailsB_all[cellNum-1];
  except:
    fit_detailsA = None; fit_detailsB = None;

  dc_str = hf_sf.get_resp_str(respMeasure=0);
  f1_str = hf_sf.get_resp_str(respMeasure=1);

  #modFit_A_dc = fitListA[cellNum-1][dc_str]['params'];
  #modFit_B_dc = fitListB[cellNum-1][dc_str]['params'];
  #modFit_A_f1 = fitListA[cellNum-1][f1_str]['params'];
  #modFit_B_f1 = fitListB[cellNum-1][f1_str]['params'];
  #lossVals = [[x[cellNum-1][y]['NLL'] for x in [fitListA, fitListB]] for y in [dc_str, f1_str]]

  modFit_A_dc = fitListA[cellNum-1][dc_str]['params'][whichKfold] if _CV else fitListA[cellNum-1][dc_str]['params'];
  modFit_B_dc = fitListB[cellNum-1][dc_str]['params'][whichKfold] if _CV else fitListB[cellNum-1][dc_str]['params'];
  modFit_A_f1 = fitListA[cellNum-1][f1_str]['params'][whichKfold] if _CV else fitListA[cellNum-1][f1_str]['params'];
  modFit_B_f1 = fitListB[cellNum-1][f1_str]['params'][whichKfold] if _CV else fitListB[cellNum-1][f1_str]['params'];
  if _CV:
      lossVals = [[np.mean(x[cellNum-1][y]['NLL%s' % ('_train' if isCV else '')][whichKfold]) for x in [fitListA, fitListB]] for y in [dc_str, f1_str]]
  else:
      lossVals = [[np.mean(x[cellNum-1][y]['NLL']) for x in [fitListA, fitListB]] for y in [dc_str, f1_str]]
  kstr = '_k%d' % whichKfold if _CV else '';


  normTypes = [normA, normB];
  lgnTypes = [lgnA, lgnB];
  conTypes = [conA, conB];

  expInd = -1;
  newMethod = 1;

  mod_A_dc  = mrpt.sfNormMod(modFit_A_dc, expInd=expInd, excType=excType, normType=normTypes[0], lossType=lossType, lgnFrontEnd=lgnTypes[0], newMethod=newMethod, lgnConType=conTypes[0], applyLGNtoNorm=_applyLGNtoNorm, toFit=False, normFiltersToOne=False)
  mod_B_dc = mrpt.sfNormMod(modFit_B_dc, expInd=expInd, excType=excType, normType=normTypes[1], lossType=lossType, lgnFrontEnd=lgnTypes[1], newMethod=newMethod, lgnConType=conTypes[1], applyLGNtoNorm=_applyLGNtoNorm, toFit=False, normFiltersToOne=False)
  mod_A_f1  = mrpt.sfNormMod(modFit_A_f1, expInd=expInd, excType=excType, normType=normTypes[0], lossType=lossType, lgnFrontEnd=lgnTypes[0], newMethod=newMethod, lgnConType=conTypes[0], applyLGNtoNorm=_applyLGNtoNorm, toFit=False, normFiltersToOne=False)
  mod_B_f1 = mrpt.sfNormMod(modFit_B_f1, expInd=expInd, excType=excType, normType=normTypes[1], lossType=lossType, lgnFrontEnd=lgnTypes[1], newMethod=newMethod, lgnConType=conTypes[1], applyLGNtoNorm=_applyLGNtoNorm, toFit=False, normFiltersToOne=False)

  # get varGain values...
  if lossType == 3: # i.e. modPoiss
    varGains  = [x[7] for x in [modFit_A_dc, modFit_A_f1]];
    varGains_A = [1/(1+np.exp(-x)) for x in varGains];
    varGains  = [x[7] for x in [modFit_B_dc, modFit_B_f1]];
    varGains_B = [1/(1+np.exp(-x)) for x in varGains];
  else:
    varGains_A = [-99, -99]; # just dummy values; won't be used unless losstype=3
    varGains_B = [-99, -99]; # just dummy values; won't be used unless losstype=3

else: # we will just plot the data
  fitList_fl = None;
  fitList_wg = None;

# set the save directory to save_loc, then create the save directory if needed
if onsetCurr is None:
  onsetStr = '';
else:
  onsetStr = '_onset%s_%03d_%03d' % (onsetMod_str, onsetDur, halfWidth)

if fitBase is not None:
  lossSuf = hf.lossType_suffix(lossType).replace('.npy', ''); # get the loss suffix, remove the file type ending
  excType_str = hf.excType_suffix(excType);
  if diffPlot == 1: 
    compDir  = str(fitBase + '_diag%s_%s_%s' % (excType_str, modA_str, modB_str) + lossSuf + '/diff');
  else:
    compDir  = str(fitBase + '_diag%s_%s_%s' % (excType_str, modA_str, modB_str) + lossSuf);
  if intpMod == 1:
    compDir = str(compDir + '/intp');
  subDir   = compDir.replace('fitList', 'fits').replace('.npy', '');
  save_loc = str(save_loc + subDir + '/sep/');
else:
  save_loc = str(save_loc + 'data_only/sep/');

if not os.path.exists(save_loc):
  os.makedirs(save_loc);

conDig = 3; # round contrast to the 3rd digit
dataList = hf.np_smart_load(data_loc + expName)

if fix_ylim == 1:
    ylim_flag = '_fixed';
else:
    ylim_flag = ''

#####################
### sfBB_core plotting
#####################

expName = 'sfBB_core';

unitNm = dataList['unitName'][cellNum-1];
cell = hf.np_smart_load('%s%s_sfBB.npy' % (data_loc, unitNm));
expInfo = cell[expName]
byTrial = expInfo['trial'];
f1f0_rat = hf_sf.compute_f1f0(expInfo)[0];

### Now, if we've got the models, get and organize those responses...
if fitBase is not None:
  trInf_dc, _ = mrpt.process_data(expInfo, expInd=expInd, respMeasure=0); 
  trInf_f1, _ = mrpt.process_data(expInfo, expInd=expInd, respMeasure=1); 
  val_trials = trInf_dc['num']; # these are the indices of valid, original trials

  resp_A_dc  = mod_A_dc.forward(trInf_dc, respMeasure=0, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm).detach().numpy();
  resp_B_dc = mod_B_dc.forward(trInf_dc, respMeasure=0, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm).detach().numpy();
  resp_A_f1  = mod_A_f1.forward(trInf_f1, respMeasure=1, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm).detach().numpy();
  resp_B_f1 = mod_B_f1.forward(trInf_f1, respMeasure=1, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm).detach().numpy();

  # now get the mask+base response (f1 at base TF)
  maskInd, baseInd = hf_sf.get_mask_base_inds();

  # note the indexing: [1][x][0][0] for [summary], [dc||f1], [unpack], [mean], respectively
  baseMean_mod_dc = [hf_sf.get_baseOnly_resp(expInfo, dc_resp=x, val_trials=val_trials)[1][0][0][0] for x in [resp_A_dc, resp_B_dc]];
  baseMean_mod_f1 = [hf_sf.get_baseOnly_resp(expInfo, f1_base=x[:,baseInd], val_trials=val_trials)[1][1][0][0] for x in [resp_A_f1, resp_B_f1]];

  # ---- model A responses
  respMatrix_A_dc, respMatrix_A_f1 = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, dc_resp=resp_A_dc, f1_base=resp_A_f1[:,baseInd], f1_mask=resp_A_f1[:,maskInd], val_trials=val_trials, vecCorrectedF1=0); # i.e. get the base response for F1
  # and get the mask only response (f1 at mask TF)
  respMatrix_A_dc_onlyMask, respMatrix_A_f1_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, dc_resp=resp_A_dc, f1_base=resp_A_f1[:,baseInd], f1_mask=resp_A_f1[:,maskInd], val_trials=val_trials, vecCorrectedF1=0); # i.e. get the maskONLY response
  # and get the mask+base response (but f1 at mask TF)
  _, respMatrix_A_f1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, dc_resp=resp_A_dc, f1_base=resp_A_f1[:,baseInd], f1_mask=resp_A_f1[:,maskInd], val_trials=val_trials, vecCorrectedF1=0); # i.e. get the maskONLY response
  # ---- model B responses
  respMatrix_B_dc, respMatrix_B_f1 = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, dc_resp=resp_B_dc, f1_base=resp_B_f1[:,baseInd], f1_mask=resp_B_f1[:,maskInd], val_trials=val_trials, vecCorrectedF1=0); # i.e. get the base response for F1
  # and get the mask only response (f1 at mask TF)
  respMatrix_B_dc_onlyMask, respMatrix_B_f1_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, dc_resp=resp_B_dc, f1_base=resp_B_f1[:,baseInd], f1_mask=resp_B_f1[:,maskInd], val_trials=val_trials, vecCorrectedF1=0); # i.e. get the maskONLY response
  # and get the mask+base response (but f1 at mask TF)
  _, respMatrix_B_f1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, dc_resp=resp_B_dc, f1_base=resp_B_f1[:,baseInd], f1_mask=resp_B_f1[:,maskInd], val_trials=val_trials, vecCorrectedF1=0); # i.e. get the maskONLY response

### Get the responses - base only, mask+base [base F1], mask only (mask F1)
baseDistrs, baseSummary, baseConds = hf_sf.get_baseOnly_resp(expInfo);
# - unpack DC, F1 distribution of responses per trial
baseDC, baseF1 = baseDistrs;
baseDC_mn, baseF1_mn = np.mean(baseDC), np.mean(baseF1);
if vecCorrected:
    baseDistrs, baseSummary, _ = hf_sf.get_baseOnly_resp(expInfo, vecCorrectedF1=1, onsetTransient=onsetCurr);
    baseF1_mn = baseSummary[1][0][0,:]; # [1][0][0,:] is r,phi mean
    baseF1_var = baseSummary[1][0][1,:]; # [1][0][0,:] is r,phi std/(circ.) var
    baseF1_r, baseF1_phi = baseDistrs[1][0][0], baseDistrs[1][0][1];
# - unpack the SF x CON of the base (guaranteed to have only one set for sfBB_core)
baseSf_curr, baseCon_curr = baseConds[0];
# now get the mask+base response (f1 at base TF)
respMatrixDC, respMatrixF1 = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, vecCorrectedF1=vecCorrected, onsetTransient=onsetCurr); # i.e. get the base response for F1
# and get the mask only response (f1 at mask TF)
respMatrixDC_onlyMask, respMatrixF1_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, vecCorrectedF1=vecCorrected, onsetTransient=onsetCurr); # i.e. get the maskONLY response
# and get the mask+base response (but f1 at mask TF)
_, respMatrixF1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, vecCorrectedF1=vecCorrected, onsetTransient=onsetCurr); # i.e. get the maskONLY response
#_, rMF1_mTf_noVec = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, vecCorrectedF1=0, onsetTransient=onsetCurr);

# -- if vecCorrected, let's just take the "r" elements, not the phi information
if vecCorrected:
    respMatrixF1 = respMatrixF1[:,:,0,:]; # just take the "r" information (throw away the phi)
    respMatrixF1_onlyMask = respMatrixF1_onlyMask[:,:,0,:]; # just take the "r" information (throw away the phi)
    respMatrixF1_maskTf = respMatrixF1_maskTf[:,:,0,:]; # just take the "r" information (throw away the phi)

## Reference tuning...
refDC, refF1 = hf_sf.get_mask_resp(expInfo, withBase=0, vecCorrectedF1=vecCorrected, onsetTransient=onsetCurr); # i.e. mask only, at mask TF
maskSf, maskCon = expInfo['maskSF'], expInfo['maskCon'];
# - get DC tuning curves
refDC_sf = refDC[-1, :, :]; # highest contrast
prefSf_ind = np.argmax(refDC_sf[:, 0]);
prefSf_DC = maskSf[prefSf_ind];
refDC_rvc = refDC[:, prefSf_ind, :];
# - get F1 tuning curves (adjust for vecCorrected?)
if vecCorrected: # get only r, not phi
    refF1 = refF1[:,:,0,:];
refF1_sf = refF1[-1, :, :];
prefSf_ind = np.argmax(refF1_sf[:, 0]);
prefSf_F1 = maskSf[prefSf_ind];
refF1_rvc = refF1[:, prefSf_ind, :];

########################
### Now, plot
########################

# set up model plot info
# i.e. flat model is red, weighted model is green
modColors = ['g', 'r']
try:
  modLabels = ['A: %s' % modA_str, 'B: %s' % modB_str]
except:
  modLabels = None

maxResp = np.maximum(np.max(respMatrixDC), np.max(respMatrixF1));
maxResp_onlyMask = np.maximum(np.max(respMatrixDC_onlyMask), np.max(respMatrixF1_onlyMask));
maxResp_total = np.maximum(maxResp, maxResp_onlyMask);
overall_ylim = [0, 1.2*maxResp_total];
# also get the bounds for the AbLe plot - only DC
AbLe_mn = np.nanmin(respMatrixDC[:,:,0]-baseDC_mn-respMatrixDC_onlyMask[:,:,0])
AbLe_mx = np.nanmax(respMatrixDC[:,:,0]-baseDC_mn-respMatrixDC_onlyMask[:,:,0])
AbLe_bounds = [np.sign(AbLe_mn)*1.2*np.abs(AbLe_mn), np.maximum(5, 1.2*AbLe_mx)]; # ensure we go at least above 0 with the max

# --- first SF
nrow, ncol = len(maskCon), 6;
fSf, axSf = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol*10, nrow*14))

#fSf.suptitle('V1 #%d [%s, f1f0: %.2f] base: %.2f cpd, %.2f%%' % (cellNum, unitNm, f1f0_rat, baseSf_curr, baseCon_curr), fontsize='x-large');
# --- then Con
nrow, ncol = len(maskCon), 6;
fCon, axCon = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol*10, nrow*14))

# --- initialize an array for storing variance explained across all conditions in a given fit
vE_A = np.nan*np.zeros((3,2)); # for diff. resp types and [f0,f1]
vE_B = np.nan*np.zeros((3,2)); # for diff. resp types and [f0,f1]
######

for measure in [0,1]:
    if measure == 0:
        baseline = expInfo['blank']['mean'];
        data = respMatrixDC;
        data_baseTf = None;
        maskOnly = respMatrixDC_onlyMask;
        baseOnly = baseDC
        refAll = refDC[:,:,0];
        refSf = refDC_sf;
        refRVC = refDC_rvc;
        refSf_pref = prefSf_DC;
        if baselineSub:
            data -= baseline
            baseOnly -= baseline;
        xlim_base = overall_ylim;
        ylim_diffsAbLe = AbLe_bounds;
        lbl = 'DC' 
        if fitBase is not None:
          data_A = respMatrix_A_dc;
          data_B = respMatrix_B_dc;
          data_A_onlyMask = respMatrix_A_dc_onlyMask;
          data_B_onlyMask = respMatrix_B_dc_onlyMask;
          data_A_baseTf = None;
          data_B_baseTf = None;
          mod_mean_A = baseMean_mod_dc[0];
          mod_mean_B = baseMean_mod_dc[1];
    elif measure == 1:
        data = respMatrixF1_maskTf;
        data_baseTf = respMatrixF1;
        maskOnly = respMatrixF1_onlyMask;
        if vecCorrected:
            mean_r, mean_phi = baseF1_mn;
            std_r, var_phi = baseF1_var;
            vec_r, vec_phi = baseF1_r, baseF1_phi;
        refAll = refF1[:,:,0];
        refSf = refF1_sf;
        refRVC = refF1_rvc;
        refSf_pref = prefSf_F1;
        xlim_base = overall_ylim
        lbl = 'F1'
        if fitBase is not None:
          data_A = respMatrix_A_f1_maskTf;
          data_B = respMatrix_B_f1_maskTf;
          data_A_onlyMask = respMatrix_A_f1_onlyMask;
          data_B_onlyMask = respMatrix_B_f1_onlyMask;
          data_A_baseTf = respMatrix_A_f1;
          data_B_baseTf = respMatrix_B_f1;
          mod_mean_A = baseMean_mod_f1[0];
          mod_mean_B = baseMean_mod_f1[1];

    # Now, subtract the baseOnly response from the base+mask response (only used if measure=0, i.e. DC)
    # -- but store it separately 
    if measure == 0:
        data_sub = np.copy(data);
        data_sub[:,:,0] = data[:,:,0]-np.mean(baseOnly);
        if fitBase is not None:
            data_A_sub = np.copy(data_A);
            data_A_sub[:,:,0] = data_A[:,:,0] - mod_mean_A;
            data_B_sub = np.copy(data_B);
            data_B_sub[:,:,0] = data_B[:,:,0] - mod_mean_B;


    ### --- First, SF tuning across contrast
    resps = [maskOnly, data, data_baseTf]; #need to plot data_baseTf for f1
    if fitBase is not None:
      modA_resps = [data_A_onlyMask, data_A, data_A_baseTf];
      modB_resps = [data_B_onlyMask, data_B, data_B_baseTf];
    labels = ['mask', 'mask+base', 'mask+base']
    measure_lbl = np.vstack((['', '', ''], ['', ' (mask TF)', ' (base TF)'])); # specify which TF, if F1 response
    labels_ref = ['blank', 'base']
    if vecCorrected == 1 and measure == 1:
      base_mn = mean_r;
    else:
      nResps = len(baseOnly[0]); # unpack the array for true length
      base_mn, base_sem = np.mean(baseOnly), np.std(baseOnly)/np.sqrt(nResps); 

    floors = [baseline, base_mn]; # i.e. the blank response, then the response to the base alone

    for ii, rsps in enumerate(resps): # first mask only, then mask+base (data)

        try:
          vE_A[ii,measure] = hf.var_explained(hf.nan_rm(rsps[:,:,0]), hf.nan_rm(modA_resps[ii][:,:,0]), None);
        except:
          pass
        try:
          vE_B[ii,measure] = hf.var_explained(hf.nan_rm(rsps[:,:,0]), hf.nan_rm(modB_resps[ii][:,:,0]), None);
        except:
          pass

        nCons = len(maskCon);
        # we don't plot the F1 at base TF for DC response...
        if measure == 0 and ii == (len(resps)-1):
            continue;

        for mcI, mC in enumerate(maskCon):
          pltInd = nCons-mcI-1;
          col = [(nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons)];
          # PLOT THE DATA
          axSf[pltInd, ii+3*measure].errorbar(maskSf, rsps[mcI,:,0], rsps[mcI,:,1], fmt='o', clip_on=False,
                                              color=col, label=str(np.round(mC, 2)) + '%')
          if fitBase is None: # then just plot a line for the data
            axSf[pltInd, ii+3*measure].plot(maskSf, rsps[mcI,:,0], clip_on=False, color=col)
          else:
            varExpl_A = hf.var_explained(hf.nan_rm(rsps[mcI,:,0]), hf.nan_rm(modA_resps[ii][mcI,:,0]), None);
            varExpl_B = hf.var_explained(hf.nan_rm(rsps[mcI,:,0]), hf.nan_rm(modB_resps[ii][mcI,:,0]), None);
            # PLOT model A (if present)
            axSf[pltInd, ii+3*measure].plot(maskSf, modA_resps[ii][mcI,:,0], color=modColors[0], alpha=1-col[0], label='%.2f' % varExpl_A)
            # PLOT model B (if present)
            axSf[pltInd, ii+3*measure].plot(maskSf, modB_resps[ii][mcI,:,0], color=modColors[1], alpha=1-col[0], label='%.2f' % varExpl_B)

          axSf[pltInd, ii+3*measure].set_xscale('log');
          if pltInd == 0:
            axSf[pltInd, ii+3*measure].set_xlabel('SF (c/deg)')
            axSf[pltInd, ii+3*measure].set_ylabel('Response (spks/s) [%s]' % lbl)
            axSf[pltInd, ii+3*measure].set_title(labels[ii] + measure_lbl[measure, ii]);
          else:
            axSf[pltInd, ii+3*measure].set_title('vE=%.2f|%.2f [con = %.2f]' % (varExpl_A, varExpl_B, mC))
          try:
            axSf[pltInd, ii+3*measure].set_ylim(overall_ylim);
          except:
            pass; # ylim won't always work...

          if measure == 0: # only do the blank response reference for DC
              axSf[pltInd, ii+3*measure].axhline(floors[0], linestyle='--', color='b', label=labels_ref[0])
          # i.e. always put the baseOnly reference line...
          axSf[pltInd, ii+3*measure].axhline(floors[1], linestyle='--', color='k', label=labels_ref[1])
          if pltInd == 0:
            axSf[pltInd, ii+3*measure].legend(fontsize='small');

    # RVC across SF
    for ii, rsps in enumerate(resps): # first mask only, then mask+base (data)
      nSfs = len(maskSf);

      # we don't plot the F1 at base TF for DC response...
      if measure == 0 and ii == (len(resps)-1):
          continue;

      for msI, mS in enumerate(maskSf):
        pltInd = msI; # nSfs-msI-1; # reverse order

        col = [0, 0, 0];  # just black for RVCs
        #col = [(nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs)];
        # PLOT THE DATA
        axCon[pltInd, ii+3*measure].errorbar(maskCon, rsps[:,msI,0], rsps[:,msI,1], fmt='o', clip_on=False,color=col, label=str(np.round(mS, 2)) + ' cpd')
        if fitBase is None: # then just plot a line for the data
          axCon[pltInd, ii+3*measure].plot(maskCon, rsps[:,msI,0], clip_on=False, color=col)
        else:
          varExpl_A = hf.var_explained(hf.nan_rm(rsps[:,msI,0]), hf.nan_rm(modA_resps[ii][:,msI,0]), None);
          varExpl_B = hf.var_explained(hf.nan_rm(rsps[:,msI,0]), hf.nan_rm(modB_resps[ii][:,msI,0]), None);
          # PLOT model A (if present)
          axCon[pltInd, ii+3*measure].plot(maskCon, modA_resps[ii][:,msI,0], color=modColors[0], alpha=1-col[0], label='%.2f vE' % varExpl_A)
          # PLOT model B (if present)
          axCon[pltInd, ii+3*measure].plot(maskCon, modB_resps[ii][:,msI,0], color=modColors[1], alpha=1-col[0], label='%.2f vE' % varExpl_B)

        axCon[pltInd, ii+3*measure].set_xscale('log');
        try:
          axCon[pltInd, ii+3*measure].set_ylim(overall_ylim);
        except:
          pass;
        if pltInd == 0:
          axCon[pltInd, ii+3*measure].set_xlabel('Contrast (%)')
          axCon[pltInd, ii+3*measure].set_ylabel('Response (spks/s) [%s]' % lbl)
          axCon[pltInd, ii+3*measure].set_title(labels[ii] + measure_lbl[measure, ii])
        else:
          axCon[pltInd, ii+3*measure].set_title('vE=%.2f|%.2f [sf = %.2f cpd]' % (varExpl_A, varExpl_B, mS))


        if measure == 0: # only do the blank response for DC
            axCon[pltInd, ii+3*measure].axhline(floors[0], linestyle='--', color='b', label=labels_ref[0])
        # i.e. always put the baseOnly reference line...
        axCon[pltInd, ii+3*measure].axhline(floors[1], linestyle='--', color='k', label=labels_ref[1])
        if pltInd == 0:
          axCon[pltInd, ii+3*measure].legend(fontsize='small');
 
    ### For DC only
    # --- SF tuning with R(m+b) - R(m) - R(b)
    nCons = len(maskCon);
    for mcI, mC in enumerate(maskCon):
      pltInd = nCons-mcI-1;
      col = [(nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons)];

      if measure == 0:
          curr_line = axSf[pltInd, 2+3*measure].errorbar(maskSf, data_sub[mcI,:,0]-maskOnly[mcI,:,0], data_sub[mcI,:,1],
                                                fmt='o', color=col, label=str(np.round(mC, 2)) + '%')
          if fitBase is None: # then just plot a line for the data
            axSf[pltInd, 2+3*measure].plot(maskSf, data_sub[mcI,:,0]-maskOnly[mcI,:,0], clip_on=False, color=col)
          else:
            # model A (if present)
            axSf[pltInd, 2+3*measure].plot(maskSf, data_A_sub[mcI,:,0]-data_A_onlyMask[mcI,:,0], color=modColors[0], alpha=1-col[0])
            # model B (if present)
            axSf[pltInd, 2+3*measure].plot(maskSf, data_B_sub[mcI,:,0]-data_B_onlyMask[mcI,:,0], color=modColors[1], alpha=1-col[0])

          axSf[pltInd, 2+3*measure].set_ylim(ylim_diffsAbLe)

      ylim_diffs = [ylim_diffsAbLe];
      diff_endings = [' - R(m))'];
      for (j,ylim),txt in zip(enumerate(ylim_diffs), diff_endings):
        axSf[pltInd+j, 2+3*measure].set_xscale('log');
        axSf[pltInd+j, 2+3*measure].set_xlabel('SF (c/deg)')
        if measure==1: # Abramov/Levine sub. -- only DC has this analysis
            pass;
        else:
            axSf[pltInd+j, 2+3*measure].axhline(0, color='k', linestyle='--')
            if pltInd == 0:
              axSf[pltInd+j, 2+3*measure].set_ylabel('Difference (R(m+b) - R(b)%s (spks/s) [%s]' % (txt,lbl))
        if pltInd == 0:
          axSf[pltInd+j, 2+3*measure].legend(fontsize='small');

    # --- 
    nSfs = len(maskSf);
    for msI, mS in enumerate(maskSf):
      pltInd = msI; #nSfs-msI-1;
      col = [0, 0, 0];  # just black for RVCs
      #col = [(nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs)];

      if measure == 0:
          curr_line = axCon[pltInd, 2+3*measure].errorbar(maskCon, data_sub[:,msI,0] - maskOnly[:,msI,0], data_sub[:,msI,1],
                                                  fmt='o', color=col, label=str(np.round(mS, 2)) + ' cpd')
          if fitBase is None: # then just plot a line for the data
            axCon[pltInd, 2+3*measure].plot(maskCon, data_sub[:, msI,0]-maskOnly[:, msI,0], clip_on=False, color=col)
          else:
            # model A (if present)
            axCon[pltInd, 2+3*measure].plot(maskCon, data_A_sub[:, msI,0]-data_A_onlyMask[:, msI,0], color=modColors[0], alpha=1-col[0])
            # model B (if present)
            axCon[pltInd, 2+3*measure].plot(maskCon, data_B_sub[:, msI,0]-data_B_onlyMask[:, msI,0], color=modColors[1], alpha=1-col[0])

          axCon[pltInd, 2+3*measure].set_ylim(ylim_diffsAbLe)

      for (j,ylim),txt in zip(enumerate(ylim_diffs), diff_endings):
        axCon[pltInd+j, 2+3*measure].set_xscale('log');
        axCon[pltInd+j, 2+3*measure].set_xlabel('Contrast (%%)')
        if measure==1: # Abramov/Levine sub. -- only DC has this analysis
            pass;
        else:
            axCon[pltInd+j, 2+3*measure].axhline(0, color='k', linestyle='--')
            if pltInd == 0:
              axCon[pltInd+j, 2+3*measure].set_ylabel('Difference (R(m+b) - R(b)%s (spks/s) [%s]' % (txt,lbl))
        if pltInd == 0:
          axCon[pltInd+j, 2+3*measure].legend(fontsize='small');

fSf.tight_layout(rect=[0, 0.03, 1, 0.95])
fCon.tight_layout(rect=[0, 0.03, 1, 0.95])

### set titles

fSf.suptitle('V1 #%d [%s, f1f0: %.2f] base: %.2f cpd, %.2f%%; loss: <--[%.2f|%.2f], [%.2f|%.2f]-->\nvarExpl <--[%.2f,%.2f|%.2f,%.2f|%.2f,%.2f], [%.2f,%.2f|%.2f,%.2f|%.2f,%.2f]-->' % (cellNum, unitNm, f1f0_rat, baseSf_curr, baseCon_curr, *lossVals[0], *lossVals[1], vE_A[0,0],vE_B[0,0],vE_A[1,0],vE_B[1,0],vE_A[2,0],vE_B[2,0],vE_A[0,1],vE_B[0,1],vE_A[1,1],vE_B[1,1],vE_A[2,1],vE_B[2,1]), fontsize='x-large');
fCon.suptitle('V1 #%d [%s, f1f0: %.2f] base: %.2f cpd, %.2f%%; loss: <--[%.2f|%.2f], [%.2f|%.2f]-->\nvarExpl <--[%.2f,%.2f|%.2f,%.2f|%.2f,%.2f], [%.2f,%.2f|%.2f,%.2f|%.2f,%.2f]-->' % (cellNum, unitNm, f1f0_rat, baseSf_curr, baseCon_curr, *lossVals[0], *lossVals[1], vE_A[0,0],vE_B[0,0],vE_A[1,0],vE_B[1,0],vE_A[2,0],vE_B[2,0],vE_A[0,1],vE_B[0,1],vE_A[1,1],vE_B[1,1],vE_A[2,1],vE_B[2,1]), fontsize='x-large');


### now save all figures (incl model details, if specified)
for whichF, whichNm in zip([fSf, fCon], ['_sf', '_con']):
  saveName = "/cell_%03d%s.pdf" % (cellNum, whichNm)
  full_save = os.path.dirname(str(save_loc + 'core%s/' % onsetStr));
  if not os.path.exists(full_save):
      os.makedirs(full_save);
  pdfSv = pltSave.PdfPages(full_save + saveName);
  pdfSv.savefig(whichF);
  plt.close(whichF);
  pdfSv.close()

'''
#########
# --- Plot secondary things - filter, normalization, nonlinearity, etc
#########
if fitBase is not None: # then we can plot some model details

  fDetails = plt.figure();
  fDetails.set_size_inches(w=60,h=25)

  # make overall title
  fDetails.suptitle('DC <---- |model details| ----> F1');

  detailSize = (4, 6); # yes, normally (3,6), but making debug plots on 21.03.14 & want the extra row

  respTypes = [None, None]; # todo: [dcResps, f1Resps], figure out how to package, esp. with f1 having mask & base
  colToAdd = [0, 3]; # we add +X if doing f1 details
  # ordering of labels/model parameters will be: modA/modB (following form of modColors/modLabels from above)
  whichParams = [[modFit_A_dc, modFit_B_dc], [modFit_A_f1, modFit_B_f1]];
  whichModels = [[mod_A_dc, mod_B_dc], [mod_A_f1, mod_B_f1]];
  for (i, resps), colAdd, currPrms, currMods, trInf, respMeas in zip(enumerate(respTypes), colToAdd, whichParams, whichModels, [trInf_dc, trInf_f1], [0,1]): # DC, then F1...

    # TODO: poisson test - mean/var for each condition (i.e. sfXdispXcon)

    # response nonlinearity
    if _sigmoidRespExp is None:
      modExps = [x[3] for x in currPrms]; # respExp is in location [3]
    else:
      modExps = [1 + _sigmoidRespExp/(1+np.exp(-x[3])) for x in currPrms]; # respExp is in location [3]
    curr_ax = plt.subplot2grid(detailSize, (0, 1+colAdd));
    # Remove top/right axis, put ticks only on bottom/left
    sns.despine(ax=curr_ax, offset=5);
    plt.plot([-1, 1], [0, 0], 'k--')
    plt.plot([0, 0], [-.1, 1], 'k--')
    [plt.plot(np.linspace(-1,1,100), np.power(np.maximum(0, np.linspace(-1,1,100)), modExp), '%s-' % cc, label=s, linewidth=2) for modExp,cc,s in zip(modExps, modColors, modLabels)]
    plt.plot(np.linspace(-1,1,100), np.maximum(0, np.linspace(-1,1,100)), 'k--', linewidth=1)
    plt.xlim([-1, 1]);
    plt.ylim([-.1, 1]);
    plt.text(-0.5, 0.5, 'respExp: %.2f, %.2f' % (modExps[0], modExps[1]), fontsize=24, horizontalalignment='center', verticalalignment='center');
    plt.legend(fontsize='small');

    # plot model details - exc/suppressive components
    ########### as copied, and now edited, from plot_diagnose_vLGN
    omega = np.logspace(-2, 2, 1000);
    sfExc = [];
    sfExcRaw = [];
    sfNorms = [];

    for (pltNum, modPrm),lgnType,lgnConType,normType in zip(enumerate(currPrms), lgnTypes, conTypes, normTypes):
      # First, excitatory stuff
      prefSf = modPrm[0];
      mWt = 1/(1+np.exp(-modPrm[-1])); # either this is mWeight parameter, or we're not fitting an LGN model anyway (ignored)

      if excType == 1:
        ### deriv. gauss
        dOrder = _sigmoidDord*1/(1+np.exp(-modPrm[1]));
        sfRel = omega/prefSf;
        s     = np.power(omega, dOrder) * np.exp(-dOrder/2 * np.square(sfRel));
        sMax  = np.power(prefSf, dOrder) * np.exp(-dOrder/2);
        sfExcV1 = s/sMax;
        sfExcLGN = s/sMax; # will be used IF there isn't an LGN front-end...
      if excType == 2:
        ### flex. gauss
        sigLow = modPrm[1] if _sigmoidSigma is None else _sigmoidSigma/(1+np.exp(-modPrm[1]));
        sigHigh = modPrm[-1-np.sign(lgnType)] if _sigmoidSigma is None else _sigmoidSigma/(1+np.exp(-modPrm[-1-np.sign(lgnType)]));
        sfRel = np.divide(omega, prefSf);
        # - set the sigma appropriately, depending on what the stimulus SF is
        sigma = np.multiply(sigLow, [1]*len(sfRel));
        sigma[[x for x in range(len(sfRel)) if sfRel[x] > 1]] = sigHigh;
        # - now, compute the responses (automatically normalized, since max gaussian value is 1...)
        s     = [np.exp(-np.divide(np.square(np.log(x)), 2*np.square(y))) for x,y in zip(sfRel, sigma)];
        sfExcV1 = s;
        sfExcLGN = s; # will be used IF there isn't an LGN front-end...
      # BUT. if this is an LGN model, we'll apply the filtering, eval. at 100% contrast
      if lgnType == 1 or lgnType == 2:
        params_m = [0, 12.5, 0.05];
        params_p = [0, 17.5, 0.50];
        DoGmodel = 2;
        if lgnType == 1:
          dog_m = [1, 3, 0.3, 0.4]; # k, f_c, k_s, j_s
          dog_p = [1, 9, 0.5, 0.4];
        elif lgnType == 2:
          dog_m = [1, 6, 0.3, 0.4]; # k, f_c, k_s, j_s
          dog_p = [1, 9, 0.5, 0.4];
        # now compute with these parameters
        resps_m = hf.get_descrResp(dog_m, omega, DoGmodel, minThresh=0.1)
        resps_p = hf.get_descrResp(dog_p, omega, DoGmodel, minThresh=0.1)
        # -- make sure we normalize by the true max response:
        sfTest = np.geomspace(0.1, 10, 1000);
        max_m = np.max(hf.get_descrResp(dog_m, sfTest, DoGmodel, minThresh=0.1));
        max_p = np.max(hf.get_descrResp(dog_p, sfTest, DoGmodel, minThresh=0.1));
        # -- then here's our selectivity per component for the current stimulus
        selSf_m = np.divide(resps_m, max_m);
        selSf_p = np.divide(resps_p, max_p);
        # - then RVC response: # rvcMod 0 (Movshon)
        rvc_mod = hf.get_rvc_model();
        stimCo = np.linspace(0,1,100);
        selCon_m = rvc_mod(*params_m, stimCo)
        selCon_p = rvc_mod(*params_p, stimCo)
        if lgnConType == 1: # DEFAULT
          # -- then here's our final responses per component for the current stimulus
          # ---- NOTE: The real mWeight will be sigmoid(mWeight), such that it's bounded between 0 and 1
          lgnSel = mWt*selSf_m*selCon_m[-1] + (1-mWt)*selSf_p*selCon_p[-1];
        elif lgnConType == 2 or lgnConType == 3 or lgnConType == 4:
          # -- Unlike the above (default) case, we don't allow for a separate M & P RVC - instead we just take the average of the two
          selCon_avg = mWt*selCon_m + (1-mWt)*selCon_p;
          lgnSel = mWt*selSf_m*selCon_avg[-1] + (1-mWt)*selSf_p*selCon_avg[-1];
        withLGN = s*lgnSel;
        sfExcLGN = withLGN/np.max(withLGN);

        # Then, plot LGN front-end, if we're here
        curr_ax = plt.subplot2grid(detailSize, (1+pltNum, colAdd));
        plt.semilogx(omega, selSf_m, label='magno', color='r', linestyle='--');
        plt.semilogx(omega, selSf_p, label='parvo', color='b', linestyle='--');
        max_joint = np.max(lgnSel);
        plt.semilogx(omega, np.divide(lgnSel, max_joint), label='joint - 100% contrast', color='k');
        conMatch = 0.20
        conValInd = np.argmin(np.square(stimCo-conMatch));
        if lgnConType == 1:
          jointAtLowCon = mWt*selSf_m*selCon_m[conValInd] + (1-mWt)*selSf_p*selCon_p[conValInd];
        elif lgnConType == 2 or lgnConType == 3 or lgnConType == 4:
          jointAtLowCon = mWt*selSf_m*selCon_avg[conValInd] + (1-mWt)*selSf_p*selCon_avg[conValInd];
        plt.semilogx(omega, np.divide(jointAtLowCon, max_joint), label='joint - %d%% contrast' % (100*conMatch), color='k', alpha=0.3);
        plt.title('lgn %s' % modLabels[pltNum]);
        plt.legend();
        plt.xlim([omega[0], omega[-1]]);
        #plt.xlim([1e-1, 1e1]);

      sfExcRaw.append(sfExcV1);
      sfExc.append(sfExcLGN);

      # Compute weights for suppressive signals
      nTrials = len(val_trials);
      gs_mean = modPrm[8] if normType == 2 or normType == 5 else None;
      gs_std = modPrm[9] if normType == 2 or normType == 5 else None;
      #gs_gain = 
      norm_weights = np.sqrt(hf.genNormWeightsSimple(omega, gs_mean, gs_std));
      sfNormSimple = norm_weights/np.amax(np.abs(norm_weights));
      sfNorms.append(sfNormSimple);

    # Plot the filters - for LGN, this is WITH the lgn filters "acting" (assuming high contrast)
    curr_ax = plt.subplot2grid(detailSize, (1, 1+colAdd));
    # Remove top/right axis, put ticks only on bottom/left
    sns.despine(ax=curr_ax, offset=5);
    # just setting up lines
    plt.semilogx([omega[0], omega[-1]], [0, 0], 'k--')
    plt.semilogx([.01, .01], [-1.5, 1], 'k--')
    plt.semilogx([.1, .1], [-1.5, 1], 'k--')
    plt.semilogx([1, 1], [-1.5, 1], 'k--')
    plt.semilogx([10, 10], [-1.5, 1], 'k--')
    plt.semilogx([100, 100], [-1.5, 1], 'k--')
    # now the real stuff
    [plt.semilogx(omega, exc, '%s' % cc, label=s) for exc, cc, s in zip(sfExc, modColors, modLabels)]
    [plt.semilogx(omega, norm, '%s--' % cc, label=s) for norm, cc, s in zip(sfNorms, modColors, modLabels)]
    plt.xlim([omega[0], omega[-1]]);
    plt.ylim([-0.1, 1.1]);
    plt.xlabel('spatial frequency (c/deg)', fontsize=12);
    plt.ylabel('Normalized response (a.u.)', fontsize=12)

    # SIMPLE normalization - i.e. the raw weights
    curr_ax = plt.subplot2grid(detailSize, (2, 1+colAdd));
    # Remove top/right axis, put ticks only on bottom/left
    sns.despine(ax=curr_ax, offset=5);
    plt.semilogx([omega[0], omega[-1]], [0, 0], 'k--')
    plt.semilogx([.01, .01], [-1.5, 1], 'k--')
    plt.semilogx([.1, .1], [-1.5, 1], 'k--')
    plt.semilogx([1, 1], [-1.5, 1], 'k--')
    plt.semilogx([10, 10], [-1.5, 1], 'k--')
    plt.semilogx([100, 100], [-1.5, 1], 'k--')
    ### now the real stuff
    [plt.semilogx(omega, exc, '%s' % cc, label=s) for exc, cc, s in zip(sfExcRaw, modColors, modLabels)]
    plt.xlim([omega[0], omega[-1]]);
    plt.ylim([-0.1, 1.1]);
    plt.title('v-- raw weights/filters --v');
    plt.xlabel('spatial frequency (c/deg)', fontsize=12);
    plt.ylabel('Normalized response (a.u.)', fontsize=12);

    # Now, plot the full denominator (including the constant term) at a few contrasts
    # --- use the debug flag to get the tuned component of the gain control as computed in the full model
    curr_ax = plt.subplot2grid(detailSize, (1, 2+colAdd));
    modRespsDebug = [mod.forward(trInf, respMeasure=respMeas, debug=1, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm) for mod in currMods];
    modA_norm, modA_sigma = [modRespsDebug[0][x].detach().numpy() for x in [1,2]]; # returns are exc, inh, sigmaFilt (c50)
    modB_norm, modB_sigma = [modRespsDebug[1][x].detach().numpy() for x in [1,2]]; # returns are exc, inh, sigmaFilt (c50)
    # --- then, simply mirror the calculation as done in the full model
    full_denoms = [np.power(sigmaFilt + np.power(norm, 2), 0.5) for sigmaFilt, norm in zip([modA_sigma, modB_sigma], [modA_norm, modB_norm])];
    # --- use hf.get_valid_trials to get high/low con, single gratings
    conVals = [maskCon[-5], maskCon[-3], maskCon[-1]]; # try to get the normResp at these contrast values
    modTrials = trInf['num']; # these are the trials eval. by the model
    # then, let's go through for the above contrasts and get the in-model response
    for cI, conVal in enumerate(conVals):
      closest_ind = np.argmin(np.abs(conVal - maskCon));
      close_enough = np.abs(maskCon[closest_ind] - conVal) < 0.03 # must be within 3% contrast
      if close_enough:
        # highest contrast, first; for all, no base, only mask
        all_trials = [hf_sf.get_valid_trials(expInfo, 1, 0, closest_ind, sfI)[0] for sfI,_ in enumerate(maskSf)];
        # then, find which corresponding index into model-eval-only trials this is
        all_trials_modInd = [np.intersect1d(modTrials, trs, return_indices=True)[1] for trs in all_trials];
        modA_resps = [np.mean(full_denoms[0][trs]) for trs in all_trials_modInd];
        modB_resps = [np.mean(full_denoms[1][trs]) for trs in all_trials_modInd];
        # -- take sqrt of con val so that it's not SO dim...
        [plt.semilogx(maskSf, denom, alpha=np.sqrt(conVal), color=clr) for clr,denom in zip(modColors, [modA_resps, modB_resps])]
        plt.title('Normalization term by contrast, model');
    # plot just the constant term (i.e. if there is NO g.c. pooled response)
    sf_vals = maskSf;
    onlySigma = [np.power(sigmaFilt + np.power(0, 2), 0.5) for sigmaFilt in [modA_sigma, modB_sigma]];
    [plt.plot(xCoord*sf_vals[0], sig, color=clr, marker='>') for xCoord,sig,clr in zip([0.95, 0.85], onlySigma, modColors)]
    plt.xlim([omega[0], omega[-1]]);
    #plt.xlim([1e-1, 1e1]);

    ##################
    # TEMP. DEBUG: plot the full response at a few contrasts (single gratings)
    # --- and, generally the progression from filter-->threhsolding-->etc
    # - excitatory alone (0), with thresholding/resrpExp (1), full response (2)
    ##################
    ### 
    # --- use the debug flag to get the tuned component of the gain control as computed in the full model
    modRespsDebug = [mod.forward(trInf, respMeasure=respMeas, debug=1, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm) for mod in currMods];
    modA_exc, modA_norm, modA_sigma = [modRespsDebug[0][x].detach().numpy() for x in [0,1,2]]; # returns are exc, inh, sigmaFilt (c50)
    modB_exc, modB_norm, modB_sigma = [modRespsDebug[1][x].detach().numpy() for x in [0,1,2]]; # returns are exc, inh, sigmaFilt (c50)
    # --- then, simply mirror the calculation as done in the full model
    full_nums   = [np.add(modPrms[5], excs) for modPrms, excs in zip(currPrms, [modA_exc, modB_exc])]; # X is the index for early noise
    full_denoms = [np.power(sigmaFilt + np.power(norm, 2), 0.5) for sigmaFilt, norm in zip([modA_sigma, modB_sigma], [modA_norm, modB_norm])];
    full_ratio = [np.divide(num, np.tile(denom, (num.shape[0], 1))) for num, denom in zip(full_nums, full_denoms)];
    # --- full response (with actual respExp, and with respExp=1)
    full_resp_dcs = [];
    for rExp in [[currPrms[0][3], currPrms[1][3]], [1, 1]]:
      full_resp_one  = [np.power(np.maximum(_globalMin, ratio), rExp_curr) for ratio, rExp_curr in zip(full_ratio, rExp)];
      full_resp_two = [np.maximum(_globalMin, modPrms[6] + (ratio*_sigmoidScale)/(1+np.exp(-modPrms[4]))) for ratio, modPrms in zip(full_resp_one, currPrms)];
      full_resp_fft = [hf.spike_fft(psth.transpose(), stimDur=1)[0] for psth in full_resp_two];
      full_resp_dc = [np.array([resp[0] for resp in ffts]) for ffts in full_resp_fft];
      full_resp_dcs.append(full_resp_dc);
    # --- just the excitatory filter (no thresholding)
    numer_fft = [hf.spike_fft(psth.transpose(), stimDur=1)[0] for psth in full_nums];
    numer_dc = [np.array([resp[0] for resp in ffts]) for ffts in numer_fft];
    # --- then, numerator with thresholding (with actual rExp, and with 1)
    thresh_resp  = [np.power(np.maximum(_globalMin, ratio), modPrms[3]) for ratio, modPrms in zip(full_nums, currPrms)];
    thresh_respNoExp  = [np.power(np.maximum(_globalMin, ratio), 1) for ratio, modPrms in zip(full_nums, currPrms)];
    thresh_fft = [[hf.spike_fft(psth.transpose(), stimDur=1)[0] for psth in threshs] for threshs in [thresh_resp, thresh_respNoExp]];
    thresh_dcs = [[np.array([resp[0] for resp in ffts]) for ffts in tfft] for tfft in thresh_fft];

    currToPlot = [numer_dc, thresh_dcs, full_resp_dcs];
    titles = ['filt', 'with thresh, nonlin', 'full resp']
    ylims = [[0, 1], [0, 1], [0, 1]];
    
    for (row, resp), title, ylim in zip(enumerate(currToPlot), titles, ylims):
      curr_ax = plt.subplot2grid(detailSize, (3, row+colAdd));
      # --- use hf.get_valid_trials to get high/low con, single gratings
      conVals = [maskCon[-5], maskCon[-3], maskCon[-1]]; # try to get the normResp at these contrast values
      modTrials = trInf['num']; # these are the trials eval. by the model
      # then, let's go through for the above contrasts and get the in-model response
      for cI, conVal in enumerate(reversed(conVals)):
        closest_ind = np.argmin(np.abs(conVal - maskCon));
        close_enough = np.abs(maskCon[closest_ind] - conVal) < 0.03 # must be within 3% contrast
        if close_enough:
          # highest contrast, first; for all, no base, only mask
          all_trials = [hf_sf.get_valid_trials(expInfo, 1, 0, closest_ind, sfI)[0] for sfI,_ in enumerate(maskSf)];
          # then, find which corresponding index into model-eval-only trials this is
          all_trials_modInd = [np.intersect1d(modTrials, trs, return_indices=True)[1] for trs in all_trials];
          if row == 0:
            modA_resps = [np.mean(resp[0][trs]) for trs in all_trials_modInd]
            modB_resps = [np.mean(resp[1][trs]) for trs in all_trials_modInd]
            if cI==0:
              max_resp = np.maximum(np.max(modA_resps), np.max(modB_resps));
          else: # then we've got two plots
            modA_resps = [[np.mean(resps[0][trs]) for trs in all_trials_modInd] for resps in resp];
            modB_resps = [[np.mean(resps[1][trs]) for trs in all_trials_modInd] for resps in resp];
            if cI==0:
              max_resp = np.maximum(np.max(modA_resps[0]), np.max(modB_resps[0]));
              max_resp_noExp = np.maximum(np.max(modA_resps[1]), np.max(modB_resps[1]));
          # -- take sqrt of con val so that it's not SO dim...
          if row == 0:
            [plt.semilogx(maskSf, resp/max_resp, alpha=np.sqrt(conVal), color=clr) for clr,resp in zip(modColors, [modA_resps, modB_resps])]
          else:
            [plt.semilogx(maskSf, resp/max_resp, alpha=np.sqrt(conVal), color=clr) for clr,resp in zip(modColors, [modA_resps[0], modB_resps[0]])]
            [plt.semilogx(maskSf, resp/max_resp_noExp, alpha=np.sqrt(conVal), color=clr, linestyle='--') for clr,resp in zip(modColors, [modA_resps[1], modB_resps[1]])]
          plt.axhline(_globalMin, linestyle='--', color='k');
          plt.axvline(0.1, linestyle='--', color='k');
          plt.axvline(1, linestyle='--', color='k');
          plt.axvline(10, linestyle='--', color='k');
      #plt.xlim([0.1, 10]);
      plt.xlim([omega[0], omega[-1]]);
      if row < 2:
        plt.yscale('symlog', linthresh=1e-10);
      plt.ylim(ylim); #plt.ylim([0, 1]);
      plt.title(title);
    ##################
    # END TEMP. DEBUG
    ##################

    # print, in text, model parameters:
    curr_ax = plt.subplot2grid(detailSize, (0, 0+colAdd));
    plt.text(0.5, 0.6, 'order: %s, %s' % (*modLabels, ), fontsize=24, horizontalalignment='center', verticalalignment='center');
    plt.text(0.5, 0.5, 'prefSf: %.2f, %.2f' % (currPrms[0][0], currPrms[1][0]), fontsize=24, horizontalalignment='center', verticalalignment='center');
    # assumes only 2 or 5 as normTypes (besides 1...)
    normA = np.exp(currPrms[0][8]) if normTypes[0]>=2 else np.nan;
    normB = np.exp(currPrms[1][8]) if normTypes[1]>=2 else np.nan;
    normGainA = _sigmoidGainNorm/(1+np.exp(-currPrms[0][10])) if normTypes[0]==5 else np.nan;
    normGainB = _sigmoidGainNorm/(1+np.exp(-currPrms[1][10])) if normTypes[1]==5 else np.nan;
    plt.text(0.5, 0.4, 'normSf: %.2f, %.2f' % (normA, normB), fontsize=24, horizontalalignment='center', verticalalignment='center');
    if normGainA is not None or normGainB is not None:
      plt.text(0.5, 0.0, 'normGain: %.2f, %.2f' % (normGainA, normGainB), fontsize=24, horizontalalignment='center', verticalalignment='center');
    if excType == 1:
      plt.text(0.5, 0.3, 'derivative order: %.2f, %.2f' % (currPrms[0][1], currPrms[1][1]), fontsize=24, horizontalalignment='center', verticalalignment='center');
    elif excType == 2:
      plt.text(0.5, 0.3, 'sig (l|r): %.2f|%.2f, %.2f|%.2f' % (currPrms[0][1], currPrms[0][-1-np.sign(lgnTypes[0])], currPrms[1][1], currPrms[1][-1-np.sign(lgnTypes[1])]), fontsize=24, horizontalalignment='center', verticalalignment='center');
    plt.text(0.5, 0.2, 'response scalar: %.2f, %.2f' % (currPrms[0][4], currPrms[1][4]), fontsize=24, horizontalalignment='center', verticalalignment='center');
    plt.text(0.5, 0.1, 'sigma (con|raw): %.2f, %.2f || %.2f, %.2f' % (np.power(10, currPrms[0][2]), np.power(10, currPrms[1][2]), currPrms[0][2], currPrms[1][2]), fontsize=24, horizontalalignment='center', verticalalignment='center');
    plt.axis('off');


  # at end, make tight layout
  fDetails.tight_layout(pad=0.05)

else:
  fDetails = None
  print('here');

### now save all figures (incl model details, if specified)
saveName = "/cell_%03d_both.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'core%s/' % onsetStr));
if not os.path.exists(full_save):
    os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
if fitBase is not None: # then we can plot some model details
  allFigs = [f, fDetails];
  for fig in allFigs:
    pdfSv.savefig(fig)
    plt.close(fig)
else:
  pdfSv.savefig(f);
  plt.close(f);
pdfSv.close()


#######################
## var* (if applicable)
#######################
# Do this only if we are't plotting model fits
### --- TODO: Must make fits to sfBB_var* expts...
if fitBase is None or useCoreFit:

  # first, load the sfBB_core experiment to get reference tuning
  expInfo_base = cell['sfBB_core']
  f1f0_rat = hf_sf.compute_f1f0(expInfo_base)[0];

  maskSf_ref, maskCon_ref = expInfo_base['maskSF'], expInfo_base['maskCon'];
  refDC, refF1 = hf_sf.get_mask_resp(expInfo_base, withBase=0);
  # - get DC tuning curves
  refDC_sf = refDC[-1, :, :];
  prefSf_ind = np.argmax(refDC_sf[:, 0]);
  prefSf_DC = maskSf_ref[prefSf_ind];
  refDC_rvc = refDC[:, prefSf_ind, :];
  # - get F1 tuning curves
  refF1_sf = refF1[-1, :, :];
  prefSf_ind = np.argmax(refF1_sf[:, 0]);
  prefSf_F1 = maskSf_ref[prefSf_ind];
  refF1_rvc = refF1[:, prefSf_ind, :];

  # now, find out which - if any - varExpts exist
  allKeys = list(cell.keys())
  whichVar = np.where(['var' in x for x in allKeys])[0];

  for wV in whichVar: # if len(whichVar) == 0, nothing will happen
      expName = allKeys[wV];

      if 'Size' in expName:
          continue; # we don't have an analysis for this yet

      expInfo = cell[expName]
      byTrial = expInfo['trial'];

      ## base information/responses
      baseOnlyTr = np.logical_and(byTrial['baseOn'], ~byTrial['maskOn'])
      respDistr, _, unique_pairs = hf_sf.get_baseOnly_resp(expInfo);
      # now get the mask+base response (f1 at base TF)
      respMatrixDC, respMatrixF1 = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0); # i.e. get the base response
      # and get the mask only response (f1 at mask TF)
      respMatrixDC_onlyMask, respMatrixF1_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1); # i.e. get the maskONLY response
      # and get the mask+base response (but f1 at mask TF)
      _, respMatrixF1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1); # i.e. get the maskONLY response
      ## mask Con/SF values
      # - note that we round the contrast values, since the calculation of mask contrast with different 
      #   base contrasts can leave slight differences -- all much less than the conDig we round to.
      maskCon, maskSf = np.unique(np.round(expInfo['maskCon'], conDig)), expInfo['maskSF'];

      if useCoreFit:
        ### Get model responses!
        trInf_dc, resps_dc = mrpt.process_data(expInfo, expInd=expInd, respMeasure=0);
        trInf_f1, resps_f1 = mrpt.process_data(expInfo, expInd=expInd, respMeasure=1);
        val_trials = trInf_dc['num']; # these are the indices of valid, original trials

        # -- the same models from the core experiment will apply here!
        resp_A_dc  = mod_A_dc.forward(trInf_dc, respMeasure=0, sigmoidSigma=_sigmoidSigma).detach().numpy();
        resp_B_dc = mod_B_dc.forward(trInf_dc, respMeasure=0, sigmoidSigma=_sigmoidSigma).detach().numpy();
        resp_A_f1  = mod_A_f1.forward(trInf_f1, respMeasure=1, sigmoidSigma=_sigmoidSigma).detach().numpy();
        resp_B_f1 = mod_B_f1.forward(trInf_f1, respMeasure=1, sigmoidSigma=_sigmoidSigma).detach().numpy();

        loss_A = [mrpt.loss_sfNormMod(mrpt._cast_as_tensor(mr_curr), mrpt._cast_as_tensor(resps_curr), lossType=lossType, varGain=mrpt._cast_as_tensor(varGain)).detach().numpy() for mr_curr, resps_curr, varGain in zip([resp_A_dc, resp_A_f1], [resps_dc, resps_f1], varGains_A)]
        loss_B = [mrpt.loss_sfNormMod(mrpt._cast_as_tensor(mr_curr), mrpt._cast_as_tensor(resps_curr), lossType=lossType, varGain=mrpt._cast_as_tensor(varGain)).detach().numpy() for mr_curr, resps_curr, varGain in zip([resp_B_dc, resp_B_f1], [resps_dc, resps_f1], varGains_B)]

        # now get the mask+base response (f1 at base TF)
        maskInd, baseInd = hf_sf.get_mask_base_inds();

        baseMean_mod_dc = [hf_sf.get_baseOnly_resp(expInfo, dc_resp=x, val_trials=val_trials)[1][0] for x in [resp_A_dc, resp_B_dc]];
        baseMean_mod_f1 = [hf_sf.get_baseOnly_resp(expInfo, f1_base=x[:,baseInd], val_trials=val_trials)[1][1] for x in [resp_A_f1, resp_B_f1]];

        # ---- model A responses
        respMatrix_A_dc, respMatrix_A_f1 = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, dc_resp=resp_A_dc, f1_base=resp_A_f1[:,baseInd], f1_mask=resp_A_f1[:,maskInd], val_trials=val_trials); # i.e. get the base response for F1
        # and get the mask only response (f1 at mask TF)
        respMatrix_A_dc_onlyMask, respMatrix_A_f1_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, dc_resp=resp_A_dc, f1_base=resp_A_f1[:,baseInd], f1_mask=resp_A_f1[:,maskInd], val_trials=val_trials); # i.e. get the maskONLY response
        # and get the mask+base response (but f1 at mask TF)
        _, respMatrix_A_f1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, dc_resp=resp_A_dc, f1_base=resp_A_f1[:,baseInd], f1_mask=resp_A_f1[:,maskInd], val_trials=val_trials); # i.e. get the maskONLY response
        # ---- model B responses
        respMatrix_B_dc, respMatrix_B_f1 = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, dc_resp=resp_B_dc, f1_base=resp_B_f1[:,baseInd], f1_mask=resp_B_f1[:,maskInd], val_trials=val_trials); # i.e. get the base response for F1
        # and get the mask only response (f1 at mask TF)
        respMatrix_B_dc_onlyMask, respMatrix_B_f1_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, dc_resp=resp_B_dc, f1_base=resp_B_f1[:,baseInd], f1_mask=resp_B_f1[:,maskInd], val_trials=val_trials); # i.e. get the maskONLY response
        # and get the mask+base response (but f1 at mask TF)
        _, respMatrix_B_f1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, dc_resp=resp_B_dc, f1_base=resp_B_f1[:,baseInd], f1_mask=resp_B_f1[:,maskInd], val_trials=val_trials); # i.e. get the maskONLY response

      # -- back to where we were
      # what's the maximum response value?
      maxResp = np.maximum(np.nanmax(respMatrixDC), np.nanmax(respMatrixF1));
      maxResp_onlyMask = np.maximum(np.nanmax(respMatrixDC_onlyMask), np.nanmax(respMatrixF1_onlyMask));
      maxResp_total = np.maximum(maxResp, maxResp_onlyMask);
      overall_ylim = [0, 1.2*maxResp_total];
      # - also make the limits to be consistent across all base conditions for the AbLe plot
      dc_meanPerBase = [np.mean(x) for x in respDistr[0]];
      f1_meanPerBase = [np.mean(x) for x in respDistr[1]];
      AbLe_mn = 100; AbLe_mx = -100; # dummy values to be overwitten
      for ii in np.arange(len(dc_meanPerBase)):
          # only DC matters for AbLe...
          curr_min = np.nanmin(respMatrixDC[ii][:,:,0]-dc_meanPerBase[ii]-respMatrixDC_onlyMask[:,:,0])
          curr_max = np.nanmax(respMatrixDC[ii][:,:,0]-dc_meanPerBase[ii]-respMatrixDC_onlyMask[:,:,0])
          if curr_min < AbLe_mn:
              AbLe_mn = curr_min;
          if curr_max > AbLe_mx:
              AbLe_mx = curr_max;
      AbLe_bounds = [np.sign(AbLe_mn)*1.2*np.abs(AbLe_mn), np.maximum(5, 1.2*AbLe_mx)]; # ensure we always go at least above 0

      for (ii, up), respDC, respF1, respF1_maskTf in zip(enumerate(unique_pairs), respMatrixDC, respMatrixF1, respMatrixF1_maskTf):

        # we have the unique pairs, now cycle through and do the same thing here we did with the other base stimulus....
        baseSf_curr, baseCon_curr = up;
        baseDC, baseF1 = respDistr[0][ii], respDistr[1][ii];
        baseDistrs, baseSummary, baseConds = hf_sf.get_baseOnly_resp(expInfo);
        baseDC_mn, baseF1_mn = np.mean(baseDC), np.mean(baseF1);
          
        if vecCorrected:
          baseDistrs, baseSummary, baseConds = hf_sf.get_baseOnly_resp(expInfo, vecCorrectedF1=1, onsetTransient=onsetCurr);
          baseF1_mn = baseSummary[1][ii][0,:]; # [1][ii][0,:] is r,phi mean
          baseF1_var = baseSummary[1][ii][1,:]; # [1][ii][0,:] is r,phi std/(circ.) var
          baseF1_r, baseF1_phi = baseDistrs[1][ii][0], baseDistrs[1][ii][1];

        ### Now, plot
        nrow, ncol = 5, 4;
        f, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol*12, nrow*12))

        f.suptitle('V1 #%d [%s, %.2f] base: %.2f cpd, %.2f%%' % (cellNum, unitNm, f1f0_rat, baseSf_curr, baseCon_curr));            
        for measure in [0,1]:
          if measure == 0:
            baseline = expInfo['blank']['mean'];
            data = respDC;
            maskOnly = respMatrixDC_onlyMask;
            baseOnly = baseDC
            refAll = refDC[:,:,0];
            refSf = refDC_sf;
            refRVC = refDC_rvc;
            refSf_pref = prefSf_DC;
            if baselineSub:
                data -= baseline
                baseOnly -= baseline;
            xlim_base = overall_ylim;
            ylim_diffsAbLe = AbLe_bounds;
            lbl = 'DC'
            if fitBase is not None:
              data_A = respMatrix_A_dc[ii];
              data_B = respMatrix_B_dc[ii];
              data_A_onlyMask = respMatrix_A_dc_onlyMask; # no need to subset [ii], since the maskOnly is the same for all bases
              data_B_onlyMask = respMatrix_B_dc_onlyMask;
              data_A_baseTf = None;
              data_B_baseTf = None;
              mod_mean_A = baseMean_mod_dc[0][ii][0];
              mod_mean_B = baseMean_mod_dc[1][ii][0];
          elif measure == 1:
            data = respF1_maskTf # mask+base, at mask TF
            data_baseTf = respF1; # mask+base, but at base TF
            maskOnly = respMatrixF1_onlyMask;
            baseOnly = baseF1;
            refAll = refF1[:,:,0];
            refSf = refF1_sf;
            refRVC = refF1_rvc;
            refSf_pref = prefSf_F1;
            xlim_base = overall_ylim;
            lbl = 'F1'
            if vecCorrected:
                mean_r, mean_phi = baseF1_mn;
                std_r, var_phi = baseF1_var;
                vec_r, vec_phi = baseF1_r, baseF1_phi;
            if fitBase is not None:
              data_A = respMatrix_A_f1_maskTf[ii];
              data_B = respMatrix_B_f1_maskTf[ii];
              data_A_onlyMask = respMatrix_A_f1_onlyMask; # no need to subset [ii], since the maskOnly is the same for all bases
              data_B_onlyMask = respMatrix_B_f1_onlyMask;
              data_A_baseTf = respMatrix_A_f1[ii];
              data_B_baseTf = respMatrix_B_f1[ii];
              mod_mean_A = baseMean_mod_f1[0][ii][0];
              mod_mean_B = baseMean_mod_f1[1][ii][0];

          #pdb.set_trace();

          if measure == 0:
            data_sub = np.copy(data);
            data_sub[:,:,0] = data[:,:,0]-np.mean(baseOnly);
            if fitBase is not None:
              data_A_sub = np.copy(data_A);
              data_A_sub[:,:,0] = data_A[:,:,0] - mod_mean_A;
              data_B_sub = np.copy(data_B);
              data_B_sub[:,:,0] = data_B[:,:,0] - mod_mean_B;

          ### first, just the distribution of base responses
          if vecCorrected == 1 and measure == 1:
            plt.subplot(nrow, 2, 1+measure, projection='polar')
            [plt.plot([0, np.deg2rad(phi)], [0, r], 'o--k', alpha=0.3) for r,phi in zip(vec_r, vec_phi)]
            plt.plot([0, np.deg2rad(mean_phi)], [0, mean_r], 'o-k')
            #, label=r'$ mu(r,\phi) = (%.1f, %.0f)$' % (mean_r, mean_phi)
            nResps = len(vec_r);
            fano = np.square(std_r*np.sqrt(nResps))/mean_r; # we actually return s.e.m., so first convert to std, then square for variance
            plt.title(r'[%s; fano=%.2f] $(R,\phi) = (%.1f,%.0f)$ & -- $(sem,circVar) = (%.1f, %.1f)$' % (lbl, fano, mean_r, mean_phi, std_r, var_phi))
            # still need to define base_mn, since it's used later on in plots
            base_mn = mean_r;
          else:
            sns.distplot(baseOnly, ax=ax[0, measure], kde=False);
            nResps = len(baseOnly); # unpack the array for true length
            base_mn, base_sem = np.mean(baseOnly), np.std(baseOnly)/np.sqrt(nResps); 

            ax[0, measure].set_xlim(xlim_base)
            fano = np.square(base_sem*np.sqrt(nResps))/base_mn; # we actually return s.e.m., so first convert to std, then square for variance
            ax[0, measure].set_title('[%s; fano=%.2f] mn|sem = %.2f|%.2f' % (lbl, fano, base_mn, base_sem))
            if measure == 0:
                ax[0, measure].axvline(baseline, linestyle='--', color='b',label='blank')
            if fitBase is not None:
                ax[0, measure].axvline(np.mean(baseOnly), linestyle='--', color='k',label='data mean')
                ax[0, measure].axvline(mod_mean_A, linestyle='--', color=modColors[0], label='%s mean' % modLabels[0])
                ax[0, measure].axvline(mod_mean_B, linestyle='--', color=modColors[1], label='%s mean' % modLabels[1])
            ax[0, measure].legend(fontsize='large');

          # SF tuning with contrast
          resps = [maskOnly, data, data_baseTf]; #need to plot data_baseTf for f1
          if fitBase is not None:
            modA_resps = [data_A_onlyMask, data_A, data_A_baseTf];
            modB_resps = [data_B_onlyMask, data_B, data_B_baseTf];

          labels = ['mask', 'mask+base', 'mask+base']
          measure_lbl = np.vstack((['', '', ''], ['', ' (mask TF)', ' (base TF)'])); # specify which TF, if F1 response
          labels_ref = ['blank', 'base']
          floors = [baseline, base_mn]; # i.e. the blank response, then the response to the base alone

          for ij, rsps in enumerate(resps): # first mask only, then mask+base (data)
            nCons = len(maskCon);
            # we don't plot the F1 at base TF for DC response...
            if measure == 0 and ij == (len(resps)-1):
              continue;

            for mcI, mC in enumerate(maskCon):
              col = [(nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons)];

              # PLOT THE DATA
              ax[1+ij, 2*measure].errorbar(maskSf, rsps[mcI,:,0], rsps[mcI,:,1], fmt='o', color=col, clip_on=False, label=str(np.round(mC, 2)) + '%')
              if fitBase is None:
                ax[1+ij, 2*measure].plot(maskSf, rsps[mcI,:,0], clip_on=False, color=col)
              else:
                # PLOT model A (if present)
                ax[1+ij, 2*measure].plot(maskSf, modA_resps[ij][mcI,:,0], color=modColors[0], alpha=1-col[0])
                # PLOT model B (if present)
                ax[1+ij, 2*measure].plot(maskSf, modB_resps[ij][mcI,:,0], color=modColors[1], alpha=1-col[0])

            ax[1+ij, 2*measure].set_xscale('log');
            ax[1+ij, 2*measure].set_xlabel('SF (c/deg)')
            ax[1+ij, 2*measure].set_ylabel('Response (spks/s) [%s]' % lbl)
            ax[1+ij, 2*measure].set_title(labels[ij] + measure_lbl[measure, ij]);
            ax[1+ij, 2*measure].set_ylim(overall_ylim);
            if measure == 0: # only do the blank response for DC
              ax[1+ij, 2*measure].axhline(baseline, linestyle='--', color='r', label=labels_ref[0])
            # i.e. always put the baseOnly reference line...
            ax[1+ij, 2*measure].axhline(base_mn, linestyle='--', color='b', label=labels_ref[1])
            ax[1+ij, 2*measure].legend();

          # RVC across SF
          for ij, rsps in enumerate(resps): # first mask only, then mask+base (data)
              nSfs = len(maskSf);

              # we don't plot the F1 at base TF for DC response...
              if measure == 0 and ij == (len(resps)-1):
                  continue;

              for msI, mS in enumerate(maskSf):

                col = [(nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs)];
                # PLOT THE DATA
                curr_line = ax[1+ij, 1+2*measure].errorbar(maskCon, rsps[:,msI,0], rsps[:,msI,1], fmt='o', color=col, clip_on=False, label=str(np.round(mS, 2)) + ' cpd')
                if fitBase is None: # then just plot a line for the data
                  ax[1+ij, 1+2*measure].plot(maskCon, rsps[:,msI,0], clip_on=False, color=col)
                else:
                  # PLOT model A (if present)
                  ax[1+ij, 1+2*measure].plot(maskCon, modA_resps[ij][:,msI,0], color=modColors[0], alpha=1-col[0])
                  # PLOT model B (if present)
                  ax[1+ij, 1+2*measure].plot(maskCon, modB_resps[ij][:,msI,0], color=modColors[1], alpha=1-col[0])

              ax[1+ij, 1+2*measure].set_xscale('log');
              ax[1+ij, 1+2*measure].set_xlabel('Contrast (%)')
              ax[1+ij, 1+2*measure].set_ylabel('Response (spks/s) [%s]' % lbl)
              ax[1+ij, 1+2*measure].set_title(labels[ij] + measure_lbl[measure, ij])
              ax[1+ij, 1+2*measure].set_ylim(overall_ylim);
              if measure == 0: # only do the blank response for DC
                  ax[1+ij, 1+2*measure].axhline(floors[0], linestyle='--', color='r', label=labels_ref[0])
              # i.e. always put the baseOnly reference line...
              ax[1+ij, 1+2*measure].axhline(floors[1], linestyle='--', color='b', label=labels_ref[1])
              ax[1+ij, 1+2*measure].legend(fontsize='small');

          ### What's the loss evaluated on these data?
          ax[4, 2*measure].text(0.5, 0.5, 'loss A|B = %.2f|%.2f' % (loss_A[measure], loss_B[measure]));
          ### Loss trajectory

          ### SF tuning with R(m+b) - R(m) - R(b) // for DC only
          nCons = len(maskCon);
          for mcI, mC in enumerate(maskCon):
            col = [(nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons)];

            if measure == 0:
              curr_line = ax[3, 2*measure].errorbar(maskSf, data_sub[mcI,:,0]-maskOnly[mcI,:,0], data_sub[mcI,:,1],
                                                    fmt='o', color=col, label=str(np.round(mC, 2)) + '%')
              if fitBase is None: # then just plot a line for the data
                ax[3, 2*measure].plot(maskSf, data_sub[mcI,:,0]-maskOnly[mcI,:,0], clip_on=False, color=col)
              else:
                # model A (if present)
                ax[3, 2*measure].plot(maskSf, data_A_sub[mcI,:,0]-data_A_onlyMask[mcI,:,0], color=modColors[0], alpha=1-col[0])
                # model B (if present)
                ax[3, 2*measure].plot(maskSf, data_B_sub[mcI,:,0]-data_B_onlyMask[mcI,:,0], color=modColors[1], alpha=1-col[0])

              ax[3, 2*measure].set_ylim(ylim_diffsAbLe)

          ylim_diffs = [ylim_diffsAbLe];
          diff_endings = [' - R(m))'];
          for (j,ylim),txt in zip(enumerate(ylim_diffs), diff_endings):
              ax[3+j, 2*measure].set_xscale('log');
              ax[3+j, 2*measure].set_xlabel('SF (c/deg)')
              ax[3+j, 2*measure].set_ylabel('Difference (R(m+b) - R(b)%s (spks/s) [%s]' % (txt,lbl))
              if measure==1: # Abramov/Levine sub. -- only DC has this analysis
                  pass;
              else:
                  ax[3+j, 2*measure].axhline(0, color='k', linestyle='--')
              ax[3+j, 2*measure].legend();

          ### RVC across SF [rows 1-4, column 2 (& 4)]
          nSfs = len(maskSf);
          for msI, mS in enumerate(maskSf):
            col = [(nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs)];

            if measure == 0:
              curr_line = ax[3, 1+2*measure].errorbar(maskCon, data_sub[:,msI,0] - maskOnly[:,msI,0], data_sub[:,msI,1],
                                                      fmt='o', color=col, label=str(np.round(mS, 2)) + ' cpd')
              if fitBase is None: # then just plot a line for the data
                ax[3, 1+2*measure].plot(maskCon, data_sub[:, msI,0]-maskOnly[:, msI,0], clip_on=False, color=col)
              else:
                # model A (if present)
                ax[3, 1+2*measure].plot(maskCon, data_A_sub[:, msI,0]-data_A_onlyMask[:, msI,0], color=modColors[0], alpha=1-col[0])
                # model B (if present)
                ax[3, 1+2*measure].plot(maskCon, data_B_sub[:, msI,0]-data_B_onlyMask[:, msI,0], color=modColors[1], alpha=1-col[0])

              ax[3, 1+2*measure].set_ylim(ylim_diffsAbLe)

          for (j,ylim),txt in zip(enumerate(ylim_diffs), diff_endings):
              ax[3+j, 1+2*measure].set_xscale('log');
              ax[3+j, 1+2*measure].set_xlabel('Contrast (%%)')
              if measure==1: # Abramov/Levine sub. -- only DC has this analysis
                  pass;
              else:
                  ax[3+j, 1+2*measure].axhline(0, color='k', linestyle='--')
              ax[3+j, 1+2*measure].legend();

        #sns.despine(offset=10)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])

        saveName = "/cell_%03d_both_sf%03d_con%03d.pdf" % (cellNum, np.int(100*baseSf_curr), np.int(100*baseCon_curr))
        full_save = os.path.dirname(str(save_loc + '%s/cell_%03d/' % (expName, cellNum)));
        if not os.path.exists(full_save):
            os.makedirs(full_save);
        pdfSv = pltSave.PdfPages(full_save + saveName);
        pdfSv.savefig(f)
        plt.close(f)
        pdfSv.close()

'''
