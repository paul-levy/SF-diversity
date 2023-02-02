# coding: utf-8

#### NOTE: Based on plot_diagnose_vLGN.py
# i.e. we'll compare two models (if NOT just plotting the data)
# As of 21.02.09, we will make the same change as in the "parent" function
# - that is, we can flexibly choose the two models we use (not just assume one with, one without LGN front end)

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
from matplotlib.ticker import FuncFormatter
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

useTex = False
# using fits where the filter sigma is sigmoid?
_sigmoidRespExp = None; # 3 or None, as of 21.03.14
_sigmoidSigma = 5; # put a value (5) or None (see model_responses_pytorch.py for details)
_sigmoidGainNorm = 5;
recenter_norm = 1; # recenter the tuned normalization around 1?
#######
## TODO: note useCoreFit is now 0
#######
useCoreFit = 0; # if useCoreFit, then we'll plot the model response to the sfBB_var* experiments, if applicable
#######
singleGratsOnly = False
_globalMin = 1e-10;
# if None, then we keep the plots as is; if a number, then we create a gray shaded box encompassing that much STD of the base response
# -- by plotting a range of base responses rather than just the mean, we can see how strong the variations in base or base+mask responses are
plt_base_band = 1; # e.g. if 1, then we plot +/- 0.5 std; if 2, then we plot +/- 1 std; and so on
f1_r_std_on_r = True; # do we compute the std for respAmpl (F1) based on vector (i.e. incl. var in phi) or ONLY on corrected F1 resps

############
# Before any plotting, fix plotting paramaters
############
from matplotlib import rcParams
tex_width = 469; # per \layout in Overleaf on document
sns_offset = 2; 
hist_width = 0.9;
hist_ytitle = 0.94; # moves the overall title a bit further down on histogram plots0

rcParams.update(mpl.rcParamsDefault)

fontsz = 10;
tick_scalar = 1.5;

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

if useTex:
  rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
  params = {'text.usetex' : True,
#              'font.size' : fontsz,
            'font.family': 'lmodern',
             'font.style': 'italic'}
  plt.rcParams.update(params)
else:
  rcParams['font.style'] = 'oblique';

# rcParams['lines.linewidth'] = 2.5;
rcParams['lines.markeredgewidth'] = 0; # no edge, since weird tings happen then
# rcParams['axes.linewidth'] = 2; # was 1.5
# rcParams['lines.markersize'] = 5;

tick_adj = ['xtick.major.size', 'xtick.minor.size', 'ytick.major.size', 'ytick.minor.size']
for adj in tick_adj:
    rcParams[adj] = rcParams[adj] * tick_scalar;


### input arguments
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

if len(sys.argv) > 12:
  force_measure = int(sys.argv[12]);
  # if 0 or 1, force DC/F1, respectively
  if force_measure<0 or force_measure>1:
    force_measure = None;
else:
  force_measure = None;

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
  useHPCfit = 1;

if len(sys.argv) > 16:
  whichKfold = int(sys.argv[16]);
  if whichKfold<0:
    whichKfold = None
else:
  whichKfold = None;
isCV = False if whichKfold is None else True;

forceLog = True # log Y?

if len(sys.argv) > 17: # norm weights determined with deriv. Gauss or log Gauss?
  dgNormFuncIn=int(sys.argv[17]);
else:
  dgNormFuncIn=11


## Unlikely to be changed, but keep flexibility
baselineSub = 1 if forceLog else 0;
plt_ylim = 1 if forceLog else None
fix_ylim = 0;

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
#loc_str = ''; # TEMP

if _sigmoidRespExp is not None:
  rExpStr = 're';
else:
  rExpStr = '';

### DATALIST
expName = hf.get_datalist(expDir);

# EXPIND ::: TODO: Make this smarter?
expInd = -1;
### FITLIST
_applyLGNtoNorm = 1;

# -- some params are sigmoid, we'll use this to unpack the true parameter
_sigmoidScale = 10
_sigmoidDord = 5;

#fitBase = 'fitList%s_pyt_nr221031j_noRE_noSched%s' % (loc_str, '_sg' if singleGratsOnly else '')
#fitBase = 'fitList%s_pyt_nr221109f_noSched%s' % (loc_str, '_sg' if singleGratsOnly else '') 
#fitBase = 'fitList%s_pyt_nr221116wwww_noRE_noSched%s' % (loc_str, '_sg' if singleGratsOnly else '') 
#fitBase = 'fitList%s_pyt_nr221119d_noRE_noSched%s' % (loc_str, '_sg' if singleGratsOnly else '') 
#fitBase = 'fitList%s_pyt_nr221231_noRE_noSched%s' % (loc_str, '_sg' if singleGratsOnly else '') 
#fitBase = 'fitList%s_pyt_nr230107_noRE_noSched%s' % (loc_str, '_sg' if singleGratsOnly else '') 
fitBase = 'fitList%s_pyt_nr230118_noRE_noSched%s' % (loc_str, '_sg' if singleGratsOnly else '') 
#fitBase = 'fitList%s_pyt_nr230118a_noRE_noSched%s' % (loc_str, '_sg' if singleGratsOnly else '') 

# NOT model for now (23.01.25)
#fitBase = None;

_CV=isCV

if excType <= 0:
  fitBase = None;

#incl_legend = True if fitBase is None else False; # incl. legend only if fitBase is None
incl_legend = False

if fitBase is not None:
  if vecCorrected:
    vecCorrected = 1;
  else:
    vecCorrected = 0;

  ### Model types
  # 0th: Unpack the norm types, con types, lgnTypes
  normA, normB = int(np.floor(normTypesIn/10)), np.mod(normTypesIn, 10)
  conA, conB = int(np.floor(conTypesIn/10)), np.mod(conTypesIn, 10)
  lgnA, lgnB = int(np.floor(lgnFrontEnd/10)), np.mod(lgnFrontEnd, 10)
  dgnfA, dgnfB = int(np.floor(dgNormFuncIn/10)), np.mod(dgNormFuncIn, 10)
  # --- NOTE: DEFAULT TO USING dgNormFunc
  fitNameA = hf.fitList_name(fitBase, normA, lossType, lgnA, conA, 0, fixRespExp=fixRespExp, kMult=kMult, excType=excType, CV=_CV, lgnForNorm=_applyLGNtoNorm, dgNormFunc=dgnfA)
  fitNameB = hf.fitList_name(fitBase, normB, lossType, lgnB, conB, 0, fixRespExp=fixRespExp, kMult=kMult, excType=excType, CV=_CV, lgnForNorm=_applyLGNtoNorm, dgNormFunc=dgnfB)
  #fitNameA = hf.fitList_name(fitBase, normA, lossType, lgnA, conA, vecCorrected, fixRespExp=fixRespExp, kMult=kMult, excType=excType)
  #fitNameB = hf.fitList_name(fitBase, normB, lossType, lgnB, conB, vecCorrected, fixRespExp=fixRespExp, kMult=kMult, excType=excType)
  # what's the shorthand we use to refer to these models...
  wtStr = 'wt';
  aWtStr = '%s%s' % ('wt' if normA>1 else 'asym', '' if normA<=2 else 'Gn' if normA==5 else 'Yk' if normA==6 else 'Mt');
  bWtStr = '%s%s' % ('wt' if normB>1 else 'asym', '' if normB<=2 else 'Gn' if normB==5 else 'Yk' if normB==6 else 'Mt');
  # -- the following two lines assume that we only use wt (norm=2) or wtGain (norm=5)
  #aWtStr = 'wt%s' % ('' if normA==2 else 'Gn');
  #bWtStr = 'wt%s' % ('' if normB==2 else 'Gn');
  aWtStr = '%s%s' % ('DG' if dgnfA==1 else '', aWtStr);
  bWtStr = '%s%s' % ('DG' if dgnfB==1 else '', bWtStr);
  lgnStrA = hf.lgnType_suffix(lgnA, conA);
  lgnStrB = hf.lgnType_suffix(lgnB, conB);
  modA_str = '%s%s' % ('fl' if normA==1 else aWtStr, lgnStrA if lgnA>0 else 'V1');
  modB_str = '%s%s' % ('fl' if normB==1 else bWtStr, lgnStrB if lgnB>0 else 'V1');

  fitListA = hf.np_smart_load(data_loc + fitNameA);
  fitListB = hf.np_smart_load(data_loc + fitNameB);

  try:
    fit_detailsA_all = hf.np_smart_load(data_loc + fitNameA.replace('.npy', '_details.npy'));
    fit_detailsA = fit_detailsA_all[cellNum-1];
    fit_detailsB_all = hf.np_smart_load(data_loc + fitNameB.replace('.npy', '_details.npy'));
    fit_detailsB = fit_detailsB_all[cellNum-1];
  except:
    fit_detailsA = None; fit_detailsB = None;

  dc_str = hf_sf.get_resp_str(respMeasure=0);
  f1_str = hf_sf.get_resp_str(respMeasure=1);

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

  newMethod = 1;
  mod_A_dc  = mrpt.sfNormMod(modFit_A_dc, expInd=expInd, excType=excType, normType=normTypes[0], lossType=lossType, lgnFrontEnd=lgnTypes[0], newMethod=newMethod, lgnConType=conTypes[0], applyLGNtoNorm=_applyLGNtoNorm, toFit=False, normFiltersToOne=False, dgNormFunc=dgnfA)
  mod_B_dc = mrpt.sfNormMod(modFit_B_dc, expInd=expInd, excType=excType, normType=normTypes[1], lossType=lossType, lgnFrontEnd=lgnTypes[1], newMethod=newMethod, lgnConType=conTypes[1], applyLGNtoNorm=_applyLGNtoNorm, toFit=False, normFiltersToOne=False, dgNormFunc=dgnfB)
  mod_A_f1  = mrpt.sfNormMod(modFit_A_f1, expInd=expInd, excType=excType, normType=normTypes[0], lossType=lossType, lgnFrontEnd=lgnTypes[0], newMethod=newMethod, lgnConType=conTypes[0], applyLGNtoNorm=_applyLGNtoNorm, toFit=False, normFiltersToOne=False, dgNormFunc=dgnfA)
  mod_B_f1 = mrpt.sfNormMod(modFit_B_f1, expInd=expInd, excType=excType, normType=normTypes[1], lossType=lossType, lgnFrontEnd=lgnTypes[1], newMethod=newMethod, lgnConType=conTypes[1], applyLGNtoNorm=_applyLGNtoNorm, toFit=False, normFiltersToOne=False, dgNormFunc=dgnfB)

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
  kstr = '';

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
  save_loc = str(save_loc + subDir + '_vertical/');
else:
  save_loc = str(save_loc + 'data_only_vertical/');

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
# can optionally force DC or F1
use_resp_measure = f1f0_rat>=1 if force_measure is None else force_measure;

### Now, if we've got the models, get and organize those responses...
if fitBase is not None:
  trInf_dc, resps_dc = mrpt.process_data(expInfo, expInd=expInd, respMeasure=0); 
  trInf_f1, resps_f1 = mrpt.process_data(expInfo, expInd=expInd, respMeasure=1); 
  val_trials = trInf_dc['num']; # these are the indices of valid, original trials

  resp_A_dc  = mod_A_dc.forward(trInf_dc, respMeasure=0, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm).detach().numpy();
  resp_B_dc = mod_B_dc.forward(trInf_dc, respMeasure=0, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm).detach().numpy();
  resp_A_f1  = mod_A_f1.forward(trInf_f1, respMeasure=1, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm).detach().numpy();
  resp_B_f1 = mod_B_f1.forward(trInf_f1, respMeasure=1, sigmoidSigma=_sigmoidSigma, recenter_norm=recenter_norm).detach().numpy();

  loss_A = [mrpt.loss_sfNormMod(mrpt._cast_as_tensor(mr_curr), mrpt._cast_as_tensor(resps_curr), lossType=lossType, varGain=mrpt._cast_as_tensor(varGain)).detach().numpy() for mr_curr, resps_curr, varGain in zip([resp_A_dc, resp_A_f1], [resps_dc, resps_f1], varGains_A)]
  loss_B = [mrpt.loss_sfNormMod(mrpt._cast_as_tensor(mr_curr), mrpt._cast_as_tensor(resps_curr), lossType=lossType, varGain=mrpt._cast_as_tensor(varGain)).detach().numpy() for mr_curr, resps_curr, varGain in zip([resp_B_dc, resp_B_f1], [resps_dc, resps_f1], varGains_B)]

  # now get the mask+base response (f1 at base TF)
  maskInd, baseInd = hf_sf.get_mask_base_inds();

  # note the indexing: [1][x][0][0] for [summary], [dc||f1], [unpack], [mean], respectively
  baseMean_mod_dc = [hf_sf.get_baseOnly_resp(expInfo, dc_resp=x, val_trials=val_trials)[1][0][0][0] for x in [resp_A_dc, resp_B_dc]];
  baseMean_mod_f1 = [hf_sf.get_baseOnly_resp(expInfo, f1_base=x[:,baseInd], val_trials=val_trials)[1][1][0][0] for x in [resp_A_f1, resp_B_f1]];
  # NOTE/TODO: Not handling any divFactor (rel. to stimDur)
  modBlank_A_dc = np.maximum(0, mod_A_dc.noiseLate.detach().numpy()); # baseline 
  modBlank_A_f1 = np.maximum(0, mod_A_f1.noiseLate.detach().numpy()); # baseline 
  modBlank_B_dc = np.maximum(0, mod_B_dc.noiseLate.detach().numpy()); # baseline
  modBlank_B_f1 = np.maximum(0, mod_B_f1.noiseLate.detach().numpy()); # baseline

  if forceLog:
    resp_A_dc -= modBlank_A_dc;
    resp_A_f1 -= modBlank_A_f1;
    resp_B_dc -= modBlank_B_dc;
    resp_B_f1 -= modBlank_B_f1;

  # ------ note: for all model responses, flag vecCorrectedF1 != 1 so that we make sure to use the passed-in model responses
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
    baseDistrs, baseSummary, _ = hf_sf.get_baseOnly_resp(expInfo, vecCorrectedF1=1, F1useSem=False);
    baseF1_mn = baseSummary[1][0][0,:]; # [1][0][0,:] is r,phi mean
    baseF1_var = baseSummary[1][0][1,:]; # [1][0][0,:] is r,phi std/(circ.) var
    baseF1_r, baseF1_phi = baseDistrs[1][0][0], baseDistrs[1][0][1];
# - unpack the SF x CON of the base (guaranteed to have only one set for sfBB_core)
baseSf_curr, baseCon_curr = baseConds[0];
# now get the mask+base response (f1 at base TF)
respMatrixDC, respMatrixF1 = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, vecCorrectedF1=vecCorrected); # i.e. get the base response for F1
# and get the mask only response (f1 at mask TF)
respMatrixDC_onlyMask, respMatrixF1_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, vecCorrectedF1=vecCorrected); # i.e. get the maskONLY response
# and get the mask+base response (but f1 at mask TF)
_, respMatrixF1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, vecCorrectedF1=vecCorrected); # i.e. get the maskONLY response

# -- if vecCorrected, let's just take the "r" elements, not the phi information
if vecCorrected:
    respMatrixF1 = respMatrixF1[:,:,0,:]; # just take the "r" information (throw away the phi)
    respMatrixF1_onlyMask = respMatrixF1_onlyMask[:,:,0,:]; # just take the "r" information (throw away the phi)
    respMatrixF1_maskTf = respMatrixF1_maskTf[:,:,0,:]; # just take the "r" information (throw away the phi)

## Reference tuning...
refDC, refF1 = hf_sf.get_mask_resp(expInfo, withBase=0, vecCorrectedF1=vecCorrected); # i.e. mask only, at mask TF
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

### Now, plot

# set up model plot info
# i.e. flat model is red, weighted model is green
modColors = ['g', 'r']
try:
  modLabels = ['A: %s' % modA_str, 'B: %s' % modB_str]
except:
  modLabels = None

##########################
### set up the figure
##########################
nrow, ncol = 2, 1;
width_frac = 0.475
extra_height = 1.5/width_frac;
f, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=hf.set_size(width_frac*tex_width, extra_height=extra_height), sharex=True);

# also get the bounds for the AbLe plot - only DC
AbLe_mn = np.nanmin(respMatrixDC[:,:,0]-baseDC_mn-respMatrixDC_onlyMask[:,:,0])
AbLe_mx = np.nanmax(respMatrixDC[:,:,0]-baseDC_mn-respMatrixDC_onlyMask[:,:,0])
AbLe_bounds = [np.sign(AbLe_mn)*1.2*np.abs(AbLe_mn), np.maximum(5, 1.2*AbLe_mx)]; # ensure we go at least above 0 with the max

varExpl_mod = np.zeros((2, 2)); # modA/modB [1st dim], f0/f1 [2nd dim]

######
for measure in [0,1]:
    
    if measure != use_resp_measure:
        continue; # skip the one we don't need

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
            maskOnly -= baseline;

        maxResp = np.maximum(np.nanmax(data), np.nanmax(maskOnly))
        minResp = np.minimum(np.nanmin(data), np.nanmin(maskOnly))
        overall_ylim = [np.minimum(0, 1.2*minResp), 1.2*maxResp];

        xlim_base = overall_ylim;
        ylim_diffsAbLe = AbLe_bounds;
        lbl = 'DC' 
        if fitBase is not None:
          modelsAsObj = [mod_A_dc, mod_B_dc]
          data_A = respMatrix_A_dc;
          data_B = respMatrix_B_dc;
          data_A_onlyMask = respMatrix_A_dc_onlyMask;
          data_B_onlyMask = respMatrix_B_dc_onlyMask;
          data_A_baseTf = None;
          data_B_baseTf = None;
          mod_mean_A = baseMean_mod_dc[0];
          mod_mean_B = baseMean_mod_dc[1];
    elif measure == 1:
        baseline = None; # won't be used anyway...
        data = respMatrixF1_maskTf;
        data_baseTf = respMatrixF1;
        maskOnly = respMatrixF1_onlyMask;
        if vecCorrected:
            mean_r, mean_phi = baseF1_mn;
            std_r, var_phi = baseF1_var;
            if f1_r_std_on_r: # i.e. rather than computing the vector variance, compute only the var/std on the resp magnitudes
              std_r = np.nanstd(baseDistrs[1][0][0]); # just the r values
            vec_r, vec_phi = baseF1_r, baseF1_phi;
        refAll = refF1[:,:,0];
        refSf = refF1_sf;
        refRVC = refF1_rvc;
        refSf_pref = prefSf_F1;

        maxResp = np.maximum(np.nanmax(data_baseTf), np.nanmax(maskOnly))
        minResp = np.minimum(np.nanmin(data_baseTf), np.nanmin(maskOnly))
        overall_ylim = [np.minimum(-5, 1.2*minResp), 1.2*maxResp];

        xlim_base = overall_ylim
        lbl = 'F1'
        if fitBase is not None:
          modelsAsObj = [mod_A_f1, mod_B_f1]
          data_A = respMatrix_A_f1_maskTf;
          data_B = respMatrix_B_f1_maskTf;
          data_A_onlyMask = respMatrix_A_f1_onlyMask;
          data_B_onlyMask = respMatrix_B_f1_onlyMask;
          data_A_baseTf = respMatrix_A_f1;
          data_B_baseTf = respMatrix_B_f1;
          mod_mean_A = baseMean_mod_f1[0][0];
          mod_mean_B = baseMean_mod_f1[1][0];

    # get baseline response
    if vecCorrected == 1 and measure == 1:
        base_mn = mean_r;
        nResps = len(vec_r);
    else:
        nResps = len(baseOnly[0]); # unpack the array for true length
        base_mn, base_sem = np.mean(baseOnly), np.std(baseOnly)/np.sqrt(nResps); 
    # --- and one std. of the base response
    base_one_std = std_r if measure==1 else np.std(baseOnly); # assumes vecCorrected?

    # Now, subtract the baseOnly response from the base+mask response (only used if measure=0, i.e. DC)
    # -- but store it separately 
    if measure == 0: # should ALSO SPECIFY baselineSub==0, since otherwise we are double subtracting...
        data_sub = np.copy(data);
        data_sub[:,:,0] = data[:,:,0]-np.mean(baseOnly);
        if fitBase is not None:
            data_A_sub = np.copy(data_A);
            data_A_sub[:,:,0] = data_A[:,:,0] - mod_mean_A;
            data_B_sub = np.copy(data_B);
            data_B_sub[:,:,0] = data_B[:,:,0] - mod_mean_B;

    # SF tuning with contrast (MASK)
    # --- FOR NOW, keep resps as it is, but skip ii==1 (i.e. mask+base together [@maskTF if F1])
    resps = [maskOnly, data, data_baseTf];
    if fitBase is not None:
      modA_resps = [data_A_onlyMask, data_A, data_A_baseTf];
      modB_resps = [data_B_onlyMask, data_B, data_B_baseTf];

      # compute variance explained
      all_resps = np.array(hf.flatten_list([hf.flatten_list(x[:,:,0]) if x is not None else [] for x in resps]));
      all_resps_modA = np.array(hf.flatten_list([hf.flatten_list(x[:,:,0]) if x is not None else [] for x in modA_resps]));
      all_resps_modB = np.array(hf.flatten_list([hf.flatten_list(x[:,:,0]) if x is not None else [] for x in modB_resps]));
      varExpl_mod[0,measure] = hf.var_explained(all_resps, all_resps_modA, None);
      varExpl_mod[1,measure] = hf.var_explained(all_resps, all_resps_modB, None);

    labels = ['mask', 'mask+base', 'mask+base']
    measure_lbl = np.vstack((['', '', ''], ['', ' (mask TF)', ' (base TF)'])); # specify which TF, if F1 response
    labels_ref = ['blank', 'base']
    floors = [baseline, base_mn]; # i.e. the blank response, then the response to the base alone

    plt_incl_i = 0;
    for ii, rsps in enumerate(resps): # first mask only, then mask+base (data)
        if measure == 0 and ii==2: # skip empty space for DC
            continue;
        elif measure == 1 and ii == 1: # and mask+base @maskTF if F1)
            continue;

        nCons = len(maskCon);
        # we don't plot the F1 at base TF for DC response...
        if measure == 0 and ii == (len(resps)-1):
            continue;

        for mcI, mC in enumerate(maskCon):

            col = [(nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons)];
            data_ok = np.arange(len(maskSf)) if plt_ylim is None else rsps[mcI,:,0]>plt_ylim;
            # PLOT THE DATA
            # errbars should be (2,n_sfs)
            if plt_ylim is not None:
              high_err = rsps[mcI,:,1][data_ok]; # no problem with going to higher values
              low_err = np.minimum(high_err, rsps[mcI,:,0][data_ok]-plt_ylim-1e-2); # i.e. don't allow us to make the err any lower than where the plot will cut-off (incl. negatives)
              errs = np.vstack((low_err, high_err));
            else:
              errs = rsps[mcI,:,1][data_ok];

            ax[plt_incl_i].errorbar(maskSf[data_ok], rsps[mcI,:,0][data_ok], errs, fmt='o', clip_on=False,
                                                color=col, label=str(np.round(mC, 2)))
            if fitBase is None: # then just plot a line for the data
              ax[plt_incl_i].plot(maskSf[data_ok], rsps[mcI,:,0][data_ok], clip_on=False, color=col)
            else:
              # PLOT model A (if present)
              modA_ok = np.arange(len(maskSf)) if plt_ylim is None else modA_resps[ii][mcI,:,0]>plt_ylim;
              ax[plt_incl_i].plot(maskSf[modA_ok], modA_resps[ii][mcI,:,0][modA_ok], color=modColors[0], alpha=1-col[0])
              # PLOT model B (if present)
              modB_ok = np.arange(len(maskSf)) if plt_ylim is None else modB_resps[ii][mcI,:,0]>plt_ylim;
              ax[plt_incl_i].plot(maskSf[modB_ok], modB_resps[ii][mcI,:,0][modB_ok], color=modColors[1], alpha=1-col[0])

        ax[plt_incl_i].set_xscale('log');
        if plt_incl_i==(nrow-1):
            ax[plt_incl_i].set_xlabel('Spatial Frequency (c/deg)')
            # and specify tick locations
            for jj, axis in enumerate([ax[plt_incl_i].xaxis, ax[plt_incl_i].yaxis]):
              if jj == 0:
                axis.set_major_formatter(FuncFormatter(lambda x,y: '%d' % x if x>=1 else '%.1f' % x)) # this will make everything in non-scientific notation!
                core_ticks = np.array([1,3]);
                pltd_sfs = maskSf;
                if np.min(pltd_sfs)<=0.4:
                    core_ticks = np.hstack((0.3, core_ticks));
                if np.max(pltd_sfs)>=7:
                    core_ticks = np.hstack((core_ticks, 10));
                axis.set_ticks(core_ticks)

        else: # otherwise, no ticks
            ax[plt_incl_i].tick_params('x', labelbottom=False)
        ax[plt_incl_i].set_ylabel('Response (spikes/s)')
        #ax[plt_incl_i].set_title(labels[ii] + measure_lbl[measure, ii], fontsize='small');
        ax[plt_incl_i].set_ylim(overall_ylim);
        if measure == 0 and not forceLog: # only do the blank response reference for DC
            ax[plt_incl_i].axhline(floors[0], linestyle='--', color='b', label=labels_ref[0])
        # i.e. always put the baseOnly reference line...
        ax[plt_incl_i].axhline(floors[1], linestyle='--', color='k', label=labels_ref[1])
        if plt_base_band is not None and plt_incl_i!=0: # why skip 0? that's just the mask
          # -- and as +/- X std?
          stdTot = plt_base_band; # this will also serve as the total STD range to encompass
          sfMin, sfMax = np.nanmin(maskSf), np.nanmax(maskSf);
          ax[plt_incl_i].add_patch(mpl.patches.Rectangle([sfMin, floors[1]-0.5*stdTot*base_one_std], sfMax-sfMin, stdTot*base_one_std, alpha=0.1, color='k'))
          
        if plt_incl_i==0 and incl_legend: # only need the legend once...
            ax[plt_incl_i].legend(fontsize='small');

        if forceLog:
          ax[plt_incl_i].set_yscale('log');
          ax[plt_incl_i].axis('scaled');

        # and after we are done, increment the plot counter
        plt_incl_i += 1;

# despine...
sns.despine(offset=sns_offset);

coreTitle = '#%d [%s, f1f0: %.2f] base: %.2f cpd, %.2f%%' % (cellNum, unitNm, f1f0_rat, baseSf_curr, baseCon_curr);
if fitBase is not None:
  lossTitle = '\nloss: <-- %.2f,%.2f | %.2f,%.2f -->' % (*lossVals[0], *lossVals[1])
  varExplTitle = '\nvarExpl: <-- %.2f,%.2f | %.2f,%.2f -->' % (*varExpl_mod[:,0], *varExpl_mod[:,1])
else:
  lossTitle = '';
  varExplTitle = '';

f.suptitle('%s%s%s' % (coreTitle, lossTitle, varExplTitle), fontsize='xx-small');
f.subplots_adjust(hspace=0.15);
f.tight_layout(rect=[0, 0.03, 1, 0.98])

### now save all figures (incl model details, if specified)
saveName = "/cell_%03d_%s%s%s.pdf" % (cellNum, hf_sf.get_resp_str(respMeasure=use_resp_measure), kstr, '_log' if forceLog else '')
full_save = os.path.dirname(str(save_loc + 'core/'));
if not os.path.exists(full_save):
    os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(f);
plt.close(f);
pdfSv.close()
