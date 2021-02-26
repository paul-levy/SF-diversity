# coding: utf-8

#### NOTE: Based on plot_diagnose_vLGN.py
# i.e. we'll compare two models (if NOT just plotting the data)
# As of 21.02.09, we will make the same change as in the "parent" function
# - that is, we can flexibly choose the two models we use (not just assume one with, one without LGN front end)

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

## Unlikely to be changed, but keep flexibility
baselineSub = 0;
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

if 'pl1465' in loc_base:
  loc_str = 'HPC';
else:
  loc_str = '';

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
if excType == 1:
  fitBase = 'fitList_pyt_200417'; # excType 1
  #fitBase = 'fitList_pyt_201017'; # excType 1
elif excType == 2:
  #fitBase = 'fitList_pyt_200507'; # excType 2
  #fitBase = 'fitList_pyt_210121'; # excType 2
  #fitBase = 'fitList_pyt_210206'; # excType 2
  fitBase = 'fitList_pyt_210222'; # excType 2
else:
  fitBase = None;

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

  fitNameA = hf.fitList_name(fitBase, normA, lossType, lgnA, conA, vecCorrected, fixRespExp=fixRespExp, kMult=kMult)
  fitNameB = hf.fitList_name(fitBase, normB, lossType, lgnB, conB, vecCorrected, fixRespExp=fixRespExp, kMult=kMult)
  # what's the shorthand we use to refer to these models...
  modA_str = '%s%s%s' % ('fl' if normA==1 else 'wt', 'LGN' if lgnA>0 else 'V1', 'avg' if conA>1 else '');
  modB_str = '%s%s%s' % ('fl' if normB==1 else 'wt', 'LGN' if lgnB>0 else 'V1', 'avg' if conB>1 else '');

  fitListA = hf.np_smart_load(data_loc + fitNameA);
  fitListB = hf.np_smart_load(data_loc + fitNameB);

  try:
    fit_detailsA = hf.np_smart_load(data_loc + fitNameA.replace('.npy', '_details.npy'));
    fit_detailsB = hf.np_smart_load(data_loc + fitNameB.replace('.npy', '_details.npy'));
    fit_detailsA = fit_detailsA[cellNum-1];
    fit_detailsB = fit_detailsB[cellNum-1];
  except:
    fit_detailsA = None; fit_detailsB = None;

  dc_str = hf_sf.get_resp_str(respMeasure=0);
  f1_str = hf_sf.get_resp_str(respMeasure=1);

  modFit_A_dc = fitListA[cellNum-1][dc_str]['params'];
  modFit_B_dc = fitListB[cellNum-1][dc_str]['params'];
  modFit_A_f1 = fitListA[cellNum-1][f1_str]['params'];
  modFit_B_f1 = fitListB[cellNum-1][f1_str]['params'];
  lossVals = [[x[cellNum-1][y]['NLL'] for x in [fitListA, fitListB]] for y in [dc_str, f1_str]]

  normTypes = [normA, normB]; # weighted, then flat
  lgnTypes = [lgnA, lgnB];
  conTypes = [conA, conB];

  expInd = -1;
  newMethod = 1;

  mod_A_dc  = mrpt.sfNormMod(modFit_A_dc, expInd=expInd, excType=excType, normType=normTypes[0], lossType=lossType, lgnFrontEnd=lgnTypes[0], newMethod=newMethod, lgnConType=conTypes[0])
  mod_B_dc = mrpt.sfNormMod(modFit_B_dc, expInd=expInd, excType=excType, normType=normTypes[1], lossType=lossType, lgnFrontEnd=lgnTypes[1], newMethod=newMethod, lgnConType=conTypes[1])
  mod_A_f1  = mrpt.sfNormMod(modFit_A_f1, expInd=expInd, excType=excType, normType=normTypes[0], lossType=lossType, lgnFrontEnd=lgnTypes[0], newMethod=newMethod, lgnConType=conTypes[0])
  mod_B_f1 = mrpt.sfNormMod(modFit_B_f1, expInd=expInd, excType=excType, normType=normTypes[1], lossType=lossType, lgnFrontEnd=lgnTypes[1], newMethod=newMethod, lgnConType=conTypes[1])

else: # we will just plot the data
  fitList_fl = None;
  fitList_wg = None;

# set the save directory to save_loc, then create the save directory if needed
if onsetCurr is None:
  onsetStr = '';
else:
  onsetStr = '_onset%s_%03d_%03d' % (onsetMod_str, onsetDur, halfWidth)

lossSuf = hf.lossType_suffix(lossType).replace('.npy', ''); # get the loss suffix, remove the file type ending
if fitBase is not None:
  if diffPlot == 1: 
    compDir  = str(fitBase + '_diag_%s_%s' % (modA_str, modB_str) + lossSuf + '/diff');
  else:
    compDir  = str(fitBase + '_diag_%s_%s' % (modA_str, modB_str) + lossSuf);
  if intpMod == 1:
    compDir = str(compDir + '/intp');
  subDir   = compDir.replace('fitList', 'fits').replace('.npy', '');
  save_loc = str(save_loc + subDir + '/');
else:
  save_loc = str(save_loc + 'data_only/');

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

  resp_A_dc  = mod_A_dc.forward(trInf_dc, respMeasure=0).detach().numpy();
  resp_B_dc = mod_B_dc.forward(trInf_dc, respMeasure=0).detach().numpy();
  resp_A_f1  = mod_A_f1.forward(trInf_f1, respMeasure=1).detach().numpy();
  resp_B_f1 = mod_B_f1.forward(trInf_f1, respMeasure=1).detach().numpy();

  # now get the mask+base response (f1 at base TF)
  maskInd, baseInd = hf_sf.get_mask_base_inds();

  ooh = hf_sf.get_baseOnly_resp(expInfo, dc_resp=resp_A_dc, val_trials=val_trials);
  # note the indexing: [1][x][0][0] for [summary], [dc||f1], [unpack], [mean], respectively
  baseMean_mod_dc = [hf_sf.get_baseOnly_resp(expInfo, dc_resp=x, val_trials=val_trials)[1][0][0][0] for x in [resp_A_dc, resp_B_dc]];
  aah = hf_sf.get_baseOnly_resp(expInfo, f1_base=resp_A_f1[:,baseInd], val_trials=val_trials); 
  baseMean_mod_f1 = [hf_sf.get_baseOnly_resp(expInfo, f1_base=x[:,baseInd], val_trials=val_trials)[1][1][0][0] for x in [resp_A_f1, resp_B_f1]];

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

### Now, plot

# set up model plot info
# i.e. flat model is red, weighted model is green
modColors = ['g', 'r']
try:
  modLabels = ['A: %s' % modA_str, 'B: %s' % modB_str]
except:
  modLabels = None

nrow, ncol = 5, 4;
f, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol*15, nrow*15))

f.suptitle('V1 #%d [%s, f1f0: %.2f] base: %.2f cpd, %.2f%%' % (cellNum, unitNm, f1f0_rat, baseSf_curr, baseCon_curr),
          fontsize='x-large');

maxResp = np.maximum(np.max(respMatrixDC), np.max(respMatrixF1));
maxResp_onlyMask = np.maximum(np.max(respMatrixDC_onlyMask), np.max(respMatrixF1_onlyMask));
maxResp_total = np.maximum(maxResp, maxResp_onlyMask);
overall_ylim = [0, 1.2*maxResp_total];
# also get the bounds for the AbLe plot - only DC
AbLe_mn = np.nanmin(respMatrixDC[:,:,0]-baseDC_mn-respMatrixDC_onlyMask[:,:,0])
AbLe_mx = np.nanmax(respMatrixDC[:,:,0]-baseDC_mn-respMatrixDC_onlyMask[:,:,0])
AbLe_bounds = [np.sign(AbLe_mn)*1.2*np.abs(AbLe_mn), np.maximum(5, 1.2*AbLe_mx)]; # ensure we go at least above 0 with the max

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

    ### first, just the distribution of base responses
    ax[0, measure] = plt.subplot(nrow, 2, 1+measure); # pretend there are only 2 columns
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
        nResps = len(baseOnly[0]); # unpack the array for true length
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

    for ii, rsps in enumerate(resps): # first mask only, then mask+base (data)
        nCons = len(maskCon);
        # we don't plot the F1 at base TF for DC response...
        if measure == 0 and ii == (len(resps)-1):
            continue;

        for mcI, mC in enumerate(maskCon):

            col = [(nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons)];
            # PLOT THE DATA
            ax[1+ii, 2*measure].errorbar(maskSf, rsps[mcI,:,0], rsps[mcI,:,1], fmt='o', clip_on=False,
                                                color=col, label=str(np.round(mC, 2)) + '%')
            if fitBase is None: # then just plot a line for the data
              ax[1+ii, 2*measure].plot(maskSf, rsps[mcI,:,0], clip_on=False, color=col)
            else:
              # PLOT model A (if present)
              ax[1+ii, 2*measure].plot(maskSf, modA_resps[ii][mcI,:,0], color=modColors[0], alpha=1-col[0])
              # PLOT model B (if present)
              ax[1+ii, 2*measure].plot(maskSf, modB_resps[ii][mcI,:,0], color=modColors[1], alpha=1-col[0])

        ax[1+ii, 2*measure].set_xscale('log');
        ax[1+ii, 2*measure].set_xlabel('SF (c/deg)')
        ax[1+ii, 2*measure].set_ylabel('Response (spks/s) [%s]' % lbl)
        ax[1+ii, 2*measure].set_title(labels[ii] + measure_lbl[measure, ii]);
        ax[1+ii, 2*measure].set_ylim(overall_ylim);
        if measure == 0: # only do the blank response reference for DC
            ax[1+ii, 2*measure].axhline(floors[0], linestyle='--', color='b', label=labels_ref[0])
        # i.e. always put the baseOnly reference line...
        ax[1+ii, 2*measure].axhline(floors[1], linestyle='--', color='k', label=labels_ref[1])
        ax[1+ii, 2*measure].legend(fontsize='small');

    # RVC across SF
    for ii, rsps in enumerate(resps): # first mask only, then mask+base (data)
        nSfs = len(maskSf);

        # we don't plot the F1 at base TF for DC response...
        if measure == 0 and ii == (len(resps)-1):
            continue;

        for msI, mS in enumerate(maskSf):

            col = [(nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs)];
            # PLOT THE DATA
            ax[1+ii, 1+2*measure].errorbar(maskCon, rsps[:,msI,0], rsps[:,msI,1], fmt='o', clip_on=False,
                                                color=col, label=str(np.round(mS, 2)) + ' cpd')
            if fitBase is None: # then just plot a line for the data
              ax[1+ii, 1+2*measure].plot(maskCon, rsps[:,msI,0], clip_on=False, color=col)
            else:
              # PLOT model A (if present)
              ax[1+ii, 1+2*measure].plot(maskCon, modA_resps[ii][:,msI,0], color=modColors[0], alpha=1-col[0])
              # PLOT model B (if present)
              ax[1+ii, 1+2*measure].plot(maskCon, modB_resps[ii][:,msI,0], color=modColors[1], alpha=1-col[0])

        ax[1+ii, 1+2*measure].set_xscale('log');
        ax[1+ii, 1+2*measure].set_xlabel('Contrast (%)')
        ax[1+ii, 1+2*measure].set_ylabel('Response (spks/s) [%s]' % lbl)
        ax[1+ii, 1+2*measure].set_title(labels[ii] + measure_lbl[measure, ii])
        ax[1+ii, 1+2*measure].set_ylim(overall_ylim);
        if measure == 0: # only do the blank response for DC
            ax[1+ii, 1+2*measure].axhline(floors[0], linestyle='--', color='b', label=labels_ref[0])
        # i.e. always put the baseOnly reference line...
        ax[1+ii, 1+2*measure].axhline(floors[1], linestyle='--', color='k', label=labels_ref[1])
        ax[1+ii, 1+2*measure].legend(fontsize='small');

    ### joint tuning (mask only)
    #ax[4, measure] = plt.subplot(nrow, 2, 2*nrow-1+measure); # pretend there are only 2 columns

    # temp try...plot contour and trajectory of best fit...
    ax[4, 2*measure].contourf(maskSf, maskCon, refAll)
    ax[4, 2*measure].set_xlabel('Spatial frequency (c/deg)');
    ax[4, 2*measure].set_ylabel('Contrast (%)');
    ax[4, 2*measure].set_xscale('log');
    ax[4, 2*measure].set_yscale('log');
    ax[4, 2*measure].set_title('Joint REF tuning (%s)' % lbl)
    try:
      curr_str = hf_sf.get_resp_str(respMeasure=measure);
      ax[4, 1+2*measure].plot(fit_detailsA[curr_str]['loss'], color=modColors[0]);
      ax[4, 1+2*measure].plot(fit_detailsB[curr_str]['loss'], color=modColors[1]);
      ax[4, 1+2*measure].set_xscale('log');
      ax[4, 1+2*measure].set_yscale('symlog');
      ax[4, 1+2*measure].set_xlabel('Optimization epoch');
      ax[4, 1+2*measure].set_ylabel('Loss');
      ax[4, 1+2*measure].set_title('Optimization (%s): %.2f|%.2f' % (lbl, *lossVals[measure]), fontsize='x-large')
    except:
      ax[4, 1+2*measure].axis('off');

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
        if measure==1: # Abramov/Levine sub. -- only DC has this analysis
            pass;
        else:
            ax[3+j, 2*measure].set_ylabel('Difference (R(m+b) - R(b)%s (spks/s) [%s]' % (txt,lbl))
            ax[3+j, 2*measure].axhline(0, color='k', linestyle='--')
        ax[3+j, 2*measure].legend(fontsize='small');

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
        ax[3+j, 1+2*measure].legend(fontsize='small');

#sns.despine(offset=10)
f.tight_layout(rect=[0, 0.03, 1, 0.95])

#########
# --- Plot secondary things - filter, normalization, nonlinearity, etc
#########
if fitBase is not None: # then we can plot some model details

  fDetails = plt.figure();
  fDetails.set_size_inches(w=40,h=25)

  # make overall title
  fDetails.suptitle('DC <---- |model details| ----> F1');

  detailSize = (3, 4);
  detailSize_filters = (3,2); # for the filters, we'll just pretend we have two columns

  respTypes = [None, None]; # todo: [dcResps, f1Resps], figure out how to package, esp. with f1 having mask & base
  colToAdd = [0, 2]; # we add +2 if doing f1 details
  # ordering of labels/model parameters will be: modA/modB (following form of modColors/modLabels from above)
  whichModels = [[modFit_A_dc, modFit_B_dc], [modFit_A_f1, modFit_B_f1]];
  for (i, resps), colAdd, currMods in zip(enumerate(respTypes), colToAdd, whichModels):

    # TODO: poisson test - mean/var for each condition (i.e. sfXdispXcon)
    '''
    curr_ax = plt.subplot2grid(detailSize, (0, 0)); # set the current subplot location/size[default is 1x1]
    sns.despine(ax=curr_ax, offset=5, trim=False);
    val_conds = ~np.isnan(respMean);
    gt0 = np.logical_and(respMean[val_conds]>0, respVar[val_conds]>0);
    plt.loglog([0.01, 1000], [0.01, 1000], 'k--');
    plt.loglog(respMean[val_conds][gt0], np.square(respVar[val_conds][gt0]), 'o');
    # skeleton for plotting modulated poisson prediction
    if lossType == 3: # i.e. modPoiss
      mean_vals = np.logspace(-1, 2, 50);
      varGains  = [x[7] for x in modFits];
      [plt.loglog(mean_vals, mean_vals + varGain*np.square(mean_vals)) for varGain in varGains];
    plt.xlabel('Mean (imp/s)');
    plt.ylabel('Variance (imp/s^2)');
    plt.title('Super-poisson?');
    plt.axis('equal');
    '''

    # response nonlinearity
    modExps = [x[3] for x in currMods]; # respExp is in location [3]
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
    plt.legend(fontsize='medium');

    # plot model details - exc/suppressive components
    omega = np.logspace(-2, 2, 1000);
    sfExc = [];
    for md, lgnOn in zip(currMods, lgnTypes):
      prefSf = md[0];

      if excType == 1:
        ### deriv. gauss
        dOrder = md[1];
        sfRel = omega/prefSf;
        s     = np.power(omega, dOrder) * np.exp(-dOrder/2 * np.square(sfRel));
        sMax  = np.power(prefSf, dOrder) * np.exp(-dOrder/2);
        sfExcCurr = s/sMax;
      if excType == 2:
        ### flex. gauss
        sigLow = md[1];
        sigHigh = md[-1-np.sign(lgnOn)]; # if we have an lgnFrontEnd, then mWeight is the last param, so we'll go back one more from end
        sfRel = np.divide(omega, prefSf);
        # - set the sigma appropriately, depending on what the stimulus SF is
        sigma = np.multiply(sigLow, [1]*len(sfRel));
        sigma[[x for x in range(len(sfRel)) if sfRel[x] > 1]] = sigHigh;
        # - now, compute the responses (automatically normalized, since max gaussian value is 1...)
        s     = [np.exp(-np.divide(np.square(np.log(x)), 2*np.square(y))) for x,y in zip(sfRel, sigma)];
        sfExcCurr = s; 

      sfExc.append(sfExcCurr);

    # -- now the simple normalization
    curr_ax = plt.subplot2grid(detailSize_filters, (1, 0+i));
    # Remove top/right axis, put ticks only on bottom/left
    sns.despine(ax=curr_ax, offset=5);
    plt.semilogx([omega[0], omega[-1]], [0, 0], 'k--')
    plt.semilogx([.01, .01], [-1.5, 1], 'k--')
    plt.semilogx([.1, .1], [-1.5, 1], 'k--')
    plt.semilogx([1, 1], [-1.5, 1], 'k--')
    plt.semilogx([10, 10], [-1.5, 1], 'k--')
    plt.semilogx([100, 100], [-1.5, 1], 'k--')
    # now the real stuff
    unwt_weights = np.sqrt(hf.genNormWeightsSimple(omega, None, None));
    sfNormSim = unwt_weights/np.amax(np.abs(unwt_weights));
    gs_mean = currMods[0][8]; # currMods[0] is the V1 model - position 8/9 are norm mean/std 
    gs_std = currMods[0][9];
    wt_weights = np.sqrt(hf.genNormWeightsSimple(omega, gs_mean, gs_std));
    sfNormTuneSim = wt_weights/np.amax(np.abs(wt_weights));
    sfNormsSimple = [sfNormTuneSim, sfNormSim]; # pack as wght, LGN
    [plt.semilogx(omega, exc, '%s-' % cc, label='exc[%s]' % s) for exc, cc, s in zip(sfExc, modColors, modLabels)]
    [plt.semilogx(omega, norm, '%s--' % cc, label='inh[%s]' % s) for norm, cc, s in zip(sfNormsSimple, modColors, modLabels)]
    plt.xlim([omega[0], omega[-1]]);
    plt.ylim([-0.1, 1.1]);
    plt.xlabel('spatial frequency (c/deg)');
    plt.ylabel('Normalized response (a.u.)');
    plt.legend(fontsize='medium');

    # print, in text, model parameters:
    curr_ax = plt.subplot2grid(detailSize, (2, 0+colAdd));
    plt.text(0.5, 0.6, 'order: %s, %s' % (*modLabels, ), fontsize=24, horizontalalignment='center', verticalalignment='center');
    plt.text(0.5, 0.5, 'prefSf: %.2f, %.2f' % (currMods[0][0], currMods[1][0]), fontsize=24, horizontalalignment='center', verticalalignment='center');
    normA = np.exp(currMods[0][8]) if normTypes[0]==2 else np.nan;
    normB = np.exp(currMods[1][8]) if normTypes[1]==2 else np.nan;
    plt.text(0.5, 0.4, 'normSf: %.2f, %.2f' % (normA, normB), fontsize=24, horizontalalignment='center', verticalalignment='center');
    if excType == 1:
      plt.text(0.5, 0.3, 'derivative order: %.2f, %.2f' % (currMods[0][1], currMods[1][1]), fontsize=24, horizontalalignment='center', verticalalignment='center');
    elif excType == 2:
      plt.text(0.5, 0.3, 'sig: %.2f|%.2f, %.2f|%.2f' % (currMods[0][1], currMods[0][-1], currMods[1][1], currMods[1][-1]), fontsize=24, horizontalalignment='center', verticalalignment='center');
    plt.text(0.5, 0.2, 'response scalar: %.2f, %.2f' % (currMods[0][4], currMods[1][4]), fontsize=24, horizontalalignment='center', verticalalignment='center');
    plt.text(0.5, 0.1, 'sigma (con): %.2f, %.2f' % (np.power(10, currMods[0][2]), np.power(10, currMods[1][2])), fontsize=24, horizontalalignment='center', verticalalignment='center');


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
if fitBase is None:

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
          baseOnly_curr = np.logical_and(baseOnlyTr, np.logical_and(byTrial['sf'][1,:]==baseSf_curr,
                                                                   byTrial['con'][1,:]==baseCon_curr))
          baseDC, baseF1 = respDistr[0][ii], respDistr[1][ii];

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

              data_sub = np.copy(data);
              data_sub[:,:,0] = data[:,:,0]-np.mean(baseOnly);

              ### first, just the distribution of base responses
              ax[0, measure] = plt.subplot(nrow, 2, 1+measure); # pretend there are only 2 columns

              sns.distplot(baseOnly, ax=ax[0, measure], kde=False);
              base_mn, base_sem = np.mean(baseOnly), np.std(baseOnly)/np.sqrt(len(baseOnly));

              ax[0, measure].set_xlim(xlim_base)
              ax[0, measure].set_title('[%s] mn|sem = %.2f|%.2f' % (lbl, base_mn, base_sem))
              if measure == 0:
                  ax[0, measure].axvline(baseline, linestyle='--', color='r')

              # SF tuning with contrast
              resps = [maskOnly, data, data_baseTf]; #need to plot data_baseTf for f1
              labels = ['mask', 'mask+base', 'mask+base']
              measure_lbl = np.vstack((['', '', ''], ['', ' (mask TF)', ' (base TF)'])); # specify which TF, if F1 response
              labels_ref = ['blank', 'base']
              floors = [baseline, base_mn]; # i.e. the blank response, then the response to the base alone

              #####

              # SF across con
              for ii, rsps in enumerate(resps): # first mask only, then mask+base (data)
                  nCons = len(maskCon);
                  # we don't plot the F1 at base TF for DC response...
                  if measure == 0 and ii == (len(resps)-1):
                      continue;

                  for mcI, mC in enumerate(maskCon):
                      col = [(nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons)];

                      curr_line = ax[1+ii, 2*measure].errorbar(maskSf, rsps[mcI,:,0], rsps[mcI,:,1], marker='o', 
                                                          color=col, label=str(np.round(mC, 2)) + '%')
                  ax[1+ii, 2*measure].set_xscale('log');
                  ax[1+ii, 2*measure].set_xlabel('SF (c/deg)')
                  ax[1+ii, 2*measure].set_ylabel('Response (spks/s) [%s]' % lbl)
                  ax[1+ii, 2*measure].set_title(labels[ii] + measure_lbl[measure, ii]);
                  ax[1+ii, 2*measure].set_ylim(overall_ylim);
                  if measure == 0: # only do the blank response for DC
                      ax[1+ii, 2*measure].axhline(baseline, linestyle='--', color='r', label=labels_ref[0])
                  # i.e. always put the baseOnly reference line...
                  ax[1+ii, 2*measure].axhline(base_mn, linestyle='--', color='b', label=labels_ref[1])
                  ax[1+ii, 2*measure].legend();

              # RVC across SF
              for ii, rsps in enumerate(resps): # first mask only, then mask+base (data)
                  nSfs = len(maskSf);

                  # we don't plot the F1 at base TF for DC response...
                  if measure == 0 and ii == (len(resps)-1):
                      continue;

                  for msI, mS in enumerate(maskSf):

                      col = [(nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs)];

                      curr_line = ax[1+ii, 1+2*measure].errorbar(maskCon, rsps[:,msI,0], rsps[:,msI,1], marker='o', 
                                                          color=col, label=str(np.round(mS, 2)) + ' cpd')
                  ax[1+ii, 1+2*measure].set_xscale('log');
                  ax[1+ii, 1+2*measure].set_xlabel('Contrast (%)')
                  ax[1+ii, 1+2*measure].set_ylabel('Response (spks/s) [%s]' % lbl)
                  ax[1+ii, 1+2*measure].set_title(labels[ii] + measure_lbl[measure, ii])
                  ax[1+ii, 1+2*measure].set_ylim(overall_ylim);
                  if measure == 0: # only do the blank response for DC
                      ax[1+ii, 1+2*measure].axhline(floors[0], linestyle='--', color='r', label=labels_ref[0])
                  # i.e. always put the baseOnly reference line...
                  ax[1+ii, 1+2*measure].axhline(floors[1], linestyle='--', color='b', label=labels_ref[1])
                  ax[1+ii, 1+2*measure].legend(fontsize='small');

              ### joint tuning
              ax[4, measure] = plt.subplot(nrow, 2, 2*nrow-1+measure); # pretend there are only 2 columns

              ax[4, measure].contourf(maskSf_ref, maskCon_ref, refAll);
              ax[4, measure].set_xlabel('Spatial frequency (c/deg)');
              ax[4, measure].set_ylabel('Contrast (%)');
              ax[4, measure].set_xscale('log');
              ax[4, measure].set_yscale('log');
              ax[4, measure].set_title('Joint REF tuning (%s)' % lbl)

              # SF tuning with contrast [rows 1-4, column 1 (& 3)]
              lines = []; linesNorm = []; linesAbLe = [];
              nCons = len(maskCon);
              for mcI, mC in enumerate(maskCon):
                  col = [(nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons), (nCons-mcI-1)/float(nCons)];

                  if measure == 0:
                      curr_line = ax[3, 2*measure].errorbar(maskSf, data_sub[mcI,:,0]-maskOnly[mcI,:,0], data_sub[mcI,:,1],
                                                            marker='o', color=col, label=str(np.round(mC, 2)) + '%')
                      linesAbLe.append(curr_line);
                      ax[3, 2*measure].set_ylim(ylim_diffsAbLe);

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

              # RVC across SF [rows 1-4, column 2 (& 4)]
              lines = []; linesNorm = []; linesAbLe = [];
              nSfs = len(maskSf);
              for msI, mS in enumerate(maskSf):
                  col = [(nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs), (nSfs-msI-1)/float(nSfs)];

                  if measure == 0:
                      curr_line = ax[3, 1+2*measure].errorbar(maskCon, data_sub[:,msI,0] - maskOnly[:,msI,0], data_sub[:,msI,1],
                                                              marker='o', color=col, label=str(np.round(mS, 2)) + ' cpd')
                      linesAbLe.append(curr_line);
                      ax[3, 1+2*measure].set_ylim(ylim_diffsAbLe)

              for (j,ylim),txt in zip(enumerate(ylim_diffs), diff_endings):
                  ax[3+j, 1+2*measure].set_xscale('log');
                  ax[3+j, 1+2*measure].set_xlabel('Contrast (%%)')
                  ax[3+j, 1+2*measure].set_ylabel('Difference (R(m+b) - R(b)%s (spks/s) [%s]' % (txt, lbl))
                  if measure==1: # Abramov/Levine sub. -- only DC has this analysis
                      pass;
                  else:
                      ax[3+j, 1+2*measure].axhline(0, color='k', linestyle='--')
                  ax[3+j, 1+2*measure].legend();

          sns.despine(offset=10)
          f.tight_layout(rect=[0, 0.03, 1, 0.95])

          saveName = "/cell_%03d_both_sf%03d_con%03d.pdf" % (cellNum, np.int(100*baseSf_curr), np.int(100*baseCon_curr))
          full_save = os.path.dirname(str(save_loc + '%s/cell_%03d/' % (expName, cellNum)));
          if not os.path.exists(full_save):
              os.makedirs(full_save);
          pdfSv = pltSave.PdfPages(full_save + saveName);
          pdfSv.savefig(f)
          plt.close(f)
          pdfSv.close()
