# coding: utf-8

#### NOTE: Based on plot_diagnose_vLGN.py
# i.e. we'll compare two models (if NOT just plotting the data)
# - WEIGHTED GAIN CONTROL WITH NO LGN 
# -- VS
# - FLAT GAIN CONTROL WITH LGN FRONT END

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
# TODO: migrate this to actual .mplstyle sheet
rcParams['font.size'] = 20;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 2.5;
rcParams['lines.markeredgewidth'] = 0; # remove edge??
rcParams['axes.linewidth'] = 1.5;
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

minorWid, minorLen = 2, 12;
majorWid, majorLen = 5, 25;

cellNum  = int(sys.argv[1]);
excType  = int(sys.argv[2]);
lossType = int(sys.argv[3]);
expDir   = sys.argv[4]; 
lgnFrontEnd = int(sys.argv[5]);
diffPlot = int(sys.argv[6]);
intpMod  = int(sys.argv[7]);
kMult  = float(sys.argv[8]);

if len(sys.argv) > 9:
  fixRespExp = float(sys.argv[9]);
  if fixRespExp <= 0: # this is the code to not fix the respExp
    fixRespExp = None;
else:
  fixRespExp = None; # default (see modCompare.ipynb for details)

if len(sys.argv) > 10:
  respVar = int(sys.argv[10]);
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
### FITLIST
if excType == 1:
  fitBase = 'fitList_pyt_200417'; # excType 1
  #fitBase = 'fitList_pyt_201017'; # excType 1
elif excType == 2:
  fitBase = 'fitList_pyt_200507'; # excType 2
  #fitBase = 'fitList_pyt_201107'; # excType 2
else:
  fitBase = None;

if fitBase is not None:
  if lossType == 4: # chiSq...
    fitBase = '%s%s' % (fitBase, hf.chiSq_suffix(kMult));

  if fixRespExp is not None:
    fitBase = '%s_re%d' % (fitBase, np.round(fixRespExp*10)); # suffix to indicate that the response exponent is fixed...

  # now, LGN-specific naming
  if fixRespExp is not None:
    fitBase_lgn = '%s_re%d' % (fitBase, np.round(fixRespExp*10)); # suffix to indicate that the response exponent is fixed...
  else:
    fitBase_lgn = fitBase;

  if lgnFrontEnd == 1:
    fitBase_lgn = '%s_LGN' % fitBase_lgn
  elif lgnFrontEnd == 2:
    fitBase_lgn = '%s_LGNb' % fitBase_lgn
  elif lgnFrontEnd == 99:
    fitBase_lgn = '%s_jLGN' % fitBase_lgn

  # first the fit type
  fitSuf_fl = '_flat';
  fitSuf_wg = '_wght';
  # then the loss type
  if lossType == 1:
    lossSuf = '_sqrt.npy';
    loss = lambda resp, pred: np.sum(np.square(np.sqrt(resp) - np.sqrt(pred)));
  elif lossType == 2:
    lossSuf = '_poiss.npy';
    loss = lambda resp, pred: poisson.logpmf(resp, pred);
  elif lossType == 3:
    lossSuf = '_modPoiss.npy';
    loss = lambda resp, r, p: np.log(nbinom.pmf(resp, r, p));
  elif lossType == 4:
    lossSuf = '_chiSq.npy';
    # LOSS HERE IS TEMPORARY
    loss = lambda resp, pred: np.sum(np.square(np.sqrt(resp) - np.sqrt(pred)));

  # NOTE: We choose weighted gain control for non-LGN model, flat gain control for the model with the LGN front end
  fitName = str(fitBase + fitSuf_wg + lossSuf);
  fitName_lgn = str(fitBase_lgn + fitSuf_fl + lossSuf);

  fitList = hf.np_smart_load(data_loc + fitName); # V1 only
  fitList_lgn = hf.np_smart_load(data_loc + fitName_lgn); # with LGN, no tuned gain control

  dc_str = hf_sf.get_resp_str(respMeasure=0);
  f1_str = hf_sf.get_resp_str(respMeasure=1);

  modFit_V1_dc = fitList[cellNum-1][dc_str]['params']; # 
  modFit_lgn_dc = fitList_lgn[cellNum-1][dc_str]['params']; # 
  modFit_V1_f1 = fitList[cellNum-1][f1_str]['params']; # 
  modFit_lgn_f1 = fitList_lgn[cellNum-1][f1_str]['params']; # 

  normTypes = [2, 1]; # weighted, then flat
  lgnTypes = [0, lgnFrontEnd];

  expInd = -1;
  newMethod = 1;

  mod_V1_dc  = mrpt.sfNormMod(modFit_V1_dc, expInd=expInd, excType=excType, normType=2, lossType=lossType, lgnFrontEnd=0, newMethod=newMethod)
  mod_LGN_dc = mrpt.sfNormMod(modFit_lgn_dc, expInd=expInd, excType=excType, normType=1, lossType=lossType, lgnFrontEnd=lgnFrontEnd, newMethod=newMethod)
  mod_V1_f1  = mrpt.sfNormMod(modFit_V1_f1, expInd=expInd, excType=excType, normType=2, lossType=lossType, lgnFrontEnd=0, newMethod=newMethod)
  mod_LGN_f1 = mrpt.sfNormMod(modFit_lgn_f1, expInd=expInd, excType=excType, normType=1, lossType=lossType, lgnFrontEnd=lgnFrontEnd, newMethod=newMethod)

else: # we will just plot the data
  fitList_fl = None;
  fitList_wg = None;

# set the save directory to save_loc, then create the save directory if needed
if fitBase is not None:
  if diffPlot == 1:
    compDir  = str(fitBase + '_diag' + lossSuf + '/diff');
  else:
    compDir  = str(fitBase + '_diag' + lossSuf);
  if intpMod == 1:
    compDir = str(compDir + '/intp');
  subDir   = compDir.replace('fitList', 'fits').replace('.npy', '');
  save_loc = str(save_loc + subDir + '/');
else:
  save_loc = str(save_loc + 'data_only/');

if not os.path.exists(save_loc):
  os.makedirs(save_loc);

conDig = 3; # round contrast to the 3rd digit

dataList = hf.np_smart_load(data_loc + 'dataList.npy')

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

  resp_V1_dc  = mod_V1_dc.forward(trInf_dc, respMeasure=0).detach().numpy();
  resp_LGN_dc = mod_LGN_dc.forward(trInf_dc, respMeasure=0).detach().numpy();
  resp_V1_f1  = mod_V1_f1.forward(trInf_f1, respMeasure=1).detach().numpy();
  resp_LGN_f1 = mod_LGN_f1.forward(trInf_f1, respMeasure=1).detach().numpy();

  # now get the mask+base response (f1 at base TF)
  maskInd, baseInd = hf_sf.get_mask_base_inds();

  ooh = hf_sf.get_baseOnly_resp(expInfo, dc_resp=resp_V1_dc, val_trials=val_trials);
  # note the indexing: [1][x][0][0] for [summary], [dc||f1], [unpack], [mean], respectively
  baseMean_mod_dc = [hf_sf.get_baseOnly_resp(expInfo, dc_resp=x, val_trials=val_trials)[1][0][0][0] for x in [resp_V1_dc, resp_LGN_dc]];
  aah = hf_sf.get_baseOnly_resp(expInfo, f1_base=resp_V1_f1[:,baseInd], val_trials=val_trials); 
  baseMean_mod_f1 = [hf_sf.get_baseOnly_resp(expInfo, f1_base=x[:,baseInd], val_trials=val_trials)[1][1][0][0] for x in [resp_V1_f1, resp_LGN_f1]];

  # ---- V1 model responses
  respMatrix_V1_dc, respMatrix_V1_f1 = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, dc_resp=resp_V1_dc, f1_base=resp_V1_f1[:,baseInd], f1_mask=resp_V1_f1[:,maskInd], val_trials=val_trials); # i.e. get the base response for F1
  # and get the mask only response (f1 at mask TF)
  respMatrix_V1_dc_onlyMask, respMatrix_V1_f1_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, dc_resp=resp_V1_dc, f1_base=resp_V1_f1[:,baseInd], f1_mask=resp_V1_f1[:,maskInd], val_trials=val_trials); # i.e. get the maskONLY response
  # and get the mask+base response (but f1 at mask TF)
  _, respMatrix_V1_f1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, dc_resp=resp_V1_dc, f1_base=resp_V1_f1[:,baseInd], f1_mask=resp_V1_f1[:,maskInd], val_trials=val_trials); # i.e. get the maskONLY response
  # ---- LGN model responses
  respMatrix_LGN_dc, respMatrix_LGN_f1 = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0, dc_resp=resp_LGN_dc, f1_base=resp_LGN_f1[:,baseInd], f1_mask=resp_LGN_f1[:,maskInd], val_trials=val_trials); # i.e. get the base response for F1
  # and get the mask only response (f1 at mask TF)
  respMatrix_LGN_dc_onlyMask, respMatrix_LGN_f1_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1, dc_resp=resp_LGN_dc, f1_base=resp_LGN_f1[:,baseInd], f1_mask=resp_LGN_f1[:,maskInd], val_trials=val_trials); # i.e. get the maskONLY response
  # and get the mask+base response (but f1 at mask TF)
  _, respMatrix_LGN_f1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1, dc_resp=resp_LGN_dc, f1_base=resp_LGN_f1[:,baseInd], f1_mask=resp_LGN_f1[:,maskInd], val_trials=val_trials); # i.e. get the maskONLY response

### Get the responses - base only, mask+base [base F1], mask only (mask F1)
baseDistrs, _, baseConds = hf_sf.get_baseOnly_resp(expInfo);
# - unpack DC, F1 distribution of responses per trial
baseDC, baseF1 = baseDistrs;
baseDC_mn, baseF1_mn = np.mean(baseDC), np.mean(baseF1);
# - unpack the SF x CON of the base (guaranteed to have only one set for sfBB_core)
baseSf_curr, baseCon_curr = baseConds[0];
# now get the mask+base response (f1 at base TF)
respMatrixDC, respMatrixF1 = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=0); # i.e. get the base response for F1
# and get the mask only response (f1 at mask TF)
respMatrixDC_onlyMask, respMatrixF1_onlyMask = hf_sf.get_mask_resp(expInfo, withBase=0, maskF1=1); # i.e. get the maskONLY response
# and get the mask+base response (but f1 at mask TF)
_, respMatrixF1_maskTf = hf_sf.get_mask_resp(expInfo, withBase=1, maskF1=1); # i.e. get the maskONLY response

## Reference tuning...
refDC, refF1 = hf_sf.get_mask_resp(expInfo, withBase=0); # i.e. mask only, at mask TF
maskSf, maskCon = expInfo['maskSF'], expInfo['maskCon'];
# - get DC tuning curves
refDC_sf = refDC[-1, :, :]; # highest contrast
prefSf_ind = np.argmax(refDC_sf[:, 0]);
prefSf_DC = maskSf[prefSf_ind];
refDC_rvc = refDC[:, prefSf_ind, :];
# - get F1 tuning curves
refF1_sf = refF1[-1, :, :];
prefSf_ind = np.argmax(refF1_sf[:, 0]);
prefSf_F1 = maskSf[prefSf_ind];
refF1_rvc = refF1[:, prefSf_ind, :];

### Now, plot

# set up model plot info
# i.e. flat model is red, weighted model is green
modColors = ['g', 'r']
modLabels = ['wght', 'LGN']

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
          data_V1 = respMatrix_V1_dc;
          data_LGN = respMatrix_LGN_dc;
          data_V1_onlyMask = respMatrix_V1_dc_onlyMask;
          data_LGN_onlyMask = respMatrix_LGN_dc_onlyMask;
          data_V1_baseTf = None;
          data_LGN_baseTf = None;
          mod_mean_V1 = baseMean_mod_dc[0];
          mod_mean_LGN = baseMean_mod_dc[1];
    elif measure == 1:
        data = respMatrixF1_maskTf;
        data_baseTf = respMatrixF1;
        maskOnly = respMatrixF1_onlyMask;
        baseOnly = baseF1;
        refAll = refF1[:,:,0];
        refSf = refF1_sf;
        refRVC = refF1_rvc;
        refSf_pref = prefSf_F1;
        xlim_base = overall_ylim
        lbl = 'F1'
        if fitBase is not None:
          data_V1 = respMatrix_V1_f1_maskTf;
          data_LGN = respMatrix_LGN_f1_maskTf;
          data_V1_onlyMask = respMatrix_V1_f1_onlyMask;
          data_LGN_onlyMask = respMatrix_LGN_f1_onlyMask;
          data_V1_baseTf = respMatrix_V1_f1;
          data_LGN_baseTf = respMatrix_LGN_f1;
          mod_mean_V1 = baseMean_mod_f1[0];
          mod_mean_LGN = baseMean_mod_f1[1];

    # Now, subtract the baseOnly response from the base+mask response (only used if measure=0, i.e. DC)
    # -- but store it separately 
    data_sub = np.copy(data);
    data_sub[:,:,0] = data[:,:,0]-np.mean(baseOnly);
    if fitBase is not None:
      data_V1_sub = np.copy(data_V1);
      data_V1_sub[:,:,0] = data_V1[:,:,0] - mod_mean_V1;
      data_LGN_sub = np.copy(data_LGN);
      data_LGN_sub[:,:,0] = data_LGN[:,:,0] - mod_mean_LGN;

    ### first, just the distribution of base responses
    ax[0, measure] = plt.subplot(nrow, 2, 1+measure); # pretend there are only 2 columns

    sns.distplot(baseOnly, ax=ax[0, measure], kde=False);
    base_mn, base_sem = np.mean(baseOnly), np.std(baseOnly)/len(baseOnly);

    ax[0, measure].set_xlim(xlim_base)
    ax[0, measure].set_title('[%s] mn|sem = %.2f|%.2f' % (lbl, base_mn, base_sem))
    if measure == 0:
        ax[0, measure].axvline(baseline, linestyle='--', color='b',label='blank')
    if fitBase is not None:
        ax[0, measure].axvline(np.mean(baseOnly), linestyle='--', color='k',label='data mean')
        ax[0, measure].axvline(mod_mean_V1, linestyle='--', color=modColors[0], label='%s mean' % modLabels[0])
        ax[0, measure].axvline(mod_mean_LGN, linestyle='--', color=modColors[1], label='%s mean' % modLabels[1])
    ax[0, measure].legend(fontsize='small');

    # SF tuning with contrast
    resps = [maskOnly, data, data_baseTf]; #need to plot data_baseTf for f1
    if fitBase is not None:
      v1_resps = [data_V1_onlyMask, data_V1, data_V1_baseTf];
      lgn_resps = [data_LGN_onlyMask, data_LGN, data_LGN_baseTf];
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
              # PLOT THE V1 model (if present)
              ax[1+ii, 2*measure].plot(maskSf, v1_resps[ii][mcI,:,0], color=modColors[0], alpha=1-col[0])
              # PLOT THE LGN model (if present)
              ax[1+ii, 2*measure].plot(maskSf, lgn_resps[ii][mcI,:,0], color=modColors[1], alpha=1-col[0])

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
              # PLOT THE V1 model (if present)
              ax[1+ii, 1+2*measure].plot(maskCon, v1_resps[ii][:,msI,0], color=modColors[0], alpha=1-col[0])
              # PLOT THE LGN model (if present)
              ax[1+ii, 1+2*measure].plot(maskCon, lgn_resps[ii][:,msI,0], color=modColors[1], alpha=1-col[0])

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
    ax[4, measure] = plt.subplot(nrow, 2, 2*nrow-1+measure); # pretend there are only 2 columns

    ax[4, measure].contourf(maskSf, maskCon, refAll)
    ax[4, measure].set_xlabel('Spatial frequency (c/deg)');
    ax[4, measure].set_ylabel('Contrast (%)');
    ax[4, measure].set_xscale('log');
    ax[4, measure].set_yscale('log');
    ax[4, measure].set_title('Joint REF tuning (%s)' % lbl)

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
              # PLOT THE V1 model (if present)
              ax[3, 2*measure].plot(maskSf, data_V1_sub[mcI,:,0]-data_V1_onlyMask[mcI,:,0], color=modColors[0], alpha=1-col[0])
              # PLOT THE LGN model (if present)
              ax[3, 2*measure].plot(maskSf, data_LGN_sub[mcI,:,0]-data_LGN_onlyMask[mcI,:,0], color=modColors[1], alpha=1-col[0])

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
              # PLOT THE V1 model (if present)
              ax[3, 1+2*measure].plot(maskCon, data_V1_sub[:, msI,0]-data_V1_onlyMask[:, msI,0], color=modColors[0], alpha=1-col[0])
              # PLOT THE LGN model (if present)
              ax[3, 1+2*measure].plot(maskCon, data_LGN_sub[:, msI,0]-data_LGN_onlyMask[:, msI,0], color=modColors[1], alpha=1-col[0])


            ax[3, 1+2*measure].set_ylim(ylim_diffsAbLe)

    for (j,ylim),txt in zip(enumerate(ylim_diffs), diff_endings):
        ax[3+j, 1+2*measure].set_xscale('log');
        ax[3+j, 1+2*measure].set_xlabel('Contrast (%%)')
        if measure==1: # Abramov/Levine sub. -- only DC has this analysis
            pass;
        else:
            ax[3+j, 1+2*measure].axhline(0, color='k', linestyle='--')
        ax[3+j, 1+2*measure].legend(fontsize='small');

sns.despine(offset=10)
f.tight_layout(rect=[0, 0.03, 1, 0.95])

saveName = "/cell_%03d_both.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'core/'));
if not os.path.exists(full_save):
    os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(f)
plt.close(f)
pdfSv.close()


#######################
## var* (if applicable)
#######################

'''

### TODO: Must make fits to sfBB_var* expts...

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
            base_mn, base_sem = np.mean(baseOnly), np.std(baseOnly)/len(baseOnly);

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

'''
