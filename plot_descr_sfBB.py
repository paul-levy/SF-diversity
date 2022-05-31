# coding: utf-8

import os, time, sys
import numpy as np
import matplotlib
import matplotlib.cm as cm
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns
sns.set(style='ticks')

import helper_fcns as hf
import helper_fcns_sfBB as hf_sf

import warnings
warnings.filterwarnings('once');

import pdb

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_style.mplstyle');
from matplotlib import rcParams
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
    rcParams['ytick.minor.size'] = 12; # i.e. don't have minor ticks on y...                                                                                                              

    rcParams['xtick.major.width'] = 2
    rcParams['xtick.minor.width'] = 2
    rcParams['ytick.major.width'] = 2
    rcParams['ytick.minor.width'] = 2

majWidth = 4;
minWidth = 4;
lblSize = 40;

peakFrac = 0.75; # plot fall of to peakFrac of peak, rather than peak or charFreq
inclLegend = 0;

plt_sf_as_rvc = 1;
comm_S_calc = 1;

cellNum   = int(sys.argv[1]);
expDir    = sys.argv[2]; 
descrMod  = int(sys.argv[3]);
descrLoss = int(sys.argv[4]);
joint     = int(sys.argv[5]);
phAdj     = int(sys.argv[6]);
rvcMod    = int(sys.argv[7]);
if len(sys.argv) > 8:
  respVar = int(sys.argv[8]);
else:
  respVar = 1;
if len(sys.argv) >9:
  isHPC = int(sys.argv[9]);
else:
  isHPC = 0;
if len(sys.argv) > 10: # plot prediction to all stimuli from spatial rep. of d-DoG-S model???
  ddogs_pred = int(sys.argv[10]);
else:
  ddogs_pred = 1;
if len(sys.argv) > 11:
  forceLog = int(sys.argv[11]); # used for byDisp/allCons_... (sf plots)
else:
  forceLog = 0;

loc_base = os.getcwd() + '/';

data_loc = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + 'figures/';

fracSig = 1;

phAdjSigned = phAdj;
phAdj = np.abs(phAdj);

### DATALIST
expName = hf.get_datalist(expDir, force_full=1);
### DESCRLIST
hpc_str = 'HPC' if isHPC else '';
descrBase = 'descrFits%s_220410' % hpc_str;
#descrBase = 'descrFits%s_220323' % hpc_str;
### RVCFITS
rvcBase = 'rvcFits%s_220220' % hpc_str;

##################
### Spatial frequency
##################

modStr  = hf.descrMod_name(descrMod)
fLname  = hf.descrFit_name(descrLoss, descrBase=descrBase, modelName=modStr, joint=joint, phAdj=1 if rvcAdj==1 else None);
descrFits = hf.np_smart_load(data_loc + fLname);
pause_tm = 2.0*np.random.rand();
time.sleep(pause_tm);
# set the save directory to save_loc, then create the save directory if needed
subDir = fLname.replace('Fits', '').replace('.npy', '');
save_loc = str(save_loc + subDir + '/');

if not os.path.exists(save_loc):
  os.makedirs(save_loc);

dataList = hf.np_smart_load(data_loc + expName);

cellName = dataList['unitName'][cellNum-1];
try:
  cellType = dataList['unitType'][cellNum-1];
except: 
  # TODO: note, this is dangerous; thus far, only V1 cells don't have 'unitType' field in dataList, so we can safely do this
  cellType = 'V1'; 
try:
  overwriteExpName = dataList['expType'][cellNum-1];
except:
  overwriteExpName = None;
expInd = -1
vecCorrected = 1;
rvcFits = hf.np_smart_load(data_loc + hf.rvc_fit_name(rvcBase, modNum=rvcMod, dir=None, vecF1=vecCorrected));
force_baseline = False; # plotting baseline will depend on F1/F0 designation

expName = 'sfBB_core';

unitNm = dataList['unitName'][cellNum-1];
cell = hf.np_smart_load('%s%s_sfBB.npy' % (data_loc, unitNm));
expInfo = cell[expName]
byTrial = expInfo['trial'];
f1f0_rat = hf_sf.compute_f1f0(expInfo)[0];
maskSf, maskCon = expInfo['maskSF'], expInfo['maskCon'];

if f1f0_rat >= 1:
    respMeasure = 1; # i.e. simple
else:
    respMeasure = 0;
respStr = 'f1' if respMeasure==1 else 'dc';
# --- get the right descrFits, rvcFits
df_curr = descrFits[cellNum-1][respStr]['mask'];
descrParams = df_curr['params'];
rvcFits = rvcFits[cellNum-1][respStr]['mask'];

### Get the responses - base only, mask+base [base F1], mask only (mask F1)
baseDistrs, baseSummary, baseConds = hf_sf.get_baseOnly_resp(expInfo);
# - unpack DC, F1 distribution of responses per trial
baseDC, baseF1 = baseDistrs;
baseDC_mn, baseF1_mn = np.mean(baseDC), np.mean(baseF1);
if vecCorrected:
    baseDistrs, baseSummary, _ = hf_sf.get_baseOnly_resp(expInfo, vecCorrectedF1=1);
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
baseline_resp = expInfo['blank']['mean'] if respMeasure==0 else int(0);
# --- now, specify the tuning
respMean = respMatrixF1_onlyMask[:,:,0] if respMeasure==1 else respMatrixDC_onlyMask[:,:,0];
respVar = respMatrixF1_onlyMask[:,:,1] if respMeasure==1 else respMatrixDC_onlyMask[:,:,1];
# ----- also transpose, to align with the [sf, con] ordering of all other experiments
respMean = np.transpose(respMean);
respVar = np.transpose(respVar);

# ### Plots

# set up colors, labels
modClr = 'b';
modTxt = 'descr';
dataClr = 'k';
dataTxt = 'data';
refClr = 'm'
refTxt ='ref';

# #### Plots by dispersion

sfs_plot = np.logspace(np.log10(maskSf[0]), np.log10(maskSf[-1]), 100);    

if ddogs_pred and descrMod==3: # i.e. is d-DoG-S model
    all_preds = hf.parker_hawken_all_stim(trialInf, expInd, descrParams, comm_s_calc=comm_S_calc&(joint>0));
    _, _, pred_org, pred_all = hf.organize_resp(all_preds, trialInf, expInd, respsAsRate=True)
else:
    pred_org = None;

n_v_cons = len(maskCon);

fDisp, dispAx = plt.subplots(n_v_cons, 2, figsize=(2*10, n_v_cons*12), sharey=False);

minResp = np.min(np.min(respMean[~np.isnan(respMean[:, :])]));
maxResp = np.max(np.max(respMean[~np.isnan(respMean[:, :])]));
ref_params = descrParams[-1] if joint>0 else None; # the reference parameter is the highest contrast for that dispersion

for c in reversed(range(n_v_cons)):
    c_plt_ind = len(maskCon) - c - 1;
    v_sfs = ~np.isnan(respMean[:, c]);

    currClr = [(n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons)];

    ### left side of plots
    sfVals = maskSf[v_sfs];
    resps  = respMean[v_sfs, c];
    ## plot data
    dispAx[c_plt_ind, 0].errorbar(sfVals, resps,
                                     respVar[v_sfs, c], color=currClr, fmt='o', clip_on=False, label=dataTxt);

    # now, let's also plot the baseline, if complex cell
    if baseline_resp > 0: # i.e. complex cell
      dispAx[c_plt_ind, 0].axhline(baseline_resp, color=currClr, linestyle='dashed');

    ## plot descr fit
    prms_curr = descrParams[c];
    descrResp = hf.get_descrResp(prms_curr, stim_sf=sfs_plot, DoGmodel=descrMod, baseline=baseline_resp, fracSig=fracSig, ref_params=ref_params);
    dispAx[c_plt_ind, 0].plot(sfs_plot, descrResp, color=currClr, label='descr. fit');
    # --- and also ddogs prediction (perhaps...)
    if pred_org is not None:
        dispAx[c_plt_ind, 0].plot(sfVals, baseline_resp + pred_org[v_sfs, c], color=currClr, linestyle='--', clip_on=False, label='pred');

    ## if flexGauss plot peak & frac of peak
    frac_freq = hf.sf_highCut(prms_curr, descrMod, frac=peakFrac, sfRange=(0.1, 15), baseline_sub=baseline_resp);
    if not hf.is_mod_DoG(descrMod): # i.e. non DoG models
      #ctr = hf.sf_com(resps, sfVals);
      pSf = hf.descr_prefSf(prms_curr, dog_model=descrMod, all_sfs=maskSf);
      for ii in range(2):
        dispAx[c_plt_ind, ii].plot(frac_freq, 2, linestyle='None', marker='v', label='(%.2f) highCut(%.1f)' % (peakFrac, frac_freq), color=currClr, alpha=1); # plot at y=1
        #dispAx[c_plt_ind, ii].plot(pSf, 1, linestyle='None', marker='v', label='pSF', color=currClr, alpha=1); # plot at y=1
    ## otherwise, let's plot the char freq. and frac of peak
    elif hf.is_mod_DoG(descrMod): # (single) DoG models
      char_freq = hf.dog_charFreq(prms_curr, descrMod);
      # if it's a DoG, let's also put the parameters in text (left side only)
      try:
        dispAx[c_plt_ind, 0].text(0.05, 0.075, '%d,%.2f' % (*prms_curr[0:2], ), transform=dispAx[c_plt_ind,0].transAxes, horizontalalignment='left', fontsize='small', verticalalignment='bottom');
        dispAx[c_plt_ind, 0].text(0.05, 0.025, '%.2f,%.2f' % (*prms_curr[2:], ), transform=dispAx[c_plt_ind,0].transAxes, horizontalalignment='left', fontsize='small', verticalalignment='bottom');
        for ii in range(2):
          dispAx[c_plt_ind, ii].plot(frac_freq, 2, linestyle='None', marker='v', label='(%.2f) highCut(%.1f)' % (peakFrac, frac_freq), color=currClr, alpha=1); # plot at y=1
          #dispAx[c_plt_ind, ii].plot(char_freq, 1, linestyle='None', marker='v', label='$f_c$', color=currClr, alpha=1); # plot at y=1
      except:
        pass; # why might this not work? If we only fit disp=0!

    dispAx[c_plt_ind, 0].set_title('contrast: %d%%' % (100*maskCon[c]));

    minResp_toPlot = 1e-0;
    ## plot everything again on log-log coordinates...
    # first data
    if baseline_resp > 0: # is not None
      to_sub = baseline_resp;
    else:
      to_sub = np.array(0);
    resps_curr = respMean[v_sfs, c] - to_sub;
    abvThresh = [resps_curr>minResp_toPlot];
    var_curr = respVar[v_sfs, c][abvThresh];
    dispAx[c_plt_ind, 1].errorbar(maskSf[v_sfs][abvThresh], resps_curr[abvThresh], var_curr, 
          fmt='o', color=currClr, clip_on=False, markersize=9, label=dataTxt);

    # plot descriptive model fit -- and inferred characteristic frequency (or peak...)
    prms_curr = descrParams[c];
    descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod, baseline=baseline_resp, fracSig=fracSig, ref_params=ref_params);
    descr_curr = descrResp - to_sub;
    abvThresh = [descr_curr>minResp_toPlot]
    dispAx[c_plt_ind, 1].plot(sfs_plot[abvThresh], descr_curr[abvThresh], color=currClr, label='descr. fit', clip_on=False)

    if pred_org is not None:
        to_cut = pred_org[v_sfs, c];
        abvThresh = [to_cut>minResp_toPlot];
        dispAx[c_plt_ind, 1].plot(sfVals[abvThresh], to_cut[abvThresh], color=currClr, linestyle='--', clip_on=False, label='pred');

    if not hf.is_mod_DoG(descrMod):
      psf = hf.descr_prefSf(prms_curr, dog_model=descrMod);
      #if psf != np.nan: 
      #  dispAx[c_plt_ind, 1].plot(psf, 1, 'b', color='k', label='peak freq', clip_on=False);
    elif hf.is_mod_DoG(descrMod): # diff-of-gauss
      # now plot characteristic frequency!  
      char_freq = hf.dog_charFreq(prms_curr, descrMod);
      #if char_freq != np.nan:
      #  dispAx[c_plt_ind, 1].plot(char_freq, 1, 'v', color='k', label='char. freq', clip_on=False);

    dispAx[c_plt_ind, 1].set_title('log-log: %.1f%% varExpl' % df_curr['varExpl'][c], fontsize='medium');
    dispAx[c_plt_ind, 1].set_xscale('log');
    dispAx[c_plt_ind, 1].set_yscale('log'); # double log
    dispAx[c_plt_ind, 1].set_ylim((minResp_toPlot, 1.5*maxResp));
    dispAx[c_plt_ind, 1].set_aspect('equal');

    ## Now, set things for both plots (formatting)
    for i in range(2):

      dispAx[c_plt_ind, i].set_xlim((min(maskSf), max(maskSf)));
      if min(maskSf) == max(maskSf):
        print('cell % has bad sfs' % cellNum);

      dispAx[c_plt_ind, i].set_xscale('log');
      if c_plt_ind == len(maskCon)-1:
        dispAx[c_plt_ind, i].set_xlabel('sf (c/deg)'); 

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      #dispAx[c_plt_ind, i].tick_params(labelsize=lblSize, width=majWidth, direction='out');
      #dispAx[c_plt_ind, i].tick_params(width=minWidth, which='minor', direction='out'); # minor ticks, too...	
      sns.despine(ax=dispAx[c_plt_ind, i], offset=10, trim=False); 

    dispAx[c_plt_ind, 0].set_ylim((np.minimum(-5, minResp-5), 1.5*maxResp));
    dispAx[c_plt_ind, 0].set_ylabel('resp (sps)');

fDisp.suptitle('%s #%d (f1f0: %.2f)' % (cellType, cellNum, f1f0_rat));
fDisp.subplots_adjust(wspace=0.1, top=0.95);

saveName = "/cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'byDisp/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(fDisp)
plt.close(fDisp)
pdfSv.close();

# #### All SF tuning on one graph, split by dispersion
sfs_plot = np.logspace(np.log10(maskSf[0]), np.log10(maskSf[-1]), 100);
minResp_toPlot = 1e-0;
n_v_cons = len(maskCon);

fDisp, dispAx = plt.subplots(1, 2, figsize=(35, 20), sharey=True, sharex=True);
fDisp.suptitle('%s #%d [%s] (f1f0 %.2f)' % (cellType, cellNum, unitNm, f1f0_rat));

maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));  
minToPlot = 5e-1;
ref_params = descrParams[-1] if joint>0 else None; # the reference parameter is the highest contrast for that dispersion

lines = [];
for c in reversed(range(n_v_cons)):
    v_sfs = ~np.isnan(respMean[:, c]);        

    # plot data [0]
    col = [(n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons)];
    plot_resp = respMean[v_sfs, c];
    if forceLog == 1:
      if baseline_resp > 0: #is not None:
        to_sub = baseline_resp;
      else:
        to_sub = np.array(0);
      plot_resp = plot_resp - to_sub;

    curr_line, = dispAx[0].plot(maskSf[v_sfs][plot_resp>minToPlot], plot_resp[plot_resp>minToPlot], '-o', clip_on=False, \
                                   color=col, label='%s%%' % (str(int(100*np.round(maskCon[c], 2)))));
    if baseline_resp > 0:
        dispAx[0].axhline(baseline_resp, linestyle='--', color='k');
    lines.append(curr_line);

    # plot descr fit [1]
    prms_curr = descrParams[c];
    descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod, baseline=baseline_resp, fracSig=fracSig, ref_params=ref_params);
    dispAx[1].plot(sfs_plot, descrResp-to_sub, color=col);

for i in range(len(dispAx)):
  dispAx[i].set_xlim((0.5*min(maskSf), 1.2*max(maskSf)));

  dispAx[i].set_xscale('log');
  if expDir == 'LGN/' or forceLog == 1: # we want double-log if it's the LGN!
    dispAx[i].set_yscale('log');
    #dispAx[i].set_ylim((minToPlot, 1.5*maxResp));
    dispAx[i].set_ylim((5e-1, 300)); # common y axis for ALL plots
    logSuffix = 'log_';
  else:
    dispAx[i].set_ylim((np.minimum(-5, minResp-5), 1.5*maxResp));
    logSuffix = '';

  dispAx[i].set_xlabel('sf (c/deg)'); 

  # Set ticks out, remove top/right axis, put ticks only on bottom/left
  #dispAx[i].tick_params(labelsize=15, width=2, length=16, direction='out');
  #dispAx[i].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
  sns.despine(ax=dispAx[i], offset=10, trim=False); 

  lbl_str = '' if i==0 else 'above baseline ';
  dispAx[i].set_ylabel('resp %s(sps)' % lbl_str);
  dispAx[i].set_title('sf tuning');
  dispAx[i].legend(fontsize='large'); 

saveName = "/allCons_%scell_%03d.pdf" % (logSuffix, cellNum)
full_save = os.path.dirname(str(save_loc + 'byDisp/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(fDisp)
plt.close(fDisp)
pdfSv.close()

##################
#### Response versus contrast (RVC; contrast response function, CRF)
##################

cons_plot = np.geomspace(np.minimum(0.01, maskCon), np.max(maskCon), 100); # go down to at least 1% contrast

# #### Plot contrast response functions with descriptive RVC model predictions

# which sfs have at least one contrast presentation? within a dispersion, all cons have the same # of sfs
n_v_sfs = len(maskSf);
n_rows = int(np.ceil(n_v_sfs/np.floor(np.sqrt(n_v_sfs)))); # make this close to a rectangle/square in arrangement (cycling through sfs)
n_cols = int(np.ceil(n_v_sfs/n_rows));
fRVC, rvcAx = plt.subplots(n_rows, n_cols, figsize=(n_cols*12, n_rows*12), sharey=True);

fRVC.suptitle('%s #%d [%s] (f1f0 %.2f)' % (cellType, cellNum, unitNm, f1f0_rat));

for sf in range(n_v_sfs):
    row_ind = int(sf/n_cols);
    col_ind = np.mod(sf, n_cols);
    sf_ind = sf;
    if n_cols > 1:
      plt_y = (row_ind, col_ind);
    else: # pyplot makes it (n_rows, ) if n_cols == 1
      plt_y = (row_ind, );


    n_cons = len(maskCon);

    # organize (measured) responses
    resp_curr = np.reshape([respMean[sf_ind, :]], (n_cons, ));
    var_curr  = np.reshape([respVar[sf_ind, :]], (n_cons, ));

    if forceLog == 1:
      if baseline_resp > 0: #is not None:
        to_sub = baseline_resp;
      else:
        to_sub = np.array(0);
      resp_curr = resp_curr - to_sub;

    rvcAx[plt_y].errorbar(maskCon[resp_curr>minResp_toPlot], resp_curr[resp_curr>minResp_toPlot], var_curr[resp_curr>minResp_toPlot], fmt='o', linestyle='-', clip_on=False, label='data', markersize=9, color=dataClr);

    # RVC descr model - TODO: Fix this discrepancy between f0 and f1 rvc structure? make both like descrFits?
    prms_curr = rvcFits['params'][sf_ind];
    c50 = hf.get_c50(rvcMod, prms_curr); # second to last entry
    if rvcMod == 1 or rvcMod == 2: # naka-rushton/peirce
      rvcResps = hf.naka_rushton(cons_plot, prms_curr)
    elif rvcMod == 0: # i.e. movshon form
      rvcResps = rvcModel(*prms_curr, cons_plot);

    if forceLog == 1:
       rvcResps = rvcResps - to_sub;
    val_inds = np.where(rvcResps>minResp_toPlot)[0];

    rvcAx[plt_y].plot(cons_plot[val_inds], rvcResps[val_inds], color=modClr, \
      alpha=0.7, clip_on=False, label=modTxt);
    rvcAx[plt_y].plot(c50, 1.5*minResp_toPlot, 'v', label='c50', color=modClr, clip_on=False);
    # now, let's also plot the baseline, if complex cell
    if baseline_resp > 0 and forceLog != 1: # i.e. complex cell (baseline_resp is not None) previously
      rvcAx[plt_y].axhline(baseline_resp, color='k', linestyle='dashed');

    rvcAx[plt_y].set_xscale('log', basex=10); # was previously symlog, linthreshx=0.01
    if col_ind == 0:
      rvcAx[plt_y].set_xlabel('contrast', fontsize='medium');
      rvcAx[plt_y].set_ylabel('response (spikes/s)', fontsize='medium');
      rvcAx[plt_y].legend();

    # set axis limits...
    rvcAx[plt_y].set_xlim([0.01, 1]);
    if forceLog == 1:
      rvcAx[plt_y].set_ylim((minResp_toPlot, 1.25*maxResp));
      rvcAx[plt_y].set_yscale('log'); # double log
      rvcAx[plt_y].set_aspect('equal'); 

    try:
      curr_varExpl = rvcFits['varExpl'][sf_ind];
    except:
      curr_varExpl = np.nan;
    rvcAx[plt_y].set_title('sf: %.3f [vE=%.2f%%]' % (maskSf[sf_ind], curr_varExpl), fontsize='large');
    if rvcMod == 0:
      try:
        cg = rvcFits['conGain'][sf_ind]
        rvcAx[plt_y].text(0, 0.95, 'conGain=%.1f' % cg, transform=rvcAx[plt_y].transAxes, horizontalalignment='left', fontsize='small', verticalalignment='top');
      except:
         pass; # not essential...

    # Set ticks out, remove top/right axis, put ticks only on bottom/left
    sns.despine(ax = rvcAx[plt_y], offset = 10, trim=False);
    #rvcAx[plt_y].tick_params(labelsize=lblSize, width=majWidth, direction='out');
    #rvcAx[plt_y].tick_params(width=minWidth, which='minor', direction='out'); # minor ticks, too...

fRVC.tight_layout(rect=[0, 0.03, 1, 0.95])

saveName = "/cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(fRVC)
plt.close(fRVC)
pdfSv.close()

# #### Plot contrast response functions - all sfs on one axis (per dispersion)

nrow, ncol = 1, 2+plt_sf_as_rvc;

fCRF, crfAx = plt.subplots(nrow, ncol, figsize=(ncol*17.5, nrow*20), sharex=False, sharey='row');

fCRF.suptitle('%s #%d [%s] (f1f0 %.2f)' % (cellType, cellNum, unitNm, f1f0_rat));

n_v_sfs = len(maskSf);

maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));
minResp_plot = 1e-0;
ref_params = descrParams[-1] if joint>0 else None; # the reference parameter is the highest contrast for that dispersion

lines_log = [];
for sf in range(n_v_sfs):
    sf_ind = sf;
    v_cons = ~np.isnan(respMean[sf_ind, :]);
    n_cons = sum(v_cons);

    col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];
    con_str = str(np.round(maskSf[sf_ind], 2));
    # plot data
    plot_resp = respMean[sf_ind, v_cons];
    if forceLog == 1:
      if baseline_resp > 0: #is not None:
        to_sub = baseline_resp;
      else:
        to_sub = np.array(0);
      plot_resp = plot_resp - to_sub;

    line_curr, = crfAx[0].plot(v_cons[plot_resp>minResp_plot], plot_resp[plot_resp>minResp_plot], '-o', color=col, \
                                  clip_on=False, markersize=9, label=con_str);
    lines_log.append(line_curr);
    crfAx[0].set_title('RVC data');

    # now RVC model [1]
    prms_curr = rvcFits['params'][sf_ind];
    if rvcMod == 0: # i.e. movshon form
      rvcResps = rvcModel(*prms_curr, cons_plot);
    elif rvcMod == 1 or rvcMod == 2: # naka-rushton (or modified version)
      rvcResps = hf.naka_rushton(cons_plot, prms_curr)

    rvcRespsAdj = rvcResps-to_sub;
    crfAx[1].plot(cons_plot[rvcRespsAdj>minResp_plot], rvcRespsAdj[rvcRespsAdj>minResp_plot], color=col, \
                     clip_on=False, label = con_str);
    crfAx[1].set_title('D: RVC fits');

    # OPTIONAL, plot RVCs as inferred from SF tuning fits - from 22.01.19 onwards
    if plt_sf_as_rvc:
        cons = maskCon;
        try:
            resps_curr = np.array([hf.get_descrResp(descrParams[vc], maskSf[sf_ind], descrMod, baseline=baseline_resp, fracSig=fracSig, ref_params=ref_params) for vc in np.where(maskCon)[0]]) - to_sub;
            crfAx[2].plot(maskCon, resps_curr, color=col, \
                     clip_on=False, linestyle='--', marker='o');
            crfAx[2].set_title('D: RVC from SF fit');
        except:
            pass # this is not essential...


for i in range(len(crfAx)):

  if expDir == 'LGN/' or forceLog == 1: # then plot as double-log
    crfAx[i].set_xscale('log');
    crfAx[i].set_yscale('log');
    crfAx[i].set_ylim((minResp_plot, 1.5*maxResp));
    crfAx[i].set_aspect('equal');
    #crfAx[i].set_ylim((minResp_plot, 300)); # common y axis for ALL plots
    logSuffix = 'log_';
  else:
    crfAx[i].set_xlim([-0.1, 1]);
    crfAx[i].set_ylim([-0.1*maxResp, 1.1*maxResp]);
    logSuffix = '';
  crfAx[i].set_xlabel('contrast');

  # Set ticks out, remove top/right axis, put ticks only on bottom/left
  #crfAx[i].tick_params(labelsize=lblSize, width=majWidth, direction='out');
  #crfAx[i].tick_params(width=minWidth, which='minor', direction='out'); # minor ticks, too...
  sns.despine(ax = crfAx[i], offset=10, trim=False);

  crfAx[i].set_ylabel('resp above baseline (sps)');
  crfAx[i].legend();

saveName = "/allSfs_%scell_%03d.pdf" % (logSuffix, cellNum)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(fCRF)
plt.close(fCRF)
pdfSv.close()

# #### Plot d-DoG-S model in space, all SF tuning curves

if descrMod == 3 or descrMod == 5: # i.e. d-DoG-s

  isMult = True if descrMod == 3 else False; # which parameterization of the d-DoG-S is it?
  sfs_plot = np.logspace(np.log10(maskSf[0]), np.log10(maskSf[-1]), 100);

  n_v_cons = len(maskCon);

  n_rows = int(np.ceil(n_v_cons/np.floor(np.sqrt(n_v_cons)))); # make this close to a rectangle/square in arrangement (cycling through cons)
  n_cols = int(np.ceil(n_v_cons/n_rows));

  fDisp, dispAx = plt.subplots(n_rows, n_cols, figsize=(n_cols*10, n_rows*12), sharey=True);

  minResp = np.min(np.min(respMean[~np.isnan(respMean[:, :])]));
  maxResp = np.max(np.max(respMean[~np.isnan(respMean[:, :])]));

  if comm_S_calc and joint>0: # this only applies when doing joint-across-con fits
      ref_params = descrParams[-1];
  else:
      ref_params = None;

  for c in reversed(range(n_v_cons)):
      row_ind = int((n_v_cons-c-1)/n_cols);
      col_ind = np.mod((n_v_cons-c-1), n_cols);

      c_plt_ind = len(maskCon) - c - 1;
      currClr = [(n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons)];

      dispAx[row_ind, col_ind].set_title('con: %s%%' % str(int(100*(np.round(maskCon[c], 2)))));

      # plot model
      prms_curr = descrParams[c];
      space, samps, dc, df1, df2 = hf.parker_hawken(np.copy(prms_curr), inSpace=True, debug=True, isMult=isMult, ref_params=ref_params);

      dispAx[row_ind, col_ind].plot(samps, space, 'k-')#, label='full');
      # and plot the constitutent parts
      dispAx[row_ind, col_ind].plot(samps, dc, 'k--')#, label='center');
      dispAx[row_ind, col_ind].plot(samps, df1, 'r--')#, label='f1');
      dispAx[row_ind, col_ind].plot(samps, df2, 'b--')#, label='f2');

      if c_plt_ind == 0:
        dispAx[row_ind, col_ind].set_ylabel('sensitivity');
        #dispAx[row_ind, col_ind].legend(fontsize='x-small');
      if c_plt_ind == n_v_cons-1:
        dispAx[row_ind, col_ind].set_xlabel('dva');

      # Add parameters! (in ax-transformed coords, (0,0) is bottom left, (1,1) is top right
      prms_curr_trans = hf.parker_hawken_transform(np.copy(prms_curr), space_in_arcmin=True, isMult=isMult, ref_params=ref_params);
      kc1,xc1,ks1,xs1 = prms_curr_trans[0:4];
      kc2,xc2,ks2,xs2 = prms_curr_trans[4:8];
      g,S = prms_curr_trans[8:];
      # -- first, dog 1; then dog2; finally gain, S
      dispAx[row_ind, col_ind].text(0, 0.16, r"""ctr: $k_c/k_s/x_c/x_s$=%.0f/%.0f/%.2f'/%.2f'""" % (kc1,ks1,xc1,xs1), transform=dispAx[row_ind, col_ind].transAxes, horizontalalignment='left', fontsize='x-small');
      dispAx[row_ind, col_ind].text(0, 0.08, r"""sd: $k_c/k_s/x_c/x_s$=%.0f/%.0f/%.2f'/%.2f'""" % (kc2,ks2,xc2,xs2), transform=dispAx[row_ind, col_ind].transAxes, horizontalalignment='left', fontsize='x-small');
      dispAx[row_ind, col_ind].text(0, 0, r"""$g/S$=%.2f/%.2f'""" % (g,S), transform=dispAx[row_ind, col_ind].transAxes, horizontalalignment='left', fontsize='x-small');

  fDisp.suptitle('%s #%d [%s] (f1f0: %.2f)' % (cellType, cellNum, unitNm, f1f0_rat));

  saveName = "/cell_%03d.pdf" % (cellNum)
  full_save = os.path.dirname(str(save_loc + 'byDisp_ddogs/'));
  if not os.path.exists(full_save):
    os.makedirs(full_save);
  pdfSv = pltSave.PdfPages(full_save + saveName);
  pdfSv.savefig(fDisp)
  plt.close(fDisp)
  pdfSv.close();
