# coding: utf-8

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
import model_responses as mod_resp

import warnings
warnings.filterwarnings('once');

import pdb

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_cluster.mplstyle');
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
lossType = int(sys.argv[2]);
expDir   = sys.argv[3]; 
rvcAdj   = int(sys.argv[4]); # if 1, then let's load rvcFits to adjust responses to F1
diffPlot = int(sys.argv[5]);
intpMod  = int(sys.argv[6]);
if len(sys.argv) > 7:
  respVar = int(sys.argv[7]);
else:
  respVar = 1;

## used for interpolation plot
sfSteps  = 45; # i.e. how many steps between bounds of interest
conSteps = -1;
#nRpts    = 500; # how many repeats for stimuli in interpolation plot?
#nRpts    = 5; # how many repeats for stimuli in interpolation plot?
nRpts    = 3000; # how many repeats for stimuli in interpolation plot?n
nRptsSingle = 5; # when disp = 1 (which is most cases), we do not need so many interpolated points

loc_base = os.getcwd() + '/';
data_loc = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + 'figures/';

### DATALIST
expName = hf.get_datalist(expDir);
#expName = 'dataList.npy';
#expName = 'dataList_glx.npy'
#expName = 'dataList_mr.npy'
### FITLIST
#fitBase = 'fitList_190321c';
#fitBase = 'fitListSPcns_181130c';
#fitBase = 'fitListSP_181202c';
#fitBase = 'fitList_190206c';
#fitBase = 'fitList_190321c';

#fitBase = 'mr_fitList_190502cA';
#fitBase = 'fitList_190502aA';
fitBase = 'fitList_190513cA';
#fitBase = 'fitList_190516cA';
#fitBase = 'holdout_fitList_190513cA';
### RVCFITS
rvcBase = 'rvcFits'; # direc flag & '.npy' are added

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

fitName_fl = str(fitBase + fitSuf_fl + lossSuf);
fitName_wg = str(fitBase + fitSuf_wg + lossSuf);

# set the save directory to save_loc, then create the save directory if needed
if diffPlot == 1:
  compDir  = str(fitBase + '_comp' + lossSuf + '/diff');
else:
  compDir  = str(fitBase + '_comp' + lossSuf);
if intpMod == 1:
  compDir = str(compDir + '/intp');
subDir   = compDir.replace('fitList', 'fits').replace('.npy', '');
save_loc = str(save_loc + subDir + '/');
if not os.path.exists(save_loc):
  os.makedirs(save_loc);

rpt_fit = 1; # i.e. take the multi-start result
if rpt_fit:
  is_rpt = '_rpt';
else:
  is_rpt = '';

conDig = 3; # round contrast to the 3rd digit

dataList = np.load(str(data_loc + expName), encoding='latin1').item();
fitList_fl = hf.np_smart_load(data_loc + fitName_fl);
fitList_wg = hf.np_smart_load(data_loc + fitName_wg);

cellName = dataList['unitName'][cellNum-1];
try:
  cellType = dataList['unitType'][cellNum-1];
except: 
  # TODO: note, this is dangerous; thus far, only V1 cells don't have 'unitType' field in dataList, so we can safely do this
  cellType = 'V1'; 

expData  = np.load(str(data_loc + cellName + '_sfm.npy'), encoding='latin1').item();
expInd   = hf.get_exp_ind(data_loc, cellName)[0];

# #### Load model fits

modFit_fl = fitList_fl[cellNum-1]['params']; # 
modFit_wg = fitList_wg[cellNum-1]['params']; # 
modFits = [modFit_fl, modFit_wg];
normTypes = [1, 2]; # flat, then weighted

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

modResps = [mod_resp.SFMGiveBof(fit, expData, normType=norm, lossType=lossType, expInd=expInd) for fit, norm in zip(modFits, normTypes)];
modResps = [x[1] for x in modResps]; # 1st return output (x[0]) is NLL (don't care about that here)
gs_mean = modFit_wg[8]; 
gs_std = modFit_wg[9];
# now organize the responses
orgs = [hf.organize_resp(mr, expData, expInd) for mr in modResps];
oriModResps = [org[0] for org in orgs]; # only non-empty if expInd = 1
conModResps = [org[1] for org in orgs]; # only non-empty if expInd = 1
sfmixModResps = [org[2] for org in orgs];
allSfMixs = [org[3] for org in orgs];

modLows = [np.nanmin(resp, axis=3) for resp in allSfMixs];
modHighs = [np.nanmax(resp, axis=3) for resp in allSfMixs];
modAvgs = [np.nanmean(resp, axis=3) for resp in allSfMixs];
modSponRates = [fit[6] for fit in modFits];

# more tabulation - stim vals, organize measured responses
_, stimVals, val_con_by_disp, validByStimVal, _ = hf.tabulate_responses(expData, expInd);
if rvcAdj == 1:
  rvcFlag = '_f1';
  rvcFits = hf.get_rvc_fits(data_loc, expInd, cellNum, rvcName=rvcBase);
else:
  rvcFlag = '';
  rvcFits = hf.get_rvc_fits(data_loc, expInd, cellNum, rvcName='None');
spikes  = hf.get_spikes(expData['sfm']['exp']['trial'], rvcFits=rvcFits, expInd=expInd);
_, _, respOrg, respAll    = hf.organize_resp(spikes, expData, expInd);

respMean = respOrg;
respStd = np.nanstd(respAll, -1); # take std of all responses for a given condition
# compute SEM, too
findNaN = np.isnan(respAll);
nonNaN  = np.sum(findNaN == False, axis=-1);
respSem = np.nanstd(respAll, -1) / np.sqrt(nonNaN);
# pick which measure of response variance
if respVar == 1:
  respVar = respSem;
else:
  respVar = respStd;

if diffPlot: # otherwise, nothing to do
  ### Now, recenter data relative to flat normalization model
  allAvgs = [respMean, modAvgs[1], modAvgs[0]]; # why? weighted is 1, flat is 0
  respsRecenter = [x - allAvgs[2] for x in allAvgs]; # recentered

  respMean = respsRecenter[0];
  modAvgs  = [respsRecenter[2], respsRecenter[1]];

blankMean, blankStd, _ = hf.blankResp(expData, expInd); 

all_disps = stimVals[0];
all_cons = stimVals[1];
all_sfs = stimVals[2];

nCons = len(all_cons);
nSfs = len(all_sfs);
nDisps = len(all_disps);

# ### Plots

# set up model plot info
# i.e. flat model is red, weighted model is green
modColors = ['r', 'g']
modLabels = ['flat', 'wght']

# #### Plots by dispersion

fDisp = []; dispAx = [];

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);    

for d in range(nDisps):

    v_cons = val_con_by_disp[d];
    n_v_cons = len(v_cons);

    fCurr, dispCurr = plt.subplots(n_v_cons, 2, figsize=(nDisps*8, n_v_cons*12), sharey=False);
    fDisp.append(fCurr)
    dispAx.append(dispCurr);

    minResp = np.min(np.min(respMean[d, ~np.isnan(respMean[d, :, :])]));
    maxResp = np.max(np.max(respMean[d, ~np.isnan(respMean[d, :, :])]));

    for c in reversed(range(n_v_cons)):
        c_plt_ind = len(v_cons) - c - 1;
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);        

        # make things nice
        for i in range(2):

          dispAx[d][c_plt_ind, i].set_xlim((min(all_sfs), max(all_sfs)));

          dispAx[d][c_plt_ind, i].set_xscale('log');
          dispAx[d][c_plt_ind, i].set_xlabel('sf (c/deg)'); 
          dispAx[d][c_plt_ind, i].set_title('D%02d: contrast: %.3f' % (d, all_cons[v_cons[c]]));

        # Set ticks out, remove top/right axis, put ticks only on bottom/left
          dispAx[d][c_plt_ind, i].tick_params(labelsize=15, width=majorWid, length=majorLen, direction='out');
          dispAx[d][c_plt_ind, i].tick_params(which='minor', direction='out'); # minor ticks, too...	
          sns.despine(ax=dispAx[d][c_plt_ind, i], offset=10, trim=False); 
   
        # plot data
        dispAx[d][c_plt_ind, 0].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]], 
                                      respVar[d, v_sfs, v_cons[c]], fmt='o', color='k', clip_on=False);

        # plot model fits
        if intpMod == 1:
          plt_sfs = np.geomspace(all_sfs[v_sfs][0], all_sfs[v_sfs][-1], sfSteps);
          interpModBoth = []; # well, flat is first, so we will subtract that off...
          if d == 0:
            nRptsCurr = nRptsSingle;
          else:
            nRptsCurr = nRpts;
          for pm, typ in zip(modFits, normTypes):
            simWrap = lambda x: mod_resp.SFMsimulateNew(pm, expData, d, v_cons[c], x, normType=typ, expInd=expInd, nRepeats=nRptsCurr)[0];
            interpMod = [np.mean(simWrap(np.array([sfCurr]))) for sfCurr in plt_sfs];
            interpModBoth.append(np.array(interpMod));
          # TODO plot, but recenter if diffPlot == 1...
          if diffPlot == 1:
            relTo = interpModBoth[0];
          else:
            relTo = np.zeros_like(interpModBoth[0]);
          for rsp, cc, s in zip(interpModBoth, modColors, modLabels):
            dispAx[d][c_plt_ind, 0].plot(plt_sfs, rsp-relTo, color=cc, label=s);
        else: # plot model evaluated only at data point
          [dispAx[d][c_plt_ind, 0].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], color=cc, alpha=0.7, clip_on=False, label=s) for modAvg, cc, s in zip(modAvgs, modColors, modLabels)];
        '''
        sponRate = dispAx[d][c_plt_ind, 0].axhline(blankMean, color='b', linestyle='dashed', label='data spon. rate');
        [dispAx[d][c_plt_ind, 0].axhline(sponRate, color=cc, linestyle='dashed') for sponRate,cc in zip(modSponRates, modColors)];
        '''

        if diffPlot == 1:
          dispAx[d][c_plt_ind, 0].set_ylim((-1.5*np.abs(minResp), 1.5*maxResp));
        else:
          dispAx[d][c_plt_ind, 0].set_ylim((0, 1.5*maxResp));
        dispAx[d][c_plt_ind, 0].set_ylabel('resp (sps)');
        dispAx[d][c_plt_ind, 1].set_ylabel('ratio (pred:measure)');
        dispAx[d][c_plt_ind, 1].set_ylim((1e-1, 1e3));
        dispAx[d][c_plt_ind, 1].set_yscale('log');
        dispAx[d][c_plt_ind, 1].legend();

    fCurr.suptitle('%s #%d, loss %.2f|%.2f' % (cellType, cellNum, fitList_fl[cellNum-1]['NLL'], fitList_wg[cellNum-1]['NLL']));

saveName = "/cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'byDisp%s/' % rvcFlag));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close();

# #### All SF tuning on one graph, split by dispersion

if diffPlot != 1:
  fDisp = []; dispAx = [];

  sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);    

  for d in range(nDisps):

      v_cons = val_con_by_disp[d];
      n_v_cons = len(v_cons);

      fCurr, dispCurr = plt.subplots(1, 3, figsize=(35, 30)); # left side for flat; middle for data; right side for weighted modelb
      fDisp.append(fCurr)
      dispAx.append(dispCurr);

      fCurr.suptitle('%s #%d' % (cellType, cellNum));

      resps_curr = [modAvgs[0], respMean, modAvgs[1]];
      labels     = [modLabels[0], 'data', modLabels[1]];

      for i in range(3):

        # Set ticks out, remove top/right axis, put ticks only on bottom/left
        dispAx[d][i].tick_params(labelsize=15, width=majorWid, length=majorLen, direction='out');
        dispAx[d][i].tick_params(which='minor', direction='out'); # minor ticks, too...
        sns.despine(ax=dispAx[d][i], offset=10, trim=False); 

        curr_resps = resps_curr[i];
        maxResp = np.max(np.max(np.max(curr_resps[~np.isnan(curr_resps)])));  

        lines = [];
        for c in reversed(range(n_v_cons)):
            v_sfs = ~np.isnan(curr_resps[d, :, v_cons[c]]);        

            # plot data
            col = [c/float(n_v_cons), c/float(n_v_cons), c/float(n_v_cons)];
            plot_resp = curr_resps[d, v_sfs, v_cons[c]];

            curr_line, = dispAx[d][i].plot(all_sfs[v_sfs][plot_resp>1e-1], plot_resp[plot_resp>1e-1], '-o', clip_on=False, \
                                           color=col, label=str(np.round(all_cons[v_cons[c]], 2)));
            lines.append(curr_line);

        #dispAx[d][i].set_aspect('equal', 'box'); 
        dispAx[d][i].set_xlim((0.5*min(all_sfs), 1.2*max(all_sfs)));
        dispAx[d][i].set_ylim((5e-2, 1.5*maxResp));

        dispAx[d][i].set_xscale('log');
        #dispAx[d][i].set_yscale('log');
        dispAx[d][i].set_xlabel('sf (c/deg)'); 

        dispAx[d][i].set_ylabel('resp above baseline (sps)');
        dispAx[d][i].set_title('D%02d - sf tuning %s' % (d, labels[i]));
        dispAx[d][i].legend(); 

  saveName = "/allCons_cell_%03d.pdf" % (cellNum)
  full_save = os.path.dirname(str(save_loc + 'byDisp%s/' % rvcFlag));
  if not os.path.exists(full_save):
    os.makedirs(full_save);
  pdfSv = pltSave.PdfPages(full_save + saveName);
  for f in fDisp:
      pdfSv.savefig(f)
      plt.close(f)
  pdfSv.close()

# #### Plot just sfMix contrasts

mixCons = hf.get_exp_params(expInd).nCons;
minResp = np.min(np.min(np.min(respMean[~np.isnan(respMean)])));
maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));

f, sfMixAx = plt.subplots(mixCons, nDisps, figsize=(20, 1.5*15));

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);

for d in range(nDisps):
    v_cons = np.array(val_con_by_disp[d]);
    n_v_cons = len(v_cons);
    v_cons = v_cons[np.arange(np.maximum(0, n_v_cons -mixCons), n_v_cons)]; # max(1, .) for when there are fewer contrasts than 4
    n_v_cons = len(v_cons);
    
    for c in reversed(range(n_v_cons)):

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        sfMixAx[c_plt_ind, d].tick_params(labelsize=15, width=majorWid, length=majorLen, direction='out');
        sfMixAx[c_plt_ind, d].tick_params(which='minor', direction='out'); # minor ticks, too...
        sns.despine(ax=sfMixAx[c_plt_ind, d], offset=10, trim=False);

        c_plt_ind = n_v_cons - c - 1;
        sfMixAx[c_plt_ind, d].set_title('con:' + str(np.round(all_cons[v_cons[c]], 2)))
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);
        
        # plot data
        sfMixAx[c_plt_ind, d].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]], 
                                       respVar[d, v_sfs, v_cons[c]], fmt='o', color='k', clip_on=False);

	# plot model fits
        if intpMod == 1:
          plt_sfs = np.geomspace(all_sfs[v_sfs][0], all_sfs[v_sfs][-1], sfSteps);
          interpModBoth = []; # well, flat is first, so we will subtract that off...
          if d == 0:
            nRptsCurr = nRptsSingle;
          else:
            nRptsCurr = nRpts;
          for pm, typ in zip(modFits, normTypes):
            simWrap = lambda x: mod_resp.SFMsimulateNew(pm, expData, d, v_cons[c], x, normType=typ, expInd=expInd, nRepeats=nRptsCurr)[0];
            interpMod = [np.mean(simWrap(np.array([sfCurr]))) for sfCurr in plt_sfs];
            interpModBoth.append(np.array(interpMod));
          # TODO plot, but recenter if diffPlot == 1...
          if diffPlot == 1:
            relTo = interpModBoth[0];
          else:
            relTo = np.zeros_like(interpModBoth[0]);
          for rsp, cc, s in zip(interpModBoth, modColors, modLabels):
            sfMixAx[c_plt_ind, d].plot(plt_sfs, rsp-relTo, color=cc, label=s, clip_on=False);
        else: # plot model evaluated only at data point
          [sfMixAx[c_plt_ind, d].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], color=cc, alpha=0.7, clip_on=False, label=s) for modAvg, cc, s in zip(modAvgs, modColors, modLabels)];

        sfMixAx[c_plt_ind, d].set_xlim((np.min(all_sfs), np.max(all_sfs)));
        if diffPlot == 1:
          sfMixAx[c_plt_ind, d].set_ylim((-1.5*np.abs(minResp), 1.5*maxResp));
        else:
          sfMixAx[c_plt_ind, d].set_ylim((0, 1.5*maxResp));
        sfMixAx[c_plt_ind, d].set_xscale('log');
        sfMixAx[c_plt_ind, d].set_xlabel('sf (c/deg)');
        sfMixAx[c_plt_ind, d].set_ylabel('resp (sps)');


f.legend();
f.suptitle('%s #%d (%s), loss %.2f|%.2f' % (cellType, cellNum, cellName, fitList_fl[cellNum-1]['NLL'], fitList_wg[cellNum-1]['NLL']));
	        
#########
# Plot secondary things - filter, normalization, nonlinearity, etc
#########

fDetails = plt.figure();
fDetails.set_size_inches(w=25,h=15)

detailSize = (3, 5);
if ~np.any([i is None for i in oriModResps]): # then we're using an experiment with ori curves
  curr_ax = plt.subplot2grid(detailSize, (0, 2));
  # Remove top/right axis, put ticks only on bottom/left
  sns.despine(ax=curr_ax, offset = 5);

  # plot ori tuning
  [plt.plot(expData['sfm']['exp']['ori'], oriResp, '%so' % c, clip_on=False, label=s) for oriResp, c, s in zip(oriModResps, modColors, modLabels)]; # Model responses
  expPlt = plt.plot(expData['sfm']['exp']['ori'], expData['sfm']['exp']['oriRateMean'], 'o-', clip_on=False); # Exp responses
  plt.xlabel('Ori (deg)', fontsize=12);
  plt.ylabel('Response (ips)', fontsize=12);
if ~np.any([i is None for i in conModResps]): # then we're using an experiment with RVC curves
  # CRF - with values from TF simulation and the broken down (i.e. numerator, denominator separately) values from resimulated conditions
  curr_ax = plt.subplot2grid(detailSize, (0, 1)); # default size is 1x1
  consUse = expData['sfm']['exp']['con'];
  plt.semilogx(consUse, expData['sfm']['exp']['conRateMean'], 'o-', clip_on=False); # Measured responses
  [plt.plot(consUse, conResp, '%so' % c, clip_on=False, label=s) for conResp, c, s in zip(conModResps, modColors, modLabels)]; # Model responses
  plt.xlabel('Con (%)', fontsize=20);

# poisson test - mean/var for each condition (i.e. sfXdispXcon)
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
plt.xlabel('Mean (sps)');
plt.ylabel('Variance (sps^2)');
plt.title('Super-poisson?');
plt.axis('equal');

#  RVC - pick center SF
curr_ax = plt.subplot2grid(detailSize, (0, 1)); # default size is 1x1
sns.despine(ax=curr_ax, offset = 5);
disp_rvc = 0;
val_cons = np.array(val_con_by_disp[disp_rvc]);
v_sfs = ~np.isnan(respMean[disp_rvc, :, val_cons[0]]); # remember, for single gratings, all cons have same #/index of sfs
sfToUse = np.int(np.floor(len(v_sfs)/2));
plt.semilogx(all_cons[val_cons], respMean[disp_rvc, sfToUse, val_cons], 'o', clip_on=False); # Measured responses
[plt.plot(all_cons[val_cons], modAvg[disp_rvc, sfToUse, val_cons], '%so-' % c, clip_on=False, label=s) for modAvg, c, s in zip(modAvgs, modColors, modLabels)]; # Model responses
plt.xlabel('Con (%)', fontsize=20);
# Remove top/right axis, put ticks only on bottom/left

# plot model details - exc/suppressive components
omega = np.logspace(-2, 2, 1000);
sfExc = [];
for i in modFits:
  prefSf = i[0];
  dOrder = i[1];
  sfRel = omega/prefSf;
  s     = np.power(omega, dOrder) * np.exp(-dOrder/2 * np.square(sfRel));
  sMax  = np.power(prefSf, dOrder) * np.exp(-dOrder/2);
  sfExcCurr = s/sMax;
  sfExc.append(sfExcCurr);

inhSfTuning = hf.getSuppressiveSFtuning();

# Compute weights for suppressive signals
nInhChan = expData['sfm']['mod']['normalization']['pref']['sf'];
nTrials =  inhSfTuning.shape[0];
inhWeight = hf.genNormWeights(expData, nInhChan, gs_mean, gs_std, nTrials, expInd);
inhWeight = inhWeight[:, :, 0]; # genNormWeights gives us weights as nTr x nFilters x nFrames - we have only one "frame" here, and all are the same
# first, tuned norm:
sfNormTune = np.sum(-.5*(inhWeight*np.square(inhSfTuning)), 1);
sfNormTune = sfNormTune/np.amax(np.abs(sfNormTune));
# then, untuned norm:
inhAsym = 0;
inhWeight = [];
for iP in range(len(nInhChan)):
    inhWeight = np.append(inhWeight, 1 + inhAsym * (np.log(expData['sfm']['mod']['normalization']['pref']['sf'][iP]) - np.mean(np.log(expData['sfm']['mod']['normalization']['pref']['sf'][iP]))));
sfNorm = np.sum(-.5*(inhWeight*np.square(inhSfTuning)), 1);
sfNorm = sfNorm/np.amax(np.abs(sfNorm));
sfNorms = [sfNorm, sfNormTune];

# just setting up lines
curr_ax = plt.subplot2grid(detailSize, (1, 1));
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=curr_ax, offset=5);
plt.semilogx([omega[0], omega[-1]], [0, 0], 'k--')
plt.semilogx([.01, .01], [-1.5, 1], 'k--')
plt.semilogx([.1, .1], [-1.5, 1], 'k--')
plt.semilogx([1, 1], [-1.5, 1], 'k--')
plt.semilogx([10, 10], [-1.5, 1], 'k--')
plt.semilogx([100, 100], [-1.5, 1], 'k--')
# now the real stuff
[plt.semilogx(omega, exc, '%s' % cc, label=s) for exc, cc, s in zip(sfExc, modColors, modLabels)]
[plt.semilogx(omega, -norm, '%s--' % cc, label=s) for norm, cc, s in zip(sfNorms, modColors, modLabels)]
plt.xlim([omega[0], omega[-1]]);
plt.ylim([-0.1, 1.1]);
plt.xlabel('spatial frequency (c/deg)', fontsize=12);
plt.ylabel('Normalized response (a.u.)', fontsize=12);

# SIMPLE normalization
curr_ax = plt.subplot2grid(detailSize, (2, 1));
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
wt_weights = np.sqrt(hf.genNormWeightsSimple(omega, gs_mean, gs_std));
sfNormTuneSim = wt_weights/np.amax(np.abs(wt_weights));
sfNormsSimple = [sfNormSim, sfNormTuneSim]
[plt.semilogx(omega, exc, '%s' % cc, label=s) for exc, cc, s in zip(sfExc, modColors, modLabels)]
[plt.semilogx(omega, norm, '%s--' % cc, label=s) for norm, cc, s in zip(sfNormsSimple, modColors, modLabels)]
plt.xlim([omega[0], omega[-1]]);
plt.ylim([-0.1, 1.1]);
plt.xlabel('spatial frequency (c/deg)', fontsize=12);
plt.ylabel('Normalized response (a.u.)', fontsize=12);

# last but not least...and not last... response nonlinearity
modExps = [x[3] for x in modFits];
curr_ax = plt.subplot2grid(detailSize, (1, 2));
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=curr_ax, offset=5);
plt.plot([-1, 1], [0, 0], 'k--')
plt.plot([0, 0], [-.1, 1], 'k--')
[plt.plot(np.linspace(-1,1,100), np.power(np.maximum(0, np.linspace(-1,1,100)), modExp), '%s-' % cc, label=s, linewidth=2) for modExp,cc,s in zip(modExps, modColors, modLabels)]
plt.plot(np.linspace(-1,1,100), np.maximum(0, np.linspace(-1,1,100)), 'k--', linewidth=1)
plt.xlim([-1, 1]);
plt.ylim([-.1, 1]);
plt.text(0.5, 1.1, 'respExp: %.2f, %.2f' % (modExps[0], modExps[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');

# print, in text, model parameters:
curr_ax = plt.subplot2grid(detailSize, (0, 4));
plt.text(0.5, 0.5, 'prefSf: %.3f, %.3f' % (modFits[0][0], modFits[1][0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.4, 'derivative order: %.3f, %.3f' % (modFits[0][1], modFits[1][1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.3, 'response scalar: %.3f, %.3f' % (modFits[0][4], modFits[1][4]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.2, 'sigma: %.3f, %.3f | %.3f, %.3f' % (np.power(10, modFits[0][2]), np.power(10, modFits[1][2]), modFits[0][2], modFits[1][2]), fontsize=12, horizontalalignment='center', verticalalignment='center');

### now save all figures (sfMix contrasts, details, normalization stuff)
allFigs = [f, fDetails];
saveName = "/cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'sfMixOnly%s/' % rvcFlag));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for fig in range(len(allFigs)):
    pdfSv.savefig(allFigs[fig])
    plt.close(allFigs[fig])
pdfSv.close()

#### Response versus contrast

# #### Plot contrast response functions with (full) model predictions

rvcAx = []; fRVC = [];

if intpMod == 0 or (intpMod == 1 and conSteps > 0): # i.e. we've chosen to do this (we have this flag since sometimes we do not need to plot RVCs in interpolated way)

  for d in range(nDisps):
      # which sfs have at least one contrast presentation? within a dispersion, all cons have the same # of sfs
      v_sf_inds = hf.get_valid_sfs(expData, d, val_con_by_disp[d][0], expInd, stimVals, validByStimVal);
      n_v_sfs = len(v_sf_inds);
      n_rows = int(np.ceil(n_v_sfs/np.floor(np.sqrt(n_v_sfs)))); # make this close to a rectangle/square in arrangement (cycling through sfs)
      n_cols = int(np.ceil(n_v_sfs/n_rows));
      fCurr, rvcCurr = plt.subplots(n_rows, n_cols, figsize=(n_cols*10, n_rows*15), sharex = True, sharey = True);
      fRVC.append(fCurr);
      rvcAx.append(rvcCurr);

      fCurr.suptitle('%s #%d' % (cellType, cellNum-1));

      #print('%d rows, %d cols\n' % (n_rows, n_cols));

      for sf in range(n_v_sfs):

          row_ind = int(sf/n_cols);
          col_ind = np.mod(sf, n_cols);
          sf_ind = v_sf_inds[sf];
          plt_x = d; 
          if n_cols > 1:
            plt_y = (row_ind, col_ind);
          else: # pyplot makes it (n_rows, ) if n_cols == 1
            plt_y = (row_ind, );
          #print(plt_y);

          # Set ticks out, remove top/right axis, put ticks only on bottom/left
          sns.despine(ax = rvcAx[plt_x][plt_y], offset = 10, trim=False);
          rvcAx[plt_x][plt_y].tick_params(labelsize=25, width=majorWid, length=majorLen, direction='out');
          rvcAx[plt_x][plt_y].tick_params(which='minor', direction='out'); # minor ticks, too...

          v_cons = val_con_by_disp[d];
          n_cons = len(v_cons);
          plot_cons = np.linspace(np.min(all_cons[v_cons]), np.max(all_cons[v_cons]), 100); # 100 steps for plotting...

          # organize (measured) responses
          resp_curr = np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, ));
          var_curr  = np.reshape([respVar[d, sf_ind, v_cons]], (n_cons, ));
          if diffPlot == 1: # don't set a baseline (i.e. response can be negative!)
            respPlt = rvcAx[plt_x][plt_y].errorbar(all_cons[v_cons], resp_curr, var_curr, fmt='o', color='k', clip_on=False, label='data');
          else:
            respPlt = rvcAx[plt_x][plt_y].errorbar(all_cons[v_cons], np.maximum(resp_curr, 0.1), var_curr, fmt='o', color='k', clip_on=False, label='data');

          # RVC with full model fits (i.e. flat and weighted)
          if intpMod == 1:
            plt_cons = np.geomspace(all_cons[v_cons][0], all_cons[v_cons][-1], conSteps);
            interpModBoth = []; # flat comes first, and we'll subtract off if diffPlot
            if d == 0:
              nRptsCurr = nRptsSingle;
            else:
              nRptsCurr = nRpts;
            for pm, typ in zip(modFits, normTypes):
              simWrap = lambda x: mod_resp.SFMsimulateNew(pm, expData, d, x, sf_ind, normType=typ, expInd=expInd, nRepeats=nRptsCurr)[0];
              interpMod = np.array([np.mean(simWrap(np.array([conCurr]))) for conCurr in plt_cons]);
              interpModBoth.append(np.array(interpMod));
            if diffPlot == 1:
              relTo = interpModBoth[0];
            else:
              relTo = np.zeros_like(interpModBoth[0]);
            for rsp, cc, s in zip(interpModBoth, modColors, modLabels):
              rvcAx[plt_x][plt_y].plot(plt_cons, rsp-relTo, color=cc, label=s, clip_on=False);
          else:
            [rvcAx[plt_x][plt_y].plot(all_cons[v_cons], np.maximum(modAvg[d, sf_ind, v_cons], 0.1), color=cc, \
              alpha=0.7, clip_on=False, label=s) for modAvg,cc,s in zip(modAvgs, modColors, modLabels)];

          # summary plots
          '''
          curr_rvc = rvcAx[0][d, 0].plot(all_cons[v_cons], resps_curr, '-', clip_on=False);
          rvc_plots.append(curr_rvc[0]);

          stdPts = np.hstack((0, np.reshape([respVar[d, sf_ind, v_cons]], (n_cons, ))));
          expPts = rvcAx[d+1][row_ind, col_ind].errorbar(np.hstack((0, all_cons[v_cons])), resps_w_blank, stdPts, fmt='o', clip_on=Fals
  e);

          sepPlt = rvcAx[d+1][row_ind, col_ind].plot(plot_cons, helper_fcns.naka_rushton(plot_cons, curr_fit_sep), linestyle='dashed');
          allPlt = rvcAx[d+1][row_ind, col_ind].plot(plot_cons, helper_fcns.naka_rushton(plot_cons, curr_fit_all), linestyle='dashed');
          # accompanying legend/comments
          rvcAx[d+1][row_ind, col_ind].legend((expPts[0], sepPlt[0], allPlt[0]), ('data', 'model fits'), fontsize='large', loc='center left')
          '''

          rvcAx[plt_x][plt_y].set_xscale('log', basex=10); # was previously symlog, linthreshx=0.01
          if col_ind == 0:
            rvcAx[plt_x][plt_y].set_xlabel('contrast', fontsize='medium');
            rvcAx[plt_x][plt_y].set_ylabel('response (spikes/s)', fontsize='medium');
            rvcAx[plt_x][plt_y].legend();

          rvcAx[plt_x][plt_y].set_title('D%d: sf: %.3f' % (d+1, all_sfs[sf_ind]), fontsize='large');


  saveName = "/cell_%03d.pdf" % (cellNum)
  full_save = os.path.dirname(str(save_loc + 'CRF%s/' % rvcFlag));
  if not os.path.exists(full_save):
    os.makedirs(full_save);
  pdfSv = pltSave.PdfPages(full_save + saveName);
  for f in fRVC:
      pdfSv.savefig(f)
      plt.close(f)
  pdfSv.close()

# #### Plot contrast response functions - all sfs on one axis (per dispersion)

if diffPlot != 1 or intpMod == 0:

  crfAx = []; fCRF = [];

  for d in range(nDisps):

      fCurr, crfCurr = plt.subplots(1, 3, figsize=(35, 30), sharex = False, sharey = True); # left side for flat; middle for data; right side for weighted model
      fCRF.append(fCurr)
      crfAx.append(crfCurr);

      fCurr.suptitle('%s #%d' % (cellType, cellNum));

      resps_curr = [modAvgs[0], respMean, modAvgs[1]];
      labels     = [modLabels[0], 'data', modLabels[1]];

      v_sf_inds = hf.get_valid_sfs(expData, d, val_con_by_disp[d][0], expInd, stimVals, validByStimVal);
      n_v_sfs = len(v_sf_inds);

      for i in range(3):
        curr_resps = resps_curr[i];
        maxResp = np.max(np.max(np.max(curr_resps[~np.isnan(curr_resps)])));

        # Set ticks out, remove top/right axis, put ticks only on bottom/left
        crfAx[d][i].tick_params(labelsize=15, width=majorWid, length=majorLen, direction='out');
        crfAx[d][i].tick_params(which='minor', direction='out'); # minor ticks, too...
        sns.despine(ax = crfAx[d][i], offset=10, trim=False);

        lines_log = [];
        for sf in range(n_v_sfs):
            sf_ind = v_sf_inds[sf];
            v_cons = ~np.isnan(curr_resps[d, sf_ind, :]);
            n_cons = sum(v_cons);

            col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];
            plot_resp = curr_resps[d, sf_ind, v_cons];

            line_curr, = crfAx[d][i].plot(all_cons[v_cons][plot_resp>1e-1], plot_resp[plot_resp>1e-1], '-o', color=col, \
                                          clip_on=False, label = str(np.round(all_sfs[sf_ind], 2)));
            lines_log.append(line_curr);

        crfAx[d][i].set_xlim([-0.1, 1]);
        crfAx[d][i].set_ylim([-0.1*maxResp, 1.1*maxResp]);
        '''
        crfAx[d][i].set_xscale('log');
        crfAx[d][i].set_yscale('log');
        crfAx[d][i].set_xlim([1e-2, 1]);
        crfAx[d][i].set_ylim([1e-2, 1.5*maxResp]);
        '''
        crfAx[d][i].set_xlabel('contrast');

        crfAx[d][i].set_ylabel('resp above baseline (sps)');
        crfAx[d][i].set_title('D%d: sf:all - log resp %s' % (d, labels[i]));
        crfAx[d][i].legend();

  saveName = "/allSfs_cell_%03d.pdf" % (cellNum)
  if not os.path.exists(full_save):
    os.makedirs(full_save);
  full_save = os.path.dirname(str(save_loc + 'CRF%s/' % rvcFlag));
  pdfSv = pltSave.PdfPages(full_save + saveName);
  for f in fCRF:
      pdfSv.savefig(f)
      plt.close(f)
  pdfSv.close()
