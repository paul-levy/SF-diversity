# coding: utf-8

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns
sns.set(style='ticks')
from scipy.stats import poisson, nbinom
from scipy.stats.mstats import gmean

import pdb

# import the "main" helper_fcns and model_responses (i.e. the common set)
import sys
sys.path.insert(0, '../'); # now hf, mod_resp will be from the parent directory
import helper_fcns as hf
import model_responses as mod_resp

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_cluster.mplstyle');
from matplotlib import rcParams
rcParams['font.size'] = 20;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 2.5;
rcParams['axes.linewidth'] = 1.5;
rcParams['lines.markersize'] = 5;
rcParams['font.style'] = 'oblique';

cellNum  = int(sys.argv[1]);
lossType = int(sys.argv[2]);
expInd   = 2; # (V1) altExp is #2

# at CNS
data_loc = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/structures/';
save_loc = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/figures/';
# personal mac
#data_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/altExp/structures/';
#save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/altExp/figures/cell4_sandbox/';
# prince cluster
#data_loc = '/home/pl1465/SF_diversity/altExp/structures/';
#save_loc = '/home/pl1465/SF_diversity/altExp/figures/';

expName = 'dataList.npy'
fitBase = 'fitList_190114c';

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
compDir  = str(fitBase + '_comp' + lossSuf);
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

expData = np.load(str(data_loc + dataList['unitName'][cellNum-1] + '_sfm.npy'), encoding='latin1').item();

# #### Load model fits

modFit_fl = fitList_fl[cellNum-1]['params']; # 
modFit_wg = fitList_wg[cellNum-1]['params']; # 
modFits = [modFit_fl, modFit_wg];
normTypes = [1, 2]; # flat, then weighted

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

modResps = [mod_resp.SFMGiveBof(fit, expData, normType=norm, lossType=lossType, expInd=expInd) for fit, norm in zip(modFits, normTypes)];
modResps = [x[1] for x in modResps]; # 1st return output is NLL (don't care about that here)
gs_mean = modFit_wg[8]; 
gs_std = modFit_wg[9];
# now organize the responses
orgs = [hf.organize_resp(mr, expData, expInd) for mr in modResps];
#orgs = [hf.organize_modResp(mr, expData) for mr in modResps];
sfmixModResps = [org[2] for org in orgs];
allSfMixs = [org[3] for org in orgs];
# now organize the measured responses in the same way
_, _, sfmixExpResp, allSfMixExp = hf.organize_resp(expData['sfm']['exp']['trial']['spikeCount'], expData, expInd);

modLows = [np.nanmin(resp, axis=3) for resp in allSfMixs];
modHighs = [np.nanmax(resp, axis=3) for resp in allSfMixs];
modAvgs = [np.nanmean(resp, axis=3) for resp in allSfMixs];
modSponRates = [fit[6] for fit in modFits];

# more tabulation
resp, stimVals, val_con_by_disp, _, _ = hf.tabulate_responses(expData, expInd, modResps[0]);

respMean = resp[0];
respStd = resp[1];

blankMean, blankStd, _ = hf.blankResp(expData); 

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
    
    fCurr, dispCurr = plt.subplots(n_v_cons, 2, figsize=(25, n_v_cons*8), sharey=False);
    fDisp.append(fCurr)
    dispAx.append(dispCurr);
    
    maxResp = np.max(np.max(respMean[d, ~np.isnan(respMean[d, :, :])]));
    
    for c in reversed(range(n_v_cons)):
        c_plt_ind = len(v_cons) - c - 1;
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);        

        # plot data
        dispAx[d][c_plt_ind, 0].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]], 
                                      respStd[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);

	# plot model fits
        # plot model average for both models (flat + weighted)
        [dispAx[d][c_plt_ind, 0].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], color=cc, alpha=0.7, clip_on=False, label=s) for modAvg, cc, s in zip(modAvgs, modColors, modLabels)];
        sponRate = dispAx[d][c_plt_ind, 0].axhline(blankMean, color='b', linestyle='dashed', label='data spon. rate');
        [dispAx[d][c_plt_ind, 0].axhline(sponRate, color=cc, linestyle='dashed') for sponRate,cc in zip(modSponRates, modColors)];

        for i in range(2):

          dispAx[d][c_plt_ind, i].set_xlim((min(all_sfs), max(all_sfs)));
        
          dispAx[d][c_plt_ind, i].set_xscale('log');
          dispAx[d][c_plt_ind, i].set_xlabel('sf (c/deg)'); 
          dispAx[d][c_plt_ind, i].set_title('D%02d: contrast: %.3f' % (d, all_cons[v_cons[c]]));

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
          dispAx[d][c_plt_ind, i].tick_params(labelsize=15, width=1, length=8, direction='out');
          dispAx[d][c_plt_ind, i].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...	
          sns.despine(ax=dispAx[d][c_plt_ind, i], offset=10, trim=False); 

        dispAx[d][c_plt_ind, 0].set_ylim((0, 1.5*maxResp));
        dispAx[d][c_plt_ind, 0].set_ylabel('resp (sps)');
        dispAx[d][c_plt_ind, 1].set_ylabel('ratio (pred:measure)');
        dispAx[d][c_plt_ind, 1].set_ylim((1e-1, 1e3));
        dispAx[d][c_plt_ind, 1].set_yscale('log');
        dispAx[d][c_plt_ind, 1].legend();

    fCurr.suptitle('cell #%d, loss %.2f|%.2f' % (cellNum, fitList_fl[cellNum-1]['NLL'], fitList_wg[cellNum-1]['NLL']));

saveName = "/cell_%02d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'byDisp/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close();

# #### All SF tuning on one graph, split by dispersion
'''
fDisp = []; dispAx = [];

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);    

for d in range(nDisps):
    
    v_cons = val_con_by_disp[d];
    n_v_cons = len(v_cons);
    
    fCurr, dispCurr = plt.subplots(1, 2, figsize=(20, 20)); # left side for data, right side for model predictions
    fDisp.append(fCurr)
    dispAx.append(dispCurr);

    for i in range(2):
    
      if i == 0:
        curr_resps = respMean;
        curr_mean = blankMean;
        maxResp = np.max(np.max(curr_resps[d, ~np.isnan(curr_resps[d, :, :])]));
      else:
        curr_resps = modAvgs;
        curr_mean = modSponRates;
        maxResp = np.max([np.max(np.max(cr[d, ~np.isnan(cr[d, :, :])])) for cr in curr_resps]);

      lines = [];
      for c in reversed(range(n_v_cons)):
          v_sfs = ~np.isnan(curr_resps[d, :, v_cons[c]]);        

          # plot data
          col = [c/float(n_v_cons), c/float(n_v_cons), c/float(n_v_cons)];
          respAbBaseline = curr_resps[d, v_sfs, v_cons[c]] - curr_mean;
          curr_line, = dispAx[d][i].plot(all_sfs[v_sfs][respAbBaseline>1e-1], respAbBaseline[respAbBaseline>1e-1], '-o', clip_on=False, color=col);
          lines.append(curr_line);

      dispAx[d][i].set_aspect('equal', 'box'); 
      dispAx[d][i].set_xlim((0.5*min(all_sfs), 1.2*max(all_sfs)));
      dispAx[d][i].set_ylim((5e-2, 1.5*maxResp));

      dispAx[d][i].set_xscale('log');
      dispAx[d][i].set_yscale('log');
      dispAx[d][i].set_xlabel('sf (c/deg)'); 

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      dispAx[d][i].tick_params(labelsize=15, width=2, length=16, direction='out');
      dispAx[d][i].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
      sns.despine(ax=dispAx[d][i], offset=10, trim=False); 

      dispAx[d][i].set_ylabel('resp above baseline (sps)');
      dispAx[d][i].set_title('D%02d - sf tuning' % (d));
      dispAx[d][i].legend(lines, [str(i) for i in reversed(all_cons[v_cons])], loc=0);

saveName = "/allCons_cell_%02d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'byDisp/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()
'''
# #### Plot just sfMix contrasts

# i.e. highest (up to) 4 contrasts for each dispersion

mixCons = 4;
maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));

f, sfMixAx = plt.subplots(mixCons, nDisps, figsize=(20, 15));

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);

for d in range(nDisps):
    v_cons = np.array(val_con_by_disp[d]);
    n_v_cons = len(v_cons);
    v_cons = v_cons[np.arange(np.maximum(0, n_v_cons -mixCons), n_v_cons)]; # max(1, .) for when there are fewer contrasts than 4
    n_v_cons = len(v_cons);
    
    for c in reversed(range(n_v_cons)):
        c_plt_ind = n_v_cons - c - 1;
        sfMixAx[c_plt_ind, d].set_title('con:' + str(np.round(all_cons[v_cons[c]], 2)))
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);
        
        # plot data
        sfMixAx[c_plt_ind, d].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]], 
                                       respStd[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);

	# plot model fits
        [sfMixAx[c_plt_ind, d].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], color=cc, alpha=0.7, clip_on=False, label=s) for modAvg, cc, s in zip(modAvgs, modColors, modLabels)];

        sfMixAx[c_plt_ind, d].set_xlim((np.min(all_sfs), np.max(all_sfs)));
        sfMixAx[c_plt_ind, d].set_ylim((0, 1.5*maxResp));
        sfMixAx[c_plt_ind, d].set_xscale('log');
        sfMixAx[c_plt_ind, d].set_xlabel('sf (c/deg)');
        sfMixAx[c_plt_ind, d].set_ylabel('resp (sps)');

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        sfMixAx[c_plt_ind, d].tick_params(labelsize=15, width=1, length=8, direction='out');
        sfMixAx[c_plt_ind, d].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
        sns.despine(ax=sfMixAx[c_plt_ind, d], offset=10, trim=False);

f.legend();
f.suptitle('cell #%d, loss %.2f|%.2f' % (cellNum, fitList_fl[cellNum-1]['NLL'], fitList_wg[cellNum-1]['NLL']));
	        
#########
# Plot secondary things - filter, normalization, nonlinearity, etc
#########

fDetails = plt.figure();
fDetails.set_size_inches(w=25,h=10)
#fDetails, all_plots = plt.subplots(3,5, figsize=(25,10))

detailSize = (3, 5);

'''
all_plots[0,2].axis('off');
all_plots[0,3].axis('off');
#all_plots[0,4].axis('off');
all_plots[1,3].axis('off');
all_plots[1,4].axis('off');
'''
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
inhWeight = hf.genNormWeights(expData, nInhChan, gs_mean, gs_std, nTrials);
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
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=curr_ax, offset=5);

# last but not least...and not last... response nonlinearity
modExps = [x[3] for x in modFits];
curr_ax = plt.subplot2grid(detailSize, (1, 2));
plt.plot([-1, 1], [0, 0], 'k--')
plt.plot([0, 0], [-.1, 1], 'k--')
[plt.plot(np.linspace(-1,1,100), np.power(np.maximum(0, np.linspace(-1,1,100)), modExp), '%s-' % cc, label=s, linewidth=2) for modExp,cc,s in zip(modExps, modColors, modLabels)]
plt.plot(np.linspace(-1,1,100), np.maximum(0, np.linspace(-1,1,100)), 'k--', linewidth=1)
plt.xlim([-1, 1]);
plt.ylim([-.1, 1]);
plt.text(0.5, 1.1, 'respExp: %.2f, %.2f' % (modExps[0], modExps[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=curr_ax, offset=5);

# print, in text, model parameters:
curr_ax = plt.subplot2grid(detailSize, (0, 4));
plt.text(0.5, 0.5, 'prefSf: %.3f, %.3f' % (modFits[0][0], modFits[1][0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.4, 'derivative order: %.3f, %.3f' % (modFits[0][1], modFits[1][1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.3, 'response scalar: %.3f, %.3f' % (modFits[0][4], modFits[1][4]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.2, 'sigma: %.3f, %.3f | %.3f, %.3f' % (np.power(10, modFits[0][2]), np.power(10, modFits[1][2]), modFits[0][2], modFits[1][2]), fontsize=12, horizontalalignment='center', verticalalignment='center');


# poisson test - mean/var for each condition (i.e. sfXdispXcon)
curr_ax = plt.subplot2grid(detailSize, (0, 0), colspan=2, rowspan=2); # set the current subplot location/size[default is 1x1]
val_conds = ~np.isnan(respMean);
gt0 = np.logical_and(respMean[val_conds]>0, respStd[val_conds]>0);
plt.loglog([0.01, 1000], [0.01, 1000], 'k--');
plt.loglog(respMean[val_conds][gt0], np.square(respStd[val_conds][gt0]), 'o');
# skeleton for plotting modulated poisson prediction
if lossType == 3: # i.e. modPoiss
  mean_vals = np.logspace(-1, 2, 50);
  plt.loglog(mean_vals, mean_vals + varGain*np.square(mean_vals));
plt.xlabel('Mean (sps)');
plt.ylabel('Variance (sps^2)');
plt.title('Super-poisson?');
plt.axis('equal');
sns.despine(ax=curr_ax, offset=5, trim=False);

### now save all figures (sfMix contrasts, details, normalization stuff)
allFigs = [f, fDetails];
saveName = "/cell_%02d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'sfMixOnly/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for fig in range(len(allFigs)):
    pdfSv.savefig(allFigs[fig])
    plt.close(allFigs[fig])
pdfSv.close()

'''
# #### Plot contrast response functions - all sfs on one axis (per dispersion)

crfAx = []; fCRF = [];

for d in range(nDisps):
    
    fCurr, crfCurr = plt.subplots(1, 2, figsize=(20, 25), sharex = False, sharey = False); # left side for data, right side for model predictions
    fCRF.append(fCurr)
    crfAx.append(crfCurr);

    for i in range(2):
      
      if i == 0:
        curr_resps = respMean;
        curr_base = blankMean;
        title_str = 'data';
      else:
        curr_resps = modAvg;
        curr_base = modBlankMean;
        title_str = 'model';
      maxResp = np.max(np.max(np.max(curr_resps[~np.isnan(curr_resps)])));

      # which sfs have at least one contrast presentation?
      v_sfs = np.where(np.sum(~np.isnan(curr_resps[d, :, :]), axis = 1) > 0);
      n_v_sfs = len(v_sfs[0])

      lines = []; lines_log = [];
      for sf in range(n_v_sfs):
          sf_ind = v_sfs[0][sf];
          v_cons = ~np.isnan(curr_resps[d, sf_ind, :]);
          n_cons = sum(v_cons);

          col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];
          plot_resps = np.reshape([curr_resps[d, sf_ind, v_cons]], (n_cons, ));
          respAbBaseline = plot_resps-curr_base;
          line_curr, = crfAx[d][i].plot(all_cons[v_cons][respAbBaseline>1e-1], respAbBaseline[respAbBaseline>1e-1], '-o', color=col, clip_on=False);
          #line_curr, = crfAx[d][i].plot(all_cons[v_cons], np.maximum(1e-1, curr_resps-blankMean), '-o', color=col, clip_on=False);
          lines_log.append(line_curr);

      crfAx[d][i].set_xlim([1e-2, 1]);
      crfAx[d][i].set_ylim([1e-2, 1.5*maxResp]);
      crfAx[d][i].set_aspect('equal', 'box')
      crfAx[d][i].set_xscale('log');
      crfAx[d][i].set_yscale('log');
      crfAx[d][i].set_xlabel('contrast');

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      crfAx[d][i].tick_params(labelsize=15, width=1, length=8, direction='out');
      crfAx[d][i].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
      sns.despine(ax = crfAx[d][i], offset=10, trim=False);

      crfAx[d][i].set_ylabel('resp above baseline (sps)');
      crfAx[d][i].set_title('D%02d: sf:all - log resp %s' % (d, title_str));
      crfAx[d][i].legend(lines_log, [str(i) for i in np.round(all_sfs[v_sfs], 2)], loc='upper left');

saveName = "/allSfs_log_cell_%02d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fCRF:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()
'''
