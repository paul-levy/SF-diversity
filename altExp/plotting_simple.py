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

which_cell = int(sys.argv[1]);
lossType = int(sys.argv[2]);
fitType = int(sys.argv[3]);
crf_fit_type = int(sys.argv[4]);
descr_fit_type = int(sys.argv[5]);
norm_sim_on = int(sys.argv[6]);

norm_type = fitType;
expInd = 2; # (V1) altExp is #2

# at CNS
dataPath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/structures/';
save_loc = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/figures/';
# personal mac
#dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/altExp/structures/';
#save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/altExp/figures/cell4_sandbox/';
# prince cluster
#dataPath = '/home/pl1465/SF_diversity/altExp/structures/';
#save_loc = '/home/pl1465/SF_diversity/altExp/figures/';

expName = 'dataList.npy'
fitBase = 'fitList_190114c';

# first the fit type
if fitType == 1:
  fitSuf = '_flat';
elif fitType == 2:
  fitSuf = '_wght';
elif fitType == 3:
  fitSuf = '_c50';
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

fitListName = str(fitBase + fitSuf + lossSuf);

# set the save directory to save_loc, then create the save directory if needed
subDir   = fitListName.replace('fitList', 'fits').replace('.npy', '');
save_loc = str(save_loc + subDir + '/');
if not os.path.exists(save_loc):
  os.makedirs(save_loc);

if crf_fit_type == 1:
  crf_type_str = '-lsq';
if crf_fit_type == 2:
  crf_type_str = '-sqrt';
if crf_fit_type == 3:
  crf_type_str = '-poiss';
if crf_fit_type == 4:
  crf_type_str = '-poissMod';

rpt_fit = 1; # i.e. take the multi-start result
if rpt_fit:
  is_rpt = '_rpt';
else:
  is_rpt = '';

conDig = 3; # round contrast to the 3rd digit

dataList = np.load(str(dataPath + expName), encoding='latin1').item();

cellStruct = np.load(str(dataPath + dataList['unitName'][which_cell-1] + '_sfm.npy'), encoding='latin1').item();

# #### Load descriptive model fits, comp. model fits
descrFitName = hf.descrFit_name(descr_fit_type);

modParams = np.load(str(dataPath + fitListName), encoding= 'latin1').item();
modParamsCurr = modParams[which_cell-1]['params'];

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

data = cellStruct['sfm']['exp']['trial'];

ignore, modRespAll = mod_resp.SFMGiveBof(modParamsCurr, cellStruct, normType=norm_type, lossType=lossType, expInd=expInd);
print('norm type %02d' % (norm_type));
if norm_type == 2:
  gs_mean = modParamsCurr[1]; # guaranteed to exist after call to .SFMGiveBof, if norm_type == 2
  gs_std = modParamsCurr[2]; # guaranteed to exist ...
resp, stimVals, val_con_by_disp, validByStimVal, modResp = hf.tabulate_responses(cellStruct, expInd, modRespAll);
blankMean, blankStd, _ = hf.blankResp(cellStruct); 
modBlankMean = modParamsCurr[6]; # late additive noise is the baseline of the model
# all responses on log ordinate (y axis) should be baseline subtracted

all_disps = stimVals[0];
all_cons = stimVals[1];
all_sfs = stimVals[2];

nCons = len(all_cons);
nSfs = len(all_sfs);
nDisps = len(all_disps);

# #### Unpack responses

respMean = resp[0];
respStd = resp[1];
predMean = resp[2];
predStd = resp[3];

# modResp is (nFam, nSf, nCons, nReps) nReps is (currently; 2018.01.05) set to 20 to accommadate the current experiment with 10 repetitions
modLow = np.nanmin(modResp, axis=3);
modHigh = np.nanmax(modResp, axis=3);
modAvg = np.nanmean(modResp, axis=3);

# ### Plots

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
#    maxResp = np.maximum(np.max(np.max(respMean[d, ~np.isnan(respMean[d, :, :])])), np.max(np.max(predMean[d, ~np.isnan(respMean[d, :, :])])));
    
    for c in reversed(range(n_v_cons)):
        c_plt_ind = len(v_cons) - c - 1;
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);        

	# plot pred/measured ratio
        dispAx[d][c_plt_ind, 1].plot(all_sfs[v_sfs], np.divide(predMean[d, v_sfs, v_cons[c]]-blankMean, respMean[d, v_sfs, v_cons[c]]-blankMean), clip_on=False);
        dispAx[d][c_plt_ind, 1].axhline(1, clip_on=False, linestyle='dashed');
        
        # plot data
        dispAx[d][c_plt_ind, 0].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]], 
                                      respStd[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);

        # plot linear superposition prediction
#        dispAx[d][c_plt_ind, 0].errorbar(all_sfs[v_sfs], predMean[d, v_sfs, v_cons[c]], 
#                                      predStd[d, v_sfs, v_cons[c]], fmt='p', clip_on=False);

        # plot descriptive model fit
        
	# plot model fits
        dispAx[d][c_plt_ind, 0].fill_between(all_sfs[v_sfs], modLow[d, v_sfs, v_cons[c]], \
                                      modHigh[d, v_sfs, v_cons[c]], color='r', alpha=0.2);
        dispAx[d][c_plt_ind, 0].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], 'r-', alpha=0.7, clip_on=False);

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

    fCurr.suptitle('cell #%d, loss %.2f' % (which_cell, modParams[which_cell-1]['NLL']));

saveName = "/cell_%02d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'byDisp/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close();

# #### All SF tuning on one graph, split by dispersion

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
      else:
        curr_resps = modAvg;
        curr_mean = modBlankMean;
      maxResp = np.max(np.max(curr_resps[d, ~np.isnan(curr_resps[d, :, :])]));

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

saveName = "/allCons_cell_%02d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'byDisp/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()

# #### Plot just sfMix contrasts

# i.e. highest (up to) 4 contrasts for each dispersion

mixCons = 4;
maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));
#maxResp = np.maximum(np.max(np.max(np.max(respMean[~np.isnan(respMean)]))), np.max(np.max(np.max(predMean[~np.isnan(predMean)]))));

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

        # plot descriptive model fit

	# plot model fits
        sfMixAx[c_plt_ind, d].fill_between(all_sfs[v_sfs], modLow[d, v_sfs, v_cons[c]], \
                                      modHigh[d, v_sfs, v_cons[c]], color='r', alpha=0.2);
        sfMixAx[c_plt_ind, d].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], 'r-', alpha=0.7, clip_on=False);

        sfMixAx[c_plt_ind, d].set_xlim((np.min(all_sfs), np.max(all_sfs)));
        sfMixAx[c_plt_ind, d].set_ylim((0, 1.5*maxResp));
        sfMixAx[c_plt_ind, d].set_xscale('log');
        sfMixAx[c_plt_ind, d].set_xlabel('sf (c/deg)');
        sfMixAx[c_plt_ind, d].set_ylabel('resp (sps)');

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        sfMixAx[c_plt_ind, d].tick_params(labelsize=15, width=1, length=8, direction='out');
        sfMixAx[c_plt_ind, d].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
        sns.despine(ax=sfMixAx[c_plt_ind, d], offset=10, trim=False);

f.suptitle('cell #%d, loss %.2f' % (which_cell, modParams[which_cell-1]['NLL']));
	        
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

# plot model details - filter
imSizeDeg = cellStruct['sfm']['exp']['size'];
pixSize   = 0.0028; # fixed from Robbe
prefSf    = modParamsCurr[0];
dOrder    = modParamsCurr[1]
prefOri = 0; # just fixed value since no model param for this
aRatio = 1; # just fixed value since no model param for this
filtTemp  = mod_resp.oriFilt(imSizeDeg, pixSize, prefSf, prefOri, dOrder, aRatio);
filt      = (filtTemp - filtTemp[0,0])/ np.amax(np.abs(filtTemp - filtTemp[0,0]));

plt.subplot2grid(detailSize, (2, 0)); # set the current subplot location/size[default is 1x1]
plt.imshow(filt, cmap='gray');
plt.axis('off');
#plt.title('Filter in space', fontsize=20)

# plot model details - exc/suppressive components
omega = np.logspace(-2, 2, 1000);

sfRel = omega/prefSf;
s     = np.power(omega, dOrder) * np.exp(-dOrder/2 * np.square(sfRel));
sMax  = np.power(prefSf, dOrder) * np.exp(-dOrder/2);
sfExc = s/sMax;

inhSfTuning = hf.getSuppressiveSFtuning();

# Compute weights for suppressive signals
inhAsym = 0;
#inhAsym = modParamsCurr[8];
nInhChan = cellStruct['sfm']['mod']['normalization']['pref']['sf'];
inhWeight = [];
for iP in range(len(nInhChan)):
    # 0* if we ignore asymmetry; inhAsym* otherwise
    inhWeight = np.append(inhWeight, 1 + inhAsym * (np.log(cellStruct['sfm']['mod']['normalization']['pref']['sf'][iP]) - np.mean(np.log(cellStruct['sfm']['mod']['normalization']['pref']['sf'][iP]))));
           
sfInh = 0 * np.ones(omega.shape) / np.amax(modHigh); # mult by 0 because we aren't including a subtractive inhibition in model for now 7/19/17
sfNorm = np.sum(-.5*(inhWeight*np.square(inhSfTuning)), 1);
sfNorm = sfNorm/np.amax(np.abs(sfNorm));

# just setting up lines
plt.subplot2grid(detailSize, (2, 1)); # set the current subplot location/size[default is 1x1]
plt.semilogx([omega[0], omega[-1]], [0, 0], 'k--')
plt.semilogx([.01, .01], [-1.5, 1], 'k--')
plt.semilogx([.1, .1], [-1.5, 1], 'k--')
plt.semilogx([1, 1], [-1.5, 1], 'k--')
plt.semilogx([10, 10], [-1.5, 1], 'k--')
plt.semilogx([100, 100], [-1.5, 1], 'k--')
# now the real stuff
plt.semilogx(omega, sfExc, 'k-')
plt.semilogx(omega, sfInh, 'r--', linewidth=2);
plt.semilogx(omega, sfNorm, 'r-', linewidth=1);
plt.xlim([omega[0], omega[-1]]);
plt.ylim([-1.5, 1]);
plt.xlabel('SF (cpd)', fontsize=20);
plt.ylabel('Normalized response (a.u.)', fontsize=20);
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=plt.subplot2grid(detailSize, (2, 1)), offset=10, trim=False);

# last but not least...and not last... response nonlinearity
curr_ax = plt.subplot2grid(detailSize, (2, 2)); # set the current subplot location/size[default is 1x1]
plt.plot([-1, 1], [0, 0], 'k--')
plt.plot([0, 0], [-.1, 1], 'k--')
plt.plot(np.linspace(-1,1,100), np.power(np.maximum(0, np.linspace(-1,1,100)), modParamsCurr[3]), 'k-', linewidth=2)
plt.plot(np.linspace(-1,1,100), np.maximum(0, np.linspace(-1,1,100)), 'k--', linewidth=1)
plt.xlim([-1, 1]);
plt.ylim([-.1, 1]);
plt.text(0.5, 1.1, 'respExp: {:.2f}'.format(modParamsCurr[3]), fontsize=12, horizontalalignment='center', verticalalignment='center');
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=curr_ax, offset=5, trim=False);
    
if norm_type == 3: # plot the c50 filter (i.e. effective c50 as function of SF)
  stimSf = np.logspace(-2, 2, 101);
  filtPeak = modFit[11];
  stdLeft = modFit[9];
  stdRight = modFit[10];
  
  filter = setSigmaFilter(filtPeak, stdLeft, stdRight);
  offset_filt = modFit[8];
  scale_filt = -(1-offset_filt); # we always scale so that range is [offset_sf, 1]
  c50_filt = evalSigmaFilter(filter, scale_filt, offset_filt, stimSf)
 
  # now plot
  curr_ax = plt.subplot2grid(detailSize, (2, 4));
  plt.semilogx(stimSf, c50_filt);
  plt.title('(mu, stdL/R, offset) = (%.2f, %.2f|%.2f, %.2f)' % (filtPeak, stdLeft, stdRight, offset_filt));
  plt.xlabel('spatial frequency (c/deg)');
  plt.ylabel('c50 (con %)')

# print, in text, model parameters:
plt.subplot2grid(detailSize, (0, 4)); # set the current subplot location/size[default is 1x1]
plt.text(0.5, 0.5, 'prefSf: {:.3f}'.format(modParamsCurr[0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.4, 'derivative order: {:.3f}'.format(modParamsCurr[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.3, 'response scalar: {:.3f}'.format(modParamsCurr[4]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.2, 'sigma: {:.3f} | {:.3f}'.format(np.power(10, modParamsCurr[2]), modParamsCurr[2]), fontsize=12, horizontalalignment='center', verticalalignment='center');
if lossType == 3: # modpoiss
  varGain = modParamsCurr[7];
  plt.text(0.5, 0.1, 'varGain: {:.3f}'.format(varGain), fontsize=12, horizontalalignment='center', verticalalignment='center');

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

#########
# Normalization pool simulations
#########
if norm_sim_on:

    conLevels = [1, 0.75, 0.5, 0.33, 0.1];
    nCons = len(conLevels);
    sfCenters = np.logspace(-2, 2, 21); # just for now...
    #sfCenters = allSfs;
    fNorm, conDisp_plots = plt.subplots(nCons, nDisps, sharey=True, figsize=(40,30));
    norm_sim = np.nan * np.empty((nDisps, nCons, len(sfCenters)));
    if len(modParamsCurr) < 9:
        modParamsCurr.append(hf.random_in_range([-0.35, 0.35])[0]); # enter asymmetry parameter

    # simulations
    for disp in range(nDisps):
        for conLvl in range(nCons):
          print('simulating normResp for family ' + str(disp+1) + ' and contrast ' + str(conLevels[conLvl]));
          for sfCent in range(len(sfCenters)):
              # if modParamsCurr doesn't have inhAsym parameter, add it!
              if norm_type == 2:
                unweighted = 1;
                _, _, _, normRespSimple, _ = mod_resp.SFMsimulate(modParamsCurr, cellStruct, disp+1, conLevels[conLvl], sfCenters[sfCent], unweighted, normType=norm_type, expInd=expInd);
                nTrials = normRespSimple.shape[0];
                nInhChan = cellStruct['sfm']['mod']['normalization']['pref']['sf'];
                inhWeightMat  = hf.genNormWeights(cellStruct, nInhChan, gs_mean, gs_std, nTrials);
                normResp = np.sqrt((inhWeightMat*normRespSimple).sum(1)).transpose();
                norm_sim[disp, conLvl, sfCent] = np.mean(normResp); # take mean of the returned simulations (10 repetitions per stim. condition)
              else: # norm_type == 1 or 3:
                _, _, _, _, normResp = mod_resp.SFMsimulate(modParamsCurr, cellStruct, disp+1, conLevels[conLvl], sfCenters[sfCent], normType = norm_type, expInd=expInd);
                norm_sim[disp, conLvl, sfCent] = np.mean(normResp); # take mean of the returned simulations (10 repetitions per stim. condition)

          if norm_type == 2:
            maxResp = np.max(norm_sim[disp, conLvl, :]);
            conDisp_plots[conLvl, disp].text(0.5, 0.0, 'contrast: {:.2f}, dispersion level: {:.0f}, mu|std: {:.2f}|{:.2f}'.format(conLevels[conLvl], disp+1, modParamsCurr[8], modParamsCurr[9]), fontsize=12, horizontalalignment='center', verticalalignment='center'); 
          else: # norm_type == 1 or 3:
            conDisp_plots[conLvl, disp].text(0.5, 1.1, 'contrast: {:.2f}, dispersion level: {:.0f}, asym: {:.2f}'.format(conLevels[conLvl], disp+1, modParamsCurr[8]), fontsize=12, horizontalalignment='center', verticalalignment='center'); 

          conDisp_plots[conLvl, disp].semilogx(sfCenters, norm_sim[disp, conLvl, :], 'b', clip_on=False);
          conDisp_plots[conLvl, disp].set_xlim([1e-2, 1e2]);

          conDisp_plots[conLvl, disp].tick_params(labelsize=15, width=1, length=8, direction='out');
          conDisp_plots[conLvl, disp].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
          if conLvl == 0:
              conDisp_plots[conLvl, disp].set_xlabel('sf center (cpd)', fontsize=20);
          if disp == 0:
              conDisp_plots[conLvl, disp].set_ylabel('Response (ips)', fontsize=20);
          # remove axis from top and right, set ticks to be only bottom and left
          conDisp_plots[conLvl, disp].spines['right'].set_visible(False);
          conDisp_plots[conLvl, disp].spines['top'].set_visible(False);
          conDisp_plots[conLvl, disp].xaxis.set_ticks_position('bottom');
          conDisp_plots[conLvl, disp].yaxis.set_ticks_position('left');
    conDisp_plots[0, 2].text(0.5, 1.2, 'Normalization pool responses', fontsize=16, horizontalalignment='center', verticalalignment='center', transform=conDisp_plots[0, 2].transAxes);

### now save all figures (sfMix contrasts, details, normalization stuff)
#pdb.set_trace()
allFigs = [f, fDetails];
saveName = "/cell_%02d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'sfMixOnly/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for fig in range(len(allFigs)):
    pdfSv.savefig(allFigs[fig])
    plt.close(allFigs[fig])
pdfSv.close()

#########
# #### Plot contrast response functions with Naka-Rushton fits
#########
'''
crfAx = []; fCRF = [];
fSum, crfSum = plt.subplots(nDisps, 2, figsize=(30, 30), sharex=False, sharey=False);
fCRF.append(fSum);
crfAx.append(crfSum);

fits = np.load(str(dataPath + crfFitName), encoding='latin1').item();
crfFitsSepC50 = fits[which_cell-1][str('fits_each' + is_rpt)];
crfFitsOneC50 = fits[which_cell-1][str('fits' + is_rpt)];

for d in range(nDisps):
    
    # which sfs have at least one contrast presentation?
    v_sfs = np.where(np.sum(~np.isnan(respMean[d, :, :]), axis = 1) > 0);
    n_v_sfs = len(v_sfs[0])
    n_rows = 3; #int(np.floor(n_v_sfs/2));
    n_cols = 4; #n_v_sfs - n_rows
    fCurr, crfCurr = plt.subplots(n_rows, n_cols, figsize=(n_cols*10, n_rows*15), sharex = True, sharey = True);
    fCRF.append(fCurr)
    crfAx.append(crfCurr);
    
    c50_sep = np.zeros((n_v_sfs, 1));
    c50_all = np.zeros((n_v_sfs, 1));

    rvc_plots = [];

    for sf in range(n_v_sfs):
        row_ind = sf/n_cols;
        col_ind = np.mod(sf, n_cols);
        sf_ind = v_sfs[0][sf];

        v_cons = ~np.isnan(respMean[d, sf_ind, :]);
        n_cons = sum(v_cons);
        plot_cons = np.linspace(0, np.max(all_cons[v_cons]), 100); # 100 steps for plotting...
	#plot_cons = np.linspace(np.min(all_cons[v_cons]), np.max(all_cons[v_cons]), 100); # 100 steps for plotting...

	# organize responses
        resps_curr = np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, ));
        resps_w_blank = np.hstack((blankMean, resps_curr));

	# CRF fit
        curr_fit_sep = crfFitsSepC50[d][sf_ind]['params'];
        curr_fit_all = crfFitsOneC50[d][sf_ind]['params'];
	# ignore varGain when reporting loss here...
        sep_pred = hf.naka_rushton(np.hstack((0, all_cons[v_cons])), curr_fit_sep[0:4]);
        all_pred = hf.naka_rushton(np.hstack((0, all_cons[v_cons])), curr_fit_all[0:4]);

        if lossType == 3:
          r_sep, p_sep = hf.mod_poiss(sep_pred, curr_fit_sep[4]);
          r_all, p_all = hf.mod_poiss(all_pred, curr_fit_all[4]);
          sep_loss = -np.sum(loss(np.round(resps_w_blank), r_sep, p_sep));
          all_loss = -np.sum(loss(np.round(resps_w_blank), r_all, p_all));
        elif lossType == 2:	
          sep_loss = -np.sum(loss(np.round(resps_w_blank), sep_pred));
          all_loss = -np.sum(loss(np.round(resps_w_blank), all_pred));
        else: # i.e. fit_type == 1 || == 2
          sep_loss = np.sum(loss(np.round(resps_w_blank), sep_pred));
          all_loss = np.sum(loss(np.round(resps_w_blank), all_pred));
	 
        c50_sep[sf] = curr_fit_sep[3];
        c50_all[sf] = curr_fit_all[3];

        # summary plots
        curr_rvc = crfAx[0][d, 0].plot(all_cons[v_cons], resps_curr, '-', clip_on=False);
        rvc_plots.append(curr_rvc[0]);

        # NR fit plots
        stdPts = np.hstack((0, np.reshape([respStd[d, sf_ind, v_cons]], (n_cons, ))));
        expPts = crfAx[d+1][row_ind, col_ind].errorbar(np.hstack((0, all_cons[v_cons])), resps_w_blank, stdPts, fmt='o', clip_on=False);

        sepPlt = crfAx[d+1][row_ind, col_ind].plot(plot_cons, hf.naka_rushton(plot_cons, curr_fit_sep), linestyle='dashed');
        allPlt = crfAx[d+1][row_ind, col_ind].plot(plot_cons, hf.naka_rushton(plot_cons, curr_fit_all), linestyle='dashed');
	# accompanying text...
        crfAx[d+1][row_ind, col_ind].text(0, 0.9, 'free [%.1f]: gain %.1f; c50 %.3f; exp: %.2f; base: %.1f, varGn: %.2f' % (sep_loss, curr_fit_sep[1], curr_fit_sep[3], curr_fit_sep[2], curr_fit_sep[0], curr_fit_sep[4]), 
		horizontalalignment='left', verticalalignment='center', transform=crfAx[d+1][row_ind, col_ind].transAxes, fontsize=30);
        crfAx[d+1][row_ind, col_ind].text(0, 0.8, 'fixed [%.1f]: gain %.1f; c50 %.3f; exp: %.2f; base: %.1f, varGn: %.2f' % (all_loss, curr_fit_all[1], curr_fit_all[3], curr_fit_all[2], curr_fit_all[0], curr_fit_all[4]), 
		horizontalalignment='left', verticalalignment='center', transform=crfAx[d+1][row_ind, col_ind].transAxes, fontsize=30);

	# legend
        crfAx[d+1][row_ind, col_ind].legend((expPts[0], sepPlt[0], allPlt[0]), ('data', 'free c50', 'fixed c50'), fontsize='large', loc='center left')

        plt_x = d+1; plt_y = (row_ind, col_ind);

        crfAx[plt_x][plt_y].set_xscale('symlog', linthreshx=0.01); # symlog will allow us to go down to 0 
        crfAx[plt_x][plt_y].set_xlabel('contrast', fontsize='medium');
        crfAx[plt_x][plt_y].set_ylabel('resp (sps)', fontsize='medium');
        crfAx[plt_x][plt_y].set_title('D%02d: sf: %.3f' % (d+1, all_sfs[sf_ind]), fontsize='large');

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        sns.despine(ax = crfAx[plt_x][plt_y], offset = 10, trim=False);
        crfAx[plt_x][plt_y].tick_params(labelsize=25, width=2, length=16, direction='out');
        crfAx[plt_x][plt_y].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...

    # make summary plots nice
    for i in range(2):
        crfAx[0][d, i].set_xscale('log');
        sns.despine(ax = crfAx[0][d, i], offset=10, trim=False);

        # Set ticks out, remove top/right axis, put ticks only on bottom/left
        crfAx[0][d, i].tick_params(labelsize=25, width=2, length=16, direction='out');
        crfAx[0][d, i].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
   
    # plot c50 as f/n of SF; plot sf tuning as reference...
    sepC50s = crfAx[0][d, 1].plot(all_sfs[v_sfs[0]], c50_sep);
    allC50s = crfAx[0][d, 1].plot(all_sfs[v_sfs[0]], c50_all);
    maxC50 = np.maximum(np.max(c50_sep), np.max(c50_all));
    v_cons = np.array(val_con_by_disp[d]);
    sfRef = respMean[d, v_sfs[0], v_cons[-1]]; # plot highest contrast spatial frequency tuning curve
	# we normalize the sf tuning, flip upside down so it matches the profile of c50, which is lowest near peak SF preference
    invSF = crfAx[0][d, 1].plot(all_sfs[v_sfs[0]],  maxC50*(1-sfRef/np.max(sfRef)), linestyle='dashed');
    crfAx[0][d, 1].set_xlim([all_sfs[0], all_sfs[-1]]);

    crfAx[0][d, 0].set_title('D%02d - all RVC' % (d), fontsize='large');
    crfAx[0][d, 0].set_xlabel('contrast', fontsize='large');
    crfAx[0][d, 0].set_ylabel('resp (sps)', fontsize='large');
    crfAx[0][d, 0].legend(rvc_plots, [str(i) for i in np.round(all_sfs[v_sfs[0]], 2)], loc='upper left');

    crfAx[0][d, 1].set_title('D%02d - C50 (fixed vs free)' % (d), fontsize='large');
    crfAx[0][d, 1].set_xlabel('sf (cpd)', fontsize='large');
    crfAx[0][d, 1].set_ylabel('c50', fontsize='large');
    crfAx[0][d, 1].legend((sepC50s[0], allC50s[0], invSF[0]), ('c50 free', 'c50 fixed', 'rescaled SF tuning'), fontsize='large', loc='center left');
    
saveName = "/cell_NR_%02d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fCRF:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()
'''
# #### Plot contrast response functions with (full) model predictions AND Naka-Rushton 
'''
rvcAx = []; fRVC = [];

# crfFitsSepC50 loaded above

for d in range(nDisps):
    
    # which sfs have at least one contrast presentation?
    v_sfs = np.where(np.sum(~np.isnan(respMean[d, :, :]), axis = 1) > 0);
    n_v_sfs = len(v_sfs[0])
    n_rows = 3; #int(np.floor(n_v_sfs/2));
    n_cols = 4; #n_v_sfs - n_rows
    fCurr, rvcCurr = plt.subplots(n_rows, n_cols, figsize=(n_cols*10, n_rows*10), sharex = True, sharey = True);
    fRVC.append(fCurr)
    rvcAx.append(rvcCurr);
    
    #rvc_plots = [];

    for sf in range(n_v_sfs):
        row_ind = sf/n_cols;
        col_ind = np.mod(sf, n_cols);
        sf_ind = v_sfs[0][sf];
       	plt_x = d; plt_y = (row_ind, col_ind);

        v_cons = ~np.isnan(respMean[d, sf_ind, :]);
        n_cons = sum(v_cons);
        plot_cons = np.linspace(0, np.max(all_cons[v_cons]), 100); # 100 steps for plotting...
	#plot_cons = np.linspace(np.min(all_cons[v_cons]), np.max(all_cons[v_cons]), 100); # 100 steps for plotting...

	# organize responses
        resps_curr = np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, ));
        resps_w_blank = np.hstack((blankMean, resps_curr));

        # plot data
        dataPlt = rvcAx[plt_x][plt_y].plot(all_cons[v_cons], np.maximum(np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, )), 0.1), '-', clip_on=False);
	# RVC with full model fit
        rvcAx[plt_x][plt_y].fill_between(all_cons[v_cons], modLow[d, sf_ind, v_cons], \
                                      modHigh[d, sf_ind, v_cons], color='r', alpha=0.2);

        # RVC from Naka-Rushton fit
        #curr_fit_sep = crfFitsSepC50[d][sf_ind]['params'];
        #nrPlt = rvcAx[plt_x][plt_y].plot(plot_cons, hf.naka_rushton(plot_cons, curr_fit_sep), linestyle='dashed');
        #pdb.set_trace();
        modPlt = rvcAx[plt_x][plt_y].plot(all_cons[v_cons], np.maximum(modAvg[d, sf_ind, v_cons], 0.1), 'r-', alpha=0.7, clip_on=False);

        rvcAx[plt_x][plt_y].set_xscale('symlog', linthreshx=0.01); # symlog will allow us to go down to 0 
        rvcAx[plt_x][plt_y].set_xlabel('contrast', fontsize='medium');
        rvcAx[plt_x][plt_y].set_ylabel('resp (sps)', fontsize='medium');
        rvcAx[plt_x][plt_y].set_title('D%02d: sf: %.3f' % (d+1, all_sfs[sf_ind]), fontsize='large');
        #rvcAx[plt_x][plt_y].legend((dataPlt[0], modPlt[0], nrPlt[0]), ('data', 'model avg', 'Naka-Rushton'), fontsize='large', loc='center left');
        rvcAx[plt_x][plt_y].legend((dataPlt[0], modPlt[0]), ('data', 'model avg'), fontsize='large', loc='center left');

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        sns.despine(ax = rvcAx[plt_x][plt_y], offset = 10, trim=False);
        rvcAx[plt_x][plt_y].tick_params(labelsize=25, width=2, length=16, direction='out');
        rvcAx[plt_x][plt_y].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...

    fCurr.suptitle('cell #%d, loss%.2f' % (which_cell, modParams[which_cell-1]['NLL']));

saveName = "/cell_%02d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fRVC:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()

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

saveName = "/allSfs_log_cell_%02d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fCRF:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()
'''
