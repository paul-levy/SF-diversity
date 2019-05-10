# coding: utf-8
# NOTE: Unlike plot_compare.py, this file is used to plot the data/model for just one type (i.e. just flat, or just weighted, etc)

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
rcParams['font.size'] = 20;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 2.5;
rcParams['axes.linewidth'] = 1.5;
rcParams['lines.markersize'] = 5;
rcParams['font.style'] = 'oblique';

cellNum  = int(sys.argv[1]);
lossType = int(sys.argv[2]);
fitType  = int(sys.argv[3]);
expDir   = sys.argv[4]; 
modRecov = int(sys.argv[5]);
descrMod = int(sys.argv[6]);
rvcAdj   = int(sys.argv[7]); # if 1, then let's load rvcFits to adjust responses to F1
if len(sys.argv) > 8:
  respVar = int(sys.argv[8]);
else:
  respVar = 1;

loc_base = os.getcwd() + '/';

data_loc = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + 'figures/';

### DATALIST
#expName = 'dataList.npy';
#expName = 'dataList_glx_mr.npy'
expName = 'dataList_glx.npy'
#expName = 'dataList_mr.npy'
### FITLIST
#fitBase = 'fitList_190321c';
#fitBase = 'fitListSPcns_181130c';
#fitBase = 'fitListSP_181202c';
#fitBase = 'fitList_190206c';
#fitBase = 'fitList_190321c';
#fitBase = 'mr_fitList_190502cA';
fitBase = 'fitList_190502cA';
### RVCFITS
rvcBase = 'rvcFits'; # direc flag & '.npy' are added

### Descriptive fits?
if descrMod > -1:
  try:
    if modRecov == 1:
      fitStr = hf.fitType_suffix(fitType);
      descrBase = 'mr%s_descrFits_190503_poiss_%s.npy' % (fitStr, hf.descrMod_name(descrMod)); 
      descrFits = hf.np_smart_load(data_loc + descrBase);
    else:
      descrBase = 'descrFits_190503_poiss_%s.npy' % hf.descrMod_name(descrMod); 
      descrFits = hf.np_smart_load(data_loc + descrBase);
  except:
    descrFits = None;
else:
  descrFits = None;

# the loss type
if lossType == 1:
  loss = lambda resp, pred: np.sum(np.square(np.sqrt(resp) - np.sqrt(pred)));
elif lossType == 2:
  loss = lambda resp, pred: poisson.logpmf(resp, pred);
elif lossType == 3:
  loss = lambda resp, r, p: np.log(nbinom.pmf(resp, r, p));
elif lossType == 4:
  loss = lambda resp, pred: np.sum(np.square(np.sqrt(resp) - np.sqrt(pred)));

fitName = hf.fitList_name(fitBase, fitType, lossType);
fitSuf  = hf.fitType_suffix(fitType);
lossSuf = hf.lossType_suffix(lossType);

# set the save directory to save_loc, then create the save directory if needed
compDir  = str(fitBase + ('%s' % fitSuf) + lossSuf);
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
fitList= hf.np_smart_load(data_loc + fitName);

cellName = dataList['unitName'][cellNum-1];
try:
  cellType = dataList['unitType'][cellNum-1];
except: 
  # TODO: note, this is dangerous; thus far, only V1 cells don't have 'unitType' field in dataList, so we can safely do this
  cellType = 'V1'; 

expData  = np.load(str(data_loc + cellName + '_sfm.npy'), encoding='latin1').item();
expInd   = hf.get_exp_ind(data_loc, cellName)[0];

# #### Load model fits

modFit = fitList[cellNum-1]['params'];
# now get normalization stuff
normParams = hf.getNormParams(modFit, fitType);
if fitType == 1:
  inhAsym = normParams;
elif fitType == 2 or fitType == 4:
  gs_mean = normParams[0];
  gs_std  = normParams[1];
elif fitType == 3:
  # sigma calculation
  offset_sigma = normParams[0];  # c50 filter will range between [v_sigOffset, 1]
  stdLeft      = normParams[1];  # std of the gaussian to the left of the peak
  stdRight     = normParams[2]; # '' to the right '' 
  sfPeak       = normParams[3]; # where is the gaussian peak?
else:
  inhAsym = normParams;

# descrFit, if exists
if descrFits is not None:
  descrParams = descrFits[cellNum-1]['params'];
else:
  descrParams = None;

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

modResp = mod_resp.SFMGiveBof(modFit, expData, normType=fitType, lossType=lossType, expInd=expInd)[1];
# now organize the responses
orgs = hf.organize_resp(modResp, expData, expInd);
oriModResp = orgs[0]; # only non-empty if expInd = 1
conModResp = orgs[1]; # only non-empty if expInd = 1
sfmixModResp = orgs[2];
allSfMix = orgs[3];

modLow = np.nanmin(allSfMix, axis=3);
modHigh = np.nanmax(allSfMix, axis=3);
modAvg = np.nanmean(allSfMix, axis=3);
modSponRate = modFit[6];

# more tabulation - stim vals, organize measured responses
if modRecov == 1:
  modParamGT, overwriteSpikes = hf.get_recovInfo(expData, fitType);
else:
  overwriteSpikes = None;
_, stimVals, val_con_by_disp, _, _ = hf.tabulate_responses(expData, expInd);
if rvcAdj == 1:
  rvcFlag = '_f1';
  rvcFits = hf.get_rvc_fits(data_loc, expInd, cellNum, rvcName=rvcBase);
else:
  rvcFlag = '';
  rvcFits = hf.get_rvc_fits(data_loc, expInd, cellNum, rvcName='None');
spikes = hf.get_spikes(expData['sfm']['exp']['trial'], rvcFits=rvcFits, expInd=expInd, overwriteSpikes=overwriteSpikes);
_, _, respOrg, respAll = hf.organize_resp(spikes, expData, expInd);

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

blankMean, blankStd, _ = hf.blankResp(expData); 

all_disps = stimVals[0];
all_cons = stimVals[1];
all_sfs = stimVals[2];

nCons = len(all_cons);
nSfs = len(all_sfs);
nDisps = len(all_disps);

# ### Plots

# set up colors, labels
modClr = 'r';
modTxt = 'model';
dataClr = 'k';
dataTxt = 'data';
descrClr = 'b';

# #### Plots by dispersion

fDisp = []; dispAx = [];

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);    
for d in range(nDisps):
    
    v_cons = val_con_by_disp[d];
    n_v_cons = len(v_cons);
    
    fCurr, dispCurr = plt.subplots(n_v_cons, 2, figsize=(nDisps*8, n_v_cons*8), sharey=False);
    fDisp.append(fCurr)
    dispAx.append(dispCurr);
    
    maxResp = np.max(np.max(respMean[d, ~np.isnan(respMean[d, :, :])]));
    
    for c in reversed(range(n_v_cons)):
        c_plt_ind = len(v_cons) - c - 1;
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);        

        ### left side of plots
        ## plot data
        dispAx[d][c_plt_ind, 0].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]], 
                                         respVar[d, v_sfs, v_cons[c]], color=dataClr, fmt='o', clip_on=False, label=dataTxt);
        dispAx[d][c_plt_ind, 0].axhline(blankMean, color=dataClr, linestyle='dashed', label='spon. rate');

	## plot model fit 
        dispAx[d][c_plt_ind, 0].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], alpha=0.7, color=modClr, clip_on=False, label=modTxt);
        dispAx[d][c_plt_ind, 0].axhline(modSponRate, color=modClr, linestyle='dashed')
        if descrParams is not None:
          prms_curr = descrParams[d, v_cons[c]];
          descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod);
          dispAx[d][c_plt_ind, 0].plot(sfs_plot, descrResp, color=descrClr, label='descr. fit');

        ### right side of plots
        if d == 0:
          ## plot everything again on log-log coordinates...
          # first data
          dispAx[d][c_plt_ind, 1].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]], 
                                                     respVar[d, v_sfs, v_cons[c]], fmt='o', color=dataClr, clip_on=False, label=dataTxt);

          # then model fits
          dispAx[d][c_plt_ind, 1].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], color=modClr, alpha=0.7, clip_on=False, label=modTxt);

          # plot descriptive model fit -- and inferred characteristic frequency
          if descrParams is not None: # i.e. descrFits isn't empty, then plot it
            prms_curr = descrParams[d, v_cons[c]];
            descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod);
            dispAx[d][c_plt_ind, 1].plot(sfs_plot, descrResp, color=descrClr, label='descr. fit', clip_on=False)
            # now plot characteristic frequency!  
            char_freq = hf.dog_charFreq(prms_curr, descrMod);
            if char_freq != np.nan:
              dispAx[d][c_plt_ind, 1].plot(char_freq, 1, 'v', color='k', label='char. freq');

          dispAx[d][c_plt_ind, 1].set_title('log-log');
          #dispAx[d][c_plt_ind, 1].set_title('log-log: %.1f%% varExpl' % dfVarExpl[d, v_cons[c]]);
          dispAx[d][c_plt_ind, 1].set_xscale('log');
          dispAx[d][c_plt_ind, 1].set_yscale('log'); # double log

        '''
        ## Now, if dispersion: plot the individual comp. responses...
        if d>0:
          xticks = np.array([]); xticklabels = np.array([]);
          for j in v_sfs:
            comps = [];

            curr_sup = respAll[d, j, v_cons[c]]; # get the component responses only at the relevant conditions
            # todo: check if this is the right measure of variance you want...
            curr_sup_var = respVar[d, v_sfs_inds[j], v_cons[c]];
            # now get the individual responses
            n_comps = all_disps[d];

            val_trials, _, _, _ = hf.get_valid_trials(data, d, v_cons[c], j, expInd)
            isolResp, _, _, _ = hf.get_isolated_responseAdj(data, val_trials, spikes);

            # first, reset color cycle so that it's the same each time around
            dispAx[d][c_plt_ind, 1].set_prop_cycle(None);
            x_pos = [j-0.25, j+0.25];
            xticks = np.append(xticks, x_pos);
            xticklabels = np.append(xticklabels, ['mix', 'isol']);

            for i in range(n_comps): # difficult to make pythonic/array, so just iterate over each component
              # NOTE: for now, we will use the response-in-mixture std for both response stds...
              curr_means = [curr_sup[i], isolResp[i][0]]; # isolResp[i] is [mean, std] --> just get mean ([0])
              curr_stds = [curr_f1_std[i], curr_f1_std[i]];
              curr_comp = dispAx[d][c_plt_ind, 1].errorbar(x_pos, curr_means, curr_stds, fmt='-o', clip_on=False);
              comps.append(curr_comp[0]);

            comp_str = [str(i) for i in range(n_comps)];
            dispAx[d][c_plt_ind, 1].set_xticks(xticks);
            dispAx[d][c_plt_ind, 1].set_xticklabels(xticklabels);
            dispAx[d][c_plt_ind, 1].legend(comps, comp_str, loc=0);
            #dispAx[d][c_plt_ind, 1].set_ylim((0, 1.5*maxPlotComp));
            dispAx[d][c_plt_ind, 1].set_title('Component responses');
        '''

        ## Now, set things for both plots (formatting)
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
        #dispAx[d][c_plt_ind, 1].set_ylabel('ratio (pred:measure)');
        #dispAx[d][c_plt_ind, 1].set_ylim((1e-1, 1e3));
        #dispAx[d][c_plt_ind, 1].set_yscale('log');
        #dispAx[d][c_plt_ind, 1].legend();

    fCurr.suptitle('%s #%d, loss %.2f' % (cellType, cellNum, fitList[cellNum-1]['NLL']));

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

fDisp = []; dispAx = [];

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);
  
for d in range(nDisps):
    
    v_cons = val_con_by_disp[d];
    n_v_cons = len(v_cons);
    
    fCurr, dispCurr = plt.subplots(1, 3, figsize=(35, 20)); # left side for flat; middle for data; right side for weighted model
    fDisp.append(fCurr)
    dispAx.append(dispCurr);

    fCurr.suptitle('%s #%d' % (cellType, cellNum));

    resps_curr = [modAvg, respMean];
    labels     = ['model', 'data'];

    for i in range(len(resps_curr)):
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

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      dispAx[d][i].tick_params(labelsize=15, width=2, length=16, direction='out');
      dispAx[d][i].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
      sns.despine(ax=dispAx[d][i], offset=10, trim=False); 

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
                                       respVar[d, v_sfs, v_cons[c]], fmt='o', clip_on=False, label=dataTxt);

        # plot model fit
        sfMixAx[c_plt_ind, d].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], color=modClr, alpha=0.7, clip_on=False, label=modTxt)
        # plot descrFit, if there
        if descrParams is not None:
          prms_curr = descrParams[d, v_cons[c]];
          descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod);
          sfMixAx[c_plt_ind, d].plot(sfs_plot, descrResp, label='descr. fit');

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
f.suptitle('%s #%d (%s), loss %.2f' % (cellType, cellNum, cellName, fitList[cellNum-1]['NLL']));
	        
#########
# Plot secondary things - filter, normalization, nonlinearity, etc
#########

fDetails = plt.figure();
fDetails.set_size_inches(w=25,h=10)

detailSize = (3, 5);
if oriModResp is not None: # then we're using an experiment with ori curves
  # plot ori tuning
  curr_ax = plt.subplot2grid(detailSize, (0, 2));
  plt.plot(expData['sfm']['exp']['ori'], oriModResp, '%so' % modClr, clip_on=False, label=modTxt); # Model response
  expPlt = plt.plot(expData['sfm']['exp']['ori'], expData['sfm']['exp']['oriRateMean'], 'o-', clip_on=False); # Exp responses
  plt.xlabel('Ori (deg)', fontsize=12);
  plt.ylabel('Response (ips)', fontsize=12);
if conModResp is not None: # then we're using an experiment with RVC curves
  # CRF - with values from TF simulation and the broken down (i.e. numerator, denominator separately) values from resimulated conditions
  curr_ax = plt.subplot2grid(detailSize, (0, 1)); # default size is 1x1
  consUse = expData['sfm']['exp']['con'];
  plt.semilogx(consUse, expData['sfm']['exp']['conRateMean'], '%so-' % dataClr, clip_on=False); # Measured responses
  plt.plot(consUse, conModResp, '%so' % modClr, clip_on=False, label=modTxt); # Model response
  plt.xlabel('Con (%)', fontsize=20);
  # Remove top/right axis, put ticks only on bottom/left
  sns.despine(ax=curr_ax, offset = 5);

# poisson test - mean/var for each condition (i.e. sfXdispXcon)
curr_ax = plt.subplot2grid(detailSize, (0, 0)); # set the current subplot location/size[default is 1x1]
val_conds = ~np.isnan(respMean);
gt0 = np.logical_and(respMean[val_conds]>0, respVar[val_conds]>0);
plt.loglog([0.01, 1000], [0.01, 1000], 'k--');
plt.loglog(respMean[val_conds][gt0], np.square(respVar[val_conds][gt0]), 'o');
# skeleton for plotting modulated poisson prediction
if lossType == 3: # i.e. modPoiss
  mean_vals = np.logspace(-1, 2, 50);
  varGain  = x[7];
  plt.loglog(mean_vals, mean_vals + varGain*np.square(mean_vals));
plt.xlabel('Mean (sps)');
plt.ylabel('Variance (sps^2)');
plt.title('Super-poisson?');
plt.axis('equal');
sns.despine(ax=curr_ax, offset=5, trim=False);

#  RVC - pick center SF
curr_ax = plt.subplot2grid(detailSize, (0, 1)); # default size is 1x1
disp_rvc = 0;
val_cons = np.array(val_con_by_disp[disp_rvc]);
v_sfs = ~np.isnan(respMean[disp_rvc, :, val_cons[0]]); # remember, for single gratings, all cons have same #/index of sfs
sfToUse = np.int(np.floor(len(v_sfs)/2));
plt.semilogx(all_cons[val_cons], respMean[disp_rvc, sfToUse, val_cons], 'o', clip_on=False); # Measured responses
plt.plot(all_cons[val_cons], modAvg[disp_rvc, sfToUse, val_cons], '%so-' % modClr, clip_on=False, label=modTxt); # Model responses
plt.xlabel('Con (%)', fontsize=20);
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=curr_ax, offset = 5);

# plot model details - exc/suppressive components
omega = np.logspace(-2, 2, 1000);
prefSf = modFit[0];
dOrder = modFit[1];
sfRel = omega/prefSf;
s     = np.power(omega, dOrder) * np.exp(-dOrder/2 * np.square(sfRel));
sMax  = np.power(prefSf, dOrder) * np.exp(-dOrder/2);
sfExc = s/sMax;

inhSfTuning = hf.getSuppressiveSFtuning();

## Compute weights for suppressive signals
nInhChan = expData['sfm']['mod']['normalization']['pref']['sf'];
nTrials =  inhSfTuning.shape[0];
# first, normalization signal
if fitType == 2 or fitType == 4: # tuned
  inhWeight = hf.genNormWeights(expData, nInhChan, gs_mean, gs_std, nTrials, expInd, fitType);
  inhWeight = inhWeight[:, :, 0]; # genNormWeights gives us weights as nTr x nFilters x nFrames - we have only one "frame" here, and all are the same
  sfNormTune = np.sum(-.5*(inhWeight*np.square(inhSfTuning)), 1);
  sfNorm = sfNormTune/np.amax(np.abs(sfNormTune));
else: # flat; and all else not yet written
  if fitType != 1:
    warnings.warn('yet to write the code for computing/plotting this normalization type in plot_simple.py; defaulting to flat');
  inhAsym = 0;
  inhWeight = [];
  for iP in range(len(nInhChan)):
      inhWeight = np.append(inhWeight, 1 + inhAsym * (np.log(expData['sfm']['mod']['normalization']['pref']['sf'][iP]) - np.mean(np.log(expData['sfm']['mod']['normalization']['pref']['sf'][iP]))));
  sfNorm = np.sum(-.5*(inhWeight*np.square(inhSfTuning)), 1);
  sfNorm = sfNorm/np.amax(np.abs(sfNorm));

# just setting up lines
curr_ax = plt.subplot2grid(detailSize, (1, 1));
plt.semilogx([omega[0], omega[-1]], [0, 0], 'k--')
plt.semilogx([.01, .01], [-1.5, 1], 'k--')
plt.semilogx([.1, .1], [-1.5, 1], 'k--')
plt.semilogx([1, 1], [-1.5, 1], 'k--')
plt.semilogx([10, 10], [-1.5, 1], 'k--')
plt.semilogx([100, 100], [-1.5, 1], 'k--')
# now the real stuff
plt.semilogx(omega, sfExc, '%s' % modClr, label=modTxt);
plt.semilogx(omega, -sfNorm, '%s--' % modClr, label=modTxt);
plt.xlim([omega[0], omega[-1]]);
plt.ylim([-0.1, 1.1]);
plt.xlabel('spatial frequency (c/deg)', fontsize=12);
plt.ylabel('Normalized response (a.u.)', fontsize=12);
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=curr_ax, offset=5);

# SIMPLE normalization
curr_ax = plt.subplot2grid(detailSize, (2, 1));
plt.semilogx([omega[0], omega[-1]], [0, 0], 'k--')
plt.semilogx([.01, .01], [-1.5, 1], 'k--')
plt.semilogx([.1, .1], [-1.5, 1], 'k--')
plt.semilogx([1, 1], [-1.5, 1], 'k--')
plt.semilogx([10, 10], [-1.5, 1], 'k--')
plt.semilogx([100, 100], [-1.5, 1], 'k--')
# now the real stuff
if fitType == 2 or fitType == 4:
  wt_weights = np.sqrt(hf.genNormWeightsSimple(omega, gs_mean, gs_std, fitType));
  sfNormSim = wt_weights/np.amax(np.abs(wt_weights));
else:
  if fitType != 1:
    warnings.warn('yet to write code for computing/plotting simple normalization signal for this type; default to flat');
  unwt_weights = np.sqrt(hf.genNormWeightsSimple(omega, None, None));
  sfNormSim = unwt_weights/np.amax(np.abs(unwt_weights));
plt.semilogx(omega, sfExc, '%s' % modClr, label=modTxt);
plt.semilogx(omega, sfNormSim, '%s--' % modClr, label=modTxt);
plt.xlim([omega[0], omega[-1]]);
plt.ylim([-0.1, 1.1]);
plt.xlabel('spatial frequency (c/deg)', fontsize=12);
plt.ylabel('Normalized response (a.u.)', fontsize=12);
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=curr_ax, offset=5);

## organize the parameters for text output
if modRecov == 1: # compare to model recovery params if they exist
  modFits = [modFit, modParamGT];
else: # otherwise we'll just print the same thing twice...
  modFits = [modFit, modFit];

# last but not least...and not last... response nonlinearity
modExps = [x[3] for x in modFits];
curr_ax = plt.subplot2grid(detailSize, (1, 2));
plt.plot([-1, 1], [0, 0], 'k--')
plt.plot([0, 0], [-.1, 1], 'k--')
plt.plot(np.linspace(-1,1,100), np.power(np.maximum(0, np.linspace(-1,1,100)), modExps[0]), '%s-' % modClr, label=modTxt, linewidth=2);
plt.plot(np.linspace(-1,1,100), np.maximum(0, np.linspace(-1,1,100)), 'k--', linewidth=1)
plt.xlim([-1, 1]);
plt.ylim([-.1, 1]);
plt.text(0.5, 1.1, 'respExp: %.2f , %.2f' % (modExps[0], modExps[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=curr_ax, offset=5);

# print, in text, model parameters:
curr_ax = plt.subplot2grid(detailSize, (0, 4));
plt.text(0.5, 0.5, 'prefSf: %.3f|%.3f' % (modFits[0][0], modFits[1][0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.4, 'derivative order: %.3f|%.3f' % (modFits[0][1], modFits[1][1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.3, 'response scalar: %.3f|%.3f' % (modFits[0][4], modFits[1][4]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.2, 'sigma: %.3f, %.3f | %.3f, %.3f' % (np.power(10, np.float(modFits[0][2])), np.power(10, np.float(modFits[1][2])), modFits[0][2], modFits[1][2]), fontsize=12, horizontalalignment='center', verticalalignment='center');

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

for d in range(nDisps):
    # which sfs have at least one contrast presentation? within a dispersion, all cons have the same # of sfs
    v_sf_inds = hf.get_valid_sfs(expData, d, val_con_by_disp[d][0], expInd);
    n_v_sfs = len(v_sf_inds);
    n_rows = int(np.ceil(n_v_sfs/np.floor(np.sqrt(n_v_sfs)))); # make this close to a rectangle/square in arrangement (cycling through sfs)
    n_cols = int(np.ceil(n_v_sfs/n_rows));
    fCurr, rvcCurr = plt.subplots(n_rows, n_cols, figsize=(n_cols*10, n_rows*10), sharex = True, sharey = True);
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

        v_cons = val_con_by_disp[d];
        n_cons = len(v_cons);
        plot_cons = np.linspace(np.min(all_cons[v_cons]), np.max(all_cons[v_cons]), 100); # 100 steps for plotting...

	# organize (measured) responses
        resp_curr = np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, ));
        respPlt = rvcAx[plt_x][plt_y].plot(all_cons[v_cons], np.maximum(resp_curr, 0.1), '-', clip_on=False, label='data');

 	# RVC with full model fit  
        rvcAx[plt_x][plt_y].fill_between(all_cons[v_cons], modLow[d,sf_ind,v_cons], modHigh[d,sf_ind,v_cons], color=modClr, \
          alpha=0.7, clip_on=False);
        rvcAx[plt_x][plt_y].plot(all_cons[v_cons], np.maximum(modAvg[d, sf_ind, v_cons], 0.1), color=modClr, \
          alpha=0.7, clip_on=False, label=modTxt);

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

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        sns.despine(ax = rvcAx[plt_x][plt_y], offset = 10, trim=False);
        rvcAx[plt_x][plt_y].tick_params(labelsize=25, width=2, length=16, direction='out');
        rvcAx[plt_x][plt_y].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...

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

crfAx = []; fCRF = [];

for d in range(nDisps):
    
    fCurr, crfCurr = plt.subplots(1, 3, figsize=(35, 20), sharex = False, sharey = True); # left side for flat; middle for data; right side for weighted model
    fCRF.append(fCurr)
    crfAx.append(crfCurr);

    fCurr.suptitle('%s #%d' % (cellType, cellNum));

    resps_curr = [modAvg, respMean];
    labels     = [modTxt, dataTxt];

    v_sf_inds = hf.get_valid_sfs(expData, d, val_con_by_disp[d][0], expInd);
    n_v_sfs = len(v_sf_inds);

    for i in range(len(resps_curr)):
      curr_resps = resps_curr[i];
      maxResp = np.max(np.max(np.max(curr_resps[~np.isnan(curr_resps)])));

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

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      crfAx[d][i].tick_params(labelsize=15, width=1, length=8, direction='out');
      crfAx[d][i].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
      sns.despine(ax = crfAx[d][i], offset=10, trim=False);

      crfAx[d][i].set_ylabel('resp above baseline (sps)');
      crfAx[d][i].set_title('D%d: sf:all - log resp %s' % (d, labels[i]));
      crfAx[d][i].legend();

saveName = "/allSfs_cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'CRF%s/' % rvcFlag));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fCRF:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()
