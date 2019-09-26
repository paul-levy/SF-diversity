# coding: utf-8
# NOTE: Unlike plot_simple.py, this file is used to plot 
# - descriptive SF tuning fit ONLY
# - (TODO) RVC with Naka-Rushton fit

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

cellNum   = int(sys.argv[1]);
expDir    = sys.argv[2]; 
descrMod  = int(sys.argv[3]);
descrLoss = int(sys.argv[4]);
rvcAdj    = int(sys.argv[5]); # if 1, then let's load rvcFits to adjust F1, as needed
if len(sys.argv) > 6:
  respVar = int(sys.argv[6]);
else:
  respVar = 1;

loc_base = os.getcwd() + '/';

data_loc = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + 'figures/';

### DATALIST
expName = hf.get_datalist(expDir);
#expName = 'dataList.npy';
#expName = 'dataList_glx_mr.npy'
#expName = 'dataList_glx.npy'
#expName = 'dataList_mr.npy'
### DESCRLIST
descrBase = 'descrFits_190916';
#descrBase = 'descrFits_190503';
### RVCFITS
rvcBase = 'rvcFits_190916'; # direc flag & '.npy' are added

##################
### Spatial frequency
##################

modStr  = hf.descrMod_name(descrMod)
fLname  = hf.descrFit_name(descrLoss, descrBase=descrBase, modelName=modStr);
descrFits = hf.np_smart_load(data_loc + fLname);
if rvcAdj == 1:
  rvcFits = hf.np_smart_load(data_loc + hf.phase_fit_name(rvcBase + '_f1', dir=1)); # i.e. positive
else:
  rvcFits = hf.np_smart_load(data_loc + rvcBase + '_f0.npy');
rvcFits = rvcFits[cellNum-1];

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

expData  = hf.np_smart_load(str(data_loc + cellName + '_sfm.npy'));
trialInf = expData['sfm']['exp']['trial'];
expInd   = hf.get_exp_ind(data_loc, cellName)[0];

descrParams = descrFits[cellNum-1]['params'];
f1f0rat = hf.compute_f1f0(trialInf, cellNum, expInd, data_loc, descrFitName_f0=fLname)[0];

# more tabulation - stim vals, organize measured responses
overwriteSpikes = None;
_, stimVals, val_con_by_disp, validByStimVal, _ = hf.tabulate_responses(expData, expInd);
rvcModel = hf.get_rvc_model();
if rvcAdj == 0:
  rvcBase = None;
  rvcFlag = '_noAdj';
else:
  rvcFlag = '_f1';
  rvcBase = '%s%s' % (rvcBase, rvcFlag);
spikes_rate = hf.get_adjusted_spikerate(trialInf, cellNum, expInd, data_loc, rvcBase, descrFitName_f0 = fLname);
###
# now, we take into account that we fit the responses re-centered such that they are non-negative
# i.e. if the adjusted responses (in particular, baseline-subtracted F0 responses) were negative,
#   we previuosly added the lowest value (i.e. most negative response) plus an additional scalar (0.1)
#   thus making the lowest response 0.1 - the descriptive fits were made on this transformed data
#   and thus we apply the reverse of that transformation here, to those fits, before plotting
###
min_resp = np.nanmin(spikes_rate);
minThresh = 0.1;
if min_resp < 0:
  modAdj_add = - min_resp + minThresh;
else:
  modAdj_add = np.array(0);
# now get the measured responses
_, _, respOrg, respAll = hf.organize_resp(spikes_rate, trialInf, expInd, respsAsRate=True);

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

all_disps = stimVals[0];
all_cons = stimVals[1];
all_sfs = stimVals[2];

nCons = len(all_cons);
nSfs = len(all_sfs);
nDisps = len(all_disps);

# ### Plots

# set up colors, labels
modClr = 'b';
modTxt = 'descr';
dataClr = 'k';
dataTxt = 'data';

# #### Plots by dispersion

fDisp = []; dispAx = [];

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);    
for d in range(nDisps):
    
    v_cons = val_con_by_disp[d];
    n_v_cons = len(v_cons);
    
    fCurr, dispCurr = plt.subplots(n_v_cons, 2, figsize=(nDisps*8, n_v_cons*8), sharey=False);
    fDisp.append(fCurr)
    dispAx.append(dispCurr);
    
    minResp = np.min(np.min(respMean[d, ~np.isnan(respMean[d, :, :])]));
    maxResp = np.max(np.max(respMean[d, ~np.isnan(respMean[d, :, :])]));
    
    for c in reversed(range(n_v_cons)):
        c_plt_ind = len(v_cons) - c - 1;
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);        

        ### left side of plots
        sfVals = all_sfs[v_sfs];
        resps  = respMean[d, v_sfs, v_cons[c]];
        ## plot data
        dispAx[d][c_plt_ind, 0].errorbar(sfVals, resps,
                                         respVar[d, v_sfs, v_cons[c]], color=dataClr, fmt='o', clip_on=False, label=dataTxt);
        # dispAx[d][c_plt_ind, 0].axhline(blankMean, color=dataClr, linestyle='dashed', label='spon. rate'); # blank is deprecated, since either f1 or f0 as baseline subtracted

        ## plot descr fit
        prms_curr = descrParams[d, v_cons[c]];
        descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod) - modAdj_add;
        dispAx[d][c_plt_ind, 0].plot(sfs_plot, descrResp, color=modClr, label='descr. fit');

        ## plot peak & c.o.m.
        ctr = hf.sf_com(resps, sfVals);
        pSf = hf.dog_prefSf(prms_curr, dog_model=descrMod, all_sfs=all_sfs);
        dispAx[d][c_plt_ind, 0].plot(ctr, 1, linestyle='None', marker='v', label='c.o.m.', color=dataClr); # plot at y=1
        dispAx[d][c_plt_ind, 0].plot(pSf, 1, linestyle='None', marker='v', label='pSF', color=modClr); # plot at y=1
        dispAx[d][c_plt_ind, 0].legend();

        ### right side of plots
        if d == 0:
          ## plot everything again on log-log coordinates...
          # first data
          dispAx[d][c_plt_ind, 1].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]], 
                                                     respVar[d, v_sfs, v_cons[c]], fmt='o', color=dataClr, clip_on=False, label=dataTxt);

          # plot descriptive model fit -- and inferred characteristic frequency (or peak...)
          prms_curr = descrParams[d, v_cons[c]];
          descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod) - modAdj_add;
          dispAx[d][c_plt_ind, 1].plot(sfs_plot, descrResp, color=modClr, label='descr. fit', clip_on=False)
          if descrMod == 0:
            psf = hf.dog_prefSf(prms_curr, dog_model=descrMod);
            if psf != np.nan:
              dispAx[d][c_plt_ind, 1].plot(psf, 1, 'v', color='k', label='peak freq');
          elif descrMod == 1 or descrMod == 2: # diff-of-gauss
            # now plot characteristic frequency!  
            char_freq = hf.dog_charFreq(prms_curr, descrMod);
            if char_freq != np.nan:
              dispAx[d][c_plt_ind, 1].plot(char_freq, 1, 'v', color='k', label='char. freq');

          dispAx[d][c_plt_ind, 1].set_title('log-log');
          dispAx[d][c_plt_ind, 1].set_xscale('log');
          dispAx[d][c_plt_ind, 1].set_yscale('log'); # double log

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

        dispAx[d][c_plt_ind, 0].set_ylim((minResp-5, 1.5*maxResp));
        dispAx[d][c_plt_ind, 0].set_ylabel('resp (sps)');

    fCurr.suptitle('%s #%d (f1f0: %.2f), varExpl %.2f%%' % (cellType, cellNum, f1f0rat, descrFits[cellNum-1]['varExpl'][d, v_cons[c]]));

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
    
    fCurr, dispCurr = plt.subplots(1, 2, figsize=(35, 20));
    fDisp.append(fCurr)
    dispAx.append(dispCurr);

    fCurr.suptitle('%s #%d (f1f0 %.2f)' % (cellType, cellNum, f1f0rat));

    maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));  

    lines = [];
    for c in reversed(range(n_v_cons)):
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);        

        # plot data [0]
        col = [c/float(n_v_cons), c/float(n_v_cons), c/float(n_v_cons)];
        plot_resp = respMean[d, v_sfs, v_cons[c]];

        curr_line, = dispAx[d][0].plot(all_sfs[v_sfs][plot_resp>1e-1], plot_resp[plot_resp>1e-1], '-o', clip_on=False, \
                                       color=col, label=str(np.round(all_cons[v_cons[c]], 2)));
        lines.append(curr_line);
 
        # plot descr fit [1]
        prms_curr = descrParams[d, v_cons[c]];
        descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod) - modAdj_add;
        dispAx[d][1].plot(sfs_plot, descrResp, color=col);

    for i in range(len(dispCurr)):
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
      dispAx[d][i].set_title('D%02d - sf tuning');
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
        
        sfVals = all_sfs[v_sfs];
        resps  = respMean[d, v_sfs, v_cons[c]];

        # plot data
        sfMixAx[c_plt_ind, d].errorbar(sfVals, resps,
                                       respVar[d, v_sfs, v_cons[c]], fmt='o', clip_on=False, label=dataTxt, color=dataClr);

        # plot descrFit
        prms_curr = descrParams[d, v_cons[c]];
        descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod) - modAdj_add;
        sfMixAx[c_plt_ind, d].plot(sfs_plot, descrResp, label=modTxt, color=modClr);

        # plot prefSF, center of mass
        ctr = hf.sf_com(resps, sfVals);
        pSf = hf.dog_prefSf(prms_curr, dog_model=descrMod, all_sfs=all_sfs);
        sfMixAx[c_plt_ind, d].plot(ctr, 1, linestyle='None', marker='v', label='c.o.m.', color=dataClr); # plot at y=1
        sfMixAx[c_plt_ind, d].plot(pSf, 1, linestyle='None', marker='v', label='pSF', color=modClr); # plot at y=1

        sfMixAx[c_plt_ind, d].set_xlim((np.min(all_sfs), np.max(all_sfs)));
        sfMixAx[c_plt_ind, d].set_ylim((minResp-5, 1.5*maxResp));
        sfMixAx[c_plt_ind, d].set_xscale('log');
        sfMixAx[c_plt_ind, d].set_xlabel('sf (c/deg)');
        sfMixAx[c_plt_ind, d].set_ylabel('resp (sps)');

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        sfMixAx[c_plt_ind, d].tick_params(labelsize=15, width=1, length=8, direction='out');
        sfMixAx[c_plt_ind, d].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
        sns.despine(ax=sfMixAx[c_plt_ind, d], offset=10, trim=False);

f.legend();
f.suptitle('%s #%d (%s; f1f0 %.2f)' % (cellType, cellNum, cellName, f1f0rat));
	        
allFigs = [f]; 
saveName = "/cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'sfMixOnly%s/' % rvcFlag));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for fig in range(len(allFigs)):
    pdfSv.savefig(allFigs[fig])
    plt.close(allFigs[fig])
pdfSv.close()

##################
#### Response versus contrast (RVC; contrast response function, CRF)
##################

cons_plot = np.geomspace(np.min(all_cons), np.max(all_cons), 100);

# #### Plot contrast response functions with descriptive RVC model predictions

rvcAx = []; fRVC = [];

for d in range(nDisps):
    # which sfs have at least one contrast presentation? within a dispersion, all cons have the same # of sfs
    v_sf_inds = hf.get_valid_sfs(expData, d, val_con_by_disp[d][0], expInd, stimVals, validByStimVal);
    n_v_sfs = len(v_sf_inds);
    n_rows = int(np.ceil(n_v_sfs/np.floor(np.sqrt(n_v_sfs)))); # make this close to a rectangle/square in arrangement (cycling through sfs)
    n_cols = int(np.ceil(n_v_sfs/n_rows));
    fCurr, rvcCurr = plt.subplots(n_rows, n_cols, figsize=(n_cols*10, n_rows*10), sharex = True, sharey = True);
    fRVC.append(fCurr);
    rvcAx.append(rvcCurr);
    
    fCurr.suptitle('%s #%d (f1f0 %.2f)' % (cellType, cellNum-1, f1f0rat));

    for sf in range(n_v_sfs):
        row_ind = int(sf/n_cols);
        col_ind = np.mod(sf, n_cols);
        sf_ind = v_sf_inds[sf];
       	plt_x = d; 
        if n_cols > 1:
          plt_y = (row_ind, col_ind);
        else: # pyplot makes it (n_rows, ) if n_cols == 1
          plt_y = (row_ind, );

        v_cons = val_con_by_disp[d];
        n_cons = len(v_cons);

	# organize (measured) responses
        resp_curr = np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, ));
        respPlt = rvcAx[plt_x][plt_y].plot(all_cons[v_cons], np.maximum(resp_curr, 0.1), '-', clip_on=False, label='data');

 	# RVC descr model - TODO: Fix this discrepancy between f0 and f1 rvc structure? make both like descrFits?
        if rvcAdj == 1: # i.e. _f1 or non-"_f0" flag on rvcFits
          prms_curr = rvcFits[d]['params'][sf_ind];
        else:
          prms_curr = rvcFits['params'][d][sf_ind]; 
        rvcAx[plt_x][plt_y].plot(cons_plot, np.maximum(rvcModel(*prms_curr, cons_plot), 0.1), color=modClr, \
          alpha=0.7, clip_on=False, label=modTxt);
        c50 = prms_curr[-1]; # last entry is c50
        rvcAx[plt_x][plt_y].plot(c50, 0.1, 'v', label='c50', color=modClr);

        rvcAx[plt_x][plt_y].set_xscale('log', basex=10); # was previously symlog, linthreshx=0.01
        if col_ind == 0:
          rvcAx[plt_x][plt_y].set_xlim([0.01, 1]);
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
    
    fCurr, crfCurr = plt.subplots(1, 2, figsize=(35, 20), sharex = False, sharey = True);
    fCRF.append(fCurr)
    crfAx.append(crfCurr);

    fCurr.suptitle('%s #%d (f1f0 %.2f)' % (cellType, cellNum, f1f0rat));

    v_sf_inds = hf.get_valid_sfs(expData, d, val_con_by_disp[d][0], expInd, stimVals, validByStimVal);
    n_v_sfs = len(v_sf_inds);

    maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));

    lines_log = [];
    for sf in range(n_v_sfs):
        sf_ind = v_sf_inds[sf];
        v_cons = ~np.isnan(respMean[d, sf_ind, :]);
        n_cons = sum(v_cons);

        col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];
        con_str = str(np.round(all_sfs[sf_ind], 2));
        plot_resp = respMean[d, sf_ind, v_cons];

        line_curr, = crfAx[d][0].plot(all_cons[v_cons][plot_resp>1e-1], plot_resp[plot_resp>1e-1], '-o', color=col, \
                                      clip_on=False, label = con_str);
        lines_log.append(line_curr);

        # now RVC model [1]
 	# RVC descr model - TODO: Fix this discrepancy between f0 and f1 rvc structure? make both like descrFits?
        if rvcAdj == 1: # i.e. _f1 or non-"_f0" flag on rvcFits
          prms_curr = rvcFits[d]['params'][sf_ind];
        else:
          prms_curr = rvcFits['params'][d][sf_ind]; 
        crfAx[d][1].plot(cons_plot, np.maximum(rvcModel(*prms_curr, cons_plot), 0.1), color=col, \
                         clip_on=False, label = con_str);

    for i in range(len(crfCurr)):

      crfAx[d][i].set_xlim([-0.1, 1]);
      crfAx[d][i].set_ylim([-0.1*maxResp, 1.1*maxResp]);
      crfAx[d][i].set_xlabel('contrast');

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      crfAx[d][i].tick_params(labelsize=15, width=1, length=8, direction='out');
      crfAx[d][i].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
      sns.despine(ax = crfAx[d][i], offset=10, trim=False);

      crfAx[d][i].set_ylabel('resp above baseline (sps)');
      crfAx[d][i].set_title('D%d: sf:all - log resp' % d);
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

