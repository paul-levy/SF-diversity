# coding: utf-8
# NOTE: Unlike plot_simple.py, this file is used to plot 
# - descriptive SF tuning fit ONLY
# - RVC with Naka-Rushton fit

import os
import sys
import numpy as np
import matplotlib
import matplotlib.cm as cm
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
rvcMod    = int(sys.argv[6]);
if len(sys.argv) > 7:
  respVar = int(sys.argv[7]);
else:
  respVar = 1;

loc_base = os.getcwd() + '/';

data_loc = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + 'figures/';

### DATALIST
expName = hf.get_datalist(expDir);
### DESCRLIST
#descrBase = 'descrFits_191003';
descrBase = 'descrFits_191023';
### RVCFITS
#rvcBase = 'rvcFits_191003'; # direc flag & '.npy' are added
rvcBase = 'rvcFits_191023'; # direc flag & '.npy' are added

##################
### Spatial frequency
##################

modStr  = hf.descrMod_name(descrMod)
fLname  = hf.descrFit_name(descrLoss, descrBase=descrBase, modelName=modStr);
descrFits = hf.np_smart_load(data_loc + fLname);
if rvcAdj == 1:
  rvcFits = hf.np_smart_load(data_loc + hf.rvc_fit_name(rvcBase, modNum=rvcMod, dir=1)); # i.e. positive
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
  rvcFlag = '_f0';
else:
  rvcFlag = '';
rvcSuff = hf.rvc_mod_suff(rvcMod);
rvcBase = '%s%s' % (rvcBase, rvcFlag);
spikes_rate = hf.get_adjusted_spikerate(trialInf, cellNum, expInd, data_loc, rvcBase, rvcMod=rvcMod, descrFitName_f0 = fLname, baseline_sub=False);
# let's also get the baseline
if f1f0rat < 1 and expDir != 'LGN/': # i.e. if we're in LGN, DON'T get baseline, even if f1f0 < 1 (shouldn't happen)
  baseline_resp = hf.blankResp(trialInf, expInd, spikes=spikes_rate, spksAsRate=True)[0];
else:
  baseline_resp = None;

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
diffClr ='r';
diffTxt = 'diff';
refClr = 'm'
refTxt ='ref';

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

        # now, let's also plot the baseline, if complex cell
        if baseline_resp is not None: # i.e. complex cell
          dispAx[d][c_plt_ind, 0].axhline(baseline_resp, color=dataClr, linestyle='dashed');

        ## plot descr fit
        prms_curr = descrParams[d, v_cons[c]];
        descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod);
        dispAx[d][c_plt_ind, 0].plot(sfs_plot, descrResp, color=modClr, label='descr. fit');

        ## if flexGauss plot peak & c.o.m.
        if descrMod == 0:
          ctr = hf.sf_com(resps, sfVals);
          pSf = hf.dog_prefSf(prms_curr, dog_model=descrMod, all_sfs=all_sfs);
          dispAx[d][c_plt_ind, 0].plot(ctr, 1, linestyle='None', marker='v', label='c.o.m.', color=dataClr); # plot at y=1
          dispAx[d][c_plt_ind, 0].plot(pSf, 1, linestyle='None', marker='v', label='pSF', color=modClr); # plot at y=1
          dispAx[d][c_plt_ind, 0].legend();
        ## otherwise, let's plot the char freq.
        elif descrMod == 1 or descrMod == 2:
          char_freq = hf.dog_charFreq(prms_curr, descrMod);
          dispAx[d][c_plt_ind, 0].plot(char_freq, 1, linestyle='None', marker='v', label='$f_c$', color=dataClr); # plot at y=1
          dispAx[d][c_plt_ind, 0].legend();

        dispAx[d][c_plt_ind, 0].set_title('D%02d: contrast: %.3f' % (d+1, all_cons[v_cons[c]]));

        ### right side of plots - BASELINE SUBTRACTED IF COMPLEX CELL
        if d == 0:
          ## plot everything again on log-log coordinates...
          # first data
          if baseline_resp is not None:
            to_sub = baseline_resp;
          else:
            to_sub = np.array(0);
          dispAx[d][c_plt_ind, 1].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]] - to_sub, 
                                                     respVar[d, v_sfs, v_cons[c]], fmt='o', color=dataClr, clip_on=False, label=dataTxt);

          # plot descriptive model fit -- and inferred characteristic frequency (or peak...)
          prms_curr = descrParams[d, v_cons[c]];
          descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod);
          dispAx[d][c_plt_ind, 1].plot(sfs_plot, descrResp - to_sub, color=modClr, label='descr. fit', clip_on=False)
          if descrMod == 0:
            psf = hf.dog_prefSf(prms_curr, dog_model=descrMod);
            if psf != np.nan:
              dispAx[d][c_plt_ind, 1].plot(psf, 1, 'b', color='k', label='peak freq', clip_on=False);
          elif descrMod == 1 or descrMod == 2: # diff-of-gauss
            # now plot characteristic frequency!  
            char_freq = hf.dog_charFreq(prms_curr, descrMod);
            if char_freq != np.nan:
              dispAx[d][c_plt_ind, 1].plot(char_freq, 1, 'v', color='k', label='char. freq', clip_on=False);

          dispAx[d][c_plt_ind, 1].set_title('log-log: %.1f%% varExpl' % descrFits[cellNum-1]['varExpl'][d, v_cons[c]]);
          dispAx[d][c_plt_ind, 1].set_xscale('log');
          dispAx[d][c_plt_ind, 1].set_yscale('log'); # double log
          dispAx[d][c_plt_ind, 1].legend();

        ## Now, set things for both plots (formatting)
        for i in range(2):

          dispAx[d][c_plt_ind, i].set_xlim((min(all_sfs), max(all_sfs)));
        
          dispAx[d][c_plt_ind, i].set_xscale('log');
          dispAx[d][c_plt_ind, i].set_xlabel('sf (c/deg)'); 

	  # Set ticks out, remove top/right axis, put ticks only on bottom/left
          dispAx[d][c_plt_ind, i].tick_params(labelsize=15, width=1, length=8, direction='out');
          dispAx[d][c_plt_ind, i].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...	
          sns.despine(ax=dispAx[d][c_plt_ind, i], offset=10, trim=False); 

        dispAx[d][c_plt_ind, 0].set_ylim((np.minimum(-5, minResp-5), 1.5*maxResp));
        dispAx[d][c_plt_ind, 0].set_ylabel('resp (sps)');

    fCurr.suptitle('%s #%d (f1f0: %.2f)' % (cellType, cellNum, f1f0rat));

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
    
    fCurr, dispCurr = plt.subplots(1, 2, figsize=(35, 20), sharey=True, sharex=True);
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
        descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod);
        dispAx[d][1].plot(sfs_plot, descrResp, color=col);

    for i in range(len(dispCurr)):
      dispAx[d][i].set_xlim((0.5*min(all_sfs), 1.2*max(all_sfs)));

      dispAx[d][i].set_xscale('log');
      if expDir == 'LGN/': # we want double-log if it's the LGN!
        dispAx[d][i].set_yscale('log');
        dispAx[d][i].set_ylim((5e-2, 1.5*maxResp));
      else:
        dispAx[d][i].set_ylim((np.minimum(-5, min_resp-5), 1.5*maxResp));

      dispAx[d][i].set_xlabel('sf (c/deg)'); 

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      dispAx[d][i].tick_params(labelsize=15, width=2, length=16, direction='out');
      dispAx[d][i].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
      sns.despine(ax=dispAx[d][i], offset=10, trim=False); 

      dispAx[d][i].set_ylabel('resp above baseline (sps)');
      dispAx[d][i].set_title('D%02d - sf tuning' % (d+1));
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

        # now, let's also plot the baseline, if complex cell
        if baseline_resp is not None: # i.e. complex cell
          sfMixAx[c_plt_ind, d].axhline(baseline_resp, color=dataClr, linestyle='dashed');

        # plot descrFit
        prms_curr = descrParams[d, v_cons[c]];
        descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod);
        sfMixAx[c_plt_ind, d].plot(sfs_plot, descrResp, label=modTxt, color=modClr);

        # plot prefSF, center of mass
        ctr = hf.sf_com(resps, sfVals);
        pSf = hf.dog_prefSf(prms_curr, dog_model=descrMod, all_sfs=all_sfs);
        sfMixAx[c_plt_ind, d].plot(ctr, 1, linestyle='None', marker='v', label='c.o.m.', color=dataClr, clip_on=False); # plot at y=1
        sfMixAx[c_plt_ind, d].plot(pSf, 1, linestyle='None', marker='v', label='pSF', color=modClr, clip_on=False); # plot at y=1

        sfMixAx[c_plt_ind, d].set_xlim((np.min(all_sfs), np.max(all_sfs)));
        sfMixAx[c_plt_ind, d].set_ylim((np.minimum(-5, minResp-5), 1.25*maxResp)); # ensure that 0 is included in the range of the plot!
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
    
    fCurr.suptitle('%s #%d (f1f0 %.2f)' % (cellType, cellNum, f1f0rat));

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
        respPlt = rvcAx[plt_x][plt_y].plot(all_cons[v_cons], np.maximum(resp_curr, 0.1), '-', clip_on=False, label='data', color=dataClr);

 	# RVC descr model - TODO: Fix this discrepancy between f0 and f1 rvc structure? make both like descrFits?
        if rvcAdj == 1: # i.e. _f1 or non-"_f0" flag on rvcFits
          prms_curr = rvcFits[d]['params'][sf_ind];
        else:
          prms_curr = rvcFits['params'][d][sf_ind]; 
        if rvcMod == 1 or rvcMod == 2: # naka-rushton/peirce
          rvcResps = hf.naka_rushton(cons_plot, prms_curr)
          c50 = prms_curr[-2]; # second to last entry
        elif rvcMod == 0: # i.e. movshon form
          rvcResps = rvcModel(*prms_curr, cons_plot);
          c50 = prms_curr[-1]; # last entry is c50
        # TODO: do you want to do the max(x, 0.1)???
        rvcAx[plt_x][plt_y].plot(cons_plot, np.maximum(rvcResps, 0.1), color=modClr, \
          alpha=0.7, clip_on=False, label=modTxt);
        rvcAx[plt_x][plt_y].plot(c50, 0.1, 'v', label='c50', color=modClr, clip_on=False);
        # now, let's also plot the baseline, if complex cell
        if baseline_resp is not None: # i.e. complex cell
          rvcAx[plt_x][plt_y].axhline(baseline_resp, color='k', linestyle='dashed');

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
full_save = os.path.dirname(str(save_loc + 'CRF%s%s/' % (rvcSuff, rvcFlag)));
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
        # plot data
        plot_resp = respMean[d, sf_ind, v_cons];

        line_curr, = crfAx[d][0].plot(all_cons[v_cons][plot_resp>1e-1], plot_resp[plot_resp>1e-1], '-o', color=col, \
                                      clip_on=False, label=con_str);
        lines_log.append(line_curr);

        # now RVC model [1]
 	# RVC descr model - TODO: Fix this discrepancy between f0 and f1 rvc structure? make both like descrFits?
        if rvcAdj == 1:
          prms_curr = rvcFits[d]['params'][sf_ind];
        else: # i.e. f0 flag on the rvc fits...
          prms_curr = rvcFits['params'][d][sf_ind]; 
        if rvcMod == 0: # i.e. movshon form
          rvcResps = rvcModel(*prms_curr, cons_plot);
        elif rvcMod == 1 or rvcMod == 2: # naka-rushton (or modified version)
          rvcResps = hf.naka_rushton(cons_plot, prms_curr)
        crfAx[d][1].plot(cons_plot, np.maximum(rvcResps, 0.1), color=col, \
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
      crfAx[d][i].set_title('D%d: sf:all - log resp' % (d+1));
      crfAx[d][i].legend();

saveName = "/allSfs_cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'CRF%s%s/' % (rvcSuff, rvcFlag)));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fCRF:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()

####################################
##### difference plots #############
####################################

# #### DIFFERENCE: Plots by dispersion
# for each cell, we will plot the high contrast SF tuning curve (for that dispersion; the "reference"), and then plot the
# difference of the given contrast relative to that reference

fDisp = []; dispAx = [];

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);    
for d in range(nDisps):
    
    v_cons = val_con_by_disp[d];
    n_v_cons = len(v_cons)-1; # "-1" since we don't need to plot the highest (refernce) contrast
    
    if n_v_cons == 1:
      continue; # TODO: this is temporary hack to avoid plotting disps with just two contrasts--causes issue in indexing

    fCurr, dispCurr = plt.subplots(n_v_cons, 2, figsize=(nDisps*8, n_v_cons*8), sharey=False);
    fDisp.append(fCurr)
    dispAx.append(dispCurr);
    
    minResp = np.min(np.min(respMean[d, ~np.isnan(respMean[d, :, :])]));
    maxResp = np.max(np.max(respMean[d, ~np.isnan(respMean[d, :, :])]));

    #########
    ### get the reference tuning
    #########
    v_sfs = ~np.isnan(respMean[d, :, v_cons[-1]]); # i.e. the highest valid contrast for this dispersion
    sfRefmn  = respMean[d, v_sfs, v_cons[-1]];
    sfRefvar = respVar[d, v_sfs, v_cons[-1]]
    
    for c in reversed(range(n_v_cons)):
        c_plt_ind = n_v_cons - c - 1;
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);

        sfVals = all_sfs[v_sfs];
        resps_curr  = respMean[d, v_sfs, v_cons[c]];
        vars_curr   = respVar[d, v_sfs, v_cons[c]];

        #########
        ### left side of plots -- NOTE: for now, including lines here, since I'm avoiding descrFits
        #########
        ## plot reference tuning
        dispAx[d][c_plt_ind, 0].errorbar(sfVals, sfRefmn, sfRefvar, color=refClr, fmt='o', clip_on=False, label=refTxt, linestyle='dashed');
        # plot 'no difference' line
        dispAx[d][c_plt_ind, 0].axhline(np.array(0), linestyle='dashed', color=diffClr);
        ## then the current response...
        dispAx[d][c_plt_ind, 0].errorbar(sfVals, resps_curr,
                                         vars_curr, color=dataClr, fmt='o', clip_on=False, label=dataTxt, linestyle='dashed');
        ## now, plot the difference!
        #  note: for now, just plotting the variance for this contrast (not any difference between vars...)
        dispAx[d][c_plt_ind, 0].errorbar(sfVals, resps_curr - sfRefmn,
                                         vars_curr, color=diffClr, fmt='o', clip_on=False, label=diffTxt, linestyle='dashed');

        # now, let's also plot the baseline, if complex cell
        if baseline_resp is not None: # i.e. complex cell
          dispAx[d][c_plt_ind, 0].axhline(baseline_resp, color=dataClr, linestyle='dashed');

        '''
        ## plot descr fit
        prms_curr = descrParams[d, v_cons[c]];
        descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod);
        dispAx[d][c_plt_ind, 0].plot(sfs_plot, descrResp, color=modClr, label='descr. fit');

        ## if flexGauss plot peak & c.o.m.
        if descrMod == 0:
          ctr = hf.sf_com(resps, sfVals);
          pSf = hf.dog_prefSf(prms_curr, dog_model=descrMod, all_sfs=all_sfs);
          dispAx[d][c_plt_ind, 0].plot(ctr, 1, linestyle='None', marker='v', label='c.o.m.', color=dataClr); # plot at y=1
          dispAx[d][c_plt_ind, 0].plot(pSf, 1, linestyle='None', marker='v', label='pSF', color=modClr); # plot at y=1
        ## otherwise, let's plot the char freq.
        elif descrMod == 1 or descrMod == 2:
          char_freq = hf.dog_charFreq(prms_curr, descrMod);
          dispAx[d][c_plt_ind, 0].plot(char_freq, 1, linestyle='None', marker='v', label='$f_c$', color=dataClr); # plot at y=1
        '''

        dispAx[d][c_plt_ind, 0].legend();
        dispAx[d][c_plt_ind, 0].set_title('D%02d: contrast: %.3f' % (d+1, all_cons[v_cons[c]]));

        ### right side of plots - BASELINE SUBTRACTED IF COMPLEX CELL
        if d == 0:
          ## plot everything again on log-log coordinates...
          # first data
          if baseline_resp is not None:
            to_sub = baseline_resp;
          else:
            to_sub = np.array(0);

          ## plot reference tuning
          dispAx[d][c_plt_ind, 1].errorbar(sfVals, sfRefmn-to_sub, sfRefvar, color=refClr, fmt='o', clip_on=False, label=refTxt, linestyle='dashed');
          # and 'no effect' line (i.e. vertical "0")
          dispAx[d][c_plt_ind, 1].axhline(np.array(0), linestyle='dashed', color=diffClr);
          ## then the current response...
          dispAx[d][c_plt_ind, 1].errorbar(sfVals, resps_curr,
                                           vars_curr, color=dataClr, fmt='o', clip_on=False, label=dataTxt, linestyle='dashed');
          ## now, plot the difference!
          #  note: for now, just plotting the variance for this contrast (not any difference between vars...)
          dispAx[d][c_plt_ind, 1].errorbar(sfVals, (resps_curr-to_sub) - (sfRefmn-to_sub),
                                           vars_curr, color=diffClr, fmt='o', clip_on=False, label=diffTxt, linestyle='dashed');

          '''
          # plot descriptive model fit -- and inferred characteristic frequency (or peak...)
          prms_curr = descrParams[d, v_cons[c]];
          descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod);
          dispAx[d][c_plt_ind, 1].plot(sfs_plot, descrResp - to_sub, color=modClr, label='descr. fit', clip_on=False)
          if descrMod == 0:
            psf = hf.dog_prefSf(prms_curr, dog_model=descrMod);
            if psf != np.nan:
              dispAx[d][c_plt_ind, 1].plot(psf, 1, 'b', color='k', label='peak freq', clip_on=False);
          elif descrMod == 1 or descrMod == 2: # diff-of-gauss
            # now plot characteristic frequency!  
            char_freq = hf.dog_charFreq(prms_curr, descrMod);
            if char_freq != np.nan:
              dispAx[d][c_plt_ind, 1].plot(char_freq, 1, 'v', color='k', label='char. freq', clip_on=False);
          '''

          dispAx[d][c_plt_ind, 1].set_title('log-log');
          #dispAx[d][c_plt_ind, 1].set_title('log-log: %.1f%% varExpl' % descrFits[cellNum-1]['varExpl'][d, v_cons[c]]);
          dispAx[d][c_plt_ind, 1].set_xscale('log');
          dispAx[d][c_plt_ind, 1].set_yscale('symlog', linthreshy=1); # double log
          dispAx[d][c_plt_ind, 1].legend();

        ## Now, set things for both plots (formatting)
        for i in range(2):

          dispAx[d][c_plt_ind, i].set_xlim((min(all_sfs), max(all_sfs)));
        
          dispAx[d][c_plt_ind, i].set_xscale('log');
          dispAx[d][c_plt_ind, i].set_xlabel('sf (c/deg)'); 

	  # Set ticks out, remove top/right axis, put ticks only on bottom/left
          dispAx[d][c_plt_ind, i].tick_params(labelsize=15, width=1, length=8, direction='out');
          dispAx[d][c_plt_ind, i].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...	
          sns.despine(ax=dispAx[d][c_plt_ind, i], offset=10, trim=False); 

        #dispAx[d][c_plt_ind, 0].set_ylim((np.minimum(-5, minResp-5), 1.5*maxResp));
        dispAx[d][c_plt_ind, 0].set_ylabel('resp (sps)');

    fCurr.suptitle('%s #%d (f1f0: %.2f)' % (cellType, cellNum, f1f0rat));

saveName = "/cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'diff/byDisp%s/' % rvcFlag));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close();

# #### DIFFERENCE: All SF tuning on one graph, split by dispersion
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
  
    # reference data
    v_sfs = ~np.isnan(respMean[d, :, v_cons[-1]]);
    refSf = respMean[d, v_sfs, v_cons[-1]];
    # reference descriptive fits
    prms_curr = descrParams[d, v_cons[-1]];
    refDescr = hf.get_descrResp(prms_curr, sfs_plot, descrMod);
    # plot 'no difference' line
    dispAx[d][0].axhline(np.array(0), linestyle='dashed', color=diffClr);
    dispAx[d][1].axhline(np.array(0), linestyle='dashed', color=diffClr);

    lines = [];
    for c in reversed(range(n_v_cons)):
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);        

        # plot data [0a]
        col = [c/float(n_v_cons), c/float(n_v_cons), c/float(n_v_cons)];
        plot_resp = respMean[d, v_sfs, v_cons[c]];
        
        curr_line, = dispAx[d][0].plot(all_sfs[v_sfs][plot_resp>1e-1], plot_resp[plot_resp>1e-1], '-o', clip_on=False, \
                                       color=col, label=str(np.round(all_cons[v_cons[c]], 2)));
        lines.append(curr_line);

        # plot differences [0b]
        if c < (n_v_cons-1):
          col = [c/float(n_v_cons), c/float(n_v_cons), c/float(n_v_cons)];
          plot_resp = respMean[d, v_sfs, v_cons[c]];

          curr_line, = dispAx[d][0].plot(all_sfs[v_sfs], plot_resp - refSf, '--o', clip_on=False, \
                                         color=col, label=str(np.round(all_cons[v_cons[c]], 2)));

 
        # plot descr fit [1a]
        prms_curr = descrParams[d, v_cons[c]];
        descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod);
        dispAx[d][1].plot(sfs_plot, descrResp, color=col);

        # plot descr fit differences [1b]
        if c < (n_v_cons-1):
          dispAx[d][1].plot(sfs_plot, descrResp-refDescr, color=col, linestyle='--');

    for i in range(len(dispCurr)):
      dispAx[d][i].set_xlim((0.5*min(all_sfs), 1.2*max(all_sfs)));

      dispAx[d][i].set_xscale('log');
      if expDir == 'LGN/': # we want double-log if it's the LGN!
        dispAx[d][i].set_yscale('symlog', linthresh=1);
        #dispAx[d][i].set_ylim((5e-2, 1.5*maxResp));
      #else:
        #dispAx[d][i].set_ylim((np.minimum(-5, min_resp-5), 1.5*maxResp));

      dispAx[d][i].set_xlabel('sf (c/deg)'); 

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      dispAx[d][i].tick_params(labelsize=15, width=2, length=16, direction='out');
      dispAx[d][i].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
      sns.despine(ax=dispAx[d][i], offset=10, trim=False); 

      dispAx[d][i].set_ylabel('resp above baseline (sps)');
      dispAx[d][i].set_title('D%02d - sf tuning' % (d+1));
      dispAx[d][i].legend(); 

saveName = "/allCons_cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'diff/byDisp%s/' % rvcFlag));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()

####################################
##### joint tuning plots ###########
####################################

fDisp = []; dispAx = [];
# NOTE: for now, only plotting single gratings
for d in range(1): #nDisps

  nr, nc = 1, 3; # 3 for: data, pred-from-rvc, pred-from-sfs
  f, ax = plt.subplots(nrows=nr, ncols=nc, figsize=(nc*15, nr*10))
  fDisp.append(f); dispAx.append(ax);
    
  ### "zeroth", get the stimulus values and labels
  val_cons = val_con_by_disp[d];
  val_sfs = hf.get_valid_sfs(expData, d, val_cons[d], expInd, stimVals, validByStimVal); # just take any contrast

  xlabels = ['%.2f' % x for x in all_sfs[val_sfs]]
  ylabels = ['%.2f' % x for x in all_cons[val_cons]]

  #########
  ### first, get the data and model predictions
  #########
  ## first, the data
  #   [X, Y] is X: increasing CON ||| Y: increasing SF
  #   "fancy" indexing (must turn dispersion into array for this to work)
  #   note that we also transpose so that SF will be on the x, contrast on the y
  curr_resps = respOrg[np.ix_([d], val_sfs, val_cons)].squeeze().transpose();
  ## now, RVC model - here, each set of parameters is for a given SF
  rvcCurr = rvcFits[d]['params']
  con_steps = 100;
  plt_cons = np.geomspace(all_cons[val_cons][0], all_cons[val_cons][-1], con_steps);
  rvcResps = np.zeros((con_steps, len(val_sfs)));
#     rvcResps = np.zeros_like(curr_resps);
  for s_itr, s in enumerate(val_sfs):
    curr_params = rvcCurr[s];
    curr_cons = plt_cons;
    if rvcMod == 1 or rvcMod == 2: # naka-rushton/peirce
      rsp = hf.naka_rushton(curr_cons, curr_params)
    elif rvcMod == 0: # i.e. movshon form
      rvc_mov = hf.get_rvc_model()
      rsp = rvc_mov(*curr_params, curr_cons);
    rvcResps[range(len(curr_cons)), s_itr] = rsp;
  ## finally, descriptive SF model - here, each set of parameters is for a given contrast
  descrCurr = descrParams[d];
  sf_steps = 100;
  plt_sfs = np.geomspace(all_sfs[0], all_sfs[-1], sf_steps);
  descrResps = np.zeros((len(val_cons), sf_steps));
  for c_itr, c in enumerate(val_cons):
    curr_params = descrCurr[c];
    curr_sfs = plt_sfs;
    descrResps[c_itr, range(len(curr_sfs))] = hf.get_descrResp(curr_params, curr_sfs, descrMod);

  ovr_min = np.minimum(np.min(curr_resps), np.minimum(np.min(rvcResps), np.min(descrResps)))
  ovr_max = np.maximum(np.max(curr_resps), np.maximum(np.max(rvcResps), np.max(descrResps)))

  #########
  ### now, plot!
  #########
  # first, data
  sns.heatmap(curr_resps, vmin=ovr_min, vmax=ovr_max, xticklabels=xlabels, yticklabels=ylabels, cmap=cm.gray, ax=ax[0])
  ax[0].set_title('data')
  # then, from RVC model
  sns.heatmap(rvcResps, vmin=ovr_min, vmax=ovr_max, xticklabels=xlabels, cmap=cm.gray, ax=ax[1])
  ax[1].set_yticks(ticks=[])
  ax[1].set_title('rvc model')
  # then, from descr model
  ### third, plot predictions as given from the descriptive fits
  sns.heatmap(descrResps, vmin=ovr_min, vmax=ovr_max, yticklabels=ylabels, cmap=cm.gray, ax=ax[2])
  ax[2].set_xticks(ticks=[])
  ax[2].set_title('sf model')
  ### fourth, hacky hacky: matmul the two model matrices together
#     ooh = ovr_max*np.matmul(np.divide(rvcResps, np.max(rvcResps.flatten())), np.divide(descrResps, np.max(descrResps.flatten())))
#     sns.heatmap(ooh, vmin=ovr_min, vmax=ovr_max, yticklabels=ylabels, cmap=cm.gray, ax=ax[3])
#     ax[3].set_xticks(ticks=[]); ax[3].set_yticks(ticks=[])
#     ax[3].set_title('mult?')

  ### finally, just set some labels
  f.suptitle('Cell #%d [%s]: Joint tuning w/SF and contrast [disp %d]' % (cellNum, cellType, d+1))
  for i in range(nc):
    ax[i].set_xlabel('sf (c/deg)');
    ax[i].set_ylabel('con (%)')

saveName = "/cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'joint%s/' % rvcFlag));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close();

