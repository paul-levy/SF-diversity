# coding: utf-8

######################## To do:

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns
sns.set(style='ticks')
import helper_fcns
from scipy.stats import poisson, nbinom, norm
from scipy.stats.mstats import gmean

import pdb

import sys # so that we can import model_responses (in different folder)
import model_responses

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/Analysis/Functions/paul_plt_cluster.mplstyle');
from matplotlib import rcParams
rcParams['font.size'] = 20;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 2.5;
rcParams['axes.linewidth'] = 1.5;
rcParams['lines.markersize'] = 5;
rcParams['font.style'] = 'oblique';

which_cell = int(sys.argv[1]);
plotType = int(sys.argv[2]);
lossType = int(sys.argv[3]);
fitType = int(sys.argv[4]);
crf_fit_type = int(sys.argv[5]);
descr_fit_type = int(sys.argv[6]);
norm_sim_on = int(sys.argv[7]);
normTypeArr = [];
argInd = 8; # we've already taken 8 arguments off (function call, which_cell, plot_type, loss_type, fit_type, crf_fit_type, descr_fit_type, norm_sim_on) 
nArgsIn = len(sys.argv) - argInd; 
while nArgsIn > 0:
  normTypeArr.append(float(sys.argv[argInd]));
  nArgsIn = nArgsIn - 1;
  argInd = argInd + 1;

# at CNS
# dataPath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/recordings/';
# savePath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/analysis/';
# personal mac
#dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/analysis/structures/';
#save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/analysis/figures/';
# prince cluster
dataPath = '/home/pl1465/SF_diversity/LGN/analysis/structures/';
save_loc = '/home/pl1465/SF_diversity/LGN/analysis/figures/';

expName = 'dataList.npy'
fitBase = 'fitList_180713';

# first the fit type
if fitType == 1:
  fitSuf = '_flat';
elif fitType == 2:
  fitSuf = '_wght';
elif fitType == 3:
  fitSuf = '_c50';
# then the loss type
if lossType == 1:
  lossSuf = '_lsq.npy';
elif lossType == 2:
  lossSuf = '_sqrt.npy';
elif lossType == 3:
  lossSuf = '_poiss.npy';
elif lossType == 4:
  lossSuf = '_modPoiss.npy';

if fitType != 0 and lossType != 0:
  fitListName = str(fitBase + fitSuf + lossSuf);

if crf_fit_type == 1:
  crf_type_str = '-lsq';
elif crf_fit_type == 2:
  crf_type_str = '-sqrt';
elif crf_fit_type == 3:
  crf_type_str = '-poiss';
elif crf_fit_type == 4:
  crf_type_str = '-poissMod';

if descr_fit_type == 1:
  descr_type_str = '_lsq';
elif descr_fit_type == 2:
  descr_type_str = '_sqrt';
elif descr_fit_type == 3:
  descr_type_str = '_poiss';

if crf_fit_type == 0:
  crfFitName = [];
else:
  crfFitName = str('crfFits' + crf_type_str + '.npy');

rpt_fit = 1; # i.e. take the multi-start result
if rpt_fit:
  is_rpt = '_rpt';
else:
  is_rpt = '';

conDig = 3; # round contrast to the 3rd digit

dataList = np.load(str(dataPath + expName), encoding='latin1').item();

cellStruct = np.load(str(dataPath + dataList['unitName'][which_cell-1] + '_sfm.npy'), encoding='latin1').item();

# #### Load descriptive model fits, comp. model fits
if descr_fit_type == 0:
  descrFits = [];
else:
  descrFitName = str('descrFits' + descr_type_str + '.npy');
  descrFits = np.load(str(dataPath + descrFitName), encoding = 'latin1').item();
  descrFits = descrFits[which_cell-1]['params']; # just get this cell

if fitType == 0 or lossType == 0:
  modParamsCurr = [];
else:
  modParams = np.load(str(dataPath + fitListName), encoding= 'latin1').item();
  modParamsCurr = modParams[which_cell-1]['params'];

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

data = cellStruct['sfm']['exp']['trial'];

if modParamsCurr: # i.e. modParamsCurr isn't []
  modBlankMean = modParamsCurr[6]; # late additive noise is the baseline of the model
  ignore, modRespAll, normTypeArr = model_responses.SFMGiveBof(modParamsCurr, cellStruct, normTypeArr);
  norm_type = normTypeArr[0];
  print('norm type %d' % (norm_type));
  if norm_type == 2:
    gs_mean = normTypeArr[1]; # guaranteed to exist after call to .SFMGiveBof, if norm_type == 2
    gs_std = normTypeArr[2]; # guaranteed to exist ...
  resp, stimVals, val_con_by_disp, validByStimVal, modResp = helper_fcns.tabulate_responses(cellStruct, modRespAll);
else:
  modResp = [];
  resp, stimVals, val_con_by_disp, validByStimVal, _ = helper_fcns.tabulate_responses(cellStruct);

blankMean, blankStd, _ = helper_fcns.blankResp(cellStruct); 

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
if len(resp)>4:
  f1MeanAll = resp[4];
  f1Mean = np.reshape([np.sum(x) for x in f1MeanAll.flatten()], f1MeanAll.shape);
  f1StdAll = resp[5];
  f1Std = np.reshape([np.sqrt(np.sum(np.square(x))) for x in f1StdAll.flatten()], f1StdAll.shape);
  # why the above computation? variance adds, so we square the std to get variance of sum, sum, and take sqrt again to put back to std
  predF1mean = resp[6];
  predF1std = resp[7];
else: # if we don't have the f1, just set the plotType to 0 (just mean, i.e. f0)
  plotType = 0;

# modResp is (nFam, nSf, nCons, nReps) nReps is (currently; 2018.01.05) set to 20 to accommadate the current experiment with 10 repetitions
if modResp:
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

    fCurr.suptitle('%s #%d' % (dataList['unitType'][which_cell-1], which_cell));
    
    maxResp = np.max(np.max(respMean[d, ~np.isnan(respMean[d, :, :])]));
    maxPred = np.max(np.max(predMean[d, ~np.isnan(predMean[d, :, :])]));
    maxf1 = np.max(np.max(f1Mean[d, ~np.isnan(f1Mean[d, :, :])]));
    maxF1Pred = np.max(np.max(predF1mean[d, ~np.isnan(predF1mean[d, :, :])]));
    if plotType == 0:
      maxPlot = np.maximum(maxResp, maxPred);
    if plotType == 1:
      maxPlot = np.maximum(maxf1, maxF1Pred);
    if plotType == 2:
      maxPlot = np.maximum(np.maximum(maxResp, maxf1), np.maximum(maxPred, maxF1Pred));

    maxPlotComp = np.nanmax([np.max(x) for x in f1MeanAll[1, :, :].flatten()]);

    for c in reversed(range(n_v_cons)):
        leftLines = []; leftStr = []; # lines/string for legend of left side of plot

        c_plt_ind = len(v_cons) - c - 1;
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);        

        #dispAx[d][c_plt_ind, 1].plot(all_sfs[v_sfs], np.divide(predMean[d, v_sfs, v_cons[c]]-blankMean, respMean[d, v_sfs, v_cons[c]]-blankMean), clip_on=False);
        #dispAx[d][c_plt_ind, 1].axhline(1, clip_on=False, linestyle='dashed');
        
        # plot data (and predicted response, if dispersion > 1)
        if plotType == 0 or plotType == 2:
          respPlt = dispAx[d][c_plt_ind, 0].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]], 
                                      respStd[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);
          leftLines.append(respPlt); leftStr.append('response');
          if d>0:
            dispAx[d][c_plt_ind, 0].plot(all_sfs[v_sfs], predMean[d, v_sfs, v_cons[c]], 'b-', alpha=0.7, clip_on=False);
            predPlt = dispAx[d][c_plt_ind, 0].fill_between(all_sfs[v_sfs], predMean[d, v_sfs, v_cons[c]] - predStd[d, v_sfs, v_cons[c]],
                                             predMean[d, v_sfs, v_cons[c]] + predStd[d, v_sfs, v_cons[c]], color='b', alpha=0.2);
            leftLines.append(predPlt); leftStr.append('prediction');
        if plotType == 1 or plotType == 2:
          respPlt = dispAx[d][c_plt_ind, 0].errorbar(all_sfs[v_sfs], f1Mean[d, v_sfs, v_cons[c]], 
                                      f1Std[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);
          leftLines.append(respPlt); leftStr.append('response');
          if d>0:
            dispAx[d][c_plt_ind, 0].plot(all_sfs[v_sfs], predF1mean[d, v_sfs, v_cons[c]], 'b-', alpha=0.7, clip_on=False);
            predPlt = dispAx[d][c_plt_ind, 0].fill_between(all_sfs[v_sfs], predF1mean[d, v_sfs, v_cons[c]] - predF1std[d, v_sfs, v_cons[c]],
                                             predF1mean[d, v_sfs, v_cons[c]] + predF1std[d, v_sfs, v_cons[c]], color='b', alpha=0.2);
            leftLines.append(predPlt); leftStr.append('prediction');

        # plot descriptive model fit
        if descrFits: # i.e. descrFits isn't empty, then plot it
          curr_mod_params = descrFits[d, v_cons[c], :];
          #dispAx[d][c_plt_ind, 0].plot(sfs_plot, helper_fcns.flexible_Gauss(curr_mod_params, sfs_plot), clip_on=False)
        
	# plot model fits
        if modParamsCurr: # i.e. modParamsCurr isn't [] 
          modPlt = dispAx[d][c_plt_ind, 0].fill_between(all_sfs[v_sfs], modLow[d, v_sfs, v_cons[c]], \
                                      modHigh[d, v_sfs, v_cons[c]], color='r', alpha=0.2);
          dispAx[d][c_plt_ind, 0].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], 'r-', alpha=0.7, clip_on=False);
          leftLines.append(modPlt); leftStr.append('model resp');

        dispAx[d][c_plt_ind, 0].legend(leftLines, leftStr, loc=0);

        # if plotType == 1 or 2 (i.e. plotting f1), plot response to individual components (right column of plot)
        if d>0 and (plotType == 1 or plotType == 2):
          comps = [];
          curr_f1 = f1MeanAll[d, v_sfs, v_cons[c]]; # get the component responses only at the relevant conditions
          curr_f1_std = f1StdAll[d, v_sfs, v_cons[c]];

          n_comps = len(curr_f1[0]); # how many components per stimulus/response?

          for i in range(n_comps):
            curr_resps = [x[i] for x in curr_f1]; # go through each response "list" and get the correct component
            curr_std = [x[i] for x in curr_f1_std];
            curr_comp = dispAx[d][c_plt_ind, 1].errorbar(all_sfs[v_sfs] + norm.rvs(0, 0.1, len(curr_resps)), curr_resps, curr_std, fmt='-o', clip_on=False); # scatter the x-coordinate for better visibility
            comps.append(curr_comp[0]);
          comp_str = [str(i) for i in range(n_comps)];
          dispAx[d][c_plt_ind, 1].legend(comps, comp_str, loc=0);
            

        for i in range(2):

          dispAx[d][c_plt_ind, i].set_xlim((min(all_sfs), max(all_sfs)));
        
          dispAx[d][c_plt_ind, i].set_xscale('log');
          dispAx[d][c_plt_ind, i].set_xlabel('sf (c/deg)'); 

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
          dispAx[d][c_plt_ind, i].tick_params(labelsize=15, width=1, length=8, direction='out');
          dispAx[d][c_plt_ind, i].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...	
          sns.despine(ax=dispAx[d][c_plt_ind, i], offset=10, trim=False); 

        dispAx[d][c_plt_ind, 0].set_title('Resp: D%d, contrast: %.3f' % (d, all_cons[v_cons[c]]));
        dispAx[d][c_plt_ind, 0].set_ylim((0, 1.5*maxPlot));
        dispAx[d][c_plt_ind, 0].set_ylabel('resp (sps)');
        dispAx[d][c_plt_ind, 1].set_title('Component responses');
        dispAx[d][c_plt_ind, 1].set_ylabel('resp (sps)');
        dispAx[d][c_plt_ind, 1].set_ylim((0, 1.5*maxPlotComp));


saveName = "/cell_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'byDisp/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close();

# #### All SF tuning on one graph, split by dispersio
# left side of plots for data, right side for model predictions
fDisp = []; dispAx = [];

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);    

for d in range(nDisps):
    
    v_cons = val_con_by_disp[d];
    n_v_cons = len(v_cons);
    
    fCurr, dispCurr = plt.subplots(1, 2, figsize=(20, 20)); 
    fDisp.append(fCurr)
    dispAx.append(dispCurr);

    fCurr.suptitle('%s #%d' % (dataList['unitType'][which_cell-1], which_cell));

    for i in range(2):
    
      if i == 0:
        curr_resps = respMean;
        curr_base_f0 = blankMean;
        f1_resps = f1Mean;

        maxResp = np.max(np.max(curr_resps[d, ~np.isnan(curr_resps[d, :, :])]));
        maxf1 = np.max(np.max(f1_resps[d, ~np.isnan(f1_resps[d, :, :])]));
        if plotType == 1:
          maxResp = maxf1;
        if plotType == 2:
          maxResp = np.maximum(maxf1, maxResp);
      elif i == 1 and modResp:
        curr_resps = modAvg;
        curr_base_f0 = modBlankMean;
        maxResp = np.max(np.max(curr_resps[d, ~np.isnan(curr_resps[d, :, :])]));
      elif i == 1 and not modResp:
        continue;

      lines = [];
      linesf1 = [];
      for c in reversed(range(n_v_cons)):
          v_sfs = ~np.isnan(curr_resps[d, :, v_cons[c]]);        

          # plot data
          col = [c/float(n_v_cons), c/float(n_v_cons), c/float(n_v_cons)];
          if i == 0:
            if plotType == 0 or plotType == 2:
              respAbBaseline = curr_resps[d, v_sfs, v_cons[c]] - curr_base_f0;
              curr_line, = dispAx[d][i].plot(all_sfs[v_sfs][respAbBaseline>1e-1], respAbBaseline[respAbBaseline>1e-1], '-o', clip_on=False, color=col);
              lines.append(curr_line);
            if plotType == 1 or plotType == 2:
              curr_f1 = f1_resps[d, v_sfs, v_cons[c]];
              curr_line, = dispAx[d][i].plot(all_sfs[v_sfs][curr_f1>1e-1], curr_f1[curr_f1>1e-1], '--o', clip_on=False, color=col);
              linesf1.append(curr_line);

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
      dispAx[d][i].set_title('D%d - sf tuning' % (d));
      con_strs = [str(i) for i in reversed(all_cons[v_cons])];
      if plotType == 0:
        dispAx[d][i].legend(lines, con_strs, loc=0);
      if plotType == 1:
        dispAx[d][i].legend(linesf1, con_strs, loc=0);
      if plotType == 2:
        dispAx[d][i].legend((lines, linesf1), (con_strs, con_strs), loc=0);

saveName = "/allCons_cell_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'byDisp/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()

# #### Plot just sfMix contrasts

# i.e. highest (up to) 4 contrasts for each dispersion

mixCons = 4;
maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));
maxf1 = np.max(np.max(np.max(f1Mean[~np.isnan(f1Mean)])));
if plotType == 1:
  maxResp = maxf1;
if plotType == 2:
  maxResp = np.maximum(maxResp, maxf1);

f, sfMixAx = plt.subplots(mixCons, nDisps, figsize=(20, 15));

f.suptitle('%s #%d' % (dataList['unitType'][which_cell-1], which_cell));

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
        if plotType == 0 or plotType == 2:
          sfMixAx[c_plt_ind, d].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]], 
                                       respStd[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);
        if plotType == 1 or plotType == 2:
          sfMixAx[c_plt_ind, d].errorbar(all_sfs[v_sfs], f1Mean[d, v_sfs, v_cons[c]], 
                                       f1Std[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);

        # plot linear superposition prediction
#        sfMixAx[c_plt_ind, d].errorbar(all_sfs[v_sfs], predMean[d, v_sfs, v_cons[c]], 
#                                       predStd[d, v_sfs, v_cons[c]], fmt='p', clip_on=False);

        # plot descriptive model fit
        if descrFits:
          curr_mod_params = descrFits[d, v_cons[c], :];
          sfMixAx[c_plt_ind, d].plot(sfs_plot, helper_fcns.flexible_Gauss(curr_mod_params, sfs_plot), clip_on=False)

	# plot model fits
        if modParamsCurr:
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
	        
#########
# Plot secondary things - filter, normalization, nonlinearity, etc
#########

fDetails = [];
if modParamsCurr:

  fDetails = plt.figure();
  fDetails.set_size_inches(w=25,h=10)
  #fDetails, all_plots = plt.subplots(3,5, figsize=(25,10))
  fDetails.suptitle('%s #%d' % (dataList['unitType'][which_cell-1], which_cell)); 
 
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
  if modParamsCurr:
    prefSf    = modParamsCurr[0];
    dOrder    = modParamsCurr[1]
  prefOri = 0; # just fixed value since no model param for this
  aRatio = 1; # just fixed value since no model param for this
  filtTemp  = model_responses.oriFilt(imSizeDeg, pixSize, prefSf, prefOri, dOrder, aRatio);
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

  inhSfTuning = helper_fcns.getSuppressiveSFtuning();

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
    filtPeak = normTypeArr[4];
    stdLeft = normTypeArr[2];
    stdRight = normTypeArr[3];

    filter = setSigmaFilter(filtPeak, stdLeft, stdRight);
    offset_filt = normTypeArr[1];
    scale_filt = -(1-offset_filt); # we always scale so that range is [offset_sf, 1]
    c50_filt = helper_fcns.evalSigmaFilter(filter, scale_filt, offset_filt, stimSf)

    # now plot
    curr_ax = plt.subplot2grid(detailSize, (2, 4));
    plt.semilogx(stimSf, c50_filt);
    plt.title('(mu, stdL/R, offset) = (%.2f, %.2f|%.2f, %.2f)' % (sfPref, stdLeft, stdRight, offset_filt));
    plt.xlabel('sf (cpd)');
    plt.ylabel('c50 (con %)')

  # print, in text, model parameters:
  plt.subplot2grid(detailSize, (0, 4)); # set the current subplot location/size[default is 1x1]
  plt.text(0.5, 0.5, 'prefSf: {:.3f}'.format(modParamsCurr[0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
  plt.text(0.5, 0.4, 'derivative order: {:.3f}'.format(modParamsCurr[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
  plt.text(0.5, 0.3, 'response scalar: {:.3f}'.format(modParamsCurr[4]), fontsize=12, horizontalalignment='center', verticalalignment='center');
  plt.text(0.5, 0.2, 'sigma: {:.3f} | {:.3f}'.format(np.power(10, modParamsCurr[2]), modParamsCurr[2]), fontsize=12, horizontalalignment='center', verticalalignment='center');
  if fit_type == 4:
    varGain = modParamsCurr[7];
    plt.text(0.5, 0.1, 'varGain: {:.3f}'.format(varGain), fontsize=12, horizontalalignment='center', verticalalignment='center');

  # poisson test - mean/var for each condition (i.e. sfXdispXcon)
  curr_ax = plt.subplot2grid(detailSize, (0, 0), colspan=2, rowspan=2); # set the current subplot location/size[default is 1x1]
  val_conds = ~np.isnan(respMean);
  gt0 = np.logical_and(respMean[val_conds]>0, respStd[val_conds]>0);
  plt.loglog([0.01, 1000], [0.01, 1000], 'k--');
  plt.loglog(respMean[val_conds][gt0], np.square(respStd[val_conds][gt0]), 'o');
  # skeleton for plotting modulated poisson prediction
  if fit_type == 4: # i.e. modPoiss
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
fNorm = [];
if norm_sim_on and modParamsCurr:

    conLevels = [1, 0.75, 0.5, 0.33, 0.1];
    nCons = len(conLevels);
    sfCenters = np.logspace(-2, 2, 21); # just for now...
    #sfCenters = allSfs;
    fNorm, conDisp_plots = plt.subplots(nCons, nDisps, sharey=True, figsize=(40,30));
    fNorm.suptitle('%s #%d' % (dataList['unitType'][which_cell-1], which_cell));
    norm_sim = np.nan * np.empty((nDisps, nCons, len(sfCenters)));
    if len(modParamsCurr) < 9:
        modParamsCurr.append(helper_fcns.random_in_range([-0.35, 0.35])[0]); # enter asymmetry parameter

    # simulations
    for disp in range(nDisps):
        for conLvl in range(nCons):
          print('simulating normResp for family ' + str(disp+1) + ' and contrast ' + str(conLevels[conLvl]));
          for sfCent in range(len(sfCenters)):
              # if modParamsCurr doesn't have inhAsym parameter, add it!
              if norm_type == 2:
                unweighted = 1;
                _, _, _, normRespSimple, _ = model_responses.SFMsimulate(modParamsCurr, cellStruct, disp+1, conLevels[conLvl], sfCenters[sfCent], unweighted, normTypeArr = normTypeArr);
                nTrials = normRespSimple.shape[0];
                nInhChan = cellStruct['sfm']['mod']['normalization']['pref']['sf'];
                inhWeightMat  = helper_fcns.genNormWeights(cellStruct, nInhChan, gs_mean, gs_std, nTrials);
                normResp = np.sqrt((inhWeightMat*normRespSimple).sum(1)).transpose();
                norm_sim[disp, conLvl, sfCent] = np.mean(normResp); # take mean of the returned simulations (10 repetitions per stim. condition)
              else: # norm_type == 1 or 3:
                _, _, _, _, normResp = model_responses.SFMsimulate(modParamsCurr, cellStruct, disp+1, conLevels[conLvl], sfCenters[sfCent], normTypeArr = normTypeArr);
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
allFigs = [f];
if fDetails:
  allFigs.append(fDetails);
if fNorm:
  allFigs.append(fNorm);
saveName = "/cell_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'sfMixOnly/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for fig in range(len(allFigs)):
    pdfSv.savefig(allFigs[fig])
    plt.close(allFigs[fig])
pdfSv.close()

#########
# #### Plot contrast response functions with Naka-Rushton fits
#########

crfAx = []; fCRF = [];
fSum, crfSum = plt.subplots(nDisps, 2, figsize=(30, 30), sharex=False, sharey=False);

fCRF.append(fSum);
crfAx.append(crfSum);

fSum.suptitle('%s #%d' % (dataList['unitType'][which_cell-1], which_cell));

if crfFitName:
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
    
    fCurr.suptitle('%s #%d' % (dataList['unitType'][which_cell-1], which_cell));

    c50_sep = np.zeros((n_v_sfs, 1));
    c50_all = np.zeros((n_v_sfs, 1));

    rvc_plots = [];

    for sf in range(n_v_sfs):
        row_ind = int(sf/n_cols);
        col_ind = np.mod(sf, n_cols);
        sf_ind = v_sfs[0][sf];

        v_cons = ~np.isnan(respMean[d, sf_ind, :]);
        n_cons = sum(v_cons);
        plot_cons = np.linspace(0, np.max(all_cons[v_cons]), 100); # 100 steps for plotting...
	#plot_cons = np.linspace(np.min(all_cons[v_cons]), np.max(all_cons[v_cons]), 100); # 100 steps for plotting...

	# organize responses
        if plotType == 0 or plotType == 2:
          resps_curr = np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, ));
          resps_w_blank = np.hstack((blankMean, resps_curr));
        if plotType == 1 or plotType == 2:
          f1_curr = np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, ));

        # CRF fit
        if crfFitName:
          curr_fit_sep = crfFitsSepC50[d][sf_ind]['params'];
          curr_fit_all = crfFitsOneC50[d][sf_ind]['params'];
          # ignore varGain when reporting loss here...
          sep_pred = helper_fcns.naka_rushton(np.hstack((0, all_cons[v_cons])), curr_fit_sep[0:4]);
          all_pred = helper_fcns.naka_rushton(np.hstack((0, all_cons[v_cons])), curr_fit_all[0:4]);

          if fit_type == 4:
            r_sep, p_sep = helper_fcns.mod_poiss(sep_pred, curr_fit_sep[4]);
            r_all, p_all = helper_fcns.mod_poiss(all_pred, curr_fit_all[4]);
            sep_loss = -np.sum(loss(np.round(resps_w_blank), r_sep, p_sep));
            all_loss = -np.sum(loss(np.round(resps_w_blank), r_all, p_all));
          elif fit_type == 3:	
            sep_loss = -np.sum(loss(np.round(resps_w_blank), sep_pred));
            all_loss = -np.sum(loss(np.round(resps_w_blank), all_pred));
          else: # i.e. fit_type == 1 || == 2
            sep_loss = np.sum(loss(np.round(resps_w_blank), sep_pred));
            all_loss = np.sum(loss(np.round(resps_w_blank), all_pred));

          C50_sep[sf] = curr_fit_sep[3];
          c50_all[sf] = curr_fit_all[3];

        # summary plots
        if plotType == 0 or plotType == 2:
          curr_rvc = crfAx[0][d, 0].plot(all_cons[v_cons], resps_curr, '-', clip_on=False);
        if plotType == 1 or plotType == 2:
          curr_rvc = crfAx[0][d, 0].plot(all_cons[v_cons], f1_curr, '-', clip_on=False);
          rvc_plots.append(curr_rvc[0]);

        # NR fit plots
        if crfFitName:
          stdPts = np.hstack((0, np.reshape([respStd[d, sf_ind, v_cons]], (n_cons, ))));
          expPts = crfAx[d+1][row_ind, col_ind].errorbar(np.hstack((0, all_cons[v_cons])), resps_w_blank, stdPts, fmt='o', clip_on=False);

          sepPlt = crfAx[d+1][row_ind, col_ind].plot(plot_cons, helper_fcns.naka_rushton(plot_cons, curr_fit_sep), linestyle='dashed');
          allPlt = crfAx[d+1][row_ind, col_ind].plot(plot_cons, helper_fcns.naka_rushton(plot_cons, curr_fit_all), linestyle='dashed');
          # accompanying text...
          crfAx[d+1][row_ind, col_ind].text(0, 0.9, 'free [%.1f]: gain %.1f; c50 %.3f; exp: %.2f; base: %.1f, varGn: %.2f' % (sep_loss, curr_fit_sep[1], curr_fit_sep[3], curr_fit_sep[2], curr_fit_sep[0], curr_fit_sep[4]), 
                  horizontalalignment='left', verticalalignment='center', transform=crfAx[d+1][row_ind, col_ind].transAxes, fontsize=30);
          crfAx[d+1][row_ind, col_ind].text(0, 0.8, 'fixed [%.1f]: gain %.1f; c50 %.3f; exp: %.2f; base: %.1f, varGn: %.2f' % (all_loss, curr_fit_all[1], curr_fit_all[3], curr_fit_all[2], curr_fit_all[0], curr_fit_all[4]), 
                  horizontalalignment='left', verticalalignment='center', transform=crfAx[d+1][row_ind, col_ind].transAxes, fontsize=30);

	# legend
        if crfFitName:
          crfAx[d+1][row_ind, col_ind].legend((expPts[0], sepPlt[0], allPlt[0]), ('data', 'free c50', 'fixed c50'), fontsize='large', loc='center left')

        plt_x = d+1; plt_y = (row_ind, col_ind);

        crfAx[plt_x][plt_y].set_xscale('symlog', linthreshx=0.01); # symlog will allow us to go down to 0 
        crfAx[plt_x][plt_y].set_xlabel('contrast', fontsize='medium');
        crfAx[plt_x][plt_y].set_ylabel('resp (sps)', fontsize='medium');
        crfAx[plt_x][plt_y].set_title('D%d: sf: %.3f cpd' % (d+1, all_sfs[sf_ind]), fontsize='large');

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
    if crfFitName:
      sepC50s = crfAx[0][d, 1].plot(all_sfs[v_sfs[0]], c50_sep);
      allC50s = crfAx[0][d, 1].plot(all_sfs[v_sfs[0]], c50_all);
      maxC50 = np.maximum(np.max(c50_sep), np.max(c50_all));
      v_cons = np.array(val_con_by_disp[d]);
      sfRef = respMean[d, v_sfs[0], v_cons[-1]]; # plot highest contrast spatial frequency tuning curve
          # we normalize the sf tuning, flip upside down so it matches the profile of c50, which is lowest near peak SF preference
      invSF = crfAx[0][d, 1].plot(all_sfs[v_sfs[0]],  maxC50*(1-sfRef/np.max(sfRef)), linestyle='dashed');
      crfAx[0][d, 1].set_xlim([all_sfs[0], all_sfs[-1]]);

      crfAx[0][d, 0].set_title('D%d - all RVC' % (d), fontsize='large');
      crfAx[0][d, 0].set_xlabel('contrast', fontsize='large');
      crfAx[0][d, 0].set_ylabel('resp (sps)', fontsize='large');
      crfAx[0][d, 0].legend(rvc_plots, [str(i) for i in np.round(all_sfs[v_sfs[0]], 2)], loc='upper left');

      crfAx[0][d, 1].set_title('D%d - C50 (fixed vs free)' % (d), fontsize='large');
      crfAx[0][d, 1].set_xlabel('sf (cpd)', fontsize='large');
      crfAx[0][d, 1].set_ylabel('c50', fontsize='large');
      crfAx[0][d, 1].legend((sepC50s[0], allC50s[0], invSF[0]), ('c50 free', 'c50 fixed', 'rescaled SF tuning'), fontsize='large', loc='center left');
    
saveName = "/cell_NR_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fCRF:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()

# #### Plot contrast response functions with (full) model predictions AND Naka-Rushton 

rvcAx = []; fRVC = [];

# crfFitsSepC50 loaded above, if crf_fit_type was given at program call

for d in range(nDisps):
    
    # which sfs have at least one contrast presentation?
    v_sfs = np.where(np.sum(~np.isnan(respMean[d, :, :]), axis = 1) > 0);
    n_v_sfs = len(v_sfs[0])
    n_rows = 3; #int(np.floor(n_v_sfs/2));
    n_cols = 4; #n_v_sfs - n_rows
    fCurr, rvcCurr = plt.subplots(n_rows, n_cols, figsize=(n_cols*10, n_rows*10), sharex = True, sharey = True);
    fRVC.append(fCurr)
    rvcAx.append(rvcCurr);
    
    fCurr.suptitle('%s #%d' % (dataList['unitType'][which_cell-1], which_cell-1));

    for sf in range(n_v_sfs):
        row_ind = int(sf/n_cols);
        col_ind = np.mod(sf, n_cols);
        sf_ind = v_sfs[0][sf];
       	plt_x = d; plt_y = (row_ind, col_ind);

        v_cons = ~np.isnan(respMean[d, sf_ind, :]);
        n_cons = sum(v_cons);
        plot_cons = np.linspace(0, np.max(all_cons[v_cons]), 100); # 100 steps for plotting...
	#plot_cons = np.linspace(np.min(all_cons[v_cons]), np.max(all_cons[v_cons]), 100); # 100 steps for plotting...

	# organize responses
        if plotType == 0 or plotType == 2:
          resps_curr = np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, ));
          resps_w_blank = np.hstack((blankMean, resps_curr));
          f0Plt = rvcAx[plt_x][plt_y].plot(all_cons[v_cons], np.maximum(resps_curr, 0.1), '-', clip_on=False);
        if plotType == 1 or plotType == 2:
          f1_curr = np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, ));
          f1Plt = rvcAx[plt_x][plt_y].plot(all_cons[v_cons], np.maximum(f1_curr, 0.1), '-', clip_on=False);

        # plot data

	# RVC with full model fit
        if modParamsCurr:
          rvcAx[plt_x][plt_y].fill_between(all_cons[v_cons], modLow[d, sf_ind, v_cons], \
                                      modHigh[d, sf_ind, v_cons], color='r', alpha=0.2);
          modPlt = rvcAx[plt_x][plt_y].plot(all_cons[v_cons], np.maximum(modAvg[d, sf_ind, v_cons], 0.1), 'r-', alpha=0.7, clip_on=False);        # RVC from Naka-Rushton fit
        if crfFitName:
          curr_fit_sep = crfFitsSepC50[d][sf_ind]['params'];
          nrPlt = rvcAx[plt_x][plt_y].plot(plot_cons, helper_fcns.naka_rushton(plot_cons, curr_fit_sep), linestyle='dashed');
          #pdb.set_trace();


        # summary plots
        '''
	curr_rvc = rvcAx[0][d, 0].plot(all_cons[v_cons], resps_curr, '-', clip_on=False);
        rvc_plots.append(curr_rvc[0]);

        stdPts = np.hstack((0, np.reshape([respStd[d, sf_ind, v_cons]], (n_cons, ))));
        expPts = rvcAx[d+1][row_ind, col_ind].errorbar(np.hstack((0, all_cons[v_cons])), resps_w_blank, stdPts, fmt='o', clip_on=False);

        sepPlt = rvcAx[d+1][row_ind, col_ind].plot(plot_cons, helper_fcns.naka_rushton(plot_cons, curr_fit_sep), linestyle='dashed');
        allPlt = rvcAx[d+1][row_ind, col_ind].plot(plot_cons, helper_fcns.naka_rushton(plot_cons, curr_fit_all), linestyle='dashed');
	# accompanying legend/comments
	rvcAx[d+1][row_ind, col_ind].legend((expPts[0], sepPlt[0], allPlt[0]), ('data', 'model fits'), fontsize='large', loc='center left')
        '''

        rvcAx[plt_x][plt_y].set_xscale('symlog', linthreshx=0.01); # symlog will allow us to go down to 0 
        rvcAx[plt_x][plt_y].set_xlabel('contrast', fontsize='medium');
        rvcAx[plt_x][plt_y].set_ylabel('resp (sps)', fontsize='medium');
        rvcAx[plt_x][plt_y].set_title('D%d: sf: %.3f' % (d+1, all_sfs[sf_ind]), fontsize='large');
        
        plotList = ();
        strList = ();
        
        if plotType == 0 or plotType == 2:
          plotList = plotList + (f0Plt[0], );
          strList = strList + ('data - f0', );
        if plotType == 1 or plotType == 2:
          plotList = plotList + (f1Plt[0], );
          strList = strList + ('data - f1', );
        if modParamsCurr:
          plotList = plotList + (modPlt[0], );
          strList = strList + ('model avg', );
        if crfFitName:
          plotList = plotList + (nrPlt[0], );
          strList = strList + ('Naka-Rushton', );

        rvcAx[plt_x][plt_y].legend(plotList, strList, fontsize='large', loc='center left');
	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        sns.despine(ax = rvcAx[plt_x][plt_y], offset = 10, trim=False);
        rvcAx[plt_x][plt_y].tick_params(labelsize=25, width=2, length=16, direction='out');
        rvcAx[plt_x][plt_y].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...

saveName = "/cell_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
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

    fCurr.suptitle('%s #%d' % (dataList['unitType'][which_cell-1], which_cell));

    for i in range(2):
      
      if i == 0:
        curr_resps = respMean;
        curr_blank_f0 = blankMean;
        curr_f1 = f1Mean;
        title_str = 'data';
      elif i == 1 and modParamsCurr:
        curr_resps = modAvg;
        curr_blank_f0 = modBlankMean;
        title_str = 'model';
      elif i == 1 and not modParamsCurr:
        continue;
      maxResp = np.max(np.max(np.max(curr_resps[~np.isnan(curr_resps)])));
      if plotType == 1 or 2:
        maxf1 = np.max(np.max(np.max(curr_f1[~np.isnan(curr_f1)])));
        maxResp = np.maximum(maxResp, maxf1);

      # which sfs have at least one contrast presentation?
      v_sfs = np.where(np.sum(~np.isnan(curr_resps[d, :, :]), axis = 1) > 0); # will be the same for f1, if we're plotting that, too
      n_v_sfs = len(v_sfs[0])

      lines_log = []; lines_f1_log = [];
      for sf in range(n_v_sfs):
          sf_ind = v_sfs[0][sf];
          v_cons = ~np.isnan(curr_resps[d, sf_ind, :]);
          n_cons = sum(v_cons);

          col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];

          if i == 1 or (i ==0 and (plotType == 0 or plotType == 2)): # if we're plotting the model OR (plotting data AND f0)
            plot_resps = np.reshape([curr_resps[d, sf_ind, v_cons]], (n_cons, ));
            respAbBaseline = plot_resps-curr_blank_f0;
            line_curr, = crfAx[d][i].plot(all_cons[v_cons][respAbBaseline>1e-1], respAbBaseline[respAbBaseline>1e-1], '-o', color=col, clip_on=False);
            lines_log.append(line_curr);
          if plotType == 1 or plotType == 2:
            plot_f1 = np.reshape([curr_f1[d, sf_ind, v_cons]], (n_cons, ));
            line_curr, = crfAx[d][i].plot(all_cons[v_cons][plot_f1>1e-1], plot_f1[plot_f1>1e-1], '-o', color=col, clip_on=False);
            lines_f1_log.append(line_curr);

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
      crfAx[d][i].set_title('D%d: sf:all - log resp %s' % (d, title_str));
      if plotType == 0 or plotType == 2:
        crfAx[d][i].legend(lines_log, [str(i) for i in np.round(all_sfs[v_sfs], 2)], loc='upper left');
      if plotType == 1 or plotType == 2:
        crfAx[d][i].legend(lines_f1_log, [str(i) for i in np.round(all_sfs[v_sfs], 2)], loc='upper left');

saveName = "/allSfs_log_cell_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fCRF:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()
