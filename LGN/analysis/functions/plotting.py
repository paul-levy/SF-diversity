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
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['font.style'] = 'oblique';

plot_type = int(sys.argv[9]);
if plot_type == 0: # for standard viewing
  rcParams['font.size'] = 20;
  rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
  rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
  rcParams['lines.linewidth'] = 2.5;
  rcParams['axes.linewidth'] = 1.5;
  rcParams['lines.markersize'] = 5;
if plot_type == 1: # for illustrator editing
  rcParams['font.size'] = 35;
  rcParams['lines.linewidth'] = 6;
  rcParams['axes.linewidth'] = 4;
  rcParams['lines.markersize'] = 15;

  rcParams['xtick.major.size'] = 25
  rcParams['xtick.minor.size'] = 12
  rcParams['ytick.major.size'] = 25
  rcParams['ytick.minor.size'] = 12

  rcParams['xtick.major.width'] = 6
  rcParams['xtick.minor.width'] = 3
  rcParams['ytick.major.width'] = 6
  rcParams['ytick.minor.width'] = 3

which_cell = int(sys.argv[1]);
lossType = int(sys.argv[2]);
normType = int(sys.argv[3]);
crf_fit_type = int(sys.argv[4]);
sf_loss_type = int(sys.argv[5]); # sf tuning curve fits (Diff. of Gaussians) - what loss function
sf_DoG_model = int(sys.argv[6]); # sf tuning curve fits (Diff. of Gaussians) - what DoG parameterization
norm_sim_on = int(sys.argv[7]);
phase_dir = int(sys.argv[8]);

# personal mac
#dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/analysis/structures/';
#save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/analysis/figures/';
# prince cluster
dataPath = '/home/pl1465/SF_diversity/LGN/analysis/structures/';
save_loc = '/home/pl1465/SF_diversity/LGN/analysis/figures/';

expName = 'dataList.npy'
fitBase = 'fitList_181115';
rvcBase = 'rvcFits';
phAdvBase = 'phaseAdvanceFits';
descrBase = 'descrFits_181102';

# first the fit (or normalization) type
if normType == 1:
  fitSuf = '_flat';
elif normType == 2:
  fitSuf = '_wght';
elif normType == 3:
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

if normType != 0 and lossType != 0:
  fitListName = str(fitBase + fitSuf + lossSuf);

if crf_fit_type == 1:
  crf_type_str = '-lsq';
elif crf_fit_type == 2:
  crf_type_str = '-sqrt';
elif crf_fit_type == 3:
  crf_type_str = '-poiss';
elif crf_fit_type == 4:
  crf_type_str = '-poissMod';

if sf_loss_type == 1:
  descr_loss_str = '_poiss';
elif sf_loss_type == 2:
  descr_loss_str = '_sqrt';
elif sf_loss_type == 3:
  descr_loss_str = '_sach';
elif sf_loss_type == 4:
  descr_loss_str = '_varExpl';
if sf_DoG_model == 1:
  descr_mod_str = '_sach';
elif sf_DoG_model == 2:
  descr_mod_str = '_tony';
fLname = str(descrBase + descr_loss_str + descr_mod_str + '.npy');

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

# load data, fits
dataList = np.load(str(dataPath + expName)).item();
cellStruct = np.load(str(dataPath + dataList['unitName'][which_cell-1] + '_sfm.npy')).item();

# #### Load rvc/phase advance models; sf tuning descriptive fits
rvcName = str(dataPath + helper_fcns.fit_name(rvcBase, phase_dir));
rvcFits = helper_fcns.np_smart_load(rvcName);
phAdvName = str(dataPath + helper_fcns.fit_name(phAdvBase, phase_dir));
phAdvFits = helper_fcns.np_smart_load(phAdvName);

if sf_loss_type == 0 or sf_DoG_model == 0:
  descrFits = None;
else:
  descrFitName = str(fLname);
  descrFits = helper_fcns.np_smart_load(str(dataPath + descrFitName));
  dfVarExpl = descrFits[which_cell-1]['varExpl']; # get the variance explained for this cell
  descrFits = descrFits[which_cell-1]['params']; # just get this cell

if lossType == 0:
  modParamsCurr = [];
else:
  modParams = np.load(str(dataPath + fitListName)).item();
  modParamsCurr = modParams[which_cell-1]['params'];

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

data = cellStruct['sfm']['exp']['trial'];

if modParamsCurr: # i.e. modParamsCurr isn't []
  modBlankMean = modParamsCurr[6]; # late additive noise is the baseline of the model
  ignore, modRespAll = model_responses.SFMGiveBof(modParamsCurr, cellStruct, normType=normType);
  print('norm type %d' % (normType));
  if normType == 2: # gaussian
    gs_mean, gs_std = helper_fcns.getNormParams(modParamsCurr, normType);
  resp, stimVals, val_con_by_disp, validByStimVal, modResp = helper_fcns.tabulate_responses(data, modRespAll);
else:
  modResp = None;
  resp, stimVals, val_con_by_disp, validByStimVal, _ = helper_fcns.tabulate_responses(data);

blankMean, blankStd, _ = helper_fcns.blankResp(data); 

all_disps = stimVals[0];
all_cons = stimVals[1];
all_sfs = stimVals[2];

nCons = len(all_cons);
nSfs = len(all_sfs);
nDisps = len(all_disps);

# #### Unpack responses - only f1 stuff!
f1Mean, f1MeanByTrial, f1MeanAll, f1Pred = helper_fcns.organize_adj_responses(data, rvcFits[which_cell-1]);

# std predictions are based on unprojected responses, since these have variance (proj are all same for cond)
f1semAll = resp[5];
f1sem = np.reshape([np.sqrt(np.sum(np.square(x))) for x in f1semAll.flatten()], f1semAll.shape);
# why the above computation? variance adds, so we square the std to get variance of sum, sum, and take sqrt again to put back to std
predF1std = resp[7];

# modResp is (nFam, nSf, nCons, nReps) nReps is (currently; 2018.01.05) set to 20 to accommadate the current experiment with 10 repetitions
if modResp is not None:
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
    fCurr.subplots_adjust(wspace=0.5, hspace=0.5);
    
    maxPred = np.max(np.max(f1Pred[d, ~np.isnan(f1Pred[d, :, :])]));
    f1resps = f1Mean[d, :, :];
    maxf1 = np.max([np.max(x) for x in f1resps]);
    maxF1Pred = np.max(np.max(f1Pred[d, ~np.isnan(f1Pred[d, :, :])]));
    maxPlot = np.maximum(maxf1, maxF1Pred);
    maxPlotComp = np.nanmax([np.max(x) for x in f1MeanAll[1, :, :].flatten()]);

    for c in reversed(range(n_v_cons)):
      ### left side of plots
      leftLines = []; leftStr = []; # lines/string for legend of left side of plot
      c_plt_ind = len(v_cons) - c - 1;
      v_sfs = ~np.isnan(f1Mean[d, :, v_cons[c]]);        

      # plot data
      respPlt = dispAx[d][c_plt_ind, 0].errorbar(all_sfs[v_sfs], f1resps[v_sfs, v_cons[c]], 
                                  f1sem[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);
      leftLines.append(respPlt); leftStr.append('response');
      if d>0: # also plot predicted response if d>0
        dispAx[d][c_plt_ind, 0].plot(all_sfs[v_sfs], f1Pred[d, v_sfs, v_cons[c]], 'b-', alpha=0.7, clip_on=False);
        predPlt = dispAx[d][c_plt_ind, 0].fill_between(all_sfs[v_sfs], f1Pred[d, v_sfs, v_cons[c]] - predF1std[d, v_sfs, v_cons[c]],
                                         f1Pred[d, v_sfs, v_cons[c]] + predF1std[d, v_sfs, v_cons[c]], color='b', alpha=0.2);
        leftLines.append(predPlt); leftStr.append('prediction');

      # plot descriptive model fit
      if descrFits is not None and d == 0: # i.e. descrFits isn't empty, then plot it; descrFits only for single gratings as of 09.24.18
        curr_mod_params = descrFits[d, v_cons[c], :];
        if sf_DoG_model == 1:
          curr_mod_resp = helper_fcns.DoGsach(*curr_mod_params, stim_sf=sfs_plot)[0];
        elif sf_DoG_model == 2:
          curr_mod_resp = helper_fcns.DiffOfGauss(*curr_mod_params, stim_sf=sfs_plot)[0];
        sfs_plot = np.logspace(np.log10(np.min(all_sfs[v_sfs])), np.log10(np.max(all_sfs[v_sfs])), 100);
        descrPlt = dispAx[d][c_plt_ind, 0].plot(sfs_plot, curr_mod_resp, color='k', clip_on=False)
        leftLines.append(descrPlt[0]); leftStr.append('DoG');
        # now plot characteristic frequency!  
        char_freq = helper_fcns.dog_charFreq(curr_mod_params, sf_DoG_model);
        freqPlt = dispAx[d][c_plt_ind, 0].plot(char_freq, 1, 'v', color='k');
        leftLines.append(freqPlt[0]); leftStr.append(r'$f_c$');

      # plot model fits - FOR NOW, only for single gratings
      if modParamsCurr and d == 0: # i.e. modParamsCurr isn't [] 
        modPlt = dispAx[d][c_plt_ind, 0].fill_between(all_sfs[v_sfs], modLow[d, v_sfs, v_cons[c]], \
                                    modHigh[d, v_sfs, v_cons[c]], color='r', alpha=0.2);
        dispAx[d][c_plt_ind, 0].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], 'r-', alpha=0.7, clip_on=False);
        leftLines.append(modPlt); leftStr.append('model resp');

      dispAx[d][c_plt_ind, 0].legend(leftLines, leftStr, loc=0);

      ### right side of plots
      # plot response to individual components (right side of figure; left) 
      # AND response to components when presented individually (right side of figure; right)
      v_sfs_inds = np.where(v_sfs)[0];
      rightLines = []; rightStr = []; # lines/string for legend of right side of plot
      if d == 0:
        # plot everything again on log-log coordinates...
        respPlt = dispAx[d][c_plt_ind, 1].errorbar(all_sfs[v_sfs], f1resps[v_sfs, v_cons[c]], 
                                    f1sem[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);
        rightLines.append(respPlt); rightStr.append('response');

        # plot descriptive model fit -- and inferred characteristic frequency
        if descrFits is not None: # i.e. descrFits isn't empty, then plot it
          curr_mod_params = descrFits[d, v_cons[c], :];
          if sf_DoG_model == 1:
            curr_mod_resp = helper_fcns.DoGsach(*curr_mod_params, stim_sf=sfs_plot)[0];
          elif sf_DoG_model == 2:
            curr_mod_resp = helper_fcns.DiffOfGauss(*curr_mod_params, stim_sf=sfs_plot)[0];
          sfs_plot = np.logspace(np.log10(np.min(all_sfs[v_sfs])), np.log10(np.max(all_sfs[v_sfs])), 100);
          descrPlt = dispAx[d][c_plt_ind, 1].plot(sfs_plot, curr_mod_resp, clip_on=False)
          rightLines.append(descrPlt[0]); rightStr.append('DoG');
          # now plot characteristic frequency!  
          char_freq = helper_fcns.dog_charFreq(curr_mod_params, sf_DoG_model);
          freqPlt = dispAx[d][c_plt_ind, 1].plot(char_freq, 1, 'v', color='k');
          rightLines.append(freqPlt[0]); rightStr.append(r'$f_c$');

        # plot model fits
        if modParamsCurr: # i.e. modParamsCurr isn't [] 
          modPlt = dispAx[d][c_plt_ind, 1].fill_between(all_sfs[v_sfs], modLow[d, v_sfs, v_cons[c]], \
                                        modHigh[d, v_sfs, v_cons[c]], color='r', alpha=0.2);
          dispAx[d][c_plt_ind, 1].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], 'r-', alpha=0.7, clip_on=False);
          rightLines.append(modPlt); rightStr.append('model resp');

        dispAx[d][c_plt_ind, 1].legend(rightLines, rightStr, loc=0);
        dispAx[d][c_plt_ind, 1].set_title('log-log: %.1f%% varExpl' % dfVarExpl[d, v_cons[c]]);
        dispAx[d][c_plt_ind, 1].set_xscale('log');
        dispAx[d][c_plt_ind, 1].set_yscale('log'); # double log

      if d>0:
        xticks = np.array([]); xticklabels = np.array([]);
        for j in range(len(v_sfs_inds)):
          comps = [];

          curr_f1 = f1MeanAll[d, v_sfs_inds[j], v_cons[c]]; # get the component responses only at the relevant conditions
          curr_f1_std = f1semAll[d, v_sfs_inds[j], v_cons[c]];
          # now get the individual responses
          n_comps = all_disps[d];

          val_trials, _, _, _ = helper_fcns.get_valid_trials(data, d, v_cons[c], v_sfs_inds[j])
          isolResp, _, _, _ = helper_fcns.get_isolated_responseAdj(data, val_trials, f1MeanByTrial);

          # first, reset color cycle so that it's the same each time around
          dispAx[d][c_plt_ind, 1].set_prop_cycle(None); 
          x_pos = [j-0.25, j+0.25];
          xticks = np.append(xticks, x_pos);
          xticklabels = np.append(xticklabels, ['mix', 'isol']);

          for i in range(n_comps): # difficult to make pythonic/array, so just iterate over each component
            # NOTE: for now, we will use the response-in-mixture std for both response stds...
            curr_means = [curr_f1[i], isolResp[i][0]]; # isolResp[i] is [mean, std] --> just get mean ([0])
            curr_stds = [curr_f1_std[i], curr_f1_std[i]];
            curr_comp = dispAx[d][c_plt_ind, 1].errorbar(x_pos, curr_means, curr_stds, fmt='-o', clip_on=False);
            comps.append(curr_comp[0]);

          comp_str = [str(i) for i in range(n_comps)];
          dispAx[d][c_plt_ind, 1].set_xticks(xticks);
          dispAx[d][c_plt_ind, 1].set_xticklabels(xticklabels);
          dispAx[d][c_plt_ind, 1].legend(comps, comp_str, loc=0);
          #dispAx[d][c_plt_ind, 1].set_ylim((0, 1.5*maxPlotComp));
          dispAx[d][c_plt_ind, 1].set_title('Component responses');

      for i in range(2):
      # Set ticks out, remove top/right axis, put ticks only on bottom/left
        dispAx[d][c_plt_ind, i].tick_params(direction='out', top='off', right='off');
        dispAx[d][c_plt_ind, i].tick_params(which='minor', direction='out', top='off', right='off'); # minor ticks, too...	
        sns.despine(ax=dispAx[d][c_plt_ind, i], offset=25, trim=False); 

      dispAx[d][c_plt_ind, 0].set_xlim((min(all_sfs), max(all_sfs)));
      dispAx[d][c_plt_ind, 0].set_xscale('log');
      dispAx[d][c_plt_ind, 0].set_xlabel('spatial frequency (c/deg)'); 
      dispAx[d][c_plt_ind, 0].set_title('Resp: D%d, contrast: %.3f' % (d, all_cons[v_cons[c]]));
      #dispAx[d][c_plt_ind, 0].set_ylim((0, 1.5*maxPlot));
      dispAx[d][c_plt_ind, 0].set_ylabel('response (spikes/s)');
      dispAx[d][c_plt_ind, 1].set_ylabel('response (spikes/s)');

saveName = "/cell_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'byDisp/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close();

# #### All SF tuning on one graph, split by dispersion
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
        curr_resps = f1Mean;
        maxf1 = np.max(np.max(curr_resps[d, ~np.isnan(curr_resps[d, :, :])]));
      elif i == 1 and modResp is not None:
        curr_resps = modAvg;
      elif i == 1 and modResp is None:
        continue;

      linesf1 = [];
      for c in reversed(range(n_v_cons)):
          v_sfs = ~np.isnan(curr_resps[d, :, v_cons[c]]);        

          col = [c/float(n_v_cons), c/float(n_v_cons), c/float(n_v_cons)];
          # plot data
          if i == 0:
            curr_f1 = curr_resps[d, v_sfs, v_cons[c]];
            curr_line, = dispAx[d][i].plot(all_sfs[v_sfs][curr_f1>1e-1], curr_f1[curr_f1>1e-1], '--o', clip_on=False, color=col);
            linesf1.append(curr_line);
          # plot model
          if i == 1:
            curr_f1 = curr_resps[d, v_sfs, v_cons[c]];
            curr_line, = dispAx[d][i].plot(all_sfs[v_sfs][curr_f1>1e-1], curr_f1[curr_f1>1e-1], '--o', clip_on=False, color=col);
            linesf1.append(curr_line);

      dispAx[d][i].set_aspect('equal', 'box'); 
      dispAx[d][i].set_xlim((0.5*min(all_sfs), 1.2*max(all_sfs)));
      dispAx[d][i].set_ylim((5e-2, 1.5*maxf1));

      dispAx[d][i].set_xscale('log');
      dispAx[d][i].set_yscale('log');
      dispAx[d][i].set_xlabel('spatial frequency (c/deg)'); 

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      dispAx[d][i].tick_params(labelsize=15, width=2, length=16, direction='out');
      dispAx[d][i].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
      sns.despine(ax=dispAx[d][i], offset=10, trim=False); 

      dispAx[d][i].set_ylabel('resp (sps)');
      dispAx[d][i].set_title('D%d - sf tuning' % (d));
      con_strs = [str(i) for i in reversed(all_cons[v_cons])];
      dispAx[d][i].legend(linesf1, con_strs, loc=0);

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
maxf1 = np.max(np.max(np.max(f1Mean[~np.isnan(f1Mean)])));
maxResp = maxf1;

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
        v_sfs = ~np.isnan(f1Mean[d, :, v_cons[c]]);
        
        # plot data
        sfMixAx[c_plt_ind, d].errorbar(all_sfs[v_sfs], f1Mean[d, v_sfs, v_cons[c]], 
                                       f1sem[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);

        # plot descriptive model fit
        if descrFits is not None:
          curr_mod_params = descrFits[d, v_cons[c], :];
          if sf_DoG_model == 1:
            curr_mod_resps = helper_fcns.DoGsach(*curr_mod_params, stim_sf=sfs_plot)[0];
          elif sf_DoG_model == 2:
            curr_mod_resps = helper_fcns.DiffOfGauss(*curr_mod_params, stim_sf=sfs_plot)[0];
          sfMixAx[c_plt_ind, d].plot(sfs_plot, curr_mod_resps, clip_on=False)

	# plot model fits
        if modParamsCurr:
          sfMixAx[c_plt_ind, d].fill_between(all_sfs[v_sfs], modLow[d, v_sfs, v_cons[c]], \
                                      modHigh[d, v_sfs, v_cons[c]], color='r', alpha=0.2);
          sfMixAx[c_plt_ind, d].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], 'r-', alpha=0.7, clip_on=False);

        sfMixAx[c_plt_ind, d].set_xlim((np.min(all_sfs), np.max(all_sfs)));
        sfMixAx[c_plt_ind, d].set_ylim((0, 1.5*maxResp));
        sfMixAx[c_plt_ind, d].set_xscale('log');
        sfMixAx[c_plt_ind, d].set_xlabel('spatial frequency (c/deg)');
        sfMixAx[c_plt_ind, d].set_ylabel('response (spikes/s)');

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
  nInhChan = cellStruct['sfm']['mod']['normalization']['pref']['sf'];
  if normType == 2: # gaussian
    nTrials =  inhSfTuning.shape[0];
    inhWeight = helper_fcns.genNormWeights(cellStruct, nInhChan, gs_mean, gs_std, nTrials);
    inhWeight = inhWeight[:, :, 0]; # genNormWeights gives us weights as nTr x nFilters x nFrames - we have only one "frame" here, and all are the same
  else:
    if len(modParamsCurr) == 9: # i.e. if right number of model parameters...
      inhAsym = modParamsCurr[8];
    else:
      inhAsym = 0;
    inhWeight = [];
    
    for iP in range(len(nInhChan)):
      inhWeight = np.append(inhWeight, 1 + inhAsym * (np.log(cellStruct['sfm']['mod']['normalization']['pref']['sf'][iP]) - np.mean(np.log(cellStruct['sfm']['mod']['normalization']['pref']['sf'][iP]))));

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
  #plt.semilogx(omega, sfInh, 'r--', linewidth=2);
  plt.semilogx(omega, sfNorm, 'r-', linewidth=1);
  plt.xlim([omega[0], omega[-1]]);
  plt.ylim([-1.5, 1]);
  plt.xlabel('spatial frequency (c/deg)', fontsize=20);
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

  if normType == 3: # plot the c50 filter (i.e. effective c50 as function of SF)
    stimSf = np.logspace(-2, 2, 101);
    offset_filt, stdLeft, stdRight, filtPeak = helper_fcns.getNormParams(modParamsCurr, normType);
    filter = setSigmaFilter(filtPeak, stdLeft, stdRight);
    scale_filt = -(1-offset_filt); # we always scale so that range is [offset_sf, 1]
    c50_filt = helper_fcns.evalSigmaFilter(filter, scale_filt, offset_filt, stimSf)
    # now plot
    curr_ax = plt.subplot2grid(detailSize, (2, 4));
    plt.semilogx(stimSf, c50_filt);
    plt.title('(mu, stdL/R, offset) = (%.2f, %.2f|%.2f, %.2f)' % (sfPref, stdLeft, stdRight, offset_filt));
    plt.xlabel('spatial frequency (c/deg)');
    plt.ylabel('c50 (con %)')

  # print, in text, model parameters:
  plt.subplot2grid(detailSize, (0, 4)); # set the current subplot location/size[default is 1x1]
  plt.text(0.5, 0.5, 'prefSf: {:.3f}'.format(modParamsCurr[0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
  plt.text(0.5, 0.4, 'derivative order: {:.3f}'.format(modParamsCurr[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
  plt.text(0.5, 0.3, 'response scalar: {:.3f}'.format(modParamsCurr[4]), fontsize=12, horizontalalignment='center', verticalalignment='center');
  plt.text(0.5, 0.2, 'sigma: {:.3f} | {:.3f}'.format(np.power(10, modParamsCurr[2]), modParamsCurr[2]), fontsize=12, horizontalalignment='center', verticalalignment='center');
  if lossType == 4:
    varGain = modParamsCurr[7];
    plt.text(0.5, 0.1, 'varGain: {:.3f}'.format(varGain), fontsize=12, horizontalalignment='center', verticalalignment='center');

  # poisson test - mean/var for each condition (i.e. sfXdispXcon)
  curr_ax = plt.subplot2grid(detailSize, (0, 0), colspan=2, rowspan=2); # set the current subplot location/size[default is 1x1]
  val_conds = ~np.isnan(f1Mean);
  gt0 = np.logical_and(f1Mean[val_conds]>0, f1sem[val_conds]>0);
  plt.loglog([0.01, 1000], [0.01, 1000], 'k--');
  plt.loglog(f1Mean[val_conds][gt0], np.square(f1sem[val_conds][gt0]), 'o');
  # skeleton for plotting modulated poisson prediction
  if lossType == 4: # i.e. modPoiss
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
              if normType == 2: # gaussian weighting...
                unweighted = 1;
                _, _, _, normRespSimple, _ = model_responses.SFMsimulate(modParamsCurr, cellStruct, disp+1, conLevels[conLvl], sfCenters[sfCent], unweighted, normType = normType);
                nTrials = normRespSimple.shape[0];
                nInhChan = cellStruct['sfm']['mod']['normalization']['pref']['sf'];
                inhWeightMat  = helper_fcns.genNormWeights(cellStruct, nInhChan, gs_mean, gs_std, nTrials);
                normResp = np.sqrt((inhWeightMat*normRespSimple).sum(1)).transpose();
                norm_sim[disp, conLvl, sfCent] = np.mean(normResp); # take mean of the returned simulations (10 repetitions per stim. condition)
              else: # normType == 1 or 3:
                _, _, _, _, normResp = model_responses.SFMsimulate(modParamsCurr, cellStruct, disp+1, conLevels[conLvl], sfCenters[sfCent], normType = normType);
                norm_sim[disp, conLvl, sfCent] = np.mean(normResp); # take mean of the returned simulations (10 repetitions per stim. condition)

          if normType == 2:
            maxResp = np.max(norm_sim[disp, conLvl, :]);
            conDisp_plots[conLvl, disp].text(0.5, 0.0, 'contrast: {:.2f}, dispersion level: {:.0f}, mu|std: {:.2f}|{:.2f}'.format(conLevels[conLvl], disp+1, modParamsCurr[8], modParamsCurr[9]), fontsize=12, horizontalalignment='center', verticalalignment='center'); 
          else: # normType == 0 or 2:
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
    conDisp_plots[0, nDisps].text(0.5, 1.2, 'Normalization pool responses', fontsize=16, horizontalalignment='center', verticalalignment='center', transform=conDisp_plots[0, 2].transAxes);

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
  fits = np.load(str(dataPath + crfFitName)).item();
  crfFitsSepC50 = fits[which_cell-1][str('fits_each' + is_rpt)];
  crfFitsOneC50 = fits[which_cell-1][str('fits' + is_rpt)];

for d in range(nDisps):
    
    # which sfs have at least one contrast presentation?
    v_sfs = np.where(np.sum(~np.isnan(f1Mean[d, :, :]), axis = 1) > 0);
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

        v_cons = ~np.isnan(f1Mean[d, sf_ind, :]);
        n_cons = sum(v_cons);
        plot_cons = np.linspace(0, np.max(all_cons[v_cons]), 100); # 100 steps for plotting...
	#plot_cons = np.linspace(np.min(all_cons[v_cons]), np.max(all_cons[v_cons]), 100); # 100 steps for plotting...

	# organize responses
        f1_curr = np.reshape([f1Mean[d, sf_ind, v_cons]], (n_cons, ));

        # CRF fit
        if crfFitName:
          curr_fit_sep = crfFitsSepC50[d][sf_ind]['params'];
          curr_fit_all = crfFitsOneC50[d][sf_ind]['params'];
          # ignore varGain when reporting loss here...
          sep_pred = helper_fcns.naka_rushton(np.hstack((0, all_cons[v_cons])), curr_fit_sep[0:4]);
          all_pred = helper_fcns.naka_rushton(np.hstack((0, all_cons[v_cons])), curr_fit_all[0:4]);

          if lossType == 4:
            r_sep, p_sep = helper_fcns.mod_poiss(sep_pred, curr_fit_sep[4]);
            r_all, p_all = helper_fcns.mod_poiss(all_pred, curr_fit_all[4]);
            sep_loss = -np.sum(loss(np.round(resps_w_blank), r_sep, p_sep));
            all_loss = -np.sum(loss(np.round(resps_w_blank), r_all, p_all));
          elif lossType == 3:	
            sep_loss = -np.sum(loss(np.round(resps_w_blank), sep_pred));
            all_loss = -np.sum(loss(np.round(resps_w_blank), all_pred));
          else: # i.e. lossType == 1 || == 2
            sep_loss = np.sum(loss(np.round(resps_w_blank), sep_pred));
            all_loss = np.sum(loss(np.round(resps_w_blank), all_pred));

          C50_sep[sf] = curr_fit_sep[3];
          c50_all[sf] = curr_fit_all[3];

        # summary plots
        curr_rvc = crfAx[0][d, 0].plot(all_cons[v_cons], f1_curr, '-', clip_on=False);
        rvc_plots.append(curr_rvc[0]);

        # NR fit plots
        if crfFitName:
          stdPts = np.hstack((0, np.reshape([f1sem[d, sf_ind, v_cons]], (n_cons, ))));
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
        crfAx[plt_x][plt_y].set_ylabel('response (spikes/s)', fontsize='medium');
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
      sfRef = f1Mean[d, v_sfs[0], v_cons[-1]]; # plot highest contrast spatial frequency tuning curve
          # we normalize the sf tuning, flip upside down so it matches the profile of c50, which is lowest near peak SF preference
      invSF = crfAx[0][d, 1].plot(all_sfs[v_sfs[0]],  maxC50*(1-sfRef/np.max(sfRef)), linestyle='dashed');
      crfAx[0][d, 1].set_xlim([all_sfs[0], all_sfs[-1]]);

      crfAx[0][d, 0].set_title('D%d - all RVC' % (d), fontsize='large');
      crfAx[0][d, 0].set_xlabel('contrast', fontsize='large');
      crfAx[0][d, 0].set_ylabel('response (spikes/s)', fontsize='large');
      crfAx[0][d, 0].legend(rvc_plots, [str(i) for i in np.round(all_sfs[v_sfs[0]], 2)], loc='upper left');

      crfAx[0][d, 1].set_title('D%d - C50 (fixed vs free)' % (d), fontsize='large');
      crfAx[0][d, 1].set_xlabel('spatial frequency (c/deg)', fontsize='large');
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
    v_sfs = np.where(np.sum(~np.isnan(f1Mean[d, :, :]), axis = 1) > 0);
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

        v_cons = ~np.isnan(f1Mean[d, sf_ind, :]);
        n_cons = sum(v_cons);
        plot_cons = np.linspace(0, np.max(all_cons[v_cons]), 100); # 100 steps for plotting...
	#plot_cons = np.linspace(np.min(all_cons[v_cons]), np.max(all_cons[v_cons]), 100); # 100 steps for plotting...

	# organize responses
        f1_curr = np.reshape([f1Mean[d, sf_ind, v_cons]], (n_cons, ));
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
        rvcAx[plt_x][plt_y].set_ylabel('response (spikes/s)', fontsize='medium');
        rvcAx[plt_x][plt_y].set_title('D%d: sf: %.3f' % (d+1, all_sfs[sf_ind]), fontsize='large');
        
        plotList = ();
        strList = ();
        
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

adjFlag = ''; # are we plotting adjusted means?
if which_cell-1 in rvcFits: # TODO: BAD - currently assume that if the fit exists, it's for single gratings (i.e. dispersion 0)
  rvcFitCurr = rvcFits[which_cell-1];
else:
  rvcFitCurr = None;

for d in range(nDisps):
    
    fCurr, crfCurr = plt.subplots(1, 2, figsize=(20, 25), sharex = False, sharey = True); # left side for data, right side for model predictions
    fCRF.append(fCurr)
    crfAx.append(crfCurr);

    fCurr.suptitle('%s #%d' % (dataList['unitType'][which_cell-1], which_cell));

    for i in range(2):
      
      if i == 0:
        curr_resps = f1Mean;
        title_str = 'data';
      elif i == 1 and modParamsCurr:
        curr_resps = modAvg;
        title_str = 'model';
      elif i == 1 and not modParamsCurr:
        continue;
      maxResp = np.max(np.max(np.max(curr_resps[~np.isnan(curr_resps)])));

      # which sfs have at least one contrast presentation?
      v_sfs = np.where(np.sum(~np.isnan(curr_resps[d, :, :]), axis = 1) > 0); # will be the same for f1, if we're plotting that, too
      n_v_sfs = len(v_sfs[0])

      lines_log = []; lines_f1_log = [];
      for sf in range(n_v_sfs):
          sf_ind = v_sfs[0][sf];
          v_cons = ~np.isnan(curr_resps[d, sf_ind, :]);
          n_cons = sum(v_cons);

          col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];
          plot_f1 = f1Mean[d, sf_ind, v_cons];

          line_curr, = crfAx[d][i].plot(all_cons[v_cons][plot_f1>1e-1], plot_f1[plot_f1>1e-1], '-o', color=col, clip_on=False);
          lines_f1_log.append(line_curr);

      crfAx[d][i].set_xlim([-0.1, 1]);
      crfAx[d][i].set_ylim([-0.1*maxResp, 1.1*maxResp]);
      #crfAx[d][i].set_aspect('equal', 'box')
      #crfAx[d][i].set_xscale('log');
      #crfAx[d][i].set_yscale('log');
      #crfAx[d][i].set_xlim([1e2, 1]);
      #crfAx[d][i].set_ylim([1e-2, 1.5*maxResp]);
      crfAx[d][i].set_xlabel('contrast');

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      crfAx[d][i].tick_params(labelsize=15, width=1, length=8, direction='out');
      crfAx[d][i].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
      sns.despine(ax = crfAx[d][i], offset=10, trim=False);

      crfAx[d][i].set_ylabel('resp (adj) above baseline (sps)');
      crfAx[d][i].set_title('D%d: sf:all - log resp %s' % (d, title_str));
      crfAx[d][i].legend(lines_f1_log, [str(i) for i in np.round(all_sfs[v_sfs], 2)], loc='upper left');

saveName = "/allSfs_cell_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fCRF:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()



### SIMULATION PLOTS###
## NOTE: NOT adjusted for changes in adjusted responses - 09.30.18
# We'll simulate from the model, now

if norm_sim_on and modParamsCurr:

  # construct by hand for now
  val_con_by_disp = [];
  val_con_by_disp.append(np.array([1, 0.688, 0.473, 0.325, 0.224, 0.154, 0.106, 0.073, 0.05, 0.01]));
  val_con_by_disp.append(np.array([1, 0.688, 0.473, 0.325]));

  v_sfs = np.logspace(np.log10(np.min(all_sfs)), np.log10(np.max(all_sfs)), 11); # for now
  print('\nSimulating enhanced range of contrasts from model\n\n');
  print('\tTesting at range of spatial frequencies: ' + str(v_sfs));

  fSims = []; simsAx = [];

  # first, just plot the (normalized) excitatory filter and normalization pool response on the same plot
  # and for ease of comparison, also duplicate the SF and RVC tuning for single gratings here
  # calculations done above in fDetails (sfExc, sfNorm)
  fFilt, axCurr = plt.subplots(1, 1, figsize=(10, 10));
  fSims.append(fFilt);
  simsAx.append(axCurr);

  # plot model details - filter
  simsAx[0].semilogx([omega[0], omega[-1]], [0, 0], 'k--')
  simsAx[0].semilogx([.01, .01], [-0.1, 1], 'k--')
  simsAx[0].semilogx([.1, .1], [-0.1, 1], 'k--')
  simsAx[0].semilogx([1, 1], [-0.1, 1], 'k--')
  simsAx[0].semilogx([10, 10], [-0.1, 1], 'k--')
  simsAx[0].semilogx([100, 100], [-0.1, 1], 'k--')
  # now the real stuff
  ex = simsAx[0].semilogx(omega, sfExc, 'k-')
  nm = simsAx[0].semilogx(omega, -sfNorm, 'r-', linewidth=2.5);
  simsAx[0].set_xlim([omega[0], omega[-1]]);
  simsAx[0].set_ylim([-0.1, 1.1]);
  simsAx[0].set_xlabel('spatial frequency (c/deg)', fontsize=12);
  simsAx[0].set_ylabel('Normalized response (a.u.)', fontsize=12);
  simsAx[0].set_title('CELL %d' % (which_cell), fontsize=20);
  simsAx[0].legend([ex[0], nm[0]], ('excitatory %.2f' % (modParamsCurr[0]), 'normalization %.2f' % (np.exp(modParamsCurr[-2]))));
  # Remove top/right axis, put ticks only on bottom/left
  sns.despine(ax=simsAx[0], offset=5);

  for d in range(len(val_con_by_disp)):

      v_cons = val_con_by_disp[d];
      n_v_cons = len(v_cons);

      fCurr, dispCurr = plt.subplots(1, 2, figsize=(20, 20)); # left side for SF simulations, right side for RVC simulations
      fSims.append(fCurr)
      simsAx.append(dispCurr);

      # SF tuning - NEED TO SIMULATE
      lines = [];
      for c in reversed(range(n_v_cons)):
          curr_resps = [];
          for sf_i in v_sfs:
            print('Testing SF tuning: disp %d, con %.2f, sf %.2f' % (d+1, v_cons[c], sf_i));
            sf_iResp, _, _, _, _ = model_responses.SFMsimulate(modParamsCurr, cellStruct, d+1, v_cons[c], sf_i, normType=normType);
            curr_resps.append(sf_iResp[0]); # SFMsimulate returns array - unpack it

          # plot data
          col = [c/float(n_v_cons), c/float(n_v_cons), c/float(n_v_cons)];
          respAbBaseline = np.asarray(curr_resps);
          print('resps: %s' % respAbBaseline);
          #print('Simulated at %d|%d sfs: %d above baseline' % (len(v_sfs), len(curr_resps), sum(respAbBaseline>1e-1)));
          curr_line, = simsAx[d+1][0].plot(v_sfs[respAbBaseline>1e-1], respAbBaseline[respAbBaseline>1e-1], '-o', clip_on=False, color=col);
          lines.append(curr_line);

      simsAx[d+1][0].set_aspect('equal', 'box'); 
      simsAx[d+1][0].set_xlim((0.5*min(v_sfs), 1.2*max(v_sfs)));
      #simsAx[d+1][0].set_ylim((5e-2, 1.5*maxResp));
      simsAx[d+1][0].set_xlabel('spatial frequency (c/deg)'); 

      simsAx[d+1][0].set_ylabel('response (spikes/s)');
      simsAx[d+1][0].set_title('D%d - sf tuning' % (d));
      simsAx[d+1][0].legend(lines, [str(i) for i in reversed(v_cons)], loc=0);

      # RVCs - NEED TO SIMULATE
      n_v_sfs = len(v_sfs)

      lines_log = [];
      for sf_i in range(n_v_sfs):
          sf_curr = v_sfs[sf_i];

          curr_resps = [];
          for con_i in v_cons:
            print('Testing RVC: disp %d, con %.2f, sf %.2f' % (d+1, con_i, sf_curr));
            con_iResp, _, _, _, _ = model_responses.SFMsimulate(modParamsCurr, cellStruct, d+1, con_i, sf_curr, normType=normType);
            curr_resps.append(con_iResp[0]); # unpack the array returned by SFMsimulate

          col = [sf_i/float(n_v_sfs), sf_i/float(n_v_sfs), sf_i/float(n_v_sfs)];
          respAbBaseline = np.asarray(curr_resps);
          print('rAB = %s ||| v_cons %s' % (respAbBaseline, v_cons));
          line_curr, = simsAx[d+1][1].plot(v_cons[respAbBaseline>1e-1], respAbBaseline[respAbBaseline>1e-1], '-o', color=col, clip_on=False);
          lines_log.append(line_curr);

      simsAx[d+1][1].set_xlim([1e-2, 1]);
      #simsAx[d+1][1].set_ylim([1e-2, 1.5*maxResp]);
      simsAx[d+1][1].set_aspect('equal', 'box')
      simsAx[d+1][1].set_xscale('log');
      simsAx[d+1][1].set_yscale('log');
      simsAx[d+1][1].set_xlabel('contrast');

      simsAx[d+1][1].set_ylabel('response (spikes/s)');
      simsAx[d+1][1].set_title('D%d: sf:all - log resp' % (d));
      simsAx[d+1][1].legend(lines_log, [str(i) for i in np.round(v_sfs, 2)], loc='upper left');

      for ii in range(2):

        simsAx[d+1][ii].set_xscale('log');
        simsAx[d+1][ii].set_yscale('log');

        # Set ticks out, remove top/right axis, put ticks only on bottom/left
        simsAx[d+1][ii].tick_params(labelsize=15, width=2, length=16, direction='out');
        simsAx[d+1][ii].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
        sns.despine(ax=simsAx[d+1][ii], offset=10, trim=False); 

  # fSims must be saved separately...
  saveName = "cell_%d_simulate.pdf" % (which_cell)
  pdfSv = pltSave.PdfPages(str(save_loc + 'simulate/' + saveName));
  for ff in fSims:
      pdfSv.savefig(ff)
      plt.close(ff)
  pdfSv.close();
