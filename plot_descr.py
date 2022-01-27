# coding: utf-8
# NOTE: Unlike plot_simple.py, this file is used to plot 
# - descriptive SF tuning fit ONLY
# - RVC with Naka-Rushton fit
# - all the basic characterization plots in one figure

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
import time

import helper_fcns as hf
import model_responses as mod_resp

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

plotMetrCorr = 0; # plot the corr. b/t sf70 and charFreq [1] or rc [2] for each condition?
plt_sf_as_rvc = 1;
comm_S_calc = 0;

cellNum   = int(sys.argv[1]);
expDir    = sys.argv[2]; 
descrMod  = int(sys.argv[3]);
descrLoss = int(sys.argv[4]);
joint     = int(sys.argv[5]);
rvcAdj    = int(sys.argv[6]); # if 1, then let's load rvcFits to adjust F1, as needed
rvcMod    = int(sys.argv[7]);
if len(sys.argv) > 8:
  respVar = int(sys.argv[8]);
else:
  respVar = 1;
if len(sys.argv) > 9:
  isHPC = int(sys.argv[8]);
else:
  isHPC = 0;
if len(sys.argv) > 10:
  forceLog = int(sys.argv[10]); # used for byDisp/allCons_... (sf plots)
else:
  forceLog = 0;

loc_base = os.getcwd() + '/';

data_loc = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + 'figures/';

fracSig = 1;
#fracSig = 0 if expDir == 'LGN/' else 1; # we only enforce the "upper-half sigma as fraction of lower half" for V1 cells! 

### DATALIST
expName = hf.get_datalist(expDir, force_full=1);
### DESCRLIST
hpc_str = 'HPC' if isHPC else '';
descrBase = 'descrFits%s_220122e' % hpc_str;
#descrBase = 'descrFits_220103';
#descrBase = 'descrFits_211214';
#descrBase = 'descrFits_211129';
#descrBase = 'descrFits_211028';
#descrBase = 'descrFits_211020_f030'; #211005'; #210929';
#descrBase = 'descrFits_210524';
#descrBase = 'descrFits_191023'; # for V1, V1_orig, LGN
#descrBase = 'descrFits_200507'; # for altExp
### RVCFITS
#rvcBase = 'rvcFits_200507'; # direc flag & '.npy' are added
#rvcBase = 'rvcFits_191023'; # direc flag & '.npy' are adde
#rvcBase = 'rvcFits_200714'; # direc flag & '.npy' are adde
if expDir == 'LGN/':
  rvcBase = 'rvcFits%s_211108' % hpc_str;
else:
  rvcBase = 'rvcFits%s_210914' % hpc_str; # if V1?
# -- rvcAdj = -1 means, yes, load the rvcAdj fits, but with vecF1 correction rather than ph fit; so, we'll 
rvcAdjSigned = rvcAdj;
rvcAdj = np.abs(rvcAdj);

##################
### Spatial frequency
##################

modStr  = hf.descrMod_name(descrMod)
fLname  = hf.descrFit_name(descrLoss, descrBase=descrBase, modelName=modStr, joint=joint);
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
expInd   = hf.get_exp_ind(data_loc, cellName, overwriteExpName)[0];

if expInd <= 2: # if expInd <= 2, then there cannot be rvcAdj, anyway!
  rvcAdj = 0; # then we'll load just below!

if rvcAdj == 1:
  vecF1 = 1 if rvcAdjSigned==-1 else 0
  dir = 1 if rvcAdjSigned==1 else None # we dont' have pos/neg phase if vecF1
  rvcFits = hf.np_smart_load(data_loc + hf.rvc_fit_name(rvcBase, modNum=rvcMod, dir=dir, vecF1=vecF1)); # i.e. positive
  force_baseline = False; # plotting baseline will depend on F1/F0 designation
if rvcAdj == 0:
  vecF1 = None
  rvcFits = hf.np_smart_load(data_loc + rvcBase + '_f0_NR.npy');
  force_baseline = True;
rvcFits = rvcFits[cellNum-1];
expData  = hf.np_smart_load(str(data_loc + cellName + '_sfm.npy'));
trialInf = expData['sfm']['exp']['trial'];

descrParams = descrFits[cellNum-1]['params'];
f1f0rat = hf.compute_f1f0(trialInf, cellNum, expInd, data_loc, descrFitName_f0=fLname)[0];

# more tabulation - stim vals, organize measured responses
overwriteSpikes = None;
_, stimVals, val_con_by_disp, validByStimVal, _ = hf.tabulate_responses(expData, expInd);
rvcModel = hf.get_rvc_model();
if rvcAdj == 0:
  rvcFlag = '_f0';
  force_dc = True;
else:
  rvcFlag = '';
  force_dc = False;
if expDir == 'LGN/':
  force_f1 = True;
else:
  force_f1 = False;
rvcSuff = hf.rvc_mod_suff(rvcMod);
rvcBase = '%s%s' % (rvcBase, rvcFlag);

#force_dc = True

# NOTE: We pass in the rvcFits where rvcBase[name] goes, and use -1 in rvcMod to indicate that we've already loaded the fits
spikes_rate, which_measure = hf.get_adjusted_spikerate(trialInf, cellNum, expInd, data_loc, rvcFits, rvcMod=-1, descrFitName_f0 = fLname, baseline_sub=False, force_dc=force_dc, force_f1=force_f1, return_measure=True, vecF1=vecF1);
# let's also get the baseline
if force_baseline or (f1f0rat < 1 and expDir != 'LGN/'): # i.e. if we're in LGN, DON'T get baseline, even if f1f0 < 1 (shouldn't happen)
  baseline_resp = hf.blankResp(trialInf, expInd, spikes=spikes_rate, spksAsRate=True)[0];
else:
  baseline_resp = int(0);

# now get the measured responses
#print('HERE!!!');
_, _, respOrg, respAll = hf.organize_resp(spikes_rate, trialInf, expInd, respsAsRate=True);
#pdb.set_trace();

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
    
    fCurr, dispCurr = plt.subplots(n_v_cons, 2, figsize=(2*10, n_v_cons*12), sharey=False);
    fDisp.append(fCurr)
    dispAx.append(dispCurr);    

    minResp = np.min(np.min(respMean[d, ~np.isnan(respMean[d, :, :])]));
    maxResp = np.max(np.max(respMean[d, ~np.isnan(respMean[d, :, :])]));
    
    for c in reversed(range(n_v_cons)):
        c_plt_ind = len(v_cons) - c - 1;
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);        

        currClr = [(n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons)];

        ### left side of plots
        sfVals = all_sfs[v_sfs];
        resps  = respMean[d, v_sfs, v_cons[c]];
        ## plot data
        dispAx[d][c_plt_ind, 0].errorbar(sfVals, resps,
                                         respVar[d, v_sfs, v_cons[c]], color=currClr, fmt='o', clip_on=False, label=dataTxt);

        # now, let's also plot the baseline, if complex cell
        if baseline_resp > 0: # i.e. complex cell
          dispAx[d][c_plt_ind, 0].axhline(baseline_resp, color=currClr, linestyle='dashed');

        ## plot descr fit
        prms_curr = descrParams[d, v_cons[c]];
        descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod, baseline=baseline_resp, fracSig=fracSig);
        dispAx[d][c_plt_ind, 0].plot(sfs_plot, descrResp, color=currClr, label='descr. fit');

        ## if flexGauss plot peak & frac of peak
        frac_freq = hf.sf_highCut(prms_curr, descrMod, frac=peakFrac, sfRange=(0.1, 15), baseline_sub=baseline_resp);
        if not hf.is_mod_DoG(descrMod): # i.e. non DoG models
          #ctr = hf.sf_com(resps, sfVals);
          pSf = hf.descr_prefSf(prms_curr, dog_model=descrMod, all_sfs=all_sfs);
          for ii in range(2):
            dispAx[d][c_plt_ind, ii].plot(frac_freq, 2, linestyle='None', marker='v', label='(%.2f) highCut(%.1f)' % (peakFrac, frac_freq), color=currClr, alpha=1); # plot at y=1
            #dispAx[d][c_plt_ind, ii].plot(pSf, 1, linestyle='None', marker='v', label='pSF', color=currClr, alpha=1); # plot at y=1
        ## otherwise, let's plot the char freq. and frac of peak
        elif hf.is_mod_DoG(descrMod): # (single) DoG models
          char_freq = hf.dog_charFreq(prms_curr, descrMod);
          # if it's a DoG, let's also put the parameters in text (left side only)
          dispAx[d][c_plt_ind, 0].text(0.05, 0.075, '%d,%.2f' % (*prms_curr[0:2], ), transform=dispAx[d][c_plt_ind,0].transAxes, horizontalalignment='left', fontsize='small', verticalalignment='bottom');
          dispAx[d][c_plt_ind, 0].text(0.05, 0.025, '%.2f,%.2f' % (*prms_curr[2:], ), transform=dispAx[d][c_plt_ind,0].transAxes, horizontalalignment='left', fontsize='small', verticalalignment='bottom');
          for ii in range(2):
            dispAx[d][c_plt_ind, ii].plot(frac_freq, 2, linestyle='None', marker='v', label='(%.2f) highCut(%.1f)' % (peakFrac, frac_freq), color=currClr, alpha=1); # plot at y=1
            #dispAx[d][c_plt_ind, ii].plot(char_freq, 1, linestyle='None', marker='v', label='$f_c$', color=currClr, alpha=1); # plot at y=1

        dispAx[d][c_plt_ind, 0].set_title('D%02d: contrast: %d%%' % (d+1, 100*all_cons[v_cons[c]]));

        ### right side of plots - BASELINE SUBTRACTED IF COMPLEX CELL
        if d >= 0:
          minResp_toPlot = 1e-0;
          ## plot everything again on log-log coordinates...
          # first data
          if baseline_resp > 0: # is not None
            to_sub = baseline_resp;
          else:
            to_sub = np.array(0);
          resps_curr = respMean[d, v_sfs, v_cons[c]] - to_sub;
          abvThresh = [resps_curr>minResp_toPlot];
          var_curr = respVar[d, v_sfs, v_cons[c]][abvThresh];
          dispAx[d][c_plt_ind, 1].errorbar(all_sfs[v_sfs][abvThresh], resps_curr[abvThresh], var_curr, 
                fmt='o', color=currClr, clip_on=False, markersize=9, label=dataTxt);

          # plot descriptive model fit -- and inferred characteristic frequency (or peak...)
          prms_curr = descrParams[d, v_cons[c]];
          descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod, baseline=baseline_resp, fracSig=fracSig);
          descr_curr = descrResp - to_sub;
          abvThresh = [descr_curr>minResp_toPlot]
          dispAx[d][c_plt_ind, 1].plot(sfs_plot[abvThresh], descr_curr[abvThresh], color=currClr, label='descr. fit', clip_on=False)
          if not hf.is_mod_DoG(descrMod):
            psf = hf.descr_prefSf(prms_curr, dog_model=descrMod);
            #if psf != np.nan: 
            #  dispAx[d][c_plt_ind, 1].plot(psf, 1, 'b', color='k', label='peak freq', clip_on=False);
          elif hf.is_mod_DoG(descrMod): # diff-of-gauss
            # now plot characteristic frequency!  
            char_freq = hf.dog_charFreq(prms_curr, descrMod);
            #if char_freq != np.nan:
            #  dispAx[d][c_plt_ind, 1].plot(char_freq, 1, 'v', color='k', label='char. freq', clip_on=False);

          dispAx[d][c_plt_ind, 1].set_title('log-log: %.1f%% varExpl' % descrFits[cellNum-1]['varExpl'][d, v_cons[c]], fontsize='medium');
          dispAx[d][c_plt_ind, 1].set_xscale('log');
          dispAx[d][c_plt_ind, 1].set_yscale('log'); # double log
          dispAx[d][c_plt_ind, 1].set_ylim((minResp_toPlot, 1.5*maxResp));
          dispAx[d][c_plt_ind, 1].set_aspect('equal');

        ## Now, set things for both plots (formatting)
        for i in range(2):

          dispAx[d][c_plt_ind, i].set_xlim((min(all_sfs), max(all_sfs)));
          if min(all_sfs) == max(all_sfs):
            print('cell % has bad sfs' % cellNum);
        
          dispAx[d][c_plt_ind, i].set_xscale('log');
          if c_plt_ind == len(v_cons)-1:
            dispAx[d][c_plt_ind, i].set_xlabel('sf (c/deg)'); 

	  # Set ticks out, remove top/right axis, put ticks only on bottom/left
          #dispAx[d][c_plt_ind, i].tick_params(labelsize=lblSize, width=majWidth, direction='out');
          #dispAx[d][c_plt_ind, i].tick_params(width=minWidth, which='minor', direction='out'); # minor ticks, too...	
          sns.despine(ax=dispAx[d][c_plt_ind, i], offset=10, trim=False); 

        dispAx[d][c_plt_ind, 0].set_ylim((np.minimum(-5, minResp-5), 1.5*maxResp));
        dispAx[d][c_plt_ind, 0].set_ylabel('resp (sps)');

    fCurr.suptitle('%s #%d (f1f0: %.2f)' % (cellType, cellNum, f1f0rat));
    fCurr.subplots_adjust(wspace=0.1, top=0.95);

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

minResp_toPlot = 1e-0;
  
for d in range(nDisps):
    
    v_cons = val_con_by_disp[d];
    n_v_cons = len(v_cons);
    
    fCurr, dispCurr = plt.subplots(1, 2, figsize=(35, 20), sharey=True, sharex=True);
    fDisp.append(fCurr)
    dispAx.append(dispCurr);

    fCurr.suptitle('%s #%d (f1f0 %.2f)' % (cellType, cellNum, f1f0rat));

    maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));  

    minToPlot = 5e-1;

    lines = [];
    for c in reversed(range(n_v_cons)):
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);        

        # plot data [0]
        col = [(n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons)];
        plot_resp = respMean[d, v_sfs, v_cons[c]];
        if forceLog == 1:
          if baseline_resp > 0: #is not None:
            to_sub = baseline_resp;
          else:
            to_sub = np.array(0);
          plot_resp = plot_resp - to_sub;

        curr_line, = dispAx[d][0].plot(all_sfs[v_sfs][plot_resp>minToPlot], plot_resp[plot_resp>minToPlot], '-o', clip_on=False, \
                                       color=col, label='%s%%' % (str(int(100*np.round(all_cons[v_cons[c]], 2)))));
        lines.append(curr_line);
 
        # plot descr fit [1]
        prms_curr = descrParams[d, v_cons[c]];
        descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod, baseline=baseline_resp, fracSig=fracSig);
        dispAx[d][1].plot(sfs_plot, descrResp-to_sub, color=col);

    for i in range(len(dispCurr)):
      dispAx[d][i].set_xlim((0.5*min(all_sfs), 1.2*max(all_sfs)));

      dispAx[d][i].set_xscale('log');
      if expDir == 'LGN/' or forceLog == 1: # we want double-log if it's the LGN!
        dispAx[d][i].set_yscale('log');
        #dispAx[d][i].set_ylim((minToPlot, 1.5*maxResp));
        dispAx[d][i].set_ylim((5e-1, 300)); # common y axis for ALL plots
        logSuffix = 'log_';
      else:
        dispAx[d][i].set_ylim((np.minimum(-5, minResp-5), 1.5*maxResp));
        logSuffix = '';

      dispAx[d][i].set_xlabel('sf (c/deg)'); 

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      #dispAx[d][i].tick_params(labelsize=15, width=2, length=16, direction='out');
      #dispAx[d][i].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
      sns.despine(ax=dispAx[d][i], offset=10, trim=False); 

      dispAx[d][i].set_ylabel('resp above baseline (sps)');
      dispAx[d][i].set_title('D%02d - sf tuning' % (d+1));
      dispAx[d][i].legend(fontsize='large'); 

saveName = "/allCons_%scell_%03d.pdf" % (logSuffix, cellNum)
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

f, sfMixAx = plt.subplots(mixCons, nDisps, figsize=(nDisps*9, mixCons*8));

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);

for d in range(nDisps):
    v_cons = np.array(val_con_by_disp[d]);
    n_v_cons = len(v_cons);
    v_cons = v_cons[np.arange(np.maximum(0, n_v_cons -mixCons), n_v_cons)]; # max(1, .) for when there are fewer contrasts than 4
    n_v_cons = len(v_cons);
    
    for c in reversed(range(n_v_cons)):
        c_plt_ind = n_v_cons - c - 1;
        sfMixAx[c_plt_ind, d].set_title('con: %s%%' % str(int(100*(np.round(all_cons[v_cons[c]], 2)))));
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);
        
        sfVals = all_sfs[v_sfs];
        resps  = respMean[d, v_sfs, v_cons[c]];

        # plot data
        if c_plt_ind == 0 and d==0: # only make the legend here
          sfMixAx[c_plt_ind, d].errorbar(sfVals, resps,
                                       respVar[d, v_sfs, v_cons[c]], fmt='o', clip_on=False, label=dataTxt, color=dataClr);
        else:
          sfMixAx[c_plt_ind, d].errorbar(sfVals, resps,
                                       respVar[d, v_sfs, v_cons[c]], fmt='o', clip_on=False, color=dataClr);

        # now, let's also plot the baseline, if complex cell
        if baseline_resp > 0: #is not None: # i.e. complex cell
          sfMixAx[c_plt_ind, d].axhline(baseline_resp, color=dataClr, linestyle='dashed');

        # plot descrFit
        prms_curr = descrParams[d, v_cons[c]];
        descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod, baseline=baseline_resp, fracSig=fracSig);
        if c_plt_ind == 0 and d==0: # only make the legend here
          sfMixAx[c_plt_ind, d].plot(sfs_plot, descrResp, label=modTxt, color=modClr);
        else:
          sfMixAx[c_plt_ind, d].plot(sfs_plot, descrResp, color=modClr);

        # plot prefSF, center of mass
        ctr = hf.sf_com(resps, sfVals);
        pSf = hf.descr_prefSf(prms_curr, dog_model=descrMod, all_sfs=all_sfs);
        if c_plt_ind == 0 and d==0: # only make the legend here
          sfMixAx[c_plt_ind, d].plot(ctr, 1, linestyle='None', marker='v', label='c.o.m.', color=dataClr, clip_on=False); # plot at y=1
          if pSf > 0.1 and pSf < 10:
            sfMixAx[c_plt_ind, d].plot(pSf, 1, linestyle='None', marker='v', label='pSF', color=modClr, clip_on=False); # plot at y=1
        else:
          sfMixAx[c_plt_ind, d].plot(ctr, 1, linestyle='None', marker='v', color=dataClr, clip_on=False); # plot at y=1
          if pSf > 0.1 and pSf < 10:
            sfMixAx[c_plt_ind, d].plot(pSf, 1, linestyle='None', marker='v', color=modClr, clip_on=False); # plot at y=1

        sfMixAx[c_plt_ind, d].set_xlim((np.min(all_sfs), np.max(all_sfs)));
        sfMixAx[c_plt_ind, d].set_ylim((np.minimum(-5, minResp-5), 1.25*maxResp)); # ensure that 0 is included in the range of the plot!
        sfMixAx[c_plt_ind, d].set_xscale('log');
        if d == 0:
          if c_plt_ind == 0:
            sfMixAx[c_plt_ind, d].set_ylabel('resp (sps)');
          if c_plt_ind == mixCons-1:
            sfMixAx[c_plt_ind, d].set_xlabel('sf (c/deg)');

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        #sfMixAx[c_plt_ind, d].tick_params(labelsize=15, width=1, length=8, direction='out');
        #sfMixAx[c_plt_ind, d].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
        sns.despine(ax=sfMixAx[c_plt_ind, d], offset=10, trim=False);

f.legend();
f.suptitle('%s #%d (%s; f1f0 %.2f)' % (cellType, cellNum, cellName, f1f0rat));
#f.tight_layout(rect=[0, 0, 1, 0.97])
	        
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

cons_plot = np.geomspace(np.minimum(0.01, all_cons), np.max(all_cons), 100); # go down to at least 1% contrast
#cons_plot = np.geomspace(np.min(all_cons), np.max(all_cons), 100);

# #### Plot contrast response functions with descriptive RVC model predictions

rvcAx = []; fRVC = [];

for d in range(nDisps):
    # which sfs have at least one contrast presentation? within a dispersion, all cons have the same # of sfs
    v_sf_inds = hf.get_valid_sfs(expData, d, val_con_by_disp[d][0], expInd, stimVals, validByStimVal);
    n_v_sfs = len(v_sf_inds);
    n_rows = int(np.ceil(n_v_sfs/np.floor(np.sqrt(n_v_sfs)))); # make this close to a rectangle/square in arrangement (cycling through sfs)
    n_cols = int(np.ceil(n_v_sfs/n_rows));
    fCurr, rvcCurr = plt.subplots(n_rows, n_cols, figsize=(n_cols*12, n_rows*12), sharey=True);
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
        var_curr  = np.reshape([respVar[d, sf_ind, v_cons]], (n_cons, ));

        if forceLog == 1:
          if baseline_resp > 0: #is not None:
            to_sub = baseline_resp;
          else:
            to_sub = np.array(0);
          resp_curr = resp_curr - to_sub;

        rvcAx[plt_x][plt_y].errorbar(all_cons[v_cons][resp_curr>minResp_toPlot], resp_curr[resp_curr>minResp_toPlot], var_curr[resp_curr>minResp_toPlot], fmt='o', linestyle='-', clip_on=False, label='data', markersize=9, color=dataClr);

 	# RVC descr model - TODO: Fix this discrepancy between f0 and f1 rvc structure? make both like descrFits?
        # NOTE: changing split of accessing rvcFits based on rvcAdj
        if rvcAdj == 1: # i.e. _f1 or non-"_f0" flag on rvcFits
          prms_curr = rvcFits[d]['params'][sf_ind];
        else:
          prms_curr = rvcFits['params'][d, sf_ind, :]; 
        c50 = hf.get_c50(rvcMod, prms_curr); # second to last entry
        if rvcMod == 1 or rvcMod == 2: # naka-rushton/peirce
          rvcResps = hf.naka_rushton(cons_plot, prms_curr)
        elif rvcMod == 0: # i.e. movshon form
          rvcResps = rvcModel(*prms_curr, cons_plot);

        if forceLog == 1:
           rvcResps = rvcResps - to_sub;
        val_inds = np.where(rvcResps>minResp_toPlot)[0];

        rvcAx[plt_x][plt_y].plot(cons_plot[val_inds], rvcResps[val_inds], color=modClr, \
          alpha=0.7, clip_on=False, label=modTxt);
        rvcAx[plt_x][plt_y].plot(c50, 1.5*minResp_toPlot, 'v', label='c50', color=modClr, clip_on=False);
        # now, let's also plot the baseline, if complex cell
        if baseline_resp > 0 and forceLog != 1: # i.e. complex cell (baseline_resp is not None) previously
          rvcAx[plt_x][plt_y].axhline(baseline_resp, color='k', linestyle='dashed');

        rvcAx[plt_x][plt_y].set_xscale('log', basex=10); # was previously symlog, linthreshx=0.01
        if col_ind == 0:
          rvcAx[plt_x][plt_y].set_xlabel('contrast', fontsize='medium');
          rvcAx[plt_x][plt_y].set_ylabel('response (spikes/s)', fontsize='medium');
          rvcAx[plt_x][plt_y].legend();

        # set axis limits...
        rvcAx[plt_x][plt_y].set_xlim([0.01, 1]);
        if forceLog == 1:
          rvcAx[plt_x][plt_y].set_ylim((minResp_toPlot, 1.25*maxResp));
          rvcAx[plt_x][plt_y].set_yscale('log'); # double log
          rvcAx[plt_x][plt_y].set_aspect('equal'); 
     
        try:
          curr_varExpl = rvcFits[d]['varExpl'][sf_ind] if rvcAdj else rvcFits['varExpl'][d, sf_ind];
        except:
          curr_varExpl = np.nan;
        rvcAx[plt_x][plt_y].set_title('D%d: sf: %.3f [vE=%.2f%%]' % (d+1, all_sfs[sf_ind], curr_varExpl), fontsize='large');
        if rvcMod == 0:
          try:
            cg = rvcFits[d]['conGain'][sf_ind] if rvcAdj else rvcFits['conGain'][d, sf_ind];
            rvcAx[plt_x][plt_y].text(0, 0.95, 'conGain=%.1f' % cg, transform=rvcAx[plt_x][plt_y].transAxes, horizontalalignment='left', fontsize='small', verticalalignment='top');
          except:
             pass; # not essential...

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        sns.despine(ax = rvcAx[plt_x][plt_y], offset = 10, trim=False);
        #rvcAx[plt_x][plt_y].tick_params(labelsize=lblSize, width=majWidth, direction='out');
        #rvcAx[plt_x][plt_y].tick_params(width=minWidth, which='minor', direction='out'); # minor ticks, too...

    fCurr.tight_layout(rect=[0, 0.03, 1, 0.95])

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
    
    nrow, ncol = 1, 2+plt_sf_as_rvc;

    fCurr, crfCurr = plt.subplots(nrow, ncol, figsize=(ncol*17.5, nrow*20), sharex=False, sharey='row');
    fCRF.append(fCurr)
    crfAx.append(crfCurr);

    fCurr.suptitle('%s #%d (f1f0 %.2f)' % (cellType, cellNum, f1f0rat));

    v_sf_inds = hf.get_valid_sfs(expData, d, val_con_by_disp[d][0], expInd, stimVals, validByStimVal);
    n_v_sfs = len(v_sf_inds);

    maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));
    minResp_plot = 1e-0;

    lines_log = [];
    for sf in range(n_v_sfs):
        sf_ind = v_sf_inds[sf];
        v_cons = ~np.isnan(respMean[d, sf_ind, :]);
        n_cons = sum(v_cons);

        col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];
        con_str = str(np.round(all_sfs[sf_ind], 2));
        # plot data
        plot_resp = respMean[d, sf_ind, v_cons];
        if forceLog == 1:
          if baseline_resp > 0: #is not None:
            to_sub = baseline_resp;
          else:
            to_sub = np.array(0);
          plot_resp = plot_resp - to_sub;

        line_curr, = crfAx[d][0].plot(all_cons[v_cons][plot_resp>minResp_plot], plot_resp[plot_resp>minResp_plot], '-o', color=col, \
                                      clip_on=False, markersize=9, label=con_str);
        lines_log.append(line_curr);
        crfAx[d][0].set_title('D%d: RVC data' % (d+1));

        # now RVC model [1]
        if rvcAdj == 1:
          prms_curr = rvcFits[d]['params'][sf_ind];
        else: # i.e. f0 flag on the rvc fits...
          prms_curr = rvcFits['params'][d, sf_ind, :]; 
        if rvcMod == 0: # i.e. movshon form
          rvcResps = rvcModel(*prms_curr, cons_plot);
        elif rvcMod == 1 or rvcMod == 2: # naka-rushton (or modified version)
          rvcResps = hf.naka_rushton(cons_plot, prms_curr)

        rvcRespsAdj = rvcResps-to_sub;
        crfAx[d][1].plot(cons_plot[rvcRespsAdj>minResp_plot], rvcRespsAdj[rvcRespsAdj>minResp_plot], color=col, \
                         clip_on=False, label = con_str);
        crfAx[d][1].set_title('D%d: RVC fits' % (d+1));

        # OPTIONAL, plot RVCs as inferred from SF tuning fits - from 22.01.19 onwards
        if plt_sf_as_rvc:
            cons = all_cons[v_cons];
            try:
                resps_curr = np.array([hf.get_descrResp(descrParams[0, vc], all_sfs[sf_ind], descrMod, baseline=baseline_resp, fracSig=fracSig) for vc in np.where(v_cons)[0]]) - to_sub;
                crfAx[d][2].plot(all_cons[v_cons], resps_curr, color=col, \
                         clip_on=False, linestyle='--', marker='o');
                crfAx[d][2].set_title('D%d: RVC from SF fit' % (d+1));
            except:
                pass # this is not essential...
            

    for i in range(len(crfCurr)):

      if expDir == 'LGN/' or forceLog == 1: # then plot as double-log
        crfAx[d][i].set_xscale('log');
        crfAx[d][i].set_yscale('log');
        crfAx[d][i].set_ylim((minResp_plot, 1.5*maxResp));
        crfAx[d][i].set_aspect('equal');
        #crfAx[d][i].set_ylim((minResp_plot, 300)); # common y axis for ALL plots
        logSuffix = 'log_';
      else:
        crfAx[d][i].set_xlim([-0.1, 1]);
        crfAx[d][i].set_ylim([-0.1*maxResp, 1.1*maxResp]);
        logSuffix = '';
      crfAx[d][i].set_xlabel('contrast');

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      #crfAx[d][i].tick_params(labelsize=lblSize, width=majWidth, direction='out');
      #crfAx[d][i].tick_params(width=minWidth, which='minor', direction='out'); # minor ticks, too...
      sns.despine(ax = crfAx[d][i], offset=10, trim=False);

      crfAx[d][i].set_ylabel('resp above baseline (sps)');
      crfAx[d][i].legend();

saveName = "/allSfs_%scell_%03d.pdf" % (logSuffix, cellNum)
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

'''

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
          dispAx[d][1].plot(sfs_plot, np.subtract(descrResp, refDescr), color=col, linestyle='--');

    for i in range(len(dispCurr)):
      dispAx[d][i].set_xlim((0.5*min(all_sfs), 1.2*max(all_sfs)));

      dispAx[d][i].set_xscale('log');
      if expDir == 'LGN/': # we want double-log if it's the LGN!
        dispAx[d][i].set_yscale('symlog', linthresh=1);
        #dispAx[d][i].set_ylim((5e-2, 1.5*maxResp));
      #else:
        #dispAx[d][i].set_ylim((np.minimum(-5, minResp-5), 1.5*maxResp));

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

'''

####################################
##### joint tuning plots ###########
####################################

fDisp = []; dispAx = [];
# NOTE: for now, only plotting single gratings
for d in range(1): #nDisps

  nr, nc = 1, 3; # 3 for: data, pred-from-rvc, pred-from-sfs
  f, ax = plt.subplots(nrows=nr, ncols=nc, figsize=(nc*25, nr*20))
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
  rvcCurr = rvcFits[d]['params'] if rvcAdj == 1 else rvcFits['params'][d, :,:];

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
    descrResps[c_itr, range(len(curr_sfs))] = hf.get_descrResp(curr_params, curr_sfs, descrMod, baseline=baseline_resp, fracSig=fracSig);

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

# #### If we have boots, plot the correlation between sf70 and char. freq [only 0 disp, for now; 21.10.23]

curr_fits = descrFits[cellNum-1];
check_boot = 'boot_params' in curr_fits;

if plotMetrCorr>0 and check_boot and (descrMod == 3 or descrMod == 1): # i.e. DoG or d-DoG-S

  fDisp = []; dispAx = [];

  for d in range(1): #nDisps): # only do single gratings, as of 21.10.23

      v_cons = val_con_by_disp[d];
      n_v_cons = len(v_cons);

      n_rows = int(np.ceil(n_v_cons/np.floor(np.sqrt(n_v_cons)))); # make this close to a rectangle/square in arrangement (cycling through cons)
      n_cols = int(np.ceil(n_v_cons/n_rows));

      fCurr, dispCurr = plt.subplots(n_rows, n_cols, figsize=(n_cols*15, n_rows*15), sharex=True, sharey=True);
      fDisp.append(fCurr)
      dispAx.append(dispCurr);    

      for c in range(n_v_cons):

          row_ind = int(c/n_cols);
          col_ind = np.mod(c, n_cols);

          if plotMetrCorr == 1:
            cfs = curr_fits['boot_charFreq'][:,d,v_cons[c]]
            nconds = len(cfs);
          elif plotMetrCorr == 2:
            rcs = curr_fits['boot_params'][:,d,v_cons[c], 1]; # get all of the characteristic radii
            nconds = len(rcs);
          sf70s = np.array([hf.sf_highCut(curr_fits['boot_params'][x,d,v_cons[c]], sfMod=descrMod, frac=0.7) for x in range(nconds)]);

          ratio_rs = curr_fits['boot_params'][:,d,v_cons[c],3] # this is the rs parameter, which is already as a ratio relative to rc
          ratio_rs_clip = np.clip(ratio_rs, 1, 8); # all sizes >= X are treated the same
          if plotMetrCorr == 1:
            no_nan = np.logical_and(~np.isnan(sf70s), ~np.isnan(cfs));
            rsq = np.corrcoef(np.log10(sf70s[no_nan]), np.log10(cfs[no_nan]))
            dispAx[d][row_ind, col_ind].scatter(sf70s, cfs, s=15+np.square(ratio_rs_clip)); # add a constant to avoid very small sizes
            ylabel = '$f_c$';
          elif plotMetrCorr == 2:
            no_nan = np.logical_and(~np.isnan(sf70s), ~np.isnan(rcs));
            rsq = np.corrcoef(sf70s[no_nan], rcs[no_nan])
            dispAx[d][row_ind, col_ind].scatter(sf70s, rcs, s=15+np.square(ratio_rs_clip)); # add a constant to avoid very small sizes
            ylabel = '$r_c$';

          if col_ind == 0:
              if row_ind == n_rows-1: # only on the last row do we make the sf70 label
                  dispAx[d][row_ind, col_ind].set_xlabel(r'$sf_{70}$')
              dispAx[d][row_ind, col_ind].set_ylabel(r'%s' % ylabel);

          mdn_gainRat = np.nanmedian(curr_fits['boot_params'][:,d,v_cons[c],2]) # this is the gs param, which is already relative to gc
          mdn_radRat = np.nanmedian(ratio_rs)

          dispAx[d][row_ind, col_ind].set_title(r'%d%%: $\frac{g_s}{g_c}$=%.2f; $\frac{r_s}{c_c}$=%.2f; r=%.2f; [n=%02d]' % (100*all_cons[v_cons[c]], mdn_gainRat, mdn_radRat, rsq[0,-1], np.sum(no_nan)));
          # per TM (21.10.26, frequency should ALWAYS be on log scale)
          dispAx[d][row_ind, col_ind].set_xscale('log');
          dispAx[d][row_ind, col_ind].set_yscale('log');

          sns.despine(offset=10, ax=dispAx[d][row_ind, col_ind]);

      fDisp[d].suptitle('%s #%d (f1f0: %.2f)' % (cellType, cellNum, f1f0rat));

  #print('saving corr.');
  saveName = "/cell_%03d_rc_sf70.pdf" % (cellNum) if plotMetrCorr == 2 else "/cell_%03d_cFreq_sf70.pdf" % (cellNum)
  full_save = os.path.dirname(str(save_loc + 'byDisp%s/' % rvcFlag));

  if not os.path.exists(full_save):
    os.makedirs(full_save);
  pdfSv = pltSave.PdfPages(full_save + saveName);
  for f in fDisp:
      pdfSv.savefig(f); # only one figure here...
      plt.close(f);
  pdfSv.close()

# #### Plot d-DoG-S model in space, just sfMix contrasts

if descrMod == 3 or descrMod == 5: # i.e. d-DoG-s

  isMult = True if descrMod == 3 else False; # which parameterization of the d-DoG-S is it?

  mixCons = hf.get_exp_params(expInd).nCons;
  minResp = np.min(np.min(np.min(respMean[~np.isnan(respMean)])));
  maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));

  f, sfMixAx = plt.subplots(mixCons, nDisps, figsize=(nDisps*9, mixCons*8), sharex=True, sharey=True);

  for d in range(nDisps):
      v_cons = np.array(val_con_by_disp[d]);
      n_v_cons = len(v_cons);
      v_cons = v_cons[np.arange(np.maximum(0, n_v_cons -mixCons), n_v_cons)]; # max(1, .) for when there are fewer contrasts than 4
      n_v_cons = len(v_cons);

      if comm_S_calc and joint>0: # this only applies when doing joint-across-con fits
          ref_params = descrParams[d, v_cons[-1]];
      else:
          ref_params = None;

      for c in reversed(range(n_v_cons)):
          c_plt_ind = n_v_cons - c - 1;
          conStr = str(int(100*(np.round(all_cons[v_cons[c]], 2))));
          sfMixAx[c_plt_ind, d].set_title('con: %s%%' % conStr);
          v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);

          sfVals = all_sfs[v_sfs];
          resps  = respMean[d, v_sfs, v_cons[c]];

          # plot model
          prms_curr = descrParams[d, v_cons[c]];
          space, samps, dc, df1, df2 = hf.parker_hawken(np.copy(prms_curr), inSpace=True, debug=True, isMult=isMult, ref_params=ref_params);

          sfMixAx[c_plt_ind, d].plot(samps, space, 'k-')#, label='full');
          # and plot the constitutent parts
          sfMixAx[c_plt_ind, d].plot(samps, dc, 'k--')#, label='center');
          sfMixAx[c_plt_ind, d].plot(samps, df1, 'r--')#, label='f1');
          sfMixAx[c_plt_ind, d].plot(samps, df2, 'b--')#, label='f2');

          if d == 0:
            if c_plt_ind == 0:
              sfMixAx[c_plt_ind, d].set_ylabel('sensitivity');
              #sfMixAx[c_plt_ind, d].legend(fontsize='x-small');
            if c_plt_ind == mixCons-1:
              sfMixAx[c_plt_ind, d].set_xlabel('dva');

          # Add parameters! (in ax-transformed coords, (0,0) is bottom left, (1,1) is top right
          prms_curr_trans = hf.parker_hawken_transform(np.copy(prms_curr), space_in_arcmin=True, isMult=isMult, ref_params=ref_params);
          kc1,xc1,ks1,xs1 = prms_curr_trans[0:4];
          kc2,xc2,ks2,xs2 = prms_curr_trans[4:8];
          g,S = prms_curr_trans[8:];
          # -- first, dog 1; then dog2; finally gain, S
          sfMixAx[c_plt_ind, d].text(0, 0.16, r"""ctr: $k_c/k_s/x_c/x_s$=%.0f/%.0f/%.2f'/%.2f'""" % (kc1,ks1,xc1,xs1), transform=sfMixAx[c_plt_ind, d].transAxes, horizontalalignment='left', fontsize='x-small');
          sfMixAx[c_plt_ind, d].text(0, 0.08, r"""sd: $k_c/k_s/x_c/x_s$=%.0f/%.0f/%.2f'/%.2f'""" % (kc2,ks2,xc2,xs2), transform=sfMixAx[c_plt_ind, d].transAxes, horizontalalignment='left', fontsize='x-small');
          sfMixAx[c_plt_ind, d].text(0, 0, r"""$g/S$=%.2f/%.2f'""" % (g,S), transform=sfMixAx[c_plt_ind, d].transAxes, horizontalalignment='left', fontsize='x-small');

          #sfMixAx[c_plt_ind, d].set_aspect('equal');

          # Set ticks out, remove top/right axis, put ticks only on bottom/left
          sns.despine(ax=sfMixAx[c_plt_ind, d], offset=10, trim=False);

  f.legend();
  f.suptitle('%s #%d (%s; f1f0 %.2f)' % (cellType, cellNum, cellName, f1f0rat));
  #f.tight_layout(rect=[0, 0, 1, 0.97])

  allFigs = [f]; 
  saveName = "/cell_%03d.pdf" % (cellNum)
  full_save = os.path.dirname(str(save_loc + 'sfMixOnly%s_ddogs/' % rvcFlag));
  if not os.path.exists(full_save):
    os.makedirs(full_save);
  pdfSv = pltSave.PdfPages(full_save + saveName);
  for fig in range(len(allFigs)):
      pdfSv.savefig(allFigs[fig])
      plt.close(allFigs[fig])
  pdfSv.close();

# #### Plot d-DoG-S model in space, all SF tuning curves

if descrMod == 3 or descrMod == 5: # i.e. d-DoG-s

  fDisp = []; dispAx = [];

  sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);    

  for d in range(1): #nDisps): # let's only do single gratings as of 22.01.13

      v_cons = val_con_by_disp[d];
      n_v_cons = len(v_cons);

      n_rows = int(np.ceil(n_v_cons/np.floor(np.sqrt(n_v_cons)))); # make this close to a rectangle/square in arrangement (cycling through cons)
      n_cols = int(np.ceil(n_v_cons/n_rows));

      fCurr, dispCurr = plt.subplots(n_rows, n_cols, figsize=(n_cols*10, n_rows*12), sharey=True);
      fDisp.append(fCurr)
      dispAx.append(dispCurr);    

      minResp = np.min(np.min(respMean[d, ~np.isnan(respMean[d, :, :])]));
      maxResp = np.max(np.max(respMean[d, ~np.isnan(respMean[d, :, :])]));

      if comm_S_calc and joint>0: # this only applies when doing joint-across-con fits
          ref_params = descrParams[d, v_cons[-1]];
      else:
          ref_params = None;

      for c in reversed(range(n_v_cons)):
          row_ind = int((n_v_cons-c-1)/n_cols);
          col_ind = np.mod((n_v_cons-c-1), n_cols);

          c_plt_ind = len(v_cons) - c - 1;
          currClr = [(n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons)];

          dispAx[d][row_ind, col_ind].set_title('con: %s%%' % str(int(100*(np.round(all_cons[v_cons[c]], 2)))));

          # plot model
          prms_curr = descrParams[d, v_cons[c]];
          space, samps, dc, df1, df2 = hf.parker_hawken(np.copy(prms_curr), inSpace=True, debug=True, isMult=isMult, ref_params=ref_params);

          dispAx[d][row_ind, col_ind].plot(samps, space, 'k-')#, label='full');
          # and plot the constitutent parts
          dispAx[d][row_ind, col_ind].plot(samps, dc, 'k--')#, label='center');
          dispAx[d][row_ind, col_ind].plot(samps, df1, 'r--')#, label='f1');
          dispAx[d][row_ind, col_ind].plot(samps, df2, 'b--')#, label='f2');

          if c_plt_ind == 0:
            dispAx[d][row_ind, col_ind].set_ylabel('sensitivity');
            #dispAx[d][row_ind, col_ind].legend(fontsize='x-small');
          if c_plt_ind == mixCons-1:
            dispAx[d][row_ind, col_ind].set_xlabel('dva');

          # Add parameters! (in ax-transformed coords, (0,0) is bottom left, (1,1) is top right
          prms_curr_trans = hf.parker_hawken_transform(np.copy(prms_curr), space_in_arcmin=True, isMult=isMult, ref_params=ref_params);
          kc1,xc1,ks1,xs1 = prms_curr_trans[0:4];
          kc2,xc2,ks2,xs2 = prms_curr_trans[4:8];
          g,S = prms_curr_trans[8:];
          # -- first, dog 1; then dog2; finally gain, S
          dispAx[d][row_ind, col_ind].text(0, 0.16, r"""ctr: $k_c/k_s/x_c/x_s$=%.0f/%.0f/%.2f'/%.2f'""" % (kc1,ks1,xc1,xs1), transform=dispAx[d][row_ind, col_ind].transAxes, horizontalalignment='left', fontsize='x-small');
          dispAx[d][row_ind, col_ind].text(0, 0.08, r"""sd: $k_c/k_s/x_c/x_s$=%.0f/%.0f/%.2f'/%.2f'""" % (kc2,ks2,xc2,xs2), transform=dispAx[d][row_ind, col_ind].transAxes, horizontalalignment='left', fontsize='x-small');
          dispAx[d][row_ind, col_ind].text(0, 0, r"""$g/S$=%.2f/%.2f'""" % (g,S), transform=dispAx[d][row_ind, col_ind].transAxes, horizontalalignment='left', fontsize='x-small');

      fCurr.suptitle('%s #%d (f1f0: %.2f)' % (cellType, cellNum, f1f0rat));

  saveName = "/cell_%03d.pdf" % (cellNum)
  full_save = os.path.dirname(str(save_loc + 'byDisp%s_ddogs/' % rvcFlag));
  if not os.path.exists(full_save):
    os.makedirs(full_save);
  pdfSv = pltSave.PdfPages(full_save + saveName);
  for f in fDisp:
      pdfSv.savefig(f)
      plt.close(f)
  pdfSv.close();
