# coding: utf-8

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

comps = [[0, -1], [1,2], [3,4], [5,6]]
def comp_ind_to_str(ind):
    if ind==0:
        return 'all';
    elif ind==1:
        return 'even';
    elif ind==2:
        return 'odd';
    elif ind==3:
        return 'first half';
    elif ind==4:
        return 'second half';
    elif ind==5:
        return 'random 50%';
    elif ind==6:
        return 'remainder';
    elif ind==-1:
        return 'model';
    else:
        return 'EMPTY OR ERROR'

nRows = len(comps);
nComps = np.sum([x is not None for x in comps]);

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

fracSig = 0 if expDir == 'LGN/' else 1; # we only enforce the "upper-half sigma as fraction of lower half" for V1 cells! 

### DATALIST
expName = hf.get_datalist(expDir, force_full=1);
### DESCRLIST
hpc_str = 'HPC' if isHPC else '';
if expDir == 'LGN/':# or expDir == 'altExp':
    #descrBase = 'descrFits%s_220418' % hpc_str;
    #descrBase = 'descrFits%s_220421' % hpc_str;
    descrBase = 'descrFits%s_220504' % hpc_str;
else:
    #descrBase = 'descrFits%s_220323' % hpc_str;
    descrBase = 'descrFits%s_220410' % hpc_str;
##############
# 
##############
if expDir == 'LGN/':
  rvcBase = 'rvcFits%s_220504' % hpc_str;
  #rvcBase = 'rvcFits%s_220421' % hpc_str;
  #rvcBase = 'rvcFits%s_220418' % hpc_str;
else:
  rvcBase = 'rvcFits%s_210914' % ''#hpc_str; # if V1?
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
_, _, respOrg, respAll = hf.organize_resp(spikes_rate, trialInf, expInd, respsAsRate=True);
nTrials = np.sum(~np.isnan(respAll[0,-1,-1])); # single grating, highest SF/CON
halfway = np.floor(nTrials/2).astype(int)

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

# #### All SF tuning on one graph, split by dispersion

fDisp = []; dispAx = [];

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);

minResp_toPlot = 1e-0;
  
for d in range(1):
    
    v_cons = val_con_by_disp[d];
    n_v_cons = len(v_cons);
    
    fCurr, dispCurr = plt.subplots(nRows, 2, figsize=(35, nRows*20), sharey=True, sharex=True);
    fDisp.append(fCurr)
    dispAx.append(dispCurr);

    fCurr.suptitle('%s #%d (f1f0 %.2f, %02d tr/cond)' % (cellType, cellNum, f1f0rat, nTrials));

    maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));  
    minToPlot = 5e-1;
    ref_params = descrParams[d, v_cons[-1]] if joint>0 else None; # the reference parameter is the highest contrast for that dispersion
    ref_rc_val = ref_params[1]; # will be used ONLY if DoG AND joint==5

    if forceLog == 1:
        if baseline_resp > 0: #is not None:
            to_sub = baseline_resp;
        else:
            to_sub = np.array(0);
    else:
        to_sub = np.array(0);

    for row_i, comps in enumerate(comps):

      for col_i, j in enumerate(comps):

        if j is None:
            continue;
        if j==0: # all data
            respsCurr = respMean;
        elif j==1: # even trials
            by_trial = respAll[..., ::2]
        elif j==2: # odd trials
            by_trial = respAll[..., 1::2]
        elif j==3: # first half of data
            by_trial = respAll[..., 0:halfway];
        elif j==4: # second half of data
            by_trial = respAll[..., (halfway+1):];
        elif j==5: # random 50% sample
            by_trial = hf.organize_resp(spikes_rate, trialInf, expInd, respsAsRate=True, resample=True, cross_val=(0.5, -1))[-1];
            by_trial_rand = by_trial;
        elif j==6: # remaining 50% of data
            nan_val = -1e3;
            training = np.copy(by_trial_rand);
            training[np.isnan(training)] = nan_val;
            all_data = np.copy(respAll);
            all_data[np.isnan(all_data)] = nan_val;
            heldout = np.abs(all_data - training) > 1e-6; # if the difference is g.t. this, it means they are different value
            test_data = np.nan * np.zeros_like(respAll);
            test_data[heldout] = all_data[heldout]; # then put the heldout values here
            by_trial = test_data

        if j>0: # need to get means..
            respsCurr = np.nanmean(by_trial, axis=-1);

        curr_loss = 0; data_loss = 0;
        vExps = [];
        for c in reversed(range(n_v_cons)):
            v_sfs = ~np.isnan(respsCurr[d, :, v_cons[c]]);        
            prms_curr = descrParams[d, v_cons[c]];

            col = [(n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons), (n_v_cons-c-1)/float(n_v_cons)];
            if j>=0:
                # plot data [0]
                plot_resp = respsCurr[d, v_sfs, v_cons[c]] - to_sub;
                
                curr_line, = dispAx[d][row_i][col_i].plot(all_sfs[v_sfs][plot_resp>minToPlot], plot_resp[plot_resp>minToPlot], '-o', clip_on=False, \
                    color=col, label='%s%%' % (str(int(100*np.round(all_cons[v_cons[c]], 2)))));
                if baseline_resp > 0:
                    dispAx[d][row_i][col_i].axhline(baseline_resp, linestyle='--', color='k');

                # compute model loss...
                curr_loss += hf.DoG_loss(prms_curr, respsCurr[d,v_sfs,v_cons[c]], all_sfs[v_sfs], loss_type=descrLoss, DoGmodel=descrMod, dir=dir, joint=0, baseline=baseline_resp, ref_params=ref_params, ref_rc_val=ref_rc_val);
                vExps.append(hf.var_explained(respsCurr[d,v_sfs,v_cons[c]], prms_curr, all_sfs[v_sfs], descrMod, baseline=baseline_resp, ref_params=ref_params, ref_rc_val=ref_rc_val));
                # AND compute data loss --> per discussion with Tim on 22.05.03, let's set a reference for the subsetted losses by computing the loss between the full dataset and the current subsample
                if descrLoss==1:
                    data_loss += np.sum(np.square(respsCurr[d,v_sfs,v_cons[c]] - respMean[d,v_sfs,v_cons[c]]));
                elif descrLoss==2:
                    rS = respsCurr[d,v_sfs,v_cons[c]];
                    rA = respMean[d,v_sfs,v_cons[c]];
                    data_loss += np.sum(np.square(np.sign(rS)*np.sqrt(np.abs(rS)) - np.sign(rA)*np.sqrt(np.abs(rA))));

            elif j==-1: # model
                # plot descr fit [1]
                descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod, baseline=baseline_resp, fracSig=fracSig, ref_params=ref_params, ref_rc_val=ref_rc_val);
                dispAx[d][row_i][col_i].plot(sfs_plot, descrResp-to_sub, color=col);


            # set the nice things
            dispAx[d][row_i][col_i].set_xlim((0.5*min(all_sfs), 1.2*max(all_sfs)));

            dispAx[d][row_i][col_i].set_xscale('log');
            if expDir == 'LGN/' or forceLog == 1: # we want double-log if it's the LGN!
                dispAx[d][row_i][col_i].set_yscale('log');
                #dispAx[d][row_i][col_i].set_ylim((minToPlot, 1.5*maxResp));
                dispAx[d][row_i][col_i].set_ylim((5e-1, 300)); # common y axis for ALL plots
                logSuffix = 'log_';
            else:
                dispAx[d][row_i][col_i].set_ylim((np.minimum(-5, minResp-5), 1.5*maxResp));
                logSuffix = '';
        # END of con loop
        
        dispAx[d][row_i][col_i].set_xlabel('sf (c/deg)'); 

        sns.despine(ax=dispAx[d][row_i][col_i], offset=10, trim=False); 

        lbl_str = '' if row_i==0 else 'above baseline ';
        curr_vExp = np.nanmedian(vExps);
        ref_loss = '||%.2f' % descrFits[cellNum-1]['totalNLL'][d] if j == 0 else '';
        dispAx[d][row_i][col_i].set_title('%s ([%.2f]--%.2f%s;%.2f)' % (comp_ind_to_str(j), data_loss, curr_loss, ref_loss, curr_vExp));
        if col_i==0:
            dispAx[d][row_i][col_i].set_ylabel('resp %s(sps)' % lbl_str);
        if row_i==0 and col_i==0:
            dispAx[d][row_i][col_i].legend(fontsize='medium');

saveName = "/allCons_%scell_%03d.pdf" % (logSuffix, cellNum)
full_save = os.path.dirname(str(save_loc + 'byDisp%s_noise_comparisons/' % rvcFlag));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()
