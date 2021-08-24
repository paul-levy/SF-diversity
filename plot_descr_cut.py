# coding: utf-8
# NOTE: Unlike plot_descr.py, this file is used to plot 
# - descriptive SF tuning fit ONLY, and ONLY for single gratings
# -- specifically, we'll plot SF tuning for 4 conditions
# ---- if we have the full set of contrasts, we'll do single gratings
# ---- at 4 contrasts; otherwise, we'll do the two contrasts at the first two dispersion levels 

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
    rcParams['lines.markeredgewidth'] = 0; # remove edge??                                  rcParams['axes.linewidth'] = 3;
    rcParams['lines.markersize'] = 12; # 8 is the default                                   rcParams['font.style'] = 'oblique';
    rcParams['xtick.major.size'] = 25
    rcParams['xtick.minor.size'] = 12
    rcParams['ytick.major.size'] = 25
    rcParams['ytick.minor.size'] = 0; # i.e. don't have minor ticks on y...                 rcParams['xtick.major.width'] = 2
    rcParams['xtick.minor.width'] = 2
    rcParams['ytick.major.width'] = 2
    rcParams['ytick.minor.width'] = 0

majWidth = 4;
minWidth = 4;
lblSize = 40;

peakFrac = 0.75; # plot fall of to peakFrac of peak, rather than peak or charFreq
inclLegend = 0;

cellNum   = int(sys.argv[1]);
expDir    = sys.argv[2]; 
descrMod  = int(sys.argv[3]);
descrLoss = int(sys.argv[4]);
descrJnt  = int(sys.argv[5]);
rvcAdj    = int(sys.argv[6]); # if 1, then let's load rvcFits to adjust F1, as needed
rvcMod    = int(sys.argv[7]);
if len(sys.argv) > 8:
  respVar = int(sys.argv[8]);
else:
  respVar = 1;
if len(sys.argv) > 9:
  forceLog = int(sys.argv[9]); # used for byDisp/allCons_... (sf plots)
else:
  forceLog = 0;

loc_base = os.getcwd() + '/';

data_loc = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + 'figures/';

fracSig = 0 if expDir == 'LGN/' else 1; # we only enforce the "upper-half sigma as fraction of lower half" for V1 cells! 

### DATALIST
expName = hf.get_datalist(expDir, force_full=1);
### DESCRLIST
descrBase = 'descrFits_210517';
#descrBase = 'descrFits_210503';
#descrBase = 'descrFits_210304';
#descrBase = 'descrFits_191023'; # for V1, V1_orig, LGN
#descrBase = 'descrFits_200507'; # for altExp
#descrBase = 'descrFits_190503';
if descrJnt == 1:
  descrBase = '%s_joint' % descrBase;
### RVCFITS
#rvcBase = 'rvcFits_200507'; # direc flag & '.npy' are added
#rvcBase = 'rvcFits_191023'; # direc flag & '.npy' are adde
#rvcBase = 'rvcFits_200714'; # direc flag & '.npy' are adde
#rvcBase = 'rvcFits_200507';
rvcBase = 'rvcFits_210517';
# -- rvcAdj = -1 means, yes, load the rvcAdj fits, but with vecF1 correction rather than ph fit; so, we'll 
rvcAdjSigned = rvcAdj;
rvcAdj = np.abs(rvcAdj);

##################
### Spatial frequency
##################

modStr  = hf.descrMod_name(descrMod)
fLname  = hf.descrFit_name(descrLoss, descrBase=descrBase, modelName=modStr);
descrFits = hf.np_smart_load(data_loc + fLname);
pause_tm = 2.5*np.random.rand();
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
expInd   = hf.get_exp_ind(data_loc, cellName)[0];

if expInd <= 2: # if expInd <= 2, then there cannot be rvcAdj, anyway!
  rvcAdj = 0; # then we'll load just below!

if rvcAdj == 1:
  vecF1 = 1 if rvcAdjSigned==-1 else 0
  dir = 1 if rvcAdjSigned==1 else None # we dont' have pos/neg phase if vecF1
  rvcFits = hf.np_smart_load(data_loc + hf.rvc_fit_name(rvcBase, modNum=rvcMod, dir=dir, vecF1=vecF1)); # i.e. positive
  force_baseline = False; # plotting baseline will depend on F1/F0 designation
if rvcAdj == 0:
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
# NOTE: We pass in the rvcFits where rvcBase[name] goes, and use -1 in rvcMod to indicate that we've already loaded the fits
spikes_rate, which_measure = hf.get_adjusted_spikerate(trialInf, cellNum, expInd, data_loc, rvcFits, rvcMod=-1, descrFitName_f0 = fLname, baseline_sub=False, force_dc=force_dc, force_f1=force_f1, return_measure=True, vecF1=vecF1);
# let's also get the baseline
force_baseline = True if force_dc else False; # plotting baseline will depend on F1/F0 designation
if force_baseline or (f1f0rat < 1 and expDir != 'LGN/'): # i.e. if we're in LGN, DON'T get baseline, even if f1f0 < 1 (shouldn't happen)
  baseline_resp = hf.blankResp(trialInf, expInd, spikes=spikes_rate, spksAsRate=True)[0];
else:
  baseline_resp = int(0);

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

# #### Plot just sfMix contrasts

mixCons = hf.get_exp_params(expInd).nCons;
minResp = np.min(np.min(np.min(respMean[~np.isnan(respMean)])));
maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));

f, sfMixAx = plt.subplots(2, 2, figsize=(15, 12), sharex=True,sharey=True);

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);

val_cons = val_con_by_disp[0];
# organize the four conditions by (disp,con)
if len(val_cons) > 2:
    # then the highest four contrasts, single gratings
    conds = [(0,val_cons[-1]), (0,val_cons[-2]), (0,val_cons[-3]), (0,val_cons[-4])]
else:
    conds = [(0,val_cons[-1]), (0,val_cons[-2]), (1,val_cons[-1]), (1,val_cons[-2])]

for ii, cond in enumerate(conds):
    curr_disp, curr_con = cond[0], cond[1]

    plt_ind_row, plt_ind_col = np.mod(ii,2), int(np.floor(ii/2))
    sfMixAx[plt_ind_row, plt_ind_col].set_title('con: %d%%' % int(100*(np.round(all_cons[curr_con], 2))))
    v_sfs = ~np.isnan(respMean[curr_disp, :, curr_con]);
        
    sfVals = all_sfs[v_sfs];
    resps  = respMean[curr_disp, v_sfs, curr_con];

    # plot data
    sfMixAx[plt_ind_row, plt_ind_col].errorbar(sfVals, resps,
                respVar[curr_disp, v_sfs, curr_con], fmt='o', clip_on=False, color=dataClr);

    # now, let's also plot the baseline, if complex cell
    if baseline_resp > 0: #is not None: # i.e. complex cell
        sfMixAx[plt_ind_row, plt_ind_col].axhline(baseline_resp, color=dataClr, linestyle='dashed');

    # plot descrFit
    prms_curr = descrParams[curr_disp, curr_con];
    descrResp = hf.get_descrResp(prms_curr, sfs_plot, descrMod, baseline=baseline_resp, fracSig=fracSig);
    sfMixAx[plt_ind_row, plt_ind_col].plot(sfs_plot, descrResp, color=modClr);
        
    # plot prefSF, center of mass
    #ctr = hf.sf_com(resps, sfVals);
    pSf = hf.descr_prefSf(prms_curr, dog_model=descrMod, all_sfs=all_sfs);
    sfMixAx[plt_ind_row, plt_ind_col].plot(pSf, 1, linestyle='None', marker='v', color=modClr, clip_on=False); # plot at y=1

    sfMixAx[plt_ind_row, plt_ind_col].set_xlim((np.min(all_sfs), np.max(all_sfs)));
    sfMixAx[plt_ind_row, plt_ind_col].set_ylim((np.minimum(-5, minResp-5), 1.25*maxResp)); # ensure that 0 is included in the range of the plot!
    sfMixAx[plt_ind_row, plt_ind_col].set_xscale('log');
    if ii == 1:
        sfMixAx[plt_ind_row, plt_ind_col].set_xlabel('sf (c/deg)');
        sfMixAx[plt_ind_row, plt_ind_col].set_ylabel('resp (sps)');

    # Set ticks out, remove top/right axis, put ticks only on bottom/left
    #sfMixAx[plt_ind_row, plt_ind_col].tick_params(labelsize=15, width=1, length=8, direction='out');
    #sfMixAx[plt_ind_row, plt_ind_col].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
    sns.despine(ax=sfMixAx[plt_ind_row, plt_ind_col], offset=10, trim=False);

f.suptitle('%s #%d (%s; f1f0 %.2f)' % (cellType, cellNum, cellName, f1f0rat));
f.tight_layout(rect=[0, 0, 1, 0.97])
	        
allFigs = [f]; 
saveName = "/cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'sfMixOnly%s_restricted/' % rvcFlag));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
for fig in range(len(allFigs)):
    pdfSv.savefig(allFigs[fig])
    plt.close(allFigs[fig])
pdfSv.close()
