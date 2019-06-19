# coding: utf-8

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
expDir   = sys.argv[3]; 
rvcAdj   = int(sys.argv[4]); # if 1, then let's load rvcFits to adjust responses to F1
if len(sys.argv) > 5:
  respVar = int(sys.argv[5]);
else:
  respVar = 1;

loc_base = os.getcwd() + '/';
data_loc = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + 'figures/';

### DATALIST
expName = 'dataList.npy';
#expName = 'dataList_glx.npy'
#expName = 'dataList_mr.npy'
### FITLIST
#fitBase = 'fitList_190321c';
#fitBase = 'fitListSPcns_181130c';
#fitBase = 'fitListSP_181202c';
#fitBase = 'fitList_190206c';
#fitBase = 'fitList_190321c';

#fitBase = 'mr_fitList_190502cA';
#fitBase = 'fitList_190502cA_glx'; # mostly deprecated...(i.e. even for GLX fits, we just use fitList_*)
fitBase = 'fitList_190502cA';
### RVCFITS
rvcBase = 'rvcFits'; # direc flag & '.npy' are added

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
compDir  = str(fitBase + '_comp' + lossSuf + 'diff/');
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

cellName = dataList['unitName'][cellNum-1];
try:
  cellType = dataList['unitType'][cellNum-1];
except: 
  # TODO: note, this is dangerous; thus far, only V1 cells don't have 'unitType' field in dataList, so we can safely do this
  cellType = 'V1'; 

expData  = np.load(str(data_loc + cellName + '_sfm.npy'), encoding='latin1').item();
expInd   = hf.get_exp_ind(data_loc, cellName)[0];

# #### Load model fits

modFit_fl = fitList_fl[cellNum-1]['params']; # 
modFit_wg = fitList_wg[cellNum-1]['params']; # 
modFits = [modFit_fl, modFit_wg];
normTypes = [1, 2]; # flat, then weighted

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

modResps = [mod_resp.SFMGiveBof(fit, expData, normType=norm, lossType=lossType, expInd=expInd) for fit, norm in zip(modFits, normTypes)];
modResps = [x[1] for x in modResps]; # 1st return output (x[0]) is NLL (don't care about that here)
gs_mean = modFit_wg[8]; 
gs_std = modFit_wg[9];
# now organize the responses
orgs = [hf.organize_resp(mr, expData, expInd) for mr in modResps];
oriModResps = [org[0] for org in orgs]; # only non-empty if expInd = 1
conModResps = [org[1] for org in orgs]; # only non-empty if expInd = 1
sfmixModResps = [org[2] for org in orgs];
allSfMixs = [org[3] for org in orgs];

modLows = [np.nanmin(resp, axis=3) for resp in allSfMixs];
modHighs = [np.nanmax(resp, axis=3) for resp in allSfMixs];
modAvgs = [np.nanmean(resp, axis=3) for resp in allSfMixs];
modSponRates = [fit[6] for fit in modFits];

# more tabulation - stim vals, organize measured responses
_, stimVals, val_con_by_disp, _, _ = hf.tabulate_responses(expData, expInd);
if rvcAdj == 1:
  rvcFlag = '_f1';
  rvcFits = hf.get_rvc_fits(data_loc, expInd, cellNum, rvcName=rvcBase);
else:
  rvcFlag = '';
  rvcFits = hf.get_rvc_fits(data_loc, expInd, cellNum, rvcName='None');
spikes  = hf.get_spikes(expData['sfm']['exp']['trial'], rvcFits=rvcFits, expInd=expInd);
_, _, respOrg, respAll    = hf.organize_resp(spikes, expData, expInd);

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

### Now, recenter data relative to flat normalization model
allAvgs = [respMean, modAvgs[1], modAvgs[0]]; # why? weighted is 1, flat is 0
respsRecenter = [x - allAvgs[2] for x in allAvgs]; # recentered

respMean = respsRecenter[0];
modAvgs  = [respsRecenter[2], respsRecenter[1]];

# ### Plots

# set up model plot info
# i.e. flat model is red, weighted model is green
modColors = ['r', 'g']
modLabels = ['flat', 'wght']

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
        
        # plot data
        sfMixAx[c_plt_ind, d].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]], 
                                       respVar[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);

	# plot model fits
        [sfMixAx[c_plt_ind, d].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], color=cc, alpha=0.7, clip_on=False, label=s) for modAvg, cc, s in zip(modAvgs, modColors, modLabels)];

        sfMixAx[c_plt_ind, d].set_xlim((np.min(all_sfs), np.max(all_sfs)));
        sfMixAx[c_plt_ind, d].set_ylim((-1.5*np.abs(minResp), 1.5*maxResp));
        sfMixAx[c_plt_ind, d].set_xscale('log');
        sfMixAx[c_plt_ind, d].set_xlabel('sf (c/deg)');
        sfMixAx[c_plt_ind, d].set_ylabel('resp (sps)');

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        sfMixAx[c_plt_ind, d].tick_params(labelsize=15, width=1, length=8, direction='out');
        sfMixAx[c_plt_ind, d].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
        sns.despine(ax=sfMixAx[c_plt_ind, d], offset=10, trim=False);

f.legend();
f.suptitle('%s #%d (%s), loss %.2f|%.2f' % (cellType, cellNum, cellName, fitList_fl[cellNum-1]['NLL'], fitList_wg[cellNum-1]['NLL']));

# now save
saveName = "/cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'sfMixOnly%s/' % rvcFlag));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(f)
plt.close(f)
pdfSv.close()
