# coding: utf-8

######################## To do:

import os
import numpy as np
from numpy.matlib import repmat
import matplotlib
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns
sns.set(style='ticks')
import helper_fcns
from scipy.stats import poisson, nbinom
from scipy.stats import norm as normpdf
from scipy.stats.mstats import gmean

import pdb

import sys # so that we can import model_responses (in different folder)
import model_responses

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/Analysis/Functions/paul_plt_cluster.mplstyle');
plt.rc('legend',fontsize='medium') # using a named size

which_cell = int(sys.argv[1]);
fit_type = int(sys.argv[2]);
norm_type = int(sys.argv[3]);

if norm_type == 1: # i.e. gaussian, not "standard asymmetry"
  if len(sys.argv) > 4:
    gs_mean = float(sys.argv[4]);
    gs_std = float(sys.argv[5]);
  else:
    gs_mean = helper_fcns.random_in_range([-1, 1])[0]; 
    gs_std = np.power(10, helper_fcns.random_in_range([-2, 2])[0]); # i.e. 1e-2, 1e2

# at CNS
# dataPath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/recordings/';
# savePath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/analysis/';
# personal mac
#dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/altExp/analysis/structures/';
#save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/altExp/analysis/figures/';
# prince cluster
dataPath = '/home/pl1465/SF_diversity/altExp/analysis/structures/';
save_loc = '/home/pl1465/SF_diversity/altExp/analysis/figures/';

if fit_type == 1:
  loss = lambda resp, pred: np.sum(np.power(resp-pred, 2)); # least-squares, for now...
  type_str = '-lsq';
if fit_type == 2:
  loss = lambda resp, pred: np.sum(np.square(np.sqrt(resp) - np.sqrt(pred)));
  type_str = '-sqrt';
if fit_type == 3:
  loss = lambda resp, pred: poisson.logpmf(resp, pred);
  type_str = '-poiss';
if fit_type == 4:
  loss = lambda resp, r, p: np.log(nbinom.pmf(resp, r, p)); # Likelihood for each pass under doubly stochastic model
  type_str = '-poissMod';

fitListName = 'fitList_180105.npy';
crfFitName = str('crfFits' + type_str + '.npy');

rpt_fit = 1; # i.e. take the multi-start result
if rpt_fit:
  is_rpt = '_rpt';
else:
  is_rpt = '';

conDig = 3; # round contrast to the 3rd digit

dataList = np.load(str(dataPath + 'dataList.npy'), encoding='latin1').item();

cellStruct = np.load(str(dataPath + dataList['unitName'][which_cell-1] + '_sfm.npy'), encoding='latin1').item();

# #### Load descriptive model fits, comp. model fits

descrFits = np.load(str(dataPath + 'descrFits.npy'), encoding = 'latin1').item();
descrFits = descrFits[which_cell-1]['params']; # just get this cell

modParams = np.load(str(dataPath + fitListName), encoding= 'latin1').item();
modParamsCurr = modParams[which_cell-1]['params'];

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

data = cellStruct['sfm']['exp']['trial'];

modRespAll = model_responses.SFMGiveBof(modParamsCurr, cellStruct)[1];
resp, stimVals, val_con_by_disp, validByStimVal, modResp = helper_fcns.tabulate_responses(cellStruct, modRespAll);
blankMean, blankStd, _ = helper_fcns.blankResp(cellStruct); 
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

# #### Plot just sfMix contrasts

# i.e. highest (up to) 4 contrasts for each dispersion

#########
# Normalization pool simulations
#########
conLevels = [1, 0.75, 0.5, 0.33, 0.1]; # what contrasts to test?
nCons = len(conLevels);
nDisps = 4;
sfCenters = np.logspace(-2, 2, 31); # just for now...

fNorm, conDisp_plots = plt.subplots(nCons, nDisps, sharey=True, figsize=(45,25))

norm_sim = np.nan * np.empty((nDisps, nCons, len(sfCenters)));
# simulations
for disp in range(nDisps):
    for conLvl in range(nCons):
      print('simulating normResp for family ' + str(disp+1) + ' and contrast ' + str(conLevels[conLvl]));

      # if modParamsCurr doesn't have inhAsym parameter, add it!
      if len(modParamsCurr) < 9: 
        modParamsCurr.append(helper_fcns.random_in_range([-0.35, 0.35])[0]); # enter asymmetry parameter

      for sfCent in range(len(sfCenters)):

        if norm_type == 1:
          ignore, ignore, ignore, normRespSimple = model_responses.SFMsimulate(modParamsCurr, cellStruct, disp+1, conLevels[conLvl], sfCenters[sfCent]);
          nTrials = normRespSimple.shape[0];
          nInhChan = cellStruct['sfm']['mod']['normalization']['pref']['sf'];
          inhWeightMat  = helper_fcns.genNormWeights(cellStruct, nInhChan, gs_mean, gs_std, nTrials);
          normResp = np.sqrt((inhWeightMat*normRespSimple).sum(1)).transpose();
          norm_sim[disp, conLvl, sfCent] = np.mean(normResp); # take mean of the returned simulations (10 repetitions per stim. condition)
        
        if norm_type == 0:
          # B: OR, calculation if using normalization calculation in model_responses.SFMsimulate
          ignore, normResp, ignore, ignore = model_responses.SFMsimulate(modParamsCurr, cellStruct, disp+1, conLevels[conLvl], sfCenters[sfCent]);
          norm_sim[disp, conLvl, sfCent] = np.mean(normResp);

      conDisp_plots[conLvl, disp].semilogx(sfCenters, norm_sim[disp, conLvl, :], 'b', clip_on=False);
      conDisp_plots[conLvl, disp].set_xlim([np.min(sfCenters), np.max(sfCenters)]);
      if norm_type == 1:
        conDisp_plots[conLvl, disp].text(0.5, 1.1, 'contrast: {:.2f}, dispersion level: {:.0f}, mu|sig: {:.2f}|{:.2f}'.format(conLevels[conLvl], disp+1, gs_mean, gs_std), fontsize=12, horizontalalignment='center', verticalalignment='center');
      elif norm_type == 0:
        conDisp_plots[conLvl, disp].text(0.5, 1.1, 'contrast: {:.2f}, dispersion level: {:.0f}, asym: {:.2f}'.format(conLevels[conLvl], disp+1, modParamsCurr[8]), fontsize=12, horizontalalignment='center', verticalalignment='center');
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
allFigs = [fNorm];
saveName = "/normResp_%d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'normSandbox/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for fig in range(len(allFigs)):
    pdfSv.savefig(allFigs[fig])
    plt.close(allFigs[fig])
pdfSv.close()
