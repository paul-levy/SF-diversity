# -*- coding: utf-8 -*-

# # Plotting

    # parameters
    # 00 = preferred spatial frequency   (cycles per degree)
    # 01 = derivative order in space
    # 02 = normalization constant        (log10 basis)
    # 03 = response exponent
    # 04 = response scalar
    # 05 = early additive noise
    # 06 = late additive noise
    # 07 = variance of response gain
    
# ### SF Diversity Project - plotting data, descriptive fits, and functional model fits

import os
import sys
import numpy as np
from helper_fcns import organize_modResp, flexible_Gauss, getSuppressiveSFtuning, compute_SF_BW, genNormWeights, random_in_range, evalSigmaFilter, setSigmaFilter, np_smart_load
import model_responses as mod_resp
from itertools import chain
import matplotlib
matplotlib.use('Agg') # why? so that we can get around having no GUI on cluster
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns

import pdb

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/Analysis/Functions/paul_plt_cluster.mplstyle');
sns.set(style='ticks');

# better plotting
from matplotlib import rcParams
rcParams['font.size'] = 20;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 2.5;
rcParams['axes.linewidth'] = 1.5;
rcParams['lines.markersize'] = 5;
rcParams['font.style'] = 'oblique';
rcParams['legend.fontsize'] ='large'; # using a named size

import pdb

cellNum = int(sys.argv[1]);
lossType = int(sys.argv[2]);
log_y = int(sys.argv[3]);

# prince
#save_loc = '/home/pl1465/SF_diversity/Analysis/Figures/';
#data_loc = '/home/pl1465/SF_diversity/Analysis/Structures/';
# CNS
save_loc = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/Analysis/Figures/';
data_loc = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/Analysis/Structures/';

expName = 'dataList.npy'
fitBase = 'fitListSPcns_181130c';
# first the fit type
fitSuf_fl = '_flat';
fitSuf_wg = '_wght';
# then the loss type
if lossType == 1:
  lossSuf = '_sqrt.npy';
elif lossType == 2:
  lossSuf = '_poiss.npy';
elif lossType == 3:
  lossSuf = '_modPoiss.npy';
elif lossType == 4:
  lossSuf = '_chiSq.npy';

fitName_fl = str(fitBase + fitSuf_fl + lossSuf);
fitName_wg = str(fitBase + fitSuf_wg + lossSuf);

# set the save directory to save_loc, then create the save directory if needed
compDir  = str(fitBase + '_comp' + lossSuf);
subDir   = compDir.replace('fitList', 'fits').replace('.npy', '');
save_loc = str(save_loc + subDir + '/');
if not os.path.exists(save_loc):
  os.makedirs(save_loc);

descrExpName = 'descrFits.npy';
descrModName = 'descrFitsModel.npy';

nFam = 5;
nCon = 2;
plotSteps = 100; # how many steps for plotting descriptive functions?
sfPlot = np.logspace(-1, 1, plotSteps);

# for bandwidth/prefSf descriptive stuff
muLoc = 2; # mu is in location '2' of parameter arrays
height = 1/2.; # measure BW at half-height
sf_range = [0.01, 10]; # allowed values of 'mu' for fits - see descr_fit.py for details

dL = np_smart_load(data_loc + expName);
fitList_fl = np_smart_load(data_loc + fitName_fl);
fitList_wg = np_smart_load(data_loc + fitName_wg);
descrExpFits = np_smart_load(data_loc + descrExpName);
descrModFits = np_smart_load(data_loc + descrModName);

# #### Load data

expData = np_smart_load(str(data_loc + dL['unitName'][cellNum-1] + '_sfm.npy'));
expResp = expData

modFit_fl = fitList_fl[cellNum-1]['params']; # 
modFit_wg = fitList_wg[cellNum-1]['params']; # 
modFits = [modFit_fl, modFit_wg];
normTypes = [1, 2]; # flat, then weighted

descrExpFit = descrExpFits[cellNum-1]['params']; # nFam x nCon x nDescrParams
descrModFit = descrModFits[cellNum-1]['params']; # nFam x nCon x nDescrParams

modResps = [mod_resp.SFMGiveBof(fit, expData, normType=norm, lossType=lossType) for fit, norm in zip(modFits, normTypes)];
modResps = [x[1] for x in modResps]; # 1st return output is NLL (don't care about that here)
gs_mean = modFit_wg[8]; 
gs_std = modFit_wg[9];
# now organize the responses
orgs = [organize_modResp(mr, expData['sfm']['exp']['trial']) for mr in modResps];
oriModResps = [org[0] for org in orgs];
conModResps = [org[1] for org in orgs];
sfmixModResps = [org[2] for org in orgs];
allSfMixs = [org[3] for org in orgs];
# now organize the measured responses in the same way
oriExpResp, conExpResp, sfmixExpResp, allSfMixExp = organize_modResp(expData['sfm']['exp']['trial']['spikeCount'], \
                                                                           expData['sfm']['exp']['trial'])

# allSfMix is (nFam, nCon, nCond, nReps) where nCond is 11, # of SF centers and nReps is usually 10
modLows = [np.nanmin(resp, axis=3) for resp in allSfMixs];
modHighs = [np.nanmax(resp, axis=3) for resp in allSfMixs];
modAvgs = [np.nanmean(resp, axis=3) for resp in allSfMixs];
modSponRates = [fit[6] for fit in modFits];

findNan = np.isnan(allSfMixExp);
nonNan = np.sum(findNan == False, axis=3); # how many valid trials are there for each fam x con x center combination?
allExpSEM = np.nanstd(allSfMixExp, axis=3) / np.sqrt(nonNan); # SEM

# Do some analysis of bandwidth, prefSf

bwMod = np.ones((nFam, nCon)) * np.nan;
bwExp = np.ones((nFam, nCon)) * np.nan;
pSfMod = np.ones((nFam, nCon)) * np.nan;
pSfExp = np.ones((nFam, nCon)) * np.nan;

for f in range(nFam):
        
      ignore, bwMod[f,0] = compute_SF_BW(descrModFit[f, 0, :], height, sf_range)
      ignore, bwMod[f,1] = compute_SF_BW(descrModFit[f, 1, :], height, sf_range)
      pSfMod[f,0] = descrModFit[f, 0, muLoc]
      pSfMod[f,1] = descrModFit[f, 1, muLoc]

      ignore, bwExp[f, 0] = compute_SF_BW(descrExpFit[f, 0, :], height, sf_range)
      ignore, bwExp[f, 1] = compute_SF_BW(descrExpFit[f, 1, :], height, sf_range)
      pSfExp[f, 0] = descrExpFit[f, 0, muLoc]
      pSfExp[f, 1] = descrExpFit[f, 1, muLoc]

#########
# Plot the main stuff - sfMix experiment with model predictions and descriptive fits
#########

# In[281]:

f, all_plots = plt.subplots(nCon, nFam, sharex=True, sharey=True, figsize=(25,8))
expSfCent = expData['sfm']['exp']['sf'][0][0];
expResponses = expData['sfm']['exp']['sfRateMean'];
# sfRateMean/Var are doubly-nested (i.e. [nFam][nCon][sf] rather than [nFam, nCon, sf]) unpack here
sfMeanFlat = list(chain.from_iterable(chain.from_iterable(expResponses)));
sfVarFlat = list(chain.from_iterable(chain.from_iterable(expData['sfm']['exp']['sfRateVar'])));

# set up model plot info
modColors = ['r', 'g']
modLabels = ['flat', 'wght']

# plot experiment and models
# in this plot_compare version, we only plot the model average (not the banded low-to-high)
for con in reversed(range(nCon)): # contrast
    maxMod = np.maximum(np.amax(modAvgs[0][0][con]), np.amax(modAvgs[1][0][con]))
    yMax = 1.25*np.maximum(np.amax(expResponses[0][con]), maxMod); # we assume that 1.25x Max response for single grating will be enough
    all_plots[con, 0].set_ylim([-1, yMax]);
    for fam in reversed(range(nFam)): # family        
        expPoints = all_plots[con, fam].errorbar(expSfCent, expResponses[fam][con], allExpSEM[fam, con, :],\
                                                 linestyle='None', marker='o', color='b', clip_on=False, label='data +- 1 s.e.m.');
        # plot model average for both models (flat + weighted)
        [all_plots[con, fam].plot(expSfCent, modAvg[fam, con,:], color=c, alpha=0.7, clip_on=False, label=s) for modAvg, c, s in zip(modAvgs, modColors, modLabels)];
        sponRate = all_plots[con, fam].axhline(expData['sfm']['exp']['sponRateMean'], color='b', linestyle='dashed', label='data spon. rate');
        [all_plots[con, fam].axhline(sponRate, color=c, linestyle='dashed') for sponRate,c in zip(modSponRates, modColors)];
        all_plots[con,fam].set_xscale('log');
        if log_y:
          all_plots[con,fam].set_yscale('log');
        
        # pretty
        all_plots[con,fam].tick_params(labelsize=15, width=1, length=8, direction='out');
        all_plots[con,fam].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
        if con == 1:
            all_plots[con, fam].set_xlabel('spatial frequency (c/deg)', fontsize=20);
        if fam == 0:
            all_plots[con, fam].set_ylabel('response (spikes/s)', fontsize=20);
       
        all_plots[con,fam].text(0.5,1.10, 'exp: {:.2f} cpd | {:.2f} oct'.format(pSfExp[fam, con], bwExp[fam, con]), fontsize=12, horizontalalignment='center', verticalalignment='top', transform=all_plots[con,fam].transAxes); 

        # Remove top/right axis, put ticks only on bottom/left, despine
        all_plots[con, fam].spines['right'].set_visible(False);
        all_plots[con, fam].spines['top'].set_visible(False);
        all_plots[con, fam].xaxis.set_ticks_position('bottom');
        all_plots[con, fam].yaxis.set_ticks_position('left');
        sns.despine(ax=all_plots[con, fam], offset = 10);

f.legend(fontsize = 15, loc='upper right');
f.suptitle('cell #%d, loss %.2f|%.2f' % (cellNum, fitList_fl[cellNum-1]['NLL'], fitList_wg[cellNum-1]['NLL']), fontsize=25);

#########
# Plot secondary things - CRF, filter, normalization, nonlinearity, etc
#########

fDetails = plt.figure();
fDetails.set_size_inches(w=25, h=10);
detailSize = (3, 5);
# plot ori tuning
curr_ax = plt.subplot2grid(detailSize, (0, 2));
[plt.plot(expData['sfm']['exp']['ori'], oriResp, '%so' % c, clip_on=False, label=s) for oriResp, c, s in zip(oriModResps, modColors, modLabels)]; # Model responses
expPlt = plt.plot(expData['sfm']['exp']['ori'], expData['sfm']['exp']['oriRateMean'], 'o-', clip_on=False); # Exp responses
plt.xlabel('Ori (deg)', fontsize=12);
plt.ylabel('Response (ips)', fontsize=12);

# CRF - with values from TF simulation and the broken down (i.e. numerator, denominator separately) values from resimulated conditions
curr_ax = plt.subplot2grid(detailSize, (0, 1)); # default size is 1x1
consUse = expData['sfm']['exp']['con'];
plt.semilogx(consUse, expData['sfm']['exp']['conRateMean'], 'o-', clip_on=False); # Measured responses
[plt.plot(consUse, conResp, '%so' % c, clip_on=False, label=s) for conResp, c, s in zip(conModResps, modColors, modLabels)]; # Model responses
plt.xlabel('Con (%)', fontsize=20);
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=curr_ax, offset = 5);
 
#poisson test - mean/var for each condition (i.e. sfXdispXcon)
curr_ax = plt.subplot2grid(detailSize, (0, 0));
lower_bound = 1e-2;
plt.loglog([lower_bound, 1000], [lower_bound, 1000], 'k--');
meanList = (expData['sfm']['exp']['conRateMean'], expData['sfm']['exp']['oriRateMean'], np.array(sfMeanFlat));
varList = (expData['sfm']['exp']['conRateVar'], expData['sfm']['exp']['oriRateVar'], np.array(sfVarFlat));

for i in range(len(meanList)):
  gtLB = np.logical_and(meanList[i]>lower_bound, varList[i]>lower_bound);
  plt.loglog(meanList[i][gtLB], varList[i][gtLB], 'o');
# skeleton for plotting modulated poisson prediction                                                                                                                                                    
if lossType == 3: # i.e. modPoiss                                                                     varGain = modFit[7];
  mean_vals = np.logspace(-1, 2, 50);
  plt.loglog(mean_vals, mean_vals + varGain*np.square(mean_vals));
plt.xlabel('Mean (sps)');
plt.ylabel('Variance (sps^2)');
plt.title('Super-poisson?');
plt.axis('equal');
sns.despine(ax=curr_ax, offset=5, trim=False);

# plot model details - exc/suppressive components
omega = np.logspace(-2, 2, 1000);
sfExc = [];
for i in modFits:
  prefSf = i[0];
  dOrder = i[1];
  sfRel = omega/prefSf;
  s     = np.power(omega, dOrder) * np.exp(-dOrder/2 * np.square(sfRel));
  sMax  = np.power(prefSf, dOrder) * np.exp(-dOrder/2);
  sfExcCurr = s/sMax;
  sfExc.append(sfExcCurr);

inhSfTuning = getSuppressiveSFtuning();

# Compute weights for suppressive signals
nInhChan = expData['sfm']['mod']['normalization']['pref']['sf'];
nTrials =  inhSfTuning.shape[0];
inhWeight = genNormWeights(expData, nInhChan, gs_mean, gs_std, nTrials);
inhWeight = inhWeight[:, :, 0]; # genNormWeights gives us weights as nTr x nFilters x nFrames - we have only one "frame" here, and all are the same
# first, tuned norm:
sfNormTune = np.sum(-.5*(inhWeight*np.square(inhSfTuning)), 1);
sfNormTune = sfNormTune/np.amax(np.abs(sfNormTune));
# then, untuned norm:
inhAsym = 0;
inhWeight = [];
for iP in range(len(nInhChan)):
    inhWeight = np.append(inhWeight, 1 + inhAsym * (np.log(expData['sfm']['mod']['normalization']['pref']['sf'][iP]) - np.mean(np.log(expData['sfm']['mod']['normalization']['pref']['sf'][iP]))));
sfNorm = np.sum(-.5*(inhWeight*np.square(inhSfTuning)), 1);
sfNorm = sfNorm/np.amax(np.abs(sfNorm));
sfNorms = [sfNorm, sfNormTune];

# just setting up lines
curr_ax = plt.subplot2grid(detailSize, (1, 1));
plt.semilogx([omega[0], omega[-1]], [0, 0], 'k--')
plt.semilogx([.01, .01], [-1.5, 1], 'k--')
plt.semilogx([.1, .1], [-1.5, 1], 'k--')
plt.semilogx([1, 1], [-1.5, 1], 'k--')
plt.semilogx([10, 10], [-1.5, 1], 'k--')
plt.semilogx([100, 100], [-1.5, 1], 'k--')
# now the real stuff
[plt.semilogx(omega, exc, '%s' % c, label=s) for exc, c, s in zip(sfExc, modColors, modLabels)]
[plt.semilogx(omega, -norm, '%s--' % c, label=s) for norm, c, s in zip(sfNorms, modColors, modLabels)]
plt.xlim([omega[0], omega[-1]]);
plt.ylim([-0.1, 1.1]);
plt.xlabel('spatial frequency (c/deg)', fontsize=12);
plt.ylabel('Normalized response (a.u.)', fontsize=12);
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=curr_ax, offset=5);

# last but not least...and not last... response nonlinearity
modExps = [x[3] for x in modFits];
curr_ax = plt.subplot2grid(detailSize, (1, 2));
plt.plot([-1, 1], [0, 0], 'k--')
plt.plot([0, 0], [-.1, 1], 'k--')
[plt.plot(np.linspace(-1,1,100), np.power(np.maximum(0, np.linspace(-1,1,100)), modExp), '%s-' % c, label=s, linewidth=2) for modExp,c,s in zip(modExps, modColors, modLabels)]
plt.plot(np.linspace(-1,1,100), np.maximum(0, np.linspace(-1,1,100)), 'k--', linewidth=1)
plt.xlim([-1, 1]);
plt.ylim([-.1, 1]);
plt.text(0.5, 1.1, 'respExp: %.2f, %.2f' % (modExps[0], modExps[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=curr_ax, offset=5);

# print, in text, model parameters:
curr_ax = plt.subplot2grid(detailSize, (0, 4));
plt.text(0.5, 0.5, 'prefSf: %.3f, %.3f' % (modFits[0][0], modFits[1][0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.4, 'derivative order: %.3f, %.3f' % (modFits[0][1], modFits[1][1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.3, 'response scalar: %.3f, %.3f' % (modFits[0][4], modFits[1][4]), fontsize=12, horizontalalignment='center', verticalalignment='center');
plt.text(0.5, 0.2, 'sigma: %.3f, %.3f | %.3f, %.3f' % (np.power(10, modFits[0][2]), np.power(10, modFits[1][2]), modFits[0][2], modFits[1][2]), fontsize=12, horizontalalignment='center', verticalalignment='center');

# and now save it
allFigs = [f, fDetails];
#allFigs = [f, fDetails, fNorm];
if log_y:
  log_str = '_logy';
else:
  log_str = '';
saveName = "cell_%02d%s.pdf" % (cellNum, log_str);
pdf = pltSave.PdfPages(str(save_loc + saveName))
for fig in range(len(allFigs)): ## will open an empty extra figure :(
    pdf.savefig(allFigs[fig])
pdf.close()

