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
    
    # 08 = asymmetry of normalization (weights +/- linearly about SF mean)
    # OR
    # 08 = mean of gaussian which is used to apply weights to norm pool filters (as f'n of sf)
    # 09 = std of ...

# ### SF Diversity Project - plotting data, descriptive fits, and functional model fits

import os
import sys
import numpy as np
from helper_fcns import organize_modResp, flexible_Gauss, getSuppressiveSFtuning, compute_SF_BW, genNormWeights, random_in_range
import model_responses as mod_resp
from itertools import chain
import matplotlib
matplotlib.use('Agg') # why? so that we can get around having no GUI on cluster
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/Analysis/Functions/paul_plt_cluster.mplstyle');
sns.set(style='ticks');
from matplotlib import rcParams
rcParams['font.size'] = 20;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 4;
rcParams['axes.linewidth'] = 3;
rcParams['lines.markersize'] = 5
rcParams['font.style'] = 'oblique';
rcParams['legend.fontsize'] ='large'; # using a named size

import pdb

cellNum = int(sys.argv[1]);
fitType = int(sys.argv[2]);
normTypeArr= [];
nArgsIn = len(sys.argv) - 3; # we've already taken 3 arguments off (function all, which_cell, fit_type)
argInd = 3;
while nArgsIn > 0:
  normTypeArr.append(float(sys.argv[argInd]));
  nArgsIn = nArgsIn - 1;
  argInd = argInd + 1;


#save_loc = '/ser/1.2/p2/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/Analysis/Figures/'# CNS
#save_loc = '/home/pl1465/SF_diversity/Analysis/Figures/'; # prince
#data_loc = '/home/pl1465/SF_diversity/Analysis/Structures/'; # prince
save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/Analysis/Figures/'; # local machine
data_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/Analysis/Structures/'; # local machine

expName = 'dataList.npy'
fitBase = 'fitList_180430_LR';
if fitType == 1:
  fitSuf = '_sqrt.npy';
elif fitType == 2:
  fitSuf = '_poiss.npy';
elif fitType == 3:
  fitSuf = '_modPoiss.npy';
fitName = str(fitBase + fitSuf);
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

dL = np.load(data_loc + expName).item();
fitList = np.load(data_loc + fitName, encoding='latin1').item();
descrExpFits = np.load(data_loc + descrExpName, encoding='latin1').item();
descrModFits = np.load(data_loc + descrModName, encoding='latin1').item();

# #### Load data

expData = np.load(str(data_loc + dL['unitName'][cellNum-1] + '_sfm.npy')).item();
expResp = expData
modFit = fitList[cellNum-1]['params']; # 
descrExpFit = descrExpFits[cellNum-1]['params']; # nFam x nCon x nDescrParams
descrModFit = descrModFits[cellNum-1]['params']; # nFam x nCon x nDescrParams

if len(normTypeArr) == 3: # i.e. we've passed in gs_mean, gs_std, then replace...
  modFit[-2] = normTypeArr[1];
  modFit[-1] = normTypeArr[2];

ignore, modResp, normTypeArr = mod_resp.SFMGiveBof(modFit, expData, normTypeArr);
norm_type = normTypeArr[0];
if norm_type == 1:
  gs_mean = normTypeArr[1]; # guaranteed to exist after call to .SFMGiveBof, if norm_type == 1
  gs_std = normTypeArr[2]; # guaranteed to exist ...
#modRespAll = mod_resp.SFMGiveBof(modParamsCurr, expData, normTypeArr)[1]; # NOTE: We're taking [1] (i.e. second) output of SFMGiveBof
oriModResp, conModResp, sfmixModResp, allSfMix = organize_modResp(modResp, expData['sfm']['exp']['trial'])
oriExpResp, conExpResp, sfmixExpResp, allSfMixExp = organize_modResp(expData['sfm']['exp']['trial']['spikeCount'], \
                                                                           expData['sfm']['exp']['trial'])
#pdb.set_trace();

# allSfMix is (nFam, nCon, nCond, nReps) where nCond is 11, # of SF centers and nReps is usually 10
modLow = np.nanmin(allSfMix, axis=3);
modHigh = np.nanmax(allSfMix, axis=3);
modAvg = np.nanmean(allSfMix, axis=3);
modSponRate = modFit[6];

findNan = np.isnan(allSfMixExp);
nonNan = np.sum(findNan == False, axis=3); # how many valid trials are there for each fam x con x center combination?
allExpSEM = np.nanstd(allSfMixExp, axis=3) / np.sqrt(nonNan); # SEM

# plot model details - exc/suppressive components
prefSf    = modFit[0];
dOrder    = modFit[1]
omega = np.logspace(-2, 2, 1000);
sfRel = omega/prefSf;
s     = np.power(omega, dOrder) * np.exp(-dOrder/2 * np.square(sfRel));
sMax  = np.power(prefSf, dOrder) * np.exp(-dOrder/2);
sfExc = s/sMax;

inhSfTuning = getSuppressiveSFtuning();

# Compute weights for suppressive signals
nInhChan = expData['sfm']['mod']['normalization']['pref']['sf'];
if norm_type == 1:
  nTrials =  inhSfTuning.shape[0];
  inhWeight = genNormWeights(expData, nInhChan, gs_mean, gs_std, nTrials);
  inhWeight = inhWeight[:, :, 0]; # genNormWeights gives us weights as nTr x nFilters x nFrames - we have only one "frame" here, and all are the same
else:
  if modFit[8]: # i.e. if this parameter exists...
    inhAsym = modFit[8];
  else:
    inhAsym = 0;

  inhWeight = [];
  for iP in range(len(nInhChan)):
      inhWeight = np.append(inhWeight, 1 + inhAsym * (np.log(expData['sfm']['mod']['normalization']['pref']['sf'][iP]) - np.mean(np.log(expData['sfm']['mod']['normalization']['pref']['sf'][iP]))));

sfNorm = np.sum(-.5*(inhWeight*np.square(inhSfTuning)), 1);
sfNorm = sfNorm/np.amax(np.abs(sfNorm));
# construct by hand for now; 5 dispersions with the old stimulus set
val_con_by_disp = [];
val_con_by_disp.append(np.array([1, 0.688, 0.473, 0.325, 0.224, 0.154, 0.106, 0.073, 0.05, 0.01]));
val_con_by_disp.append(np.array([1, 0.688, 0.473, 0.325]));
val_con_by_disp.append(np.array([1, 0.688, 0.473, 0.325]));
val_con_by_disp.append(np.array([1, 0.688, 0.473, 0.325]));
val_con_by_disp.append(np.array([1, 0.688, 0.473, 0.325]));

v_sfs = np.logspace(np.log10(0.3), np.log10(10), 11); # for now
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
simsAx[0].semilogx([.01, .01], [-1.5, 1], 'k--')
simsAx[0].semilogx([.1, .1], [-1.5, 1], 'k--')
simsAx[0].semilogx([1, 1], [-1.5, 1], 'k--')
simsAx[0].semilogx([10, 10], [-1.5, 1], 'k--')
simsAx[0].semilogx([100, 100], [-1.5, 1], 'k--')
# now the real stuff
ex = simsAx[0].semilogx(omega, sfExc, 'k-')
nm = simsAx[0].semilogx(omega, -sfNorm, 'r-', linewidth=2.5);
simsAx[0].set_xlim([omega[0], omega[-1]]);
simsAx[0].set_ylim([-1.5, 1]);
simsAx[0].set_xlabel('SF (cpd)', fontsize=12);
simsAx[0].set_ylabel('Normalized response (a.u.)', fontsize=12);
simsAx[0].set_title('CELL %d' % (cellNum), fontsize=20);
simsAx[0].legend([ex[0], nm[0]], ('excitatory %.2f' % (modFit[0]), 'normalization %.2f' % (np.exp(modFit[-2]))));
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=simsAx[0], offset=5);

for d in range(nFam):
    
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
          sf_iResp, ignore, ignore, ignore = mod_resp.SFMsimulate(modFit, expData, d+1, v_cons[c], sf_i);
          curr_resps.append(sf_iResp[0]); # SFMsimulate returns array - unpack it

        # plot data
        col = [c/float(n_v_cons), c/float(n_v_cons), c/float(n_v_cons)];
        respAbBaseline = curr_resps - modSponRate;
        #print('Simulated at %d|%d sfs: %d above baseline' % (len(v_sfs), len(curr_resps), sum(respAbBaseline>1e-1)));
        curr_line, = simsAx[d+1][0].plot(v_sfs[respAbBaseline>1e-1], respAbBaseline[respAbBaseline>1e-1], '-o', clip_on=False, color=col);
        lines.append(curr_line);

    simsAx[d+1][0].set_aspect('equal', 'box'); 
    simsAx[d+1][0].set_xlim((0.5*min(v_sfs), 1.2*max(v_sfs)));
    #simsAx[d+1][0].set_ylim((5e-2, 1.5*maxResp));
    simsAx[d+1][0].set_xlabel('sf (c/deg)'); 

    simsAx[d+1][0].set_ylabel('resp above baseline (sps)');
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
          con_iResp, ignore, ignore, ignore = mod_resp.SFMsimulate(modFit, expData, d+1, con_i, sf_curr);
          curr_resps.append(con_iResp[0]); # unpack the array returned by SFMsimulate

        col = [sf_i/float(n_v_sfs), sf_i/float(n_v_sfs), sf_i/float(n_v_sfs)];
        respAbBaseline = curr_resps - modSponRate;
        print('rAB = %s ||| v_cons %s' % (respAbBaseline, v_cons));
        line_curr, = simsAx[d+1][1].plot(v_cons[respAbBaseline>1e-1], respAbBaseline[respAbBaseline>1e-1], '-o', color=col, clip_on=False);
        lines_log.append(line_curr);

    simsAx[d+1][1].set_xlim([1e-2, 1]);
    #simsAx[d+1][1].set_ylim([1e-2, 1.5*maxResp]);
    simsAx[d+1][1].set_aspect('equal', 'box')
    simsAx[d+1][1].set_xscale('log');
    simsAx[d+1][1].set_yscale('log');
    simsAx[d+1][1].set_xlabel('contrast');

    simsAx[d+1][1].set_ylabel('resp above baseline (sps)');
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
saveName = "cell_%d_simulate.pdf" % (cellNum)
pdfSv = pltSave.PdfPages(str(save_loc + saveName));
for ff in fSims:
    pdfSv.savefig(ff)
    plt.close(ff)
pdfSv.close();
