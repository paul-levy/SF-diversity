
# -*- coding: utf-8 -*-

# ### SF Diversity Project - plotting data summaries (e.g. distribution of bandwidths, prefSf, etc.)

import sys
import numpy as np
from helper_fcns import organize_modResp, flexible_Gauss, getSuppressiveSFtuning, compute_SF_BW
import model_responses as mod_resp
import matplotlib
matplotlib.use('Agg') # why? so that we can get around having no GUI on cluster
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave

import pdb # if you need to debug, eh?

save_loc = '/home/pl1465/SF_diversity/Analysis/Figures/';
data_loc = '/home/pl1465/SF_diversity/Analysis/Structures/';
expName = 'dataList.npy'
fitName = 'fitListSimplified.npy'
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

# Load data
dL = np.load(data_loc + expName).item();
fitList = np.load(data_loc + fitName, encoding='latin1'); # no '.item()' because this is array of dictionaries...
descrExpFits = np.load(data_loc + descrExpName, encoding='latin1').item();
descrModFits = np.load(data_loc + descrModName, encoding='latin1').item();

nCells = len(descrExpFits);

# Do some analysis of bandwidth, prefSf

bwMod = np.ones((nCells, nFam, nCon)) * np.nan;
bwExp = np.ones((nCells, nFam, nCon)) * np.nan;
pSfMod = np.ones((nCells, nFam, nCon)) * np.nan;
pSfExp = np.ones((nCells, nFam, nCon)) * np.nan;

for c in range(nCells):
  for f in range(nFam):
        
      ignore, bwExp[c, f, 0] = compute_SF_BW(descrExpFits[c]['params'][f, 0, :], height, sf_range)
      ignore, bwExp[c, f, 1] = compute_SF_BW(descrExpFits[c]['params'][f, 1, :], height, sf_range)
      pSfExp[c, f, 0] = descrExpFits[c]['params'][f, 0, muLoc]
      pSfExp[c, f, 1] = descrExpFits[c]['params'][f, 1, muLoc]

# start the plotting

nPlotTypes = 4;
fig, all_plots = plt.subplots(nPlotTypes, nFam, sharex='col', sharey='row', figsize=(25,8))

# first plot - prefSf by dispersion level, plotted at high & low contrast
nBinsPSF = 15;
pSfBins = np.logspace(np.log2(0.1), np.log2(10), nBinsPSF, base=2);

#pdb.set_trace();

for f in range(nFam):
  all_plots[0, f].hist(pSfExp[:, f, 0], pSfBins, label='high con', alpha=0.5, rwidth=0.8);   
  all_plots[0, f].hist(pSfExp[:, f, 1], pSfBins, label='low con', alpha=0.5, rwidth=0.8);   
  all_plots[0, f].set_xscale('log');
  all_plots[0, f].tick_params(labelsize=15, width=1, length=8, which='major', direction='out');
  all_plots[0, f].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
  all_plots[0, f].text(0.5,1.1, 'median (hi/lo): {:.2f} | {:.2f} cpd'.format(np.median(pSfExp[:, f, 0]), np.median(pSfExp[:, f, 1])), fontsize=12, horizontalalignment='center', verticalalignment='top', transform=all_plots[0, f].transAxes);

# second plot - sfBW by dispersion level, plotted at high & low contrast
nBinsBW = 15;
bwExpBins = np.linspace(0, 10, nBinsBW);

#pdb.set_trace();

for f in range(nFam):
  valid = np.logical_and(~np.isnan(bwExp[:, f, 0]), ~np.isnan(bwExp[:, f, 1]));
  all_plots[1, f].hist(bwExp[valid, f, 0], pSfBins, label='high con', alpha=0.5, rwidth=0.8);   
  all_plots[1, f].hist(bwExp[valid, f, 1], pSfBins, label='low con', alpha=0.5, rwidth=0.8);   
  all_plots[1, f].tick_params(labelsize=15, width=1, length=8, which='major', direction='out');
  all_plots[1, f].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
  all_plots[1, f].text(0.5,1.1, 'median (hi/lo): {:.2f} | {:.2f} oct'.format(np.median(bwExp[valid, f, 0]), np.median(bwExp[valid, f, 1])), fontsize=12, horizontalalignment='center', verticalalignment='top', transform=all_plots[1, f].transAxes);

# and now save it
bothFigs = [fig];
saveName = "popSummary.pdf";
pdf = pltSave.PdfPages(str(save_loc + saveName))
for fig in range(len(bothFigs)):
    pdf.savefig(bothFigs[fig])
pdf.close()

