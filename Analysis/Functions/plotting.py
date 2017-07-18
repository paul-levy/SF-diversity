
# -*- coding: utf-8 -*-

# # Plotting

    # parameters
    # 00 = preferred direction of motion (degrees)
    # 01 = preferred spatial frequency   (cycles per degree)
    # 02 = aspect ratio 2-D Gaussian
    # 03 = derivative order in space
    # 04 = directional selectivity
    # 05 = normalization constant        (log10 basis)
    # 06 = response exponent
    # 07 = response scalar
    # 08 = early additive noise
    # 09 = late additive noise
    # 10 = variance of response gain



# ### SF Diversity Project - plotting data, descriptive fits, and functional model fits

# #### Pick your cell

# In[ ]:

# #### Set constants

# In[441]:

import sys
import numpy as np
from helper_fcns import organize_modResp, flexible_Gauss, getSuppressiveSFtuning, compute_SF_BW
import model_responses as mod_resp
import matplotlib
matplotlib.use('Agg') # why? so that we can get around having no GUI on cluster
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave

import pdb

cellNum = int(sys.argv[1]);

save_loc = '/home/pl1465/SF_diversity/Analysis/Figures/';
#save_loc = '/ser/1.2/p2/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/Analysis/Figures/'# CNS
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

dL = np.load(data_loc + expName).item();
fitList = np.load(data_loc + fitName, encoding='latin1'); # no '.item()' because this is array of dictionaries...
descrExpFits = np.load(data_loc + descrExpName, encoding='latin1').item();
descrModFits = np.load(data_loc + descrModName, encoding='latin1').item();

# #### Load data

# In[258]:

expData = np.load(str(data_loc + dL['unitName'][cellNum-1] + '_sfm.npy')).item();
modFit = fitList[cellNum-1]['params']; # 
descrExpFit = descrExpFits[cellNum-1]['params']; # nFam x nCon x nDescrParams
descrModFit = descrModFits[cellNum-1]['params']; # nFam x nCon x nDescrParams

a, modResp = mod_resp.SFMGiveBof(modFit, expData);
expResp = expData
oriModResp, conModResp, sfmixModResp, allSfMix = organize_modResp(modResp, expData['sfm']['exp']['trial'])
oriExpResp, conExpResp, sfmixExpResp, allSfMixExp = organize_modResp(expData['sfm']['exp']['trial']['spikeCount'], \
                                                                           expData['sfm']['exp']['trial'])
modLow = np.nanmin(allSfMix, axis=3)
modHigh = np.nanmax(allSfMix, axis=3)

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

# #### Plot the main stuff - sfMix experiment with model predictions and descriptive fits

# In[281]:

f, all_plots = plt.subplots(nCon, nFam, sharex=True, sharey=True, figsize=(25,8))
expSfCent = expData['sfm']['exp']['sf'][0][0];
expResponses = expData['sfm']['exp']['sfRateMean'];

# plot experiment and models
for con in reversed(range(nCon)): # contrast
    yMax = 1.25*np.maximum(np.amax(expResponses[0][con]), np.amax(modHigh[0, con, :])); # we assume that 1.25x Max response for single grating will be enough
    all_plots[con, 0].set_ylim([-1, yMax]);
    for fam in reversed(range(nFam)): # family        
        expPoints = all_plots[con, fam].errorbar(expSfCent, expResponses[fam][con], allExpSEM[fam, con, :],\
                                                 linestyle='None', marker='o', color='b');
        modRange = all_plots[con, fam].fill_between(expSfCent, modLow[fam,con,:], \
                                                    modHigh[fam, con,:], color='r', alpha=0.2)
        sponRate = all_plots[con, fam].axhline(expData['sfm']['exp']['sponRateMean'], color='k', linestyle='dashed');
        all_plots[con,fam].set_xscale('log');
        
        # pretty
        all_plots[con,fam].tick_params(labelsize=15, width=1, length=8);
        all_plots[con,fam].tick_params(width=1, length=4, which='minor'); # minor ticks, too...
        if con == 1:
            all_plots[con, fam].set_xlabel('sf center (cpd)', fontsize=20);
        if fam == 0:
            all_plots[con, fam].set_ylabel('Response (ips)', fontsize=20);
       
        all_plots[con,fam].text(0.5,1.05, 'mod: {:.2f} cpd | {:.2f} oct'.format(pSfMod[fam, con], bwMod[fam, con]), fontsize=12, horizontalalignment='center', verticalalignment='top', transform=all_plots[con,fam].transAxes); 
        all_plots[con,fam].text(0.5,1.10, 'exp: {:.2f} cpd | {:.2f} oct'.format(pSfExp[fam, con], bwExp[fam, con]), fontsize=12, horizontalalignment='center', verticalalignment='top', transform=all_plots[con,fam].transAxes); 
            
f.legend((expPoints[0], modRange, sponRate), ('data +- 1 s.e.m.', 'model range', 'spontaneous f.r.'), fontsize = 15, loc='right');
f.suptitle('SF mixture experiment', fontsize=25);

# In[439]:

fDetails, all_plots = plt.subplots(3,5, figsize=(25,10))
# plot ori, CRF tuning
modPlt=all_plots[0, 0].plot(expData['sfm']['exp']['ori'], oriModResp, 'ro'); # Model responses
expPlt=all_plots[0, 0].plot(expData['sfm']['exp']['ori'], expData['sfm']['exp']['oriRateMean'], 'o-'); # Exp responses
all_plots[0, 0].set_xlabel('Ori (deg)', fontsize=20);
all_plots[0, 0].set_ylabel('Response (ips)', fontsize=20);

all_plots[0, 1].semilogx(expData['sfm']['exp']['con'], conModResp, 'ro'); # Model responses
all_plots[0, 1].semilogx(expData['sfm']['exp']['con'], expData['sfm']['exp']['conRateMean'], 'o-'); # Model responses
all_plots[0, 1].set_xlabel('Con (%)', fontsize=20);

all_plots[0,2].axis('off');
all_plots[0,3].axis('off');
#all_plots[0,4].axis('off');
all_plots[1,3].axis('off');
all_plots[1,4].axis('off');

# plot model details - filter
imSizeDeg = expData['sfm']['exp']['size'];
pixSize   = 0.0028; # fixed from Robbe
prefSf    = modFit[1];
prefOri   = np.pi/180 * modFit[0];
dOrder    = modFit[3]
aRatio    = modFit[2];
filtTemp  = mod_resp.oriFilt(imSizeDeg, pixSize, prefSf, prefOri, dOrder, aRatio);
filt      = (filtTemp - filtTemp[0,0])/ np.amax(np.abs(filtTemp - filtTemp[0,0]));
all_plots[1,0].imshow(filt, cmap='gray');
all_plots[1,0].axis('off');
all_plots[1,0].set_title('Filter in space', fontsize=20)

# plot model details - exc/suppressive components
omega = np.logspace(-2, 2, 1000);

sfRel = omega/prefSf;
s     = np.power(omega, dOrder) * np.exp(-dOrder/2 * np.square(sfRel));
sMax  = np.power(prefSf, dOrder) * np.exp(-dOrder/2);
sfExc = s/sMax;

inhSfTuning = getSuppressiveSFtuning();

# Compute weights for suppressive signals
nInhChan = expData['sfm']['mod']['normalization']['pref']['sf'];
inhWeight = [];
for iP in range(len(nInhChan)):
    # '0' because no asymmetry
    inhWeight = np.append(inhWeight, 1 + 0 * (np.log(expData['sfm']['mod']['normalization']['pref']['sf'][iP]) - np.mean(np.log(expData['sfm']['mod']['normalization']['pref']['sf'][iP]))));
           
sfInh = np.ones(omega.shape) / np.amax(modHigh);
sfNorm = np.sum(-.5*(inhWeight*np.square(inhSfTuning)), 1);
sfNorm = sfNorm/np.amax(np.abs(sfNorm));

# just setting up lines
all_plots[1,1].semilogx([omega[0], omega[-1]], [0, 0], 'k--')
all_plots[1,1].semilogx([.01, .01], [-1.5, 1], 'k--')
all_plots[1,1].semilogx([.1, .1], [-1.5, 1], 'k--')
all_plots[1,1].semilogx([1, 1], [-1.5, 1], 'k--')
all_plots[1,1].semilogx([10, 10], [-1.5, 1], 'k--')
all_plots[1,1].semilogx([100, 100], [-1.5, 1], 'k--')
# now the real stuff
all_plots[1,1].semilogx(omega, sfExc, 'k-')
all_plots[1,1].semilogx(omega, sfInh, 'r--', linewidth=2);
all_plots[1,1].semilogx(omega, sfNorm, 'r-', linewidth=1);
all_plots[1,1].set_xlim([omega[0], omega[-1]]);
all_plots[1,1].set_ylim([-1.5, 1]);
all_plots[1, 1].set_xlabel('SF (cpd)', fontsize=20);
all_plots[1, 1].set_ylabel('Normalized response (a.u.)', fontsize=20);

for i in range(len(all_plots)):
    for j in range (len(all_plots[0])):
        all_plots[i,j].tick_params(labelsize=15, width=1, length=8);
        all_plots[i,j].tick_params(width=1, length=4, which='minor'); # minor ticks, too...

# last but not least...and not last... response nonlinearity
all_plots[1,2].plot([-1, 1], [0, 0], 'k--')
all_plots[1,2].plot([0, 0], [-.1, 1], 'k--')
all_plots[1,2].plot(np.linspace(-1,1,100), np.power(np.maximum(0, np.linspace(-1,1,100)), modFit[6]), 'k-', linewidth=2)
all_plots[1,2].plot(np.linspace(-1,1,100), np.maximum(0, np.linspace(-1,1,100)), 'k--', linewidth=1)
all_plots[1,2].set_xlim([-1, 1]);
all_plots[1,2].set_ylim([-.1, 1]);
    
# actually last - CRF at different dispersion levels
crf_row = len(all_plots)-1; # we're putting the CRFs in the last row of this plot
crf_sfIndex = np.argmin(abs(expSfCent - descrExpFit[0][0][2])); # get mu (i.e. prefSf) as measured at high contrast, single grating and find closest presented SF (index)
crf_sfVal = expSfCent[crf_sfIndex]; # what's the closest SF to the pref that was presented?
crf_cons = expData['sfm']['exp']['con']; # what contrasts to sim. from model? Same ones used in exp
crf_sim = np.zeros((nFam, len(crf_cons))); # create nparray for results
# first, run the CRFs...
for i in range(nFam):
    print('simulating CRF for family ' + str(i+1));
    for j in range(len(crf_cons)):
        crf_sim[i, j] = np.mean(mod_resp.SFMsimulate(modFit, expData, i+1, crf_cons[j], crf_sfVal)); # take mean of the returned simulations (10 repetitions per stim. condition)

# now plot!
for i in range(len(all_plots[0])):
    all_plots[crf_row, i].semilogx(1, expResponses[i][0][crf_sfIndex], 'bo'); # exp response - high con
    all_plots[crf_row, i].semilogx(0.33, expResponses[i][1][crf_sfIndex], 'bo'); # exp response - low con
    all_plots[crf_row, i].semilogx(crf_cons, crf_sim[i, :], 'ro-'); # model resposes - range of cons
    all_plots[crf_row, i].set_xlabel('Con (%)', fontsize=20);    
    all_plots[crf_row, i].set_xlim([1e-2, 1e0]);
    all_plots[crf_row, i].set_ylim([0, 1.05*np.amax(np.maximum(crf_sim[0, :], expResponses[0][0][crf_sfIndex]))]);
    if i == 0:
      all_plots[crf_row, i].set_ylabel('Resp. amp (sps)');

fDetails.legend((modPlt[0], expPlt[0]), ('model', 'experiment'), fontsize = 15, loc='center left');
fDetails.suptitle('SF mixture - details', fontsize=25);

# print, in text, model parameters:
all_plots[0, 4].text(0.5, 0.6, 'prefOri: {:.3f}'.format(modFit[0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
all_plots[0, 4].text(0.5, 0.5, 'dir. selectivity: {:.3f}'.format(modFit[3]), fontsize=12, horizontalalignment='center', verticalalignment='center');
all_plots[0, 4].text(0.5, 0.4, 'prefSf: {:.3f}'.format(modFit[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
all_plots[0, 4].text(0.5, 0.3, 'aspect ratio: {:.3f}'.format(modFit[2]), fontsize=12, horizontalalignment='center', verticalalignment='center');
all_plots[0, 4].text(0.5, 0.2, 'response scalar: {:.3f}'.format(modFit[7]), fontsize=12, horizontalalignment='center', verticalalignment='center');
all_plots[0, 4].text(0.5, 0.1, 'sigma: {:.3f} | {:.3f}'.format(np.power(10, modFit[5]), modFit[5]), fontsize=12, horizontalalignment='center', verticalalignment='center');

# In[444]:

# and now save it
bothFigs = [f, fDetails];
saveName = "cellSimpMod_%d.pdf" % cellNum
pdf = pltSave.PdfPages(str(save_loc + saveName))
for fig in range(len(bothFigs)): ## will open an empty extra figure :(
    pdf.savefig(bothFigs[fig])
pdf.close()

