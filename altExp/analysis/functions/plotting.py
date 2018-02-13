# coding: utf-8

######################## To do:
#2/5/18 - "Clean up" plotting code s.t. you only make the plots you want, keep # plots to a minimum

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import helper_fcns

import pdb

import sys # so that we can import model_responses (in different folder)
import model_responses

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/Analysis/Functions/paul_plt_cluster.mplstyle');
plt.rc('legend',fontsize='medium') # using a named size

which_cell = int(sys.argv[1]);

# at CNS
# dataPath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/recordings/';
# savePath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/analysis/';
# personal mac
#dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/altExp/analysis/structures/';
#save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/altExp/analysis/figures/';
# prince cluster
dataPath = '/home/pl1465/SF_diversity/altExp/analysis/structures/';
save_loc = '/home/pl1465/SF_diversity/altExp/analysis/figures/';

fitListName = 'fitList_180105.npy';

conDig = 3; # round contrast to the 3rd digit

dataList = np.load(dataPath + 'dataList.npy', encoding='latin1').item();

cellStruct = np.load(dataPath + dataList['unitName'][which_cell-1] + '_sfm.npy', encoding='latin1').item();

# #### Load descriptive model fits, comp. model fits

descrFits = np.load(dataPath + 'descrFits.npy', encoding = 'latin1').item();
descrFits = descrFits[which_cell-1]['params']; # just get this cell

modParams = np.load(dataPath + fitListName, encoding= 'latin1').item();
modParamsCurr = modParams[which_cell-1]['params'];

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

data = cellStruct['sfm']['exp']['trial'];

modRespAll = model_responses.SFMGiveBof(modParamsCurr, cellStruct)[1];
resp, stimVals, val_con_by_disp, validByStimVal, modResp = helper_fcns.tabulate_responses(cellStruct, modRespAll);
blankMean, _ = helper_fcns.blankResp(cellStruct); 
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

# #### Plots by dispersion

fDisp = []; dispAx = [];

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);    

for d in range(nDisps):
    
    v_cons = val_con_by_disp[d];
    n_v_cons = len(v_cons);
    
    fCurr, dispCurr = plt.subplots(n_v_cons, 2, figsize=(40, n_v_cons*10), sharey=False);
    fDisp.append(fCurr)
    dispAx.append(dispCurr);
    
    maxResp = np.max(np.max(respMean[d, ~np.isnan(respMean[d, :, :])]));
#    maxResp = np.maximum(np.max(np.max(respMean[d, ~np.isnan(respMean[d, :, :])])), np.max(np.max(predMean[d, ~np.isnan(respMean[d, :, :])])));
    
    for c in reversed(range(n_v_cons)):
        c_plt_ind = len(v_cons) - c - 1;
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);        

	# plot pred/measured ratio
        dispAx[d][c_plt_ind, 1].plot(all_sfs[v_sfs], np.divide(predMean[d, v_sfs, v_cons[c]]-blankMean, respMean[d, v_sfs, v_cons[c]]-blankMean), clip_on=False);
	dispAx[d][c_plt_ind, 1].axhline(1, clip_on=False, linestyle='dashed');
        
        # plot data
        dispAx[d][c_plt_ind, 0].errorbar(all_sfs[v_sfs], respMean[d, v_sfs, v_cons[c]], 
                                      respStd[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);

        # plot linear superposition prediction
#        dispAx[d][c_plt_ind, 0].errorbar(all_sfs[v_sfs], predMean[d, v_sfs, v_cons[c]], 
#                                      predStd[d, v_sfs, v_cons[c]], fmt='p', clip_on=False);

        # plot descriptive model fit
        curr_mod_params = descrFits[d, v_cons[c], :];
        #dispAx[d][c_plt_ind, 0].plot(sfs_plot, helper_fcns.flexible_Gauss(curr_mod_params, sfs_plot), clip_on=False)
        
	# plot model fits
	dispAx[d][c_plt_ind, 0].fill_between(all_sfs[v_sfs], modLow[d, v_sfs, v_cons[c]], \
                                      modHigh[d, v_sfs, v_cons[c]], color='r', alpha=0.2);
	dispAx[d][c_plt_ind, 0].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], 'r-', alpha=0.7, clip_on=False);

	for i in range(2):

          dispAx[d][c_plt_ind, i].set_xlim((min(all_sfs), max(all_sfs)));
        
          dispAx[d][c_plt_ind, i].set_xscale('log');
          dispAx[d][c_plt_ind, i].set_xlabel('sf (c/deg)'); 
          dispAx[d][c_plt_ind, i].set_title('D%d: contrast: %.3f' % (d, all_cons[v_cons[c]]));

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
          dispAx[d][c_plt_ind, i].tick_params(labelsize=15, width=1, length=8, direction='out');
          dispAx[d][c_plt_ind, i].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
	  dispAx[d][c_plt_ind, i].spines['right'].set_visible(False);
	  dispAx[d][c_plt_ind, i].spines['top'].set_visible(False);
	  dispAx[d][c_plt_ind, i].xaxis.set_ticks_position('bottom');
	  dispAx[d][c_plt_ind, i].yaxis.set_ticks_position('left');

        dispAx[d][c_plt_ind, 0].set_ylim((0, 1.5*maxResp));
        dispAx[d][c_plt_ind, 0].set_ylabel('resp (sps)');
        dispAx[d][c_plt_ind, 1].set_ylabel('ratio (pred:measure)');
	dispAx[d][c_plt_ind, 1].set_ylim((1e-1, 1e3));
        dispAx[d][c_plt_ind, 1].set_yscale('log');


saveName = "/cell_%d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'byDisp/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close();

# #### All SF tuning on one graph, split by dispersion

fDisp = []; dispAx = [];

sfs_plot = np.logspace(np.log10(all_sfs[0]), np.log10(all_sfs[-1]), 100);    

for d in range(nDisps):
    
    v_cons = val_con_by_disp[d];
    n_v_cons = len(v_cons);
    
    fCurr, dispCurr = plt.subplots(1, 1, figsize=(40, 20));
    fDisp.append(fCurr)
    dispAx.append(dispCurr);
    
    maxResp = np.max(np.max(respMean[d, ~np.isnan(respMean[d, :, :])]));
    
    lines = [];
    for c in reversed(range(n_v_cons)):
        v_sfs = ~np.isnan(respMean[d, :, v_cons[c]]);        

        # plot data
	col = [c/float(n_v_cons), c/float(n_v_cons), c/float(n_v_cons)];
	respAbBaseline = respMean[d, v_sfs, v_cons[c]] - blankMean;
        curr_line, = dispAx[d].plot(all_sfs[v_sfs][respAbBaseline>1e-1], respAbBaseline[respAbBaseline>1e-1], '-o', clip_on=False, color=col);
	lines.append(curr_line);

    dispAx[d].set_aspect('equal', 'box'); 
    dispAx[d].set_xlim((0.5*min(all_sfs), 1.2*max(all_sfs)));
    dispAx[d].set_ylim((5e-2, 1.5*maxResp));

    dispAx[d].set_xscale('log');
    dispAx[d].set_yscale('log');
    dispAx[d].set_xlabel('sf (c/deg)'); 

    # Set ticks out, remove top/right axis, put ticks only on bottom/left
    dispAx[d].tick_params(labelsize=15, width=2, length=16, direction='out');
    dispAx[d].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
    dispAx[d].spines['right'].set_visible(False);
    dispAx[d].spines['top'].set_visible(False);
    dispAx[d].xaxis.set_ticks_position('bottom');
    dispAx[d].yaxis.set_ticks_position('left');

    dispAx[d].set_ylabel('resp above baseline (sps)');
    dispAx[d].set_title('D%d - sf tuning' % (d));
    dispAx[d].legend(lines, [str(i) for i in reversed(all_cons[v_cons])], loc=0);

saveName = "/allCons_cell_%d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'byDisp/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fDisp:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()

# #### Plot just sfMix contrasts

# i.e. highest (up to) 4 contrasts for each dispersion

mixCons = 4;
maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));
#maxResp = np.maximum(np.max(np.max(np.max(respMean[~np.isnan(respMean)]))), np.max(np.max(np.max(predMean[~np.isnan(predMean)]))));

f, sfMixAx = plt.subplots(mixCons, nDisps, figsize=(40, 30));

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
                                       respStd[d, v_sfs, v_cons[c]], fmt='o', clip_on=False);
        # plot linear superposition prediction
#        sfMixAx[c_plt_ind, d].errorbar(all_sfs[v_sfs], predMean[d, v_sfs, v_cons[c]], 
#                                       predStd[d, v_sfs, v_cons[c]], fmt='p', clip_on=False);

        # plot descriptive model fit
        curr_mod_params = descrFits[d, v_cons[c], :];
        sfMixAx[c_plt_ind, d].plot(sfs_plot, helper_fcns.flexible_Gauss(curr_mod_params, sfs_plot), clip_on=False)

	# plot model fits
	sfMixAx[c_plt_ind, d].fill_between(all_sfs[v_sfs], modLow[d, v_sfs, v_cons[c]], \
                                      modHigh[d, v_sfs, v_cons[c]], color='r', alpha=0.2);
	sfMixAx[c_plt_ind, d].plot(all_sfs[v_sfs], modAvg[d, v_sfs, v_cons[c]], 'r-', alpha=0.7, clip_on=False);

        sfMixAx[c_plt_ind, d].set_xlim((np.min(all_sfs), np.max(all_sfs)));
        sfMixAx[c_plt_ind, d].set_ylim((0, 1.5*maxResp));
        sfMixAx[c_plt_ind, d].set_xscale('log');
        sfMixAx[c_plt_ind, d].set_xlabel('sf (c/deg)');
        sfMixAx[c_plt_ind, d].set_ylabel('resp (sps)');

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        sfMixAx[c_plt_ind, d].tick_params(labelsize=15, width=1, length=8, direction='out');
        sfMixAx[c_plt_ind, d].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
	sfMixAx[c_plt_ind, d].spines['right'].set_visible(False);
	sfMixAx[c_plt_ind, d].spines['top'].set_visible(False);
	sfMixAx[c_plt_ind, d].xaxis.set_ticks_position('bottom');
	sfMixAx[c_plt_ind, d].yaxis.set_ticks_position('left');
        
#########
# Plot secondary things - filter, normalization, nonlinearity, etc
#########

fDetails, all_plots = plt.subplots(3,5, figsize=(25,10))

all_plots[0,2].axis('off');
all_plots[0,3].axis('off');
#all_plots[0,4].axis('off');
all_plots[1,3].axis('off');
all_plots[1,4].axis('off');

# plot model details - filter
imSizeDeg = cellStruct['sfm']['exp']['size'];
pixSize   = 0.0028; # fixed from Robbe
prefSf    = modParamsCurr[0];
dOrder    = modParamsCurr[1]
prefOri = 0; # just fixed value since no model param for this
aRatio = 1; # just fixed value since no model param for this
filtTemp  = model_responses.oriFilt(imSizeDeg, pixSize, prefSf, prefOri, dOrder, aRatio);
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

inhSfTuning = helper_fcns.getSuppressiveSFtuning();

# Compute weights for suppressive signals
inhAsym = 0;
#inhAsym = modParamsCurr[8];
nInhChan = cellStruct['sfm']['mod']['normalization']['pref']['sf'];
inhWeight = [];
for iP in range(len(nInhChan)):
    # '0' because no asymmetry
    inhWeight = np.append(inhWeight, 1 + inhAsym * (np.log(cellStruct['sfm']['mod']['normalization']['pref']['sf'][iP]) - np.mean(np.log(cellStruct['sfm']['mod']['normalization']['pref']['sf'][iP]))));
           
sfInh = 0 * np.ones(omega.shape) / np.amax(modHigh); # mult by 0 because we aren't including a subtractive inhibition in model for now 7/19/17
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
# Remove top/right axis, put ticks only on bottom/left
all_plots[1, 1].spines['right'].set_visible(False);
all_plots[1, 1].spines['top'].set_visible(False);
all_plots[1, 1].xaxis.set_ticks_position('bottom');
all_plots[1, 1].yaxis.set_ticks_position('left');

for i in range(len(all_plots)):
    for j in range (len(all_plots[0])):
        all_plots[i,j].tick_params(labelsize=15, width=1, length=8, direction='out');
        all_plots[i,j].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...

# last but not least...and not last... response nonlinearity
all_plots[1,2].plot([-1, 1], [0, 0], 'k--')
all_plots[1,2].plot([0, 0], [-.1, 1], 'k--')
all_plots[1,2].plot(np.linspace(-1,1,100), np.power(np.maximum(0, np.linspace(-1,1,100)), modParamsCurr[3]), 'k-', linewidth=2)
all_plots[1,2].plot(np.linspace(-1,1,100), np.maximum(0, np.linspace(-1,1,100)), 'k--', linewidth=1)
all_plots[1,2].set_xlim([-1, 1]);
all_plots[1,2].set_ylim([-.1, 1]);
all_plots[1,2].text(0.5, 1.1, 'respExp: {:.2f}'.format(modParamsCurr[3]), fontsize=12, horizontalalignment='center', verticalalignment='center');
# Remove top/right axis, put ticks only on bottom/left
all_plots[1, 2].spines['right'].set_visible(False);
all_plots[1, 2].spines['top'].set_visible(False);
all_plots[1, 2].xaxis.set_ticks_position('bottom');
all_plots[1, 2].yaxis.set_ticks_position('left');
    
# print, in text, model parameters:
all_plots[0, 4].text(0.5, 0.5, 'prefSf: {:.3f}'.format(modParamsCurr[0]), fontsize=12, horizontalalignment='center', verticalalignment='center');
all_plots[0, 4].text(0.5, 0.4, 'derivative order: {:.3f}'.format(modParamsCurr[1]), fontsize=12, horizontalalignment='center', verticalalignment='center');
all_plots[0, 4].text(0.5, 0.3, 'response scalar: {:.3f}'.format(modParamsCurr[4]), fontsize=12, horizontalalignment='center', verticalalignment='center');
all_plots[0, 4].text(0.5, 0.2, 'sigma: {:.3f} | {:.3f}'.format(np.power(10, modParamsCurr[2]), modParamsCurr[2]), fontsize=12, horizontalalignment='center', verticalalignment='center');

### now save both figures (sfMix contrasts and details)

allFigs = [f, fDetails];
saveName = "/cell_%d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'sfMixOnly/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for fig in range(len(allFigs)):
    pdfSv.savefig(allFigs[fig])
    plt.close(allFigs[fig])
pdfSv.close()

# #### Plot contrast response functions

crfAx = []; fCRF = [];
fSum, crfSum = plt.subplots(nDisps, 2, figsize=(40, 40), sharex=False, sharey=False);
fCRF.append(fSum);
crfAx.append(crfSum);

crfFitsSepC50 = helper_fcns.fit_all_CRF(cellStruct, 1);
crfFitsOneC50 = helper_fcns.fit_all_CRF(cellStruct, 0);

crf_loss = lambda resp, pred: np.sum(np.square(np.sqrt(resp) - np.sqrt(pred)));
#crf_loss = lambda resp, pred: np.sum(np.power(resp-pred, 2)); # least-squares, for now...

for d in range(nDisps):
    
    # which sfs have at least one contrast presentation?
    v_sfs = np.where(np.sum(~np.isnan(respMean[d, :, :]), axis = 1) > 0);
    n_v_sfs = len(v_sfs[0])
    n_rows = 3; #int(np.floor(n_v_sfs/2));
    n_cols = 4; #n_v_sfs - n_rows
    fCurr, crfCurr = plt.subplots(n_rows, n_cols, figsize=(n_cols*20, n_rows*20), sharex = True, sharey = True);
    fCRF.append(fCurr)
    crfAx.append(crfCurr);
    
    c50_sep = np.zeros((n_v_sfs, 1));
    c50_all = np.zeros((n_v_sfs, 1));

    for sf in range(n_v_sfs):
	row_ind = sf/n_cols;
	col_ind = np.mod(sf, n_cols);
        sf_ind = v_sfs[0][sf];

        v_cons = ~np.isnan(respMean[d, sf_ind, :]);
        n_cons = sum(v_cons);
	plot_cons = np.linspace(np.min(all_cons[v_cons]), np.max(all_cons[v_cons]), 100); # 100 steps for plotting...

	# organize responses
	resps_curr = np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, ));

	# CRF fit
	curr_fit_sep = crfFitsSepC50[d][sf_ind]['params'];
	curr_fit_all = crfFitsOneC50[d][sf_ind]['params'];
	sep_loss = np.sum(np.log(crf_loss(resps_curr, helper_fcns.naka_rushton(all_cons[v_cons], curr_fit_sep))));
	all_loss = np.sum(np.log(crf_loss(resps_curr, helper_fcns.naka_rushton(all_cons[v_cons], curr_fit_all))));
	 
        c50_sep[sf] = curr_fit_sep[3];
        c50_all[sf] = curr_fit_all[3];

        # summary plots
	crfAx[0][d, 0].plot(all_cons[v_cons], np.maximum(resps_curr, 0.1), '-', clip_on=False);

        # 0.1 minimum to keep plot axis range OK...should find alternative
        expPts = crfAx[d+1][row_ind, col_ind].errorbar(all_cons[v_cons], np.maximum(resps_curr, 0.1), np.reshape([respStd[d, sf_ind, v_cons]], (n_cons, )), fmt='o', clip_on=False);

        sepPlt = crfAx[d+1][row_ind, col_ind].plot(plot_cons, helper_fcns.naka_rushton(plot_cons, curr_fit_sep), linestyle='dashed');
        allPlt = crfAx[d+1][row_ind, col_ind].plot(plot_cons, helper_fcns.naka_rushton(plot_cons, curr_fit_all), linestyle='dashed');
	# accompanying text...
	crfAx[d+1][row_ind, col_ind].text(0, 0.9, 'free [%.1f]: gain %.1f; c50 %.3f; exp: %.2f; baseline: %.1f' % (sep_loss, curr_fit_sep[1], curr_fit_sep[3], curr_fit_sep[2], curr_fit_sep[0]), 
		horizontalalignment='left', verticalalignment='center', transform=crfAx[d+1][row_ind, col_ind].transAxes, fontsize=30);
	crfAx[d+1][row_ind, col_ind].text(0, 0.8, 'fixed [%.1f]: gain %.1f; c50 %.3f; exp: %.2f; baseline: %.1f' % (all_loss, curr_fit_all[1], curr_fit_all[3], curr_fit_all[2], curr_fit_all[0]), 
		horizontalalignment='left', verticalalignment='center', transform=crfAx[d+1][row_ind, col_ind].transAxes, fontsize=30);

	# legend
	crfAx[d+1][row_ind, col_ind].legend((expPts[0], sepPlt[0], allPlt[0]), ('data', 'free c50', 'fixed c50'), fontsize='medium', loc='center left')

	plt_x = d+1; plt_y = (row_ind, col_ind);

	crfAx[plt_x][plt_y].set_xscale('log');
        crfAx[plt_x][plt_y].set_xlabel('contrast', fontsize='medium');
        crfAx[plt_x][plt_y].set_ylabel('resp (sps)', fontsize='medium');
	crfAx[plt_x][plt_y].set_title('D%d: sf: %.3f' % (d+1, all_sfs[sf_ind]), fontsize='large');

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        crfAx[plt_x][plt_y].tick_params(labelsize=25, width=2, length=16, direction='out');
        crfAx[plt_x][plt_y].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
	crfAx[plt_x][plt_y].spines['right'].set_visible(False);
	crfAx[plt_x][plt_y].spines['top'].set_visible(False);
	crfAx[plt_x][plt_y].xaxis.set_ticks_position('bottom');
	crfAx[plt_x][plt_y].yaxis.set_ticks_position('left');

    # make summary plots nice
    for i in range(2):
        crfAx[0][d, i].set_xscale('log');

        # Set ticks out, remove top/right axis, put ticks only on bottom/left
        crfAx[0][d, i].tick_params(labelsize=25, width=2, length=16, direction='out');
        crfAx[0][d, i].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
        crfAx[0][d, i].spines['right'].set_visible(False);
        crfAx[0][d, i].spines['top'].set_visible(False);
        crfAx[0][d, i].xaxis.set_ticks_position('bottom');
        crfAx[0][d, i].yaxis.set_ticks_position('left');
    
    # plot c50 as f/n of SF; plot sf tuning as reference...
    crfAx[0][d, 1].plot(all_sfs[v_sfs[0]], c50_sep);
    crfAx[0][d, 1].plot(all_sfs[v_sfs[0]], c50_all);
    maxC50 = np.maximum(np.max(c50_sep), np.max(c50_all));
    v_cons = np.array(val_con_by_disp[d]);
    sfRef = respMean[d, v_sfs[0], v_cons[-1]]; # plot highest contrast spatial frequency tuning curve
	# we normalize the sf tuning, flip upside down so it matches the profile of c50, which is lowest near peak SF preference
    crfAx[0][d, 1].plot(all_sfs[v_sfs[0]],  maxC50*(1-sfRef/np.max(sfRef)), linestyle='dashed');
    crfAx[0][d, 1].set_xlim([all_sfs[0], all_sfs[-1]]);

    crfAx[0][d, 0].set_title('D%d - all CRF' % (d), fontsize='large');
    crfAx[0][d, 0].set_xlabel('contrast', fontsize='large');
    crfAx[0][d, 0].set_ylabel('resp (sps)', fontsize='large');

    crfAx[0][d, 1].set_title('D%d - C50 (fixed vs free)' % (d), fontsize='large');
    crfAx[0][d, 1].set_xlabel('sf (cpd)', fontsize='large');
    crfAx[0][d, 1].set_ylabel('c50', fontsize='large');

	

saveName = "/cell_%d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fCRF:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()

# #### Plot contrast response functions - all sfs on one axis (per dispersion)

crfAx = []; fCRF = [];

maxResp = np.max(np.max(np.max(respMean[~np.isnan(respMean)])));

for d in range(nDisps):
    
    # which sfs have at least one contrast presentation?
    v_sfs = np.where(np.sum(~np.isnan(respMean[d, :, :]), axis = 1) > 0);
    n_v_sfs = len(v_sfs[0])
    fCurr, crfCurr = plt.subplots(1, 1, figsize=(15, 20), sharex = True, sharey = False);
    fCRF.append(fCurr)
    crfAx.append(crfCurr);
    
    lines = []; lines_log = [];
    for sf in range(n_v_sfs):
        sf_ind = v_sfs[0][sf];
        v_cons = ~np.isnan(respMean[d, sf_ind, :]);
        n_cons = sum(v_cons);
	
	col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];
 	curr_resps = np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, ));
        #line_curr, = crfAx[d][0].plot(all_cons[v_cons], curr_resps, '-o', color=col, clip_on=False);
	#lines.append(line_curr);
	respAbBaseline = curr_resps-blankMean;
        line_curr, = crfAx[d].plot(all_cons[v_cons][respAbBaseline>1e-1], respAbBaseline[respAbBaseline>1e-1], '-o', color=col, clip_on=False);
        #line_curr, = crfAx[d].plot(all_cons[v_cons], np.maximum(1e-1, curr_resps-blankMean), '-o', color=col, clip_on=False);
	lines_log.append(line_curr);

    crfAx[d].set_xlim([1e-2, 1]);
    crfAx[d].set_ylim([1e-2, 1.5*maxResp]);
    crfAx[d].set_aspect('equal', 'box')
    crfAx[d].set_xscale('log');
    crfAx[d].set_yscale('log');
    crfAx[d].set_xlabel('contrast');

    # Set ticks out, remove top/right axis, put ticks only on bottom/left
    crfAx[d].tick_params(labelsize=15, width=1, length=8, direction='out');
    crfAx[d].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
    crfAx[d].spines['right'].set_visible(False);
    crfAx[d].spines['top'].set_visible(False);
    crfAx[d].xaxis.set_ticks_position('bottom');
    crfAx[d].yaxis.set_ticks_position('left');

    crfAx[d].set_ylabel('resp above baseline (sps)');
    crfAx[d].set_title('D%d: sf:all - log resp' % (d));
    crfAx[d].legend(lines_log, [str(i) for i in np.round(all_sfs[v_sfs], 2)], loc='upper left');

saveName = "/allSfs_log_cell_%d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fCRF:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()
