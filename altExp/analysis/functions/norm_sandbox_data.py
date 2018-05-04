# coding: utf-8

######################## To do:

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns
sns.set(style='ticks')
import helper_fcns
from scipy.stats import poisson, nbinom
from scipy.stats.mstats import gmean

import pdb

import sys # so that we can import model_responses (in different folder)
import model_responses

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/Analysis/Functions/paul_plt_cluster.mplstyle');
plt.rc('legend',fontsize='medium') # using a named size

which_cell = int(sys.argv[1]);
fit_type = int(sys.argv[2]);
normTypeArr = [];
nArgsIn = len(sys.argv) - 3; # we've already taken 3 arguments off (function all, which_cell, fit_type)
argInd = 3;
while nArgsIn > 0:
  normTypeArr.append(float(sys.argv[argInd]));
  nArgsIn = nArgsIn - 1;
  argInd = argInd + 1;

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

fitListName = 'fitList_180426_slowLR.npy';

rpt_fit = 1; # i.e. take the multi-start result
if rpt_fit:
  is_rpt = '_rpt';
else:
  is_rpt = '';

conDig = 3; # round contrast to the 3rd digit

dataList = np.load(str(dataPath + 'dataList.npy'), encoding='latin1').item();

cellStruct = np.load(str(dataPath + dataList['unitName'][which_cell-1] + '_sfm.npy'), encoding='latin1').item();

# #### Load model fits

modParams = np.load(str(dataPath + fitListName), encoding= 'latin1').item();
modParamsCurr = modParams[which_cell-1]['params'];

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

data = cellStruct['sfm']['exp']['trial'];

ignore, modRespAll, normTypeArr = model_responses.SFMGiveBof(modParamsCurr, cellStruct, normTypeArr);
norm_type = normTypeArr[0];
if norm_type == 1:
  gs_mean = normTypeArr[1]; # guaranteed to exist after call to .SFMGiveBof, if norm_type == 1
  gs_std = normTypeArr[2]; # guaranteed to exist ...
#modRespAll = model_responses.SFMGiveBof(modParamsCurr, cellStruct, normTypeArr)[1]; # NOTE: We're taking [1] (i.e. second) output of SFMGiveBof
resp, stimVals, val_con_by_disp, validByStimVal, modResp = helper_fcns.tabulate_responses(cellStruct, modRespAll);
blankMean, blankStd, _ = helper_fcns.blankResp(cellStruct); 
modBlankMean = modParamsCurr[6]; # late additive noise is the baseline of the model
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

# modResp is (nDisps, nSf, nCons, nReps) nReps is (currently; 2018.01.05) set to 20 to accommadate the current experiment with 10 repetitions
modLow = np.nanmin(modResp, axis=3);
modHigh = np.nanmax(modResp, axis=3);
modAvg = np.nanmean(modResp, axis=3);
modSponRate = modParamsCurr[6];

# ### Plots	        
#########
# Plot secondary things - filter, normalization, nonlinearity, etc
#########

# plot model details - filter
imSizeDeg = cellStruct['sfm']['exp']['size'];
pixSize   = 0.0028; # fixed from Robbe
prefSf    = modParamsCurr[0];
dOrder    = modParamsCurr[1]
prefOri = 0; # just fixed value since no model param for this
aRatio = 1; # just fixed value since no model param for this
filtTemp  = model_responses.oriFilt(imSizeDeg, pixSize, prefSf, prefOri, dOrder, aRatio);
filt      = (filtTemp - filtTemp[0,0])/ np.amax(np.abs(filtTemp - filtTemp[0,0]));

plt.subplot2grid(detailSize, (2, 0)); # set the current subplot location/size[default is 1x1]
plt.imshow(filt, cmap='gray');
plt.axis('off');
#plt.title('Filter in space', fontsize=20)

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
    # 0* if we ignore asymmetry; inhAsym* otherwise
    inhWeight = np.append(inhWeight, 1 + inhAsym * (np.log(cellStruct['sfm']['mod']['normalization']['pref']['sf'][iP]) - np.mean(np.log(cellStruct['sfm']['mod']['normalization']['pref']['sf'][iP]))));
           
sfInh = 0 * np.ones(omega.shape) / np.amax(modHigh); # mult by 0 because we aren't including a subtractive inhibition in model for now 7/19/17
sfNorm = np.sum(-.5*(inhWeight*np.square(inhSfTuning)), 1);
sfNorm = sfNorm/np.amax(np.abs(sfNorm));

#### HERE
fSims = []; simsAx = [];

# first, just plot the (normalized) excitatory filter and normalization pool response on the same plot
# and for ease of comparison, also duplicate the SF and RVC tuning for single gratings here
# calculations done above in fDetails (sfExc, sfNorm)
fFilt, axCurr = plt.subplots(1, 1, figsize=(10, 10));
fSims.append(fFilt);
simsAx.append(axCurr);

# plot model details - filter
simsAx[0][0].semilogx([omega[0], omega[-1]], [0, 0], 'k--')
simsAx[0][0].semilogx([.01, .01], [-1.5, 1], 'k--')
simsAx[0][0].semilogx([.1, .1], [-1.5, 1], 'k--')
simsAx[0][0].semilogx([1, 1], [-1.5, 1], 'k--')
simsAx[0][0].semilogx([10, 10], [-1.5, 1], 'k--')
simsAx[0][0].semilogx([100, 100], [-1.5, 1], 'k--')
# now the real stuff
ex = simsAx[0][0].semilogx(omega, sfExc, 'k-')
#simsAx[0][0].semilogx(omega, sfInh, 'r--', linewidth=2);
nm = simsAx[0][0].semilogx(omega, -sfNorm, 'r-', linewidth=1);
simsAx[0][0].set_xlim([omega[0], omega[-1]]);
simsAx[0][0].set_ylim([-1.5, 1]);
simsAx[0][0].set_xlabel('SF (cpd)', fontsize=12);
simsAx[0][0].set_ylabel('Normalized response (a.u.)', fontsize=12);
simsAx[0][0].set_title('CELL %d' % (cellNum), fontsize=20);
simsAx[0][0].legend([ex[0], nm[0]], ('excitatory %.2f' % (modParamsCurr[0]), 'normalization %.2f' % (np.exp(modParamsCurr[-2]))));
# Remove top/right axis, put ticks only on bottom/left
sns.despine(ax=simsAx[0][0], offset=5);

#### Now simulate ####
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

for d in range(nDisps):
    
    v_cons = val_con_by_disp[d];
    n_v_cons = len(v_cons);
    
    fCurr, dispCurr = plt.subplots(1, 2, figsize=(40, 40)); # left side for SF simulations, right side for RVC simulations
    fSims.append(fCurr)
    simsAx.append(dispCurr);

    # SF tuning - NEED TO SIMULATE
    lines = [];
    for c in reversed(range(n_v_cons)):
        curr_resps = [];
        for sf_i in v_sfs:
          print('Testing SF tuning: disp %d, con %.2f, sf %.2f' % (d+1, v_cons[c], sf_i));
          sf_iResp, ignore, ignore, ignore = mod_resp.SFMsimulate(modParamsCurr, cellStruct, d+1, v_cons[c], sf_i);
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
          con_iResp, ignore, ignore, ignore = mod_resp.SFMsimulate(modParamsCurr, cellStruct, d+1, con_i, sf_curr);
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

# fix subplots to not overlap
fDetails.tight_layout();
# fSims must be saved separately...
saveName = "cell_%d_simulate.pdf" % (cellNum)
pdfSv = pltSave.PdfPages(str(save_loc + saveName));
for ff in fSims:
    pdfSv.savefig(ff)
    plt.close(ff)
pdfSv.close();

'''
### END OF HERE

# #### Plot contrast response functions with (full) model predictions; i.e. not Naka-Rushton 

rvcAx = []; fRVC = [];

for d in range(nDisps):
    
    # which sfs have at least one contrast presentation?
    v_sfs = np.where(np.sum(~np.isnan(respMean[d, :, :]), axis = 1) > 0);
    n_v_sfs = len(v_sfs[0])
    n_rows = 3; #int(np.floor(n_v_sfs/2));
    n_cols = 4; #n_v_sfs - n_rows
    fCurr, rvcCurr = plt.subplots(n_rows, n_cols, figsize=(n_cols*20, n_rows*20), sharex = True, sharey = True);
    fRVC.append(fCurr)
    rvcAx.append(rvcCurr);
    
    #rvc_plots = [];

    for sf in range(n_v_sfs):
	row_ind = sf/n_cols;
	col_ind = np.mod(sf, n_cols);
        sf_ind = v_sfs[0][sf];
       	plt_x = d; plt_y = (row_ind, col_ind);

        v_cons = ~np.isnan(respMean[d, sf_ind, :]);
        n_cons = sum(v_cons);
	plot_cons = np.linspace(0, np.max(all_cons[v_cons]), 100); # 100 steps for plotting...
	#plot_cons = np.linspace(np.min(all_cons[v_cons]), np.max(all_cons[v_cons]), 100); # 100 steps for plotting...

	# organize responses
        resps_curr = np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, ));
        resps_w_blank = np.hstack((blankMean, resps_curr));

        # plot data
        dataPlt = rvcAx[plt_x][plt_y].plot(all_cons[v_cons], np.maximum(np.reshape([respMean[d, sf_ind, v_cons]], (n_cons, )), 0.1), '-', clip_on=False);
	# RVC with full model fit
        rvcAx[plt_x][plt_y].fill_between(all_cons[v_cons], modLow[d, sf_ind, v_cons], \
                                      modHigh[d, sf_ind, v_cons], color='r', alpha=0.2);
        #pdb.set_trace();
        modPlt = rvcAx[plt_x][plt_y].plot(all_cons[v_cons], np.maximum(modAvg[d, sf_ind, v_cons], 0.1), 'r-', alpha=0.7, clip_on=False);

        # summary plots
        '''
        '''
	curr_rvc = rvcAx[0][d, 0].plot(all_cons[v_cons], resps_curr, '-', clip_on=False);
        rvc_plots.append(curr_rvc[0]);

        stdPts = np.hstack((0, np.reshape([respStd[d, sf_ind, v_cons]], (n_cons, ))));
        expPts = rvcAx[d+1][row_ind, col_ind].errorbar(np.hstack((0, all_cons[v_cons])), resps_w_blank, stdPts, fmt='o', clip_on=False);

        sepPlt = rvcAx[d+1][row_ind, col_ind].plot(plot_cons, helper_fcns.naka_rushton(plot_cons, curr_fit_sep), linestyle='dashed');
        allPlt = rvcAx[d+1][row_ind, col_ind].plot(plot_cons, helper_fcns.naka_rushton(plot_cons, curr_fit_all), linestyle='dashed');
	# accompanying legend/comments
	rvcAx[d+1][row_ind, col_ind].legend((expPts[0], sepPlt[0], allPlt[0]), ('data', 'model fits'), fontsize='large', loc='center left')
        '''

        '''
	rvcAx[plt_x][plt_y].set_xscale('symlog', linthreshx=0.01); # symlog will allow us to go down to 0 
        rvcAx[plt_x][plt_y].set_xlabel('contrast', fontsize='medium');
        rvcAx[plt_x][plt_y].set_ylabel('resp (sps)', fontsize='medium');
	rvcAx[plt_x][plt_y].set_title('D%d: sf: %.3f' % (d+1, all_sfs[sf_ind]), fontsize='large');
	rvcAx[plt_x][plt_y].legend((dataPlt[0], modPlt[0]), ('data', 'model avg'), fontsize='large', loc='center left');

	# Set ticks out, remove top/right axis, put ticks only on bottom/left
        sns.despine(ax = rvcAx[plt_x][plt_y], offset = 10, trim=False);
        rvcAx[plt_x][plt_y].tick_params(labelsize=25, width=2, length=16, direction='out');
        rvcAx[plt_x][plt_y].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...

saveName = "/cell_%d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fRVC:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()

# #### Plot contrast response functions - all sfs on one axis (per dispersion)

crfAx = []; fCRF = [];

for d in range(nDisps):
    
    fCurr, crfCurr = plt.subplots(1, 2, figsize=(15, 40), sharex = False, sharey = False); # left side for data, right side for model predictions
    fCRF.append(fCurr)
    crfAx.append(crfCurr);

    for i in range(2):
      
      if i == 0:
        curr_resps = respMean;
        curr_base = blankMean;
        title_str = 'data';
      else:
        curr_resps = modAvg;
        curr_base = modBlankMean;
        title_str = 'model';
      maxResp = np.max(np.max(np.max(curr_resps[~np.isnan(curr_resps)])));

      # which sfs have at least one contrast presentation?
      v_sfs = np.where(np.sum(~np.isnan(curr_resps[d, :, :]), axis = 1) > 0);
      n_v_sfs = len(v_sfs[0])

      lines = []; lines_log = [];
      for sf in range(n_v_sfs):
          sf_ind = v_sfs[0][sf];
          v_cons = ~np.isnan(curr_resps[d, sf_ind, :]);
          n_cons = sum(v_cons);

          col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];
          plot_resps = np.reshape([curr_resps[d, sf_ind, v_cons]], (n_cons, ));
          respAbBaseline = plot_resps-curr_base;
          line_curr, = crfAx[d][i].plot(all_cons[v_cons][respAbBaseline>1e-1], respAbBaseline[respAbBaseline>1e-1], '-o', color=col, clip_on=False);
          #line_curr, = crfAx[d][i].plot(all_cons[v_cons], np.maximum(1e-1, curr_resps-blankMean), '-o', color=col, clip_on=False);
          lines_log.append(line_curr);

      crfAx[d][i].set_xlim([1e-2, 1]);
      crfAx[d][i].set_ylim([1e-2, 1.5*maxResp]);
      crfAx[d][i].set_aspect('equal', 'box')
      crfAx[d][i].set_xscale('log');
      crfAx[d][i].set_yscale('log');
      crfAx[d][i].set_xlabel('contrast');

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      crfAx[d][i].tick_params(labelsize=15, width=1, length=8, direction='out');
      crfAx[d][i].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...
      sns.despine(ax = crfAx[d][i], offset=10, trim=False);

      crfAx[d][i].set_ylabel('resp above baseline (sps)');
      crfAx[d][i].set_title('D%d: sf:all - log resp %s' % (d, title_str));
      crfAx[d][i].legend(lines_log, [str(i) for i in np.round(all_sfs[v_sfs], 2)], loc='upper left');

saveName = "/allSfs_log_cell_%d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'CRF/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
for f in fCRF:
    pdfSv.savefig(f)
    plt.close(f)
pdfSv.close()
'''
