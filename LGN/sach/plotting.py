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

import pdb

import sys

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/Analysis/Functions/paul_plt_cluster.mplstyle');
from matplotlib import rcParams
rcParams['font.size'] = 20;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 3;
rcParams['axes.linewidth'] = 3;
rcParams['lines.markersize'] = 3
rcParams['font.style'] = 'oblique';

which_cell = int(sys.argv[1]);
sf_loss_type = int(sys.argv[2]);
sf_DoG_model = int(sys.argv[3]);

# personal mac
#dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/sach-data/';
#save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/figures/';
# prince cluster
dataPath = '/home/pl1465/SF_diversity/LGN/sach/structures/';
save_loc = '/home/pl1465/SF_diversity/LGN/sach/figures/';

conDig = 3; # round contrast to the 3rd digit

allData = np.load(dataPath + 'sachData.npy', encoding='latin1').item();
cellStruct = allData[which_cell-1];

# #### Load descriptive model fits, [RVC Naka-Rushton fits], [comp. model fits]

fLname = 'descrFits_181006';
if sf_loss_type == 1:
  loss_str = '_poiss';
elif sf_loss_type == 2:
  loss_str = '_sqrt';
elif sf_loss_type == 3:
  loss_str = '_sach';
if sf_DoG_model == 1:
  mod_str = '_sach';
elif sf_DoG_model == 2:
  mod_str = '_tony';
fLname = str(dataPath + fLname + loss_str + mod_str + '.npy');

descrFits = helper_fcns.np_smart_load(fLname);
descrFits = descrFits[which_cell-1]; # just get this cell

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

data = cellStruct['data'];

resps, stimVals, _ = helper_fcns.tabulateResponses(data);
# all responses on log ordinate (y axis) should be baseline subtracted

all_cons = stimVals[0];
all_sfs = stimVals[1];

nCons = len(all_cons);
nSfs = len(all_sfs);

# Unpack responses
f1 = resps[1]; # power at fundamental freq. of stimulus

# ### Plots

#########
# #### All SF tuning on one graph
#########

plotThresh = 1e-1;
maxResp = np.max(np.max(f1['mean']));

f, ax = plt.subplots(1, 1, figsize=(20, 10));
 
lines = [];
for c in reversed(range(nCons)):

    # summary plot (just data) first
    val_sfs = np.where(all_sfs>0); # there is one SF which is zero; ignore this one
    # plot data
    col = [c/float(nCons), c/float(nCons), c/float(nCons)];
    respAbBaseline = f1['mean'][c, val_sfs];
    threshResps = respAbBaseline>plotThresh;
    curr_line, = ax.plot(all_sfs[val_sfs][threshResps[0]], respAbBaseline[threshResps], '-o', clip_on=False, color=col);
    lines.append(curr_line);

ax.set_aspect('equal', 'box'); 
ax.set_xlim((0.5*np.min(all_sfs[val_sfs]), 1.2*np.max(all_sfs[val_sfs])));
#ax.set_ylim((5e-2, 1.5*maxResp));

#ax.set_xscale('symlog', linthreshx=all_sfs[1]); # all_sfs[0] = 0, but all_sfs[0]>1
ax.set_xscale('log');
ax.set_yscale('log');
ax.set_xlabel('sf (c/deg)'); 

# Set ticks out, remove top/right axis, put ticks only on bottom/left
ax.tick_params(labelsize=15, width=2, length=16, direction='out');
ax.tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
sns.despine(ax=ax, offset=10, trim=False); 

ax.set_ylabel('resp above baseline (sps)');
ax.set_title('SF tuning - %s #%d' % (cellStruct['cellType'], which_cell));
ax.legend(lines, [str(np.round(i, conDig)) for i in reversed(all_cons)], loc=0);

saveName = "/allCons_cell_%d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'sfTuning/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(f)
plt.close(f)
pdfSv.close()

#########
# #### SF tuning - split by contrast, data and model
#########

fSfs, sfsAx = plt.subplots(nCons, 1, figsize=(20, 10*nCons), sharey=False);

val_sfs = np.where(all_sfs>0); # do not plot the zero sf condition
sfs_plot = np.logspace(np.log10(np.min(all_sfs[val_sfs])), np.log10(np.max(all_sfs[val_sfs])), 51);

for c in reversed(range(nCons)):
    c_plt_ind = nCons - c - 1;
   
    curr_resps = f1['mean'][c, val_sfs][0]; # additional layer of array to unwrap
    curr_sem = f1['sem'][c, val_sfs][0];
    v_sfs = ~np.isnan(curr_resps);

    # plot data
    sfsAx[c_plt_ind].errorbar(all_sfs[val_sfs][v_sfs], curr_resps[v_sfs], 
                                  curr_sem[v_sfs], fmt='o', clip_on=False);

    # plot descriptive model fit and inferred characteristic frequency
    curr_mod_params = descrFits['params'][c]; 
    if sf_DoG_model == 1:
      sfsAx[c_plt_ind].plot(sfs_plot, helper_fcns.DoGsach(*curr_mod_params, stim_sf=sfs_plot)[0], clip_on=False)
    elif sf_DoG_model == 2:
      sfsAx[c_plt_ind].plot(sfs_plot, helper_fcns.DiffOfGauss(*curr_mod_params, stim_sf=sfs_plot)[0], clip_on=False)
    f_c = helper_fcns.dog_charFreq(curr_mod_params, sf_DoG_model);
    # note we take DiffOfGaus(.)[0] since that is the unnormalized version of the DoG response

    sfsAx[c_plt_ind].set_xlim((min(sfs_plot), max(sfs_plot)));

    sfsAx[c_plt_ind].set_xscale('log');
    sfsAx[c_plt_ind].set_xlabel('spatial frequency (c/deg)'); 
    sfsAx[c_plt_ind].set_title('SF tuning: contrast: %.3f%%, %.1f%% varExpl' % (all_cons[c], descrFits['varExpl'][c]));

    # Set ticks out, remove top/right axis, put ticks only on bottom/left
    sfsAx[c_plt_ind].tick_params(labelsize=15, width=1, length=8, direction='out');
    sfsAx[c_plt_ind].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...	
    sns.despine(ax=sfsAx[c_plt_ind], offset=10, trim=False); 

    #sfsAx[c_plt_ind].set_ylim((0, 1.5*maxResp));
    yAxStr = 'response ';
    sfsAx[c_plt_ind].set_ylabel(str(yAxStr + '(spikes/s)'));

saveName = "/cell_%d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'sfTuning/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(fSfs)
plt.close(fSfs)
pdfSv.close()


#########
# #### Plot response versus contrast curves (RVC)
#########

v_sfs = all_sfs;
n_v_sfs = len(all_sfs);

maxResp = np.max(np.max(f1['mean']));

f, ax = plt.subplots(1, 1, figsize=(20, 10));

lines = [];
for sf in reversed(range(n_v_sfs)):

    # plot data
    col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];
    respAbBaseline = f1['mean'][:, sf];
    curr_line, = ax.plot(all_cons[respAbBaseline>1e-1], respAbBaseline[respAbBaseline>1e-1], '-o', clip_on=False, color=col);
    lines.append(curr_line);

ax.set_aspect('equal', 'box'); 
ax.set_xlim((0.5*min(all_cons), 1.2*max(all_cons)));
ax.set_ylim((5e-2, 1.5*maxResp));

ax.set_xscale('log');
ax.set_yscale('log');
ax.set_xlabel('con (%)'); 

# Set ticks out, remove top/right axis, put ticks only on bottom/left
ax.tick_params(labelsize=15, width=2, length=16, direction='out');
ax.tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
sns.despine(ax=ax, offset=10, trim=False); 

ax.set_ylabel('resp above baseline (sps)');
ax.set_title('RVC - %s #%d' % (cellStruct['cellType'], which_cell));
ax.legend(lines, [str(np.round(i, conDig)) for i in reversed(all_sfs)], loc=0);

saveName = "/cell_NR_%d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'RVC/'));
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(f)
plt.close(f)
pdfSv.close()
