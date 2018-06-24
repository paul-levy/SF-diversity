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
baseline_sub = int(sys.argv[2]); # subtract the baseline when plotting one curve per axis?
sf_fit_type = int(sys.argv[3]);

# personal mac
dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/sach-data/';
save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/figures/';

conDig = 3; # round contrast to the 3rd digit

allData = np.load(dataPath + 'sachData.npy', encoding='latin1').item();
cellStruct = allData[which_cell-1];

# #### Load descriptive model fits, [RVC Naka-Rushton fits], [comp. model fits]

fLname = 'descrFits';
if baseline_sub:
  fLname = str(fLname + '_baseSub');
if sf_fit_type == 1:
  type_str = '_poiss';
elif sf_fit_type == 2:
  type_str = '_sqrt';
elif sf_fit_type == 3:
  type_str = '_sach';
fLname = str(dataPath + fLname + type_str + '.npy');

descrFits = helper_fcns.np_smart_load(fLname);
descrFits = descrFits[which_cell-1]['params']; # just get this cell

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

data = cellStruct['data'];

resps, stimVals, _ = helper_fcns.tabulateResponses(data);
blankMean, blankStd = helper_fcns.blankResp(data); 
# all responses on log ordinate (y axis) should be baseline subtracted

all_cons = stimVals[0];
all_sfs = stimVals[1];

nCons = len(all_cons);
nSfs = len(all_sfs);

# #### Unpack responses

f0 = resps[0]; # mean rate (power at DC)
f1 = resps[1]; # power at fundamental freq. of stimulus

# ### Plots

#########
# #### All SF tuning on one graph
#########

plotThresh = 1e-1;
maxResp = np.max(np.max(f0['mean']));

f, ax = plt.subplots(1, 1, figsize=(20, 10));
 
lines = [];
for c in reversed(range(nCons)):

    # summary plot (just data) first
    val_sfs = np.where(all_sfs>0); # there is one SF which is zero; ignore this one
    # plot data
    col = [c/float(nCons), c/float(nCons), c/float(nCons)];
    respAbBaseline = f0['mean'][c, val_sfs] - blankMean;
    threshResps = respAbBaseline>plotThresh;
    curr_line, = ax.plot(all_sfs[val_sfs][threshResps[0]], respAbBaseline[threshResps], '-o', clip_on=False, color=col);
    lines.append(curr_line);

    # specific contrast - data and 

ax.set_aspect('equal', 'box'); 
ax.set_xlim((0.5*np.min(all_sfs[val_sfs]), 1.2*np.max(all_sfs[val_sfs])));
ax.set_ylim((5e-2, 1.5*maxResp));

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
   
    curr_resps = f0['mean'][c, val_sfs][0]; # additional layer of array to unwrap
    curr_sem = f0['sem'][c, val_sfs][0];
    if baseline_sub:
      curr_resps = curr_resps - blankMean;
    v_sfs = ~np.isnan(curr_resps);

    # plot data
    sfsAx[c_plt_ind].errorbar(all_sfs[val_sfs][v_sfs], curr_resps[v_sfs], 
                                  curr_sem[v_sfs], fmt='o', clip_on=False);

    # plot descriptive model fit
    curr_mod_params = descrFits[c]; 
    sfsAx[c_plt_ind].plot(sfs_plot, helper_fcns.DiffOfGauss(*curr_mod_params, stim_sf=sfs_plot)[0], clip_on=False)
    # note we take DiffOfGaus(.)[0] since that is the unnormalized version of the DoG response

    sfsAx[c_plt_ind].set_xlim((min(sfs_plot), max(sfs_plot)));

    sfsAx[c_plt_ind].set_xscale('log');
    sfsAx[c_plt_ind].set_xlabel('sf (c/deg)'); 
    sfsAx[c_plt_ind].set_title('SF tuning at contrast: %.3f%%' % (all_cons[c]));

    # Set ticks out, remove top/right axis, put ticks only on bottom/left
    sfsAx[c_plt_ind].tick_params(labelsize=15, width=1, length=8, direction='out');
    sfsAx[c_plt_ind].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...	
    sns.despine(ax=sfsAx[c_plt_ind], offset=10, trim=False); 

    sfsAx[c_plt_ind].set_ylim((0, 1.5*maxResp));
    yAxStr = 'resp ';
    if baseline_sub:
      yAxStr = str(yAxStr + 'above baseline ');
    sfsAx[c_plt_ind].set_ylabel(str(yAxStr + '(sps)'));

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

maxResp = np.max(np.max(f0['mean']));

f, ax = plt.subplots(1, 1, figsize=(20, 10));

lines = [];
for sf in reversed(range(n_v_sfs)):

    # plot data
    col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];
    respAbBaseline = f0['mean'][:, sf] - blankMean;
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
