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

# personal mac
dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/sach-data/';
save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/figures/';

conDig = 3; # round contrast to the 3rd digit

allData = np.load(dataPath + 'sachData.npy', encoding='latin1').item();
cellStruct = allData[which_cell-1];

# #### Load descriptive model fits, comp. model fits

'''
descrFits = np.load(str(dataPath + 'descrFits.npy'), encoding = 'latin1').item();
descrFits = descrFits[which_cell-1]['params']; # just get this cell

modParams = np.load(str(dataPath + fitListName), encoding= 'latin1').item();
modParamsCurr = modParams[which_cell-1]['params'];
'''

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

data = cellStruct['data'];

resps, stimVals = helper_fcns.tabulateResponses(data);
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

n_v_cons = len(all_cons);

maxResp = np.max(np.max(f0['mean']));

f, ax = plt.subplots(1, 1, figsize=(20, 10));
 
lines = [];
for c in reversed(range(n_v_cons)):

    val_sfs = np.where(all_sfs>0); # there is one SF which is zero; ignore this one
    # plot data
    col = [c/float(n_v_cons), c/float(n_v_cons), c/float(n_v_cons)];
    respAbBaseline = f0['mean'][c, val_sfs] - blankMean;
    threshResps = respAbBaseline>1e-1;
    #curr_line = ax.axhline(maxResp, clip_on=False, color=col);
    curr_line, = ax.plot(all_sfs[val_sfs][threshResps[0]], respAbBaseline[threshResps], '-o', clip_on=False, color=col);
    lines.append(curr_line);

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
