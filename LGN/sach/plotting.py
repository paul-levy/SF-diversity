# coding: utf-8

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns
sns.set(style='ticks')
import helper_fcns_sach as hf
from scipy.stats import poisson, nbinom

import pdb

import sys

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_style.mplstyle');
from matplotlib import rcParams
rcParams['font.size'] = 20;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 2.5;
rcParams['axes.linewidth'] = 1.5;
rcParams['lines.markersize'] = 5;
rcParams['font.style'] = 'oblique';

which_cell   = int(sys.argv[1]);
sf_loss_type = int(sys.argv[2]);
sf_DoG_model = int(sys.argv[3]);
rvcMod       = int(sys.argv[4]);
joint        = int(sys.argv[5]);
isHPC        = int(sys.argv[6]);
phAdj        = int(sys.argv[7]);
fromFile     = int(sys.argv[8]); 

loc_base = os.getcwd() + '/';

dataPath = loc_base + 'structures/';
save_loc = loc_base + 'figures/';

conDig = 2; # round contrast to the 3rd digit

allData = hf.np_smart_load(dataPath + 'sachData.npy');
#allData = np.load(dataPath + 'sachData.npy', encoding='latin1').item();
cellStruct = allData[which_cell-1];

# #### Load descriptive model fits, RVC Naka-Rushton fits, [comp. model fits]
zSub = 0; # are we loading fits that were fit to responses adjusted s.t. the lowest value is 1?
#######
## NOTE: SF tuning curves with with zSub; RVCs are not, so we must subtract respAdj from the RVC curve to align with what we fit (i.e. the zSub'd data)
#######
HPC = 'HPC' if isHPC else '';
fLname = 'descrFits%s_s220518' % HPC;
#fLname = 'descrFits%s_s220412' % HPC;
#fLname = 'descrFits%s_s220410a' % HPC;
#fLname = 'descrFits%s_s220227' % HPC;
mod_str = hf.descrMod_name(sf_DoG_model);
fLname_full = hf.descrFit_name(sf_loss_type, fLname, mod_str, joint=joint, phAdj=phAdj);
descrFits = hf.np_smart_load(dataPath + fLname_full);
descrFits = descrFits[which_cell-1]; # just get this cell

rvcSuff = hf.rvc_mod_suff(rvcMod);
rvcBase = 'rvcFits%s_220518' % HPC;
#rvcBase = 'rvcFits%s_220412' % HPC;
#rvcBase = 'rvcFits%s_220219' % HPC;
#rvcBase = 'rvcFits_211006';
#rvcBase = 'rvcFits_210721';
vecF1 = 1 if phAdj==0 else 0;
if phAdj<1: # either vec corr or neither
  dir=None;
else:
  dir=1; # default...
rvcFits = hf.np_smart_load(dataPath + hf.rvc_fit_name(rvcBase, rvcMod, vecF1=vecF1, dir=dir));
rvcFits = rvcFits[which_cell-1];

### now, get the FULL save_loc (should have fit name)
subDir = fLname_full.replace('Fits', '').replace('.npy', '');
save_loc = str(save_loc + subDir + '/');
if not os.path.exists(save_loc):
  os.makedirs(save_loc);

# ### Organize data
# #### determine contrasts, center spatial frequency, dispersions

data = cellStruct['data'];

resps, stimVals, _ = hf.tabulateResponses(data, phAdjusted=phAdj);
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
maxResp = np.max(np.max(f1['mean']));
minResp = np.nanmin(f1['mean'].flatten());
if zSub == 1:
  respAdj = minResp-1;
else:
  respAdj = np.array([0]);

f, ax = plt.subplots(1, 2, figsize=(35, 20), sharey=True);

minResp_toPlot = 5e-1; 

lines = [];
for c in reversed(range(nCons)):

    # summary plot (just data) first
    val_sfs = np.where(all_sfs>0)[0]; # there is one SF which is zero; ignore this one
    # plot data
    col = [(nCons-c-1)/float(nCons), (nCons-c-1)/float(nCons), (nCons-c-1)/float(nCons)];
    respAbBaseline = np.reshape(f1['mean'][c, val_sfs], (len(val_sfs), ));
    respVar = np.reshape(f1['sem'][c, val_sfs], (len(val_sfs), )); 
    abvBound = respAbBaseline>minResp_toPlot;
    # DO NOT actually plot error bar on these...
    ax[0].plot(all_sfs[val_sfs][abvBound], respAbBaseline[abvBound] - respAdj, '-o', clip_on=False, color=col, label='%d%%' % (int(100*np.round(all_cons[c], conDig))))
 
    # then descriptive fit
    sfs_plot = np.geomspace(all_sfs[val_sfs[0]], all_sfs[val_sfs[-1]], 100);
    prms_curr = descrFits['params'][c];
    descrResp = hf.get_descrResp(prms_curr, sfs_plot, sf_DoG_model);
    abvZero = descrResp>minResp_toPlot;
    ax[1].plot(sfs_plot[abvZero], descrResp[abvZero], '-', clip_on=False, color=col); #, label=str(np.round(all_cons[c], conDig)))

for i in range(2):
  #ax[i].set_aspect('equal', 'box'); 
  ax[i].set_xlim((0.5*np.min(all_sfs[val_sfs]), 1.2*np.max(all_sfs[val_sfs])));
  ax[i].set_ylim((minResp_toPlot, 300)); # common y axis for ALL plots
  #ax[i].set_ylim((5e-2, 1.5*maxResp));

  #ax.set_xscale('symlog', linthreshx=all_sfs[1]); # all_sfs[0] = 0, but all_sfs[0]>1
  ax[i].set_xscale('log');
  ax[i].set_yscale('log');
  ax[i].set_aspect('equal'); # if both axes are log, must make equal scales!
  ax[i].set_xlabel('sf (c/deg)'); 

  # Set ticks out, remove top/right axis, put ticks only on bottom/left
  #ax[i].tick_params(labelsize=15, width=2, length=16, direction='out');
  #ax[i].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...

  if i == 0:
    ax[i].set_ylabel('resp above baseline (sps)');
    ax[i].set_title('SF tuning - %s #%d' % (cellStruct['cellType'], which_cell));
    ax[i].legend();

sns.despine(offset=10, trim=False); 

saveName = "/allCons_cell_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'sfTuning/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(f)
plt.close(f)
pdfSv.close()

#########
# #### SF tuning - split by contrast, data and model
#########

### TODO: make log, too...
fSfs, sfsAx = plt.subplots(nCons, 2, figsize=(2*10, 8*nCons), sharey=False);

val_sfs = np.where(all_sfs>0); # do not plot the zero sf condition
sfs_plot = np.logspace(np.log10(np.min(all_sfs[val_sfs])), np.log10(np.max(all_sfs[val_sfs])), 51);

minResp_toPlot = 5e-1;

for c in reversed(range(nCons)):
    c_plt_ind = nCons - c - 1;
   
    curr_resps = f1['mean'][c, val_sfs][0]; # additional layer of array to unwrap
    curr_sem = f1['sem'][c, val_sfs][0];
    v_sfs = ~np.isnan(curr_resps);
    data_sfs = all_sfs[val_sfs][v_sfs]

    curr_resps_adj = np.array(curr_resps[v_sfs]-respAdj);
    abvThresh = [curr_resps_adj > minResp_toPlot];

    col = [(nCons-c-1)/float(nCons), (nCons-c-1)/float(nCons), (nCons-c-1)/float(nCons)];

    for i in range(2):

      # plot data
      sfsAx[c_plt_ind, i].errorbar(data_sfs[abvThresh], curr_resps_adj[abvThresh], 
                                    curr_sem[v_sfs][abvThresh], fmt='o', markersize=10, clip_on=False, color=col);

      # plot descriptive model fit and inferred characteristic frequency
      if fromFile:
        curr_mod_params = hf.load_modParams(which_cell, c);
      else:
        curr_mod_params = descrFits['params'][c]; 
      mod_resps = hf.get_descrResp(curr_mod_params, data_sfs, sf_DoG_model);
      mod_resps_plt = hf.get_descrResp(curr_mod_params, sfs_plot, sf_DoG_model);
      sfsAx[c_plt_ind, i].plot(sfs_plot, mod_resps_plt, clip_on=False, color=col)
      # -- also put text of DoG parameters, if applicable
      if sf_DoG_model == 1 or sf_DoG_model == 2:
        try:
          # if it's a DoG, let's also put the parameters in text (left side only)
          sfsAx[c_plt_ind, 0].text(0.05, 0.075, '%d,%.2f' % (*curr_mod_params[0:2], ), transform=sfsAx[c_plt_ind,0].transAxes, horizontalalignment='left', fontsize='small', verticalalignment='bottom');
          sfsAx[c_plt_ind, 0].text(0.05, 0.025, '%.2f,%.2f' % (*curr_mod_params[2:], ), transform=sfsAx[c_plt_ind,0].transAxes, horizontalalignment='left', fontsize='small', verticalalignment='bottom');
        except:
          pass

      # now plot characteristic frequency!
      f_c = hf.dog_charFreq(curr_mod_params, sf_DoG_model);
      sfsAx[c_plt_ind, i].plot(f_c, 1, 'v', color='k');

      sfsAx[c_plt_ind, i].set_xlim((min(sfs_plot), max(sfs_plot)));

      sfsAx[c_plt_ind, i].set_xscale('log');
      if fromFile:
        varExpl = hf.var_expl_direct(curr_resps[v_sfs], mod_resps);
      else:
        varExpl = descrFits['varExpl'][c];
      sfsAx[c_plt_ind, i].set_title('SF tuning: contrast: %.3f%%, %.1f%% varExpl' % (all_cons[c], varExpl));

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      #sfsAx[c_plt_ind, i].tick_params(labelsize=15, width=1, length=8, direction='out');
      #sfsAx[c_plt_ind, i].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...	
      sns.despine(ax=sfsAx[c_plt_ind, i], offset=10, trim=False); 

      yAxStr = 'response ';
      if i == 0 and c_plt_ind == (nCons-1):
        sfsAx[c_plt_ind, i].set_xlabel('spatial frequency (c/deg)'); 
        sfsAx[c_plt_ind, i].set_ylabel(str(yAxStr + '(spikes/s)'));

      if i == 0: # linear...
        sfsAx[c_plt_ind, i].set_ylim((0, 1.5*maxResp));
      elif i == 1: # log
        sfsAx[c_plt_ind, i].set_yscale('log');
        sfsAx[c_plt_ind, i].set_ylim((minResp_toPlot, 300));
        sfsAx[c_plt_ind, i].set_aspect('equal'); # if both axes are log, must make equal scales!


saveName = "/cell_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'sfTuning/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(fSfs)
plt.close(fSfs)
pdfSv.close()

#########
# #### Plot response versus contrast curves (RVC)
#########

# #### Plot contrast response functions with descriptive RVC model predictions

# which sfs have at least one contrast presentation? within a dispersion, all cons have the same # of sfs
v_sfs = all_sfs;
n_v_sfs = len(all_sfs);
n_rows = int(np.ceil(n_v_sfs/np.floor(np.sqrt(n_v_sfs)))); # make this close to a rectangle/square in arrangement (cycling through sfs)
n_cols = int(np.ceil(n_v_sfs/n_rows));
f, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*10, n_rows*10), sharex = True, sharey = 'col');

f.suptitle('RVC - %s #%d' % (cellStruct['cellType'], which_cell));

for sf in range(n_v_sfs):
    row_ind = int(sf/n_cols);
    col_ind = np.mod(sf, n_cols);
    if n_cols > 1:
      plt_y = (row_ind, col_ind);
    else: # pyplot makes it (n_rows, ) if n_cols == 1
      plt_y = (row_ind, );

    # organize (measured) responses
    respAbBaseline = np.reshape(f1['mean'][:, sf], (len(all_cons), )); 
    respVar = np.reshape(f1['sem'][:, sf], (len(all_cons), ));
    col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];
    ax[plt_y].errorbar(all_cons, respAbBaseline - respAdj, respVar, fmt='o', clip_on=False, markersize=10, color=col);

    # now plot model
    cons_plot = np.geomspace(np.min(all_cons), np.max(all_cons), 100);
    prms_curr = rvcFits['params'][sf];
    rvcResps = hf.get_rvcResp(prms_curr, cons_plot, rvcMod);
    ax[plt_y].plot(cons_plot, np.maximum(rvcResps - respAdj, 0.1), 'r--', clip_on=False, color=col);

    ax[plt_y].set_xscale('log', basex=10); # was previously symlog, linthreshx=0.01
    if col_ind == 0:
      ax[plt_y].set_xlim([0.01, 1]);
      if row_ind == (n_rows-1): # i.e. only at the last one
        ax[plt_y].set_xlabel('contrast', fontsize='medium');
        ax[plt_y].set_ylabel('response (spikes/s)', fontsize='medium');
      ax[plt_y].legend();
    varExpl_curr = rvcFits['varExpl'][sf];

    ax[plt_y].set_title('sf: %.3f [vExp=%.2f]' % (all_sfs[sf], varExpl_curr), fontsize='large');

    # Set ticks out, remove top/right axis, put ticks only on bottom/left
    #ax[plt_y].tick_params(labelsize=25, width=2, length=16, direction='out');
    #ax[plt_y].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...

sns.despine(offset = 10, trim=False);

saveName = "/cell_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'RVC%s/' % rvcSuff));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(f)
plt.close(f)
pdfSv.close()


######## all on one axis

v_sfs = all_sfs;
n_v_sfs = len(all_sfs);

maxResp = np.max(np.max(f1['mean']));
ymin = 1e-1
f, ax = plt.subplots(1, 2, figsize=(35, 20), sharex=True, sharey='row');

lines = [];
for sf in reversed(range(n_v_sfs)):

    # plot data
    col = [sf/float(n_v_sfs), sf/float(n_v_sfs), sf/float(n_v_sfs)];
    resps = f1['mean'][:, sf];
    whereAbBaseline = np.where(resps>ymin)[0];
    curr_line, = ax[0].plot(all_cons[whereAbBaseline], resps[whereAbBaseline] - respAdj, '-o', clip_on=False, color=col, label=str(np.round(all_sfs[sf], conDig)));
    lines.append(curr_line);

    # now plot model
    cons_plot = np.geomspace(np.min(all_cons), np.max(all_cons), 100);
    prms_curr = rvcFits['params'][sf];
    if rvcMod == 0: # i.e. movshon form
      rvcModel = hf.get_rvc_model();
      rvcResps = rvcModel(*prms_curr, cons_plot);
    elif rvcMod == 1 or rvcMod == 2: # naka-rushton (or modified version)
      rvcResps = hf.naka_rushton(cons_plot, prms_curr)
    ax[1].plot(cons_plot, np.maximum(rvcResps, 0.1), color=col, \
                     clip_on=False);

for i in range(2):

  #ax[i].set_aspect('equal', 'box'); 
  ax[i].set_xlim((0.5*min(all_cons), 1.2*max(all_cons)));
  ax[i].set_ylim((ymin, 1.5*maxResp));

  ax[i].set_xscale('log');
  ax[i].set_yscale('log');
  ax[i].set_aspect('equal');
  ax[i].set_xlabel('con (%)'); 

  # Set ticks out, remove top/right axis, put ticks only on bottom/left
  #ax[i].tick_params(labelsize=15, width=2, length=16, direction='out');
  #ax[i].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
  ax[i].set_ylabel('resp (sps)');
  ax[i].set_title('RVC - %s #%d' % (cellStruct['cellType'], which_cell));
  ax[i].legend();

sns.despine(offset=10, trim=False); 

saveName = "/allSfs_cell_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'RVC%s/' % rvcSuff));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(full_save + saveName);
pdfSv.savefig(f)
plt.close(f)
pdfSv.close()
