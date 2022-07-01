# coding: utf-8

import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
from matplotlib.ticker import FuncFormatter
import seaborn as sns
sns.set(style='ticks')
import helper_fcns_sach as hf
from scipy.stats import poisson, nbinom

import pdb

import sys

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_style.mplstyle');
from matplotlib import rcParams
for i in range(2):
    # must run twice for changes to take effect?
    from matplotlib import rcParams, cm
    rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = ['Helvetica']
    rcParams['font.style'] = 'oblique'
    rcParams['font.size'] = 40;
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['lines.linewidth'] = 3;
    rcParams['lines.markeredgewidth'] = 0; # remove edge??
    rcParams['axes.linewidth'] = 3;
    rcParams['lines.markersize'] = 12; # 8 is the default

    rcParams['xtick.major.size'] = 25
    rcParams['xtick.minor.size'] = 12
    rcParams['ytick.major.size'] = 25
    rcParams['ytick.minor.size'] = 12; # i.e. don't have minor ticks on y...
    rcParams['xtick.major.width'] = 2
    rcParams['xtick.minor.width'] = 2
    rcParams['ytick.major.width'] = 2
    rcParams['ytick.minor.width'] = 2

y_lblpad = 6;
x_lblpad = 8;
lblSize = 40;

specify_ticks = True; # specify the SF ticks (x-axis for SF plots?)

which_cell   = int(sys.argv[1]);
sf_loss_type = int(sys.argv[2]);
sf_DoG_model = int(sys.argv[3]);
rvcMod       = int(sys.argv[4]);
joint        = int(sys.argv[5]);
isHPC        = int(sys.argv[6]);
phAdj        = int(sys.argv[7]);
if len(sys.argv) > 8:
  plot_sem_on_log = int(sys.argv[8]);
else:
  plot_sem_on_log = 1; # plot the S.E.M. for log SF plots?
if len(sys.argv) > 9:
  plot_zFreq = int(sys.argv[9]);
else:
  plot_zFreq = 1; # plot the zero frequency?
if len(sys.argv) > 10:
  fromFile     = int(sys.argv[10]); 
else:
  fromFile = 0; # basically deprecated...

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
fLname = 'descrFits%s_s220609' % HPC;
#fLname = 'descrFits%s_s220520' % HPC;
mod_str = hf.descrMod_name(sf_DoG_model);
fLname_full = hf.descrFit_name(sf_loss_type, fLname, mod_str, joint=joint, phAdj=phAdj);
descrFits = hf.np_smart_load(dataPath + fLname_full);
descrFits = descrFits[which_cell-1]; # just get this cell

rvcSuff = hf.rvc_mod_suff(rvcMod);
rvcBase = 'rvcFits%s_220531' % HPC;
#rvcBase = 'rvcFits%s_220512' % HPC;
#rvcBase = 'rvcFits%s_220518' % HPC;
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

f, ax = plt.subplots(1, 2, figsize=(35, 20))#, sharey=True);

minResp_toPlot = 1; 
#minResp_toPlot = 5e-1; 

lines = [];
for c in reversed(range(nCons)):

    # summary plot (just data) first
    if plot_zFreq: 
        # we want to plot zero frequency, but need to keep axes equal
        # --- so, we'll pretend that the zero frequency is just a slightly lower, non-zero frequency
        val_sfs = range(len(all_sfs)); # get all indices
        curr_sfs = np.maximum(1e-2, all_sfs[val_sfs]);
    else:
        val_sfs = np.where(all_sfs>0); # do not plot the zero sf condition
        curr_sfs = all_sfs;
    sfs_plot = np.geomspace(curr_sfs[val_sfs[0]], curr_sfs[val_sfs[-1]], 100);

    # plot data
    col = [(nCons-c-1)/float(nCons), (nCons-c-1)/float(nCons), (nCons-c-1)/float(nCons)];
    respAbBaseline = np.reshape(f1['mean'][c, val_sfs], (len(val_sfs), ));
    respVar = np.reshape(f1['sem'][c, val_sfs], (len(val_sfs), )); 
    abvBound = respAbBaseline>minResp_toPlot;
    # DO NOT actually plot error bar on these...unless plot_sem_on_log (for publications)
    if plot_sem_on_log:
      # errbars should be (2,n_sfs)
      high_err = respVar; # no problem with going to higher values
      low_err = np.minimum(respVar, respAbBaseline-minResp_toPlot-1e-2); # i.e. don't allow us to make the err any lower than where the plot will cut-off (incl. negatives)
      errs = np.vstack((low_err, high_err));
      ax[0].errorbar(curr_sfs[val_sfs][abvBound], respAbBaseline[abvBound] - respAdj, errs[:, abvBound], fmt='o', linestyle='-', clip_on=False, color=col, label='%d%%' % (int(100*np.round(all_cons[c], conDig))));
      # AND add it to the model plot, too (without label, line)
      ax[1].errorbar(curr_sfs[val_sfs][abvBound], respAbBaseline[abvBound] - respAdj, errs[:, abvBound], fmt='o', clip_on=False, color=col);
    else:
      ax[0].plot(curr_sfs[val_sfs][abvBound], respAbBaseline[abvBound] - respAdj, '-o', clip_on=False, color=col, label='%d%%' % (int(100*np.round(all_cons[c], conDig))))
 
    # then descriptive fit
    sfs_plot = np.geomspace(curr_sfs[val_sfs[0]], curr_sfs[val_sfs[-1]], 100);
    prms_curr = descrFits['params'][c];
    descrResp = hf.get_descrResp(prms_curr, sfs_plot, sf_DoG_model);
    abvZero = descrResp>minResp_toPlot;
    ax[1].plot(sfs_plot[abvZero], descrResp[abvZero], '-', clip_on=False, color=col); #, label=str(np.round(all_cons[c], conDig)))

for i in range(2):
  ax[i].set_xlim((0.5*np.min(curr_sfs[val_sfs]), 1.2*np.max(curr_sfs[val_sfs])));
  if not specify_ticks or maxResp>90:
    ax[i].set_ylim((minResp_toPlot, 300)); # common y axis for ALL plots
  else:
    ax[i].set_ylim((minResp_toPlot, 110));

  ax[i].set_xscale('log');
  ax[i].set_yscale('log');
  ax[i].set_aspect('equal'); # if both axes are log, must make equal scales!
  ax[i].set_xlabel('Spatial frequency (c/deg)'); 

  ax[i].set_ylabel('Response (spikes/s)', labelpad=y_lblpad);
  if i == 0:
    ax[i].set_title('SF tuning - %s #%d' % (cellStruct['cellType'], which_cell));
    ax[i].legend(fontsize='x-small');

  for jj, axis in enumerate([ax[i].xaxis, ax[i].yaxis]):
      axis.set_major_formatter(FuncFormatter(lambda x,y: '%d' % x if x>=1 else '%.1f' % x if x>=0.1 else '%.2f' % x)) # this will make everything in non-scientific notation!
      if jj == 0 and specify_ticks: # i.e. x-axis
          core_ticks = np.array([1]); # always include 1 c/deg
          pltd_sfs = curr_sfs[val_sfs];
          if np.min(pltd_sfs)<=0.3:
            core_ticks = np.hstack((0.3, core_ticks, 3));
            if np.min(pltd_sfs)<=0.15:
              core_ticks = np.hstack((0.1, core_ticks));
              if np.min(pltd_sfs)<=0.05:
                core_ticks = np.hstack((0.03, core_ticks));
          else:
            core_ticks = np.hstack((0.5, core_ticks, 5));
          if np.max(pltd_sfs)>=7:
            core_ticks = np.hstack((core_ticks, 10));
          axis.set_ticks(core_ticks);
      elif jj == 1: # y axis, make sure we also show the ticks, even though the axes are shared
        axis.set_tick_params(labelleft=True); 

sns.despine(offset=10, trim=False); 

sem_str = '_sem' if plot_sem_on_log else '';
saveName = "/allCons_cell_%03d.pdf" % (which_cell)
full_save = os.path.dirname(str(save_loc + 'sfTuning%s/' % sem_str));
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
fSfs, sfsAx = plt.subplots(nCons, 2, figsize=(2*10, 12*nCons))#, sharey=False);

if plot_zFreq: 
    # we want to plot zero frequency, but need to keep axes equal
    # --- so, we'll pretend that the zero frequency is just a slightly lower, non-zero frequency
    val_sfs = range(len(all_sfs)); # get all indices
    curr_sfs = np.maximum(1e-2, all_sfs[val_sfs]);
    #curr_sfs = np.maximum(np.min(all_sfs[all_sfs>0])/2, all_sfs[val_sfs]);
else:
    val_sfs = np.where(all_sfs>0); # do not plot the zero sf condition
    curr_sfs = all_sfs;
sfs_plot = np.geomspace(curr_sfs[val_sfs[0]], curr_sfs[val_sfs[-1]], 100);

minResp_toPlot = 5e-1;

for c in reversed(range(nCons)):
    c_plt_ind = nCons - c - 1;
   
    curr_resps = f1['mean'][c, val_sfs];
    curr_sem = f1['sem'][c, val_sfs];
    v_sfs = ~np.isnan(curr_resps);
    data_sfs = curr_sfs[val_sfs][v_sfs]

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
      if i == 0:
        sfsAx[c_plt_ind, i].set_title('con: %.1f%%, %.1f%% vExp' % (100*all_cons[c], varExpl));
      #sfsAx[c_plt_ind, i].set_title('SF tuning: contrast: %.3f%%, %.1f%% varExpl' % (all_cons[c], varExpl));

      # Set ticks out, remove top/right axis, put ticks only on bottom/left
      #sfsAx[c_plt_ind, i].tick_params(labelsize=15, width=1, length=8, direction='out');
      #sfsAx[c_plt_ind, i].tick_params(width=1, length=4, which='minor', direction='out'); # minor ticks, too...	
      sns.despine(ax=sfsAx[c_plt_ind, i], offset=10, trim=False); 

      yAxStr = 'Response ';
      if i == 0 and c_plt_ind == (nCons-1):
        sfsAx[c_plt_ind, i].set_xlabel('Spatial frequency (c/deg)', labelpad=x_lblpad); 
        sfsAx[c_plt_ind, i].set_ylabel(str(yAxStr + '(spikes/s)'), labelpad=y_lblpad);

      if i == 0: # linear...
        sfsAx[c_plt_ind, i].set_ylim((0, 1.5*maxResp));
      elif i == 1: # log
        sfsAx[c_plt_ind, i].set_yscale('log');
        sfsAx[c_plt_ind, i].set_ylim((minResp_toPlot, 300));
        sfsAx[c_plt_ind, i].set_aspect('equal'); # if both axes are log, must make equal scales!

      for jj, axis in enumerate([sfsAx[c_plt_ind, i].xaxis, sfsAx[c_plt_ind, i].yaxis]):
          axis.set_major_formatter(FuncFormatter(lambda x,y: '%d' % x if x>=1 else '%.1f' % x)) # this will make everything in non-scientific notation!
          if jj == 0 and specify_ticks: # i.e. x-axis
              core_ticks = np.array([1]); # always include 1 c/deg
              pltd_sfs = data_sfs;
              if np.min(pltd_sfs)<=0.3:
                  core_ticks = np.hstack((0.3, core_ticks, 3));
              else:
                  core_ticks = np.hstack((0.5, core_ticks, 5));
              if np.max(pltd_sfs)>=7:
                  core_ticks = np.hstack((core_ticks, 10));
              axis.set_ticks(core_ticks)
          elif jj == 1: # y axis, make sure we also show the ticks, even though the axes are shared
              axis.set_tick_params(labelleft=True); 

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
        ax[plt_y].set_xlabel('Contrast', fontsize='medium', labelpad=x_lblpad);
        ax[plt_y].set_ylabel('Response (spikes/s)', fontsize='medium', labelpad=y_lblpad);
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
f, ax = plt.subplots(1, 2, figsize=(35, 20))#, sharex=True, sharey='row');

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
  ax[i].set_xlabel('Contrast (%)', labelpad=x_lblpad); 

  # Set ticks out, remove top/right axis, put ticks only on bottom/left
  #ax[i].tick_params(labelsize=15, width=2, length=16, direction='out');
  #ax[i].tick_params(width=2, length=8, which='minor', direction='out'); # minor ticks, too...
  ax[i].set_ylabel('Response (spikes/s)', labelpad=y_lblpad);
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
