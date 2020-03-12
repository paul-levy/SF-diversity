##################
### plot_basics.py - use to plot the basic characterizations
### -- i.e., ori16, tf11, rfsize10, rvc10asc, sf11
##################

import os
import sys
import numpy as np
import matplotlib
import matplotlib.cm as cm
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns
sns.set(style='ticks')
from scipy.stats import poisson, nbinom
from scipy.stats.mstats import gmean

import helper_fcns as hf
import model_responses as mod_resp

import warnings
warnings.filterwarnings('once');

import pdb

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_style.mplstyle');
from matplotlib import rcParams
rcParams['font.size'] = 20;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 2.5;
rcParams['axes.linewidth'] = 1.5;
rcParams['lines.markersize'] = 5;
rcParams['font.style'] = 'oblique';

cellNum   = int(sys.argv[1]);
expDir    = sys.argv[2];

loc_base = os.getcwd() + '/';

data_loc = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + 'figures/';

### Load datalist, specific cell information
dataListNm = hf.get_datalist(expDir);
dataList = hf.np_smart_load(data_loc + dataListNm);

cellName = dataList['unitName'][cellNum-1];
expInd = hf.get_exp_ind(data_loc, cellName)[0]

### Load the "basics" information
basic_names, basic_order = dataList['basicProgName'][cellNum-1], dataList['basicProgOrder']
basics = hf.get_basic_tunings(basic_names, basic_order)
# - and get the basic characterization outputs
[rf, tf, sf, rvc, ori] = [basics[x] for x in ['rfsize', 'tf', 'sf', 'rvc', 'ori']];

### set up the figure
nrow, ncol = 3, 2;
f, ax = plt.subplots(nrow, ncol, figsize=(ncol*10, nrow*10));

#############
### Response versus contrast
#############
if rvc is not None:
  # data
  convals, conresps, constderr = [rvc['rvc_exp'][x] for x in ['conVals', 'counts_mean', 'counts_stderr']]
  ax[0, 0].errorbar(convals, conresps, constderr, fmt='o', color='k');
  # model
  modNum = rvc['rvcMod'];
  plt_cons = np.geomspace(convals[0], convals[-1], 100);
  mod_resp = hf.get_rvcResp(rvc['params'], plt_cons, modNum)
  c50 = rvc['c50']
  ax[0, 0].plot(c50, 0, 'kv', label='c50 (%d%%)' % (100*c50));
  ax[0, 0].semilogx(plt_cons, mod_resp, 'k-')
  ax[0, 0].legend();

#############
### RF size tuning
#############
if rf is not None:
  # first the data
  diskResp, annResp = rf['rf_exp']['counts_mean'][:, 0], rf['rf_exp']['counts_mean'][:, 1]
  diskStdErr, annStdErr = rf['rf_exp']['counts_stderr'][:, 0], rf['rf_exp']['counts_stderr'][:, 1]
  diskVals, annVals = rf['rf_exp']['diskVals'], rf['rf_exp']['annulusVals']
  ax[0, 1].errorbar(diskVals, diskResp, diskStdErr, fmt='o', color='k', label='disk');
  ax[0, 1].errorbar(annVals, annResp, annStdErr, fmt='o', color='r', label='annulus');
  # then the model fits
  ax[0, 1].semilogx(rf['to_plot']['diams'], rf['to_plot']['resps'], 'k-');
  ax[0, 1].semilogx(rf['to_plot']['ann'], rf['to_plot']['ann_resp'], 'r-');
  ax[0, 1].set_xlabel('size (deg)');
  ax[0, 1].set_ylabel('response (spks/s)');
  ax[0, 1].legend();

#############
### SF tuning [1, 1]
#############
if sf is not None:
  # data
  sfvals, sfresps, sfstderr = [sf['sf_exp'][x] for x in ['sfVals', 'counts_mean', 'counts_stderr']];
  ax[1, 1].errorbar(sfvals, sfresps[:,0], sfstderr[:,0], fmt='o', color='k');
  # model
  plt_sfs = np.geomspace(sfvals[0], sfvals[-1], 100);
  mod_resp = hf.flexible_Gauss(sf['sfParams'], plt_sfs)
  ax[1, 1].plot(sf['charFreq'], 0, 'kv', label='char freq (%.1f Hz)' % sf['charFreq'])
  ax[1, 1].semilogx(plt_sfs, mod_resp, 'k-')
  ax[1, 1].set_title('Peak %.1f Hz, bw %.1f oct' % (sf['sfPref'], sf['sfBW_oct']))
  #ax[1, 1].legend();

#############
### TF tuning
#############
if tf is not None:
  # data
  tfvals, tfresps, tfstderr = [tf['tf_exp'][x] for x in ['tfVals', 'counts_mean', 'counts_stderr']];
  ax[1, 0].errorbar(tfvals, tfresps[:,0], tfstderr[:,0], fmt='o', color='k');
  # model
  plt_tfs = np.geomspace(tfvals[0], tfvals[-1], 100);
  mod_resp = hf.flexible_Gauss(tf['tfParams'], plt_tfs)
  ax[1, 0].plot(tf['charFreq'], 0, 'kv', label='char freq (%.1f Hz)' % tf['charFreq'])
  ax[1, 0].semilogx(plt_tfs, mod_resp, 'k-')
  ax[1, 0].set_title('Peak %.1f Hz, bw %.1f oct' % (tf['tfPref'], tf['tfBW_oct']))
  #ax[1, 0].legend();

#############
### Orientation tuning
#############
if ori is not None:
  # data
  orivals, oriresps, oristderr = [ori['ori_exp'][x] for x in ['oriVals', 'counts_mean', 'counts_stderr']];
  ax[2, 0].errorbar(orivals, oriresps, oristderr, fmt='o', color='k');
  # model
  plt_oris = np.linspace(0, 2*np.pi, 100);
  oriMod, _ = hf.get_ori_mod();
  mod_resp = oriMod(*ori['params'], plt_oris)
  ax[2, 0].plot(np.rad2deg(plt_oris), mod_resp, 'k-');
  ax[2, 0].set_title('Peak %d deg, bw %d deg, cv %.2f, ds %.2f' % (ori['pref'], ori['bw'], ori['cv'], ori['DS']))
  #ax[2, 0].legend();

sns.despine(fig=f, offset=10);

#############
### Finally, save
#############
saveName = "/cell_%03d.pdf" % (cellNum)
full_save = os.path.dirname(str(save_loc + 'basics/'));
if not os.path.exists(full_save):
  os.makedirs(full_save);
pdfSv = pltSave.PdfPages(str(full_save + saveName));
pdfSv.savefig(f)
pdfSv.close();
