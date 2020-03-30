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

rvcName = 'rvcFits_191023' # updated - computes RVC for best responses (i.e. f0 or f1)
rvcMod = 1; # naka rushton

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
### Let's load the sfMix responses, just for comparison (we can compare sf and rvc responses)
#############

S = hf.np_smart_load(data_loc + cellName + '_sfm.npy')
expData = S['sfm']['exp']['trial'];

### compute f1f0 ratio, and load the corresponding F0 or F1 responses
f1f0_rat = hf.compute_f1f0(expData, cellNum, expInd, data_loc)[0];

if f1f0_rat > 1 or expDir == 'LGN/': # i.e. if we're looking at a simple cell, then let's get F1
  if rvcName is not None:
    rvcFits = hf.get_rvc_fits(data_loc, expInd, cellNum, rvcName=rvcName, rvcMod=rvcMod);
  else:
    rvcFits = None
  spikes_byComp = hf.get_spikes(expData, get_f0=0, rvcFits=rvcFits, expInd=expInd);
  spikes = np.array([np.sum(x) for x in spikes_byComp]);
  rates = True; # when we get the spikes from rvcFits, they've already been converted into rates (in hf.get_all_fft)
  baseline = None; # f1 has no "DC", yadig?
else: # otherwise, if it's complex, just get F0
  spikes = hf.get_spikes(expData, get_f0=1, rvcFits=None, expInd=expInd);
  rates = False; # get_spikes without rvcFits is directly from spikeCount, which is counts, not rates!
  baseline = hf.blankResp(expData, expInd)[0]; # we'll plot the spontaneous rate
  # why mult by stimDur? well, spikes are not rates but baseline is, so we convert baseline to count (i.e. not rate, too)
  spikes = spikes - baseline*hf.get_exp_params(expInd).stimDur; 

_, _, respOrg, respAll = hf.organize_resp(spikes, expData, expInd);
resps, stimVals, val_con_by_disp, _, _ = hf.tabulate_responses(expData, expInd, overwriteSpikes=spikes, respsAsRates=rates);
predResps = resps[2];

respMean = resps[0]; # equivalent to resps[0];
respStd = np.nanstd(respAll, -1); # take std of all responses for a given condition
# compute SEM, too
findNaN = np.isnan(respAll);
nonNaN  = np.sum(findNaN == False, axis=-1);
respSem = np.nanstd(respAll, -1) / np.sqrt(nonNaN);

# organize stimulus values
all_disps = stimVals[0];
all_cons = stimVals[1];
all_sfs = stimVals[2];

### get the sfRef and rvcRef - i.e. single grating, high contrast/prefSf SF/RVC tuning
sfRef = hf.nan_rm(respMean[0, :, -1]); # high contrast tuning
sfRefSEM = hf.nan_rm(respSem[0, :, -1]);
rvc10_sf = np.unique(rvc['rvc_exp']['byTrial']['sf']);
# now, let's get two RVCs - one at nearest sf to what was used in rvc10, one at peak sfMix response 
sfPeak = np.argmax(sfRef); # stupid/simple, but just get the rvc for the max response
rvcRef_sf = all_sfs[sfPeak];
sfComp = np.argmin(np.square(np.log2(all_sfs) - np.log2(rvc10_sf)));
rvcComp_sf = all_sfs[sfComp];
rvcRef_sfs = [rvcRef_sf, rvcComp_sf];
v_cons_single = val_con_by_disp[0]
rvcRefs = [hf.nan_rm(respMean[0, sfInd, v_cons_single]) for sfInd in (sfPeak, sfComp)];
rvcRefsSEM = [hf.nan_rm(respSem[0, sfInd, v_cons_single]) for sfInd in (sfPeak, sfComp)];
rvcRefsColor = ['r', 'b']

#############
### Response versus contrast
#############
if rvc is not None:
  # data
  respInd = rvc['isSimple'];
  convals, conresps, constderr = [rvc['rvc_exp'][x] for x in ['conVals', 'counts_mean', 'counts_stderr']]
  rvc10_sf = np.unique(rvc['rvc_exp']['byTrial']['sf']);
  ax[0, 0].errorbar(convals, conresps[:, respInd], constderr[:, respInd], fmt='o', color='k');
  # model
  modNum = rvc['rvcMod'];
  plt_cons = np.geomspace(convals[0], convals[-1], 100);
  mod_resp = hf.get_rvcResp(rvc['params'], plt_cons, modNum)
  c50 = rvc['c50']; c50_eval = rvc['c50_eval'];
  ax[0, 0].plot(c50, 0, 'kv', label='c50/eval (%d%%, %d%%) at %.1f cpd' % (100*c50, 100*c50_eval, rvc10_sf));
  ax[0, 0].semilogx(plt_cons, mod_resp, 'k-')
  ax[0, 0].set_xlabel('contrast (%%)');
### NOW, let's also plot the RVC for the nearest optimal SF as it appears in sfMix
# if possible, let's also plot the RVC fit from sfMix's prefSf RVC
if rvcName is not None and (expDir == 'V1/' or expDir == 'LGN/'): 
  # we CANNOT do this for V1_orig; we SHOULD be able to do for altExp 
  #TODO: will need to fix how we get the parameters to account for differing way in 
  # fit_rvc_f0 vs. rvc_adjusted_fit on how we store parameters [d,sf,prm] vs. [d]['params'][sf]
  rvcFits = hf.get_rvc_fits(data_loc, expInd, cellNum, rvcName=rvcName, rvcMod=rvcMod);
  rel_rvcs = [rvcFits[0]['params'][sfInd] for sfInd in (sfPeak, sfComp)]; # we get 0 dispersion, peak SF
  plt_cons = np.geomspace(all_cons[0], all_cons[-1], 50);
  c50s = [hf.get_c50(rvcMod, rel_rvc) for rel_rvc in rel_rvcs];
  c50_emps = [hf.c50_empirical(rvcMod, rel_rvc)[1] for rel_rvc in rel_rvcs]; # determine c50 by optimization, numerical approx. (take the latter)
  if rvcMod == 0:
    rvc_mod = hf.get_rvc_model();
    rvcmodResps = [rvc_mod(*rel_rvc, plt_cons) for rel_rvc in rel_rvcs];
  else: # i.e. mod=1 or mod=2
    rvcmodResps = [hf.naka_rushton(plt_cons, rel_rvc) for rel_rvc in rel_rvcs];
  if baseline is not None:
    rvcmodResp = [rvcmodResp + baseline for rvcmodResp in rvcmodResps]; # why? we subtract baseline before fitting RVC; add it back, here
  [ax[0, 0].plot(plt_cons, rvcmodResp, color=clr, linestyle='-', label='rvc fit (c50/eval=%d%%, %d%%)' % (100*c50, 100*c50_eval)) for rvcmodResp,c50,c50_eval,clr in zip(rvcmodResps, c50s, c50_emps, rvcRefsColor)]
  # and save it
### and plot the data from sfMix
if baseline is not None:
  rvcRefs = [rvcRef + baseline for rvcRef in rvcRefs];
[ax[0, 0].errorbar(all_cons[v_cons_single], rvcRef, rvcRefSEM, color=clr, fmt='o', label='sfMix (d0, %.1f cpd)' % rvcRef_sf, clip_on=False) for rvcRef, rvcRefSEM, rvcRef_sf, clr in zip(rvcRefs, rvcRefsSEM, rvcRef_sfs, rvcRefsColor)]
ax[0, 0].legend();


#############
### RF size tuning
#############
if rf is not None:
  respInd = rf['isSimple'];
  # first the data
  diskResp, annResp = rf['rf_exp']['counts_mean'][:, 0, respInd], rf['rf_exp']['counts_mean'][:, 1, respInd]
  diskStdErr, annStdErr = rf['rf_exp']['counts_stderr'][:, 0, respInd], rf['rf_exp']['counts_stderr'][:, 1, respInd]
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
  respInd = sf['isSimple'];
  # data
  sfvals, sfresps, sfstderr = [sf['sf_exp'][x] for x in ['sfVals', 'counts_mean', 'counts_stderr']];
  ax[1, 1].errorbar(sfvals, sfresps[:,0, respInd], sfstderr[:,0, respInd], fmt='o', color='k');
  # model
  plt_sfs = np.geomspace(sfvals[0], sfvals[-1], 100);
  mod_resp = hf.flexible_Gauss(sf['sfParams'], plt_sfs)
  ax[1, 1].plot(sf['charFreq'], 0, 'kv', label='char freq (%.1f Hz)' % sf['charFreq'])
  ax[1, 1].semilogx(plt_sfs, mod_resp, 'k-')
  ax[1, 1].set_title('SF: Peak %.1f cyc/deg, bw %.1f oct' % (sf['sfPref'], sf['sfBW_oct']))
  ax[1, 1].set_xlabel('spatial frequency (cyc/sec)');
# also plot the sfMix high contrast, single grating sf tuning
### plot reference tuning [row 1 (i.e. 2nd row)]
if baseline is not None:
  sfRef = sfRef + baseline;
ax[1, 1].errorbar(all_sfs, sfRef, sfRefSEM, color='r', fmt='o', label='ref. tuning (d0, high con)', clip_on=False)
ax[1, 1].legend();

#############
### TF tuning
#############
if tf is not None:
  respInd = tf['isSimple'];
  # data
  tfvals, tfresps, tfstderr = [tf['tf_exp'][x] for x in ['tfVals', 'counts_mean', 'counts_stderr']];
  ax[1, 0].errorbar(tfvals, tfresps[:,0, respInd], tfstderr[:,0, respInd], fmt='o', color='k');
  # model
  plt_tfs = np.geomspace(tfvals[0], tfvals[-1], 100);
  mod_resp = hf.flexible_Gauss(tf['tfParams'], plt_tfs)
  ax[1, 0].plot(tf['charFreq'], 0, 'kv', label='char freq (%.1f Hz)' % tf['charFreq'])
  ax[1, 0].semilogx(plt_tfs, mod_resp, 'k-')
  ax[1, 0].set_title('TF: Peak %.1f Hz, bw %.1f oct' % (tf['tfPref'], tf['tfBW_oct']))
  ax[1, 0].set_xlabel('temporal frequency (cyc/sec)');
  #ax[1, 0].legend();

#############
### Orientation tuning
#############
if ori is not None:
  respInd = ori['isSimple'];
  # data
  orivals, oriresps, oristderr = [ori['ori_exp'][x] for x in ['oriVals', 'counts_mean', 'counts_stderr']];
  ax[2, 0].errorbar(orivals, oriresps[:, respInd], oristderr[:, respInd], fmt='o', color='k');
  # model
  plt_oris = np.linspace(0, 2*np.pi, 100);
  oriMod, _ = hf.get_ori_mod();
  mod_resp = oriMod(*ori['params'], plt_oris)
  ax[2, 0].plot(np.rad2deg(plt_oris), mod_resp, 'k-');
  ax[2, 0].set_title('Peak %d deg, bw %d deg, cv %.2f, ds %.2f' % (ori['pref'], ori['bw'], ori['cv'], ori['DS']))
  ax[2, 0].set_xlabel('orientation (deg)');
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
