import helper_fcns as hf
import helper_fcns_sfBB as hf_sf
import numpy as np
import scipy.optimize as opt
import os, sys
import importlib as il
import pdb

import matplotlib
import matplotlib.cm as cm
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import seaborn as sns
sns.set(style='ticks')

import warnings
warnings.filterwarnings('once');

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_cluster.mplstyle');
from matplotlib import rcParams
rcParams['font.size'] = 40;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 4;
rcParams['axes.linewidth'] = 4;
rcParams['lines.markersize'] = 9;
rcParams['font.style'] = 'oblique';
rcParams['xtick.major.size'] = 24;
rcParams['xtick.minor.size'] = 12;
rcParams['ytick.major.size'] = 24;
rcParams['ytick.minor.size'] = 12;

majWidth = 4;
minWidth = 4;
lblSize = 40;

### Load input parameters
cellNum   = int(sys.argv[1]);
expDir    = sys.argv[2]; 
# -- how wide to model the onset transient?
onsetDur  = int(sys.argv[3]); # in mS
# -- how wide to make each averaged bin in the sliding PSTH
halfWidth = int(sys.argv[4]); # in mS
toPlot = int(sys.argv[5]);

loc_base = os.getcwd() + '/';

data_loc = loc_base + expDir + 'structures/';
save_loc = loc_base + expDir + 'figures/onset/';

### Load datalist, cell, and "onset list"
expName = hf.get_datalist(expDir);
onsetName = 'onset_transients.npy';

# auto...
path = '%sstructures/' % expDir;
dataList = hf.np_smart_load(path + expName)
expName = 'sfBB_core';
unitNm = dataList['unitName'][cellNum-1];
cell = hf.np_smart_load('%s%s_sfBB.npy' % (path, unitNm));
expInfo = cell[expName]
byTrial = expInfo['trial'];

# we're enforcing expInd = -1 for now...
expInd = -1;
stimDur = hf.get_exp_params(expInd).stimDur;

msTenthToS = 1e-4; # the spike times are in 1/10th ms, so multiply by 1e-4 to convert to S
nonBlank = np.where(np.logical_or(byTrial['maskOn'], byTrial['maskOn']))[0]
nTrials = len(nonBlank);
# print('%d non-blank trials considered' % nTrials);
allSpikes = np.hstack(expInfo['spikeTimes'][nonBlank] * msTenthToS)

maskInd, baseInd = hf_sf.get_mask_base_inds();

# what is the length (in mS) of one cycle (mask, base separately)
maskTf = np.unique(expInfo['trial']['tf'][maskInd,:])[0]
baseTf = np.unique(expInfo['trial']['tf'][baseInd,:])[0]
cycleDur_mask = 1e3/maskTf # guaranteed only one TF value
cycleDur_base = 1e3/baseTf # guaranteed only one TF value

# X (e.g. 25) ms SLIDING bins (will take longer)
psth_slide, bins_slide = hf.make_psth_slide([allSpikes], binWidth=halfWidth*1e-3);
psth_slide = psth_slide[0];
rate_slide = psth_slide/nTrials;

# Fit onset transient...
full_transient = hf.fit_onset_transient(rate_slide, bins_slide, onsetWidth=onsetDur, toNorm=0)
max_rate = np.max(rate_slide);
onsetInd = int((1e-3*onsetDur/stimDur)*len(psth_slide));
onset_rate, onset_bins = rate_slide[0:onsetInd], bins_slide[0:onsetInd]
# do it again so we have the normalized version
full_transient_norm = hf.fit_onset_transient(rate_slide, bins_slide, onsetWidth=onsetDur, toNorm=1)

### Load what we need to 
curr_key = (onsetDur, halfWidth)
if not os.path.exists(data_loc + onsetName):
  onset_list = dict();
  onset_list[cellNum-1] = dict();
  onset_list[cellNum-1][curr_key] = dict();
else:
  onset_list = hf.np_smart_load(data_loc + onsetName);
  if cellNum-1 not in onset_list:
    onset_list[cellNum-1] = dict();
  curr_key = (onsetDur, halfWidth)
  if curr_key not in onset_list[cellNum-1]: # tuple as key: (onset duration, half width) is the dictionary key
    onset_list[cellNum-1][curr_key] = dict();
# NOTE: We aren't handling overwriting right now (just assuming we'll use the most recent fit...)
### Then put the transient there...
curr_transient = dict();
curr_transient['onsetDur'] = onsetDur; # yes, it's already the key, but w/e...
curr_transient['halfWidth'] = halfWidth; # yes, it's already the key, but w/e...
curr_transient['transient'] = full_transient_norm; # yes, it's already the key, but w/e...

onset_list[cellNum-1][curr_key] = curr_transient;
np.save(data_loc + onsetName, onset_list);
print('Saved onset transient for cell #%d, onset duration of %dms and sliding psth with half-width of %dms' % (cellNum,onsetDur,halfWidth));

if toPlot:

  nrow, ncol = 3, 1
  f, ax = plt.subplots(nrow, ncol, figsize=(ncol*25, nrow*20))

  f.suptitle('Cell #%d -- PSTH for %d trials' % (cellNum, nTrials))

  # 1 ms bins
  psth, bins = hf.make_psth([allSpikes], binWidth=1e-3)
  rate = psth[0]/nTrials;
  # - init params (we'll fit a Gaussian to the first X ms)
  toFit = 100; # in mS

  psth_floor = np.min(rate);
  wherePeak = np.argmax(rate);

  max_rate = np.max(rate)
  ax[0].plot([0, cycleDur_mask], [1.05*max_rate, 1.05*max_rate], 'k-', label='Mask cycle')
  ax[0].plot([0, cycleDur_base], [1.025*max_rate, 1.025*max_rate], 'r-', label='Base cycle')
  ax[0].plot(1e3*bins[0][0:-1], rate); # put times in mS
  ax[0].set_ylim([-0.2*max_rate, 1.2*max_rate])
  ax[0].set_ylabel('Spike rate');
  # ax[0].set_xlabel('Time (ms)');
  ax[0].set_title('Bins (non-overlapping) are 1ms');

  # 10 ms bins
  psth, bins = hf.make_psth([allSpikes], binWidth=1e-2)
  rate = psth[0]/nTrials;
  # - init params
  psth_floor = np.min(rate);
  # wherePeak = np.argmax(rate);
  # init_params = [0.050,0.05,0.011,1.2*psth_floor];
  init_params = [0.50,0.001,0.001,1.2*psth_floor];
  # -- fit PSTH
  max_rate = np.max(rate);
  ax[1].plot([0, cycleDur_mask], [1.05*max_rate, 1.05*max_rate], 'k-', label='Mask cycle')
  ax[1].plot([0, cycleDur_base], [1.025*max_rate, 1.025*max_rate], 'r-', label='Base cycle')
  ax[1].plot(1e3*bins[0][0:-1], rate); # put times in mS
  ax[1].set_ylabel('Spike rate');
  # ax[1].set_ylim([-0.2*max_rate, 1.2*max_rate])
  ax[1].set_xlabel('Time (ms)');
  ax[1].set_title('Bins (non-overlapping) are 10ms');
  # -- show sliding PSTH, onset transient
  max_rate = np.max(rate_slide);

  ax[2].plot([0, cycleDur_mask], [1.05*max_rate, 1.05*max_rate], 'k-', label='Mask cycle')
  ax[2].plot([0, cycleDur_base], [1.025*max_rate, 1.025*max_rate], 'r-', label='Base cycle')
  ax[2].plot(1e3*bins_slide, rate_slide); # put times in mS
  ax[2].plot(1e3*onset_bins, full_transient[0:onsetInd], 'r--')
  ax[2].set_ylabel('Spike rate');
  # ax[1].set_ylim([-0.2*max_rate, 1.2*max_rate])
  ax[2].set_xlabel('Time (ms)');
  ax[2].set_title('Bins are overlapping, +/- %d ms' % halfWidth);

  # Now save
  if not os.path.exists(save_loc):
    os.makedirs(save_loc);
  save_name = 'cell_%03d_onsetDur%d_halfWidth%d.pdf' % (cellNum, onsetDur, halfWidth)
  pdfSv = pltSave.PdfPages(str(save_loc + save_name));
  pdfSv.savefig(f)
  pdfSv.close();

'''

#################
### Now, just the PSTH for one trial
#################

##### pick trial
#whichTrial = 705; # good examples for cell #12: 10, 100, 560 (best), 561, 569
whichTrial = 690; # good examples for cell #4: 705, 274, 862, 
##### pick trial

curr_spikes = expInfo['spikeTimes'][whichTrial-1] * msTenthToS
psth, bins = hf.make_psth_slide([curr_spikes], binWidth=halfWidth*1e-3)
psth = psth[0];
psth_raw, bins_raw = hf.make_psth([curr_spikes], binWidth=1e-3)
psth_raw = psth_raw[0]; bins_raw = bins_raw[0];

### Actually get the FFT coefficients??
input_mat, coeffs, spectrum, amplitudes = hf.manual_fft(psth_raw, tfs=np.array([int(maskTf), int(baseTf)]), onsetTransient=full_transient)
input_mat_noTransient, coeffs_noTransient, spectrum_noTr, amplitudes_noTr = hf.manual_fft(psth_raw, tfs=np.array([int(maskTf), int(baseTf)]), onsetTransient=None)

max_rate = np.max(psth);

onsetInd = int((1e-3*onsetDur/stimDur)*len(psth));
onset_rate, onset_bins = psth[0:onsetInd], bins[0:onsetInd]
onset_toFit = onset_rate - np.min(rate);

# compute amplitudes - first from real FFT, then our manual FFT*
_, tf_amps, amps = hf.spike_fft([psth_raw], tfs=[[int(maskTf), int(baseTf)]], stimDur=1)
ax[3].text(cycleDur_mask*1.2, 1.025*max_rate, 'FFT: DC %.1f, mask %.1f, base %.2f' % (amps[0][0], tf_amps[0][0], tf_amps[0][1]))
ax[3].text(cycleDur_mask*1.2, 1.05*max_rate, 'FFT+: DC %.1f, mask %.1f, base %.2f' % (amplitudes[0], amplitudes[1], amplitudes[2]))

ax[3].plot([0, cycleDur_mask], [1.05*max_rate, 1.05*max_rate], 'k-', label='Mask cycle')
ax[3].plot([0, cycleDur_base], [1.025*max_rate, 1.025*max_rate], 'r-', label='Base cycle')
# plot the fit??
ax[3].plot(1e3*bins[1:], np.matmul(input_mat, coeffs), 'k--', label='FFT+')
ax[3].plot(1e3*bins[1:], np.matmul(input_mat_noTransient, coeffs_noTransient), 'b--', label='FFT')

ax[3].plot(1e3*bins_raw[1:], .85*max_rate*psth_raw, 'ko'); # put times in mS
ax[3].plot(1e3*bins, psth, 'k-'); # put times in mS
ax[3].plot(1e3*onset_bins, onset_rate[0:onsetInd], 'r--', label='transient')
ax[3].set_ylabel('Spike rate');
# ax[1].set_ylim([-0.2*max_rate, 1.2*max_rate])
ax[3].set_xlabel('Time (ms)');
ax[3].set_title('Trial %d, 1 ms bins' % whichTrial);
ax[3].legend();

sns.despine(offset=5)

'''
