import numpy as np
import helper_fcns as hf
import sys
import os

import pdb

# Use this script to analyze the spike train for each trial for a given cell
# First, we create the PSTH (from the raw spike times), then we perform the Fourier transform
# to determine the strength of response at each temporal frequency. We'll save the full spectrum and
# stimulus-relevant subset (i.e. the spectrum power at the stimulus component temporal frequenc(y/ies) 
# ASSUMPTIONS: This only works with 1s stimulus duration, integer temporal frequencies

which_cell = int(sys.argv[1]);
binWidth = 1e-3; # in seconds
stimDur = 1; # in seconds

# at CNS
# dataPath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/recordings/';
# savePath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/analysis/';
# personal mac
#dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/analysis/structures/';
#save_loc = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/analysis/figures/';
# prince cluster
dataPath = '/home/pl1465/SF_diversity/LGN/analysis/structures/';
save_loc = '/home/pl1465/SF_diversity/LGN/analysis/figures/';

expName = 'dataList.npy'

dataList = hf.np_smart_load(str(dataPath + expName));
loadName = str(dataPath + dataList['unitName'][which_cell-1] + '_sfm.npy');
cellStruct = hf.np_smart_load(loadName);
data = cellStruct['sfm']['exp']['trial'];

allSpikes = data['spikeTimes'];

# Create PSTH, spectrum for all cells
psth_all, _ = hf.make_psth(allSpikes, binWidth, stimDur);
spect_all, _, _ = hf.spike_fft(psth_all);

# now for valid trials (i.e. non-blanks), compute the stimulus-relevant power
n_stim_comp = np.max(data['num_comps']);
n_tr = len(psth_all);
stim_power = np.array(np.nan*np.zeros(n_tr, ), dtype='O'); # object dtype; more flexible

val_trials = np.where(~np.isnan(data['tf'][0]))[0]; # nan in TF means a blank

all_tfs = np.vstack((data['tf'][0], data['tf'][1], data['tf'][2], data['tf'][3], data['tf'][4]));
all_tfs = np.transpose(all_tfs)[val_trials];
all_tfs = all_tfs.astype(int)
rel_tfs = [[x[0]] if x[0] == x[1] else x for x in all_tfs];

psth_val, _ = hf.make_psth(allSpikes[val_trials], binWidth, stimDur);
_, rel_power, _ = hf.spike_fft(psth_val, rel_tfs);

stim_power[val_trials] = rel_power;

# now organize the data structure
data['power_f1'] = stim_power;
data['fourier'] = dict();
data['fourier']['fs'] = 22050; # sampling rate - make "smarter", i.e. not just magic number
data['fourier']['binWidth'] = 1e-3; # what was bin width before FFT
data['fourier']['spectrum'] = spect_all;
data['fourier']['psth'] = psth_all;

# now save
cellStruct['sfm']['exp']['trial'] = data;
np.save(loadName, cellStruct);
