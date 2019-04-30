import numpy as np
import helper_fcns as hf
import makeStimulus
import math, os

loc_matData = 'V1/structures/';
loc_pyData = loc_matData;

matFiles = ['m676l01_glx162_sfm.mat', 'm676l01_glx182_sfm.mat'];
moveGLXspikes = 1; # move the spikeTimesGLX info to usual spikeTimes/Count loc

for f in matFiles:
  matData = makeStimulus.loadmat(loc_matData + f);
  S = matData.get('S'); # the actual data structure

  print("now saving...")
  saveName = loc_pyData + f.replace('.mat', '.npy');
  np.save(saveName, S)

  # if move spikes
  pyDat = hf.np_smart_load(saveName);
  pyDat['sfm']['exp']['trial']['spikeCount'] = pyDat['sfm']['exp']['trial']['spikeTimesGLX']['spikeCount'];
  pyDat['sfm']['exp']['trial']['spikeTimes'] = pyDat['sfm']['exp']['trial']['spikeTimesGLX']['spikeTimes'];
  np.save(saveName, pyDat);
