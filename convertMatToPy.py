#### This file is a more streamlined process than convertMatToPy.ipynb if you are
#### planning to convert a .mat to .npy file, particularly if spikeTimesGLX was created
#### - that is, if you used mountainsort to get spikes from the neuropixel, this function
####   will take the .mat file, convert to .npy file, AND overwrite spikeCount/Times with the 
####   values from GLX 
#### - note that you still must use converMatToPy.ipynb to create/regenerate the dataList


import numpy as np
import helper_fcns as hf
import makeStimulus
import math, os

loc_matData = 'V1/structures/';
loc_pyData = loc_matData;
moveGLXspikes = 0; # move the spikeTimesGLX info to usual spikeTimes/Count loc

# name files specifically
#matFiles = ['m678p6l18_c45_sfm.mat', 'm678p6l18_c59_sfm.mat', 'm678p6l18_c17_sfm.mat', 'm678p6l18_c69_sfm.mat', 
#            'm678p7r03_c72_sfm.mat', 'm678p7r03_c32_sfm.mat', 'm678p7r03_c36_sfm.mat', 'm678p7r03_c27_sfm.mat', 'm678p7r03_c39_sfm.mat', 'm678p7r03_c69_sfm.mat'];
# or generate the list 
matFiles = None;
if matFiles is None: # then, let's build a list of files that we need
  allFiles = os.listdir(loc_matData);
  matFiles = [];
  for f in allFiles:
    if f.startswith('m681') and f.endswith('_sfm.mat'):
      if os.path.exists(loc_pyData + f.replace('.mat', '.npy')):
        continue;
      else:
        matFiles.append(f);

for f in matFiles:
  matData = makeStimulus.loadmat(loc_matData + f);
  S = matData.get('S'); # the actual data structure

  print("now saving...%s" % f.replace('.mat', '.npy'))
  saveName = loc_pyData + f.replace('.mat', '.npy');
  np.save(saveName, S)

  # if move spikes
  if moveGLXspikes:
    pyDat = hf.np_smart_load(saveName);
    pyDat['sfm']['exp']['trial']['spikeCount'] = pyDat['sfm']['exp']['trial']['spikeTimesGLX']['spikeCount'];
    pyDat['sfm']['exp']['trial']['spikeTimes'] = pyDat['sfm']['exp']['trial']['spikeTimesGLX']['spikeTimes'];
    np.save(saveName, pyDat);
