import numpy as np
import helper_fcns as hf
import makeStimulus
import math, os

loc_matData = 'V1/structures/';
loc_pyData = loc_matData;
moveGLXspikes = 1; # move the spikeTimesGLX info to usual spikeTimes/Count loc

# name files specifically
#matFiles = ['m678p5l06_glx174_sfm.mat'];
# or generate the list 
matFiles = None;
if matFiles is None: # then, let's build a list of files that we need
  allFiles = os.listdir(loc_matData);
  matFiles = [];
  for f in allFiles:
    if f.startswith('m') and f.endswith('_sfm.mat'):
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
