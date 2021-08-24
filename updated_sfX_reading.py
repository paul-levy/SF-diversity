import os, sys, glob
import helper_fcns as hf
import numpy as np
import pandas as pd

import pdb

exp_dir = sys.argv[1];

dl_name = 'dataList_210721.npy'

test_files = glob.glob('%s/structures/m*_sf*.npy' % exp_dir)

unitNames = [];
expTypes = [];
unitArea = [];

for t in sorted(test_files):
    
    curr_file = hf.np_smart_load(t);
    
    if 'sfm' in t:
       unitNames.append(curr_file['unitLabel'])
    else:
       unitNames.append(curr_file['sfBB_core']['unitLabel'])
    if 'sfm' in t:
      expName = curr_file['sfm']['exp']['filename'];
    elif 'sfBB' in t:
      expName = curr_file['sfBB_core']['filename'];
    expTypes.append(hf.parse_exp_name(expName)[-2]);
    unitArea.append('%s' % exp_dir.split('_')[0])
    
dataList = dict();
dataList['unitName'] = unitNames;
dataList['expType'] = expTypes
dataList['unitArea'] = unitArea;

print('%d cells' % len(unitNames))

np.save('%s/structures/%s' % (exp_dir, dl_name), dataList)
