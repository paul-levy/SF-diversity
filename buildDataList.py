import numpy as np
import helper_fcns as hf
import makeStimulus
import math, os

loc_data = 'V1/structures/';
dl_name = 'dataList_glx_full.npy'
recArea = 'V1';

files = os.listdir(loc_data)
files = sorted(files)

# convert individual files
unitName = [];
expType  = [];
unitArea = [];
for i in files:
    # if file has 'sfm' in it and starts with m then 
    if i.find('sfm.npy') >= 0 and i.startswith('m'):

        print('substr: %s' % i[0:i.rfind('_')]);
        _, expName = hf.get_exp_ind(loc_data, i[0:i.rfind('_')])
        unitName.append(i[0:i.rfind('_')]) # go up to the '_' character
        expType.append(expName);
        unitArea.append(recArea);

dl_save = loc_data + dl_name;
if os.path.exists(dl_save):
    dataList = np.load(dl_save).item();
    dataList['unitName'] = unitName;
    dataList['unitArea'] = unitArea;
    dataList['expType'] = expType;
    np.save(dl_save, dataList);
else: # unitType, isolation, comment must be filled in by hand at later time
    dataList = dict();
    dataList['unitName'] = unitName;
    dataList['unitArea'] = unitArea;
    dataList['expType'] = expType;
    dataList['expType'] = expType;
    dataList['isolation'] = [];
    dataList['comment'] = [];
    np.save(dl_save, dataList);
