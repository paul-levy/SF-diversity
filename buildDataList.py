import numpy as np
import helper_fcns as hf
import build_basics_list as bl
import sys, os
import glob
import pdb

######################
## NOTE: if you are starting a dataList from scratch, run this whole function (e.g. python buildDataList.py arg1 arg2 ...)
## If you are simply adding something to the list (e.g. basic_list), then it is best to run just pieces of this code manually, since otherwise any manual curation of the datalist 
## will be overwritten!
######################

######################
## Now, build the real data list
######################

def update_data_list(loc_data, dl_name, recArea):

  files = os.listdir(loc_data)
  files = sorted(files)

  # convert individual files
  unitName = [];
  expType  = [];
  unitArea = [];
  for i in files:
      # if file has 'sfm' in it and starts with m then 
      if i.find('sfm.npy') >= 0 and i.startswith('m'):

          #print('substr: %s' % i[0:i.rfind('_')]);
          _, expName = hf.get_exp_ind(loc_data, i[0:i.rfind('_')])
          unitName.append(i[0:i.rfind('_')]) # go up to the '_' character
          expType.append(expName);
          unitArea.append(recArea);

  dl_save = loc_data + dl_name;

  #########
  ### now, let's get the corresponding basic characterization files!
  ###### IF MANUAL EDIT OF DATALIST (just to add basic list) RUN THIS NEXT LINE
  ###### -- and then simply patch the basic_list, basic_order to the loaded datalist, and save (np.save(path, dataList))
  #########
  basic_list, _, _, basic_order = bl.build_basic_lists(unitName, '', loc='V1_orig/', subfolder='recordings/', folderByExpt=False, reduceNums=True);

  #########
  ### finally, let's save it all
  #########
  if os.path.exists(dl_save):
      dataList = np.load(dl_save).item();
      dataList['unitName'] = unitName;
      dataList['unitArea'] = unitArea;
      dataList['expType'] = expType;
      dataList['basicProgName'] = basic_list;
      dataList['basicProgOrder'] = basic_order;
      np.save(dl_save, dataList);
  else: # unitType, isolation, comment must be filled in by hand at later time
      dataList = dict();
      dataList['unitName'] = unitName;
      dataList['unitArea'] = unitArea;
      dataList['expType'] = expType;
      dataList['expType'] = expType;
      dataList['isolation'] = [];
      dataList['comment'] = [];
      dataList['basicProgName'] = basic_list;
      dataList['basicProgOrder'] = basic_order;
      np.save(dl_save, dataList);

if __name__ == "__main__":

  loc_data, dl_name, recArea = sys.argv[1], sys.argv[2], sys.argv[3];
  update_data_list(loc_data, dl_name, recArea)
