import numpy as np
import sys
import model_responses

def comp_norm_resp(cellInd, data_loc, expDir, dlName, overwrite=0):
   model_responses.GetNormResp(cellInd, data_loc, expDir=expDir, dataListName=dlName, overwrite=overwrite);

if __name__ == '__main__':

    # on LCV machines
    dataPath = '/users/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/';
    # at CNS
    #dataPath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/';
    # personal mac
    #dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/';
    # on cluster
    #dataPath = '/home/pl1465/SF_diversity/';

    if len(sys.argv) < 4:
      print('uhoh...you need three arguments here');
      print('Should be cell number & expDir & dataListName for the experiment...');
      exit();

    cellNum = int(sys.argv[1]);
    expDir  = sys.argv[2];
    dlName = sys.argv[3];
    if len(sys.argv) > 4:
      overwrite = int(sys.argv[4]);
    else:
      overwrite = 0;
    print('Running cell ' + str(cellNum) + '...');
    
    comp_norm_resp(cellNum, dataPath, expDir, dlName, overwrite)
