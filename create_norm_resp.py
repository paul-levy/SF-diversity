import numpy as np
import sys
import model_responses

def comp_norm_resp(cellInd, data_loc, expInd):
  # subtract one, since cell 1 is at location 0, etc, etc (zero-index)...
   model_responses.GetNormResp(cellInd-1, data_loc, expInd=expInd);

if __name__ == '__main__':

    # on LCV machines
    dataPath = '/users/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/';
    # at CNS
    #dataPath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/';
    # personal mac
    #dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/';
    # on cluster
    #dataPath = '/home/pl1465/SF_diversity/';

    if len(sys.argv) < 3:
      print('uhoh...you two arguments here');
      print('Should be cell number & index for the experiment...');
      exit();

    cellNum = int(sys.argv[1]);
    expInd  = int(sys.argv[2]);
    print('Running cell ' + str(cellNum) + '...');

    comp_norm_resp(cellNum, dataPath, expInd)
