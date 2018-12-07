import numpy as np
import sys
import model_responses

def comp_norm_resp(cellInd, data_loc):
  # subtract one, since cell 1 is at location 0, etc, etc (zero-index)...
   model_responses.GetNormResp(cellInd-1, data_loc);

if __name__ == '__main__':

    # at CNS
    # dataPath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/recordings/';
    # personal mac
    dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/LGN/analysis/structures/';
    # on cluster
    #dataPath = '/home/pl1465/SF_diversity/altExp/analysis/structures/';

    if len(sys.argv) < 2:
      print('uhoh...you one argument here'); # and one is the script itself...
      print('Should be cell number...');
      exit();

    print('Running cell ' + sys.argv[1] + '...');

    comp_norm_resp(int(sys.argv[1]), dataPath)
