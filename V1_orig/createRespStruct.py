import numpy as np

def createRespStruct(data_loc):
  dataList = np.load(data_loc + 'dataList.npy').item();

  nCells = len(dataList['unitName']);
  nFam = 5; # definition, from the experiment
  nCon = 2; # el mismo
  nSfs = 11; # same as above; from 11 spatial frequency centers per condition

  allResp = np.zeros((nCells, nFam, nCon, nSfs));

  for c in range(nCells):
    data = np.load(data_loc + dataList['unitName'][c] + '_sfm.npy').item();

    for disp in range(nFam):
      for con in range(nCon):
      # store the data as mean subtracted
        allResp[c, disp, con, :] = data['sfm']['exp']['sfRateMean'][disp][con] - data['sfm']['exp']['sponRateMean'];

  np.save(data_loc + 'respAboveBase.npy', allResp);

if __name__ == '__main__':

  data_loc = '/home/pl1465/SF_diversity/Analysis/Structures/';

  createRespStruct(data_loc);

