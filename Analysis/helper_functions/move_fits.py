import numpy as np
import sys

# The goal here is to make the data structures truly static and have all of the "moving parts" (i.e. model fits) be in a separate
# data structure. This makes things easier for passing around...

def organizeFits(data_loc):
   dataList = np.load(data_loc + 'dataList.npy').item();
   fitList = [];
 
   for i in dataList['unitName']:
     print(i)
     curr = np.load(data_loc + i + '_sfm.npy').item();
     currFit = curr['sfm']['mod']['fit'];
     
     fitList.append(currFit);

   np.save(data_loc + 'fitList.npy', fitList);

if __name__ == '__main__':
 
   organizeFits(sys.argv[1]);
     
