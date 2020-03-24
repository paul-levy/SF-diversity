#!/bin/bash

# params are: cellNum, expDir, datalistname, [overwrite flag = 0 (default) or 1]
# see model_responses.py or create_norm_resp.py for additional details

source activate lcv-python

#python create_norm_resp.py 6 V1/ dataList_glx.npy & 
#python create_norm_resp.py 7 V1/ dataList_glx.npy & 
#python create_norm_resp.py 9 V1/ dataList_glx.npy & 

for run in {1..56}
do
  python create_norm_resp.py $run V1/ dataList_glx.npy & 
done
