#!/bin/bash

# params are: cellNum, expDir, datalistname, [overwrite flag = 0 (default) or 1]
# see model_responses.py or create_norm_resp.py for additional details

source activate lcv-python

for run in {46..46}
do
  python create_norm_resp.py $run V1_orig/ dataList.npy & 
done
