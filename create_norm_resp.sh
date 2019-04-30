#!/bin/bash

# params are: cellNum, expDir, [overwrite flag = 0 (default) or 1]
# see model_responses.py or create_norm_resp.py for additional details

source activate lcv-python

for run in {1..19}
do
  python create_norm_resp.py $run V1/ & 
done
