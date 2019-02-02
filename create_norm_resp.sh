#!/bin/bash

# params are: cellNum, expDir
# see model_responses.py or create_norm_resp.py for additional details

source activate lcv-python

for run in {1..5}
do
  python create_norm_resp.py $run V1/ & 
done
