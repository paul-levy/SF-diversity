#!/bin/bash

# params are: cellNum, expDir, lossType, fitType, initFromCurr
# see model_responses.py for additional details

source activate lcv-python

for run in {1..5}; do
  python model_responses.py $run V1/ 4 1 0 &
  python model_responses.py $run V1/ 4 2 0 &
done
