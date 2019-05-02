#!/bin/bash

# params are: cellNum, expDir, lossType, fitType, initFromCurr, trackSteps
# see model_responses.py for additional details

source activate lcv-python

for run in {1..59}; do
  python model_responses.py $run V1_orig/ 4 1 0 1 &
  python model_responses.py $run V1_orig/ 4 2 0 1 &
done
