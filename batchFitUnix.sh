#!/bin/bash

# params are: cellNum, expDir, lossType, fitType, initFromCurr
# see model_responses.py for additional details

source activate lcv-python

for run in {1..34}; do
  python model_responses.py $run LGN/ 4 1 1 &
  python model_responses.py $run LGN/ 4 2 1 &
done
