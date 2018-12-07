#!/bin/bash                                                                                                                                                                                     
# params are: cellNum, stopThresh (NOT CURRENTLY USED), learning rate (NOT CURRENTLY USED), lossType, fitType, subset_frac (NOT CURRENTLY USED), initFromCurr
# see batchFit.s for help

for run in {1..8}
do
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py $run 1e-5 0.1 4 1 1 1 &
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py $run 1e-5 0.1 4 2 1 1 &
done
