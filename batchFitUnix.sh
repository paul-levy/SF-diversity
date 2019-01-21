#!/bin/bash

# params are: cellNum, expInd, lossType, fitType, initFromCurr
# see model_responses.py for additional details

for run in {1..34}
do
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py $run 3 4 1 1 0 &
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py $run 3 4 2 1 0 &
done
