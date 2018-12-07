#!/bin/bash

# params are cellNum, nRepeats, fromModelSim (0 or 1), fitLossType, [if modSim1: baseStr, normType, lossType]
# fitLossType - lsq (1), sqrt (2), poiss(3)

for run in {1..59}
do
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 descr_fit.py $run 50 0 1 &
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 descr_fit.py $run 50 1 1 fitListSPcns_181130c 1 4 & 
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 descr_fit.py $run 50 1 1 fitListSPcns_181130c 2 4 & 
done
