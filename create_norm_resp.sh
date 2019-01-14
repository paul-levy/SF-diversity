#!/bin/bash

# params are: cellNum, expInd
# see model_responses.py or create_norm_resp.py for additional details

for run in {1..34}
do
  /e/2.3/p3/wangzhuo/anaconda3/bin/python3 create_norm_resp.py $run 3 &
done
