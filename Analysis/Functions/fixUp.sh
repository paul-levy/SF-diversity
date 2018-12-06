#!/bin/bash

# params are: cellNum, stopThresh (NOT CURRENTLY USED), learning rate (NOT CURRENTLY USED), lossType, fitType, subset_frac (NOT CURRENTLY USED), initFromCurr
# see batchFit.s for help

# from tony on 12.03.18 a.m.: "In the meantime, can you work on some of the worst misfit cases? Specifically #1 and 31 for both, # 10, 28, 38, 41, 48, 58 for flat, and 50 for tuned? These are mostly good data, they should be fit"

/e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py 1 1e-5 0.1 4 1 1 0 &
/e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py 1 1e-5 0.1 4 2 1 0 &

/e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py 31 1e-5 0.1 4 1 1 0 &
/e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py 31 1e-5 0.1 4 2 1 0 &

/e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py 10 1e-5 0.1 4 1 1 0 &
/e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py 28 1e-5 0.1 4 2 1 0 &
/e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py 38 1e-5 0.1 4 1 1 0 &
/e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py 41 1e-5 0.1 4 2 1 0 &
/e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py 48 1e-5 0.1 4 1 1 0 &
/e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py 58 1e-5 0.1 4 2 1 0 &

/e/2.3/p3/wangzhuo/anaconda3/bin/python3 model_responses.py 50 1e-5 0.1 4 2 1 0 &

