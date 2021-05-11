#!/bin/bash

source activate lcv-python

# 1 - cell #
# 2 - rvcModel (should be 0 [tony] or 1 [naka-rushton])
# 3 - nRpts (for DoG fit)
# 4 - loss type (1: poiss; 2: sqrt; 3: sach; 4: varExpl)
# 5 - dogModel (0: flex || 1: sach || 2: tony)
# 6 - joint fitting (0 - no; 1 - yes) //see hf.dog_fit for details

DOG_MOD=${1:=1}
DOG_LOSS=${2:=3}
RVC_MOD=${3:=0}

for run in {1..34}
do
  python3.6 descr_fit.py $run $RVC_MOD 250 $DOG_LOSS $DOG_MOD 0 & 
done
