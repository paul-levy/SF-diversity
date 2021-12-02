#!/bin/bash

source activate lcv-python

# 1 - cell #
# 2 - rvcModel (should be 0 [tony] or 1 [naka-rushton])
# 3 - nRpts (for DoG fit)
# 4 - loss type (1: lsq; 2: sqrt; 3: poiss; 4: sach)
# 5 - dogModel (0: flex || 1: sach || 2: tony)
# 6 - joint fitting (0 - no; 1 - yes) //see hf.dog_fit for details

DOG_MOD=${1:-1}
DOG_LOSS=${2:-3}
RVC_MOD=${3:-0}
BOOT_ITER=${4:-0}
JOINT=${5:-0}

for run in {1..11}
do
  python3.6 descr_fit.py $run $RVC_MOD 25 $DOG_LOSS $DOG_MOD $JOINT $BOOT_ITER & 
done
wait
for run in {12..22}
do
  python3.6 descr_fit.py $run $RVC_MOD 25 $DOG_LOSS $DOG_MOD $JOINT $BOOT_ITER & 
done
wait
for run in {23..34}
do
  python3.6 descr_fit.py $run $RVC_MOD 25 $DOG_LOSS $DOG_MOD $JOINT $BOOT_ITER & 
done
