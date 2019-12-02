#!/bin/bash

source activate lcv-python

# 1 - cell #
# 2 - rvcModel (should be 0)
# 3 - nRpts (for DoG fit)
# 4 - loss type (3: sach)
# 5 - dogModel (1: sach || 2: tony)
# 6 - joint fitting (0 - no; 1 - yes) //see hf.dog_fit for details

for run in {1..34}
do
  python descr_fit.py $run 0 250 3 2 1 &
  #python descr_fit.py $run 0 250 3 1 1 &
done
