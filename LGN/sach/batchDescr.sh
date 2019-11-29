#!/bin/bash


source activate lcv-python

# 1 - cell #
# 2 - rvcModel (should be 0)
# 3 - nRpts (for DoG fit)
# 4 - loss type (3: sach)
# 5 - dogModel (1: sach || 2: tony)

for run in {1..34}
do
  python descr_fit.py $run 0 100 3 2 &
  #python descr_fit.py $run 0 100 3 1 &
done
