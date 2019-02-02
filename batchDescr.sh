#!/bin/sh

# arguments are
#   1 - cell #
#   2 - dispersion (index into the list of dispersions for that cell; not used in descr/DoG fits)
#   3 - data directory (e.g. V1/ or LGN/)
#   4 - make phase advance fits
#   5 - make RVC fits
#   6 - make descriptive (DoG) fits (1 or 0)
#   7 - DoG model (sach or tony)
#   [8 - phase direction (pos or neg)]; default is pos
#   [9 - regularization for gain term (>0 means penalize for high gain)] default is 0

source activate lcv-python

for run in {1..5}
do
  # phAdv fits
  # rvcFits
  # DoG fits
  python descr_fits.py $run -1 V1/ 0 0 1 1 1 & 
done
