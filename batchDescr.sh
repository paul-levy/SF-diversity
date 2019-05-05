#!/bin/bash

### README
# Have you set the dataList name?
# Have you set the phAdv name?
# Have you set the RVC name?#
# Have you set the descriptive fit name?
# Have you set the modelRecovery status/type?
### Go to descr_fits.py first

# arguments are
#   1 - cell #
#   2 - dispersion (index into the list of dispersions for that cell; not used in descr/DoG fits)
#   3 - data directory (e.g. LGN/ or V1/)
#   4 - make phase advance fits
#   5 - make RVC fits
#   6 - make descriptive (DoG) fits (1 or 0)
#   7 - DoG model (flexGauss [0; not DoG] or sach [1] or tony [2])
#   [8 - phase direction (pos or neg)]; default is pos (check...)
#   [9 - regularization for gain term (>0 means penalize for high gain)] default is 0

source activate lcv-python

# NOTE: If running only SF descr fits, do not need to run separately for all disp

for run in {1..17}
do
  python descr_fits.py $run 0 V1/ 0 0 1 0 1 & 
  #python descr_fits.py $run 1 V1/ 0 0 1 0 1 & 
  #python descr_fits.py $run 2 V1/ 0 0 1 0 1 & 
  #python descr_fits.py $run 3 V1/ 0 0 1 0 1 & 
done
