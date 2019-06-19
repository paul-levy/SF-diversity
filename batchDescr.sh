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
#   6 - make RVC f0-only fits
#   7 - make descriptive (DoG) fits (1 or 0)
#   8 - DoG model (flexGauss [0; not DoG] or sach [1] or tony [2])
#   [9 - phase direction (pos or neg)]; default is pos (check...)
#   [10 - regularization for gain term (>0 means penalize for high gain)] default is 0

### GUIDE (as of 19.05.04)
# V1/ - use dataList_glx.npy, 17 cells
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells

source activate lcv-python

# NOTE: If running only SF descr or RVC-f0 fits, do not need to run separately for all disp

for run in {1..59}
do
  python descr_fits.py $run 0 V1_orig/ 0 0 0 1 1 1 & # F0 DoG
  #python descr_fits.py $run 0 V1_orig/ 0 0 1 1 0 1 & # F0 RVC and DoG
  #python descr_fits.py $run 0 V1_orig/ 0 0 1 0 0 1 & # F0 RVC only
  #python descr_fits.py $run 0 V1/ 0 0 0 1 0 1 & # DoG only
done
