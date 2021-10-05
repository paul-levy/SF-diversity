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
#   2 - fit_rvc (yes or no)
#   3 - fit_sf (yes or no)
#   4 - rvcMod (movshon/naka-rushton/pierce)
#   5 - sfMod (flex.gauss/sachDoG/tonyDoG)
#   6 - lossType (lsq/sqrt/poiss/sach)
#   7 - bootIter (0 for no resampling; otherwise, e.g. 100)

source activate pytorch-lcv

FIT_RVC=$1
FIT_SFS=$2
SF_MOD=$3
BOOT_ITER=$4

python3.6 descr_fits_sfBB.py -147 $FIT_RVC $FIT_SFS 1 $SF_MOD 2 $BOOT_ITER # 147, as of 21.08.23
#python3.6 descr_fits_sfBB.py -147 $FIT_RVC $FIT_SFS 1 $SF_MOD 4 $BOOT_ITER # 147, as of 21.08.23; SACH loss
#python3.6 descr_fits_sfBB.py -141 $FIT_RVC $FIT_SFS 1 $SF_MOD 2 $BOOT_ITER # -141 means from cell 1 to cell 41
