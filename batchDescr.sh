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
#   4 - make phase advance fits (yes [1], no [0], vec correction for F1 [-1])
#   5 - make RVC fits
#   6 - make RVC f0-only fits
#   7 - which RVC model? (see hf::rvc_fit_name)
#       0 - Movshon/Kiorpes (only for LGN)
#       1 - Naka-Rushton 
#       2 - Peirce-adjustment of Naka-Rushton (can super-saturate)
#   8 - make descriptive (DoG) fits (1 or 0)
#   9 - DoG model (flexGauss [0; not DoG] or sach [1] or tony [2])
#   10 - loss type (for DoG fit); 
#       1 - lsq
#       2 - sqrt
#       3 - poiss [was previously default]
#       4 - Sach sum{[(exp-obs)^2]/[k+sigma^2]} where
#           k := 0.01*max(obs); sigma := measured variance of the response
#   11 - joint fitting (0 - no; 1 - yes) //see hf.dog_fit for details
#   [12 - phase direction (pos or neg)]; default is pos (1); neg (-1); or NEITHER (0)
#   [13 - regularization for gain term (>0 means penalize for high gain)] default is 0

### GUIDE (as of 19.11.05)
# V1/ - use dataList_glx.npy, was 35 cells -- now 56 (as of m681)
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
# LGN/ - standard, 77 cells

source activate lcv-python

########
### NOTES: 
###   If running only SF descr or RVC-f0 fits, do not need to run separately for all disp
###   
########

EXP_DIR=$1
RVC_FIT=$2
DESCR_FIT=$3

if [ "$EXP_DIR" = "V1/" ]; then
  if [[ $RVC_FIT -eq 1 ]]; then
    ## RVCs ONLY with NO phase adjustment (instead, vector correction for F1)
    # -- Naka-Rushton
    for run in {1..40}
    do
      python3.6 descr_fits.py $run 0 V1/ -1 1 0 1 0 0 2 0 0 &
      python3.6 descr_fits.py $run 1 V1/ -1 1 0 1 0 0 2 0 0 &
      python3.6 descr_fits.py $run 2 V1/ -1 1 0 1 0 0 2 0 0 &
      python3.6 descr_fits.py $run 3 V1/ -1 1 0 1 0 0 2 0 0 &
      # -- Movshon RVC
      python3.6 descr_fits.py $run 0 V1/ -1 1 0 0 0 0 2 0 0 &
      python3.6 descr_fits.py $run 1 V1/ -1 1 0 0 0 0 2 0 0 &
      python3.6 descr_fits.py $run 2 V1/ -1 1 0 0 0 0 2 0 0 &
      python3.6 descr_fits.py $run 3 V1/ -1 1 0 0 0 0 2 0 0 &
    done
    wait
    for run in {41..81}
    do
      python3.6 descr_fits.py $run 0 V1/ -1 1 0 1 0 0 2 0 0 &
      python3.6 descr_fits.py $run 1 V1/ -1 1 0 1 0 0 2 0 0 &
      python3.6 descr_fits.py $run 2 V1/ -1 1 0 1 0 0 2 0 0 &
      python3.6 descr_fits.py $run 3 V1/ -1 1 0 1 0 0 2 0 0 &
      # -- Movshon RVC
      python3.6 descr_fits.py $run 0 V1/ -1 1 0 0 0 0 2 0 0 &
      python3.6 descr_fits.py $run 1 V1/ -1 1 0 0 0 0 2 0 0 &
      python3.6 descr_fits.py $run 2 V1/ -1 1 0 0 0 0 2 0 0 &
      python3.6 descr_fits.py $run 3 V1/ -1 1 0 0 0 0 2 0 0 &
    done
    wait
  fi
  if [[ $DESCR_FIT -eq 1 ]]; then
    # then, just SF tuning (again, vec corr. for F1, not phase adjustment);
    # -- responses derived from vecF1 corrections, if F1 responses
    for run in {1..40}
    do
      python3.6 descr_fits.py $run 0 V1/ -1 0 0 1 1 0 2 0 0 &# flex gauss
      python3.6 descr_fits.py $run 0 V1/ -1 0 0 1 1 2 2 0 0 &# Tony DoG
      #python3.6 descr_fits.py $run 0 V1/ -1 0 0 1 1 1 2 0 0 &# sach DoG
    done
    wait
    for run in {41..81}
    do
      python3.6 descr_fits.py $run 0 V1/ -1 0 0 1 1 0 2 0 0 &# flex gauss
      python3.6 descr_fits.py $run 0 V1/ -1 0 0 1 1 2 2 0 0 &# Tony DoG
      #python3.6 descr_fits.py $run 0 V1/ -1 0 0 1 1 1 2 0 0 &# sach DoG
    done
    wait
  fi
fi

if [ "$EXP_DIR" = "V1_orig/" ]; then
  if [[ $RVC_FIT -eq 1 ]]; then
    for run in {1..30}
    do
      # V1_orig/ -- rvc_f0 and descr only
      python3.6 descr_fits.py $run 0 V1_orig/ -1 0 1 1 0 0 2 0 &
    done
    wait
    for run in {31..59}
    do
      # V1_orig/ -- rvc_f0 and descr only
      python3.6 descr_fits.py $run 0 V1_orig/ -1 0 1 1 0 0 2 0 &
    done
    wait
  fi
  wait
  if [[ $DESCR_FIT -eq 1 ]]; then
    for run in {1..30}
    do
      # then, just SF tuning (again, vec corr. for F1, not phase adjustment);
      python3.6 descr_fits.py $run 0 V1_orig/ -1 0 0 1 1 0 2 0 &# flex. gauss
      python3.6 descr_fits.py $run 0 V1_orig/ -1 0 0 1 1 2 2 0 &# Tony DoG
      #python3.6 descr_fits.py $run 0 V1_orig/ -1 0 0 1 1 1 2 0 &# sach DoG
    done
    wait
    for run in {31..59}
    do
      # then, just SF tuning (again, vec corr. for F1, not phase adjustment);
      python3.6 descr_fits.py $run 0 V1_orig/ -1 0 0 1 1 0 2 0 &# flex. gauss
      python3.6 descr_fits.py $run 0 V1_orig/ -1 0 0 1 1 2 2 0 &# Tony DoG
      #python3.6 descr_fits.py $run 0 V1_orig/ -1 0 0 1 1 1 2 0 &# sach DoG
    done
    wait
  fi
fi

if [ "$EXP_DIR" = "altExp/" ]; then
  if [[ $RVC_FIT -eq 1 ]]; then
    for run in {1..8}
    do
      # altExp/ -- rvc_f0 and descr only
      python3.6 descr_fits.py $run 0 altExp/ -1 0 1 1 0 0 2 0 &
    done
    wait
  fi
  wait
  if [[ $DESCR_FIT -eq 1 ]]; then
    for run in {1..8}
    do
      # then, just SF tuning (again, vec corr. for F1, not phase adjustment);
      python3.6 descr_fits.py $run 0 altExp/ -1 0 0 1 1 0 2 0 &# flex. gauss
      python3.6 descr_fits.py $run 0 altExp/ -1 0 0 1 1 2 2 0 &# Tony DoG
      #python3.6 descr_fits.py $run 0 altExp/ -1 0 0 1 1 1 2 0 &# sach DoG
    done
    wait
  fi
fi

if [ "$EXP_DIR" = "LGN/" ]; then
  ## LGN - phase adjustment (will be done iff LGN/ 1; not if LGN/ 0 ) and F1 rvc
  if [[ $RVC_FIT -eq 1 ]]; then
    for run in {1..38}
    do
      # phase adj
      python3.6 descr_fits.py $run 0 LGN/ 1 0 0 0 0 0 3 1 &
    done
    wait
    for run in {39..77}
    do
      # phase adj
      python3.6 descr_fits.py $run 0 LGN/ 1 0 0 0 0 0 3 1 &
    done
    wait
  fi
  if [[ $RVC_FIT -eq 1 ]]; then
    for run in {1..38}
    do
      # RVC (movshon)
      python3.6 descr_fits.py $run 0 LGN/ 0 1 0 0 0 0 3 1 &
      python3.6 descr_fits.py $run 1 LGN/ 0 1 0 0 0 0 3 1 &
      python3.6 descr_fits.py $run 2 LGN/ 0 1 0 0 0 0 3 1 &
      python3.6 descr_fits.py $run 3 LGN/ 0 1 0 0 0 0 3 1 &
      # RVC (Naka-Rushton)
      python3.6 descr_fits.py $run 0 LGN/ 0 1 0 1 0 0 3 1 &
      python3.6 descr_fits.py $run 1 LGN/ 0 1 0 1 0 0 3 1 &
      python3.6 descr_fits.py $run 2 LGN/ 0 1 0 1 0 0 3 1 &
      python3.6 descr_fits.py $run 3 LGN/ 0 1 0 1 0 0 3 1 &
    done
    wait
    for run in {39..77}
    do
      # RVC (movshon)
      python3.6 descr_fits.py $run 0 LGN/ 0 1 0 0 0 0 3 1 &
      python3.6 descr_fits.py $run 1 LGN/ 0 1 0 0 0 0 3 1 &
      python3.6 descr_fits.py $run 2 LGN/ 0 1 0 0 0 0 3 1 &
      python3.6 descr_fits.py $run 3 LGN/ 0 1 0 0 0 0 3 1 &
      # RVC (Naka-Rushton)
      python3.6 descr_fits.py $run 0 LGN/ 0 1 0 1 0 0 3 1 &
      python3.6 descr_fits.py $run 1 LGN/ 0 1 0 1 0 0 3 1 &
      python3.6 descr_fits.py $run 2 LGN/ 0 1 0 1 0 0 3 1 &
      python3.6 descr_fits.py $run 3 LGN/ 0 1 0 1 0 0 3 1 &
    done
    wait
  fi
  if [[ $DESCR_FIT -eq 1 ]]; then
    for run in {1..38}
    do
      # Descr fits (based on Movshon RVCs)
      python3.6 descr_fits.py $run 0 LGN/ 0 0 0 0 1 0 2 0 1 &# flex gauss, not joint
      python3.6 descr_fits.py $run 0 LGN/ 0 0 0 0 1 2 2 0 1 &# Tony DoG, not joint (sqrt)
      #python3.6 descr_fits.py $run 0 LGN/ 0 0 0 0 1 1 2 0 1 &# sach DoG, not joint (sqrt)
      #python3.6 descr_fits.py $run 0 LGN/ 0 0 0 0 1 1 4 0 1 &# sach DoG, not joint (sach loss)
    done
    wait
    for run in {39..77}
    do
      # Descr fits (based on Movshon RVCs)
      python3.6 descr_fits.py $run 0 LGN/ 0 0 0 0 1 0 2 0 1 &# flex gauss, not joint
      python3.6 descr_fits.py $run 0 LGN/ 0 0 0 0 1 2 2 0 1 &# Tony DoG, not joint (sqrt)
      #python3.6 descr_fits.py $run 0 LGN/ 0 0 0 0 1 1 2 0 1 &# sach DoG, not joint (sqrt)
      #python3.6 descr_fits.py $run 0 LGN/ 0 0 0 0 1 1 4 0 1 &# sach DoG, not joint (sach loss)
    done
    wait
  fi
fi
