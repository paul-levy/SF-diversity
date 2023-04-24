#!/bin/bash

### README
# Have you set the dataList name?
# Have you set the phAdv name?
# Have you set the RVC name?
# Have you set the descriptive fit name?
# Have you set the modelRecovery status/type?
#
# e.g. calls
# sh batchDescr_par.sh V1_orig/ 1 1 0 // fit both rvc, sf; no bootstrapping
# sh batchDescr_par.sh V1/ 1 1 250 // fit both rvc, sf; 250 bootstrap reps.
#
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
#   11 - bootstrap fits (0 - no; nBoots[>0] - yes) //see hf.dog_fit for details
#   12 - joint fitting (0 - no; 1 - yes) //see hf.dog_fit for details
#   [13 - phase direction (pos or neg)]; default is pos (1); neg (-1); or NEITHER (0)
#   [14 - modRecov]; default is None; 1 for yes, 0 for no
#   [15 - cross_val]; default is None (i.e. don't do cross-validation); 1 if you want to do so
#   [16 - vol_lam]; default is 0; if doing cross_val, what's the constant on the gain penalty
#   [17 - regularization for gain term (>0 means penalize for high gain)] default is 0

### GUIDE (as of 19.11.05)
# V1/ - use dataList_glx.npy, was 35 cells -- now 56 (as of m681)
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
# LGN/ - standard, 88 cells (as of 21.05.24)

source activate lcv-python

########
### NOTES: 
###   If running only SF descr or RVC-f0 fits, do not need to run separately for all disp
###   
########

EXP_DIR=$1
RVC_FIT=$2
DESCR_FIT=$3
BOOT_REPS=$4
JOINT=${5:-0}
MOD_RECOV=${6:-0}
LOSS=${7:-2}
CROSS_VAL=${8:--1}
PH_ADJ=${9:-1}
DOGMOD=${10:-3} # used only for V1

if [ "$EXP_DIR" = "V1/" ]; then
  #NCELLS=-181
  NCELLS=-156  
  if [[ $PH_ADJ -eq 1 ]]; then
      DIR=1
  else
      DIR=0
  fi
  echo $DIR
  if [[ $RVC_FIT -eq 1 ]]; then
    # -- Naka-Rushton
    python3.6 descr_fits.py $NCELLS 0 V1/ $PH_ADJ 1 0 1 0 0 $LOSS $BOOT_REPS $JOINT $DIR $MOD_RECOV $CROSS_VAL
    python3.6 descr_fits.py $NCELLS 1 V1/ $PH_ADJ 1 0 1 0 0 $LOSS $BOOT_REPS $JOINT $DIR $MOD_RECOV $CROSS_VAL
    python3.6 descr_fits.py $NCELLS 2 V1/ $PH_ADJ 1 0 1 0 0 $LOSS $BOOT_REPS $JOINT $DIR $MOD_RECOV $CROSS_VAL
    python3.6 descr_fits.py $NCELLS 3 V1/ $PH_ADJ 1 0 1 0 0 $LOSS $BOOT_REPS $JOINT $DIR $MOD_RECOV $CROSS_VAL
    # -- Movshon RVC
    python3.6 descr_fits.py $NCELLS 0 V1/ $PH_ADJ 1 0 0 0 0 $LOSS $BOOT_REPS $JOINT $DIR $MOD_RECOV $CROSS_VAL
    python3.6 descr_fits.py $NCELLS 1 V1/ $PH_ADJ 1 0 0 0 0 $LOSS $BOOT_REPS $JOINT $DIR $MOD_RECOV $CROSS_VAL
    python3.6 descr_fits.py $NCELLS 2 V1/ $PH_ADJ 1 0 0 0 0 $LOSS $BOOT_REPS $JOINT $DIR $MOD_RECOV $CROSS_VAL
    python3.6 descr_fits.py $NCELLS 3 V1/ $PH_ADJ 1 0 0 0 0 $LOSS $BOOT_REPS $JOINT $DIR $MOD_RECOV $CROSS_VAL
  fi
  if [[ $DESCR_FIT -eq 1 ]]; then
    python3.6 descr_fits.py $NCELLS 0 V1/ $PH_ADJ 0 0 1 1 $DOGMOD $LOSS $BOOT_REPS $JOINT $DIR $MOD_RECOV $CROSS_VAL # d-DoG-S or Sach
  fi
fi

if [ "$EXP_DIR" = "V1_orig/" ]; then
  if [[ $RVC_FIT -eq 1 ]]; then
    # V1_orig/ -- rvc_f0 and descr only
    python3.6 descr_fits.py -159 0 V1_orig/ 0 0 1 1 0 0 $LOSS $BOOT_REPS $JOINT
  fi
  wait
  if [[ $DESCR_FIT -eq 1 ]]; then
    # then, just SF tuning (again, vec corr. for F1, not phase adjustment);
    python3.6 descr_fits.py -159 0 V1_orig/ 0 0 0 1 1 $DOGMOD $LOSS $BOOT_REPS $JOINT 0 $MOD_RECOV $CROSS_VAL # d-DoG-S or Sach
  fi
fi

if [ "$EXP_DIR" = "altExp/" ]; then
  if [[ $RVC_FIT -eq 1 ]]; then
    # altExp/ -- rvc_f0 and descr only
    python3.6 descr_fits.py -108 0 altExp/ -1 0 1 1 0 0 $LOSS $BOOT_REPS $JOINT
  fi
  wait
  if [[ $DESCR_FIT -eq 1 ]]; then
    python3.6 descr_fits.py -108 0 altExp/ -1 0 0 1 1 $DOGMOD $LOSS $BOOT_REPS $JOINT 0 $MOD_RECOV $CROSS_VAL # d-DoG-S or Sach
  fi
fi

if [ "$EXP_DIR" = "V1_BB/" ]; then
    # first, only RVC, no boot
  if [[ $RVC_FIT -eq 1 ]]; then
    python3.6 descr_fits_sfBB.py -147 $RVC_FIT 0 1 $DOGMOD $LOSS 0 $JOINT 0 # 47 cells, as of 21.08.23
  fi    
  if [[ $DESCR_FIT -eq 1 ]]; then
    # then, no RVC, allow boot for SF
    python3.6 descr_fits_sfBB.py -147 0 $DESCR_FIT 1 $DOGMOD $LOSS $BOOT_REPS $JOINT $CROSS_VAL # 47 cells, as of 21.08.23
  fi    
fi

if [ "$EXP_DIR" = "LGN/" ]; then
  ## LGN - phase adjustment (will be done iff LGN/ 1; not if LGN/ 0 ) and F1 rvc
  if [[ $RVC_FIT -eq 1 ]]; then
    # phase adj
    python3.6 descr_fits.py -181 0 LGN/ $PH_ADJ 0 0 0 0 0 3 $BOOT_REPS $JOINT
    # RVC (movshon)
    python3.6 descr_fits.py -181 0 LGN/ $PH_ADJ 1 0 0 0 0 3 $BOOT_REPS $JOINT
    python3.6 descr_fits.py -181 1 LGN/ $PH_ADJ 1 0 0 0 0 3 $BOOT_REPS $JOINT
    python3.6 descr_fits.py -181 2 LGN/ $PH_ADJ 1 0 0 0 0 3 $BOOT_REPS $JOINT
    python3.6 descr_fits.py -181 3 LGN/ $PH_ADJ 1 0 0 0 0 3 $BOOT_REPS $JOINT
    # RVC (Naka-Rushton)
    python3.6 descr_fits.py -181 0 LGN/ $PH_ADJ 1 0 1 0 0 3 $BOOT_REPS $JOINT
    python3.6 descr_fits.py -181 1 LGN/ $PH_ADJ 1 0 1 0 0 3 $BOOT_REPS $JOINT
    python3.6 descr_fits.py -181 2 LGN/ $PH_ADJ 1 0 1 0 0 3 $BOOT_REPS $JOINT
    python3.6 descr_fits.py -181 3 LGN/ $PH_ADJ 1 0 1 0 0 3 $BOOT_REPS $JOINT
  fi
  wait
  if [[ $DESCR_FIT -eq 1 ]]; then
    # Descr fits (based on Movshon RVCs)
    ### with model recovery
    python3.6 descr_fits.py -181 0 LGN/ $PH_ADJ 0 0 0 1 1 $LOSS $BOOT_REPS $JOINT 1 $MOD_RECOV $CROSS_VAL # sach DoG (sqrt)
    #python3.6 descr_fits.py -181 0 LGN/ 0 0 0 0 1 2 $LOSS $BOOT_REPS $JOINT 1 $MOD_RECOV $CROSS_VAL # tony DoG (sqrt)
  fi
  wait
fi
