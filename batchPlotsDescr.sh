#!/bin/bash

### README
# have you set the dataList name?
# have you set the fitList base name?
# have you set the descrFits base name?
### see plot_simple.py for changes/details

# second param is expDir (e.g. V1/ or LGN/)
# third param is descrMod: (descr)
	# 0 - flex
	# 1 - sach
	# 2 - tony
# fourth param is loss type (descr)
        # 1 - lsq
        # 2 - sqrt
        # 3 - poiss
        # 4 - sach
# fifth param is joint (descr) - 0/1
# sixth param is f0/f1 (i.e. if 1, load rvcFits; if -1, load vec-adjusted rvcFits, rather than phase-corrected responses)
# seventh param is which rvcModel to get/plot (0: movshon; 1: naka-rushton; 2: peirce)
# eigth param is std/sem as variance measure: (1 sem (default))
# ninth param is force log Y for byDisp/allCon and CRF/allSfs: (0/no (default))

source activate lcv-python

### TODO: Fix here and in helper_fcns (and thus in all plotting...) to access correct rvcFits
###       ALSO must make rvcFits (or get_spikes - make the choice) access F0/F1 depending on 

### GUIDE (as of 19.11.05)
# V1/ - use dataList_glx.npy, was 35 cells -- now 56 (as of m681)
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
# LGN/ - standard, 81 cells

EXP_DIR=$1
DOG_MOD=$2
LOSS_TYPE=${3:-2}
JOINT=${4:-0}
HPC=${5:-0}

# note: dog_mod=0 means flex; 1 means sach (DoG)

if [ "$EXP_DIR" = "V1/" ]; then
  for run in {1..30}
  do 
    python3.6 plot_descr.py $run V1/ $DOG_MOD $LOSS_TYPE $JOINT -1 1 1 $HPC &
  done
  wait
  for run in {31..60}
  do 
    python3.6 plot_descr.py $run V1/ $DOG_MOD $LOSS_TYPE $JOINT -1 1 1 $HPC &
  done
  wait
  for run in {61..81}
  do 
    python3.6 plot_descr.py $run V1/ $DOG_MOD $LOSS_TYPE $JOINT -1 1 1 $HPC &
  done
  wait
fi
if [ "$EXP_DIR" = "LGN/" ]; then
  for run in {1..25}
  do 
    python3.6 plot_descr.py $run LGN/ $DOG_MOD $LOSS_TYPE $JOINT 1 0 1 $HPC 0 &
  done
  wait
  for run in {26..50}
  do 
    python3.6 plot_descr.py $run LGN/ $DOG_MOD $LOSS_TYPE $JOINT 1 0 1 $HPC 0 &
  done
  wait
  for run in {50..81}
  do 
    python3.6 plot_descr.py $run LGN/ $DOG_MOD $LOSS_TYPE $JOINT 1 0 1 $HPC 0 &
  done
fi
if [ "$EXP_DIR" = "V1_orig/" ]; then
  for run in {1..20}
  do 
    python3.6 plot_descr.py $run V1_orig/ $DOG_MOD $LOSS_TYPE $JOINT 0 1 1 $HPC &
  done
  wait
  for run in {21..40}
  do 
    python3.6 plot_descr.py $run V1_orig/ $DOG_MOD $LOSS_TYPE $JOINT 0 1 1 $HPC &
  done
  wait
  for run in {41..59}
  do 
    python3.6 plot_descr.py $run V1_orig/ $DOG_MOD $LOSS_TYPE $JOINT 0 1 1 $HPC &
  done
 fi
if [ "$EXP_DIR" = "altExp/" ]; then
  for run in {1..8}
  do 
    python3.6 plot_descr.py $run altExp/ $DOG_MOD $LOSS_TYPE $JOINT 0 1 1 $HPC &
  done
fi

# leave a blank line at the end
