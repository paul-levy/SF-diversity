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
#   [12 - phase direction (pos or neg)]; default is pos
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

for run in {1..77}
do
  ## LGN - phase adjustment (if LGN/ 1; not if LGN/ 0 ) and F1 rvc
  #python descr_fits.py $run 0 LGN/ 1 0 0 0 0 0 3 1
  #python descr_fits.py $run 0 LGN/ 0 1 0 0 0 0 3 1 &
  #python descr_fits.py $run 1 LGN/ 0 1 0 0 0 0 3 1 &
  #python descr_fits.py $run 2 LGN/ 0 1 0 0 0 0 3 1 &
  #python descr_fits.py $run 3 LGN/ 0 1 0 0 0 0 3 1 &

  ## LGN - descrFits (DoG)
  python descr_fits.py $run 0 LGN/ 0 0 0 0 1 2 4 1 1 &

  ## V1 - phase adjustment and F1 rvc
  #python descr_fits.py $run 0 altExp/ 1 0 0 1 0 0 3 1 # phAdv fits, only; each cell, in turn (not in background)
  #python descr_fits.py $run 0 altExp/ 0 1 0 2 0 0 3 1 & # without phAdv fits
  #python descr_fits.py $run 1 altExp/ 0 1 0 2 0 0 3 1 &
  #python descr_fits.py $run 2 altExp/ 0 1 0 2 0 0 3 1 &
  #python descr_fits.py $run 3 altExp/ 0 1 0 2 0 0 3 1 & 

  # V1 - descr fits
  #python descr_fits.py $run 0 altExp/ 0 0 0 1 1 1 4 1 &
  #python descr_fits.py $run 0 altExp/ 0 0 0 1 1 0 2 1 &

done
