#!/bin/bash

### README
### WARN: have you set the fitList base name?

# 2nd param is excType: 1 (gaussian deriv); 2 (flex. gauss)
# 3rd param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# 4 param is expDir (e.g. altExp/ or LGN/)
# 5 param is normTypes (XY, where X is for 1st model, Y is for 2nd model; e.g. 12 meaning flat, then wght; 22 meaning both wght)
# 6 param is conType (XY; as above, but for lgnConType; e.g. 11 (default; separate RVC for M,P), 12 (separate, then fixed)
# 7 param is lgnFrontEnd (choose LGN type; will be comparing against non-LGN type)
# -------- [1] (m::p f_c is 1::3), [2] (2::3), [9] (joint fit one LGN front end for all cells in the directory) or none [<=0]
# -- As with 5,6 will be XY, where we use X for modA, y for modB
# 8 param is f0/f1 (i.e. load rvcFits?) (OR if -1, do the vector correction for F1 responses; leave DC untouched)
# 9 param is rvcMod (0/1/2 - see hf.rvc_fit_name)
# 10 param is diffPlot (i.e. plot everything relative to flat model prediction)
# 11 param is interpModel (i.e. interpolate model?)
# 12th param is kMult (0.01, 0.05, 0.10, usually...)
# 13th param is respExpFixed (-1 for not fixed, then specific value for a fit with fixed respExp [e.g. 1 or 2])
# 14th param is std/sem as variance measure: (1 sem (default))
# 15th param is whether or not to use pytorch model (0 is default)

source activate pytorch-lcv
#source activate lcv-python

# Used with newest version of calling the plotting function
EXP_DIR=$1
END=$2
RVC_ADJ=${3:-0} # RVC ADJ should be 0 for altExp/V1_orig, 1 for V1/
WHICH_PLOT=${4:-0}
KFOLD=${5:--1}
DIFF_PLOT=${6:--0}
INTP=${7:-0} # smoot hmodel fits by evaluating in-between values?
EXC_TYPE=${8:-1}
DG_NORM_FUNC=${9:-0} # should be double-digit number from {00 [default], 10, 11, 01}
LOSS=${10:-1}
HPC=${11:-1}
START=${12:-1}

CORES=$(($(getconf _NPROCESSORS_ONLN)-4))

if [[ $WHICH_PLOT -eq 0 ]]; then
  PYCALL="plot_diagnose_vLGN.py"
elif [[ $WHICH_PLOT -eq 1 ]]; then
  PYCALL="plot_diagnose_vLGN_tex.py"
fi

for (( run=$START; run<=$END; run++ ))
do
  ### 23.01.29 plots
  # no LGN --> flat, wght
  python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 12 11 00 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD 01 & # no diff, not interpolated
  # no LGN --> wght (log Gauss), wght
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 22 11 00 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD 01 & # no diff, not interpolated
  # no LGN --> wght, wghtMatch
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 27 11 00 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD 11 & # no diff, not interpolated
  # wght, V1, flat LGNsi
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 21 11 04 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD 10 & # no diff, not interpolated
  # LGNsi --> fflat,wght
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 12 11 44 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD 01 & # no diff, not interpolated

  ############## mostly unused below?

  #python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 12 55 44 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD 01 & # no diff, not interpolated
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 11 15 44 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD 00 & # no diff, not interpolated

  # asym vs. wght
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 02 11 00 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD & # no diff, not interpolated


  # asym, LGNsi, wght LGNsi
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 02 11 44 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD 01 & # no diff, not interpolated
  
  # flat, V1, flat LGNsi
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 11 11 04 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD & # no diff, not interpolated
  # w/LGN
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 12 11 11 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD & # no diff, not interpolated
  # w/LGNsi
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 12 11 44 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD & # no diff, not interpolated
  # flat, V1/LGNsi
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS $EXP_DIR 11 11 04 $RVC_ADJ 1 $DIFF_PLOT $INTP 0.05 -1 1 1 $HPC $KFOLD & # no diff, not interpolated

  # Check how many background jobs there are, and if it
  # is equal to the number of cores, wait for anyone to
  # finish before continuing.
  while :; do
      background=( $(jobs -p))
      if (( ${#background[@]} <= $CORES )); then
	  echo 'not waiting'
	  break
      fi
      echo 'waiting'
      sleep 5
  done

done

# leave a blank line at the end

