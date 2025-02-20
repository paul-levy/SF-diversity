#!/bin/bash

### README
### WARN: have you set the fitList base name?

# NOTE: assumes normType = 1 (flat) and normType = 2 (gauss) are present
# -- will choose normType = 1 for LGN, =2 for non-LGN front-end

# 2nd param is excType: 1 (gaussian deriv); 2 (flex. gauss)
# 3rd param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# 4 param is expDir (e.g. altExp/ or LGN/)
# 5 param is lgnFrontEnd (choose LGN type; will be comparing against non-LGN type)
# 6 param is diffPlot (i.e. plot everything relative to flat model prediction)
# 7 param is interpModel (i.e. interpolate model?)
# 8th param is kMult (0.01, 0.05, 0.10, usually...)
# 9th param is whether (1) or not (0) to do vector correction F1
# 10th param is whether to include the onset transient correction for F1 responses (use onsetDur in mS to use (e.g. 100); 0 to do without)
# 11th param is respExpFixed (-1 for not fixed, then specific value for a fit with fixed respExp [e.g. 1 or 2])
# 12th param is std/sem as variance measure: (1 sem (default))

source activate pytorch-lcv

EXC_TYPE=${1:-1}
WHICH_PLOT=${2:-1}
KFOLD=${3:--1} # if 2048, then we make difference plots (why multiplexing kfold? don't want to adjust actual input arguments)
INTP=${4:-0} # smooth curves from evaluating model at intermediate values
DATA_ONLY=${5:-0}
LOSS=${6:-1}
HPC=${7:-1}
VEC_F1=${8:-1}

if [[ $WHICH_PLOT -eq 1 ]]; then
  PYCALL="plot_sfBB.py"
elif [[ $WHICH_PLOT -eq -1 ]]; then
  PYCALL="plot_sfBB_vertical.py"
elif [[ $WHICH_PLOT -eq -2 ]]; then
  PYCALL="plot_sfBB_vertical_sep.py"
else
  PYCALL="plot_sfBB_sep.py"
fi

CORES=$(($(getconf _NPROCESSORS_ONLN)-4))

for run in {1..47} # was ..58 before cutting dataList_210721
do
  ######
  ## New version, model fits - the doubled (i.e. two-digit) inputs are, in order, normTypes, (lgn)conTypes, lgnFrontEnd
  ######
  # ------------------------e-------l------dir--nrm---lgn-dif-kmul--onsr--sem-----
  # -----------------------------------------------con---inp----cor-rExp-------
  # modA: flat, fixed RVC, lgn A; modB: wght, fixed RVC, lgnA
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 12 44 11 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD & # no diff, not interpolated
  # modA: flat, fixed RVC, lgn A; modB: wght, standard RVC, lgnA
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 12 41 11 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD & # no diff, not interpolated
  # modA: flat, standard RVC, lgn A; modB: wght, standard RVC, lgnA
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 12 11 11 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD & # no diff, not interpolated
  # pytorch mod; modA: wght, fixed RVC, lgn A; modB: wght, standard RVC, lgnA
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 22 41 11 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD & # no diff, not interpolated
  if [[ $DATA_ONLY -eq 1 ]]; then
      if [[ $WHICH_PLOT -eq 1 ]]; then
	  python3.6 $PYCALL $run -1 -1 V1_BB/ -1 -1 -1 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD & # no diff, not interpolated
      elif [[ $WHICH_PLOT -eq -1 ]]; then
	  # need to make sure that we don't force DC or F1 (given by -1 after VEC_F1)
	  # --- change that -1 to 0 or 1 to force DC/F1, respectively
	  python3.6 $PYCALL $run -1 -1 V1_BB/ -1 -1 -1 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD & # no diff, not interpolated
      fi
  else
    ### 23.01.29 plots
    #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 12 11 00 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD 01 & # no diff, not interpolated
    #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 27 11 00 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD 11 & # no diff, not interpolated
    #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 21 11 04 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD 10 & # no diff, not interpolated
    python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 11 11 04 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD 11 & # no diff, not interpolated
    python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 12 11 44 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD 11 & # no diff, not interpolated
    python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 17 11 44 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD 11 & # no diff, not interpolated
    python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 77 11 04 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD 11 & # no diff, not interpolated
    #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 22 11 00 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD 01 & # no diff, not interpolated

    # ---------- less used (incl. some that already appear above)
    # modA: flat, no LGN; modB: wght, no LGN
    #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 12 11 00 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD & # no diff, not interpolated
    # modA: flat, LGN; modB: wght, LGN
    #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 12 11 11 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD & # no diff, not interpolated
    # modA: flat, LGNsi; modB: wght, LGN si
    #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 12 11 44 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD & # no diff, not interpolated
    # modA: wght, no LGN; modB: flat, LGN si
    #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 21 11 04 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD & # no diff, not interpolated
    # modA: flat, LGN; modB: flat, LGN si
    #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 11 11 14 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD & # no diff, not interpolated
  fi
  # modA: flat, LGN; modB: flat, LGN yk
  #python3.6 $PYCALL $run $EXC_TYPE $LOSS V1_BB/ 11 11 13 0 $INTP 0.05 $VEC_F1 -1 -1 1 $HPC $KFOLD & # no diff, not interpolated

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
