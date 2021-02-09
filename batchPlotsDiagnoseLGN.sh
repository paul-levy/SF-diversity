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
EXP_DIR="altExp/" 
LOSS=2

for run in {1..8}
do
  
  # jointLGN plots
  #python3.6 plot_diagnose_vLGN.py $run 2 1 V1/ 99 1 1 0 0 0.05 -1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 1 V1_orig/ 99 1 1 0 0 0.05 -1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 1 altExp/ 99 1 1 0 0 0.05 -1 1 & # no diff, not interpolated

  # pytorch mod, loss; modA: flat, fixed RVC, lgn A; modB: wght, fixed RVC, lgnA
  #python3.6 plot_diagnose_vLGN.py $run 2 $LOSS $EXP_DIR 12 22 11 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated
  # pytorch mod, poiss loss; modA: flat, fixed RVC, lgn A; modB: wght, standard RVC, lgnA
  #python3.6 plot_diagnose_vLGN.py $run 2 $LOSS $EXP_DIR 12 21 11 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated
  # pytorch mod, poiss loss; modA: flat, standard RVC, lgn A; modB: wght, standard RVC, lgnA
  #python3.6 plot_diagnose_vLGN.py $run 2 $LOSS $EXP_DIR 12 11 11 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated
  # pytorch mod, poiss loss; modA: wght, fixed RVC, lgn A; modB: wght, standard RVC, lgnA
  python3.6 plot_diagnose_vLGN.py $run 2 $LOSS $EXP_DIR 22 21 11 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated

  # LGN type 1, pytorch model, poiss loss
  #python3.6 plot_diagnose_vLGN.py $run 2 2 V1/ 1 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 2 V1_orig/ 1 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 2 altExp/ 1 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated

  # LGN type 1, pytorch model, sqrt loss
  #python3.6 plot_diagnose_vLGN.py $run 2 1 V1/ 1 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 1 V1_orig/ 1 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 1 altExp/ 1 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated

  # LGN type 1, pytorch model, modPoiss loss
  #python3.6 plot_diagnose_vLGN.py $run 2 3 V1/ 1 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 3 V1_orig/ 1 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 3 altExp/ 1 -1 1 0 0 0.05 -1 1 1 & # no diff, not interpolated

done

# leave a blank line at the end

