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
# 6 param is f0/f1 (i.e. load rvcFits?)
# 7 param is rvcMod (0/1/2 - see hf.rvc_fit_name)
# 8 param is diffPlot (i.e. plot everything relative to flat model prediction)
# 9 param is interpModel (i.e. interpolate model?)
# 10th param is kMult (0.01, 0.05, 0.10, usually...)
# 11th param is respExpFixed (-1 for not fixed, then specific value for a fit with fixed respExp [e.g. 1 or 2])
# 11th param is std/sem as variance measure: (1 sem (default))

source activate lcv-python

for run in {1..8}
do
  
  # jointLGN plots
  #python3.6 plot_diagnose_vLGN.py $run 2 1 V1/ 99 1 1 0 0 0.05 -1 1 & # no diff, not interpolated
  python3.6 plot_diagnose_vLGN.py $run 2 1 V1_orig/ 99 1 1 0 0 0.05 -1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 1 altExp/ 99 1 1 0 0 0.05 -1 1 & # no diff, not interpolated

  #python3.6 plot_diagnose_vLGN.py $run 2 1 V1/ 99 1 1 0 0 0.05 1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 1 V1_orig/ 99 1 1 0 0 0.05 1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 1 altExp/ 99 1 1 0 0 0.05 1 1 & # no diff, not interpolated

  #python3.6 plot_diagnose_vLGN.py $run 2 1 V1/ 99 1 1 0 0 0.05 2 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 1 V1_orig/ 99 1 1 0 0 0.05 2 1 & # no diff, not interpolated
  #python3.6 plot_diagnose_vLGN.py $run 2 1 altExp/ 99 1 1 0 0 0.05 2 1 & # no diff, not interpolated
  '''

  python3.6 plot_diagnose_vLGN.py $run 2 4 V1/ 1 1 1 0 0 0.05 2 1 & # no diff, not interpolated
  python3.6 plot_diagnose_vLGN.py $run 2 4 V1_orig/ 1 1 1 0 0 0.05 2 1 & # no diff, not interpolated
  python3.6 plot_diagnose_vLGN.py $run 2 4 altExp/ 1 1 1 0 0 0.05 2 1 & # no diff, not interpolated

  python3.6 plot_diagnose_vLGN.py $run 2 4 V1/ 2 1 1 0 0 0.05 2 1 & # no diff, not interpolated
  python3.6 plot_diagnose_vLGN.py $run 2 4 V1_orig/ 2 1 1 0 0 0.05 2 1 & # no diff, not interpolated
  python3.6 plot_diagnose_vLGN.py $run 2 4 altExp/ 2 1 1 0 0 0.05 2 1 & # no diff, not interpolated

  '''

done

# leave a blank line at the end

