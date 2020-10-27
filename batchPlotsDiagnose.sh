#!/bin/bash

### README
### WARN: have you set the fitList base name?

# NOTE: assumes normType = 1 (flat) and normType = 2 (gauss) are present

# 2nd param is excType: 1 (gaussian deriv); 2 (flex. gauss)
# 3rd param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# 4 param is expDir (e.g. altExp/ or LGN/)
# 5 param is lgnFrontOn (1 or 0)
# 6 param is f0/f1 (i.e. load rvcFits?)
# 7 param is rvcMod (0/1/2 - see hf.rvc_fit_name)
# 8 param is diffPlot (i.e. plot everything relative to flat model prediction)
# 9 param is interpModel (i.e. interpolate model?)
# 10th param is kMult (0.01, 0.05, 0.10, usually...)
# 11th param is fixRespExp (None is default, pass in negative number for None)
# 12th param is std/sem as variance measure: (1 sem (default))

source activate lcv-python

for run in {1..8}
do
  #python3.6 plot_diagnose.py $run 1 1 V1/ 0 1 1 0 0 0.10 -1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose.py $run 1 1 V1_orig/ 0 1 1 0 0 0.10 -1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose.py $run 1 1 altExp/ 0 1 1 0 0 0.10 -1 1 & # no diff, not interpolated

  python3.6 plot_diagnose.py $run 2 1 V1/ 0 1 1 0 0 0.10 -1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose.py $run 2 1 V1_orig/ 0 1 1 0 0 0.10 -1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose.py $run 2 1 altExp/ 0 1 1 0 0 0.10 -1 1 & # no diff, not interpolated

  #python3.6 plot_diagnose.py $run 2 1 V1/ 1 1 1 0 0 0.10 -1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose.py $run 2 1 V1_orig/ 1 1 1 0 0 0.10 -1 1 & # no diff, not interpolated
  #python3.6 plot_diagnose.py $run 2 1 altExp/ 1 1 1 0 0 0.10 -1 1 & # no diff, not interpolated

  #python3.6 plot_diagnose.py $run 2 1 V1/ 1 1 0 0 1 & # no diff, not interpolated
  #python3.6 plot_diagnose.py $run 2 1 altExp/ 1 1 0 0 1 & # no diff, not interpolated
  #python3.6 plot_diagnose.py $run 2 1 V1_orig/ 1 1 0 0 1 & # no diff, not interpolated

  #python3.6 plot_diagnose.py $run 1 1 V1/ 1 1 0 0 1 & # no diff, not interpolated
  #python3.6 plot_diagnose.py $run 1 1 altExp/ 1 1 0 0 1 & # no diff, not interpolated
  #python3.6 plot_diagnose.py $run 1 1 V1_orig/ 1 1 0 0 1 & # no diff, not interpolated
done

# leave a blank line at the end

