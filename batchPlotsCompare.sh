#!/bin/bash

### README
# have you set the dataList name?
# have you set the fitList base name?
# have you set the directory (below)
### see plot_simple.py for changes/details

# second param is loss_type:
	# 1 - square root
	# 2 - poisson
	# 3 - modulated poission
	# 4 - chi squared
# third param is expDir (e.g. altExp/ or LGN/)
# fourth param is f0/f1 (i.e. load rvcFits?)
# fifth param is rvcMod (0/1/2 - see hf.rvc_fit_name)
# sixth param is diffPlot (i.e. plot everything relative to flat model prediction)
# 7th param is interpModel (i.e. interpolate model?)
# 8th param is std/sem as variance measure: (1 sem (default))

source activate lcv-python

for run in {1..56}
do
  #python plot_compare.py $run 4 altExp/ 0 0 0 1 & # original (simple)
  #python plot_compare.py $run 4 V1_orig/ 0 1 0 1 & # diff plots

  python plot_compare.py $run 4 V1/ 1 1 0 0 1 & # no diff, not interpolated
  #python plot_compare.py $run 4 V1/ 1 0 2 1 1 & # interpolated
  #python plot_compare.py $run 4 V1/ 1 1 2 1 1 & # diff plots, interpolated
done

# leave a blank line at the end

