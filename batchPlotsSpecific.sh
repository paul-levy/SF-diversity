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
# third param is expDir (e.g. V1/ or LGN/)
# fourth param is f0/f1 (i.e. load rvcFits?)
# fifth param is diffPlot (i.e. plot everything relative to flat model prediction)
# sixth param is interpModel (i.e. interpolate model?)
# seventh param is std/sem as variance measure: (1 sem (default))

source activate lcv-python

######
## V1_orig
######

### for disp
python plot_compare.py 10 4 V1_orig/ 0 0 1 1 & # interpolated
python plot_compare.py 10 4 V1_orig/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 41 4 V1_orig/ 0 0 1 1 & # interpolated
#python plot_compare.py 41 4 V1_orig/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 59 4 V1_orig/ 0 0 1 1 & # interpolated
#python plot_compare.py 59 4 V1_orig/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 34 4 V1_orig/ 0 0 1 1 & # interpolated
#python plot_compare.py 34 4 V1_orig/ 0 1 1 1 & # diff plots, interpolated
python plot_compare.py 28 4 V1_orig/ 0 0 1 1 & # interpolated
python plot_compare.py 28 4 V1_orig/ 0 1 1 1 & # diff plots, interpolated

### general
#python plot_compare.py 13 4 V1_orig/ 0 0 1 1 & # interpolated
#python plot_compare.py 13 4 V1_orig/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 17 4 V1_orig/ 0 0 1 1 & # interpolated
#python plot_compare.py 17 4 V1_orig/ 0 1 1 1 & # diff plots, interpolated


######
## altExp
######

### for disp
python plot_compare.py 4 4 altExp/ 0 0 1 1 & # interpolated
python plot_compare.py 4 4 altExp/ 0 1 1 1 & # diff plots, interpolated

### general
#python plot_compare.py 8 4 altExp/ 0 0 1 1 & # interpolated
#python plot_compare.py 8 4 altExp/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 6 4 altExp/ 0 0 1 1 & # interpolated
#python plot_compare.py 6 4 altExp/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 1 4 altExp/ 0 0 1 1 & # interpolated
#python plot_compare.py 1 4 altExp/ 0 1 1 1 & # diff plots, interpolated

######
## V1
######

### for disp

#python plot_compare.py 1 4 V1/ 0 0 1 1 & # interpolated
#python plot_compare.py 1 4 V1/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 8 4 V1/ 0 0 1 1 & # interpolated
#python plot_compare.py 8 4 V1/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 5 4 V1/ 0 0 1 1 & # interpolated
#python plot_compare.py 5 4 V1/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 7 4 V1/ 0 0 1 1 & # interpolated
#python plot_compare.py 7 4 V1/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 17 4 V1/ 0 0 1 1 & # interpolated
#python plot_compare.py 17 4 V1/ 0 1 1 1 & # diff plots, interpolated

#python plot_compare.py 11 4 V1/ 0 0 1 1 & # interpolated
#python plot_compare.py 11 4 V1/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 3 4 V1/ 0 0 1 1 & # interpolated
#python plot_compare.py 3 4 V1/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 17 4 V1/ 0 0 1 1 & # interpolated
#python plot_compare.py 17 4 V1/ 0 1 1 1 & # diff plots, interpolated

### general

#python plot_compare.py 9 4 V1/ 0 0 1 1 & # interpolated
#python plot_compare.py 9 4 V1/ 0 1 1 1 & # diff plots, interpolated
#python plot_compare.py 2 4 V1/ 0 0 1 1 & # interpolated
#python plot_compare.py 2 4 V1/ 0 1 1 1 & # diff plots, interpolated


######
## other cases
######


#python plot_compare.py $run 4 V1_orig/ 0 0 0 1 & # original (simple)
#python plot_compare.py $run 4 V1_orig/ 0 1 0 1 & # diff plots

#python plot_compare.py $run 4 V1/ 0 0 1 1 & # interpolated
#python plot_compare.py $run 4 V1/ 0 1 1 1 & # diff plots, interpolated

# leave a blank line at the end

