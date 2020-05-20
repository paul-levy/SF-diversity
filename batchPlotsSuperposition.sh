#!/bin/bash

source activate lcv-python

### GUIDE (as of 19.11.05)
# V1/ - use dataList_glx.npy, was 35 cells -- now 56 (as of m681)
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
# LGN/ - standard, 77 cells

## optional arguments after expDir/
## - use_mod_resp (default is 0; put 1 to plot superposition with model responses)
## - fitType (default is 2 [weighted]; use 1 for flat; etc)
## - excType (1 for gauss deriv [default]; 2 for flex. gauss)

for run in {1..56}
do
  #python3.6 plot_superposition.py $run altExp/ & # with data
  #python3.6 plot_superposition.py $run V1/ & # with data
  #python3.6 plot_superposition.py $run V1/ 1 1 2 & # model, flat, flex. gauss
  #python3.6 plot_superposition.py $run V1/ 1 2 2 & # model, wght, flex. gauss
  python3.6 plot_superposition.py $run V1/ 1 1 1 & # model, flat, gauss deriv.
  python3.6 plot_superposition.py $run V1/ 1 2 1 & # model, wght, gauss deriv.
done

#python3.6 plot_superposition.py 5 V1/ 
#python3.6 plot_superposition.py 7 V1/ 
#python3.6 plot_superposition.py 23 V1/ 
#python3.6 plot_superposition.py 33 V1/ 
#python3.6 plot_superposition.py 41 V1/ 
#python3.6 plot_superposition.py 42 V1/ 
#python3.6 plot_superposition.py 46 V1/ 
#python3.6 plot_superposition.py 47 V1/ 
#python3.6 plot_superposition.py 52 V1/ 
#python3.6 plot_superposition.py 54 V1/ 

# leave a blank line at the end
