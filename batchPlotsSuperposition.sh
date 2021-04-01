#!/bin/bash

source activate pytorch-lcv

### GUIDE (as of 19.11.05)
# V1/ - use dataList_glx.npy, was 35 cells -- now 56 (as of m681)
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
# LGN/ - standard, 77 cells

## optional arguments after expDir/
## - use_mod_resp (default is 0; put 1 to plot superposition with model responses; in 2, it uses the pytorch model fits)
## - fitType (default is 2 [weighted]; use 1 for flat; etc)
## - excType (1 for gauss deriv [default]; 2 for flex. gauss)
## - useHPCfit (1 to use it; 0 for not)
## - conType (what lgnConType for the model fit?)
## - lgnFrontEnd (which lgnFrontEnd for the model fit?)

EXP_DIR=$1

for run in {1..8}
do
  #python3.6 plot_superposition.py $run altExp/ & # with data
  #python3.6 plot_superposition.py $run V1/ & # with data
  #python3.6 plot_superposition.py $run LGN/ & # with data
  # DATA
  python3.6 plot_superposition.py $run $EXP_DIR 0 5 2 1 1 1 &
  # Weighted model (5)
  python3.6 plot_superposition.py $run $EXP_DIR 2 5 2 1 1 1 &
  # Flat model
  python3.6 plot_superposition.py $run $EXP_DIR 2 1 2 1 1 1 &

done

#python3.6 plot_superposition.py 1 LGN/
#python3.6 plot_superposition.py 2 LGN/
#python3.6 plot_superposition.py 5 LGN/
#python3.6 plot_superposition.py 6 LGN/
#python3.6 plot_superposition.py 10 LGN/
#python3.6 plot_superposition.py 15 LGN/
#python3.6 plot_superposition.py 17 LGN/
#python3.6 plot_superposition.py 19 LGN/
#python3.6 plot_superposition.py 21 LGN/
#python3.6 plot_superposition.py 24 LGN/
#python3.6 plot_superposition.py 26 LGN/
#python3.6 plot_superposition.py 28 LGN/
#python3.6 plot_superposition.py 30 LGN/
#python3.6 plot_superposition.py 31 LGN/
#python3.6 plot_superposition.py 32 LGN/
#python3.6 plot_superposition.py 41 LGN/
#python3.6 plot_superposition.py 42 LGN/
#python3.6 plot_superposition.py 43 LGN/
#python3.6 plot_superposition.py 45 LGN/
#python3.6 plot_superposition.py 51 LGN/
#python3.6 plot_superposition.py 53 LGN/
#python3.6 plot_superposition.py 54 LGN/
#python3.6 plot_superposition.py 70 LGN/

#python3.6 plot_superposition.py 48 LGN/
#python3.6 plot_superposition.py 53 LGN/
#python3.6 plot_superposition.py 54 LGN/

#python3.6 plot_superposition.py 5 V1/ 
#python3.6 plot_superposition.py 7 V1/ 
#python3.6 plot_superposition.py 11 V1/ 
#python3.6 plot_superposition.py 23 V1/ 
#python3.6 plot_superposition.py 39 V1/ 
#python3.6 plot_superposition.py 42 V1/ 
#python3.6 plot_superposition.py 46 V1/ 
#python3.6 plot_superposition.py 47 V1/ 
#python3.6 plot_superposition.py 52 V1/ 
#python3.6 plot_superposition.py 54 V1/ 

# leave a blank line at the end
