#!/bin/bash

source activate lcv-python

### GUIDE (as of 19.11.05)
# V1/ - use dataList_glx.npy, was 35 cells -- now 56 (as of m681)
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
# LGN/ - standard, 77 cells

#for run in {1..56}
#do
#  python3.6 plot_superposition.py $run V1/ &
#done

python3.6 plot_superposition.py 2 V1/ 
python3.6 plot_superposition.py 4 V1/ 
python3.6 plot_superposition.py 15 V1/ 
python3.6 plot_superposition.py 16 V1/ 
python3.6 plot_superposition.py 18 V1/ 
python3.6 plot_superposition.py 19 V1/ 
python3.6 plot_superposition.py 22 V1/ 
python3.6 plot_superposition.py 30 V1/ 
python3.6 plot_superposition.py 35 V1/ 
python3.6 plot_superposition.py 36 V1/ 
python3.6 plot_superposition.py 37 V1/ 
python3.6 plot_superposition.py 40 V1/ 
python3.6 plot_superposition.py 51 V1/ 
python3.6 plot_superposition.py 55 V1/ 


# leave a blank line at the end
