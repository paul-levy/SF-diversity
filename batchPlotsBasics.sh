#!/bin/bash

source activate lcv-python

### GUIDE (as of 19.11.05)
# V1/ - use dataList_glx.npy, was 35 cells -- now 56 (as of m681)
# V1/ - model recovery (dataList_glx_mr; mr_fitList...), 10 cells
# V1_orig/ - model recovery (dataList_mr; mr_fitList...), 10 cells
# V1_orig/ - standard, 59 cells
# altExp   - standard, 8 cells
# LGN/ - standard, 77 cells

### expDir/ compareSfMix forceSimple

for run in {1..20}
do
  python plot_basics.py $run V1_BB/ 0 0 &
  python plot_basics.py $run V1_BB/ 0 1 &
done

# leave a blank line at the end
