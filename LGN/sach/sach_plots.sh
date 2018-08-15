#!/bin/bash
# Proper header for a Bash script

for i in {1..32}; do
    echo "making plots for cell $i"
    # python plotting.py $i
    python descr_fit.py $i 2
done