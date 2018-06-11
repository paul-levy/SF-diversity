#!/bin/bash
# Proper header for a Bash script

for i in {1..32}; do
    echo "making plots for cell $i"
    python plotting.py $i
done