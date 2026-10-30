#!/bin/bash
# CVQ baseline on the GMM channels, via cvq_proj.py. Each "Tu TuM" pair is one feedback budget
# (Tu direction bits + TuM magnitude bits = 8, 12, 12, 16, 20 bits/user); N=32 skips the 14-bit
# direction codebook.

N_list=(4 8 32)
K=(1 2 4)
SNR=(-10 -5 -2.5 0 2.5 5 10 15)
pairs=("5 3" "7 5" "8 4" "11 5" "14 6")   # "Tu TuM"
TRAIN_SAMPLES=200000

cd "$(dirname "$0")"
source ../experiment_grid.sh   # naming conventions + parameter checks

for N in "${N_list[@]}"; do
    read -r -a UT <<< "$(grid_allowed_uts "$N")"
    for pair in "${pairs[@]}"; do
        read -r Tu MB <<< "$pair"
        if [ "$N" -eq 32 ] && [ "$Tu" -eq 14 ]; then
            continue
        fi
        python3 cvq_proj.py \
            --num-codebooks-to-test 3 \
            --num-bs-ant "$N" \
            --num-uts "${UT[@]}" \
            --num-gmm-components "${K[@]}" \
            --Td "$N" \
            --num-feedback-bits "$Tu" \
            --num-magnitude-bits "$MB" \
            --max-wmmse-iter 20 \
            --channel-inversion-type lmmse \
            --snr-db "${SNR[@]}" \
            --scenario 'gmm' \
            --num-kmeans-train-samples $TRAIN_SAMPLES
    done
done
