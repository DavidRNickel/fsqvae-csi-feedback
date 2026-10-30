#!/bin/bash
# DVQ (rotated DFT codebook) baseline on the GMM channels, via dft.py. Each "N O TuM" config is
# one feedback budget of log2(N*O) + TuM bits/user (O is an oversampling factor, not a bit
# count): 12 bits for every N, plus 16 bits at N=32.

K=(1 2 4)
SNR=(-10 -5 -2.5 0 2.5 5 10 15)
configs=("4 32 5" "8 32 4" "32 4 5" "32 32 6")   # "N O TuM"

cd "$(dirname "$0")"
source ../experiment_grid.sh   # naming conventions + parameter checks

for config in "${configs[@]}"; do
    read -r N O MB <<< "$config"
    read -r -a UT <<< "$(grid_allowed_uts "$N")"
    for k in "${K[@]}"; do
        python3 dft.py \
            --num-codebooks-to-test 2 \
            --num-bs-ant "$N" \
            --num-uts "${UT[@]}" \
            --num-gmm-components "$k" \
            --Td "$N" \
            --num-feedback-bits "$O" \
            --num-magnitude-bits "$MB" \
            --max-wmmse-iter 20 \
            --channel-inversion-type lmmse \
            --snr-db "${SNR[@]}" \
            --scenario 'gmm' \
            --rotate-codebook True
    done
done
