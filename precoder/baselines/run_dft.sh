#!/bin/bash
# DVQ (rotated DFT codebook) baseline on the deepmimo channels, via dft.py. Each "N O TuM" config
# is one feedback budget of log2(N*O) + TuM bits/user (O is an oversampling factor, not a bit
# count): 12 bits for every N, plus 16 bits at N=32. Needs ../channel_data/ from make_channels.py.

SNR=(-10 -5 -2.5 0 2.5 5 10 15)
configs=("4 32 5" "8 16 5" "32 4 5" "32 64 5")   # "N O TuM"

cd "$(dirname "$0")"
source ../experiment_grid.sh   # naming conventions + parameter checks

for config in "${configs[@]}"; do
    read -r N O MB <<< "$config"
    Td="$(grid_expected_td deepmimo "$N")"
    read -r -a UT <<< "$(grid_allowed_uts "$N")"
    python3 dft.py \
        --num-codebooks-to-test 1 \
        --num-bs-ant "$N" \
        --num-uts "${UT[@]}" \
        --Td "$Td" \
        --num-feedback-bits "$O" \
        --num-magnitude-bits "$MB" \
        --max-wmmse-iter 20 \
        --channel-inversion-type lmmse \
        --snr-db "${SNR[@]}" \
        --rotate-codebook True
done
