#!/bin/bash
# KVQ (KLT + reverse water-filling transform coding) baseline on the deepmimo channels, via
# transform_vq.py. Tu is the total feedback bits/user. Needs ../channel_data/ from make_channels.py.

N_list=(4 8 32)
Tu=(8 12 16 20 40)
SNR=(-10 -5 -2.5 0 2.5 5 10 15)

cd "$(dirname "$0")"
source ../experiment_grid.sh   # naming conventions + parameter checks

for N in "${N_list[@]}"; do
    Td="$(grid_expected_td deepmimo "$N")"
    read -r -a UT <<< "$(grid_allowed_uts "$N")"
    python3 transform_vq.py \
        --num-codebooks-to-test 1 \
        --num-bs-ant "$N" \
        --num-uts "${UT[@]}" \
        --Td "$Td" \
        --num-feedback-bits "${Tu[@]}" \
        --max-wmmse-iter 20 \
        --channel-inversion-type lmmse \
        --snr-db "${SNR[@]}"
done
