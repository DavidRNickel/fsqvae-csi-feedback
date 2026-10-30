#!/bin/bash
# CVQ baseline on the deepmimo channels, via cvq_proj.py. Each "Tu TuM" pair is one feedback
# budget (Tu direction bits + TuM magnitude bits = 8, 12, 12, 16, 20 bits/user); N=32 skips the
# 14-bit direction codebook. Needs ../channel_data/ from make_channels.py.

N_list=(4 8 32)
SNR=(-10 -5 -2.5 0 2.5 5 10 15)
pairs=("5 3" "7 5" "8 4" "11 5" "14 6")   # "Tu TuM"
DIRNUM=''   # CVQ results go to wmmse_results/ (aggregate_results.py reads only that); a suffix hides them

cd "$(dirname "$0")"
source ../experiment_grid.sh   # naming conventions + parameter checks
grid_require_wmmse_dirnum "$DIRNUM"

for N in "${N_list[@]}"; do
    Td="$(grid_expected_td deepmimo "$N")"
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
            --Td "$Td" \
            --num-feedback-bits "$Tu" \
            --num-magnitude-bits "$MB" \
            --max-wmmse-iter 20 \
            --channel-inversion-type lmmse \
            --snr-db "${SNR[@]}" \
            --dirnum "$DIRNUM"
    done
done
