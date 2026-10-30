#!/bin/bash
# FSQ (DFT pilot matrix) and FSQM (learned pilot matrix) on the GMM channels, via main.py.
# Each "N J Tu K" config is one the paper figures use (Td = N, see experiment_grid.sh);
# three independent training runs each (results_fsq_1..3, averaged by
# analysis/aggregate_results.py).

dirnum_list=("_1" "_2" "_3")
configs=("8 8 12 2"
         "32 12 12 1" "32 12 12 2" "32 12 12 4"
         "32 12 16 1" "32 12 16 2" "32 12 16 4"
         "32 20 12 2")   # "N J Tu K"
dft_list=("False" "True")

cd "$(dirname "$0")"
source ../experiment_grid.sh   # naming conventions + parameter checks
for config in "${configs[@]}"; do
    read -r N ut tu k <<< "$config"
    grid_check gmm "$N" "$N" -- "$ut"
done
for DIRNUM in "${dirnum_list[@]}"; do
    grid_require_fsq_dirnum "$DIRNUM"
done

for DIRNUM in "${dirnum_list[@]}"; do
    for config in "${configs[@]}"; do
        read -r N ut tu k <<< "$config"
        for dft in "${dft_list[@]}"; do
            echo "--------------------------------------------"
            echo "Running with N: $N | UTs: $ut | Tu: $tu | K: $k | DFT Matrix: $dft | Dirnum: $DIRNUM"
            echo "--------------------------------------------"

            python3 main.py \
                --num-bs-ant "$N" \
                --num-uts "$ut" \
                --num-gmm-components "$k" \
                --Td "$N" \
                --Tu "$tu" \
                --snr-db 100 \
                --use-dft-matrix "$dft" \
                --save-model False \
                --dirnum "$DIRNUM"
        done
    done
done
