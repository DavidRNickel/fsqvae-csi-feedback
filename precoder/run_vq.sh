#!/bin/bash
# VQ (DFT pilot matrix) and VQM (learned pilot matrix) on the deepmimo channels, via main_vq.py.
# Same grid as run_fsq.sh: N = 4, 8, 32 (J and Td from experiment_grid.sh), Tu = 8 12 16 20,
# three independent training runs (results_vq_1..3). Needs channel_data/ from make_channels.py.

dirnum_list=("_1" "_2" "_3")
N_list=(4 8 32)
tu_list=(8 12 16 20)
dft_list=("False" "True")

cd "$(dirname "$0")"
source ./experiment_grid.sh   # naming conventions + parameter checks
for DIRNUM in "${dirnum_list[@]}"; do
    grid_require_fsq_dirnum "$DIRNUM"   # VQ shares FSQ's _<run> convention (results_vq_<run>)
done

for DIRNUM in "${dirnum_list[@]}"; do
    for N in "${N_list[@]}"; do
        Td="$(grid_expected_td deepmimo "$N")"
        read -r -a uts_list <<< "$(grid_allowed_uts "$N")"
        for ut in "${uts_list[@]}"; do
            for tu in "${tu_list[@]}"; do
                for dft in "${dft_list[@]}"; do
                    echo "--------------------------------------------"
                    echo "Running with N: $N | UTs: $ut | Td: $Td | Tu: $tu | DFT Matrix: $dft | Dirnum: $DIRNUM"
                    echo "--------------------------------------------"

                    python3 main_vq.py \
                        --num-bs-ant "$N" \
                        --num-uts "$ut" \
                        --Td "$Td" \
                        --Tu "$tu" \
                        --use-dft-matrix "$dft" \
                        --save-model False \
                        --dirnum "$DIRNUM"
                done
            done
        done
    done
done
