#!/bin/bash
# FSQ (DFT pilot matrix) and FSQM (learned pilot matrix) on the deepmimo channels, via main.py.
# Covers every config in the paper figures: N = 4, 8, 32 (J and Td from experiment_grid.sh),
# Tu = 8 12 16 20, three independent training runs (results_fsq_1..3, averaged by
# analysis/aggregate_results.py). Needs channel_data/ from make_channels.py.
#
# The last block trains the one checkpoint the latent-space figure loads (results_fsq_eval/).

dirnum_list=("_1" "_2" "_3")
N_list=(4 8 32)
tu_list=(8 12 16 20)
dft_list=("False" "True")

cd "$(dirname "$0")"
source ./experiment_grid.sh   # naming conventions + parameter checks
for DIRNUM in "${dirnum_list[@]}"; do
    grid_require_fsq_dirnum "$DIRNUM"
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

                    python3 main.py \
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

# latent-space figure (analysis/make_paper_figures.py, fig_latent_space): FSQM, N=32, J=20, Tu=12
python3 main.py \
    --num-bs-ant 32 \
    --num-uts 20 \
    --Td "$(grid_expected_td deepmimo 32)" \
    --Tu 12 \
    --use-dft-matrix False \
    --save-model True \
    --dirnum _eval
