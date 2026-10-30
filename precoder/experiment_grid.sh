#!/bin/bash
# Shared guard rails for the sum-rate run scripts (source this file; don't execute it).
#
# Naming conventions the rest of the pipeline (analysis/aggregate_results.py) depends on:
#
#   FSQ / FSQM   (main.py, gmm/main.py)          results_fsq_<run>/...   <run> = 1, 2, 3, ... are independent
#                                                training runs of the same experiment and are averaged
#                                                (mean +/- std). results_fsq_eval is the model used for the
#                                                latent-space plot and is NOT averaged. Pass --dirnum _<run>.
#   VQ / VQM     (main_vq.py, gmm/main_vq.py)    results_vq_<run>/...    same _<run> convention as FSQ
#                                                (grid_require_fsq_dirnum covers both).
#   CVQ / DVQ / KVQ (cvq_proj.py, dft.py,        wmmse_results/...       no run suffix: aggregate_results.py
#                    transform_vq.py)                                    reads ONLY wmmse_results/, so a
#                                                                        --dirnum here silently hides results.
#
#   Users per BS antenna count (every sum-rate result, all methods, both channel models):
#       N=4  -> J = 2 3 4        N=8 -> J = 2 4 6 8        N=32 -> J = 4 8 12 20
#   Pilot symbols Td:  gmm      Td = N
#                      deepmimo N=4 -> 4,  N=8 -> 6,  N=32 -> 8
#
# Usage in a run script:
#     source "$(dirname "$0")/../experiment_grid.sh"            # path to this file
#     grid_check gmm      "$N" "${Td[@]}" -- "${UT[@]}"          # or: deepmimo
#     grid_require_wmmse_dirnum "$DIRNUM"                        # CVQ/DVQ/KVQ scripts that take --dirnum
#     grid_require_fsq_dirnum   "$DIRNUM"                        # FSQ scripts

_grid_fail() {
    echo "experiment_grid: $*" >&2
    exit 1
}

# grid_allowed_uts N -> prints the allowed J values for that N
grid_allowed_uts() {
    case "$1" in
        4)  echo "2 3 4" ;;
        8)  echo "2 4 6 8" ;;
        32) echo "4 8 12 20" ;;
        *)  _grid_fail "N=$1 has no defined J set (known: 4, 8, 32); add it to experiment_grid.sh" ;;
    esac
}

# grid_expected_td CHANNEL N -> prints the expected Td (CHANNEL = gmm | deepmimo)
grid_expected_td() {
    local channel=$1 N=$2
    case "$channel" in
        gmm)      echo "$N" ;;
        deepmimo) case "$N" in 4) echo 4 ;; 8) echo 6 ;; 32) echo 8 ;; *) _grid_fail "N=$N has no defined deepmimo Td" ;; esac ;;
        *) _grid_fail "unknown channel model '$channel' (use gmm or deepmimo)" ;;
    esac
}

# grid_check CHANNEL N TD... -- J...   -> exits with a message unless every Td and J is on the grid
grid_check() {
    local channel=$1 N=$2
    shift 2
    local tds=() uts=() seen_sep=0 x
    for x in "$@"; do
        if [ "$x" = "--" ]; then seen_sep=1; continue; fi
        if [ $seen_sep -eq 0 ]; then tds+=("$x"); else uts+=("$x"); fi
    done
    [ $seen_sep -eq 1 ] || _grid_fail "grid_check needs '--' between the Td list and the J list"

    local want_td allowed_uts
    want_td="$(grid_expected_td "$channel" "$N")" || exit 1
    allowed_uts="$(grid_allowed_uts "$N")" || exit 1

    for x in "${tds[@]}"; do
        [ "$x" = "$want_td" ] || _grid_fail "$channel N=$N must use Td=$want_td, but this script has Td=$x"
    done
    for x in "${uts[@]}"; do
        case " $allowed_uts " in
            *" $x "*) ;;
            *) _grid_fail "N=$N allows J in {$allowed_uts}, but this script has J=$x" ;;
        esac
    done
}

# CVQ / DVQ / KVQ results must land in wmmse_results/ (see the header): --dirnum must be empty
grid_require_wmmse_dirnum() {
    [ -z "$1" ] || _grid_fail "DIRNUM='$1' would write to wmmse_results$1/, which aggregate_results.py never reads; leave it empty"
}

# FSQ results: _<run number> (averaged) or _eval (latent-plot model)
grid_require_fsq_dirnum() {
    case "$1" in
        _eval) ;;
        _[0-9]*) case "${1#_}" in *[!0-9]*) _grid_fail "DIRNUM='$1' is not _<run number> or _eval" ;; esac ;;
        *) _grid_fail "DIRNUM='$1' is not _<run number> (results_fsq_<run>, averaged) or _eval" ;;
    esac
}
