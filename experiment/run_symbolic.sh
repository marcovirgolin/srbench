#!/bin/bash

# for data in "../../pmlb/datasets/strogatz_" "../../pmlb/datasets/feynman_" ; do # feynman and strogatz datasets
#     for TN in 0 0.001 0.01 0.1; do 
#         python analyze.py \
#             $data"*" \
#             -ml PySRRegressor \
#             --local \
#             -results ../results_sym_data \
#             -target_noise $TN \
#             -sym_data \
#             -n_trials 10 \
#             -m 16384 \
#             -time_limit 9:00 \
#             -job_limit 100000 \
#             -n_jobs 40 \
#             -tuned 
#         if [ $? -gt 0 ] ; then
#             break
#         fi
#     done
# done

# assess the ground-truth models that were produced using sympy
for data in "../../pmlb/datasets/strogatz_" "../../pmlb/datasets/feynman_" ; do # feynman and strogatz datasets
    for TN in 0 0.001 0.01 0.1; do # noise levels
        python analyze.py \
            -script assess_symbolic_model \
            -ml PySRRegressor \
            $data"*" \
            -results ../results_sym_data \
            -target_noise $TN \
            -sym_data \
            -n_trials 10 \
            -n_jobs 40 \
            -m 8192 \
            -time_limit 1:00 \
            -job_limit 100000 \
            -tuned \
            --local 
        if [ $? -gt 0 ] ; then
            break
        fi
    done
done