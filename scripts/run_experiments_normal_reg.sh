#!/bin/bash

# Logging in a file with timestamp
timestamp=$(date +"%y%m%d-%H-%M")
logfile="log-exp-combined-${timestamp}.txt"
touch $logfile


function call_buffer() {
    echo "======================"
    echo "Calling next training run"
    echo "======================"
    sleep 0 # change number of seconds to wait depending on usage
}



python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_gt_normal --scale_depths --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.2 --lambda_normal 0.1 -m output/80_test

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_gt_normal --scale_depths --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.2 --lambda_normal 0.1 --use_mask  -m output/80_m_test

# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --scale_depths --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths --iterations 30000 --save_iterations 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.2 --position_lr_init 0.000016 --scaling_lr 0.001 -m output/80_lower_lr_ds_combined_crossroad_dense_depth_reg
