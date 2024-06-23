#!/bin/bash

# Logging in a file with timestamp
timestamp=$(date +"%y%m%d-%H-%M")
logfile="log-exp-depth-reg2-${timestamp}.txt"
touch $logfile


function call_buffer() {
    echo "======================"
    echo "Calling next training run"
    echo "======================"
    sleep 0 # change number of seconds to wait depending on usage
}

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --use_tv_loss --iterations 30000 --save_iterations 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.2 -m output/76_ds_01_depth_reg_True_sfm_scaled_cloud_tv_loss

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --use_tv_loss --iterations 30000 --save_iterations 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.2 -m output/76_2_ds_01_depth_reg_True_sfm_scaled_cloud_tv_loss


# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --use_tv_loss --use_mask --iterations 30000 --save_iterations 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.2 -m output/76_m_ds_01_depth_reg_True_sfm_scaled_cloud_tv_loss