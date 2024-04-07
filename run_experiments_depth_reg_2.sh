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

# Temporary
python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --use_mask --use_gt_depth --iterations 40000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 1500 30000 40000 --lambda_depth 0.2 -m output/55_redo_ds_01_depth_reg_true_sfm


python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.2 --scale_depths -m output/65_redo_ds_01_depth_reg_False_sfm_colmap_intr_scaled_cloud || echo "Experiment 65 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --iterations 40000 --save_iterations 1000 5000 30000 40000 --test_iterations 1 1000 7000 15000 30000 40000 --lambda_depth 0.2 -m output/66_ds_01_depth_reg_True_sfm_scaled_cloud

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --iterations 40000 --save_iterations 1000 5000 30000 40000 --test_iterations 1 1000 7000 15000 30000 40000 --lambda_depth 0.2 --position_lr_init 0.000016 --scaling_lr 0.001 -m output/67_ds_01_depth_reg_True_colmap_intr_lower_lr

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --iterations 40000 --save_iterations 1000 5000 30000 40000 --test_iterations 1 1000 7000 15000 30000 40000 --lambda_depth 0.1 -m output/68_ds_01_depth_reg_True_sfm_scaled_cloud_lambda01

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --iterations 40000 --save_iterations 1000 5000 30000 40000 --test_iterations 1 1000 7000 15000 30000 40000 --lambda_depth 0.01 -m output/69_ds_01_depth_reg_True_sfm_scaled_cloud_lambda001

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --iterations 40000 --save_iterations 1000 5000 30000 40000 --test_iterations 1 1000 7000 15000 30000 40000 --lambda_depth 0.2 --position_lr_init 0.000016 --scaling_lr 0 -m output/70_ds_01_depth_reg_False_sfm_scaled_cloud_scale_fixed

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --iterations 40000 --save_iterations 1000 5000 30000 40000 --test_iterations 1 1000 7000 15000 30000 40000 --lambda_depth 0.2 --position_lr_init 0.000016 --scaling_lr 0 -m output/71_ds_01_depth_reg_True_sfm_scaled_cloud_scale_fixed

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --iterations 40000 --save_iterations 1000 5000 30000 40000 --test_iterations 1 1000 7000 15000 30000 40000 --lambda_depth 0.2 --position_lr_init 0.0 --scaling_lr 0.001 -m output/72_ds_01_depth_reg_False_sfm_scaled_cloud_pos_fixed

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --iterations 40000 --save_iterations 1000 5000 30000 40000 --test_iterations 1 1000 7000 15000 30000 40000 --lambda_depth 0.2 --position_lr_init 0 --scaling_lr 0.001 -m output/73_ds_01_depth_reg_True_sfm_scaled_cloud_pos_fixed

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --iterations 40000 --save_iterations 1000 5000 30000 40000 --test_iterations 1 1000 7000 15000 30000 40000 --lambda_depth 0.2 --min_gt_depth 5.0 -m output/74_ds_01_depth_reg_True_sfm_scaled_cloud_min_depth5