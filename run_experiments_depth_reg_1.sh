#!/bin/bash

# Logging in a file with timestamp
timestamp=$(date +"%y%m%d-%H-%M")
logfile="log-exp-depth-reg1-${timestamp}.txt"
touch $logfile

python train.py --source_path  /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.2 -m output/52_ds_01_depth_reg_false || echo "Experiment 52 failed!" >> $logfile && true


function call_buffer() {
    echo "======================"
    echo "Calling next training run"
    echo "======================"
    sleep 0 # change number of seconds to wait depending on usage
}

# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --use_mask --use_gt_depth --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 1500 30000 --lambda_depth 0.2 -m output/53_ds_01_depth_reg_true || echo "Experiment 53 failed!" >> $logfile && true

call_buffer

python train.py --source_path  /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 1500 30000 --lambda_depth 0.2 -m output/54_ds_01_depth_reg_false_sfm || echo "Experiment 54 failed!" >> $logfile && true

call_buffer

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --use_mask --use_gt_depth --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 1500 30000 --lambda_depth 0.2 -m output/55_ds_01_depth_reg_true_sfm || echo "Experiment 55 failed!" >> $logfile && true

call_buffer

# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --use_mask --use_gt_depth --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.1 -m output/56_ds_01_depth_reg_true_sfm_lower_lambda || echo "Experiment 56 failed!" >> $logfile && true

call_buffer

# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --use_mask --use_gt_depth --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 1500 30000 --lambda_depth 0.2 --position_lr_init 0.000016 --scaling_lr 0.001 -m output/57_ds_01_depth_reg_true_sfm_lower_lr || echo "Experiment 57 failed!" >> $logfile && true

call_buffer

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_colmap_intrinsics  --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 15000 30000  --test_iterations 1 1000 7000 1500 30000 -m output/58_ds_01_depth_reg_False_colmap_intr || echo "Experiment 58 failed!" >> $logfile && true

call_buffer

# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_colmap_intrinsics --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --use_mask --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.2 -m output/59_ds_01_depth_reg_True_colmap_intr || echo "Experiment 59 failed!" >> $logfile && true

call_buffer

# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.2  -m output/60_ds_01_depth_reg_False_sfm_colmap_intr || echo "Experiment 60 failed!" >> $logfile && true

call_buffer

# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --use_mask --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.2 -m output/61_ds_01_depth_reg_True_sfm_colmap_intr || echo "Experiment 61 failed!" >> $logfile && true

call_buffer

# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --use_mask --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.1 -m output/62_ds_01_depth_reg_False_colmap_intr_lower_lambda || echo "Experiment 62 failed!" >> $logfile && true

call_buffer

# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --use_mask --use_gt_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths_colmap_intrinsics --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.2 --position_lr_init 0.000016 --scaling_lr 0.001 -m output/63_ds_01_depth_reg_True_colmap_intr_lower_lr || echo "Experiment 63 failed!" >> $logfile && true

call_buffer

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.2 -m output/64_ds_01_depth_reg_False_sfm_scaled_cloud --scale_depths || echo "Experiment 64 failed!" >> $logfile && true

call_buffer

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_depth_reg_sfm_colmap_intrinsics_scaled_cloud --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 --test_iterations 1 1000 7000 15000 30000 --lambda_depth 0.2 --scale_depths -m output/65_ds_01_depth_reg_False_sfm_colmap_intr_scaled_cloud || echo "Experiment 65 failed!" >> $logfile && true