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


python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_mask --use_gt_normal --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.3 --lambda_normal 0.2  -m output/82_redo_ds_combined_crossroad_dense_dn_reg_inverse_depth

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense_normals --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_mask --use_gt_normal --init_from_normals --scale_depths --use_gt_depth  --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.3 --lambda_normal 0.2  -m output/83_ds_combined_crossroad_dense_dn_reg_n_init

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense_normals --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_mask --use_gt_normal --init_from_normals --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.3 --lambda_normal 0.2  -m output/84_ds_combined_crossroad_dense_dn_reg_n_init_inverse_depth

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense_normals --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_mask --use_gt_normal --init_from_normals --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.3 --lambda_normal 0.2  -m output/84_lr_ds_combined_crossroads_dense_dn_reg_n_init_inverse_depth --scaling_lr 0.001

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense_normals --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_mask --use_gt_normal --init_from_normals --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.3 --lambda_normal 0.2  -m output/84_lr_rot_ds_combined_crossroad_dense_dn_reg_n_init_lower_scale_rot_lr --scaling_lr 0.001 --rotation_lr 0.00001

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense_normals --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_mask --use_gt_normal --init_from_normals --scale_depths --use_gt_depth --use_inverse_depth --use_tv_loss_normal --use_tv_loss_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.3 --lambda_normal 0.2  -m output/85_ds_combined_crossroads_dense_dn_reg_n_init_inverse_depthds_combined_crossroad_dn_reg_n_init_tv_losses --scaling_lr 0.001


python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense_normals --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_mask --use_gt_normal --init_from_normals --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.3 --lambda_normal 0.2  -m output/86_ds_combined_crossroad_dn_reg_n_init_late_densification --scaling_lr 0.001  --densify_from_iter 5000

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense_normals --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_mask --use_gt_normal --init_from_normals --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.2 --lambda_normal 0.2  -m output/87_ds_combined_crossroad_dn_reg_n_init_opacity_inverse_depth --scaling_lr 0.001 --use_constant_opacity_loss --init_opacity 0.1 --opacity_target 0.95

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense_normals --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_mask --use_gt_normal --init_from_normals --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.2 --lambda_normal 0.2  -m output/88_ds_combined_crossroad_dn_reg_n_init_opacity_inverse_depth --scaling_lr 0.001 --use_constant_opacity_loss --init_opacity 0.5 --opacity_target 0.95


python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense_normals --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_mask --use_gt_normal --init_from_normals --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.2 --lambda_normal 0.2  --lambda_opacity 0.01 -m output/89_ds_combined_crossroad_dn_reg_n_init_opacity_inverse_depth --scaling_lr 0.001 --use_constant_opacity_loss --init_opacity 0.5 --opacity_target 0.95

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_combined_crossroad_dense_normals --images /usr/stud/kaa/data/deepscenario/combined_crossroad/images --use_mask --use_gt_normal --init_from_normals --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/combined_crossroad/depths  --iterations 30000 --save_iterations 1 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  --lambda_depth 0.2 --lambda_normal 0.2  -m output/90_ds_combined_crossroad_dn_reg_n_init_opacity_init_late_densification --scaling_lr 0.001 --init_opacity 0.5 --densify_from_iter 5000