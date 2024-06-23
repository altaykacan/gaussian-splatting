#!/bin/bash

# Logging in a file with timestamp
timestamp=$(date +"%y%m%d-%H-%M")
logfile="log-exp-prune-entropy-${timestamp}.txt"
touch $logfile


function call_buffer() {
    echo "======================"
    echo "Calling next training run"
    echo "======================"
    sleep 0 # change number of seconds to wait depending on usage
}

# Baseline with colmap data
# python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_colmap --iterations 30000 --save_iterations 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000  -m output/91_ds_01_baseline_colmap

# Baseline with dense cloud
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images  --scale_depths --use_inverse_depth --use_mask --iterations 30000 --save_iterations 1 5000 30000 --test_iterations 1 1000 7000 15000 30000 -m output/92_ds_01_baseline_dense

# dn-reg
python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000 -m output/93_ds_01_dense_dn_reg

# late densification
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images   --scale_depths --use_inverse_depth --use_mask --iterations 30000 --save_iterations 1 5000 30000 --test_iterations 1 1000 7000 15000 30000 --densify_from_iter 5000 -m output/94_ds_01_dense_late_densification

# late densification dn-reg
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images  --use_gt_depth --scale_depths --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --iterations 30000 --save_iterations 1 5000 30000 --test_iterations 1 1000 7000 15000 30000 --densify_from_iter 5000 -m output/95_ds_01_dense_dn_reg_late_densification


# Minimum opacity ablation
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images  --use_gt_depth --scale_depths --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --lambda_depth 0.2 --lambda_normal 0.2 --init_opacity 0.5 --minimum_opacity 0.001 --use_mask --iterations 30000 --save_iterations 1 5000 30000 --test_iterations 1 1000 7000 15000 30000 -m output/96_1_ds_01_dense_dn_reg_min_opacity

# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images  --use_gt_depth --scale_depths --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --lambda_depth 0.2 --lambda_normal 0.2 --init_opacity 0.5 --minimum_opacity 0.005 --use_mask --iterations 30000 --save_iterations 1 5000 30000 --test_iterations 1 1000 7000 15000 30000 -m output/96_2_ds_01_dense_dn_reg_min_opacity

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images  --use_gt_depth --scale_depths --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --lambda_depth 0.2 --lambda_normal 0.2 --init_opacity 0.5 --minimum_opacity 0.01 --use_mask --iterations 30000 --save_iterations 1 5000 30000 --test_iterations 1 1000 7000 15000 30000 -m output/96_3_ds_01_dense_dn_reg_min_opacity

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images  --use_gt_depth --scale_depths --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --lambda_depth 0.2 --lambda_normal 0.2 --init_opacity 0.5 --minimum_opacity 0.1 --use_mask --iterations 30000 --save_iterations 1 5000 30000 --test_iterations 1 1000 7000 15000 30000 -m output/96_4_ds_01_dense_dn_reg_min_opacity

python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images  --use_gt_depth --scale_depths --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --lambda_depth 0.2 --lambda_normal 0.2 --init_opacity 0.5 --minimum_opacity 0.4 --use_mask --iterations 30000 --save_iterations 1 5000 30000 --test_iterations 1 1000 7000 15000 30000 -m output/96_5_ds_01_dense_dn_reg_min_opacity

# constant opacity loss
python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask  --save_iterations 1000 5000 30000 --test_iterations 1 1000 7000 15000 30000 -m output/97_ds_01_dense_dn_reg_opacity_reg --use_constant_opacity_loss --lambda_opacity 0.01