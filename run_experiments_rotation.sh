#!/bin/bash




# dn-reg, less rotation lr, 1/10
python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.5 --use_mask --sh_degree 0 --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 --rotation_lr 0.0001 -m output/111_1_ds_01_dn_reg_rot

# dn-reg, less rotation lr, 1/100
python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.5 --use_mask --sh_degree 0 --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 --rotation_lr 0.00001 -m output/111_2_ds_01_dn_reg_rot

# dn-reg, less rotation lr, 1/1000
python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.5 --use_mask --sh_degree 0 --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 --rotation_lr 0.000001 -m output/111_3_ds_01_dn_reg_rot



