#!/bin/bash


# Baseline with colmap data
#python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_colmap --iterations 30000 --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000  -m output/91_ent_ds_01_baseline_colmap

# Baseline with dense cloud
#python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images  --scale_depths --use_inverse_depth --use_mask --iterations 30000 --save_iterations 1 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/92_ent_ds_01_baseline_dense

# dn-reg
#python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/93_ent_ds_01_dense_dn_reg

# dn-reg and entropy, lambda 0.001
#python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 --use_entropy_regularization --lambda_entropy 0.001 -m output/106_1_ds_01_dn_reg_entropy

# dn-reg and entropy, lambda 0.01
#python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 --use_entropy_regularization --lambda_entropy 0.01 -m output/106_2_ds_01_dn_reg_entropy

# # dn-reg and entropy, lambda 1.1
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_02_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 --use_entropy_regularization --lambda_entropy 0.1 -m output/106_3_ds_01_dn_reg_entropy


# # dn-reg sh 1
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_02_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/107_1_dn_reg_sh --sh_degree 0

# # dn-reg sh 2
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_02_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/107_2_dn_reg_sh --sh_degree 1

# # dn-reg sh 2
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/107_3_dn_reg_sh --sh_degree 2

# # disk loss, dense, sh 0
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/108_1_dn_reg_disk_loss_sh --sh_degree 0 --use_disk_loss

# # disk loss, dense, sh 3
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/108_2_dn_reg_disk_loss_sh --use_disk_loss

# # disk loss, dense, sh 3, higher lambda for disk loss and normal
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.5 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/108_3_dn_reg_disk_loss_sh --use_disk_loss --lambda_disk 0.5

# # disk loss, dense, sh 0, with alpha entropy loss
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/109_dn_reg_entropy_disk_loss --sh_degree 0 --use_disk_loss --use_entropy_regularization --lambda_entropy 0.1

# # dn-reg and entropy, entropy lambda 0.01, sh 0
# python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 --use_entropy_regularization --lambda_entropy 0.01 -m output/110_1_ds_01_dn_reg_entropy_sh --sh_degree 0

# Late onset entropy regularization, 20k-25k
python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 --use_entropy_regularization --lambda_entropy 0.1 -m output/112_1_ds_01_dn_reg_late_entropy --sh_degree 0 --apply_entropy_losses_from_iter 20000 --apply_entropy_losses_until_iter 25000

# Late onset entropy regularization, 25k-30k
python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 --use_entropy_regularization --lambda_entropy 0.1 -m output/112_2_ds_01_dn_reg_late_entropy --sh_degree  0 --apply_entropy_losses_from_iter 25000 --apply_entropy_losses_until_iter 30000

# Late onset entropy regularization, 25k-30k, entropy lambda 0.5
python train.py --source_path /usr/stud/kaa/data/splats/custom/ds_01_1024_576_dense_colmap_normal --images /usr/stud/kaa/data/deepscenario/GX010061_1024_576/images --scale_depths --use_gt_depth --use_inverse_depth --gt_depth_path /usr/stud/kaa/data/deepscenario/GX010061_1024_576/depths --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 --use_entropy_regularization --lambda_entropy 0.5 -m output/112_3_ds_01_dn_reg_late_entropy --sh_degree 0 --apply_entropy_losses_from_iter 25000 --apply_entropy_losses_until_iter 30000

