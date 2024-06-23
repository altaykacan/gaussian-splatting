#!/bin/bash


# # Baseline with colmap data
# python train.py --source_path /usr/stud/kaa/data/splats/custom/munich_ottendichler_colmap --iterations 30000 --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000  -m output/113_ottendichler_colmap

# # Baseline with dense cloud
# python train.py --source_path /usr/stud/kaa/data/splats/custom/munich_ottendichler_1024_576_dense_orb_rgbd --images /usr/stud/kaa/data/munich_suburbs/ottendichler/images  --scale_depths --use_inverse_depth --use_mask --iterations 30000 --save_iterations 1 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/114_ottendichler_dense

# # dn-reg, lambda_normal 0.2, sh 0
# python train.py --source_path /usr/stud/kaa/data/splats/custom/munich_ottendichler_1024_576_dense_orb_rgbd --images /usr/stud/kaa/data/munich_suburbs/ottendichler/images --scale_depths --use_gt_depth --use_inverse_depth --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/115_1_ottendichler_dense_dn_reg --sh_degree 0

# # dn-reg, lambda_normal 0.5, sh 0
# python train.py --source_path /usr/stud/kaa/data/splats/custom/munich_ottendichler_1024_576_dense_orb_rgbd --images /usr/stud/kaa/data/munich_suburbs/ottendichler/images --scale_depths --use_gt_depth --use_inverse_depth --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.5 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/115_2_ottendichler_dense_dn_reg --sh_degree 0


# # den-reg, sh0
# python train.py --source_path /usr/stud/kaa/data/splats/custom/munich_ottendichler_1024_576_dense_orb_rgbd --images /usr/stud/kaa/data/munich_suburbs/ottendichler/images --scale_depths --use_gt_depth --use_inverse_depth --use_gt_normal --init_from_normals --iterations 35000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 35000 --sh_degree 0 --apply_entropy_losses_from_iter 30000 --apply_entropy_losses_until_iter 35000 -m output/116_1_ottendichler_dense_den_reg --use_entropy_regularization

# # den-reg, sh3
# python train.py --source_path /usr/stud/kaa/data/splats/custom/munich_ottendichler_1024_576_dense_orb_rgbd --images /usr/stud/kaa/data/munich_suburbs/ottendichler/images --scale_depths --use_gt_depth --use_inverse_depth  --use_gt_normal --init_from_normals --iterations 35000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 35000 --apply_entropy_losses_from_iter 30000 --apply_entropy_losses_until_iter 35000 -m output/116_2_ottendichler_dense_den_reg --use_entropy_regularization


# # dense using colmap poses
# python train.py --source_path /usr/stud/kaa/data/splats/custom/munich_ottendichler_1024_576_dense_colmap  --images /usr/stud/kaa/data/munich_suburbs/ottendichler/images  --scale_depths --use_inverse_depth --use_mask --iterations 30000 --save_iterations 1 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/117_ottendichler_dense_colmap

# dn reg using colmap poses
python train.py --source_path  /usr/stud/kaa/data/splats/custom/munich_ottendichler_1024_576_dense_colmap --images /usr/stud/kaa/data/munich_suburbs/ottendichler/images --scale_depths --use_gt_depth --use_inverse_depth --use_gt_normal --init_from_normals --iterations 30000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 -m output/118_1_ottendichler_dense_colmap_dn_reg --sh_degree 0

# den reg using colmap poses
python train.py --source_path /usr/stud/kaa/data/splats/custom/munich_ottendichler_1024_576_dense_colmap --images /usr/stud/kaa/data/munich_suburbs/ottendichler/images --scale_depths --use_gt_depth --use_inverse_depth --use_gt_normal --init_from_normals --iterations 35000  --lambda_depth 0.2 --lambda_normal 0.2 --use_mask --save_iterations 1000 5000 30000 --test_iterations 1 1000 5000 7000 15000 30000 35000 --sh_degree 0 --apply_entropy_losses_from_iter 30000 --apply_entropy_losses_until_iter 35000 -m output/119_1_ottendichler_dense_colmap_den_reg --use_entropy_regularization