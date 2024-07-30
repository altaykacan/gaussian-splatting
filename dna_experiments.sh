#!/usr/bin/bash

##############
#  Best performing config for DNA
##############
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 1.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad --reset_normals
##############

# more iters
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_more_iters --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 1.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad --iterations 40000 --reset_normals

# more depth lambda
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_lambda_d_1 --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 1.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 1.0 --dna_zero_grad  --reset_normals

# tv loss depth
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_tv_loss_depth --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 1.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad --use_tv_loss_depth --lambda_tv_depth 0.1 --reset_normals

# tv loss normal
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_tv_loss_normal --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 1.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad --use_tv_loss_normal --lambda_tv_normal 0.1 --reset_normals


# tv loss normal and depth
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_tv_loss_depth_normal --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 1.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad --use_tv_loss_depth --lambda_tv_depth 0.1 --use_tv_loss_normal --lambda_tv_normal 0.1  --reset_normals


# more dna lambda
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_lambda_n_2 --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 2.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad --reset_normals


# no normal reset higher dna lambda
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_no_normal_reset_lambda_n_10 --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 10.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad

# no opacity reg
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_no_opacity_reg --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 1.0 --lambda_depth 0.4 --dna_zero_grad --reset_normals

# with normal init
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_with_normal_init --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 1.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad --reset_normals --init_from_normals

# more frequent normal resets
 python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_more_normal_reset --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 1.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad --reset_normals --reset_normals_interval 500

 # more tv loss lambda
 python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_tv_loss_depth_normal_higher_tv_lambdas --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 1.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad --use_tv_loss_depth --lambda_tv_depth 0.5 --use_tv_loss_normal --lambda_tv_normal 0.5  --reset_normals

# with road mask regularization
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_no_normal_reset_road_regularization --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 1.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad --use_gt_road_mask

# with road mask regularization and higher lambda normal for dna
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_no_normal_reset_road_regularization_lambda_n_10 --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6629 --lambda_normal 10.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad --use_gt_road_mask

# with road mask regularization and no xyz grad cancelling
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_no_normal_reset_road_regularization_no_zero_grad --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 1.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --use_gt_road_mask

# with road mask regularization and normal init and way higher DNA lambda
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/best_no_normal_reset_road_regularization_normal_init_lambda_n_10k --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_dna --use_gt_depth --port 6679 --lambda_normal 10000.0 --use_constant_opacity_loss --lambda_opacity 0.01 --lambda_depth 0.4 --dna_zero_grad --use_gt_road_mask --init_from_normals



# standard normal reg with road opacity reg
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/standard_normal_reg --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_gt_depth --port 6669 --lambda_normal 0.4 --use_constant_opacity_loss --lambda_opacity 0.1 --lambda_depth 0.4 --dna_zero_grad --init_from_normals

# standard normal reg with road opacity reg and road reg
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/standard_normal_reg_road_regularization --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_gt_depth --port 6669 --lambda_normal 0.4 --use_constant_opacity_loss --lambda_opacity 0.1 --lambda_depth 0.4 --dna_zero_grad --init_from_normals --use_gt_road_mask


# standard normal reg with road opacity reg and road reg, higher lambda normal
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/standard_normal_reg_road_regularization_lambda_n_1 --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_gt_depth --port 6669 --lambda_normal 1.0 --use_constant_opacity_loss --lambda_opacity 0.1 --lambda_depth 0.4 --dna_zero_grad --init_from_normals --use_gt_road_mask

# standard normal reg, road mask regularization and no opacity reg and no xyz grad cancelling
python train.py --source_path /usr/stud/kaa/data/root/ds01/reconstructions/debug_dna --images /usr/stud/kaa/data/root/ds01/data/rgb --iterations 30000 --test_iterations 10 1000 5000 20000 25000 30000 -m /usr/stud/kaa/data/root/ds01/splats/standard_n_reg_road_reg_no_opacity_no_zero_grad --eval --scale_depths --use_inverse_depth --llffhold 10 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --use_gt_normal --use_gt_depth --port 6679 --lambda_normal 0.5 --lambda_depth 0.4 --use_gt_road_mask