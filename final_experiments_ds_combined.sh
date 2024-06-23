python train.py --source_path /usr/stud/kaa/data/splats/custom_new/ds_combined_colmap --images /usr/stud/kaa/data/root/ds_combined/data/rgb --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/01_ds_combined_colmap

python train.py --source_path /usr/stud/kaa/data/splats/custom_new/ds_combined_colmap --images /usr/stud/kaa/data/root/ds_combined/data/rgb --iterations 30000 --save_iterations 1000 5000 10000 20000 --use_mask -m output/02_ds_combined_colmap_masked

python train.py --source_path /usr/stud/kaa/data/splats/custom_new/ds_combined_dense --images /usr/stud/kaa/data/root/ds_combined/data/rgb --use_gt_depth  --gt_depth_path /usr/stud/kaa/data/root/ds_combined/data/depths/arrays  --use_inverse_depth --lambda_depth 0.2 --iterations 30000 --save_iterations 1000 5000 10000 20000 --use_mask -m output/03_ds_combined_dense_d_reg



