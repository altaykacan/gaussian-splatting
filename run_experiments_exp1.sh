#!/bin/bash

# Logging in a file with timestamp
timestamp=$(date +"%y%m%d-%H-%M")
logfile="log-exp1-${timestamp}.txt"
touch $logfile

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_colmap --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/01_baseline || echo "Experiment 01 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_colmap --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/02_baseline_low_lr --position_lr_init 0.000016 --scaling_lr 0.001 || echo "Experiment 02 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_colmap_custom_datareader_colmap_intrinsic --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/03_datareader || echo "Experiment 03 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_colmap_custom_datareader_colmap_intrinsic --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/04_datareader_mask_avg --use_mask || echo "Experiment 04 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/05_dense || echo "Experiment 05 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/06_dense_mask_avg --use_mask || echo "Experiment 06 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_ddrop --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/07_dense_ddrop || echo "Experiment 07 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_ddrop --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/08_dense_ddrop_mask_avg --use_mask || echo "Experiment 08 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_ddrop --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/09_dense_ddrop_low_lr --position_lr_init 0.000016 --scaling_lr 0.001 || echo "Experiment 09 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_ddrop --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/10_dense_ddrop_mask_avg_low_lr --use_mask  --position_lr_init 0.000016 --scaling_lr 0.001 || echo "Experiment 10 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/drjohnson_colmap/ --images /usr/stud/kaa/data/splats/custom/drjohnson_colmap/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/20_drjohnson_baseline || echo "Experiment 20 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/drjohnson_colmap_custom_datareader/ --images /usr/stud/kaa/data/splats/custom/drjohnson_colmap/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/21_drjohnson_custom_datareader || echo "Experiment 21 failed!" >> $logfile && true
