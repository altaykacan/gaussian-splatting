#!/bin/bash

# Logging in a file with timestamp
timestamp=$(date +"%y%m%d-%H-%M")
logfile="log-exp1-${timestamp}.txt"
touch $logfile

# python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_02_1024_576_colmap_opencv --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/21_baseline_02_opencv || echo "Experiment 21 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/22_dense_sfm || echo "Experiment 22 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_sky_amp --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/23_dense_skyamp || echo "Experiment 23 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_skybox --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/24_dense_skybox || echo "Experiment 24 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_skybox_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/25_dense_skybox_sfm || echo "Experiment 25 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_skydome   --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/26_dense_skydome || echo "Experiment 26 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_skydome_sfm   --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/27_dense_skydome_sfm || echo "Experiment 27 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/munich_01_1024_576_colmap  --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/28_munich_01_baseline || echo "Experiment 28 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/munich_02_1024_576_colmap  --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/30_munich_02_baseline || echo "Experiment 30 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/munich_02_1024_576_colmap_opencv  --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/31_munich_02_baseline_opencv || echo "Experiment 31 failed!" >> $logfile && true