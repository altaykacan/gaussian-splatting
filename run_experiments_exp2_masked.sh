#!/bin/bash

# Logging in a file with timestamp
timestamp=$(date +"%y%m%d-%H-%M")
logfile="log-exp2-${timestamp}.txt"
touch $logfile


python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/22_m_dense_sfm --use_mask || echo "Experiment 22_m failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_sky_amp --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/23_m_dense_skyamp --use_mask || echo "Experiment 23_m failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_skybox --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/24_m_dense_skybox || echo "Experiment 24_m failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_skybox_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/25_m_dense_skybox_sfm --use_mask || echo "Experiment 25_m failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_skydome   --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/26_m_dense_skydome --use_mask || echo "Experiment 26_m failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_01_1024_576_dense_skydome_sfm   --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX010061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/27_m_dense_skydome_sfm --use_mask || echo "Experiment 27_m failed!" >> $logfile && true