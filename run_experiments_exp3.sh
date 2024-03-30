#!/bin/bash

# Logging in a file with timestamp
timestamp=$(date +"%y%m%d-%H-%M")
logfile="log-exp3-${timestamp}.txt"
touch $logfile

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_02_1024_576_colmap --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/36_baseline_02 || echo "Experiment 36 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_02_1024_576_dense --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX020061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/32_deepscenario_02 || echo "Experiment 32 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_02_1024_576_dense --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX020061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/32_m_deepscenario_02 --use_mask || echo "Experiment 33_m failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_02_1024_576_dense_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX020061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/33_deepscenario_02_sfm || echo "Experiment 33 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_02_1024_576_dense_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX020061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/33_m_deepscenario_02_sfm --use_mask || echo "Experiment 33_m failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_02_1024_576_dense_skydome --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX020061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/34_deepscenario_02_skydome || echo "Experiment 34 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_02_1024_576_dense_skydome --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX020061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/34_m_deepscenario_02_skydome --use_mask || echo "Experiment 34_m failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_02_1024_576_dense_skydome_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX020061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/35_deepscenario_02_skydome_sfm || echo "Experiment 35 failed!" >> $logfile && true

python train.py --source_path /usr/stud/kaa/data/splats/custom/deepscenario_02_1024_576_dense_skydome_sfm --images /usr/stud/kaa/storage/user/kaa/data/deepscenario/GX020061_1024_576/images --iterations 30000 --save_iterations 1000 5000 10000 20000 -m output/35_m_deepscenario_02_skydome_sfm || echo "Experiment 35_m failed!" >> $logfile && true
