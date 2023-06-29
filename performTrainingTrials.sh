#!/bin/bash
sleeptime=10
TRIALS=10
sync="none" # neptune.ai api key needs to be configured first in SNCLtraining.py
trial_id=`date +%Y-%m-%d_%H-%M-%S`

# Requires ~24 GB VRAM to run in parallel
# nvidia-smi get free vram
# nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
# echo "Free VRAM:" $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}' | sort -n | head -n 1)

ncl_values=(0 0.25 0.5 0.75 0.9 1.0 )
base_models=(3 5 10 15 20 )
for ncl_weight in "${ncl_values[@]}"
do
    for base_model in "${base_models[@]}"
    do
        NCL="GNCL"
        echo 'run trial: models' $base_model 'NCL' $NCL 'weight' $ncl_weight
        'C:\ProgramData\Anaconda3\python.exe' 'SNCLtraining.py' --ncl_weight $ncl_weight --ncl_type $NCL --base_models $base_model --neptuneai $sync --trial_id $trial_id --silent True &
        sleep $sleeptime
    done
    wait
    for base_model in "${base_models[@]}"
    do
        NCL="SNCL"
        echo 'run trial: models' $base_model 'NCL' $NCL 'weight' $ncl_weight
        'C:\ProgramData\Anaconda3\python.exe' 'SNCLtraining.py' --ncl_weight $ncl_weight --ncl_type $NCL --base_models $base_model --neptuneai $sync --trial_id $trial_id --silent True &
        sleep $sleeptime
    done
    wait
done