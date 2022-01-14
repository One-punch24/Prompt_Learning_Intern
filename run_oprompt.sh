#!/usr/bin/env sh
PARTITION="shlab_nlp_klp"
NUM_GPUS="1"
CPUS="8"
JOB_NAME="OPrompt_5_10"
python="/mnt/lustre/kennethkong/anaconda3/envs/op/bin/python3"

# export PATH="/mnt/lustre/share/cuda-11.1/bin:$PATH"
# export RUN_NAME=prefix_tapas_bsz3_accum8_gpu4_eval100
# --optim_prefix yes --preseqlen 5 --epoch 5 --learning_rate 0.00005 --mode webnlg --bsz 1 --seed 101
CMD="python Prefix_tune.py --bsz 5 "



srun -s -p $PARTITION --gres=gpu:${NUM_GPUS:-1} --cpus-per-task=${CPUS:-1} \
    --job-name=$JOB_NAME $CMD