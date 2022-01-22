# Prompt Tuning Convergence

python Prefix_tune.py  --model_mode FineTune --epochs 5  --log_mode No_Log_Bleu
python Prefix_tune.py  --model_mode PrefixModel --epochs 5   --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=1 python Prefix_tune.py --dataset e2e --model_mode FineTune --epochs 5 --lr 0.00008 --bsz 10 --log_mode No_Log_Bleu
CUDA_VISIBLE_DEVICES=1 python Prefix_tune.py --dataset e2e --model_mode PrefixModel --epochs 5 --lr 0.00008 --bsz 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=2 python Prefix_tune.py --dataset dart --model_mode FineTune --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu
CUDA_VISIBLE_DEVICES=2 python Prefix_tune.py --dataset dart --model_mode PrefixModel --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu


## Log Bleu

python Prefix_tune.py --dataset e2e --model_mode PrefixModel --epochs 5 --lr 0.00008 --bsz 10 --project_name Bleu_dynamic
python Prefix_tune.py --dataset e2e --model_mode FineTune --epochs 5 --lr 0.00008 --bsz 10 --project_name Bleu_dynamic

CUDA_VISIBLE_DEVICES=3 python Prefix_tune.py  --model_mode FineTune --epochs 5 --log_mode Log_Bleu --project_name training_speed
CUDA_VISIBLE_DEVICES=3  python Prefix_tune.py  --model_mode PrefixModel --epochs 5  --log_mode No_log_Bleu --project_name training_speed

CUDA_VISIBLE_DEVICES=3 python Prefix_tune.py --dataset e2e --model_mode FineTune --epochs 5 --lr 0.00008 --bsz 10 --log_mode No_log_Bleu --project_name training_speed
CUDA_VISIBLE_DEVICES=3 python Prefix_tune.py --dataset e2e --model_mode PrefixModel --epochs 5 --lr 0.00008 --bsz 10 --log_mode No_log_Bleu --project_name training_speed

python Prefix_tune.py --dataset dart --model_mode FineTune --epochs 10 --lr 0.00005 --num_token 10 --log_mode Log_Bleu
python Prefix_tune.py --dataset dart --model_mode PrefixModel --epochs 10 --lr 0.00005 --num_token 10 --log_mode Log_Bleu --project_name Bleu_dynamic