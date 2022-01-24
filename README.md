# Prompt Tuning Convergence

## No Log Bleu

### DART
CUDA_VISIBLE_DEVICE=1 python Prefix_tune.py --dataset dart --model_mode FineTune --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICE=1 python Prefix_tune.py --dataset dart --model_mode PrefixModel --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICE=1 python Prefix_tune.py --dataset dart --model_mode BiasTune
--epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICE=1 python Prefix_tune.py --dataset dart --model_mode PT_plus_FineTune --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICE=1 python Prefix_tune.py --dataset dart --model_mode PT_plus_BiasTune --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

### E2E


## Log Bleu

python Prefix_tune.py --dataset e2e --model_mode PrefixModel --epochs 5 --lr 0.00008 --bsz 10 --project_name Bleu_dynamic
python Prefix_tune.py --dataset e2e --model_mode FineTune --epochs 5 --lr 0.00008 --bsz 10 --project_name Bleu_dynamic

