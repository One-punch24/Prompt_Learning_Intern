# Prompt Tuning Convergence

## No Log Bleu

### DART
CUDA_VISIBLE_DEVICES=1 python Prefix_tune.py --dataset dart --model_mode PrefixModel --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=1 python Prefix_tune.py --dataset dart --model_mode FineTune --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=1 python Prefix_tune.py --dataset dart --model_mode BiasTune --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=1 python Prefix_tune.py --dataset dart --model_mode PT_plus_FineTune --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=1 python Prefix_tune.py --dataset dart --model_mode PT_plus_BiasTune --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

#### ADD

CUDA_VISIBLE_DEVICES=0 python Prefix_tune.py --dataset dart --model_mode FineTuneEval --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=0 python Prefix_tune.py --dataset dart --model_mode BiasTuneEval --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=1 python Prefix_tune.py --dataset dart --model_mode PT_plus_FineTuneEval --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=1 python Prefix_tune.py --dataset dart --model_mode PT_plus_BiasTuneEval --epochs 10 --lr 0.00005 --num_token 10 --log_mode No_Log_Bleu

### E2E

CUDA_VISIBLE_DEVICES=2  python Prefix_tune.py --dataset e2e --model_mode PrefixModel --epochs 50 --lr 0.00008 --bsz 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=2  python Prefix_tune.py --dataset e2e --model_mode FineTune --epochs 50 --lr 0.00008 --bsz 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=2  python Prefix_tune.py --dataset e2e --model_mode BiasTune --epochs 50 --lr 0.00008 --bsz 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=2  python Prefix_tune.py --dataset e2e --model_mode PT_plus_FineTune --epochs 50 --lr 0.00008 --bsz 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=2  python Prefix_tune.py --dataset e2e --model_mode PT_plus_BiasTune --epochs 50 --lr 0.00008 --bsz 10 --log_mode No_Log_Bleu

#### ADD

CUDA_VISIBLE_DEVICES=2  python Prefix_tune.py --dataset e2e --model_mode FineTuneEval --epochs 50 --lr 0.00008 --bsz 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=2  python Prefix_tune.py --dataset e2e --model_mode BiasTuneEval --epochs 50 --lr 0.00008 --bsz 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=3  python Prefix_tune.py --dataset e2e --model_mode PT_plus_FineTuneEval --epochs 50 --lr 0.00008 --bsz 10 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=3  python Prefix_tune.py --dataset e2e --model_mode PT_plus_BiasTuneEval --epochs 50 --lr 0.00008 --bsz 10 --log_mode No_Log_Bleu

### Webnlg

CUDA_VISIBLE_DEVICES=3 python Prefix_tune.py --model_mode PrefixModel --epochs 50 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=3 python Prefix_tune.py --model_mode FineTune --epochs 50 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=3 python Prefix_tune.py --model_mode BiasTune --epochs 50 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=3 python Prefix_tune.py --model_mode PT_plus_FineTune --epochs 50 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=3 python Prefix_tune.py --model_mode PT_plus_BiasTune --epochs 50 --log_mode No_Log_Bleu

#### ADD:

CUDA_VISIBLE_DEVICES=4 python Prefix_tune.py --model_mode FineTuneEval --epochs 50 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=4 python Prefix_tune.py --model_mode BiasTuneEval --epochs 50 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=5 python Prefix_tune.py --model_mode PT_plus_FineTuneEval --epochs 50 --log_mode No_Log_Bleu

CUDA_VISIBLE_DEVICES=5 python Prefix_tune.py --model_mode PT_plus_BiasTuneEval --epochs 50 --log_mode No_Log_Bleu

## Log Bleu

python Prefix_tune.py --dataset e2e --model_mode PrefixModel --epochs 5 --lr 0.00008 --bsz 10 --project_name Bleu_dynamic
python Prefix_tune.py --dataset e2e --model_mode FineTune --epochs 5 --lr 0.00008 --bsz 10 --project_name Bleu_dynamic

