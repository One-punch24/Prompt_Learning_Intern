# encoding: utf-8
import argparse
import torch
from transformers import AutoModel,AutoTokenizer,AutoModelWithLMHead,GPT2Tokenizer,GPT2LMHeadModel
from transformers import Trainer
from utils import Model_Class

import wandb
from prompt_models import PrefixTuningModel

from utils import Model_Class
from utils import *
from Prefixtrainer import PrefixTrainer
from transformers import set_seed
import _locale
import os
_locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])



parser = argparse.ArgumentParser("Prefix Tuning")

# There are several types of paths: dataset_path, pretrained_model_path(cache_dir), trained model out_dir, gen_dir
parser.add_argument("--total_step",type=int,default=1000)
parser.add_argument("--model", type=str, default='gpt2')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='gpt2-medium')

# parser.add_argument("--cache_dir",default='D:/Github/REPO/MODELS/gpt2-medium-s3/')  # the location of gpt2 model
# parser.add_argument("--dataset_path",default="D:/Github/REPO/DATA/webnlg_challenge_2017/")
# parser.add_argument("--out_dir",default='D:/Github/REPO/CHECKPOINT/PrefixTuning')  # the location of gpt2 model
# parser.add_argument("--gen_dir",default="D:/Github/REPO/Gen/PrefixGeneration")

parser.add_argument("--cache_dir",default='../model_ckpt/gpt2-medium-s3/')  # the location of gpt2 model
parser.add_argument("--dataset",default='webnlg',help="dart or webnlg or e2e")
# parser.add_argument("--dataset_path",default="../data/webnlg_challenge_2017/")
parser.add_argument("--out_dir",default='prefix_model_new/')  # the location of gpt2 model
parser.add_argument("--gen_dir",default="gen_dir/")
parser.add_argument("--model_mode",default="PrefixModel")
parser.add_argument("--save_path",type=str,default="model_ckpt/")
parser.add_argument("--bsz",type=int,default=5)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--weight_decay",type=float,default=0.0)
parser.add_argument("--adam_epsilon",type=float,default=1e-8)
parser.add_argument("--epochs",type=float,default=5)
parser.add_argument("--num_token",type=int,default=5)

parser.add_argument("--plm_eval_mode", action="store_true")


parser.add_argument("--local_rank",type=int,default=-1)

parser.add_argument("--num_workers",type=int,default=0)
parser.add_argument("--seed",type=int,default=101)
parser.add_argument("--prefix_save_path",type=str,default=None)
parser.add_argument("--prefix_load_path",type=str,default=None)
parser.add_argument("--write_path",type=str,default="Generation.txt")
parser.add_argument("--sel",type=int,default=0)


parser.add_argument("--step_size",type=int,default=500)
parser.add_argument("--project_name",default="Webnlg_PT_FT")
parser.add_argument("--start_wandb_log", type=int, default=1)

args = parser.parse_args()
args.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)
# args.device='cpu'
'''
Code includes 3 parts: model, dataset, dealing(train,evaluate,test).
'''

'''
PART 1: Dataset & Datacollactor
'''
os.system("wandb login 1457ccb79dfb6d28df18319ee5e63bfcaad44d5c")


# test_dataset=LineByLineWebNLGTextDataset(tokenizer,args.test_dataset_path,
#                                           blocksize=-1,
#                                           bos_tok=tokenizer.bos_token,
#                                           eos_tok=tokenizer.eos_token,)


'''
PART 2: Model Definition and some initialization 
'''

# wandb.log("weather_sample" : wandb.plot.line_series(
#   xs=xs,
#   ys=ys,
#   keys=columns,
#   title="Weather Metrics")})

pretrained_model=GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=args.cache_dir,cache_dir=args.cache_dir)
tokenizer=GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=args.cache_dir,cache_dir=args.cache_dir)
set_seed(args.seed)

#out=pretrained_model.generate(input_ids=torch.tensor([2,3,4]))
#print(out)
num_added_tokens = tokenizer.add_special_tokens(
    {'pad_token': '[PAD]'})
print(type(tokenizer))
'''
文件里存的是那个啥，lisa prefix 的 embedding. 50258维度
'''
pretrained_model.resize_token_embeddings(len(tokenizer))
# weights_af=pretrained_model.state_dict()['transformer.wte.weight'][50527,:]
# torch.save({"weightemb":pretrained_model.state_dict()['transformer.wte.weight'],},"weight_set_bf.ckpt")
#ckpt=torch.load("weight_set_bf.ckpt")
#weights_bf=ckpt['weightemb']
#print(weights_bf.shape)
#weights_bf=weights_bf[50257]

#weights_af=pretrained_model.state_dict()['transformer.wte.weight'][50257]
#print(weights_bf==weights_af)
# print(type(pretrained_model))
#print(tokenizer.add_prefix_space)
PrefixModel=PrefixTuningModel(
    pretrained_model=pretrained_model,
    args=args,
    model_mode=args.model_mode,
    num_token=args.num_token,
    bsz=args.bsz,
)
print(PrefixModel.state_dict().keys())


# PrefixModel.to(args.device)
# PrefixModel.generate_to_files(test_dataset_path)
Prefix_trainer=PrefixTrainer(
    model=PrefixModel,
    tokenizer=tokenizer,
    args=args,
    )

Prefix_trainer.train()
# gen_s,refs_S=PrefixModel.generate_to_files(current_dataset_path=test_dataset_path)

# gen_s,refs_S=PrefixModel.get_past_key_values(PrefixModel.test_dataset_path)
#pretrained_model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# #PreTrainedTokenizerFast(name_or_path='gpt2-medium', vocab_size=50257, model_max_len=1024, is_fast=True, padding_side='right',
# #special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'})
# #print(config)
# #print(model.config)
# print(pretrained_model.config.is_encoder_decoder)  # False
# model=PrefixTuningModel(pretrained_model=pretrained_model,
#                         tokenizer=tokenizer,
#                         )

# The pipeline of huggingface is more like a generator in Prefix tuning, it finishes the work of generation


'''
Train, Evaluate, Test
'''
# trainer=PrefixTrainer()
# eval_output = trainer.evaluate(train_dataset)


# def Training_Args_Setup(args):
#     args.beta1=0.9
#     args.beta2=0.999
