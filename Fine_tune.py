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
parser.add_argument("--model", type=str, default='gpt2')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='gpt2-medium')
# parser.add_argument("--cache_dir",default='D:/Github/REPO/MODELS/gpt2-medium-s3/')  # the location of gpt2 model
# parser.add_argument("--dataset_path",default="D:/Github/REPO/DATA/webnlg_challenge_2017/")
# parser.add_argument("--out_dir",default='D:/Github/REPO/CHECKPOINT/PrefixTuning')  # the location of gpt2 model
# parser.add_argument("--gen_dir",default="D:/Github/REPO/Gen/PrefixGeneration")

parser.add_argument("--cache_dir",default='../model_ckpt/gpt2-medium-s3/')  # the location of gpt2 model
parser.add_argument("--dataset_path",default="../data/webnlg_challenge_2017/")
parser.add_argument("--out_dir",default='prefix_model_new/')  # the location of gpt2 model
parser.add_argument("--gen_dir",default="gen_dir/")

parser.add_argument("--save_path",type=str,default="model_ckpt/")

parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--bsz",type=int,default=5)
parser.add_argument("--weight_decay",type=float,default=0.0)
parser.add_argument("--adam_epsilon",type=float,default=1e-8)
parser.add_argument("--epochs",type=float,default=5)
parser.add_argument("--num_token",default=5)

parser.add_argument("--plm_eval_mode", action="store_true")


parser.add_argument("--local_rank",type=int,default=-1)

parser.add_argument("--num_workers",type=int,default=0)
parser.add_argument("--seed",type=int,default=101)
parser.add_argument("--prefix_save_path",type=str,default=None)
parser.add_argument("--prefix_load_path",type=str,default=None)
parser.add_argument("--write_path",type=str,default="Generation.txt")
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

'''
PART 2: Model Definition and some initialization 
'''
pretrained_model=GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=args.cache_dir,cache_dir=args.cache_dir)
tokenizer=GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=args.cache_dir,cache_dir=args.cache_dir)
set_seed(args.seed)

num_added_tokens = tokenizer.add_special_tokens(
    {'pad_token': '[PAD]'})
print(type(tokenizer))
'''
文件里存的是那个啥，lisa prefix 的 embedding. 50258维度
'''
pretrained_model.resize_token_embeddings(len(tokenizer))

'''
定义 Trainer
'''
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from packaging import version
import wandb
import collections
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerBase, DataCollator
from transformers import set_seed
from utils import *

'''
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):

'''


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


# def Norm_params(model):
#     # 'decoder_wte.weight', 'decoder_control_trans.0.weight', \
#     # 'decoder_control_trans.0.bias', 'decoder_control_trans.2.weight', 'decoder_control_trans.2.bias'
#     weight_1_norm = torch.norm(model.state_dict()['decoder_control_trans.0.weight'])
#     weight_2_norm = torch.norm(model.state_dict()['decoder_control_trans.2.weight'])
#     bias_1_norm = torch.norm(model.state_dict()['decoder_control_trans.0.bias'])
#     bias_2_norm = torch.norm(model.state_dict()['decoder_control_trans.2.bias'])
#     embed_norm = torch.norm(model.state_dict()['decoder_wte.weight'])
#     print(weight_1_norm, weight_2_norm, bias_1_norm, bias_2_norm, embed_norm)
#     return weight_1_norm, weight_2_norm, bias_1_norm, bias_2_norm, embed_norm

    # print(weight_1_norm)


class PrefixTrainer:
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module],
            args,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            # model_init: Callable[[], PreTrainedModel] = None,
            # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            # callbacks: Optional[List[TrainerCallback]] = None,
            # optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        self.model = model
        self.tokenizer = tokenizer
        # Normal Args
        self.args = args
        set_seed(self.args.seed)
        self.train_dataset_path = args.dataset_path + "train.json"
        self.valid_dataset_path = args.dataset_path + "dev.json"
        self.test_dataset_path = args.dataset_path + "test.json"

        self.train_dataset = LineByLineWebNLGTextDataset(self.tokenizer, self.train_dataset_path,
                                                         block_size=1024,
                                                         bos_tok=self.tokenizer.bos_token,
                                                         eos_tok=self.tokenizer.eos_token, )
        self.valid_dataset = LineByLineWebNLGTextDataset(self.tokenizer, self.valid_dataset_path,
                                                         block_size=1024,
                                                         bos_tok=self.tokenizer.bos_token,
                                                         eos_tok=self.tokenizer.eos_token, )
        self.test_dataset = LineByLineWebNLGTextDataset(self.tokenizer, self.test_dataset_path,
                                                        block_size=1024,
                                                        bos_tok=self.tokenizer.bos_token,
                                                        eos_tok=self.tokenizer.eos_token, )
        self.data_collator = DataCollatorForData2TextLanguageModeling(
            tokenizer=self.tokenizer, mlm=False, mlm_probability=0.15,
            format_mode='cat'
        )

    '''
    Part I : Train Data Loader
    '''

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        return (
            RandomSampler(self.train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(self.train_dataset)
        )

    def get_train_dataloader(self, train_dataset) -> DataLoader:

        # train_sampler = self._get_train_sampler()
        train_sampler = self._get_eval_sampler(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self.args.bsz,  # modified
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.num_workers,
            worker_init_fn=np.random.seed(self.args.seed)
        )

    def _get_eval_sampler(self, eval_dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        if self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_sampler = self._get_eval_sampler(eval_dataset)
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.bsz,  # modified
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.num_workers,
            # worker_init_fn=np.random.seed(self.args.seed)
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if
                           (not any(nd in n for nd in no_decay)) and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.lr,
            eps=self.args.adam_epsilon,
        )
        # for n, p in self.model.named_parameters():
        #     print(n,p.requires_grad)
        print(self.optimizer.state_dict())
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 0, num_training_steps=num_training_steps,
        )

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        return inputs

    def train(self, lresume_from_checkpoint=None, ):
        wandb.init(
            project="Prefix_norm",
            # name="Prefix_norm",
            # entity='Shuailong Zhu',
        )
        train_dataloader = self.get_train_dataloader(self.train_dataset)
        global_step = 0
        tot_loss = 0
        log_loss = 0

        total_step = self.args.epochs * len(train_dataloader)

        self.create_optimizer_and_scheduler(total_step)
        # dict=read_webnlg_files(self.test_dataset_path,self.tokenizer)

        for epoch in range(int(self.args.epochs)):
            # self.model.eval()
            # self.model.pretrained_model.eval()
            # set_seed(self.args.seed)  这里加不加属实不影响
            self.model.eval()
            # self.model.pretrained_model.eval()

            for step, inputs in enumerate(train_dataloader):
                ### TRIAL ###
                # print(self.model.state_dict().keys())

                input_ = self._prepare_inputs(inputs)

                # self.load_prefix_debug()  有问题，表示会干扰；
                self.model.to(self.args.device)
                # self.load_prefix_debug()
                # self.model.generate_to_files(self.test_dataset_path)
                global_step += 1

                # if use_cuda:
                #     inputs = inputs.cuda()
                #      input_['input_ids']=torch.tensor([[  220,   930,   317,   283,  7537,    62, 16170,   634,  1058,  1748,
                #  50,  8520,  1058,   366,    32,   283,  7537,    11, 16490,     1,
                # 220, 50256,   383,   317,   283,  7537,   318,   262,  9003,   286,
                # 317,   283,  7537,    11, 16490,    13,   220, 50256]])
                out, loss = self.model(**input_)

                # loss=o[0]
                loss.backward()
                tot_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                if global_step % 500 == 0:
                    # self.save_prefix_()
                    print("Epoch {}, global_step {} average loss: {} lr: {}".format(
                        epoch, global_step, (tot_loss - log_loss) / 500, self.lr_scheduler.get_last_lr()[0]),
                        flush=True)
                    log_loss = tot_loss
                    weight_1_norm, weight_2_norm, bias_1_norm, bias_2_norm, embed_norm = Norm_params(self.model)
                    wandb.log({"weight_1": weight_1_norm,
                               "weight_2": weight_2_norm,
                               "bias_1": bias_1_norm,
                               "bias_2": bias_2_norm,
                               "embed_weight": embed_norm,
                               "step": step,
                               })
                    print("Norm", weight_1_norm, weight_2_norm, bias_1_norm, bias_2_norm, embed_norm)

        self.save_prefix_()

    def evaluation(self):
        pass

    def save_prefix_(self, save_path=None):
        if save_path == None:
            save_path = self.args.save_path + 'prefix-new_transformers.ckpt'
            torch.save({
                "prefix_control": self.model.decoder_control_trans.state_dict(),
                "prefix_wte": self.model.decoder_wte.state_dict()}, save_path)

    def load_prefix_(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.decoder_control_trans.load_state_dict(checkpoint['model_state_dict'])

    def load_prefix_debug(self, load_path='prefix_lisa.ckpt'):
        # odict_keys(['wte.weight', 'control_trans.0.weight', 'control_trans.0.bias', 'control_trans.2.weight',
        # 'control_trans.2.bias'])
        model_dict = self.model.state_dict()
        checkpoint = torch.load(load_path)
        # self.model.load_state_dict(load_path,strict=False)
        model_dict['decoder_wte.weight'] = checkpoint['prefix']['wte.weight']
        model_dict['decoder_control_trans.0.weight'] = checkpoint['prefix']['control_trans.0.weight']
        model_dict['decoder_control_trans.2.weight'] = checkpoint['prefix']['control_trans.2.weight']
        # print(model_dict['decoder_control_trans.0.bias']==checkpoint['prefix']['control_trans.0.bias'])
        model_dict['decoder_control_trans.0.bias'] = checkpoint['prefix']['control_trans.0.bias']
        model_dict['decoder_control_trans.2.bias'] = checkpoint['prefix']['control_trans.2.bias']
        self.model.load_state_dict(model_dict, strict=False)


Fine_Tune_trainer=PrefixTrainer(
    model=pretrained_model,
    tokenizer=tokenizer,
    args=args,
    )

Fine_Tune_trainer.train()
PrefixModel.get_past_key_values(PrefixModel.test_dataset_path)

