# encoding: utf-8
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
import numpy as np
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
from transformers import PreTrainedModel,PreTrainedTokenizer,PreTrainedTokenizerBase,DataCollator
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
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def Norm_params(model):
    # 'decoder_wte.weight', 'decoder_control_trans.0.weight', \
    # 'decoder_control_trans.0.bias', 'decoder_control_trans.2.weight', 'decoder_control_trans.2.bias'
    weight_1_norm = torch.norm(model.state_dict()['decoder_control_trans.0.weight'])
    weight_2_norm = torch.norm(model.state_dict()['decoder_control_trans.2.weight'])
    bias_1_norm = torch.norm(model.state_dict()['decoder_control_trans.0.bias'])
    bias_2_norm = torch.norm(model.state_dict()['decoder_control_trans.2.bias'])
    embed_norm=torch.norm(model.state_dict()['decoder_wte.weight'])
    print(weight_1_norm,weight_2_norm,bias_1_norm,bias_2_norm,embed_norm)
    return weight_1_norm,weight_2_norm,bias_1_norm,bias_2_norm,embed_norm


def Norm_params_finetune(model):
    attn_attn_norms=[]
    attn_proj_norms=[]
    mlp_fc_norms=[]
    mlp_proj_norms=[]
    attn_attn_norms_bias=[]
    attn_proj_norms_bias=[]
    mlp_fc_norms_bias=[]
    mlp_proj_norms_bias=[]
    for name,param in model.named_parameters():
        if name.endswith("attn.c_attn.weight"):
            norm=torch.norm(param).item()
            attn_attn_norms.append(norm)
        elif name.endswith("attn.c_proj.weight"):
            norm=torch.norm(param).item()
            attn_proj_norms.append(norm)
        elif name.endswith("mlp.c_proj.weight"):
            norm=torch.norm(param).item()
            mlp_proj_norms.append(norm)
        elif name.endswith("mlp.c_fc.weight"):
            norm=torch.norm(param).item()
            mlp_fc_norms.append(norm)
#################################################################
        elif name.endswith("attn.c_attn.bias"):
            norm = torch.norm(param).item()
            attn_attn_norms_bias.append(norm)
        elif name.endswith("attn.c_proj.bias"):
            norm = torch.norm(param).item()
            attn_proj_norms_bias.append(norm)
        elif name.endswith("mlp.c_proj.bias"):
            norm = torch.norm(param).item()
            mlp_proj_norms_bias.append(norm)
        elif name.endswith("mlp.c_fc.bias"):
            norm = torch.norm(param).item()
            mlp_fc_norms_bias.append(norm)

    return   attn_attn_norms, attn_proj_norms,  mlp_fc_norms,mlp_proj_norms,attn_attn_norms_bias, attn_proj_norms_bias, mlp_fc_norms_bias,mlp_proj_norms_bias



    #print(weight_1_norm)
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
        self.model=model
        self.tokenizer=tokenizer
        # Normal Args
        self.args=args
        set_seed(self.args.seed)
        if self.args.dataset=='webnlg':
            dataset_path="../data/webnlg_challenge_2017/"
            self.train_dataset_path =dataset_path + "train.json"
            self.valid_dataset_path = dataset_path + "dev.json"

            self.train_dataset = LineByLineWebNLGTextDataset(self.tokenizer, self.train_dataset_path,
                                                        block_size=1024,
                                                        bos_tok=self.tokenizer.bos_token,
                                                        eos_tok=self.tokenizer.eos_token, )
            self.valid_dataset = LineByLineWebNLGTextDataset(self.tokenizer, self.valid_dataset_path,
                                                        block_size=1024,
                                                        bos_tok=self.tokenizer.bos_token,
                                                        eos_tok=self.tokenizer.eos_token, )
            self.test_dataset_path =dataset_path +'test.json'

        elif self.args.dataset=='dart':
            dataset_path="../data/dart/dart-v1.1.1-full-"
            self.train_dataset_path = dataset_path + "train.json"
            self.valid_dataset_path = dataset_path + "dev.json"

            self.train_dataset = LineByLineTriplesTextDataset(self.tokenizer, self.train_dataset_path,
                                                        block_size=1024,
                                                        bos_tok=self.tokenizer.bos_token,
                                                        eos_tok=self.tokenizer.eos_token, )
            self.valid_dataset = LineByLineTriplesTextDataset(self.tokenizer, self.valid_dataset_path,
                                                        block_size=1024,
                                                        bos_tok=self.tokenizer.bos_token,
                                                        eos_tok=self.tokenizer.eos_token, )
            self.test_dataset_path =dataset_path +'test.json'
        elif self.args.dataset=='e2e':
            dataset_path="../data/e2e_data/src1_"
            self.train_dataset_path = dataset_path + "train.txt"
            self.valid_dataset_path = dataset_path + "valid.txt"
            self.test_dataset_path =dataset_path +'test.txt'

            self.train_dataset = LineByLineData2TextTextDataset(self.tokenizer, self.train_dataset_path,
                                                        block_size=1024,
                                                        bos_tok=self.tokenizer.bos_token,
                                                        eos_tok=self.tokenizer.eos_token,lowdata_token=None, )
            self.valid_dataset = LineByLineData2TextTextDataset(self.tokenizer, self.valid_dataset_path,
                                                        block_size=1024,
                                                        bos_tok=self.tokenizer.bos_token,
                                                        eos_tok=self.tokenizer.eos_token,lowdata_token=None ,)
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

    def get_train_dataloader(self,train_dataset) -> DataLoader:

        train_sampler = self._get_train_sampler()
        #train_sampler = self._get_eval_sampler(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self.args.bsz, # modified
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.num_workers,
            worker_init_fn=np.random.seed(self.args.seed)
        )

    def _get_eval_sampler(self,eval_dataset) -> Optional[torch.utils.data.sampler.Sampler]:
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
            #worker_init_fn=np.random.seed(self.args.seed)
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
            "params": [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
            "weight_decay": self.args.weight_decay,
        },
        {
            "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
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

    def train(self,lresume_from_checkpoint=None,):
        layer_attn_attn_norms = []
        layer_attn_proj_norms = []
        layer_mlp_fc_norms = []
        layer_mlp_proj_norms = []
        name=self.args.dataset+"_"+self.args.model_mode+\
            "_epochs_"+str(self.args.epochs)+"_lr_"+str(self.args.lr)+"_"+self.args.log_mode
        wandb.init(
            project=self.args.project_name,
            name=name,
            # entity='Shuailong Zhu',
        )
        train_dataloader=self.get_train_dataloader(self.train_dataset)
        global_step = 0
        tot_loss = 0
        log_loss = 0

        total_step = self.args.epochs * len(train_dataloader)

        self.create_optimizer_and_scheduler(total_step)
        #dict=read_webnlg_files(self.test_dataset_path,self.tokenizer)

        self.model.to(self.args.device)
        #total_time=0.0
        fd_total_time=0.0
        bd_total_time=0.0
        update_total_time=0.0
        best_log_step=0
        best_eval_ppl=0
        for epoch in range(int(self.args.epochs)):
            # self.model.eval()
            # self.model.pretrained_model.eval()
            #set_seed(self.args.seed)  这里加不加属实不影响
            if self.model.model_mode in ["PrefixModel"]:
                self.model.eval()
            elif self.model.model_mode in ["FineTune","BiasTune","PT_plus_BiasTune","PT_plus_FineTune"]:
                self.model.pretrained_model.train()
            #self.model.pretrained_model.eval()

            for step, inputs in enumerate(train_dataloader):
                input_=self._prepare_inputs(inputs)

                global_step += 1

                fd_start_time=time.time()
                out,loss = self.model(**input_)
                fd_end_time=time.time()

                bd_start_time=time.time()
                loss.backward()
                bd_end_time=time.time()

                tot_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                update_start_time=time.time()
                self.optimizer.step()
                update_end_time=time.time()

                fd_total_time+=fd_end_time-fd_start_time
                bd_total_time+=bd_end_time-bd_start_time
                update_total_time+=update_end_time-update_start_time

                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                if global_step % self.args.step_size == 0:
                    t=global_step / self.args.step_size
                    # if global_step==1500:
                    #     break


                    #self.save_prefix_()
                    loss_=(tot_loss - log_loss) / self.args.step_size
                    eval_ppl = self.evaluation()
                    if best_eval_ppl==0:
                        best_eval_ppl=eval_ppl
                    elif eval_ppl<best_eval_ppl:
                        best_log_step=global_step


                    print("Fd_time {} Bd_Time {} Update_Time {}  Epoch {}, global_step {} average loss: {} lr: {} eval_ppl: {}".format(fd_total_time,bd_total_time,update_total_time,
                        epoch, global_step,(tot_loss - log_loss) / self.args.step_size,self.lr_scheduler.get_last_lr()[0],eval_ppl),flush=True)
                    log_loss = tot_loss




                    if self.args.log_mode=="No_Log_Bleu":
                        weight_1_norm, weight_2_norm, bias_1_norm, bias_2_norm, embed_norm = Norm_params(self.model)
                        wandb.log({
                            "step": step,
                            "loss":loss_,
                            "eval_ppl":eval_ppl,
                            "total_time":fd_total_time+bd_total_time+update_total_time,
                            })

                    elif self.args.log_mode=="Log_Bleu":
                        gen_s, refs_s = self.model.generate_to_files(current_dataset_path=self.test_dataset_path)
                        sacrebleu_score=evaluate_bleu(gen_s,refs_s,"sacrebleu")
                        bleu_score=evaluate_bleu(gen_s,refs_s,"bleu")
                        wandb.log({
                            "step": step,
                            "loss":loss_,
                            "bleu_score":bleu_score,
                            "sacrebleu_score":sacrebleu_score,
                            "eval_ppl":eval_ppl,
                            "total_time":fd_total_time+bd_total_time+update_total_time,
                            })

        print("Best Log Step:", best_log_step)
        print("Best Eval Ppl:",best_eval_ppl)

        # Save Model

        Save_path="ckpt/"+str(self.args.dataset)+\
        "_"+str(self.args.model_mode)+"_"+str(self.args.epochs)+"_"+str(self.args.lr)+".ckpt"
        self.save_prefix_or_ptm(Save_path)
        # Generate
        Write_path="generation/"+str(self.args.dataset)+\
        "_"+str(self.args.model_mode)+"_"+str(self.args.epochs)+"_"+str(self.args.lr)+".txt"

        gen_s, refs_s = self.model.generate_to_files(current_dataset_path=self.test_dataset_path,write_path=Write_path)

        # Scoring
        # os.system("cd ../evaluation")
        # os.system("./run_eval_on_webnlg.sh > ../OnepunchPrompt/bleu_tmp.txt")
        # os.system("cd ../OnepunchPrompt")
        # if self.args.dataset=="webnlg":
        #     with open("bleu_tmp.txt") as f:
        #         line=f.readline()
        #         while line:
        #             T=line.split("BLEU ALL:")
        #             if len(T)>1:
        #                 bleu_official=float(T[1])
        #                 break
        #             line=f.readline()
        #     print("Final Officla Bleu",bleu_official)

        # sacrebleu_score=evaluate_bleu(gen_s,refs_s,"sacrebleu")
        # print("Final Gen SacreBleu",sacrebleu_score)
        # bleu_score=evaluate_bleu(gen_s,refs_s,"bleu")
        # print("Final Gen Bleu",bleu_score)

        


    def evaluation(self):
        valid_dataloader = self.get_eval_dataloader(self.valid_dataset)
        loss_s=[]
        for step, inputs in enumerate(valid_dataloader):
            ### TRIAL ###
            # print(self.model.state_dict().keys())

            input_ = self._prepare_inputs(inputs)

            # self.load_prefix_debug()  有问题，表示会干扰；
            self.model.to(self.args.device)
            # self.load_prefix_debug()
            # self.model.generate_to_files(self.test_dataset_path)

            out, loss = self.model(**input_)
            loss_s.append(loss.item())
        loss_s=torch.tensor(loss_s)
        return loss_s.mean()
    def save_prefix_or_ptm(self,save_path):
        if self.args.model_mode=="PrefixModel":
            torch.save({
            "prefix_control":self.model.decoder_control_trans.state_dict(),
            "prefix_wte":self.model.decoder_wte.state_dict()}, save_path)
        elif self.args.model_mode=="FineTune":
            torch.save({
                "ptm":self.model.pretrained_model.state_dict(),
            } ,save_path)

    def load_prefix_or_ptm(self,load_path):
        checkpoint = torch.load(load_path)
        if self.args.model_mode=="PrefixModel":
            
            self.model.decoder_control_trans.load_state_dict(checkpoint['prefix_control'])
            self.model.decoder_wte.load_state_dict(checkpoint['prefix_wte'])
        elif self.model_mode=="FineTune":
            self.model.pretrained_model.load_state_dict(checkpoint["ptm"])
    
    # def load_prefix_debug(self,load_path='prefix_lisa.ckpt'):
    #     #odict_keys(['wte.weight', 'control_trans.0.weight', 'control_trans.0.bias', 'control_trans.2.weight',
    #     #'control_trans.2.bias'])
    #     model_dict = self.model.state_dict()
    #     checkpoint = torch.load(load_path)
    #     #self.model.load_state_dict(load_path,strict=False)
    #     model_dict['decoder_wte.weight']=checkpoint['prefix']['wte.weight']
    #     model_dict['decoder_control_trans.0.weight']=checkpoint['prefix']['control_trans.0.weight']
    #     model_dict['decoder_control_trans.2.weight'] = checkpoint['prefix']['control_trans.2.weight']
    #     #print(model_dict['decoder_control_trans.0.bias']==checkpoint['prefix']['control_trans.0.bias'])
    #     model_dict['decoder_control_trans.0.bias']=checkpoint['prefix']['control_trans.0.bias']
    #     model_dict['decoder_control_trans.2.bias'] = checkpoint['prefix']['control_trans.2.bias']
    #     self.model.load_state_dict(model_dict,strict=False)








