# encoding: utf-8
import torch
import os
from torch import nn
from typing import *
from torch.nn import CrossEntropyLoss
import json
import logging
import transformers
import copy
from transformers import PretrainedConfig
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
#from transformers.generation_utils import generate

from utils import *

from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from utils.data_util import read_e2e_files

'''
GPT2Config {
  "_name_or_path": "gpt2-medium",
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 1024,
  "n_head": 16,
  "n_inner": null,
  "n_layer": 24,
  "n_positions": 1024,
  "n_special": 0,
  "predict_special_tokens": true,
  "reorder_and_upcast_attn": false,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": false,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "transformers_version": "4.12.5",
  "use_cache": true,
  "vocab_size": 50257
}
'''

# GPT2 预留了 special toeknize embedding
# GPT2 中是有generate 函数的，可以想办法wrap一下

class PrefixTuningModel(nn.Module):
    # group_prefix, group_input_tokens
    def __init__(self,
                 pretrained_model:PreTrainedModel,
                 #tokenizer: PreTrainedTokenizer,
                 num_token,

                 args,
                 model_mode="FineTune",
                 bsz=5,
                 prefix_load_path=None,
                 gen_args=None,
                 prefix_save_path=None,
                 placeholder_mapping= None,
                 mask_token: str = '<mask>',
                 mid_dim: Optional[int] =  512,
                 prefix_dropout: Optional[float] = 0.0,

                 ):
        super().__init__()
        self.model_mode=model_mode
        self.bsz=bsz
        self.args=args
        self.pretrained_model=pretrained_model
        #self.tokenizer=tokenizer
        self.num_token=num_token
        self.mask_token=mask_token
        self.config=pretrained_model.config

        raw_embedding = pretrained_model.get_input_embeddings()

        self.n_decoder_layer = self.config.n_layer
        self.n_embd = self.config.n_embd
        self.n_head = self.config.n_head

        # Match Mapping Network
        self.match_n_decoder_layer = self.n_decoder_layer
        self.mid_dim = mid_dim
        self.match_n_head = self.n_head
        self.match_n_embd = self.n_embd // self.n_head


        self.prefix_dropout = prefix_dropout
        self.dropout = nn.Dropout(self.prefix_dropout)
        self.generate_parameters()

        # Initialization using fine-trained prefix embedding
        if prefix_load_path:
            self.load(prefix_load_path)
        # freeze pretrained parameters
        if self.model_mode=="PrefixModel":
            self.freeze_pretrained_model()
        elif self.model_mode in ['FineTune',"PT_plus_FineTune",'FineTuneEval',"PT_plus_FineTuneEval"]:
            self.activate_pretrained_model()
        elif self.model_mode in ["PT_plus_BiasTune","BiasTune","PT_plus_BiasTuneEval","BiasTuneEval"]:
            self.activate_PTM_Bias()
        # generation configuration
        if gen_args==None:
            self.generation_arguments = {
            "max_length": 100,
            "max_new_tokens": None,
            "min_length": 5,
            "temperature": 1.0,
            "do_sample": False,
            "top_k": 0,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "num_beams": 5,
            "bad_words_ids": [[628], [198]],
            "num_return_sequences":1,
            }



    def generate_parameters(self) -> None:
        r"""
        Generate parameters needed for new tokens' embedding in Prefix Tuning
        """

        self.input_tokens = (torch.arange(self.num_token).long()).to(self.args.device)
        if (not self.config.is_encoder_decoder):
            self.decoder_wte = nn.Embedding(self.num_token, self.n_embd)
            self.decoder_control_trans = nn.Sequential(
                # in_features, out_features
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                # nn.Linear(self.mid_dim, self.mid_dim),
                # nn.Tanh(),
                nn.Linear(self.mid_dim, self.n_decoder_layer * 2 * self.n_embd))

    def get_past_key_values(self, batch_size=1):
        pvs = []
        if (not self.config.is_encoder_decoder) or self.using_decoder_past_key_values:
            decoder_input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1).to(self.args.device)
            #print(decoder_input_tokens)
            decoder_temp_control = self.decoder_wte(decoder_input_tokens)
            #print(decoder_temp_control)
            decoder_past_key_values = self.decoder_control_trans(decoder_temp_control) #bsz, seqlen, layer*emb*2
            _, decoder_seqlen, _ = decoder_past_key_values.shape
            #print(decoder_seqlen)
            decoder_past_key_values = decoder_past_key_values.view(batch_size, decoder_seqlen, self.match_n_decoder_layer * 2, self.match_n_head,
                                                self.match_n_embd)
            decoder_past_key_values = self.dropout(decoder_past_key_values)
            # past key and values regular shape: (2, batch_size, num_heads, sequence_length, embed_size_per_head)).

            # match_n_decoder_layer * 2, bsz, match_n_head, num_tokens, match_n_embed
            decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            #pvs.append(decoder_past_key_values)
        else:
            pvs.append(None)
        return decoder_past_key_values
    # Before finishing this function, we can analysis the GPT input parameters
    # (input_ids = None past_key_values = None attention_mask = None token_type_ids =
    # Noneposition_ids = Nonehead_mask = Noneinputs_embeds = Noneencoder_hidden_states =
    # Noneencoder_attention_mask = Nonelabels = Noneuse_cache = Noneoutput_attentions =
    # Noneoutput_hidden_states = Nonereturn_dict = None) ⟶
    # CausalLMOutputWithCrossAttentions or tuple(torch.FloatTensor)

    def forward(
            self,
            # group_prefix,
            # group_input_tokens,
            input_ids=None,
            labels=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids = None,
            head_mask = None,
            inputs_embeds = None,
            encoder_hidden_states =None,
            encoder_attention_mask = None,
            use_cache = None,
            output_attentions =None,
            output_hidden_states = None,
            src_attn=None,
            tgt_attn=None,
            src=None,
            return_dict = None):
        if self.model_mode  in ['PrefixModel',"PT_plus_FineTune","PT_plus_BiasTune","PT_plus_FineTuneEval","PT_plus_BiasTuneEval"]:
            past_key_values=self.get_past_key_values(input_ids.size()[0])
        elif self.model_mode in ['FineTune',"BiasTune",'FineTuneEval',"BiasTuneEval"]:
            past_key_values=None    #group_prefix group_input_tokens

        output=self.pretrained_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids =position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            encoder_hidden_states =encoder_hidden_states,
            encoder_attention_mask = encoder_attention_mask,
            labels = labels,
            use_cache = use_cache,
            output_attentions =output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict)

        loss=self.compute_loss(loss_ids=labels, output=output)
        return output,loss

    # 看一下这个应该怎么写： wrap gpt2 的generate 和lisa code 中的run_generation部分)
    def freeze_pretrained_model(self):
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
    def activate_PTM_Bias(self):
        for key,param in self.pretrained_model.named_parameters():
            if key.endswith('bias'):
                param.requires_grad = True
            else:
                param.requires_grad = False   


    def activate_pretrained_model(self):
        for param in self.pretrained_model.parameters():
            param.requires_grad = True


        # Open Prompt Implementation: Pay attention that the shift label mask is
    def shift_logits_and_labels(self,
                                logits,
                                labels):
        shift_logits = logits[..., :-1, :].contiguous()
        #shift_loss_ids = loss_ids[..., 1:].contiguous()
        shift_input_ids = labels[..., 1:].contiguous()
        #shift_input_ids = torch.where(shift_loss_ids > 0, shift_input_ids, -100)
        return shift_logits, shift_input_ids

    # def _forward(self, batch: Union[Dict, InputFeatures]) -> torch.Tensor:
    #     r"""
    #     This is the forward method of the training of generation in prompt-learning framework.
    #
    #     Args:
    #         batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
    #
    #     Returns:
    #         loss(:obj:torch.Tensor): The loss of the current generation procedure.
    #     """
    #     if self.config.is_encoder_decoder:
    #         reference_ids = batch['decoder_input_ids']
    #     else:
    #         reference_ids = batch['input_ids']  # in case in some template, these field is dropped
    #     outputs = self.prompt_model(batch)
    #     logits = outputs.logits
    #     logits, labels = self.shift_logits_and_labels(logits, loss_ids) #loss_ids comes from datacollator
    #     batch_size, seq_len, vocab_size = logits.shape
    #     loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    #     loss = loss.view(batch_size, -1).sum(dim=-1)  # TODO support more objectives
    #     loss = loss.mean()
    #     return loss
    def compute_loss(self,output,loss_ids,objective=None):
        # objective_mode=1 in Lisa's code
        # loss_fct = CrossEntropyLoss(reduction='none')
        # bsz, seqlen, vocab_size = shift_logits.shape
        # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # loss = loss.view(bsz, seqlen).sum(dim=-1)
        logits=output.logits
        shifted_logits, shifted_labels = self.shift_logits_and_labels(logits, loss_ids)
        batch_size, seq_len, vocab_size = logits.shape
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        loss = loss.view(batch_size, -1).sum(dim=-1)
        # TODO support more objectives
        loss = loss.mean()
        return loss



    

    def generate(self,
                 input_ids,
                 ptm,
                 group_prefix=None,
                 group_input_tokens=None,
                 ):
        prefix_prompt=self.get_past_key_values(batch_size=1)
        prefix_prompt = [x.expand(-1, self.generation_arguments['num_beams'], -1, -1, -1) for x in prefix_prompt]

        if self.model_mode  in ['PrefixModel',"PT_plus_FineTune","PT_plus_BiasTune","PT_plus_FineTuneEval","PT_plus_BiasTuneEval"]:
            past_key_values=prefix_prompt
        elif self.model_mode in ['FineTune',"BiasTune",'FineTuneEval',"BiasTuneEval"]:
            past_key_values=None   

        output_sentences=ptm.generate(
            input_ids=input_ids,
            emb_match=None,
            past_key_values= past_key_values,
            max_length=self.generation_arguments['max_length']+input_ids.size()[1],
            min_length=5,
            temperature=self.generation_arguments['temperature'],
            top_k=self.generation_arguments['top_k'],
            top_p=self.generation_arguments['top_p'],
            repetition_penalty=self.generation_arguments['repetition_penalty'],
            do_sample=self.generation_arguments['do_sample'],
            num_beams=self.generation_arguments['num_beams'],
            bad_words_ids=self.generation_arguments['bad_words_ids'],
            num_return_sequences=self.generation_arguments['num_return_sequences'],
        )

        return output_sentences




    def generate_to_files(self,current_dataset_path,write_path=None,previous_output_file=None,):

        if write_path==None:
            write_path=self.args.write_path
        out=[]
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path="gpt2-medium",
                                                  cache_dir="gpt2-medium-s3/")
        tokenizer.pad_token=tokenizer.eos_token
        if self.args.dataset=="webnlg":
            full_dict=read_webnlg_files(current_dataset_path,tokenizer)
        elif self.args.dataset=="e2e":
            full_dict=read_e2e_files(current_dataset_path,tokenizer)
        elif self.args.dataset=='dart':
            full_dict=read_triples_files(current_dataset_path,tokenizer)
        print(len(tokenizer))
        # if self.args.model_mode=="PrefixModel":
        #     del self.pretrained_model
        #     self.pretrained_model=GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path=self.args.model_name_or_path,cache_dir=self.args.cache_dir)
        #     self.pretrained_model.to(self.args.device)
        

        previous_pretrained_model=copy.deepcopy(self.pretrained_model)
        previous_pretrained_model.resize_token_embeddings(len(tokenizer))
        self.to(self.args.device)
        refs_s=[]
        for idx, text_array in enumerate(full_dict):
            
            #prefix = self.args.prefix if self.args.prefix else self.args.padding_text
            if self.args.dataset=="webnlg":
                text=text_array[0]
            elif self.args.dataset=="e2e":
                text=text_array
            elif self.args.dataset=="dart":
                text=text_array[0]
            refs=full_dict[text_array]
            refs_s.append(refs)
            input_ids = tokenizer.encode( text, add_special_tokens=False, return_tensors="pt")
            input_ids = input_ids.to(self.args.device)

            out_sentence=self.generate(input_ids,ptm=previous_pretrained_model)[0]
            # print(out_sentence)
            print("=== GENERATED SEQUENCE {} ===".format(idx + 1))
            # args.stop_token = tokenizer.eos_token
            generated_sequence = out_sentence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            '''
            # print(text)
            # text_output = text[len(generated_sequence):]
            # idx_end = text_output.find(tokenizer.eos_token)
            # if idx_end >= 0:
            #     text_output = text_output[:idx_end]
            # text_output = text_output.strip()
            '''
            print(text)
            text_output = text
            idx_end = text_output.find(tokenizer.eos_token)
            if idx_end >= 0:
                text_output = text_output[idx_end:]
            idx_end_2= text_output.find(tokenizer.eos_token)
            # Generate endoftext at the end
            if idx_end_2>=0:
                text_output = text_output[14:-14]
            # Generate no endoftext at the end   
            else:
                text_output = text_output[14:]
            if idx==1:
                print("\n CHECK \n",text_output)
                print(refs)
            print(text_output)
            out.append(text_output)
        with open(write_path, 'w', ) as f:
            for i in out:
                f.write(i + '\n')
        return out,refs_s








