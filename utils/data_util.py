from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import GPT2Config,GPT2Tokenizer,GPT2LMHeadModel



import torch
import logging
import json
import os
import copy
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding,PaddingStrategy

#from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
logger = logging.getLogger(__name__)
import _locale
_locale._getdefaultlocale = (lambda *args: ['zh_CN', 'utf8'])
Model_Class={'gpt2': {
    'config': GPT2Config,
    'tokenizer': GPT2Tokenizer,
    'model': GPT2LMHeadModel,
}}


@dataclass
class DataCollatorForData2TextLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    format_mode: str = 'cat'
    mlm_probability: float = 0.15

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        # print(examples[0])
        # print(len(examples))
        input_ids, labels, src, tgt, cate = zip(*examples)
        # print(len(input_ids), len(labels), len(weights))
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:

            # print(self.format_mode)
            if self.format_mode == 'cat':
                mode_input = 3
            elif self.format_mode == 'peek':
                mode_input = 1
            elif self.format_mode == 'nopeek':
                mode_input = 2
            elif self.format_mode == 'infix':
                mode_input = 4

            # mode_input = 1 # means that we take the input again.
            # mode_input = 2 # means that we do not peek at src again.
            # mode_input = 3 # means that we look at the categories, and see the input again.

            # print(self.format_mode, mode_input)

            if mode_input == 1:
                # input, batch
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(src)
                cate_batch, cate_attn = None, None
                # tgt = self._tensorize_batch(tgt)
            elif mode_input == 2:
                # nopeek.
                batch = self._tensorize_batch(tgt)
                labels = batch.clone()
                src = self._tensorize_batch(src)
                cate_batch, cate_attn = None, None
            elif mode_input == 3:
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(cate)
                cate_batch, cate_attn = None, None
            elif mode_input == 4:
                batch = self._tensorize_batch(tgt)
                labels = batch.clone()
                src = self._tensorize_batch(src)

                cate_batch = self._tensorize_batch(cate)
                cate_attn = (cate_batch != self.tokenizer.pad_token_id)

            labels[labels == self.tokenizer.pad_token_id] = -100 # tgt
            src_attn = (src != self.tokenizer.pad_token_id) # src
            tgt_attn = (batch != self.tokenizer.pad_token_id) # tgt

            if cate_batch is None:
                return {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn,
                        'src':src}
            else:
                return {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn': tgt_attn,
                        'src': src, "cate_batch":cate_batch, "cate_attn":cate_attn}

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class LineByLineWebNLGTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)


        with open(file_path) as f:
            lines_dict = json.load(f)


        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []

        for i, example in enumerate(lines_dict['entries']):
            sents = example[str(i + 1)]['lexicalisations']
            triples = example[str(i + 1)]['modifiedtripleset']

            rela_lst = []
            temp_triples = ''
            for j, tripleset in enumerate(triples):
                subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
                rela_lst.append(rela)
                temp_triples += ' | '

                temp_triples += '{} : {} : {}'.format(subj, rela, obj)



            for sent in sents:
                if sent["comment"] == 'good':
                    full_tgt_lst.append(sent["lex"])
                    full_src_lst.append(temp_triples)
                    full_rela_lst.append(rela_lst)



        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)


        edited_sents = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = full_rela_lst

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0

        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1]) # does not contain the BOS separator
                self.tgt_sent.append(self.examples[i][sep_idx-1:]) # contains the BOS separator.
                self.labels[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx - 1
                temp_tgt_len += len(elem) - (sep_idx - 1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len / temp_tgt_len)




        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        print()
        print(self.labels[1])
        print(self.examples[1])
        print(edited_sents[1])
        print(self.src_sent[1])
        print(self.tgt_sent[1])
        print(self.src_cat[1])

        print(self.labels[-1])
        print(self.examples[-1])
        print(edited_sents[-1])
        print(self.src_sent[-1])
        print(self.tgt_sent[-1])
        print(self.src_cat[-1])
        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.src_sent[i], dtype=torch.long),
                torch.tensor(self.tgt_sent[i], dtype=torch.long),
                torch.tensor(self.src_cat[i], dtype=torch.long),
                )
def read_webnlg_files(path,tokenizer):
        file_dict = {}
        #tokenizer=self.tokenizer

        with open(path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        # full_tgt_lst = []
        total_count = 0
        for i, example in enumerate(lines_dict['entries']):
            sents = example[str(i + 1)]['lexicalisations']
            triples = example[str(i + 1)]['modifiedtripleset']

            rela_lst = []
            temp_triples = ''
            for j, tripleset in enumerate(triples):
                subj, rela, obj = tripleset['subject'], tripleset['property'], tripleset['object']
                rela_lst.append(rela)
                # if i > 0:
                temp_triples += '  '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)
            ### 注意  ' {} {} '  还是 ' {} {}'
            # [21607],
            # [220],
            # [50256],
            # [220]])  会变得不幸 如果选 ' {} {} '

            temp_triples = ' {} {}'.format(temp_triples, tokenizer.bos_token)

            for sent in sents:
                if True:  # sent["comment"] == 'good'
                    if (temp_triples, tuple(rela_lst)) not in file_dict:
                        file_dict[(temp_triples, tuple(rela_lst))] = []
                        full_src_lst.append(temp_triples)
                        full_rela_lst.append(tuple(rela_lst))
                    file_dict[(temp_triples, tuple(rela_lst))].append(sent["lex"])

        print(len(file_dict), len(full_src_lst))
        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(file_dict)

        return file_dict
# if __name__=='__main__':
#     from transformers import GPT2Tokenizer
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium", cache_dir="../model/gpt2-medium-s3")
#     tokenizer("Hello world")['input_ids']
#
#     num_added_tokens = tokenizer.add_special_tokens(
#         {'pad_token': '[PAD]'})
#     print(tokenizer.eos_token)
#     print(tokenizer.bos_token) #<endoftext> 50526
#
#     # num_added_tokens = tokenizer.add_special_tokens(
#     #     {'eos_token': '[EOS]'})
#     print(num_added_tokens)
#     #C:\Users\14828\Desktop\Intern_TG\Code\PrefixTuning\data\webnlg_challenge_2017
#     file_path="C:\\Users\\14828\\Desktop\\Intern_TG\\Code\\PrefixTuning\\data\\webnlg_challenge_2017\\train.json"
#     block_size=1000
#     dataset = LineByLineWebNLGTextDataset(tokenizer=tokenizer, file_path=file_path,
#                                              block_size=block_size, bos_tok=tokenizer.bos_token,
#                                              eos_tok=tokenizer.eos_token)

class LineByLineTriplesTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)


        with open(file_path) as f:
            lines_dict = json.load(f)

        full_rela_lst = []
        full_src_lst = []
        full_tgt_lst = []
        for example in lines_dict:
            rela_lst = []
            temp_triples = ''
            for i, tripleset in enumerate(example['tripleset']):
                subj, rela, obj = tripleset
                rela = rela.lower()
                rela_lst.append(rela)
                if i > 0:
                    temp_triples += ' | '
                temp_triples += '{} : {} : {}'.format(subj, rela, obj)

            for sent in example['annotations']:
                full_tgt_lst.append(sent['text'])
                full_src_lst.append(temp_triples)
                full_rela_lst.append(rela_lst)


        assert len(full_rela_lst) == len(full_src_lst)
        assert len(full_rela_lst) == len(full_tgt_lst)


        edited_sents = []
        for src, tgt in zip(full_src_lst, full_tgt_lst):
            sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = full_rela_lst

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1]) # does not contain the BOS separator
                self.tgt_sent.append(self.examples[i][sep_idx-1:]) # contains the BOS separator.
                self.labels[i][:sep_idx] = [-100] * sep_idx

                temp_src_len += sep_idx - 1
                temp_tgt_len += len(elem) - (sep_idx - 1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len / temp_tgt_len)


        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.src_sent[i], dtype=torch.long),
                torch.tensor(self.tgt_sent[i], dtype=torch.long),
                torch.tensor(self.src_cat[i], dtype=torch.long),

                )

class LineByLineData2TextTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok:str, eos_tok:str, lowdata_token:str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        if lowdata_token is None:
            edited_sents = []
            for src, tgt in zip(src_lines, tgt_lines):
                sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)
        else:
            edited_sents = []
            for src, tgt in zip(src_lines, tgt_lines):
                sent = ' {} {} {} '.format(lowdata_token, src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = []
        for ss in src_lines:
            ssl = [la.split(':')[0].strip() for la in ss.split('|')]
            # print(ssl)
            ssl_lst.append(ssl)

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                            is_split_into_words=True)['input_ids']


        self.src_sent = []
        self.tgt_sent = []

        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx-1])
                self.tgt_sent.append(self.examples[i][sep_idx-1:])
                self.labels[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx-1
                temp_tgt_len += len(elem) - (sep_idx-1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len/temp_tgt_len)




        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        assert len(self.src_cat) == len(self.examples)


    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.src_sent[i], dtype=torch.long),
                torch.tensor(self.tgt_sent[i], dtype=torch.long),
                torch.tensor(self.src_cat[i], dtype=torch.long),

                )
