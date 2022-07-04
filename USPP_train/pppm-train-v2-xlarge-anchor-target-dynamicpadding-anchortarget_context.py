#!/usr/bin/env python
# coding: utf-8

# # CFG

# In[1]:


import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# In[2]:


class CFG:
    debug = False
    wandb = False
    fp16 = True
    print_freq = 400
    num_workers = 0
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5

    task = 'deberta-v2-xlarge-anchor-target-310-dynamicpadding-anchortarget-context_text_test1'
    model = "deberta-v2-xlarge"
    # 已应用：difflr, coattention, AttPool
    # Todo: gc, Myfgm, lower_context, basedrop, mdrop, cat_gru

    # Task
    seed = 42
    aug = False  # 数据增强
    data_type = 'ab-c'  # 'ab-c' / 'a-bc' / 'b-ac' / 'ca-b'
    lower_context = False
    warmup_ratio = 0.1
    epochs = 5  # 5
    epochs_stage2 = 0
    batch_size = 16
    gradient_accumulation_steps = 2
    encoder_lr = 8e-6
    decoder_lr = 8e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0.01
    max_grad_norm = 1000

    # Model
    coattention = False
    cat_gru_h = False
    cat_gru_o = False
    att_pool = True
    num_layer = 1

    basedrop = False
    dropout = 0.2
    mdrop = False

    #     basedrop=True
    #     dropout=0.1
    #     mdrop=True

    dropout1 = 0.1
    dropout2 = 0.2
    dropout3 = 0.3
    dropout4 = 0.4
    dropout5 = 0.5

    # Train
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    train = True
    do_gc = False
    do_fgm = True
    skf = False
    mskf = False
    sgkf = True
    alldata = False
    pseudo_file = None


if CFG.alldata:
    CFG.trn_fold = [0]

if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]

# In[3]:


# # Library

# In[4]:


import os
import gc
import re
import time
import math
import random
import warnings

warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
# from ai.optim import AdamWGC
# from ai.helper import ensure_dir

import tokenizers
import transformers
from transformers import AutoModel, AutoConfig, AutoTokenizer
# from tokenization_fast import DebertaV2TokenizerFast as AutoTokenizer
from transformers.models.deberta.modeling_deberta import ContextPooler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# get_ipython().run_line_magic('env', 'TOKENIZERS_PARALLELISM=true')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = f'psudo614/{CFG.task}'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# In[5]:


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Two further options for CuDNN backend
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(seed=CFG.seed)


# # Helper functions for scoring

# In[6]:


def accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {"acc": acc, "f1": f1}


def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score


# # Utils

# In[7]:


def get_logger(filename=f'{OUTPUT_DIR}/train.log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=filename)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()

# # Data Loading

# In[8]:


titles = pd.read_csv('data/titles.csv')

train = pd.read_csv('data/train.csv')
train = train.merge(titles, left_on='context', right_on='code')
print(f"train.shape: {train.shape}")
# display(train.head())

if CFG.pseudo_file != None:
    pseudo = pd.read_csv(CFG.pseudo_file)
    pseudo = pseudo.merge(titles, left_on='context', right_on='code')
    print(f"pseudo.shape: {pseudo.shape}")
    # display(pseudo.head())

# In[9]:


from copy import deepcopy
from collections import defaultdict

if CFG.aug:
    anchor2synonyms = defaultdict(list)
    aug = deepcopy(train)
    # display(aug.head())

    aug_pos = aug[aug['score'] == 1.0].reset_index(drop=True)
    for a, t in zip(aug_pos['anchor'], aug_pos['target']):
        anchor2synonyms[a].append(t)
    aug['anchor'] = aug['anchor'].apply(lambda x: random.choice(anchor2synonyms[x]) if anchor2synonyms[x] else '')
    aug = aug[aug['anchor'] != ''].reset_index(drop=True)
    print(len(train), len(aug))
    # display(aug.head())

    train = pd.concat([train, aug], axis=0).reset_index(drop=True)
    print(len(train))


# In[10]:


# ====================================================
# CPC Data
# ====================================================
def get_cpc_texts():
    contexts = []
    pattern = '[A-Z]\d+'
    for file_name in os.listdir('data/CPCSchemeXML202105'):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
        with open(f'data/CPCTitleList202202/cpc-section-{cpc}_20220201.txt') as f:
            s = f.read()
        pattern = f'{cpc}\t\t.+'
        result = re.findall(pattern, s)
        cpc_result = result[0].lstrip(pattern)
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f'{context}\t\t.+'
            result = re.findall(pattern, s)
            context_text = cpc_result + ". " + result[0].lstrip(pattern)
            if CFG.lower_context:
                context_text = context_text.lower()
            results[context] = context_text
    return results


cpc_texts = get_cpc_texts()
torch.save(cpc_texts, f"{OUTPUT_DIR}/cpc_texts.pth")
train['context_text'] = train['context'].map(cpc_texts)
# display(train.head())


group = train.groupby(['anchor', "context"]).agg({"target":list}).reset_index()
group["target"] = group["target"].map(lambda s: "; ".join(s))
group.rename(columns={"target":"anchor_target"}, inplace=True)
train = train.merge(group, how='left', on=["anchor", "context"])

# group = train.groupby(['anchor']).agg({"target":list}).reset_index()
# group["target"] = group["target"].map(lambda s: ";".join(s))
# group.rename(columns={"target":"anchor_target"}, inplace=True)
#
# # display(group.head())
# train = train.merge(group, how='left', on=["anchor"])
# train.head()

    # display(pseudo.head())

# # CV split

# In[11]:


if CFG.data_type == 'ab-c':
    train['text'] = train['anchor'] + ' [SEP] ' + train['target']  +  '; ' + train['anchor_target'] +  ' [SEP] ' + train['context_text']
    # if CFG.pseudo_file != None:
    #     pseudo['text'] = pseudo['anchor'] + ' [SEP] ' + pseudo['target']
elif CFG.data_type == 'a-bc':
    train['text'] = train['anchor']
    train['context_text'] = train['target'] + ' [SEP] ' + train['context_text']
    if CFG.pseudo_file != None:
        pseudo['text'] = pseudo['anchor']
        pseudo['context_text'] = pseudo['target'] + ' [SEP] ' + pseudo['context_text']
elif CFG.data_type == 'b-ac':
    train['text'] = train['target']
    train['context_text'] = train['anchor'] + ' [SEP] ' + train['context_text']
    if CFG.pseudo_file != None:
        pseudo['text'] = pseudo['target']
        pseudo['context_text'] = pseudo['anchor'] + ' [SEP] ' + pseudo['context_text']
elif CFG.data_type == 'ca-b':
    # train['text'] = train['title'] + ' [SEP] ' + train['anchor']
    train['text'] = train['context_text'] + ' [SEP] ' + train['anchor']
    train['context_text'] = train['target']
    if CFG.pseudo_file != None:
        # pseudo['text'] = pseudo['title'] + ' [SEP] ' + pseudo['anchor']
        pseudo['text'] = pseudo['context_text'] + ' [SEP] ' + pseudo['anchor']
        pseudo['context_text'] = pseudo['target']
# display(train.head())


# In[12]:


# train['score'].hist()
# display(train['context'].apply(lambda x: x[0]).value_counts())


# In[13]:


# ====================================================
# CV split
# ====================================================
train['score_map'] = train['score'].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})
if CFG.pseudo_file != None:
    pseudo['score_map'] = pseudo['score'].map({0.00: 0, 0.25: 1, 0.50: 2, 0.75: 3, 1.00: 4})

encoder = LabelEncoder()
train['anchor_map'] = encoder.fit_transform(train['anchor'])

if CFG.sgkf:
    kf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    for n, (train_idx, val_idx) in enumerate(kf.split(X=train, y=train['score_map'], groups=train['anchor_map'])):
        train.loc[val_idx, 'fold'] = int(n)
    if CFG.pseudo_file != None:
        for n, (train_idx, val_idx) in enumerate(kf.split(X=pseudo, y=pseudo['score_map'], groups=pseudo['context'])):
            pseudo.loc[val_idx, 'fold'] = int(n)
elif CFG.mskf:
    kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = ['context', 'score_map']
    for n, (train_idx, val_idx) in enumerate(kf.split(train, train[labels])):
        train.loc[val_idx, 'fold'] = int(n)
    if CFG.pseudo_file != None:
        for n, (train_idx, val_idx) in enumerate(kf.split(pseudo, pseudo[labels])):
            pseudo.loc[val_idx, 'fold'] = int(n)
elif CFG.skf:
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for n, (train_idx, val_idx) in enumerate(kf.split(X=train, y=train['score_map'])):
        train.loc[val_idx, 'fold'] = int(n)
    if CFG.pseudo_file != None:
        for n, (train_idx, val_idx) in enumerate(kf.split(X=pseudo, y=pseudo['score_map'])):
            pseudo.loc[val_idx, 'fold'] = int(n)

train['fold'] = train['fold'].astype(int)
# display(train.groupby('fold').size())

if CFG.pseudo_file != None:
    pseudo['fold'] = pseudo['fold'].astype(int)
    # display(pseudo.groupby('fold').size())

# In[14]:


if CFG.debug:
    display(train.groupby('fold').size())
    train = train.sample(n=1000, random_state=0).reset_index(drop=True)
    display(train.groupby('fold').size())

# # tokenizer

# In[15]:


# from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast

tokenizer = AutoTokenizer.from_pretrained(CFG.model)
tokenizer.save_pretrained(OUTPUT_DIR + 'tokenizer/')
CFG.tokenizer = tokenizer

# # Dataset

# In[16]:


# ====================================================
# Define max_len
# ====================================================
lengths_dict = {}

for text_col in ['text', 'context_text']:
    lengths = []
    tk0 = tqdm(train[text_col].fillna("").values, total=len(train))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    lengths_dict[text_col] = lengths

# CFG.max_len = max(lengths_dict['text']) + max(lengths_dict['context_text']) + 3  # CLS + SEP + SEP

# CFG.q_len, CFG.p_len = max(lengths_dict['text']), max(lengths_dict['context_text'])
arr1 = np.asarray(lengths_dict['text'])
CFG.max_len = int(np.percentile(arr1, 95))
LOGGER.info(f"max_len: {CFG.max_len}")

# arr2 = np.asarray(lengths_dict['context_text'])
# CFG.p_len = int(np.percentile(arr2, 99))
#
# print(CFG.q_len, CFG.p_len)
# CFG.max_len = CFG.q_len + CFG.p_len + 3


# In[17]:


def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           truncation='longest_first',
                           return_offsets_mapping=False)
    # for k, v in inputs.items():
    #     inputs[k] = torch.tensor(v, dtype=torch.long)

    # len_text_a = min(CFG.q_len, len(cfg.tokenizer(text, add_special_tokens=False)['input_ids']))
    # len_text_b = min(CFG.p_len, len(cfg.tokenizer(context_text, add_special_tokens=False)['input_ids']))
    # q_end_index = len_text_a + 1
    # p_end_index = q_end_index + len_text_b
    # qp_end_pos = [q_end_index, p_end_index]
    # qp_end_pos = torch.tensor(qp_end_pos, dtype=torch.long)

    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.labels = df['score'].values
        self.texts = df['text'].values
        # self.context_texts = df['context_text'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs = prepare_input(
            self.cfg,
            self.texts[item]
        )
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label


def anchor_target_random_shuffle(target, anchor_target):
    shuffled = [element for element in anchor_target if element != target]
    shuffled = random.sample(shuffled, min(len(shuffled), 29))
    shuffled = shuffled + [target]
    random.shuffle(shuffled)

    return shuffled


class PatentPhraseDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, is_train=True):
        self.is_train = is_train
        self.tokenizer = tokenizer

        self.anchor = df["anchor"].tolist()
        self.target = df["target"].tolist()
        self.context = df["context_text"].tolist()
        self.anchor_target = df["anchor_target"].tolist()

        if self.is_train:
            self.score = df["score"].tolist()

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, index):
        anchor = self.anchor[index]
        target = self.target[index]
        context = self.context[index]

        anchor_target = self.anchor_target[index]
        anchor_target = ";".join(anchor_target_random_shuffle(target, anchor_target))

        inputs = anchor + "[SEP]" + target + ". " + anchor_target + "[SEP]" + context

        inputs = self.tokenizer(
            inputs,
            max_length=CFG.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
        )

        if self.is_train:
            #             return {
            #                 "ids": torch.LongTensor(encoded["input_ids"]),
            #                 "masks": torch.LongTensor(encoded["attention_mask"]),
            #                 "labels": torch.FloatTensor([self.score[index]])
            #             }
            label = torch.FloatTensor([self.score[index]])
            return inputs, label
        else:
            #             return {
            #                 "ids": torch.LongTensor(encoded["input_ids"]),
            #                 "masks": torch.LongTensor(encoded["attention_mask"]),
            #             }
            return inputs

class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        #         output = dict()
        #         inputs = [sample for sample in batch[0]]
        #         label = [sample for sample in batch[1]]
        #         print(inputs,label)
        #         print(len(inputs))
        inputs = dict()
        inputs["input_ids"] = [sample[0]["input_ids"] for sample in batch]
#         inputs["token_type_ids"] = [sample[0]["token_type_ids"] for sample in batch]
        inputs["attention_mask"] = [sample[0]["attention_mask"] for sample in batch]
        label = [sample[1] for sample in batch]

        #         print(label)
        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in inputs["input_ids"]])
        #         padding_label_id = '-100'
        # add padding

        inputs["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in inputs["input_ids"]]
#         inputs["token_type_ids"] = [s + (batch_max - len(s)) * [s[-1]] for s in inputs["token_type_ids"]]
        inputs["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in inputs["attention_mask"]]
        #         inputs["label"] = [s + (batch_max - len(s)) * [padding_label_id] for s in inputs["label"]]

        # convert to tensors
        inputs["input_ids"] = torch.tensor(inputs["input_ids"], dtype=torch.long)
#         inputs["token_type_ids"] = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        # label = torch.tensor(label, dtype=torch.float)
        label = torch.stack(label)

        return inputs, label

# # Model

# In[18]:


class CoAttention(nn.Module):
    def __init__(self, config):
        super(CoAttention, self).__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = F.gelu

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_ids, input_ids_1, attention_mask=None, head_mask=None):
        """attention_mask 对应 input_ids"""
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask

        mixed_query_layer = self.query(input_ids_1)
        mixed_key_layer = self.key(input_ids)
        mixed_value_layer = self.value(input_ids)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        reshaped_context_layer = context_layer.view(*new_context_layer_shape)

        # Should find a better way to do this
        w = self.dense.weight.t().view(self.num_attention_heads, self.attention_head_size, self.hidden_size).to(
            context_layer.dtype)
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_1 + projected_context_layer_dropout)

        ffn_output = self.ffn(layernormed_context_layer)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + layernormed_context_layer)
        return hidden_states

#
# def split_ques_context(sequence_output, qp_end_pos, sep_tok_len=1, ques_max_len=CFG.q_len, context_max_len=CFG.p_len):
#     ques_sequence_output = sequence_output.new(
#         torch.Size((sequence_output.size(0), ques_max_len, sequence_output.size(2)))).zero_()
#     context_sequence_output = sequence_output.new_zeros(
#         (sequence_output.size(0), context_max_len, sequence_output.size(2)))
#     context_attention_mask = sequence_output.new_zeros((sequence_output.size(0), context_max_len))
#     ques_attention_mask = sequence_output.new_zeros((sequence_output.size(0), ques_max_len))
#     for i in range(0, sequence_output.size(0)):
#         q_end = int(qp_end_pos[i][0].item())
#         p_end = int(qp_end_pos[i][1].item())
#         ques_sequence_output[i, :min(ques_max_len, q_end)] = sequence_output[i, 1: 1 + min(ques_max_len, q_end)]
#         context_sequence_output[i, :min(context_max_len, p_end - q_end - sep_tok_len)] = sequence_output[i,
#                                                                                          q_end + sep_tok_len + 1: q_end + sep_tok_len + 1 + min(
#                                                                                              p_end - q_end - sep_tok_len,
#                                                                                              context_max_len)]
#         context_attention_mask[i, :min(context_max_len, p_end - q_end - sep_tok_len)] = sequence_output.new_ones(
#             (1, context_max_len))[0, :min(context_max_len, p_end - q_end - sep_tok_len)]
#         ques_attention_mask[i, : min(ques_max_len, q_end)] = sequence_output.new_ones((1, ques_max_len))[0,
#                                                              : min(ques_max_len, q_end)]
#     return ques_sequence_output, context_sequence_output, ques_attention_mask, context_attention_mask


# In[19]:


class ClsModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.num_labels = 1
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)

        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.mdrop:
            self.dropout1 = nn.Dropout(cfg.dropout1)
            self.dropout2 = nn.Dropout(cfg.dropout2)
            self.dropout3 = nn.Dropout(cfg.dropout3)
            self.dropout4 = nn.Dropout(cfg.dropout4)
            self.dropout5 = nn.Dropout(cfg.dropout5)

        # self.pooler = ContextPooler(self.config)

        if cfg.coattention:
            self.coatt = CoAttention(self.config)
        if cfg.cat_gru_h or cfg.cat_gru_o:
            self.gru = nn.GRU(
                self.config.hidden_size,
                int(self.config.hidden_size / 2),
                num_layers=1,
                bidirectional=True,
                batch_first=True).cuda()
            self.fc = nn.Linear(2 * self.config.hidden_size, self.num_labels)
        else:
            self.fc = nn.Linear(self.config.hidden_size, self.num_labels)
        self._init_weights(self.fc)

        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.config.hidden_size, 1)
        )
        self._init_weights(self.attention)

        if cfg.cat_gru_o:
            self.attention_gru = nn.Sequential(
                nn.Linear(self.config.hidden_size, 512),
                nn.Tanh(),
                nn.Linear(512, 1),
                nn.Softmax(dim=1)
            )
            self._init_weights(self.attention_gru)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)

        if self.cfg.num_layer <= 1:
            sequence_output = outputs[0]
        else:
            hidden_states = outputs[-1]
            sequence_output = sum([hidden_states[-i] for i in range(1, self.cfg.num_layer + 1)])

        # if self.cfg.coattention:
        #     query_sequence_output, context_sequence_output, query_attention_mask, context_attention_mask = split_ques_context(
        #         sequence_output, qp_end_pos)
        #     sequence_output = self.coatt(query_sequence_output, sequence_output, query_attention_mask)
        #     sequence_output += outputs[0]

        # if self.cfg.att_pool:
        weights = self.attention(sequence_output)
        pooled_output = torch.sum(weights * sequence_output, dim=1)
        # else:
        #     pooled_output = self.pooler(sequence_output)



        return pooled_output

    def forward(self, inputs):
        feature = self.feature(inputs)

        if self.cfg.basedrop:
            feature = self.dropout(feature)

        if self.cfg.mdrop:
            logits1 = self.fc(self.dropout1(feature))
            logits2 = self.fc(self.dropout2(feature))
            logits3 = self.fc(self.dropout3(feature))
            logits4 = self.fc(self.dropout4(feature))
            logits5 = self.fc(self.dropout5(feature))
            output = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        else:
            # output = self.fc(feature)

            output = self.regressor(feature)
        return output


# # Helpler functions

# In[20]:


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.1, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# class FGM():
#     """ emb_name 为模型中 embedding 的参数名
#     """
#     def __init__(self, model):
#         self.model = model
#         self.backup = {}

#     def attack(self, epsilon=1., emb_name='word_embedding'):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and emb_name in name:
#                 self.backup[name] = param.data.clone()
#                 norm = torch.norm(param.grad)
#                 if norm != 0 and not torch.isnan(norm):
#                     r_at = epsilon * param.grad / norm
#                     param.data.add_(r_at)

#     def restore(self, emb_name='word_embedding'):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad and emb_name in name:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#         self.backup = {}


# In[21]:


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, ema, fgm=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.fp16)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.fp16):
            y_preds = model(inputs)

        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1)).mean()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        if fgm != None:
            # 对抗训练-FGM
            fgm.attack()  # 在 embedding 上添加对抗扰动
            with torch.cuda.amp.autocast(enabled=CFG.fp16):
                y_preds_adv = model(inputs)
            loss_adv = criterion(y_preds_adv.view(-1, 1), labels.view(-1, 1)).mean()
            if CFG.gradient_accumulation_steps > 1:
                loss_adv = loss_adv / CFG.gradient_accumulation_steps
            scaler.scale(loss_adv).backward()  # 反向传播，并在正常的 grad 基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复 embedding 参数

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model.parameters())
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if (step + 1) % CFG.gradient_accumulation_steps == 0 and (
                (step + 1) % CFG.print_freq == 0 or step == (len(train_loader) - 1)):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        # if CFG.wandb:
        #     wandb.log({f"[fold{fold}] loss": losses.val,
        #                f"[fold{fold}] lr": scheduler.get_lr()[0]})
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1)).mean()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step + 1) / len(valid_loader))))
    predictions = np.concatenate(preds)
    predictions = np.concatenate(predictions)
    return losses.avg, predictions


# In[22]:


def pearson_cumsom_loss(y_true, y_pred):
    '''
    optmize negative pearson coefficient loss
    :param y_true:
    :param y_pred:
    :return:
    '''
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    n = len(y_true)
    y_bar = y_true.mean()
    yhat_bar = y_pred.mean()
    c = 1 / ((y_true - y_bar) ** 2).sum().sqrt()  # constant variable
    b = ((y_pred - yhat_bar) ** 2).sum().sqrt()  # std of pred

    a_i = y_true - y_bar
    d_i = y_pred - yhat_bar
    a = (a_i * d_i).sum()
    gradient = c * (a_i / b - a * d_i / b ** 3)
    hessian = - (np.matmul(a_i.reshape(-1, 1), d_i.reshape(1, -1)) + np.matmul(d_i.reshape(-1, 1), a_i.reshape(1,
                                                                                                               -1))) / b ** 3 + 3 * a * np.matmul(
        d_i.reshape(-1, 1), d_i.reshape(1, -1)) / b ** 5 + a / (n * b ** 3)
    hessian = hessian - np.ones(shape=(n, n)) * a / b ** 3
    hessian *= c
    return -hessian


# In[23]:

class EMA(object):
    """
    Maintains (exponential) moving average of a set of parameters.
    使用ema累积模型参数
    Args:
        parameters (:obj:`list`): 需要训练的模型参数
        decay (:obj:`float`): 指数衰减率
        use_num_updates (:obj:`bool`, optional, defaults to True): Whether to use number of updates when computing averages
    Examples::
        >>> ema = EMA(module.parameters(), decay=0.995)
        >>> # Train for a few epochs
        >>> for _ in range(epochs):
        >>>     # 训练过程中，更新完参数后，同步update shadow weights
        >>>     optimizer.step()
        >>>     ema.update(module.parameters())
        >>> # eval前，进行ema的权重替换；eval之后，恢复原来模型的参数
        >>> ema.store(module.parameters())
        >>> ema.copy_to(module.parameters())
        >>> # evaluate
        >>> ema.restore(module.parameters())
    Reference:
        [1]  https://github.com/fadel/pytorch_ema
    """  # noqa: ignore flake8"

    def __init__(
            self,
            parameters,
            decay,
            use_num_updates=True
    ):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone()
                                 for param in parameters
                                 if param.requires_grad]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            if param.requires_grad:
                param.data.copy_(c_param.data)


def train_loop(folds, pseudo_folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")
    collate_fn = Collate(tokenizer)
    # ====================================================
    # loader
    # ====================================================
    if CFG.alldata:
        train_folds = folds
        mix_folds = train_folds
        if CFG.pseudo_file != None:
            mix_folds = pd.concat([pseudo_folds, train_folds], axis=0).reset_index(drop=True)
    else:
        train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
        mix_folds = train_folds
        if CFG.pseudo_file != None:
            pseudo_folds = pseudo_folds[pseudo_folds['fold'] != fold].reset_index(drop=True)
            mix_folds = pd.concat([pseudo_folds, train_folds], axis=0).reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds['score'].values

    mix_dataset = TrainDataset(CFG, mix_folds)
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    mix_loader = DataLoader(mix_dataset,
                            batch_size=CFG.batch_size,
                            shuffle=True,
                            num_workers=CFG.num_workers, pin_memory=True, drop_last=True,collate_fn=collate_fn)
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True,collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False,collate_fn=collate_fn)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = ClsModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, f'{OUTPUT_DIR}/config.pth')
    model.to(device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr,
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    Optim = AdamW
    optimizer = Optim(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        num_warmup_steps = int(cfg.warmup_ratio * num_train_steps)
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles
            )
        return scheduler

    # num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    num_train_steps = int(len(mix_folds) / CFG.batch_size * (CFG.epochs - CFG.epochs_stage2)) + int(
        len(train_folds) / CFG.batch_size * CFG.epochs_stage2)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    if CFG.do_fgm:
        fgm = FGM(model)
    else:
        fgm = None
    ema = EMA(model.parameters(), decay=0.999)
    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    best_score = 0.

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        if epoch >= CFG.epochs - CFG.epochs_stage2:
            avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, ema, fgm=fgm)
        else:
            avg_loss = train_fn(fold, mix_loader, model, criterion, optimizer, epoch, scheduler, device, ema, fgm=fgm)

        # eval
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        ema.restore(model.parameters())
        # scoring
        score = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}')
        if CFG.wandb:
            wandb.log({f"[fold{fold}] epoch": epoch + 1,
                       f"[fold{fold}] avg_train_loss": avg_loss,
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] score": score})

        if best_score < score:
            best_score = score
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 'predictions': predictions}, f"{OUTPUT_DIR}/{CFG.task}_{fold}.pth")

    predictions = torch.load(f"{OUTPUT_DIR}/{CFG.task}_{fold}.pth", map_location=torch.device('cpu'))['predictions']
    valid_folds['pred'] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


# In[24]:


if __name__ == '__main__':

    def get_result(oof_df):
        labels = oof_df['score'].values
        preds = oof_df['pred'].values
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')


    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                if CFG.pseudo_file != None:
                    _oof_df = train_loop(train, pseudo, fold)
                else:
                    _oof_df = train_loop(train, None, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(f'{OUTPUT_DIR}/oof_df.pkl')

    if CFG.wandb:
        wandb.finish()

# In[ ]:


# 8320 8253 8393 8327 8463

