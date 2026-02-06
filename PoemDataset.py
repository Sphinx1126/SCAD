# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:17:53 2024

@author: 28257
"""

import torch
import torch.nn as nn
import pandas as pd
import random
from collections import defaultdict
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from enum import Enum
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from os.path import join
from loguru import logger

class PoemDataset(Dataset):
    def __init__(self, data_path,
                 alltxts,poem2all,
                 negative_sample_size,
                 bert_path,bert_max_len=512):
        df=pd.read_csv(data_path)
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        
        input_ids,token_type_ids,attention_masks=[],[],[]
        for _,row in df.iterrows():
            poem=[row['poem']]*(negative_sample_size+1)
            positive_text=[row['text']]
            negative_text=random.sample(list(set(alltxts)-set(poem2all[poem[0]])), negative_sample_size)
            inputs=tokenizer(text = poem,
                             text_pair = positive_text+negative_text,
                             truncation = True,
                             padding = 'max_length',
                             max_length = bert_max_len,
                             return_tensors = "pt",
                             return_token_type_ids = True,
                             return_attention_mask = True,
                             return_special_tokens_mask = True,
                             return_length = True)
            input_ids.append(inputs.input_ids)
            token_type_ids.append(inputs.token_type_ids)
            attention_masks.append(inputs.attention_mask)
        
        self.input_ids=input_ids
        self.token_type_ids=token_type_ids
        self.attention_masks=attention_masks
        
        

        
    def __len__(self) -> int:
        return len(self.input_ids)
    
            
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        input_ids=self.input_ids[index]
        token_type_ids=self.token_type_ids[index]
        attention_masks=self.attention_masks[index]
        return input_ids,token_type_ids,attention_masks
